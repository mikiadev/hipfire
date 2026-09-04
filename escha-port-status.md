# Escha-W2 port status — feat/escha-w2

Working notes for the MoE-replay + dense-pipeline port. Scratch truth lives in
`/home/mika/git/hipfire-beta` (esha work = 3 commits on `e2f7dd1`, which IS an
ancestor of this master).

## Provenance

- hipfire-beta local escha work = commits `8c57663`, `12e7d15`, `da09de7` on top
  of `e2f7dd1` (upstream beta base). `e2f7dd1` is an ancestor of this master
  (1660 commits between). The scratch objects are fetched here as
  `beta-scratch/escha-head` (da09de7) and `beta-scratch/beta` (12e7d15).
- Full changeset vs `e2f7dd1`: 55 files, +12 582/−91 (diffstat captured in
  shell history; see `git diff e2f7dd1 beta-scratch/escha-head --stat`).
- Cherry-pick is NOT viable mechanically (conflicts across all 3 commits).
  This is a structured re-application onto the modular codebase.

## Decode math (authoritative, from higgs escha_ref.py + beta escham_decode.rs)

- `escha_code` I16 [*, in/16, out/16, 16*K]; K=2 gate_up/others, K=3 down/up.
- Trellis unpack → 256 uint16 windows (tail-biting 256K-bit circular stream,
  funnel shift formula in escha_ref.py `unpack_trellis`).
- 3INST codebook: `x = (idx*0xCBAC1FED & 0x8FFF8FFF) ^ 0x3B603B60`, sum fp16
  halves. Dep table is COMPUTABLE (pi(r) formula) — llama.cpp computes it
  inline, ships no dep table. escha exports ship NO dep/lut at all.
- `W[out,in] = (had128(had128(w_bare · rin · rout))).T`; both scales OUTSIDE WHT.
- Folded GEMV form: `M_folded = diag(rout) @ H_out @ W_bare^T @ H_in`,
  `y = M_folded @ (x·s_in·rin)`; rout applied post-GEMV.
- Dense model 3.8-27B: dim 5120, 64 layers, 48 linear-attn + 16 full-attn,
  head_dim 256, n_heads 24, n_kv_heads 4, vocab 248320. All linear projections
  are escha-coded EXCEPT embed/lm_head (int8 + per-row f16 scale) and the small
  dense tensors (norms, conv1d, A_log, dt_bias, biases).

## Dense export facts (Qwen3.8-27B-Escha-W2)

- Nested config: `Qwen3_5ForConditionalGeneration` with `text_config` +
  `vision_config` present (VL-shaped export; `language_model_only`).
- Weight prefix: `model.language_model.layers.<i>.<submodule>.<proj>.escha_*`.
- Per coded proj: `escha_code I16 [in/16, out/16, 16K]`, `escha_config I32[6]`
  (actually 6 f32-ish values near 1.0 — scale per something; NOT the 
  [tile,K,bits,mcg,...] of the MoE doc), `escha_rin F16`, `escha_rout F16`,
  `escha_s_in F32`, `escha_s_out F32` (near 1.0, per-channel), plus `.bias F16`.
- Linear-attn projs: in_proj_qkv (5120→10240), in_proj_z (5120→6144),
  out_proj (6144→5120); K=2.
- Full-attn projs: q 5120→12288, k 5120→1024, v 5120→1024, o 6144→5120; K=2.
- MLP per layer: gate 5120→17408 (K=2), up 5120→17408 (K=3), down 17408→5120
  (K=3).
- llama.cpp dense escha path reads code/rin/rout/bias ONLY; ignores s_in/s_out;
  computes dep inline; does NOT apply bias (`ESCHA_APPLY_BIAS 0`).

## Fresh-tree integration anchors

- Config: `Qwen35Config` parser descends into `text_config` already; reads
  quant_method at TOP-LEVEL node only — needs `is_escham_moe` /
  `is_escha_dense` extension (config.json `quantization_config.quant_method`
  is top-level here: `eschamoe` (MoE) / `escha` (dense) + `format_version`).
- Loader: `SafetensorsSource` dir model already supported; `paro_text_prefix`
  matches `model.language_model.`. `load_weights_from_safetensors` equivalent
  lives in qwen35/load.rs.
- Forward decode arms live in qwen35/forward.rs `forward_scratch_layers`; MoE
  (A3B) already present via `LayerWeights::{DeltaNetMoe, FullAttnMoe}` +
  `moe_ffn_decode_impl` → `hipfire_dispatch::families::moe` executor (MoeFamily).
  Escha needs NEW enum variants + arms or an Escha-aware moe path.
- WeightTensor/WeightRef + Step/GemvInput moved to `hipfire-dispatch` (pipeline
  & families) — beta's direct `Step::Gemv` call sites don't exist verbatim.
- rdna-compute Gpu API parity: bind_thread / ensure_kernel / launch_maybe_blob /
  profile::begin_timer all still exist on master → beta escham.rs dispatch
  module ports ~verbatim (modulo `pub(crate)` visibility).
- kernel src registration: rdna-compute/src/kernels.rs include_str! pattern.
  kernels live kernels/src/*.hip (escham dir absent on master — add
  kernels/src/escham/hip/*.hip + registration).

## TODO / decision points (high-level)

1. MoE replay onto master (loader flag, Escha FFN decode+fold+grouped f16,
   attention arms, int8/embed handling, tokenizer added tokens, gemv_f32
   perf fix) — get /data/rocmfpx/Escha-W2 coherent on this tree.
2. Dense loader + pipeline for /data/rocmfpx/Qwen3.8-27B-Escha-W2.
3. Full-kernel decode + (later) prefill; throughput targets per handoff.

## A5 status (MoE decode coherent — 2026-09-04)

Root cause found and fixed (per-expert input-scale indexing in the routed-FFN
cache fill). With `HIPFIRE_TOKEN_TRACE=1` per-token prints, decode emitted a
conditioned 2-6 token prefix then collapsed to period-2/period-5 attractors
(`220,16,220,16…`, `11,220,1,423,198,…`) — identical symptoms across prompts.
The framework itself was exonerated by a control run of the HFQ A3B MoE
(`qwen3.6-35b-a3b.mq4r`, byte-identical text_config) which decoded
flawlessly on this tree ("The capital of France is **Paris**.").

**Root cause:** `escha_ffn.rs::escham_moe_ffn_decode` filled each routed
expert's cached folded weight with the *entire stacked* per-expert input
scale tensor (`&ffn.gate_up_in_scale`, `[n_exp, dim]` / `&ffn.down_in_scale`,
`[n_exp, mi]`) passed to `escham_apply_rowcol_scales_f32`. That kernel
indexes `in_scale[j]` for `j in 0..in_p`, so every expert's folded weight
was scaled by **expert 0's** `s_in·rin` column vector. The export's rin/s_in
are genuinely per-expert (rel diff vs expert 0: 1.3 for e1, 1.5 for e200),
so all routed experts but (by luck) the first got the wrong input rotation —
plausible but corrupt FFN output. Beta's working fill slices per-expert:
`gate_up_in_scale.sub_offset(exp_idx * hidden, hidden)` (and the same for
down). The fresh clone dropped that per-expert sub-offset. Note the
single-expert FFN verifications (rel ~3e-4) never caught this: the
production-path check tool (`examples/check_escha_ffn.rs`) exercises one
expert at a time via an already-sliced host upload.

**Fix (crates/hipfire-arch-qwen35/src/qwen35/escha_ffn.rs):** bind
`gu_in_scale = ffn.gate_up_in_scale.sub_offset(exp_idx * gu_in_p, gu_in_p)`
and `dn_in_scale = ffn.down_in_scale.sub_offset(exp_idx * down_in_p,
down_in_p)` before the row-col scale absorb, matching beta.

**Verified coherent (all --temp 0):**
- "The capital of France is" → "The capital of France is **Paris**." —
  token ids byte-identical to the HFQ A3B control (760, 6511, 314, 9338,
  369, 2972, 57590, 159034, 248046).
- "The capital of Japan is" → "…**Tokyo**. Tokyo is the largest
  metropolitan area…" (fluent past 30 tokens).
- "Write a haiku about the ocean" → coherent haiku.
- "Explain why the sky is blue." → fluent 60-token Rayleigh-scattering
  answer.
- "def fibonacci(n):" / "def is_prime(n):" → open ```python fences.

## A4 status (MoE decode on the fresh tree — 2026-09-04)

Commits: baseline bf1214ca → A1 scaffolding 944ff9d0 → match-sweep 0c487847 →
loader 262109b4 → CLI/tokenizer b7a62c03 → forward arms e4f40d66 → int8
augmentor + debug round ee47589c.

**Works end-to-end:** model /data/rocmfpx/Escha-W2 loads (arch 6, quant
eschamoe, 40 layers, all Escha arms), the daemon decodes at ~10 tok/s, and:
- GPU trellis decode + fold vs host reference: rel ~1.4e-4, 0/8192 wrong
  tiles (verified layers 0/5, experts 0/200/100, gate_up K2 + down K3).
- Production single-expert FFN cache path (decode+fold → rout·s_in·rin scale
  absorb → f16 → grouped f16 gemv) vs host: rel ~3e-4.
- Residual x-norms stay in a stable band across all 40 layers; logits finite
  and differentiated (top-5 gap ~3) at token 1.
- shared-expert path, router softmax, grouped-8 pointers, batched silu/add:
  structurally identical to beta's validated port.

**Not yet coherent:** first-token logits are only weakly prompt-conditioned
(the top token tends to a near-prior space/newline), and decode decays into a
token attractor (". 2 . 2" with reasoning on; "0 0 0" / "1 1 1" with
reasoning off) after ~3–5 tokens — the classic DeltaNet-recurrence divergence
profile (works while recurrent state ≈ 0, collapses once the S-matrix update
accumulates). Framework notes:
- thinking default-on for Qwen (family_default) → `open_think` assistant
  prefix; the daemon fail-closes with "open think span at end of generation"
  whenever a run ends inside the think block. For coherence tests set
  `hipfire config set reasoning.mode off`.
- Prefill for Escha correctly takes the per-token `forward_scratch` fallback
  (Escha layers are batched-prefill inadmissible by design on this tree).

**Next debugging steps (in priority order):**
1. ✅ RESOLVED (b967a721): the routed-FFN cache fill passed the whole stacked
   per-expert input scale to escham_apply_rowcol_scales_f32, so every expert
   was folded with expert 0's s_in·rin. Fix: slice per-expert
   `sub_offset(exp_idx*in_p, in_p)` like beta. MoE decode is now coherent.

Open questions: s_in/s_out role (dense), config [6] meaning, whether Qwen3.5
dense attention path on master can host an escha-coded wqkv/qkv with minimal
surgery, and the split of "int8 dense attention kept f32" perf fix vs a real
fp16/int8 gemv path.

## A5 — MoE coherent on the fresh tree (2026-09-04, commit b967a721)

Root cause of the A4 decode collapse: per-expert input-scale slicing in the
routed-FFN cache fill (escha_ffn.rs ~144) — the WHOLE stacked
`gate_up_in_scale [n_exp, dim]` / `down_in_scale [n_exp, mi]` was handed to
`escham_apply_rowcol_scales_f32`, whose kernel reads `in_scale[j], j < in_p`,
so every expert ≠ 0 was folded with expert 0's s_in·rin. The export's rin/s_in
are genuinely per-expert (rel diff vs e0 ≈1.3 e1 / ≈1.5 e200). Fixed by
binding per-expert sub_offset slices before the absorb (mirrors beta
qwen35.rs:14887/14913). The single-expert verification examples never caught
it because they fed already-sliced host uploads.

Verified coherent on gfx1151 (ROCm 10 / therock), reasoning off
(`hipfire config set reasoning.mode off`), temp 0:
- "The capital of France is" → "The capital of France is **Paris**." (9 tok,
  finish stop) — token ids byte-identical to the HFQ A3B control
  (`qwen3.6-35b-a3b.mq4r`): 760,6511,314,9338,369,2972,57590,159034,248046.
- Japan → Tokyo (fluent 40-token continuation); ocean haiku (proper form,
  stop); `def is_prime(n):` → clean ```python fence (40 tok).
- ~4–7 tok/s at temp 0 on Strix Halo gfx1151 (decode-bandwidth-bound, per
  the handoff's analysis).
Control: HFQ A3B MoE (qwen3.6-35b-a3b.mq4r) decodes perfectly on this tree
— the shared DeltaNet/attention/MoE + framework path is exonerated.
bin md5 (this milestone): hipfire be74e7222dc28905940dd2d5bbd736d8,
daemon d901b59967c77cafc564747c71449cc1.

## B1 — Dense loader + kernels + forward arms land; decode deterministic but incoherent (2026-09-04)

Commits on feat/escha-w2 (dense Phase B):
- 039f9023 LayerWeights DeltaNetEscha/FullAttnEscha + seal/free/screen/sweep
- 642b0856 DenseEschaSource coded-projection loader + carrier routing
  (quant_method=escha → EschaSource), embed/lm_head int8, per-proj upload of
  code + in_scale=rin.s_in + out_scale=rout.s_out (f32 folds; s_in/s_out MUST
  apply for exactness), escha_config sanity
- 43bf070b dense decode kernels (escha_dense_kernels.hip: rotate-in T128 /
  in-kernel decode-gemm / finalize) + rdna-compute dispatch + engine
- 73eebd14 exact funnel pairing fix (see below)
- b9328ad7 forward arms (DeltaNetEscha/FullAttnEscha) + kv dispatch wo:Option
- 13b5e664 debug instrumentation (env-gated)

Verified on /data/rocmfpx/Qwen3.8-27B-Escha-W2 (gfx1151):
- Model LOADS (64 layers, all coded projections, escha_config+shape checks).
- Per-projection decode is numerically EXACT: host trellis reference ==
  activation-side == fold-side reconstruction == GPU, rel ~2e-4 on every
  class (linear qkv/z/out, mlp gate K2/up K3/down K3, self_attn q/k/v/o;
  layers 0/3/40). examples/pin_funnel.rs + check_escha_dense.rs.
- Decode runs deterministically (no NaN, no faults after the kv-layer-index
  fix); ~1-2 tok/s decode (per-token decode-gemm, expected at this stage).

Funnel pairing discovery (73eebd14): llama.cpp's dense kernel uses an
overlapping-uint2 payload trick that does NOT match the safetensors layout of
this export. Empirically (real model tiles, both K): the exact codebook index
of weight (r,c) is funnel(hi = payload word w0-1, lo = word w0) >> (sp&31),
low 16 bits, w0 = (NW - (sp>>5)) mod NW, with a PLAIN word array. This is
decode_tiles-verified. (llama.cpp's GGUF presumably stores the code transposed,
so its pairing differs.)

Not yet coherent: decode echoes recent prompt tokens then decays to garbage
("Tokyo is the capital of" → "Tokyoyo巴尔onos…"; MoE control on the same
prompt → "**Japan"). Decode-position logits ARE prompt-dependent and contain
real word tokens in the top-k, but greedy picks collapse after 1-2 tokens —
the DeltaNet recurrent-state divergence profile (same signature as the MoE A4
failure pre-fix). Per-projection math is exonerated (fold==activation==GPU).

Next-debug candidates (unresolved):
- DeltaNet state quant (Q8) vs dims — try FP32 state for 5120-dim LA.
- Small-dense attention tensor convention (alpha/beta/A_log/dt_bias/conv1d/
  norm weights) vs the EschaLabs reference runtime (transform.py /
  gptoss_experts.py found under /home/mika/test-2/escha — authoritative
  forward: y = s_out ⊙ T128(T128(x·s_in·rin) @ w_bare)·rout, matching hipfire
  folds; bias optionally added, escha runtime APPLIES bias, llama.cpp does not).
- Whether the export's norms are (gamma-1) needing +1 (hipfire convention,
  MoE-proven) — both exports store layer norms ~N(0,0.03) and final norm as
  true gamma; greedy argmax is final-norm-scale-invariant.
- KV/attention path or a subtle integration detail between decode outputs and
  the conv/state kernels (all shared with the coherent MoE arms).

## B1b — debug round 2 evidence (2026-09-04, commit ab574ebc5)

Excluded as causes (all A/B tested on "Tokyo is the capital of", n=8, temp 0):
- DeltaNet state quant: Q8 vs FP32 → byte-identical garbage.
- KV mode: only q8 valid on the qwen35-dir site; the coherent MoE control runs
  the same q8 flash path with the same head_dim=256 and decodes correctly.
- Embed dequant: L0 input embed rms ~0.012-0.015 (sane), occasional exact-0
  entries (int8 quantization) but no structural error.
- Norm magnitudes: post-rmsnorm is unit-rms with a heavy tail (max ~20-40 is
  expected RMSNorm behavior for sparse large embed entries); decode inputs are
  properly scaled. Layer-0 decode output magnitudes are consistent with the
  verified per-projection math.
- Per-projection decode: fold-side == activation-side == GPU to rel 2e-4;
  examples/pin_funnel.rs proves the funnel pairing on real tiles (K2/K3).

Confirmed symptom: decode-position logits ARE prompt-dependent and contain real
top-k words (e.g. pos 12 top 57590 after "Paris"), but greedy output echoes the
prompt prefix ("Tokyo is the capital of" → "Tokyoyo巴尔onos…", "The capital of
France is" → "The capital ―ouravist…") then decays — the classic
recurrent-state divergence profile (same as MoE A4 pre-fix). Since every
projection is exact and all surrounding kernels are shared with the coherent
MoE + HFQ controls, the remaining suspect is a narrow integration/layout bug in
the DeltaNetEscha/FullAttnEscha arms (attention input placement, FA gate/
deinterleave subtlety, or conv/state ordering) rather than the decode math.

Suggested next step (not yet run): materialize one dense projection to a folded
f32 WeightTensor at load and route the PLAIN DeltaNet/FullAttn arm over it —
decodes coherent → bug in the escha arm; still broken → kernel interaction.

## B1c — q/k norm convention fix: first tokens now correct (2026-09-04, commit 888ef3b08)

BREAKTHROUGH via A/B norm-convention sweep: the dense export stores the FA
q_norm/k_norm RMSNorm weights as TRUE gamma (raw mean ~0.23 — NOT gamma-1
offsets like the MoE export and like the layer input/ffn norms, which are
stored ~N(0,0.03) and need +1.0). Loading q/k norms with bias 0.0 (raw) flips
the model from garbage to CORRECT first tokens:
  - "Tokyo is the capital of" → "Japan" (was "Tokyoyo巴尔onos…")
  - "The capital of France is" → "The capital cathedral" (grammatical)
FA attention was confirmed as the corrupting path pre-fix (passthrough test
gave natural "It looks"), and q/k raw fixed it.

Root rule: hipfire's +1.0 norm-bias convention (gamma-1 storage) applies to the
layer input/ffn norms in BOTH exports, but the dense export's q/k norms are
plain gamma. (Why the MoE export differs for q/k is an open question — its
q_norm may be stored gamma-1, or the EschaLabs pipelines differ between
exports.)

Remaining: after the correct first token, decode decays toward template/special
tokens (248068 <think>, 248046 <im_end>, newlines) — the DeltaNet recurrent-
state divergence profile. FA path is now producing correct early tokens; next
suspects are the LA DeltaNet path across decode steps (state persistence /
conv ordering) or a template/reasoning interaction for this VL-shaped model.

Debug toggles now in tree (all env-gated, MoE path untouched):
HIPFIRE_ESCHA_DENSE_TRACE / _LOGITS / _NO_FFN / _NO_ATTN / _NO_FA_GATE /
_STATE_FP32 / _RAW_NORMS (q/k raw is now the hard-coded default).

## B1d — France→Paris & Tokyo→Japan coherent (2026-09-04, commit 1ea28000)

Fix: linear_attn.norm.weight (gated-output norm, raw mean 0.87) is stored as a
gamma-1 offset in the dense export — loading with +1.0 (default) makes short
factual completions correct and STABLE where raw loading decayed into the fixed
"TokenNameertoolsuttle…" attractor:
  - "The capital of France is" → "Paris" (finish stop), deterministic ×2.
  - "Tokyo is the capital of" → "Japan" (finish stop).
  - MoE control (same prompts): "**Japan" / "**Tokyo". Dense answers differ in
    formatting but are semantically right for these.

Not yet coherent (remaining, narrow):
  - "The capital of Japan is" → miscues (MoE control: "Tokyo"); logits for that
    prompt sit on template tokens (think/EOS/newline) — conditioning on certain
    prompt tokens still imperfect.
  - Longer generations (>2-3 tokens) drift toward the fixed attractor for many
    prompts. Profile = DeltaNet recurrent-state divergence, not weight decode.
  - Note: LA gated-norm +1 vs MoE raw — dense and MoE exports store this tensor
    differently; verified by A/B not by theory. Same class of finding as the
    q/k-norm (dense stores q/k true-gamma, layer/LA-gated norms gamma-1).

All Phase-B work is on feat/escha-w2; MoE path byte-identical (control rerun
coherent). Debug toggles (env-gated): HIPFIRE_ESCHA_DENSE_TRACE/_LOGITS/_NO_FFN/
_NO_ATTN/_NO_FA_GATE/_STATE_FP32/_RAW_NORMS. examples/pin_funnel.rs +
check_escha_dense.rs remain the decode oracles.

## B1e — final evidence (2026-09-04): partial conditioning confirmed

Directional A/B (all temp 0, deterministic):
  "The capital of France is" → Paris (correct)
  "Paris is the capital of"  → France (correct)
  "Tokyo is the capital of"  → Japan (correct)
  "The capital of Japan is"  → miscue; "Japan is the capital of" → miscue;
  "The capital of Germany is" → miscue.
France/Paris work both directions; Japan/Germany subjects fail both directions.
This asymmetry — some short completions correct via next-token prior, others
exposing the loss — confirms the model's CONTEXT isn't reliably integrated for
all prompt tokens (DeltaNet state or KV over the prompt), NOT a decode-math or
norm-convention error (those are fixed/proven). The fixed attractor after ~2-3
generated tokens ("TokenNameertoolsuttle…") is the same recurrent-divergence
signature.

Phase B status: M1 LOADS; M2 kernels EXACT + arms run with several correct
deterministic short completions; coherent-decode GATE for the full model is the
remaining M2 item (narrow integration bug, candidates documented above);
M3 (prefill/throughput) not started.
