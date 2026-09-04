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
