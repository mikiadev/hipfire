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

Open questions: s_in/s_out role (dense), config [6] meaning, whether Qwen3.5
dense attention path on master can host an escha-coded wqkv/qkv with minimal
surgery, and the split of "int8 dense attention kept f32" perf fix vs a real
fp16/int8 gemv path.
