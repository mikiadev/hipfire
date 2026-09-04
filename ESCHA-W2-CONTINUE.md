# Escha-W2 — continue here (fresh-context handoff)

Branch: **`feat/escha-w2`** in `/home/mika/git/hipfire`. Working tree clean at
`baf35dd17` (39 commits vs upstream master 8cd15a62).

This is the one-file entry point for a fresh session. The full investigation
trail lives in `escha-port-status.md` (repo root, committed); perf evidence in
`docs/perf-checkpoints/2026-09-04-escha-w2-moe-qwen36-35b-a3b-gfx1151.md` and
`docs/perf-checkpoints/2026-09-04-escha-w2-dense-qwen38-27b-gfx1151.md`.

---

## What is done (verified coherent on gfx1151 / ROCm 10 / TheRock)

### A. MoE — `/data/rocmfpx/Escha-W2` (Qwen3.6-35B-A3B-Escha-W2, eschamoe)
Replayed the working port from scratch `/home/mika/git/hipfire-beta` (da09de7)
onto the fresh modular tree: escham trellis/3INST/fold/grouped kernels +
`rdna-compute::escham` dispatch, `EschaSource` loader, `LayerWeights`
DeltaNetEschaMoe/FullAttnEschaMoe arms, FFN engine `qwen35/escha_ffn.rs`.
Coherent: France→"**Paris**", Japan→Tokyo, haiku, code fences, 128-token
rainbow explanation at **10.6 tok/s**.

### B. Dense — `/data/rocmfpx/Qwen3.8-27B-Escha-W2` (Qwen3.8 27B, escha)
Full in-kernel decode pipeline (no full-size dequant): `DenseEschaSource`
loader, `LayerWeights` DeltaNetEscha/FullAttnEscha, HIP decode-gemm kernels
(`kernels/src/escham/hip/escha_dense_kernels.hip`: rotate-in T128 →
in-kernel tile decode → finalize) + `rdna-compute/src/escha_dense.rs` +
`qwen35/escha_dense_{decode,forward}.rs`. Per-projection decode **exact** vs
host/EschaLabs (rel 2e-4). Coherent multi-token prose/haiku at **~1.6–2.5
tok/s** (unoptimized per-token decode-gemm).

---

## How to build + run (mandatory env)

```bash
cd /home/mika/git/hipfire
nix develop .#therock --command bash -c '
  cargo build --release -p hipfire-arch-qwen35 -p hipfire-daemon -p hipfire-cli
  export HIPFIRE_HOME=/tmp/hf-$(date +%s)      # FRESH per run (stale singleton + config live here)
  mkdir -p $HIPFIRE_HOME
  ./target/release/hipfire config set reasoning.mode off >/dev/null 2>&1
  HIPFIRE_LOCAL=1 HIPFIRE_DAEMON_BIN=$PWD/target/release/daemon \
    ./target/release/hipfire run /data/rocmfpx/Qwen3.8-27B-Escha-W2 -n 128 "Why is the sky blue?"
'
```

Why these env vars (all three are load-bearing):
- `HIPFIRE_LOCAL=1` + `HIPFIRE_DAEMON_BIN=$PWD/target/release/daemon` — a
  **stale system-installed serve** (nix store, user `hipfire`) listens on
  :11435 and would otherwise intercept `run` with an OLD binary that fails
  config parsing (`unknown configuration key 'serve.continuous_batch_size'`).
- `HIPFIRE_HOME=/tmp/<fresh>` — the daemon is a per-home singleton
  (`daemon.pid`) and older configs in a reused home can break startup.
- `reasoning.mode off` — the Qwen chat default opens a `<think>` block; the
  daemon fail-closes ("open think span at end of generation") when a run ends
  inside it. This is a framework artifact, not a model bug.

MoE runs identically with `/data/rocmfpx/Escha-W2`.

---

## Key files (dense + MoE)

| Area | Path |
|---|---|
| Handoff / status trail | `escha-port-status.md` (repo root) |
| MoE FFN engine | `crates/hipfire-arch-qwen35/src/qwen35/escha_ffn.rs` |
| MoE loader | `crates/hipfire-arch-qwen35/src/qwen35/escha_load.rs` |
| Dense loader (+norms) | `crates/hipfire-arch-qwen35/src/qwen35/escha_load.rs` (DenseEschaSource) |
| Dense decode engine/ref | `crates/hipfire-arch-qwen35/src/qwen35/escha_dense_decode.rs` |
| Dense forward arms | `crates/hipfire-arch-qwen35/src/qwen35/escha_dense_forward.rs` |
| Dense kernels (HIP) | `kernels/src/escham/hip/escha_dense_kernels.hip` |
| Dense dispatch | `crates/rdna-compute/src/escha_dense.rs` |
| MoE kernels (HIP) | `kernels/src/escham/hip/{escham_moe_decode_trellis,escham_moe_fold,escham_moe_grouped_kernels}.hip` |
| MoE dispatch | `crates/rdna-compute/src/escham.rs` |
| Host decode reference + tests | `crates/hipfire-arch-qwen35/src/escham_decode.rs` |
| Oracles (GPU-vs-host) | `crates/hipfire-arch-qwen35/examples/{compare_decode,check_escha_ffn,pin_funnel,check_escha_dense,check_wrow_norm}.rs` |

---

## Milestones recap (root causes worth remembering — do NOT re-investigate)

1. **Decode model**: EXL3 trellis + 3INST codebook + tensor-core-perm — pure
   function of the int16 codes, no codebook file. GPU decode == host ==
   EschaLabs (rel 2e-4). Kernel math in `escham_decode.rs` / the .hip files.
2. **Fold convention**: `y = s_out ⊙ T128(T128(x·s_in·rin) @ w_bare) · rout`
   (scales OUTSIDE the blockwise-128 WHT; MoE folds rout·s_in·rin into cached
   fp16 weights — an exact identity).
3. **MoE per-expert scale bug** (b967a721): the routed-FFN cache fill passed
   the whole stacked `[n_exp, in]` scale tensor to `escham_apply_rowcol_scales`
   (kernel reads the first `in_p` elements) — every expert folded with expert
   0's s_in·rin. Fix: per-expert `sub_offset(exp_idx*in_p, in_p)`.
4. **Dense funnel pairing** (73eebd14): llama.cpp's overlapping-uint2 payload
   trick does NOT match this export's safetensors layout. Exact index =
   `funnel(hi=word w0−1, lo=word w0) >> (sp&31)` over a plain word array.
   Pinned empirically on real model tiles (`examples/pin_funnel.rs`).
5. **Dense norm conventions** (888ef3b08, 1ea28000): `q_norm`/`k_norm` are
   TRUE gamma (raw, no +1); the LA gated-output norm (`linear_attn.norm`)
   is gamma−1 (+1). Layer input/ffn norms are gamma−1 in both exports.
6. **Dense multi-token decay root cause** (5bb373af): AR hipGraph
   capture/replay engaged on escha-dense (`use_graph` excluded
   `is_escham_moe` but not `is_escha_dense`); the per-projection decode allocs
   pool scratch per call (capture-unsafe) → replay diverged at decode token ~3.
   Fix: add `&& !config.is_escha_dense` to the `use_graph` predicate in
   `crates/hipfire-arch-qwen35/src/qwen35/forward.rs`.

Env-gated debug toggles (all harmless, MoE path untouched):
`HIPFIRE_ESCHA_DENSE_TRACE/_LOGITS/_NO_FFN/_NO_ATTN/_STATE_FP32/_RAW_NORMS`,
`HIPFIRE_ESCHA_DEBUG` (MoE probes were stripped from escha_ffn but the env
read sites may remain).

---

## Next steps (M3 / follow-up, in order)

1. **Hoist dense decode scratch**: the per-projection decode
   (`escha_dense_decode_proj`) allocs `u` + `partial` pool tensors per call.
   Move them into `Qwen35Scratch` (like the MoE down-expand buffers) so the
   escha-dense path can rejoin AR hipGraph capture (drop the 5bb373af
   exclusion after verifying).
2. **Dense prefill + throughput**: add the batched / WMMA prefill path (port
   llama.cpp-escha `2940b807c` tensor-core kernels to HIP WMMA
   `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, see
   `kernels/src/gemm_f16_wmma.hip`). Currently decode is ~1.6–2.5 tok/s
   (per-token decode-gemm, no batching); prefill needs the multi-row
   decode-gemm (`escha_matmul_dense_tiled`) + slice-of-reduction.
3. **MoE throughput**: beyond 10.6 tok/s requires the fp16/int8 dense
   attention path (handoff note: dense attention is int8→f32 at load,
   ~4.5 GB/token f32 traffic on the MoE).
4. **Dense decode scratch on host** — the export carries per-projection
   `escha_config[6]` (6 floats near 1.0, not the MoE doc's
   `[tile,K,bits,mcg,...]`) and `s_in/s_out` ≈ ±0.6% around 1 that MUST be
   applied (the loader folds s_in·rin / s_out·rout). llama.cpp ignores
   s_in/s_out; we apply them for exactness.

External references: llama.cpp-escha fork
`/home/mika/git/llama.cpp-escha` (branch `escha-w2-dense`, commits 2a238a40d +
2940b807c); EschaLabs authoritative reference runtime `/home/mika/test-2/escha/`;
AMD HIP validation of the same dense path: `/home/mika/git/escha-amd-port`
(gfx1030, yaminerl) — confirms fp32-FMA decode-gemm runs on AMD and the
portable codebook spelling is exact under HIP.

---

*Last updated 2026-09-04 (B5 dense milestone).*
