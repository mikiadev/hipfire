# GSQ-RCO native decode perf tuning (IQ3_S family)

**Status:** plan record (2026-09-12). Branch `exp/gsq-rco-iq3s` @
`b11d0cef0` (Stage 4 shipped). This file is the working plan for
closing the native-I-quant decode speed gap; it is NOT a product
claim. All numbers are `measured` on the stated fixture.

**Why this exists:** the Stage-4 native hybrid serves the release's own
RCO allocation at 12.99 GB / PPL 9.68, but its DECODE is ~2x slower
than the re-quant bridge / MQ3 siblings. The plan's Stage-2 note
already flagged this: *"Decode tok/s drops on IQ3_S (9.2 vs 12.3
Stage-1) — grid lookup is the cost; perf tuning is a follow-up."*
This file is that follow-up.

---

## 1. Measured decode gap (2026-09-12, gfx1151, single run each)

Prompt: `Why is the sky blue?` (prompt md5 `2c8abce9…`, 18 tok,
`--max-tokens` default, greedy):

| file | size | PPL¹ | prefill | decode |
|---|---|---|---|---|
| `~/.hipfire/models/qwen3.8-27b.mq3` | 12.62 GB | 10.54 | 39.0 | 14.2 |
| `/tmp/gsqrco-iq3s.hfq` (bridge, all-HFQ4G256 re-quant) | 14.97 GB | 10.32 | 7.2 | 13.8 |
| `/tmp/gsqrco-stage4.hfq` (native hybrid) | 12.99 GB | **9.68** | 9.6 | **7.0** |

¹ PPL = wikitext2 200 KB slice, `flash_prefill_quality` ctx512/chunks16/
stride8 (512 scored), md5 `538eb71f…`.

Single-run numbers are directional only; any claim needs the fresh-
process protocol in §5. The direction is unambiguous: native I-quant
decode is ~2x slower than the tuned HFQ4/MQ3 GEMV family.

## 2. Root cause (three compounding costs)

1. **Grid-lookup GEMV kernels are scalar.** `gemv_iq3_s` /
   `gemv_iq4_xs` / `gemv_q4k` / `gemv_iq3_xxs` (kernels/src/) are
   32-thread warps with per-element codebook lookups
   (`IQ3S_GRID[grid_idx]`, sign tables, nibble scales). The HFQ4 GEMV
   (`gemv_hfq4g256_residual.hip`) is arch-tuned (dual-row, float4 x
   loads, buffer SRDs, gfx1151 k4096 variants). This is the dominant
   cost and is a kernel-tuning project (§3 item 2), not a wiring bug.
2. **No fused residual GEMV for I-quant.** `for_gemv_residual(IQ3S)`
   returns Err (crates/hipfire-dispatch/src/types.rs:823), so
   `Step::GemvResidual` with a native I-quant wo falls to the
   un-fused path (crates/hipfire-dispatch/src/pipeline/steps.rs:840):
   plain GEMV into tmp + `add_inplace`. DeltaNet out_proj decode is
   therefore *un-permute → plain GEMV → add* = 3 launches vs the
   bridge's 1 fused residual GEMV. FA o_proj (also native I-quant in
   the Stage-4 file) is 2 launches vs 1. **This is §3 item 1.**
3. **Un-permute launch per LA layer.** `vhead_unpermute_f32_batched`
   (kernels/src/vhead_unpermute.hip) is one tiny launch per DeltaNet
   layer (6144 floats). ~0.3% of decode time — not the main cost, but
   it disappears if a future kernel folds the PERM into the GEMV.

## 3. Work items

### Item 1 — fused residual I-quant GEMV (DONE 2026-09-12, modest win)

Goal: collapse the un-fused `GemvResidual` fallback to a single
launch for the four out_proj / FA-o_proj dtypes (IQ3_S, IQ4_XS,
Q4_K, IQ3_XXS — 22+16+6+4 out_proj + FA o_proj tensors in the
Stage-4 file).

**Shipped:** new kernels `kernels/src/gemv_iq{3_s,4_xs,3_xxs}
_residual.hip` + `gemv_q4k_residual.hip` (base GEMV with the
epilogue `y[row] = sum` → `y[row] += sum`; identical W·x FMA order
→ greedy parity), rdna-compute methods, `KernelKey` variants +
`for_gemv_residual` + `dispatch_residual` arms. The standard
`Step::GemvResidual` fused path (steps.rs) now serves I-quant wo in
one launch.

**Measured (gfx1151, fresh-process ×3, prompt md5 `2c8abce9…`,
daemon md5 `1f3bf177…`):** decode 7.1/7.2/7.3 → median **7.2 tok/s**
vs 7.0 pre-fused (single run). +0.2 tok/s — smaller than the
0.5-1 estimate; the grid-lookup GEMV itself dominates, so removing
the extra launches is worth only ~3%. Coherence unchanged: battery
greedy 5/5 (0 attractor / 0 empty), "The capital of France is
Paris", residual parity harness max_abs 2.6e-5.

**Verdict:** ship it (correctness-neutral, small win, no risk), but
the decode gap only closes with Item 2.

### Item 2 — tune the grid-lookup GEMV kernels (the real win, bigger)

Pick ONE lever per the kernel-tuning skill (profile → ISA → fresh-
process measure):
- multi-row (2 rows/warp like `gemv_hfq4g256_residual` dual-row)
- K-tile / group prefetch (hoist x as float4 like the HFQ4 kernel)
- LDS staging of the codebook grid (512-entry IQ3S grid, 256-entry
  IQ3XXS, 512-entry IQ2XS u64) so lookups hit LDS not constant
- wave-size / launch-bounds changes
- WMMA for the 4-bit+ dtypes (IQ4_XS / Q4_K) in prefill GEMMs is a
  separate track; the plan's non-goal is DFlash.

Start with per-kernel attribution (rocprof or the kernel-atlas skill)
to confirm which GEMV dominates before writing kernels. Success bar:
decode ≥ 10 tok/s on the §5 fixture without a PPL/serve regression.

### Item 3 (optional) — fold the V-head PERM into the out_proj GEMV

A `gemv_iq*_residual_gguf` variant that reads x in ENGINE order and
applies `PERM[e]` internally removes the un-permute launch entirely
(DeltaNet out_proj: 1 launch). Only worth it after Item 1 shows the
launch overhead matters; the un-permute is ~0.3% today.

## 4. Environment (this machine — read before measuring)

- GPU: gfx1151 (AMD Radeon 8060S, 137 GB VRAM). GPU lock:
  `source scripts/gpu-lock.sh && gpu_acquire "<branch>"` (manual).
- Build: `cargo build --release` (daemon = `target/release/daemon`,
  CLI = `target/release/hipfire`). Examples need explicit
  `cargo build --release --example <name>`.
- **Kernel JIT is broken in this Nix env** (clang++ can't find
  cstdlib). Workaround for any process that JITs kernels (daemon,
  examples): export
  `CPATH=/nix/store/qxaq7jz61a6zkr2mq49i0zvqip2m2jj8-gcc-15.2.0/include/c++/15.2.0:/nix/store/qxaq7jz61a6zkr2mq49i0zvqip2m2jj8-gcc-15.2.0/include/c++/15.2.0/x86_64-unknown-linux-gnu:/nix/store/qxaq7jz61a6zkr2mq49i0zvqip2m2jj8-gcc-15.2.0/include/c++/15.2.0/backward:/nix/store/qxaq7jz61a6zkr2mq49i0zvqip2m2jj8-gcc-15.2.0/lib/gcc/x86_64-unknown-linux-gnu/15.2.0/include:/nix/store/39l6rhxg9qb07q3862p4zcr2s4146p1r-glibc-2.40-224-dev/include`
  (do NOT use `HIPFIRE_HIPCC_EXTRA_FLAGS` for this — it changes the
  kernel cache hash and forces a full recompile).
- **`hipfire run` prefers the stale `~/.hipfire/bin/daemon`** over
  `target/release/daemon`. Always run with
  `HIPFIRE_DAEMON_BIN=/home/mika/git/hipfire/target/release/daemon`
  when testing local changes.
- Model files: source GGUF
  `/data/rocmfpx/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`; Stage-4 hybrid
  `/tmp/gsqrco-stage4.hfq` (md5 `d148a992…`, 12,987,317,248 B);
  bridge `/tmp/gsqrco-iq3s.hfq` (md5 `3c60064f…`); ladder
  `~/.hipfire/models/qwen3.8-27b.{mq3,mq4,mq6}`.
- PPL tool: `./target/release/examples/flash_prefill_quality
  <model.hfq> /tmp/ppl_slice.txt <out.bin> --ctx 512 --chunks 16
  --stride 8`; PPL = `exp(mean(nll))` over the 44-byte records
  (u32 next_tok | f32 nll | f32 lse | u32 top8[8]).

## 5. Measurement protocol (mandatory for any claim)

- Fresh process per run; ≥3 runs; report the median.
- Byte-identical prompt as a committed file or recorded md5 (the
  "Why is the sky blue?" fixture md5 `2c8abce9…`; the serve battery
  uses its own built-in prompts, md5s recorded by the harness).
- Record binary md5s: `md5sum target/release/daemon
  target/release/hipfire` (current: daemon `6a2c90a8…`,
  hipfire `a556c173…`) and the model md5.
- Decode via the daemon path the product uses: `hipfire run` (with
  `HIPFIRE_DAEMON_BIN`, §4) or `scripts/serve_harness.py --mode
  battery`. Do NOT bench through the demo harnesses.
- Eyeball the decoded text; a suspiciously tight stddev or a high
  tok/s on garbage is a single-token attractor failure, not a win.

## 6. Key code locations

- Kernels: `kernels/src/gemv_iq3_s.hip`, `gemv_iq4_xs.hip`,
  `gemv_q4k.hip`, `gemv_iq3_xxs.hip` (+ `_batched` prefill GEMMs);
  residual reference: `kernels/src/gemv_hfq4g256_residual.hip`;
  un-permute: `kernels/src/vhead_unpermute.hip`.
- rdna-compute GEMV arms: `crates/rdna-compute/src/gemv.rs`
  (gemv_iq3_s at ~452; gemv_hfq4g256_residual at ~7425);
  kernel consts: `crates/rdna-compute/src/kernels.rs`.
- Dispatch: `crates/hipfire-dispatch/src/types.rs`
  (`for_gemv_residual` :823, `KernelKey` :283);
  `crates/hipfire-dispatch/src/families/gemv.rs`
  (`dispatch_residual` :546);
  `crates/hipfire-dispatch/src/pipeline/steps.rs` (:840, the
  un-fused fallback that Item 1 replaces);
  `crates/hipfire-dispatch/src/tables/gemv_table.rs`
  (`register_residual`).
- Decode wo paths (all already un-permute-aware; Item 1 only changes
  the residual fusion): lowered executor RESID_WO
  (`crates/hipfire-arch-qwen35/src/qwen35/forward.rs`, the
  `q35_op::RESID_WO` arm), hand arms in the same file, batched
  prefill `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs`
  (`batch_chunk_delta_net_attn`, `dispatch_batched_gemm_epilogue`).
- Harnesses: `crates/hipfire-quantize/examples/test_outproj_decode.rs`
  (decode GEMV + un-permute vs CPU, max_abs 2.6e-5),
  `crates/rdna-compute/examples/test_vhead_unpermute.rs` (2/2 exact).
- Plan record / admission: `docs/plans/GSQ-RCO-plan.md`,
  `docs/admissions.yml`.

## 7. Artifacts

- Branch `exp/gsq-rco-iq3s`; Stage 4 commit `b11d0cef0`.
- Stage-4 file `/tmp/gsqrco-stage4.hfq`; bridge
  `/tmp/gsqrco-iq3s.hfq`; PPL slice `/tmp/ppl_slice.txt`
  (md5 `538eb71f…`); PPL records `/tmp/fpq16_stage4_final.bin`
  (PPL 9.6767); battery `/tmp/gsqrco_harness_stage4.log`.
- User's decode comparison (2026-09-12): see §1 table.

## 8. Non-goals

- DFlash / spec decode (AR must be correct + tuned first, per the
  arch-port skill).
- Changing the release's RCO allocation (authoritative).
- IQ2_XS/IQ2_XXS native re-enable (decode-recurrence instability,
  plan Stage 3 — separate problem, not a perf issue).
- Prefill GEMM WMMA tuning for I-quant (separate track; prefill is
  already FASTER than the bridge: 9.6 vs 7.2).