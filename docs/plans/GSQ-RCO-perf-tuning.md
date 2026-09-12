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
| `/tmp/gsqrco-stage4.hfq` + Item-2/2b dual-row GEMVs | 12.99 GB | 9.68² | 9.6 | **11.7**³ |

¹ PPL = wikitext2 200 KB slice, `flash_prefill_quality` ctx512/chunks16/
stride8 (512 scored), md5 `538eb71f…`.
² PPL is prefill-only (batched GEMMs); the decode-GEMV change does not
touch it — see §3 Item 2.
³ Fresh-process ×3 median, prompt md5 `2c8abce9…`, daemon md5
`e6d1f7b7…` (Item-2b build). §1 rows above are single-run directional.

Single-run numbers are directional only; any claim needs the fresh-
process protocol in §5. Post-Item-2b, native I-quant decode (11.7) is
now within ~1.2x of the tuned HFQ4/MQ3 GEMV family (13.8-14.2)
instead of ~2x slower.

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

### Item 2 — tune the grid-lookup GEMV kernels (DONE 2026-09-12, decode 7.1 → 10.2)

Pick ONE lever per the kernel-tuning skill (profile → ISA → fresh-
process measure):
- multi-row (2 rows/warp like `gemv_hfq4g256_residual` dual-row)
- K-tile / group prefetch (hoist x as float4 like the HFQ4 kernel)
- LDS staging of the codebook grid (512-entry IQ3S grid, 256-entry
  IQ3XXS, 512-entry IQ2XS u64) so lookups hit LDS not constant
- wave-size / launch-bounds changes
- WMMA for the 4-bit+ dtypes (IQ4_XS / Q4_K) in prefill GEMMs is a
  separate track; the plan's non-goal is DFlash.

**Lever chosen (after profile + ISA):** dual-row + contiguous-8-chunk
remap. Per-kernel attribution (internal profiler, HIPFIRE_GRAPH=0,
32-token decode, gfx1151): the 8 I-quant GEMV kernels (plain +
residual) = 91.7% of serialized decode kernel time — gemv_iq3_s 31.7%
(364 µs/call), gemv_iq3_xxs 27.5% (485 µs/call), gemv_iq4_xs 18.5%
(305 µs/call), gemv_q4k 5.5% (237 µs/call) + residual arms. ISA: 48
VGPR, no spills, ~33 loads + ~80 integer decode ops per 8 elements —
instruction-bound, NOT bandwidth-bound (73 GiB/s vs the tuned
`gemv_hfq4g256` at 130 GiB/s / 23 µs/call on the same shape).

The fix: remap each thread to a CONTIGUOUS 8-element chunk of a
256-block, which collapses the IQ3_S decode to 2 grid lookups + 1 qh +
1 scales + 1 signs + 2 qs bytes + 2 float4 x loads per 8 elements (the
scalar kernel did 8 grid + 8 qs + 8 qh + 8 scales + 8 signs + 8 scalar
x), and process TWO rows per 32-thread wave so x is shared. Same
remap applies to IQ3_XXS (2 grid + 1 aux32 + 1 ksigns + 2 qs) and
IQ4_XS (8 qs nibbles as one u64). FP-reduction order changes (per-lane
partials are contiguous chunks, not group-strided) → greedy output can
diverge on borderline argmaxes (observed once per ~300 tokens:
"The sun" vs "The Sun"); max_abs vs CPU stays ~4e-7. **This is a
measured trade, not a bug — PPL/serve are unchanged (below).**

**Shipped (gfx1151-only, env kill-switches `HIPFIRE_{IQ3S,IQ3XXS,IQ4XS}
_DUALROW=0`):** `kernels/src/gemv_iq{3_s,3_xxs,4_xs}_dualrow.hip` +
`*_dualrow_residual.hip`, rdna-compute methods + `gemv_iq_bytes`
profile helper, dispatch arms in `families/gemv.rs`. Q4_K dual-row was
**REJECTED** (measured 310 µs/call vs scalar 237 µs/call — the remap's
sub-group branch + wider live range costs more than the u64 qbyte load
saves; kept opt-in behind `HIPFIRE_Q4K_DUALROW=1`).

**Measured (gfx1151, fresh-process ×3, prompt md5 `2c8abce9…`, daemon
md5 `dd765164…` [A/B also verified on the fmt-only-identical `05b9d865`],
hipfire md5 `d73b577d…`, model md5 `d148a992…`):**
decode OFF (scalar) 7.2/7.1/7.1 → median **7.1 tok/s**; decode ON
(dual-row) 10.2/9.8/10.2 → median **10.2 tok/s** (+43%). Serialized
decode kernel time 4114 → 2877 ms (−30%): gemv_iq3_s 364→235 µs/call,
gemv_iq3_xxs 485→238, gemv_iq4_xs 305→268, residual arms 186→116 /
151→129 / 224→109. **Success bar met (≥ 10 tok/s).**

**Correctness:** `test_iq_dualrow_parity` (new, real GGUF tensors) ALL
PASS — every dual-row kernel vs CPU decoder max_abs ~1e-6, odd-M tail
clean, residual dual-row vs scalar residual ~6e-8. Serve battery
(`--mode battery --thinking off --max-tokens 256`): avg decode 10.1
tok/s, 0 attractor / 0 empty / 1 runaway (same runaway profile as the
Item-1 baseline — the "reason" prompt hits max-tokens). PPL
(wikitext2 200 KB slice, ctx512/chunks16/stride8): structurally
unaffected (prefill uses the batched GEMMs, not decode GEMVs);
measured 9.68 → 9.68-equivalent (see §7 record).

**Verdict:** ship it (gfx1151). The +43% decode closes the gap to the
MQ3/bridge siblings and meets the plan's success bar. Greedy parity is
NOT preserved (documented above); any future admission must weigh the
1e-7 reordering against the 43% win.

### Item 2b — post-Item-2 follow-up (2026-09-12, decode 10.2 → 10.9)

Attempts to close the remaining ~35% gap to the MQ3/bridge siblings
(13.8-14.2) after Item 2 shipped. Profile/ISA root-cause: the dual-row
kernels are latency-bound (not bandwidth- or instruction-bound); the
HFQ4 23 µs/call reference is not apples-to-apples (different M, no
codebook). Levers tried on the dual-row kernels:

- **Coalesced 4-block LDS staging in gemv_iq3_s_dualrow (+ residual):
  SHIPPED.** The scalar kernel's scattered u8/u16 weight loads (qs/qh/
  scales/signs at different offsets) coalesce poorly. Staging 4 blocks
  (2 rows × 4 × 110 B = 880 B) into LDS with u32 loads + one sync per
  group turns the scattered reads into ds_read. 235 → 204 µs/call
  (plain, −13%), 116 → 108 (residual).
- **Coalesced 2-block LDS staging in gemv_iq3_xxs_dualrow (+ residual):
  SHIPPED.** 4-block staging only gave +2% (VGPR 38→72); 2-block
  (2 rows × 2 × 98 B = 392 B) at 86 VGPR / 0 spills gives 238 → 211
  µs/call (plain, −11%), residual 109 → ~100.
- **Coalesced 2-block LDS staging in gemv_iq4_xs_dualrow (+ residual):
  SHIPPED.** 4-block staging spilled (96 VGPR / 32 spills); 2-block
  (2 rows × 2 × 136 B = 544 B) fits at 79 VGPR / 0 spills. On top of the
  KV-table LDS: 243 → 209 µs/call (plain, −14%), 118 → 110 (residual).
  Total iq4_xs improvement vs the Item-2 baseline: 268 → 209 µs/call
  (−22%).
- **LDS-stage the KV table in gemv_iq4_xs_dualrow (+ residual): SHIPPED.**
  The 16-entry kvalues_iq4nl lookups were 16 `global_load_b32` per
  iteration (8/row × 2 rows); staging the 64-byte table in LDS once per
  kernel turns them into `ds_read`. 268 → 243 µs/call (plain), 129 → 118
  (residual). Parity unchanged (ALL PASS).
- **LDS-stage the IQ3_S codebook grid: REJECTED.** 512-entry grid staged
  in LDS once per kernel: 235 → 247 µs/call (−5%). The grid was already
  L1-hot (2 KB), so the setup + ds_read cost more than the L1-hit global
  loads.
- **LDS-stage the IQ3_XXS grid + ksigns: REJECTED (flat).** 236 vs 238
  µs/call — no measurable effect, reverted.
- **`#pragma unroll 4` on the gemv_iq3_s bi-loop (4-block K-tile):**
  235 → 223 µs/call (+5%) but VGPR 41 → 81 (occupancy 16 → 12 waves), so
  the ILP gain is largely offset; combined with LDS grid staging it
  regressed. Not shipped.

Measured (gfx1151, fresh-process ×3, prompt md5 `2c8abce9…`): decode
OFF 7.1/7.1/6.7 → ON 11.5/11.7/11.7 → median **11.7 tok/s** (+65%
over the Item-2 scalar baseline 7.1). Serialized decode kernel time
2877 → ~2549 ms.

Remaining headroom is small and structural (the I-quant codebook/sign/
scale decode is fundamentally ~4× more instructions/element than HFQ4's
direct 4-bit decode). Documented rejections above; no further lever is
queued without a new mechanism (e.g. packed `v_cvt_f32_i8`).

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

- Branch `exp/gsq-rco-iq3s`; Stage 4 commit `b11d0cef0`; Item 2
  commit (this work) — see §3 Item 2.
- Stage-4 file `/tmp/gsqrco-stage4.hfq`; bridge
  `/tmp/gsqrco-iq3s.hfq`; PPL slice `/tmp/ppl_slice.txt`
  (md5 `538eb71f…`); PPL records `/tmp/fpq16_stage4_final.bin`
  (PPL 9.6767); battery `/tmp/gsqrco_harness_stage4.log`.
- Item-2 A/B (fresh-process ×3, prompt md5 `2c8abce9…`, daemon md5
  `dd765164…`): `.codeinsight+research/gsq-rco/ab_{off,on}_{1,2,3}.{json,err}`
  (off 7.2/7.1/7.1 → on 10.2/9.8/10.2); decode profiles
  `prof_dualrow*.log`; parity `parity_all.log` (ALL PASS); battery
  `gsqrco_harness_dualrow_all.log` (avg decode 10.1, 0 attractor/empty).
- Prompt fixture `benchmarks/prompts/gsq_rco_why_sky_blue.txt`
  (md5 `2c8abce9…`).
- User's decode comparison (2026-09-12): see §1 table.

## 8. Non-goals

- DFlash / spec decode (AR must be correct + tuned first, per the
  arch-port skill).
- Changing the release's RCO allocation (authoritative).
- IQ2_XS/IQ2_XXS native re-enable (decode-recurrence instability,
  plan Stage 3 — separate problem, not a perf issue).
- Prefill GEMM WMMA tuning for I-quant (separate track; prefill is
  already FASTER than the bridge: 9.6 vs 7.2).