# GSQ-RCO decode: next levers after Item 2b (pwilkin/strix-halo cross-read)

**Status:** analysis record (2026-09-14). Companion to
`docs/plans/GSQ-RCO-perf-tuning.md`. Branch `exp/gsq-rco-iq3s` @ `bceb29885`.
Nothing here is a product claim; every estimate is a directional range to be
measured with the §8 protocol.

**Inputs read for this analysis:**
- `~/git/pwilkin-strix-halo` — the pwilkin/strix-halo **installer repo** the
  user pointed at (pins the whole tuned stack; see §1).
- `/home/mika/git/pwilkin-llama.cpp` — the actual **llama.cpp fork** it pins
  (`pwilkin/llama.cpp:strix-halo` @ `d67d58836b4987fa9dbc03b3d87c95eae6ceaddf`,
  cloned for this analysis).
- hipfire ground truth: `.codeinsight+research/gsq-rco/prof_blk.log` (32-token
  decode profile, HIPFIRE_GRAPH=0), `prof_dualrow*.log`, `ab_*.sh`, the stage-4
  tensor table (`hfq_dump`), and the current decode/dispatch sources.

## 0. TL;DR — ranked recommendations

| # | Lever | Kind | Est. decode impact | Effort |
|---|---|---|---|---|
| 1 | **q8_1 + dp4a I-quant GEMV** — SHIPPED as prototype, MEASURED | kernel | kernel 1.5–1.6×; E2E +3.4% (IQ3_S only) | high |
| 2 | **Split-K for the latency-bound dual-row kernels** | kernel | unknown (untried) | medium |
| 3 | **DFlash2/spec decode** (pwilkin headline: 26 t/s) | engine | ×1.5–2.5 on code/reasoning | high (plan non-goal) |
| 4 | Port q8dot to IQ4_XS / IQ3_XXS (43% more of decode kernel time) | kernel | +? (memory-ceiling-limited) | medium |
| 5 | Fused I-quant QKVZA / GATE_UP | kernel | +2–4% | medium |
| 6 | TOP_K wave32-native (pwilkin) | sampling | spec-only | low |
| — | Prefill: **bf16 WMMA dequant GEMM** (pwilkin) | prefill | prefill 9.5 → 30–50 t/s | high |

> **Measured update (2026-09-14):** recommendation #1 is implemented and
> measured. The kernel win is real (1.46–1.62×/call on real IQ3_S tensors —
> see §3.1a) but the end-to-end decode gain is only **+3.4%** (11.6 → 12.0
> tok/s, fresh-process ×3, prompt md5 `2c8abce9…`, daemon `518ca732…`). That
> gap is the single most important finding here: **stage-4 decode is close to
> the APU's effective memory ceiling (~130–190 GiB/s of weight traffic on a
> ~238 GiB/s peak), so kernel-arithmetic wins translate to much less wall-clock
> than the arithmetic implies.** #2 (split-K, targeting latency/parallelism,
> not instruction count) and #4 (apply the same win to the other 43% of the
> I-quant GEMVs) are the follow-ups; #3 (speculation) is the only lever that
> breaks the per-weight-pass ceiling structurally.

The honest headline: **hipfire's AR decode of stage-4 (11.7 tok/s on IQ3_S) is
already competitive with llama.cpp's AR decode of a *faster-to-decode* IQ4_XS
model on the same GPU.** pwilkin's 26 t/s is not better GEMV kernels — his fork
ships **zero decode-GEMV changes** (upstream mmvq); the win is DFlash2
speculation + a 4.97-bpw quant + retained-PM4 dispatch.

---

## 1. What the pwilkin stack actually is

`~/git/pwilkin-strix-halo` is the **installer repo** (`github.com/pwilkin/strix-halo`),
not the llama.cpp fork. It pins, reproduces, and launches a complete gfx1151
Qwen3.8-27B stack:

| Component | Pin | What it contributes |
|---|---|---|
| ROCr + HIP | `pwilkin/rocm-systems:ilintar-experiments` @ `7dda3ac6` | custom ROCr/HIP with **retained PM4** (`DEBUG_HIP_GRAPH_PM4=1`, `ENABLE_RETAINED_PM4=1`) |
| llama.cpp | `pwilkin/llama.cpp:strix-halo` @ `d67d5883` | kernel tunings below; **no decode-GEMV changes** |
| Target model | `ilintar/qwen3.8-27b-gguf-strix-halo` `Qwen3.8-27B-IQ4_XS-ALL-IMATRIX-Q8-OUT-MTP.gguf` | IQ4_XS everywhere + Q8_0 lm_head |
| Draft model | `Qwen3.8-27B-DFlash2-IQ4_XS.gguf` | DFlash2 spec draft |
| Launcher | `qwen3.8-strix-halo-server` | `-fa on -fit off -b 2048 -ub 512 -c 65536 --spec-type draft-dflash` width 6, `GGML_HIP_ENABLE_UNIFIED_MEMORY=1` |

Their matched reproduction (Radeon 8060S, 31,497-token prompt, 256 generated):
**256.8 prompt t/s, 26.26 decode t/s** with DFlash2 width 6 (acceptance
0.29 prose / 0.58 reasoning / 0.96 json). AR decode of that IQ4_XS model is not
published, but llama.cpp mmvq AR on a 27B hybrid is ~8–11 t/s class — i.e. the
spec multiplier is the entire margin.

### 1a. What the llama.cpp fork's 47 strix commits actually contain

Decode-relevant (for a 27B GDN+FA hybrid):

- **Nothing for mmvq/GEMV decode.** `git log 718f7b41..HEAD -- mmvq.cu mmq.cu vecdotq.cuh` is
  empty except `e5508525` (a prefill MMB path). Decode GEMVs are upstream
  llama.cpp: activation quantized to **q8_1**, weight grid/sign decode, then
  **`__dp4a` (v_dot4_i32_i32)** — 1 op per 4 elements (`vecdotq.cuh:1211`
  `vec_dot_iq3_s_q8_1`).
- **TOP_K wave32-native** (`fd6e5c31`, `9f38edf2`): hybrid radix-select top-k,
  wave32. Sampling-side; matters only when spec decode makes sampling a
  fraction of wall time.
- **`c5d69235` restore `prop.integrated`**: unified-memory bookkeeping.

Prefill-relevant (the user invited prefill ideas):

- **`e5508525` bf16 WMMA dequant GEMM for large prefill batches** — dequant
  I-quant weights to bf16 **once per graph**, cache the shadow, run WMMA GEMM
  (tall 384×64 tile; fused gate/up + SwiGLU; routed down writing bf16 in
  place). Rationale: prefill is compute-bound, MMQ's integer path leaves WMMA
  idle.
- **`964c6f2f` tiled gated delta-net** — token-tile the recurrence, state in
  registers across the tile, DPP/permlanex16 reduction (bit-identical order).
- **`03733c23` head-256 WMMA FA** — Qwen4exp-only (DV=256); not the 27B
  (FA hd=128).
- **`2e3ee2af` fused gated rms-norm + indexer relu-sum + MoE weighted
  reduction**; **`0540c694` depthwise conv1d + vectorized get_rows**;
  **`be39c4ff` MMQ compaction for large **routed** batches** — MoE/Flash-Next
  flavored.

Not transferable to stage-4: QSA sparse attention (`df8ad5b1`, `d67d5883`),
hyper-connection (`90aba037`/`6ec4a5f0`/`816667e9`), conv1d PLE/GDN — all
Qwen3.8-**Flash-Next**; MMQ compaction — MoE.

---

## 2. hipfire stage-4 decode ground truth (this branch)

From `.codeinsight+research/gsq-rco/prof_blk.log` (32-token decode,
HIPFIRE_GRAPH=0, gfx1151, stage-4 file, dual-row + LDS staging build):

| Kernel | calls/tok | ms/tok | % serialized |
|---|---|---|---|
| gemv_iq3_s_dualrow | 112 | 23.0 | 27.3 |
| gemv_iq4_xs_dualrow | 78 | 19.0 | 22.5 |
| gemv_iq3_xxs_dualrow | 73 | 17.2 | 20.5 |
| gemv_q4k | 30 | 7.1 | 8.4 |
| I-quant residual arms (×4) | ~64 | 7.2 | 8.5 |
| **I-quant family total** | **~357** | **73.5** | **87.2** |
| gemv_hfq4g256 (+residual) | 106 | 2.6 | 3.7 |
| gated_delta_net_q8_fast | 48 | 1.8 | 2.1 |
| rmsnorm/gated_norm/silu/add/conv/rope/… | ~600 | ~6 | ~7 |
| **Serialized kernel time** | **1118 launches/tok** | **84.2** | 100 |
| Wall (nograph) | | **102.9** | 9.7 tok/s |

- Effective BW **117.6 GiB/s** vs the APU's **~256 GB/s** (LPDDR5X-8000 × 32 B
  = 256 GB/s, confirmed from KFD topology) → **~46% of peak**. The I-quant
  GEMVs run at 109–134 GiB/s (q4k 188); there is ~2× headroom to
  bandwidth-bound.
- Non-kernel overhead in the nograph profile: **18.7 ms/token** (1118
  launches/token of host+dispatch). The shipped 11.7 tok/s build uses graph
  capture (verify-forward ON), which removes most of that.
- Root cause per the plan still holds: the I-quant decode is
  instruction/latency-bound, not bandwidth-bound — per-element cost is
  dominated by codebook/sign/scale decode (grid lookup + sign + scale ≈ 10
  integer ops per element vs 1 FMA), confirmed by ISA (48 VGPR, no spills,
  ~33 loads + ~80 int ops per 8 elements in the scalar; the dual-row remap
  cut that to ~2 grid lookups + 1 qh + 1 scales + 1 signs + 2 qs + 8 FMA per 8
  elements per row).

---

## 3. Decode levers, ranked

### 3.1 q8_1 + dp4a I-quant GEMV — SHIPPED (prototype), MEASURED

**What:** port llama.cpp's `vec_dot_*_q8_1` technique to the I-quant decode
GEMVs. Implemented as `gemv_iq3_s_q8dot` (+ residual) in
`kernels/src/gemv_iq3_s_q8dot.hip`, fed by `quantize_q8_1`
(`kernels/src/quantize_q8_1.hip`); rdna-compute methods
(`quantize_q8_1`, `gemv_iq3_s_q8dot`, `gemv_iq3_s_q8dot_residual`), dispatch
wiring in `families/gemv.rs` behind `HIPFIRE_IQ3S_Q8DOT=1` (opt-in, gfx1151),
gated scratch + kernel prewarm at `Gpu::init`.

**Measured kernel win** (gfx1151, real GGUF IQ3_S tensors, 200 iters, µs/call):

| tensor | shape | dualrow | q8dot | speedup |
|---|---|---|---|---|
| attn_output | 6144×5120 | 96.0 | 60.0 | 1.60× |
| attn_qkv | 5120×10240 | 148.3 | 98.6 | 1.51× |
| ffn_gate | 5120×17408 | 255.3 | 174.7 | 1.46× |
| ffn_down | 17408×5120 | 259.4 | 177.7 | 1.46× |

Correctness: vs a CPU q8_1 oracle max_abs ~2e-5 (rel ~1e-6); vs the fp32
dualrow rel ~3e-4 (the q8_1 activation delta). VGPR 69/0 spills vs dualrow 81.

**Measured E2E** (fresh-process ×3, prompt md5 `2c8abce9…`, daemon
`518ca732…`, hipfire `172b68ca…`, model `d148a992…`):
q8dot off 11.6/11.6/11.7 → median **11.6 tok/s**; on 12.0/12.0/12.1 → median
**12.0 tok/s** (**+3.4%**). Decoded text coherent; output diverges from the
fp32 path after ~160 chars (expected q8_1 logit perturbation — same
greedy-parity trade as Item 2), no attractor/empty.

**Why +3.4% and not the ~+12% the arithmetic implies:** IQ3_S is only 31.7% of
serialized decode kernel time, and — the important part — the decode is close
to the APU's *effective* memory ceiling, so a kernel-level instruction win does
not fully convert to wall-clock. This is consistent with the standalone bench:
the dualrow kernel already runs at 130 GiB/s and q8dot at ~140–190 GiB/s
against a ~238 GiB/s (256 GB/s) peak.

**gfx1151 gotchas found while implementing (portable to the other dtypes):**
- `__builtin_amdgcn_sdot4` fails with `needs target feature 'dot1-insts'`.
  `-Xclang -target-feature +dot1-insts` silently DROPS `__global__` kernels
  (empty device object), and `__attribute__((target("dot1-insts")))` on a
  kernel does the same. `__builtin_amdgcn_sudot4(true, a, true, b, …)` works
  with no flag and emits `v_dot4_i32_i8` (both signed; note `true` = *signed*,
  inverted from the LLVM argument name).
- gfx1151 has **no** `v_pk_sub_u8` / `v_sub_u8`. clang's
  `__builtin_elementwise_sub_sat(i8x4)` emulation costs **~14 instructions per
  pack** (measured: the sign block was 57 µs of a 99 µs kernel; a signs-off
  probe ran 2.3× faster). Replaced with `out = (g ^ mask) + bits`,
  `mask = bits*0xFF`: every byte of that sum stays ≤ 255, so a **plain 32-bit
  add is the correct packed-byte op** (~10 ops/pack).
- Anything that allocates (scratch, first kernel launch) must happen OUTSIDE a
  captured decode graph; the AR decode is hipGraph-captured by default
  (`HIPFIRE_AR_GRAPH` on). Solved with an eager scratch + warm-launch prewarm
  at `Gpu::init`. The take/put scratch cycle must preserve the *capacity*, not
  the per-call `k` — otherwise the alternating K (5120↔17408) reallocs mid-graph.

**Remaining work on this lever:** port to IQ4_XS (22.5%) and IQ3_XXS (20.5%)
and Q4K (8.4%); share one q8_1 quantize across the 4 consumers of `x_rot` and
the out_proj input; then re-measure. Given the memory-ceiling effect, the E2E
gain per dtype will be sub-arithmetic.

### 3.2 Split-K for the latency-bound dual-row kernels (P1, untried)

**What:** grid is currently `[ceil(M/2),1,1]` = 2560 blocks for M=5120 with a
serial K loop. Split K into 2 (grid `[M/2, 2]`), each block accumulates half,
then `atomicAdd` f32 to `y[row]` (needs a 20 KB memset of y per GEMV, or fold
the zeroing into the residual epilogue where present). Doubles wave count →
better latency hiding on the 40-CU APU.

**Why:** plan Item 2b explicitly concluded the dual-row kernels are
*latency-bound*. Split-K is the textbook fix and was never tried. Determinism:
fp32 atomicAdd order is non-deterministic — already acceptable (greedy parity
is a documented trade).

**Estimate:** +10–20% on the 4 plain GEMVs → decode +8–15%. Cheaper to try
than 3.1 (one kernel parameter + atomic epilogue), good as a complement to
3.1 (3.1 removes instructions, split-K removes latency stalls).

### 3.3 DFlash2/spec decode (P1 for throughput, plan non-goal)

**What:** pwilkin's entire decode margin is DFlash2 width 6 (26.26 t/s on
IQ4_XS). hipfire has the full DFlash machinery
(`crates/hipfire-arch-qwen35/src/qwen35/speculative.rs`, draft auto-discovery,
`dflash_mode`). The GSQ-RCO plan lists spec decode as a non-goal *"AR must be
correct + tuned first"* — AR is now 11.7 tok/s (success bar ≥10 met). A
paired draft for the stage-4 file would multiply decode on code/reasoning
genres.

**Caveats:** (a) needs a draft artifact matching the stage-4 target (the
canonical DFlash drafts are MQ4-based; compatibility with the native-IQ target
must be checked — the cross-quant matrix in AGENTS.md covers MQ3↔MQ4, not
I-quant targets); (b) the plan's §8 non-goal still stands until the user
revises it; (c) prose acceptance on z-lab drafts is genre-conditional.

### 3.4 Fused I-quant QKVZA / GATE_UP (P2, est. +2–4%)

`in_proj_qkv` + `in_proj_z` (48+48) and `gate` + `up` (64+64) are separate
I-quant GEMVs on the **same x**. A fused I-quant qkvza / gate_up (mirroring
`fused_qkvza_hfq4g256` / `gate_up_via_execute_steps`'s HFQ4 path) shares the x
load and halves launches for 224 GEMVs/token. Launch overhead is small in the
graph build, so this is a modest win — but it is *free* alongside 3.1 (write
the fused kernel once in the new q8_1 style).

### 3.5 TOP_K wave32-native (P2, spec-only)

hipfire's sampler is GPU-side with an argmax fast path for greedy — already
fine for AR. pwilkin's radix-select TOP_K (`top-k.cu`) only matters when spec
decode raises sampling to a measurable fraction of wall time. Defer until 3.3
lands.

### 3.6 Redline / retained-PM4 as a decode transport (P3 for AR)

The user's point — "we have redline in this hipfire" — is accurate: the
direct-KMD machinery exists (`crates/redline`, `redline-dispatch`,
`redline-rocr`, capture/replay fixtures in `hipfire-generate/src/redline.rs`,
`docs/REDLINE.md`). But two facts bound its value here:

- `docs/REDLINE.md` explicitly classifies *"the experimental direct-KMD
  `crates/redline` crate as a serving transport"* as **out of scope** unless
  separately certified. It is a capture/replay research surface today.
- AR decode is already hipGraph-captured (verify-forward ON by default), so
  per-launch dispatch is already amortized; the remaining host overhead in the
  shipped build is small. Redline would buy maybe 1–3% on AR, and real value
  only if decode becomes spec-driven with many small kernels *and* graph
  capture is not doing the job. pwilkin's retained PM4 is the same idea on the
  ROCm side — worth revisiting as a follow-up, not as the next decode lever.

---

## 4. Prefill levers (welcome, separate track)

### 4.1 bf16 WMMA dequant GEMM — the 6× prefill gap (P1)

§1a of the plan already corrected the direction: **stage-4 prefill is 9.5 t/s
vs the all-HFQ4 bridge's 59.7** — the native I-quant prefill GEMMs
(`gemm_iq3_s_batched` etc.) are SIMT and slow. pwilkin's `e5508525` is exactly
the fix: dequant the I-quant weights to **bf16 once per graph** (cached
shadow), then WMMA GEMM. On the 256 GB/s APU the shadow costs 4.6× bytes
(2 B/elem vs 0.43), but prefill at batch 512+ is compute-bound, so WMMA wins —
this is why the bridge (HFQ4, WMMA-capable) prefills 6× faster. The tall
384×64 tile + fused gate/up + SwiGLU epilogue pattern transfers directly.
**Estimate: prefill 9.5 → 30–50 t/s.** This is the highest-value prefill item
and the plan's §8 "prefill is a separate track" note should be updated to name
it.

### 4.2 Tiled GDN for prefill (P2)

pwilkin `964c6f2f` token-tiles the recurrence (state in registers across the
tile, DPP/permlanex16 reduce). hipfire has `gated_delta_net_f32_batch_seq` /
`_chunked`; compare their inner loop against the tiled variant. GDN is only
2.1% of *decode* time but is a real prefill component; the DPP reduce is
already partially in hipfire (`HIPFIRE_GFX1151_GDN_DPP_REDUCE`).

### 4.3 Not applicable

Head-256 WMMA FA (`03733c23`) is Qwen4exp-only (DV=256); the 27B FA layers are
hd=128. Fused gated rms-norm: hipfire already has `gated_norm_f32`. Depthwise
conv1d / QSA / hyper-connection: Flash-Next arch.

---

## 5. Explicitly NOT transferable / rejected

| pwilkin item | Why not |
|---|---|
| QSA sparse attention, hyper-connection, PLE/GDN conv1d | Qwen3.8-Flash-Next arch, not the 27B |
| MMQ compaction for routed batches | MoE-only; 27B is dense |
| f16 KV (`-ctk f16 -ctv f16` in their flash-next launcher) | hipfire q8/asym KV reads *fewer* bytes/head than f16 — better for decode bandwidth |
| bf16 shadow for **decode** | 4.6× bytes → bandwidth-bound at 256 GB/s; decode is instruction-bound, so the shadow would be *slower* (only prefill is compute-bound enough to justify it) |
| Q6_K bf16 shadow (`f5daaa3c`) | MMB prefill only; Q6_K not in stage-4 |
| `-ub 512` / batched decode | single-stream AR; hipfire's `--concurrency` answers a different question |
| Dual-row Q4_K, 2-block IQ3_S staging, LDS grid staging, `#pragma unroll 4` | already measured and rejected in Item 2b (plan §3) |

---

## 6. What I would do next (suggested order)

1. **3.1 q8_1/dp4a I-quant GEMV** — prototype on `gemv_iq3_s_dualrow` first
   (largest share, 27.3%), with the q8_1 quantize fused into the preceding
   norm. Gate behind `HIPFIRE_IQ3S_Q8DOT=1` like the dual-row kill-switches.
   Validate with `test_iq_dualrow_parity` (new q8_1 arm) + fresh-process A/B
   (prompt md5 `2c8abce9…`).
2. **3.2 Split-K** on the same kernel (independent of 3.1; can be tested
   first, it's a smaller diff).
3. If both land: re-run the decode profile; expect the I-quant family to drop
   from 87% toward ~60% of serialized time, then re-evaluate whether the GDN /
   norm / launch tail matters.
4. **Prefill 4.1** on a separate track (bf16 WMMA shadow) — it is the bigger
   number (9.5 → 30–50 t/s) and pwilkin proves the mechanism on gfx1151.
5. Revisit 3.3 (DFlash2) and 3.6 (redline transport) only after the AR kernel
   work is closed out, per the plan's non-goal ordering.

## 7. Key code locations

- I-quant decode kernels: `kernels/src/gemv_iq{3_s,3_xxs,4_xs}_dualrow[_residual].hip`,
  `gemv_q4k[_residual].hip`; scalar siblings.
- dp4a pattern to copy: `kernels/src/fused_qkv_hfq4g256_wave64_dp4a.hip`,
  `kernels/src/gemm_qkvza_tq2g128.hip`.
- Dispatch: `crates/hipfire-dispatch/src/families/gemv.rs` (`dispatch_residual`
  :546, dual-row arms :545–571); rdna-compute methods
  `crates/rdna-compute/src/gemv.rs`; kill-switches `HIPFIRE_{IQ3S,IQ3XXS,IQ4XS}_DUALROW`,
  `HIPFIRE_Q4K_DUALROW`.
- Decode graph: `crates/hipfire-arch-qwen35/src/qwen35/forward.rs`
  (`gate_up_via_execute_steps` :3424, `qkv_via_execute_steps` :3290, RESID_WO
  :5514).
- Prefill GEMMs: `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs`
  (`batch_chunk_delta_net_attn`, `dispatch_batched_gemm_epilogue`); kernels
  `kernels/src/gemm_iq*_batched.hip`.
- Reference (llama.cpp): `ggml/src/ggml-cuda/vecdotq.cuh:1211`
  (`vec_dot_iq3_s_q8_1`), `ggml/src/ggml-cuda/mmb.cu` (bf16 WMMA dequant GEMM),
  `ggml/src/ggml-cuda/gated_delta_net.cu` (tiled GDN).

## 8. Measurement protocol (unchanged from plan §5)

Fresh process ×3, median; byte-identical prompt (md5 `2c8abce9…`); record
daemon/hipfire/model md5s; decode via `hipfire run` with
`HIPFIRE_DAEMON_BIN=…/target/release/daemon` (or `serve_harness.py battery`);
warm DPM + JIT per cell; eyeball decoded text. PPL = wikitext2 200 KB slice,
`flash_prefill_quality` ctx512/chunks16/stride8. JIT CPATH workaround from plan
§4 applies.