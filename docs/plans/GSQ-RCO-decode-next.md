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
| 1 | **q8_1 + dp4a I-quant GEMV** (llama.cpp `vec_dot` technique) | kernel rewrite | +25–40% (11.7 → ~14–16 tok/s) | high |
| 2 | **Split-K for the latency-bound dual-row kernels** | kernel | +8–15% | medium |
| 3 | **DFlash2/spec decode** (pwilkin headline: 26 t/s) | engine | ×1.5–2.5 on code/reasoning | high (plan non-goal) |
| 4 | Fused I-quant QKVZA / GATE_UP | kernel | +2–4% | medium |
| 5 | TOP_K wave32-native (pwilkin) | sampling | spec-only | low |
| 6 | Redline / retained-PM4 as decode transport | infra | ~1–3% for AR; real for spec | high |
| — | Prefill: **bf16 WMMA dequant GEMM** (pwilkin) | prefill | prefill 9.5 → 30–50 t/s | high |

The honest headline: **hipfire's AR decode of stage-4 (11.7 tok/s on IQ3_S) is
already competitive with llama.cpp's AR decode of a *faster-to-decode* IQ4_XS
model on the same GPU.** pwilkin's 26 t/s is not better GEMV kernels — his fork
ships **zero decode-GEMV changes** (upstream mmvq); the win is DFlash2
speculation + a 4.97-bpw quant + retained-PM4 dispatch. The transferable
kernel-level "new mechanism" the plan said was missing is the **q8_1/dp4a
decode**, which hipfire already uses for its HFQ4 GEMMs but not for I-quant
decode GEMVs.

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

### 3.1 q8_1 + dp4a I-quant GEMV — the "new mechanism" (P1, est. +25–40%)

**What:** port llama.cpp's `vec_dot_*_q8_1` technique to the four I-quant
decode families (`gemv_iq3_s/iq3_xxs/iq4_xs/q4k` + residual arms):

1. Quantize the activation `x` to **q8_1** (int8 + per-32 scale) — one tiny
   kernel per layer, or better: fuse it into the preceding rmsnorm/gated-norm
   kernel that already produces `x_rot`/`dn_normed` (zero extra launch).
2. In the GEMV: grid-lookup the weight bytes as today, but apply the sign with
   packed `__vcmpne4`/`__vsub4` and accumulate with **`__dp4a`** — 1
   instruction per 4 elements instead of 4 FMAs, and the per-element scalar
   weight×x multiply disappears.
3. Rescale once per 32-block by `d × ds`.

**Why it is the right next mechanism:** the plan's own ISA analysis says the
I-quant decode is instruction-bound; the dual-row kernel still spends ~5–6 ops
per element on `(int8)grid_byte → w = db*gb*sign → FMA(x)` where llama.cpp
spends 2 dp4a + 2 packed-sign ops per 8 elements. That is a genuine **~3×
instruction reduction on the dominant cost**, and it is the *same class of
mechanism* hipfire already ships for HFQ4 GEMMs (`fused_qkv_hfq4g256_wave64_dp4a`,
`gemm_qkvza_tq2g128`, 29 dp4a kernel files) — the pattern exists in-tree.

**Numerics note:** q8_1 activations change decode numerics slightly (x is
quantized to int8+scale). Greedy parity is already *not* preserved (documented
Item-2 trade); PPL is prefill-only and unaffected. llama.cpp ships this for
every quant on every arch; it is not a correctness cliff. The parity harness
(`test_iq_dualrow_parity`) needs a q8_1 oracle arm.

**Estimate:** I-quant GEMVs 109–134 GiB/s → ~180–220 GiB/s (q4k already sits
at 188 with the same technique upstream). The 4 plain GEMVs (66 ms/tok) drop
to ~40–50 ms/tok → serialized kernel 84 → ~60–70 ms/tok → decode 9.7 → ~12–13.5
tok/s (nograph), and proportionally on the shipped graph build: **11.7 → ~14–16
tok/s**. This is the single biggest kernel-level lever and the only one that
attacks the *instruction count* rather than latency hiding.

### 3.2 Split-K for the latency-bound dual-row kernels (P1, est. +8–15%)

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