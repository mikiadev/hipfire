# Escha W2 dense — parity check vs `escha-runtime-qwen3dense` + ranked perf plan

Static verification round only. **No model was run** (the box hosting this session
shares its memory bandwidth with the engine). Everything below is from source
reading, the safetensors headers, the reference extension's extracted cubins, and
the ISA metadata already in `.hipfire_kernels/gfx1151/`.

Reference: `./escha-runtime-qwen3dense/` (sglang + `escha/_C*.so`), model
`/data/rocmfpx/Qwen3.8-27B-Escha-W2`, device gfx1151 (Radeon 8060S, 40 CU, wave32,
LPDDR5x ~256 GB/s).

---

## Part A — "Are we missing anything?" (parity)

### A1. Bias — **not missing, do not apply it.** ✅

Every coded projection ships a `*.bias` tensor on disk (`in_proj_qkv`, `in_proj_z`,
`out_proj`, `q/k/v/o_proj`, `gate/up/down_proj`). That looked like a gap. It is not:

* `sglang/srt/models/qwen3_5.py:121,134,143,152,161,232,510,520` construct every one
  of those Linears with `bias=False`.
* `qwen3_5.py:868,881,1386,1406` — `if name.endswith(".bias") and name not in
  params_dict: continue` ("Skip loading extra bias for GPTQ models").
* `qwen2_moe.py` `Qwen2MoeMLP` (`:109`, `:118`) — `bias=False`.

So the authoritative runtime **loads and then discards** those tensors. hipfire's
"bias not applied" convention (matching llama.cpp `build_escha_mm`) is correct.
Closing this question permanently.

### A2. RoPE — consistent. ✅

Reference: `get_rope(rotary_dim=self.head_dim, partial_rotary_factor=0.25,
is_neox_style=True)` (`qwen3_5.py:488-495`) → 64 of 256 head dims rotate, NeoX =
half-split. hipfire launches `rope_partial_halfsplit_f32` (confirmed present in the
gfx1151 JIT cache). Same convention.

`mrope_interleaved: true` / `mrope_section [11,11,10]` only decides which of the
(t,h,w) position sources feeds each rotation pair; for text-only `t==h==w`, so it is
numerically inert. hipfire already carries `MropeCtx`. No action.

### A3. Codebook / `escha_config` — correct, but our note about it is wrong. ⚠️

`escha_config` is **int32 `[tile=16, K, V=2, codebook_id, IC, OC]`** (verified across
all 400 projections against the matching `escha_code` shapes). `codebook_id == 1` for
every projection → single codebook, matching hipfire's hardcoded
`x = ((idx*0xCBAC1FED)&0x8FFF8FFF)^0x3B603B60; w = fp16add(lo16,hi16)`.

The handoff's claim that `escha_config` is "6 floats near 1.0" is incorrect — it was
reading int32 as float. Harmless today (hipfire derives K from `code.shape[2]/16`,
which is right), but the note will mislead the next reader. Fix the comment.

### A4. Mixed code rate — **K is 2 *and* 3, and it constrains fusion.** ⚠️

| projection | K | bytes / 16×16 tile |
|---|---|---|
| `in_proj_qkv`, `in_proj_z`, `out_proj`, `q/k/v/o_proj`, `gate_proj` | 2 | 32 (0.25 B/w) |
| `up_proj`, `down_proj` | 3 | 48 (0.375 B/w) |

hipfire handles both (`DENSE_DMAX_W = 24 = 8·3`), so this is a **performance**
constraint, not a correctness one:

> `gate_proj` is K=2 and `up_proj` is K=3. **They cannot be fused into one coded GEMM
> launch** with a single K template parameter. The reference solves this with a
> K-grouped `escham_multi_gemv` (one launch per K group, results sliced back into the
> gate/up halves). Any future gate_up fusion in hipfire must do the same or fuse
> around the K boundary rather than across it.

### A5. Ops hipfire runs separately that the reference runs in one kernel. ⚠️

Reference op surface (`torch.ops.escha.*`, recovered from `_C*.so` via cuobjdump —
54 cubins, 9 modules):

* `escham_decode_gemv` (M ≤ 32) — **one** kernel: `had_in(x·rin·s_in)` → decode →
  MMA-GEMV → `had_out` → `rout·s_out`.
* `escham_multi_gemv` — merged multi-shard GEMV, per-shard `rin/s_in` stacks,
  1.87× at M=16, ~1.39× whole-step.
* `escham_code_gemm` — large-M prefill, `BM ∈ {32,64,128} × BN ∈ {32,64}`, shared mem
  up 37888 B, `acc_mode` fp32/fp16.
* `escha_gemv` — int8 W8A16, used for `lm_head` **as stored**.

hipfire's equivalent (`escha_dense_forward.rs::escha_dense_decode_proj`) is
**3 launches + 2 global round-trips per projection** —
`escha_dense_rotate_in` → `escha_dense_decode_gemv` (writes `partial[n_slices][OC]`
fp32) → `escha_dense_finalize` — × ~400 projections/layer-stack = **~1200 launches
and ~380 MB of split-K partial traffic per decoded token**.

### A6. Things already verified matching (no action)

Fold convention `in_scale=rin·s_in`, `out_scale=rout·s_out`, `rin` already carrying
Wscale (never re-apply); trellis index permutation `pi(r)`, funnel-shift word pairing,
codebook; GDN `g = -exp(A_log)·softplus(a+dt_bias)` with `threshold=20` stabilization
(not `F.softplus`), `beta = sigmoid(b)`, in-kernel L2-norm of q/k; `RMSNormGated` =
`RMSNorm(x)·silu(z)` with a **plain** weight (no +1) while `q/k_norm` and input norms
are GemmaRMSNorm with `(1+w)` (fixed in `18871f381`); conv state bf16 / ssm state fp32.

**Verdict: no correctness gap found against the reference.** All deltas are
performance.

---

## Part B — Prefill: root cause found

> **Status as of 2026-09-07:** B1 was confirmed and fixed (batched prefill is now
> ON and measures 4.8×), B2 and B3 are done, **B4 is the remaining prefill work**
> and B6 is the ceiling-raiser. See "Suggested execution order" for current numbers.

### B1. **Batched prefill is switched off for this model.** ← this is the whole story

`qwen35_layer_batch_admissible()` (`prefill.rs:1946-1951`) returns

```rust
LayerWeights::DeltaNetEscha(_) | LayerWeights::FullAttnEscha(_) => Err(
    HipError::new(0, "Escha code-quant dense layers: batched prefill deferred (attractor risk)"),
),
```

→ `all_layers_ok = false` → `prefill_batch_pbs_eligible()` = false (`prefill.rs:2496`)
→ `if !eligible` per-token fallback loop (`prefill.rs:1176+`), described in-source as
"byte-identical to decode".

So `forward_batch_chunk_impl`'s Escha arms (`prefill.rs:7414+`) — including the
`DeltaNetEscha` batched body `deltanet_escha_layer_prefill` — are **dead code today**.

Arithmetic check: 58 tokens × 285 ms/token (measured decode step) = **16.5 s** vs
measured **16.755 s**. That is a 1.5% match. `pp == tg` is not a coincidence: prefill
*is* 58 sequential decode steps.

Timeline: `70d860bb9` closed this gate because the batched path produced an attractor;
`9daf925bf` then fixed the attractor's root cause (the `s_u` shared-memory overflow).
**The gate was never re-opened.** The handoff's "batched-prefill path re-enabled for
DeltaNetEscha layers" describes the arm inside `forward_batch_chunk_impl`, not the
eligibility gate in front of it.

**Action P0: re-open the gate and validate.** One-line change, plus the coherence
route. Expected ≥2.7× from their own profiler numbers, before any kernel work.

### B2. After B1, the next prefill limiter is the `acc[R]` spill

`escha_dense_matmul_prefill_kernel` declares `float acc[64]` and loops
`for (int m = 0; m < R; ++m)` with **`R` a runtime argument**. ISA metadata
(`.hipfire_kernels/gfx1151/*.radiowave.json`):

```
escha_dense_matmul_prefill_kernel  VGPR=19  vSpill=0  private_segment_fixed_size=272
```

`VGPR=19` + `private_segment_fixed_size=272` = the accumulators live in **scratch
(global/L1), not registers**. Every `acc[m] +=` is a scratch load + store. At 272 B ×
32 lanes ≈ 8.7 KB of private state per wavefront against a 32 KB L1, the arrays thrash.

This is why the batched path only ever measured 1.39–2.68× instead of ~8×, and is the
reason it looked "not worth re-enabling". **Fix: make `R` a compile-time constant**
(template or `-DR` specializations for 1, 2, 4, 8, 16; drop 32/64 or pair them with a
row-block loop so `acc[]` stays register-resident). Verify with the
`gfx-kernel-metadata` skill that `private_segment_fixed_size` returns to 0.

### B3. `n_slices ≥ R` inflates the split-K scratch

`escha_dense_n_slices_prefill` (`rdna-compute/src/escha_dense.rs`) does
`n_slices = n_slices.max(r as usize)` before bumping to a divisor of `nit`. The partial
buffer is `n_slices × n_rows × OC` fp32. Measured for a 58-row chunk:

| projection | n_slices (decode) | n_slices (prefill) | partial |
|---|---|---|---|
| `in_proj_qkv` | 12 | 64 | 152 MB |
| `in_proj_z` | 21 | 64 | 91 MB |
| `out_proj` | 25 | 64 | 76 MB |
| `q_proj` | 10 | 64 | 183 MB |
| `gate` / `up` | 7 | 64 | 259 MB each |
| `down` | 25 | 64 | 76 MB |

≈ **125 GB of partial write+read traffic per 58-token chunk** (~0.5 s of the DRAM
budget), and ~4.4× that at the 256-row default chunk (`prefill_max_batch_for_arch` →
`PREFILL_MAX_BATCH`, since escha-dense fails `dense_layers_are_all_mq4v2`) — 550 MB+
for `gate`/`up` alone. Drop the `.max(r)`, cut split-K to ~4–8, and hoist the prefill
partial into `PrefillBatchScratch` (`70d860bb9` hoisted the *decode* scratch only;
prefill still alloc/frees per projection per chunk).

### B4. Full-attention layers still run per-token inside the batched chunk

`prefill.rs:7437+`, the `FullAttnEscha` arm, is a gather/scatter loop: per token
`memcpy_dtod_at` row in → `memcpy_htod_auto(pos)` → `forward_scratch_layers` → memcpy
row out. 16 FA layers × 58 tokens = **928 per-token forwards + 928 H2D copies**
(each a potential sync) per chunk. Batch these like the `DeltaNetEscha` arm —
`q/k/v/o_proj` are ordinary coded projections and the FA kernel already has a batched
form used by the MQ paths.

### B5. GDN recurrence is sequential; a chunked kernel exists but is off

`deltanet_escha_layer_prefill` → `gated_delta_net_f32_batch_seq` (serial token scan).
`HIPFIRE_GDN_CHUNKED` (`rdna-compute/src/norm.rs:61`) selects the parallel chunked
kernel, "numerically EQUAL to batch_seq (oracle gdn_chunked_f32, 1.3e-15)", default
off. Reference uses `CHUNK_SIZE = 64`. Worth an A/B once B1 lands; GDN is not the
dominant cost, so this is a second-tier lever.

### B6. No tensor-core path for coded prefill

hipfire's escha GEMM is scalar-FMA. Reference `escham_code_gemm` at M=2048:
1793 ms (reconstruct-then-GEMM) → 669 ms fused fp32-acc (146 TFLOPS) → 453 ms fused
fp16-acc (215 TFLOPS), decoding the code **directly into MMA B-fragments**. hipfire
already has WMMA `gemm_*` families on gfx1151 for MQ/HFQ — the escha decoder needs the
same treatment. This is the ceiling-raiser for pp (hipfire's WMMA quants reach
~10³ tok/s prefill on this class of card; escha-dense is at 3.5).

**Prefill order: B1 → B2 → B3/B4 → B6 (WMMA) → B5.**

---

## Part C — Decode

> **Status as of 2026-09-07:** C1 and C2 are **refuted as tg levers** by
> measurement — decode is not DRAM-bound (39 GB/s used of ~104 GB/s sustained)
> and the two extra launches are 2% of a projection. C3 is partially done (the
> `% NW` division and the double LDS are gone: tg 3.4 → 5.0). C4 is demoted.
> C5 (MTP) is now the top decode item, and blocked on a container-format gap.
> See the retraction section below for the numbers.

Traffic model per token (verified against the header):

| bytes | source |
|---|---|
| 7.52 GB | 24.35 G coded weights (12.95 G @ K=2 = 0.25 B, 11.40 G @ K=3 = 0.375 B) |
| 5.08 GB | `lm_head` **dequantized to F32** (`escha_load.rs:710-729`) |
| 0.38 GB | split-K partials (write+read) |
| **≈ 13 GB** | measured 3.5 tok/s → **~45 GB/s effective, 18% of peak** |

At a realistic 55–65% of 256 GB/s, the weight-bound step is 55–70 ms →
**14–18 tok/s**. The reference is at 60–72% of roofline on 4090/5090, so that target
is not aspirational.

### C1. `lm_head` is read at 4× its stored size — 39% of decode traffic

`escha_load.rs:710-729` dequantizes `lm_head.weight_int8` + `weight_scale` to **F32**
(`upload_f32`, `DType::F32`, 248320 × 5120). The reference ships the head **as int8**
to a dedicated `escha_gemv` W8A16 kernel and measured +7.4% on a 5090 — where the head
is a smaller fraction of the step. On this bandwidth-starved APU it is 3.8 of ~13 GB.

**Cheapest large win in the whole plan.** Keep int8, add a dp4a/W8A16 GEMV
(`rdna-compute` already has MQ8 dp4a machinery at `gemv.rs:6124,6193` to model it on).
Expected **+35–40% tok/s** on its own. Note `HIPFIRE_LM_HEAD_F16` exists for the
hfq qt=1 path but the escha loader hardcodes F32; F16 alone would already recover ~2.5 GB.

Same treatment for `embed_tokens` (`escha_load.rs:657-688`, also int8→F32) is
bandwidth-neutral (row lookup) but frees 5.08 GB of VRAM, which buys context length.

### C2. Fuse the 3-kernel projection into one

`rotate_in → decode_gemv → finalize` per projection: 1200 launches/token plus a
`u[IC]` and a `partial[n_slices][OC]` global round-trip each. The reference's
`escham_decode_gemv` does Hadamard-in, decode, accumulate, Hadamard-out and the
`rout·s_out` epilogue in one launch. Merging at minimum `rotate_in` into the GEMV
(doa the 128-block Hadamard in registers/shared, once per block) removes 400 launches
and the `u` buffer entirely. With `n_slices=1` for the common case, `finalize` also
disappears.

Also merge shards: `q/k/v_proj` are three separate on-disk coded tensors but one
logical QKV — concatenating along `OC/16` is a clean concat of `escha_code`. A
reference-style `multi_gemv` (per-shard `rin/s_in`, one launch) measured 1.87× at
M=16 and ~1.39× whole-step. Respect the K-grouping from A4.

### C3. The decode GEMV's per-weight instruction count

Per weight it is ~10–17 VALU ops for one FMA (funnel shift + codebook mul/xor/add +
unpack). Two concrete reductions:

* **Two `LDS.32` per weight where one `LDS.64` suffices.** The header of
  `escha_dense_kernels.hip` describes llama.cpp's **overlapping `uint2` payload pairs**
  (`pay[w] = {word w+1, word w}`) so a single 64-bit shared load yields both words, but
  the body deliberately keeps "a plain uint32 array … two reads per weight". Switching
  to the overlapping-pair staging halves the LDS instruction count with the identical
  funnel-shift pairing already pinned by the attractor fix.
* Hoist the codebook constants and the `pi(r)`/`s`/`w0` index math out of the inner
  loop where `K`/`NW` allow (compile-time `NW` turns `% NW` into a compare+subtract).

### C4. hipGraph for the decode step

escha-dense is excluded from AR graph capture. `70d860bb9` hoisted the decode scratch
specifically to make it capture-safe — that precondition is now met. Today ~1200
launches × 2–3 µs ≈ 3–4 ms of 285 ms (~1.2%), so it is *not* the fix; it becomes
decisive once C2/C3 cut the kernels to a few ms each. Sequence it **after** C2.
Per `CLAUDE.md`, anything touching graphs owes `scripts/redline_daemon_harness.py`.

### C5. MTP draft is already on disk — ~1.8× multiplier
> **Update:** the draft is a safetensors *directory*; the loader wants a
> `<trunk>.mtp` HFQM file or a bundled trailer, so `--spec mtp` silently runs AR
> (measured 5.0 tok/s, same as `off`). Needs an HF-safetensors → HFQM converter.

`/data/rocmfpx/Qwen3.8-27B-Escha-W2/mtp/model.safetensors` (849 MB) + `mtp/config.json`.
The reference's `serve.sh` gets **1.77–1.92× decode** from it. hipfire has `--spec mtp`
plumbing; the open question is whether the MTP layer's projections can be loaded through
`escha_load.rs`. Big multiplier, but it stacks *on top of* C1–C3 — do it second.

**Decode order: C1 → C2 → C3 → C4 (graphs) → C5 (MTP).**

---

## Suggested execution order

| # | change | risk | expected | status |
|---|---|---|---|---|
| 0 | Batched-path oracles at real activations | low | prerequisite for 1/5/9 | ✅ `check_escha_dense_batched` + `check_gdn_batched`. They refuted my own "batched is broken" claim — see the retraction below. |
| 1 | Re-open `qwen35_layer_batch_admissible` for `DeltaNetEscha`/`FullAttnEscha` | low | pp ≥ 2.7× | ✅ **DEFAULT ON**. Measured **4.8×** (16.8 s → 3.47 s TTFT). `HIPFIRE_ESCHA_DENSE_BATCHED=0` reverts. |
| 1b | FA-escha fallback inside the batched chunk replayed the **whole model** per token | — | prerequisite for 1 | ✅ fixed — was latent, unreachable only because the gate was shut. |
| 2 | ~~`lm_head` as int8 + W8A16 GEMV~~ | tg +35–40% | ❌ **PREMISE WRONG.** Decode is not DRAM-bound (39 GB/s used vs ~104 GB/s sustained), so fewer bytes cannot buy tg. Still ~3.8 GB of VRAM for context length. |
| 3 | Compile-time `R` (+ `K`) in `matmul_prefill` | pp +2–4× on top of 1 | ✅ `private_segment_fixed_size` **272 → 0**, VGPR 19 → 48 @ R=16. But K compile-time did **not** fold the modulo (see 7b) — the range proof was the blocker, not the constant. |
| 4 | Drop `n_slices ≥ R`; partial scratch bound | pp +10–20%, −GB VRAM | ✅ done; the floor was accidentally capping smem → crash at 600 rows, replaced by an explicit bound (`b8dcefcb9`), test shown non-vacuous (8 violating shapes → 0). **Prefill scratch is still alloc'd per projection per chunk** — hoisting into `PrefillBatchScratch` is open. |
| 5 | Batch the 16 `FullAttnEscha` layers inside the chunk loop | med | pp ~2× on top | **NOW THE TOP PREFILL ITEM** — FA is still per-token inside the batched chunk, and the chunk's GDN half is already batched, so FA ≈ 16/64 × 58 tokens of full-speed decode is the bulk of the remaining 3.47 s |
| 6 | ~~Fuse `rotate_in`+GEMV+`finalize`~~ | tg +50–100% | ❌ **PREMISE WRONG (measured).** rotate = 9.4 µs, finalize = 3.8 µs: **13 µs of a 570 µs projection (2%)**. Fusing them cannot matter. Launch count still matters via hipGraph (8), not here. |
| 7 | `LDS.64` overlapping-pair payload | tg +20–40% | ✅ 0.63 → 0.57 ms; **tg 4.6 → 5.0**, bit-identical vs the host reference. Overlapping `uint2` pairs staged once per tile. |
| 7b | *found here:* the non-folding `% NW` | — | ✅ **biggest decode win.** `(w0+NW-1)%NW` → `w0 ? w0-1 : NW-1`: 36 `s_mul_hi_u32` + 39 `s_addc_u32` were a *division* in the inner loop. 1.10 → 0.63 ms, **tg 3.4 → 4.6**. |
| 8 | hipGraph AR capture for escha-dense | tg +10–25% | todo, but demoted: per-launch cost is now small against gemv time. |
| 9 | WMMA/MFMA coded prefill GEMM | pp → O(100–500) | todo — still the real ceiling-raiser. gemv is at 156 G weights/s vs ~515 G that measured bandwidth implies, so ~3× is on the table. |
| 10 | MTP speculation from `mtp/` | tg ×1.8 | **DOES NOT ENGAGE.** `--spec mtp` measured 5.0 tok/s, identical to `off`. The on-disk draft is silently falling back — investigate before valuing it; it is the only tg lever that doesn't require beating the gemv. |
| — | *tested and reverted:* split accumulator ×4 | — | ❌ 0.63 → 0.83 ms, **worse**. The FFMA dependency chain is not the limiter; the flat `n_slices` sweep (952→43520 blocks all ≈0.65 ms) says occupancy isn't either. |

**Cumulative measured, gfx1151, 58-token prompt, warm:**

| | start of round | now |
|---|---|---|
| decode | 3.4 tok/s | **5.0 tok/s** (+47%) |
| prefill | 3.5 tok/s (16.8 s TTFT) | **16.7 tok/s (3.47 s)** (4.8×) |

**Revised order:** 5 → 10 → 9 → 8. Items 2 and 6 are dropped for the reasons
above; item 2 (`lm_head` int8) still buys ~3.8 GB of VRAM, just not speed.

### GPU validation (gfx1151, measured — including a retraction of my own conclusion)

Everything here is measured, not inferred. Two earlier drafts of this section
were wrong and are corrected below; the corrections matter more than the
numbers.

Final numbers, warm, 3-run, 58-token prompt:

| | per-token | batched prefill | after both decode fixes |
|---|---|---|---|
| prefill | 17 080–17 537 ms | 4 767–4 821 ms | **3 465–3 536 ms** (4.8×) |
| decode | 3.3–3.4 tok/s | 3.3–3.4 tok/s | **5.0 tok/s** |
| pp vs tg | identical | separated | separated |

#### Retraction: "the batched path is wrong" was a bad inference

I first concluded batched prefill was numerically broken because at `--temp 0`
it diverges from per-token on the **first generated token**, and I flipped the
eligibility gate back off on that basis. The oracles say that reasoning was
backwards — per-token is not ground truth, it is just a different summation
order:

| probe | result |
|---|---|
| `check_escha_dense_batched` (batched projection vs **host** reference) | ≤ 4.07e-4 rel, **identical at every R 1→32** |
| `check_gdn_batched` (batched conv carry vs N per-token calls) | **bit-exact (0.0)** |
| `check_gdn_batched` (batched GDN S-matrix carry) | 1–2e-7 |
| **control** — two *batched* runs differing only in chunk size (R=64 vs R=2) | disagree with each other as much as either disagrees with per-token (L0 5.5e-3 vs 9.8e-3; L63 2.1e-1 vs 2.5e-1) |

The control row is decisive: a difference that appears between two batched runs
of identical kernels is reshuffling, not error. At 2e-3 per layer through 64
residual layers, 2e-3 at layer output is exactly the expected size. This repo
documents elsewhere that ~1 ULP flips an argmax over 2k greedy tokens, so greedy
divergence is the most sensitive metric available for the least information.
Gate is **DEFAULT ON**; `HIPFIRE_ESCHA_DENSE_BATCHED=0` reverts in one env var.

Still worth knowing, from the same investigation (all still true):
* **Not the R/K specialization.** A control build reverting to the *original*
  `matmul_prefill` kernel and the original slice heuristic behaves identically.
* **Not a chunk-carry bug.** `HIPFIRE_PREFILL_MAX_BATCH` ∈ {2,4,8,16,32,default}
  all show the same magnitude of difference.
* `9daf925bf`'s shared-memory fix landed while the gate was shut, so it was
  never exercised end-to-end — which is why nobody knew the path's status.

**Measurement trap, learned the hard way:** the per-token `dump_hidden_localize`
call site records `s.x` under `layer_idx - 1`, the batched site under
`layer_idx`. Pairing dumps by recorded layer index therefore compares batched
layer L with per-token layer L−1, which at sequence row 0 degenerates to
embedding-vs-embedding and looks *identical*. That produced a confidently wrong
"exact at row 0, ramps with position" conclusion; only prompt-internal rows give
an aligned comparison.

**The missing oracle was the real gap.** `check_escha_dense` validated the decode
gemv only; the batched path had zero coverage, which is how it could be assumed
broken for a day. Both new examples exit nonzero on FAIL and should be wired into
a gate — they are also the prerequisite for verifying any WMMA prefill GEMM
(item 9), where the same silent-corruption class is on tap.

#### Two premises I wrote into this plan and then measured against

* ~~"`lm_head` int8 → **+35–40% tg**"~~ — assumed decode is DRAM-bound. It is
  not: the gemv reads 22.3 MB in 0.566 ms = **39 GB/s** against **~104 GB/s**
  measured sustained (`rdna-compute/examples/mem_bw.rs`: 208 GB/s D2D at 2 GiB
  = 81% of theoretical; 718 GB/s at 8 MiB is L2, so the 2 GiB figure is real
  DRAM). Fewer bytes cannot buy tg here. Still saves ~3.8 GB of VRAM.
* ~~"fuse `rotate_in`+gemv+`finalize` → **+50–100% tg**"~~ — per-stage
  measurement: rotate **9.4 µs**, finalize **3.8 µs**, gemv **566 µs**. Those two
  launches are **2%** of a projection.
* **Tested and reverted:** splitting the accumulator 4 ways to shorten the FFMA
  dependency chain made the kernel **30% slower** (0.63 → 0.83 ms). And the
  `n_slices` sweep is **flat from 952 to 43 520 blocks** (0.63–0.70 ms), ruling
  out occupancy. So the remaining ~3× gap to bandwidth-limited is neither chain
  latency nor block parallelism, and **I do not know what it is** — which is
  exactly why item 9 (WMMA) is next rather than another scalar tweak.

What *did* work was found by reading the disassembly instead of reasoning: the
`% NW` in the per-weight loop emitted a full integer division (36
`s_mul_hi_u32` + 39 `s_addc_u32`) because the compiler cannot prove `w0 < NW`;
one select took the gemv 1.10 → 0.63 ms (tg 3.4 → 4.6), and a single `LDS.64` per
weight (overlapping `uint2` pairs, as this file's kernel header always described
but never used) took it to 0.566 ms (tg 5.0). Both bit-identical vs the host
reference.

#### Caveats to carry forward

* Under greedy at 1500 tokens **both** arms degenerate (per-token 6-gram repeat
  count 33, batched 213). Greedy cannot adjudicate quality either way — but note
  batched is not obviously *better*. Ship acceptance is the owner's 10+ prompt
  battery at serving temperature on both arms, not my three prompts.
* `open think span at end of generation`: this model always reasons and
  `reasoning.mode off` only zeroes the budget and hides the trace, so a run
  truncated mid-`think` is a config artifact. `-n ≥ 400` with reasoning on, or
  set `off`. Several of my early "failures" were this.
* A whole round was spent on unchanged numbers because `cargo check` does not
  build binaries — `target/release` predated the commits. Rebuild before
  measuring, and run inside `nix develop .#therock` so the JIT recompile of a
  changed `.hip` has a working hipcc.

R sweep for the next attempt: register cost of the new instantiations is R=8 →
40 VGPR, R=16 → 48, R=32 → 82, all at `private_segment_fixed_size = 0`. R=16 is
the likely sweet spot.

---

## Updated targets for this hardware

Decode is **not** DRAM-bound (39 GB/s used vs ~104 GB/s sustained), so the tg
ceiling is per-kernel efficiency, not bytes:

* **tg 5.0 → 8–12 tok/s** by getting the gemv from 157 G weights/s toward the
  ~515 G/s that measured bandwidth implies (WMMA/MMA decode, as the reference's
  `escham_decode_gemv` does — it decodes straight into MMA B-fragments).
* **MTP on top of that**, ~1.77–1.92× per the reference, once the draft is
  convertible (see handoff Next steps 2). It multiplies tg without requiring any
  kernel to get faster, so it is the highest-leverage remaining tg item.
* **pp 16.7 → 60+ tok/s** by batching the 16 FA layers (now the dominant term),
  then a WMMA coded prefill GEMM for the ceiling. hipfire's WMMA quants already
  prefill 27B near 10^3 tok/s on this card class, so pp is engine-limited, not
  hardware-limited.

Housekeeping from the parity round is **done**: the `escha_config` field layout and
the dead-`.bias` decision are recorded in `ESCHA-W2-CONTINUE.md` § "Corrected field
notes", and the dense loader's own comment (`escha_load.rs:378,423`) already had
`escha_config` right.
