# GSQ-RCO IQ3_S plan: re-quant bridge (shipped) → native IQ3_S support (proposed)

**Status:** execution trace + proposal (2026-09-09 / 2026-09-10). Branch
`exp/gsq-rco-iq3s`. This file is the plan record, not a product claim.
Quality numbers below are `measured` on the stated fixture; admission
state is fail-closed (`docs/admissions.yml` empty).

**Source under test:**
[ISTA-DASLab `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/raw/main/README.md)
(11.77 GB, mixed-precision: Q4_K 10% / IQ3_S 32% / IQ3_XXS 20% / IQ4_XS 21% /
IQ2_S 9% / BF16+F32 smalls; method:
[GSQ](https://arxiv.org/abs/2604.18556) +
[RCO](https://arxiv.org/abs/2605.00649)).
Local copy: `/data/rocmfpx/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`.

**Question answered here:** can we *serve* it, and what does it cost.
**Question proposed:** can we serve it *without re-quanting* (native
IQ3_S decode), keeping its 11.8 GB size and its win over our MQ4.

---

## 1. What shipped (re-quant bridge, coherent)

`/tmp/gsqrco-iq3s.hfq` (md5 `3c60064fb2d33f7153f69543511d7298`, 14.97 GB,
arch 5, 851 tensors, `--format mq4v1`) serves coherently on gfx1151.
Recipe: GGUF → dequant (`tensor_to_f32`, all 11 dtypes C-verified +
NRMSE ~0.10 vs host kernel-rule decode) → MQ4 re-quant.

Three conversion bugs found and fixed (commits below). All three were
*convention mismatches between the GGUF release and the safetensors-born
loader*, not decoder bugs:

| # | Bug | Evidence | Fix | Commit |
|---|---|---|---|---|
| 1 | GGUF norms store TRUE γ (~1.0); loader adds `QWEN35_NORM_BIAS=1.0` (expects residuals) → every norm loaded ~2.0 → token soup | good-3.6 norms store ~0.0; ours stored ~1.0 | store TRUE−1 for the five `norm()`-read names | `9edc06ac9` |
| 2 | GGUF DeltaNet V-heads sequential; engine consumes interleaved 3V/K (`escha-head j == gguf-head PERM[j]`, PERM=`[0,16,32,1,17,33,…]`) | good-3.6 dt_bias == Escha-3.8 dt_bias 48/48; GGUF matches 8/48; conv V-groups map identically | `maybe_interleave_deltanet_v_heads()` on dequantized f32 (A_log/dt/in_proj_a/b/z, conv/qkv-V segs, out_proj-V cols) | `9edc06ac9` |
| 3 | GGUF `ssm_a` is raw decay A (−0.04); engine wants `ln(−A)` (kernel does `alpha *= -exp(a_log)`); without it decay≈1.0 | Escha-A_log[e] == ln(−ssm_a[PERM[e]]), 2304/2304 elems, 48 layers | `ln(−A)` in the same helper | `365ba3337` (PPL 35.6 → 10.3) |

Falsified along the way (do not re-litigate): GGUF dim-order transpose
(flat pipeline = no-op), FWHT sign mismatch, Q4_K/Q2_K/IQ decoder ports
(C-verified), tokenizer divergence (ids identical to good-3.6),
chat template, KV-mode, RoPE flag, MQ4V2-vs-MQ4V1 (both souped pre-fix).

Serve-harness battery (`scripts/serve_harness.py --mode battery
--thinking off`, daemon `27b78167`): 5/5 `finish=stop`, 0 runaway /
0 empty / 0 attractor. Eyeballed: Rayleigh essay, Paris, coherent
code fence. Per Astrea rules this is serve-semantics evidence only,
not a KLD/PPL claim and not an admission.

## 2. Size accounting (why ours is bigger than the release)

| File | Bytes | Whole-file bpw |
|---|---|---|
| Release GGUF | 11.77 GB | ~3.5 |
| Ours (`/tmp/gsqrco-iq3s.hfq`) | 14.97 GB | ~4.4 |
| Local `qwen3.8-27b.mq4` | 15.66 GB | ~4.6 |
| Local `qwen3.8-27b.mq3` | 12.62 GB | ~3.7 |
| Local `qwen3.8-27b.mq6` (new) | 21.75 GB | ~6.5 |

The growth is the 497 dense 2D projections (~25.6B params) going from
~3.4 bpw mixed I-quant to flat 4.25 bpw MQ4 (+2.7 GB), plus embed→Q8
(+0.5 GB). Ours < local mq4 because we carry 305 F16s + 1 Q8 table
where mq4 carries 801 + 50 (AWQ sidecars etc.). Nothing is wasted;
it is the format math. mq3 vs mq4 on disk is the clean one-variable
pair (identical census, 496× qt49 vs qt44).

## 3. Quality ladder (measured 2026-09-09/10, same slice+harness)

Fixture: first 200 KB of
`benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt`
(md5 `83b0205a…`), `flash_prefill_quality` ctx512/chunks16/stride8
(512 scored), except the GGUF row (llama-perplexity, own tokenizer —
cross-engine caveat: tokenizers differ by 238 ids on the slice).

| Model | PPL | Size | Source |
|---|---|---|---|
| GGUF release (llama.cpp) | **7.17** ± 0.12 | 11.8 GB | measured |
| mq6 (`qwen3.8-27b.mq6`) | **9.16** | 21.8 GB | measured (new anchor) |
| mq4 trunk | 9.23 | 15.7 GB | measured |
| ours v5 | **10.32** | 15.0 GB | measured |
| mq3 | 10.54 | 12.6 GB | measured |

Registry teacher-KLD (same teacher/protocol, comparable): mq4 base
WT2 0.039 / mq3 base 0.154 / mq6 base 0.0028. Both ladders agree:
ours ≈ mq3-class, ~1.1 PPL behind mq4, ~1.2 behind mq6. The gap to
the release is the expected double-quant cost (GSQ-RCO 3.5 bpw →
MQ4 4.25 bpw); the release's own table claims IQ3_S ≈ BF16-lossless,
and we are one more lossy hop down.

## 4. Native IQ3_S: why it needs kernels, not a loader tweak

Verified by inventory, not by reasoning from docs:

- `kernels/src/` has exactly one kernel for this file's dtypes:
  `gemv_q4k.hip` (10% of params). No `gemv_iq3s/iq4_xs/q2k`, no I-quant
  GEMM, no fused prefill path. 87% of params have no compute behind them.
- `qwen35/load.rs::load_weight_tensor_raw` has no Q4K arm and no
  I-quant arms (unknown dtypes fall to `dequant_weight_raw`, which
  only speaks F16/F32/BF16 + raw codecs).
- Prefill is the wall even for Q4_K alone: the fused `qkvza` matcher
  falls through to an HFQ4-stride kernel reading Q4_K blocks
  (fluent garbage, no error). Each LA/FA/FFN matcher + the
  `batched_gemm_single_weight` mixed-format arm needs a per-dtype case.
- There is **no GGUF-serve path** in the tree (llama-arch GGUF code is
  dequant-for-quantize tooling, not serving). Native support means HFQ
  container + dispatch entries, same as every other format.

Cost estimate: ~10 kernels (GEMV + batched-GEMM per dtype that matters)
+ loader/preflight arms + matcher routing + parity tests. Order by
param share: IQ4_XS+Q2_K first (23%), then IQ3_S (31%), then the tail.
Phase 0 (Q4_K only: passthrough + loader + `gemm_q4k_batched` +
matchers) was prototyped to prove this analysis — it serves fluent
garbage (PPL 2933) at exactly the predicted matcher gap, and is parked,
not deleted (stash `gsqrco-native-phase0-wip`, patch `/tmp/phase0_wip.patch`,
kernel `kernels/src/gemm_q4k_batched.hip` untracked). Per the
kernel-tuning skill: correct micro-pieces, flat end-to-end → reject
with evidence. Resume from the stash, don't restart.

## 5. Suggested route (staged, each stage shippable)

- **Stage 0 — land the bridge fixes (done, committed).** Norm residual,
  V-head interleave, A_log domain. They are GGUF-path corrections any
  native route inherits; the C-oracle gates + NRMSE audits stay as
  regression cover.
- **Stage 1 — IQ4_XS + Q2_K GEMV pair (23% of params).** Smallest kernels
  that prove the I-quant pattern end-to-end: `gemv_iq4_xs` (+`gemm`
  batched), `gemv_q2k` (+`gemm`), DType + table + `for_gemv` keys,
  loader/preflight arms, unfused-matcher routing (TQ2 pattern),
  `test_kernels` channel parity vs the CPU decoders in
  `gguf_iq.rs`, then serve-harness + PPL delta on a hybrid file where
  only these two dtypes go native. Success bar: hybrid PPL moves
  10.32 → ~9.6 (half the gap, proportional to share).
  **Measured (2026-09-10, gfx1151):** hybrid PPL **10.02** (vs
  all-HFQ4G256 same-pipeline baseline 10.38, vs v5 bridge 10.32) —
  +0.36 for 23.9% native, right at the proportional-to-share line
  (~0.26 expected). Serve coherent (battery 4/5 stop + 1 length-capped
  code, 0 empty/attractor). Bar partially met: the ~9.6 target
  over-estimated per-tensor gain; native replaces 4.06–2.06 bpw RCO
  choices, not a uniform-4.25 upgrade, so the win is the RCO allocation
  quality, not bit-depth.
  **Root-cause fix landed:** the Stage-1 hybrid served garbage because
  the packed passthrough skipped the DeltaNet V-head interleave the f32
  arm applies (qkv/z kept GGUF order while alpha/beta were interleaved →
  recurrence gated wrong heads). Falsified via
  `HIPFIRE_GSQRCO_SKIP_VHEADS=1` (436ec6c3c), fixed with a packed
  128-row block interleave for `in_proj_qkv`/`in_proj_z`; `out_proj`
  (128-col half-group, not a chunk move) falls back to interleaved-f32
  HFQ4G256 (34b4ae862). Follow-up if wanted: packed half-group
  interleave for native `out_proj`.
- **Stage 2 — IQ3_S (31%, the file's backbone).** Same pattern; grid
  decode is already C-verified in-tree. Success bar: hybrid → ~8.5,
  file shrinks toward ~12 GB. (Note: IQ3_S is 3.06 bpw vs the HFQ4G256
  fallback's 4.25 — the win here is size + RCO allocation quality, not
  bit-depth; expect a Stage-1-like proportional PPL move, not 8.5.)
  **Measured (2026-09-10, gfx1151):** `gemv_iq3_s` +
  `gemm_iq3_s_batched` (110 B/group, 512-entry grid) landed; decode
  mapping verified EXACT vs the C-verified `dequant_iq3_s` port (0.0
  max diff, 500 random blocks); GPU parity harness 33/33 PASS. Full
  Stage-2 hybrid (215 native: +93 IQ3_S incl. 22 in_proj_qkv + 18
  in_proj_z interleaved, out_proj still falls back) serves coherently
  (battery 5/5 stop, 0 empty/attractor). PPL **9.97** — only +0.05
  over Stage-1's 10.02. The RCO allocation puts the 3.06 bpw tier on
  low-sensitivity tensors, so native IQ3_S ≈ HFQ4G256 re-quant on this
  slice; the ~8.5 bar is not attainable. **Outlook for Stage 4:** the
  measured progression (10.38 → 10.02 → 9.97) implies full-native
  lands ~9.9, NOT within 0.5 of the release row (7.17, different
  tokenizer — llama-perplexity, 238-id divergence on the slice). The
  native route's value is SIZE (11.8 GB) + serving the release's own
  RCO allocation, not a PPL win over the in-engine mq4/mq6 ladder
  (9.23/9.16). Stage-4 admission should be re-baselined against
  in-engine formats, not the cross-tokenizer release row. Decode tok/s
  drops on IQ3_S (9.2 vs 12.3 Stage-1) — grid lookup is the cost;
  perf tuning is a follow-up.
- **Stage 3 — tail (IQ3_XXS/IQ2_XS/IQ2_XXS/IQ2_S/IQ1_M/BF16-smalls).**
  Smallest-first; BF16 smalls can stay host-F32 (0.09%, not worth a kernel).
  **Measured (2026-09-11, gfx1151):**
  - Q4_K (14%, kernels were already wired from Stage 1): passthrough
    enabled after adding Q4K to the `la/fa_has_native_iq` matcher lists
    (the Stage-1 lists checked IQ4XS/Q2K/IQ3S but not Q4K → mixed layers
    fell to the fused kernel and read Q4K blocks at the HFQ4 stride →
    '!!!' attractor). PPL 9.77, coherent.
  - IQ3_XXS (18%): new 98 B/group grid kernels, verified EXACT + 55/55
    harness. PPL 9.69 (best), greedy coherent.
  - IQ2_S (7%): new 82 B/group kernels, verified EXACT + 66/66 harness.
    PPL 9.72, greedy coherent.
  - **IQ2_XS/IQ2_XXS (2.3%/1%): kernels + full wiring landed and verified
    (88/88 harness, .hfq orientation test ~1.7e-7) but PASSTHROUGH
    DISABLED** — serving the 2.06-2.31 bpw weights natively destabilizes
    the DeltaNet DECODE recurrence: first-decode hidden state grows
    ~2.8x vs the HFQ4G256 fallback (197 vs 71 max) → `\n`/`\r\n`
    attractors on short prompts, despite IDENTICAL prefill PPL (9.72)
    and verified kernels. Root cause is decode-recurrence stability
    sensitivity to 2-bit weights, not a kernel/routing bug (exhaustively
    traced). Re-enable when the decode recurrence stability is addressed.
  - Final Stage-3 hybrid (338 native): PPL 9.72 (unchanged), **13.07 GB**
    (vs 14.03 Stage-2, ~1 GB saved), greedy serve coherent. Battery
    (temp 1.0) shows sampling weakness (1 attractor, off-topic) from the
    deeper 2-bit mixture — greedy is clean, documented honestly.
  - **Stage-3 conclusion:** the tail dtypes below ~3 bpw (IQ2_S and
    especially IQ2_XS/XXS) degrade generation quality beyond what PPL
    captures. The size win (~1 GB) comes with a real generation-quality
    cost. out_proj (48 tensors, V-head half-group interleave) remains the
    biggest non-native chunk.
- **Stage 4 — full-native file + admission.** All-native `.hfq` at
  ~11.8 GB, PPL within ~0.5 of the release row, `docs/admissions.yml`
  row (fail-closed until then). Retire the re-quant bridge per model,
  keep it per pipeline (other GSQ-RCO releases reuse it).

Non-goals: Q4_K-embedding GEMV (embed stays Q8 — lookup path, no win),
DFlash draft work (after AR is correct per the arch-port skill),
changing the release's RCO allocation (take it as authoritative, same
as the ternary precedent).

## 6. Artifact index

- Branch: `exp/gsq-rco-iq3s` @ `365ba3337` (bridge). Stash:
  `gsqrco-native-phase0-wip` (+ `/tmp/phase0_wip.patch`,
  `kernels/src/gemm_q4k_batched.hip` untracked).
- Models: `/data/rocmfpx/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` (source);
  `/tmp/gsqrco-iq3s.hfq` == `/tmp/gsqrco-v5.hfq` (md5 `3c60064f…`,
  serving artifact); `/tmp/gsqrco-native.hfq` (Phase-0 hybrid,
  diagnostic only); `~/.hipfire/models/qwen3.8-27b.{mq3,mq4,mq6}`.
- Eval: `/tmp/ppl_slice.txt` (200 KB slice head); `/tmp/fpq16_*.bin`
  per-model records; `/tmp/llama_ppl.err` (release row);
  `/tmp/gsqrco_harness3.err` (serve battery).
- Reference export for the kernel work: `/data/rocmfpx/Qwen38-27b-Escha-W2`
  (BF16 safetensors; norm-residual + dt/A_log ground truth) and
  `/data/rocmfpx/Qwen3.8F/llama.cpp` (`ggml-quants.c` / `ggml-common.h`
  C references all I-quant decoders were verified against).
