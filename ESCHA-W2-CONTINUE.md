# Escha-W2 — continue here (fresh-context handoff)

Branch: **`feat/escha-w2`** in `/home/mika/git/hipfire` (66+ commits vs upstream
master 8cd15a62). Working tree clean; last **code** change is `82637504f`
(decode gemv `LDS.64`), after which only docs moved. Verify with
`git log --oneline -8` — and note `cargo check` does not build binaries, so
rebuild `target/release` before measuring anything.

This is the one-file entry point for a fresh session. Read **this file first**,
then `ESCHA-W2-PERF-PLAN.md` (repo root) for the ranked perf work with measured
evidence and the withdrawn-estimate post-mortems. The full investigation trail
lives in `escha-port-status.md` (repo root, committed); perf evidence in
`docs/perf-checkpoints/2026-09-04-escha-w2-moe-qwen36-35b-a3b-gfx1151.md` and
`docs/perf-checkpoints/2026-09-04-escha-w2-dense-qwen38-27b-gfx1151.md`.

---

## Where it stands right now

### A. MoE — `/data/rocmfpx/Escha-W2` (Qwen3.6-35B-A3B-Escha-W2, eschamoe)

Replayed the working port from scratch `/home/mika/git/hipfire-beta` (da09de7)
onto the fresh modular tree: escham trellis/3INST/fold/grouped kernels +
`rdna-compute::escham` dispatch, `EschaSource` loader, `LayerWeights`
DeltaNetEschaMoe/FullAttnEschaMoe arms, FFN engine `qwen35/escha_ffn.rs`.
Coherent: France→"**Paris**", Japan→Tokyo, haiku, code fences, 128-token
rainbow explanation at **10.6 tok/s**. Unchanged by this round's work.

### B. Dense — `/data/rocmfpx/Qwen3.8-27B-Escha-W2` (Qwen3.8 27B, escha)

Full in-kernel decode pipeline (no full-size dequant): `DenseEschaSource`
loader, `LayerWeights` DeltaNetEscha/FullAttnEscha, HIP decode-gemm kernels
(`kernels/src/escham/hip/escha_dense_kernels.hip`: rotate-in T128 →
in-kernel tile decode → finalize) + `rdna-compute/src/escha_dense.rs` +
`qwen35/escha_dense_{decode,forward}.rs`. Per-projection decode **exact** vs
host/EschaLabs (rel ≤ 6e-4).

**Current perf (2026-09-07, gfx1151 / Strix Halo, 58-token prompt, warm, 3-run):**

| | 2026-09-05 | now |
|---|---|---|
| decode (tg) | 1.9 → 3.4 tok/s | **5.0 tok/s** |
| prefill (pp) | 3.4 tok/s ≡ decode (16.8 s TTFT) | **16.7 tok/s, 3.47 s TTFT** (4.8×) |

Prefill finally separated from decode this session. Output coherent, verified
factual (Canberra), correct arithmetic (17×23=391) at serving temperature.

**The long-horizon decay / attractor is FIXED** — see B7 below (`18871f381`,
q/k norms were gamma-1 loaded raw). Do not re-open it.

---

## What changed this session (all measured, all committed)

1. **Prefill was silently running the per-token path.** `qwen35_layer_batch_admissible`
   still returned `Err` for `DeltaNetEscha`/`FullAttnEscha` long after
   `9daf925bf` "fixed" the attractor that closed it, so
   `prefill_batch_pbs_eligible()` was false for the **whole model** and prefill
   fell through to the loop the source calls "byte-identical to decode". That is
   why pp == tg exactly: 58 × 285 ms reproduced the measured 16.8 s TTFT to
   within 1.5%. Gate re-opened on shape validation. `HIPFIRE_ESCHA_DENSE_BATCHED=0`
   restores per-token.
2. **Latent bug found in the batched FA arm**: it called `forward_scratch_layers`,
   which iterates `0..config.n_layers` — a whole-model forward, replayed once per
   FA layer per token. Now calls `fullattn_escha_layer_forward` (one layer).
   Unreachable until (1), so it had never fired.
3. **R and K are now compile-time** in `matmul_prefill` (12 `extern "C"`
   instantiations, K∈{2,3} × R∈{1,2,4,8,16,32}). Measured ISA:
   `private_segment_fixed_size` **272 → 0**, VGPR 19 → 48 @ R=16. The `acc[R]`
   array was in scratch, not registers — that is why row-batching only ever
   bought 1.39–2.68×.
4. **`n_slices >= R` floor removed.** It was accidentally capping dynamic shared
   memory; without a real bound, a 600-row chunk asked for 68.75 KB and
   `hipModuleLaunchKernel` failed with `invalid argument`. Replaced by an
   explicit smem bound (`b8dcefcb9`), with a test shown non-vacuous (8 violating
   shapes → 0).
5. **Biggest decode win came from reading the disassembly, not from the plan**:
   `(w0 + NW - 1) % NW` in the per-weight loop emitted a **full integer division**
   (36 `s_mul_hi_u32` + 39 `s_addc_u32`), because the compiler cannot prove
   `w0 < NW`. Written as `w0 ? w0-1 : NW-1` it is one select: gemv 1.10 → 0.63 ms,
   **tg 3.4 → 4.6**. Note that making K compile-time did *not* fix this — the
   range proof was the blocker, not the constant.
6. **One `LDS.64` per weight** instead of two `LDS.32`, via the overlapping
   `uint2` payload pairs the file's own header already described but never used:
   gemv → 0.566 ms (157.6 G weights/s), **tg 4.6 → 5.0**.
7. **New oracles** (`check_escha_dense_batched`, `check_gdn_batched`) — see
   "Batched-path verification" below. These are what let me retract a wrong
   conclusion about (1).

---

## Batched-path verification — read this before doubting (1)

I initially concluded the batched path was *wrong* because it diverges from
per-token at `--temp 0` from the first generated token, and I flipped the gate
back off on that reasoning. **That reasoning was backwards**, and the oracles
refute it:

| probe | result |
|---|---|
| `check_escha_dense_batched` — batched projection vs **host** reference | ≤ 4.07e-4 rel, **identical at every R from 1 to 32** |
| `check_gdn_batched` — batched conv carry vs N per-token calls | **bit-exact (0.0)** |
| `check_gdn_batched` — batched GDN S-matrix carry | 1–2e-7 |
| **control**: two *batched* runs differing only in chunk size (R=64 vs R=2) | disagree with each other **as much as either disagrees with per-token** (L0 5.5e-3 vs 9.8e-3; L63 2.1e-1 vs 2.5e-1) |

The control row is the one that settles it: a difference that appears between two
batched runs of identical kernels is fp32 **summation-order reshuffling**, not
error. Per-token is not ground truth — the host reference is, and batched passes
it. This repo documents repeatedly that ~1 ULP flips an argmax over 2k greedy
tokens, so greedy divergence is the most sensitive possible metric for the least
information. Gate is therefore **DEFAULT ON**.

**Honest caveat for the owner:** under greedy at 1500 tokens *both* arms
degenerate (per-token worst 6-gram repeat 33, batched 213), so greedy cannot
adjudicate either way, and batched is not obviously *better*. Ship acceptance
should be your own 10+ prompt battery at serving temperature on both arms, not my
three prompts. `HIPFIRE_ESCHA_DENSE_BATCHED=0` is a one-env-var revert.

---

## Two premises in the old plan are refuted by measurement

Withdrawn in `25865f7e5`; do not re-estimate them from the old numbers.

- ~~"`lm_head` int8 → **tg +35–40%**"~~ assumed decode is DRAM-bound. It is not:
  the gemv reads 22.3 MB in 0.566 ms = **39 GB/s** against **~104 GB/s** measured
  sustained read (new `rdna-compute/examples/mem_bw.rs`: 208 GB/s D2D r+w at
  2 GiB = 81% of LPDDR5x theoretical; 718 GB/s at 8 MiB is L2, so the 2 GiB
  number is real DRAM). Cutting bytes cannot deliver tg. It still saves ~3.8 GB
  of VRAM, which is worth something for context length.
- ~~"fuse `rotate_in`+gemv+`finalize` → **tg +50–100%**"~~ Measured per stage:
  rotate **9.4 µs**, finalize **3.8 µs**, gemv **566 µs**. The two "wasted"
  launches are **2%** of a projection. Fusion is not a lever here.

Also recorded as a negative result: **splitting the accumulator 4 ways to shorten
the FFMA dependency chain made it 30% *slower*** (0.63 → 0.83 ms) and was
reverted. And the `n_slices` sweep is **flat from 952 to 43 520 blocks**
(0.63–0.70 ms), which rules out occupancy. So the remaining ~3× gap to
bandwidth-limited is neither chain latency nor block parallelism — **I do not
know what it is**, which is why the next decode step should be WMMA (item 9), not
another scalar micro-optimisation.

---

## Next steps, in priority order

1. **Batch the 16 `FullAttnEscha` layers inside the chunk loop** — *now the
   dominant prefill cost*. The chunk's GDN half is batched and fast; FA is still
   a per-token gather/scatter loop, so FA ≈ 16/64 of weights × 58 tokens of
   full-speed decode is essentially all of the remaining 3.47 s. The MQ path
   already solves this shape in `batch_chunk_full_attn_attn` (`prefill.rs:5146`) /
   `batch_chunk_full_attn_ffn` (`prefill.rs:5597`), dispatched at `prefill.rs:7457` — mirror it for coded
   wq/wk/wv/wo. Task #71.
2. **MTP does not engage, and the reason is now known.** `--spec mtp` measured
   **5.0 tok/s, identical to `--spec off`**. The loader
   (`hipfire-loader/src/lib.rs:2007,2032`) looks only for a bundled `.mq4-mtp`
   trailer or a **sibling `<trunk>.mtp` file in HFQM container format**
   (`mtp_head::load_mtp_head` → `HfqFile::open_at_offset`, arch_id 21, produced
   by `mtp_extract`). The draft on disk is `/data/rocmfpx/Qwen3.8-27B-Escha-W2/mtp/`
   — a **directory** containing raw HF `model.safetensors` + `config.json`. So it
   is a *format* gap, not a dispatch bug. Work is: write an HF-safetensors-MTP →
   HFQM converter (or teach the loader to read the directory). Worth it — it's
   the only tg lever that doesn't require beating the gemv, and the reference
   runtime reports 1.77–1.92×.
3. **WMMA/MFMA coded prefill GEMM** (plan item 9) — the real ceiling-raiser for
   pp, and probably for decode too. gemv is at 157 G weights/s vs ~515 G implied
   by measured bandwidth. The reference's `escham_code_gemm` reaches 146 TFLOPS
   (fp32 acc) / 215 TFLOPS (fp16 acc) by decoding straight into MMA B-fragments.
4. hipGraph AR capture for escha-dense (plan item 8) — demoted: per-launch cost
   is now small against gemv time. Prereq (scratch hoisting) is already done.
5. MoE throughput: beyond 10.6 tok/s requires the fp16/int8 dense attention path
   (dense attention is int8→f32 at load, ~4.5 GB/token f32 traffic on the MoE).

---

## How to build + run (mandatory env)

```bash
cd /home/mika/git/hipfire
nix develop .#therock --command bash -c '
  cargo build --release
  export HIPFIRE_HOME=/tmp/hf-$(date +%s)      # FRESH per run (stale singleton + config live here)
  mkdir -p $HIPFIRE_HOME
  ./target/release/hipfire config set reasoning.mode off >/dev/null 2>&1
  HIPFIRE_LOCAL=1 HIPFIRE_DAEMON_BIN=$PWD/target/release/daemon \
    ./target/release/hipfire run /data/rocmfpx/Qwen3.8-27B-Escha-W2 -n 128 "Why is the sky blue?"
'
```

Why each part is load-bearing:
- **`nix develop .#therock` wrapping the whole thing.** If a `.hip` file changed,
  the JIT cache is source-hash invalidated and the daemon must recompile at run
  time; outside the dev shell hipcc has no C++ stdlib and the run dies with
  `"Could not find standard C++ header 'cmath'"` *after* a 4 s model load. This
  bit me twice. Same reason `cargo run --example …` needs the dev shell.
- `HIPFIRE_LOCAL=1` + `HIPFIRE_DAEMON_BIN=$PWD/target/release/daemon` — a
  **stale system-installed serve** (nix store, user `hipfire`) listens on :11435
  and would otherwise intercept `run` with an OLD binary that fails config
  parsing (`unknown configuration key 'serve.continuous_batch_size'`).
- `HIPFIRE_HOME=/tmp/<fresh>` — the daemon is a per-home singleton (`daemon.pid`)
  and older configs in a reused home can break startup. (Using `$PWD` as the home
  works but writes `config.toml` and rewrites the tracked `registry.cache.json`
  into the repo — revert that file before committing.)
- **`target/release` really has to be rebuilt.** `cargo check` does not produce
  binaries. A whole round was spent on identical numbers because the binaries
  predated the commits.
- `reasoning.mode off` — **this model always reasons; `off` only zeroes the
  budget and hides the trace** (owner's correction). It is a short-run device:
  with `on`, a run that ends inside `<think>` fail-closes with
  `open think span at end of generation`, and `-n 4/100/200` is not enough for
  the dense 27B. So either set `off` for quick checks, or use `on` with
  `-n ≥ 400` for real quality readings. My `-n 200` runs produced spurious
  failures and one false alarm.
- `[stats]` lines go to **stderr**, and the daemon can also fail-close a
  perfectly good generation — so don't `2>/dev/null` when you need a measurement.

MoE runs identically with `/data/rocmfpx/Escha-W2`.

Useful debug levers, all env-only: `HIPFIRE_DEBUG_BATCH=1` (prints the
batch-eligibility decision — `result=true` is what makes batched prefill
happen), `HIPFIRE_ESCHA_DENSE_AUDIT[_LAYER][_POS]` (live GPU-vs-host decode
audit on real activations), `HIPFIRE_ESCHA_DENSE_TRACE`,
`HIPFIRE_ESCHA_DENSE_NO_ATTN/_NO_FA_GATE/_NO_FFN` (arm ablation),
`HIPFIRE_PREFILL_MAX_BATCH` (chunk size),
`HIPFIRE_GDN_CHUNKED` (parallel chunked GDN, default off — untested on escha).

Gotcha: `pkill -f "target/release/daemon"` matches its **own** wrapper argv and
kills the shell running it. Use `pgrep -x daemon` / kill by PID.

---

## Key files (dense + MoE)

| Area | Path |
|---|---|
| Perf plan, ranked + measured | `ESCHA-W2-PERF-PLAN.md` (repo root) |
| Handoff / status trail | `escha-port-status.md` (repo root) |
| MoE FFN engine | `crates/hipfire-arch-qwen35/src/qwen35/escha_ffn.rs` |
| MoE loader | `crates/hipfire-arch-qwen35/src/qwen35/escha_load.rs` |
| Dense loader (+norms) | `crates/hipfire-arch-qwen35/src/qwen35/escha_load.rs` (DenseEschaSource) |
| Dense decode engine/ref | `crates/hipfire-arch-qwen35/src/qwen35/escha_dense_decode.rs` |
| Dense forward arms | `crates/hipfire-arch-qwen35/src/qwen35/escha_dense_forward.rs` |
| Dense decode scratch | `Qwen35Scratch.escha_dense_u` / `escha_dense_partial` in `forward.rs` |
| Batched prefill driver + eligibility gate | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs` (`qwen35_layer_batch_admissible`, `prefill_batch_pbs_eligible`, `forward_batch_chunk_impl`) |
| Dense kernels (HIP) | `kernels/src/escham/hip/escha_dense_kernels.hip` |
| Dense dispatch + slice heuristics | `crates/rdna-compute/src/escha_dense.rs` (has unit tests) |
| MoE kernels (HIP) | `kernels/src/escham/hip/{escham_moe_decode_trellis,escham_moe_fold,escham_moe_grouped_kernels}.hip` |
| MoE dispatch | `crates/rdna-compute/src/escham.rs` |
| Host decode reference + tests | `crates/hipfire-arch-qwen35/src/escham_decode.rs` |
| Oracles (GPU-vs-host) | `examples/{compare_decode,check_escha_ffn,pin_funnel,check_escha_dense,check_wrow_norm}` + **`check_escha_dense_batched`**, **`check_gdn_batched`** (both exit nonzero on FAIL) |
| Perf probes | `examples/profile_escha_dense_prefill.rs` (pp, per-row speedup), `hipfire-arch-qwen35/examples/decode_stage_probe.rs` (**tg, per-stage + n_slices sweep**), `rdna-compute/examples/mem_bw.rs` (device bandwidth) |
| Reference runtime (untracked) | `./escha-runtime-qwen3dense/` — sglang + `escha/_C*.so`; 54 cubins recoverable with `cuobjdump -xelf all`. Left untracked on purpose (51 MB of build artifacts). |

Kernel ISA metadata without a rebuild: the JIT cache writes
`.hipfire_kernels/<arch>/*.radiowave.json` (VGPR/SGPR/spill/`private_segment_fixed_size`).
A standalone `hipcc`/`clang++` invocation fails on C++ headers, so read the
radiowave JSON, or bundle-compile and parse the AMDGPU metadata note:
`hipcc --genco --offload-arch=gfx1151 -O3 --no-offload-compress -o x.hsaco f.hip`
then carve the ELF (magic `__CLANG_OFFLOAD_BUNDLE__`, device blob at offset 4096)
and scan for `vgpr_count` / `private_segment_fixed_size`.

---

## Milestones recap (root causes worth remembering — do NOT re-investigate)

1. **Decode model**: EXL3 trellis + 3INST codebook + tensor-core-perm — pure
   function of the int16 codes, no codebook file. GPU decode == host ==
   EschaLabs (rel 2e-4). Kernel math in `escham_decode.rs` / the .hip files.
2. **Fold convention**: `y = s_out ⊙ T128(T128(x·s_in·rin) @ w_bare) · rout`
   (scales OUTSIDE the blockwise-128 WHT; MoE folds rout·s_in·rin into cached
   fp16 weights — an exact identity). `rin` already has Wscale folded — never
   re-apply it.
3. **MoE per-expert scale bug** (b967a721): the routed-FFN cache fill passed
   the whole stacked `[n_exp, in]` scale tensor to `escham_apply_rowcol_scales`
   (kernel reads the first `in_p` elements) — every expert folded with expert
   0's s_in·rin. Fix: per-expert `sub_offset(exp_idx*in_p, in_p)`.
4. **Dense funnel pairing** (73eebd14): llama.cpp's overlapping-uint2 payload
   trick does NOT match this export's safetensors layout. Exact index =
   `funnel(hi=word w0−1, lo=word w0) >> (sp&31)` over a plain word array.
   Pinned empirically on real model tiles (`examples/pin_funnel.rs`).
   ⚠️ **Careful, this note is about global→shared staging order, and it has since
   been superseded for the *shared* layout:** the decode gemv now stores
   overlapping `uint2` pairs in shared memory and reads one `LDS.64` per weight
   (`82637504f`), which is bit-identical to the old two-word version per
   `check_escha_dense`. What must NOT change is *which pair of words* forms
   (hi, lo): hi is word `w0−1 mod NW`. Do not re-derive the pairing.
5. **Dense norm conventions** (888ef3b08, 1ea28000): `q_norm`/`k_norm` are
   TRUE gamma (raw, no +1); the LA gated-output norm (`linear_attn.norm`)
   is gamma−1 (+1). Layer input/ffn norms are gamma−1 in both exports.
   (Superseded by **B7** below for FA q/k — those are gamma−1 / +1 too.)
6. **Dense multi-token decay root cause** (5bb373af): AR hipGraph
   capture/replay engaged on escha-dense (`use_graph` excluded
   `is_escham_moe` but not `is_escha_dense`); the per-projection decode allocs
   pool scratch per call (capture-unsafe) → replay diverged at decode token ~3.
   Fix: add `&& !config.is_escha_dense` to the `use_graph` predicate in
   `crates/hipfire-arch-qwen35/src/qwen35/forward.rs`.
7. **Decode scratch hoisting** (70d860bb9): `escha_dense_decode_proj` allocs
   `u` + `partial` pool tensors per call (capture-unsafe). Hoisted into
   `Qwen35Scratch` as `escha_dense_u` / `escha_dense_partial` (gated on
   `config.is_escha_dense`). `escha_dense_rotate_in` now takes explicit `ic`
   (scratch is oversized for max projection). Per-call alloc/free removed.
   **Prefill scratch is still allocated per projection per chunk** — hoisting it
   into `PrefillBatchScratch` is open (see plan item 4 follow-up).

Env-gated debug toggles (all harmless, MoE path untouched):
`HIPFIRE_ESCHA_DENSE_TRACE/_LOGITS/_NO_FFN/_NO_ATTN/_STATE_FP32/_RAW_NORMS`,
`HIPFIRE_ESCHA_DEBUG` (MoE probes were stripped from escha_ffn but the env
read sites may remain).

---

## B6 — long-horizon decay re-diagnosis (2026-09-04, fresh session)

User reports: "Why is the sky blue?" on the dense 27B produces a coherent
Rayleigh start then collapses into a "* Blue: Blue. Blue. Blue…" verbatim
attractor, and reasoning is not visible. Model exonerated by the user (coherent
to 256K ctx on CUDA 3090/4090/5090 AND llama.cpp-escha on 7900 XTX, "quality
comparable to Q8").

**Reproduced + characterized:**
- B5's four gate prompts still pass EXACTLY as claimed (France→"Paris",
  self-intro, haiku, sky ≤50 tok). The B5 gate horizon (≤50 tok) was simply too
  short — the decay starts at ~60–150 generated tokens.
- Deterministic at temp 0 AND at temp 0.3/0.8 with repeat penalty up to 1.15:
  the escha-dense hidden state settles into a stable-but-wrong fixed point
  (verbatim self-echo) past the horizon. Story prompt ("robot Spark") starts
  beautifully coherent, decays into stream-of-consciousness echo by ~tok 100.
- Prompt memory is PERFECT (recalls "code word = zebra" at 10/50/120 tokens
  back) — not a context/KV/prefill problem.
- Reasoning DOES run with a budget (think-cap force-closes correctly) but the
  THINK stream itself decays into the same verbatim echo ("especially nitrogen
  and nitrogen? Actually molecules of nitrogen and nitrogen?…"). The think
  stream was invisible because `hipfire run`'s generate callback only printed
  `type:"token"` events and dropped `type:"reasoning"` — FIXED (run now shows
  reasoning and reports it in --json).

**Excluded with evidence (this session):**
- Per-projection in-kernel decode == host == llama funnel formula, BIT-EXACT on
  real tiles (all 256/256, K2+K3, deep layers) AND live-decode audit on real
  activations ≤ rel 2e-4 at every position through the collapse (new env-gated
  audit: HIPFIRE_ESCHA_DENSE_AUDIT[_LAYER][_POS]).
- DeltaNet state FP32 == Q8 byte-identical text (state quant not involved).
- Controls coherent on the same tree at the SAME shape: MoE escha 2-bit
  (attention int8) → 800 tok; HFQ-dense qwen3.6-27b.mq4r (plain MQ4, identical
  5120/64L/24H/4KV/16kH/48vH/128hd) → 800 tok story coherent. Shared kernels
  at this shape exonerated.
- Bias NOT applied (matches llama build_escha_mm comment: EschaLabs "ignoring
  them is what reproduces the results published here"); s_in/s_out fold correct.
- FA passthrough still decays (LA implicated); FA removed entirely so not a
  clean isolation. Structure mirrors llama qwen35.cpp graph op-for-op.
- Embed/lm_head int8 scale sanity: 304/248320 rare-token rows carry extreme
  scales (32–16320 vs ~1e-3) — benign-looking but NOT yet ruled out.

**OPEN (the actual remaining bug):** long-horizon-only coherence decay in the
escha-dense 2-bit path. Every per-token mechanism is proven exact; the decay is
a temporal-accumulation property that NO coherent control exercises (no control
runs 2-bit ATTENTION q/k/v/o at 5120-dim/64L). Highest-value next experiments:
1. Golden token stream from the EschaLabs sglang/escha reference runtime on the
   same model (needs torch+wheel setup; ~28G free disk on /home may be tight) —
   compare token-level where hipfire first diverges.
2. Materialize the dense projections to folded fp16 WeightTensors at load and
   route through the PLAIN DeltaNet/FullAttn arms (the "decisive never-run" B3
   experiment) — distinguishes arm wiring from decoded-value issues.
3. Bisect by layer-count: reduce effective LA depth (config surgery) to see if
   decay horizon scales with recurrence depth.

---

## B7 — ROOT CAUSE SOLVED: FA q/k norms were gamma-1, loaded raw (2026-09-04, commit 18871f381)

The B6 "long-horizon decay" is FIXED. Root cause: the dense export stores
`self_attn` q/k RMSNorm weights as **gamma-1 offsets** (raw means
+0.23/+0.22 with a handful of NEGATIVES — impossible for a true gamma),
and hipfire loaded them **raw** (bias 0.0), running the whole FA path at
~5x under-scale. The B1c A/B (888ef3b08) that chose raw was confounded:
+1 was tested while the LA gated-norm was still raw (fixed later in
1ea28000), so the +1 arm was judged inside a still-broken model.

Authoritative evidence (not another A/B): llama.cpp's HF→GGUF conversion
(`conversion/qwen.py::modify_tensors`) adds +1 to every `norm.weight`
EXCEPT `linear_attn.norm.weight`. `q_norm.weight`/`k_norm.weight` match
the `norm.weight` suffix → the reference runtime (coherent to 256K ctx)
runs them as (raw+1). Fix: bias 0.0 → 1.0 in `escha_load.rs`.

Verified on gfx1151 (temp 0, reasoning off; OLD vs NEW same-prompt A/B):
- Spark n=120: OLD top 8-gram x4 verbatim loop from para 1 → NEW top
  8-gram x2, varied prose, no verbatim loop.
- Directional factual battery 5/5 (B1g was 3/6): France→Paris,
  Japan→Tokyo, Paris→France, Tokyo→Japan, Germany→Berlin.
- Sky n=200/300, quantum n=300, rainbow n=128, fibonacci n=100, haiku,
  self-intro: coherent. MoE unregressed (Paris, haiku).

Residual (second, smaller effect — NOT this bug): greedy temp-0 creative
prose still thematically loops at ~150–260 tok on BOTH Q8 and FP32 state;
MoE/HFQ controls stay clean to 800. The q/k fix moves the horizon
12 → ~200 tok and converts immediate verbatim echo into late thematic
looping. With temp 0.3 + repeat-penalty 1.1 the dense path is clean to
300 tok (top 8-gram x1). Next lever for the residual: state precision /
2-bit attention fidelity work (M3), not FA scaling.

**Confirmed again 2026-09-07:** at greedy `--temp 0` and `-n 1500` the dense path
still degenerates (**both** with batched prefill on *and* off, repeat counts 33
vs 213), while at serving temperature (0.3) short runs are clean and factually
correct. So the residual is greedy-only, pre-existing, and independent of the
prefill path — consistent with this section. Do not mistake it for a prefill
regression.

---

## Corrected field notes (do not re-derive)

- `escha_config` is **int32 `[tile=16, K, V=2, codebook_id, IC, OC]`**, NOT "6 floats
  near 1.0" — that earlier reading was int32 reinterpreted as float. Verified across
  all 400 projections against the matching `escha_code` shapes. `codebook_id == 1`
  everywhere, so the single hardcoded codebook is right. `escha_load.rs` already
  parsed this correctly.
- The checkpoint is **mixed-rate**: `gate_proj` and all attention / linear-attn
  projections are **K=2** (32 B per 16×16 tile, 0.25 B/weight); `up_proj` and
  `down_proj` are **K=3** (48 B, 0.375 B/weight). Consequence: gate and up can never
  share one coded-GEMM launch with a single K — fusing them needs K-grouped slicing.
  The reference handles this in `escham_multi_gemv` with per-K launches.
- **The `*.bias` tensors on disk are dead.** Every coded projection ships one, but the
  authoritative sglang runtime builds all of them with `bias=False`
  (`qwen3_5.py:121/134/143/152/161/232/510/520`, `qwen2_moe.py:109/118`) and its
  `load_weights` skips `.bias` names absent from `params_dict`. hipfire not applying
  bias matches the reference; stop treating this as an open question.
- RoPE: reference is `rotary_dim=head_dim`, `partial_rotary_factor=0.25`,
  `is_neox_style=True` → 64 of 256 dims, half-split = hipfire's
  `rope_partial_halfsplit_f32`. `mrope_interleaved` is inert for text (t==h==w).
- GDN reference semantics hipfire matches: `g = -exp(A_log)·softplus(a+dt_bias)`
  with **stabilized** softplus (`threshold=20`, not `F.softplus`), `beta=sigmoid(b)`,
  in-kernel L2-norm of q/k, `RMSNormGated = RMSNorm(x)·silu(z)` with a **plain**
  weight (no +1), `norm_before_gate=True`, conv state bf16 / ssm state fp32.
- Model shape (verified from `config.json`, not from Qwen3Next defaults):
  **64 layers, hidden 5120, intermediate 17408, vocab 248320, 24 attn heads /
  4 KV heads / head_dim 256, 48 GDN + 16 FA** (FA at idx 3,7,…,63),
  `attn_output_gate=True` (so q_proj OC = 24·256·2 = 12288), GDN 16 k-heads×128 +
  48 v-heads×128 → conv kernel input OC = 2·2048 + 6144 = 10240.
- FA layers store q/k/v/o **separately** on disk; concatenating along the `OC/16`
  axis is a clean code concat, so a merged QKV gemv is available as a win.
- `mtp/` on disk is a **safetensors directory**, not a `.mtp` HFQM container —
  see Next steps item 2.

External references: llama.cpp-escha fork
`/home/mika/git/llama.cpp-escha` (branch `escha-w2-dense`, commits 2a238a40d +
2940b807c); EschaLabs authoritative reference runtime `/home/mika/test-2/escha/`;
AMD HIP validation of the same dense path: `/home/mika/git/escha-amd-port`
(gfx1030, yaminerl) — confirms fp32-FMA decode-gemm runs on AMD and the
portable codebook spelling is exact under HIP. Packaged reference with kernels:
`./escha-runtime-qwen3dense/` (sglang + `escha/_C*.so`).

---

*Last updated 2026-09-07 (batched prefill on: 4.8× pp; decode gemv modulo +
LDS.64: 3.4 → 5.0 tok/s; oracles added; two plan premises refuted by measurement).*
