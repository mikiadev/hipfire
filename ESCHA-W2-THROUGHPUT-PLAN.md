# Escha-W2 throughput plan, rev 2 — beat PR #694, don't duplicate it

Status: DRAFT for owner review. This supersedes the decode-first ordering in
`ESCHA-W2-PERF-PLAN.md` (§ "Suggested execution order" / "Revised order:
5 → 10 → 9 → 8") and reframes the branch trajectory. Nothing here is measured
yet — every number attributed to PR #694 is *their* number, quoted with source.

Branch: `feat/escha-w2` @ `30ec1cb8d`. PR: warpfront/hipfire#694
(`nw_escha_w2` @ `33ba046e0`, 101 commits, +21857/−803, OPEN, `meta:wip`).
Author: nwoolmer. Local checkout of the PR head: `git fetch origin
pull/694/head:pr-694` (already fetched in this repo as `pr-694`).

Owner directive (2026-09-07): throughput is the goal; do not discontinue this
branch's work, but change trajectory toward *beating* the PR's numbers; run a
quality-vs-repack experiment on how repacking/quanting affects quality vs
reading the safetensors directly.

---

## 1. What the PR actually ships (and what it doesn't)

Three artifacts per model, six total, published on HF as
`hipfire-models/qwen3.6-35b-a3b-escha` and `hipfire-models/qwen3.8-27b-escha`.
None of them is on this box (22 GB free on `/`, 27 GB on `/data` — no room to
pull all six; see §4). Quoted numbers are gfx1151 unless noted:

| artifact | 27B size | 27B prefill | 27B decode | 27B PPL | 35B decode |
|---|---|---|---|---|---|
| `.escha-xt` | 10.45 GB | 122 tok/s | 12.3 tok/s | 9.7242 | 63 tok/s |
| `.escha` (default) | 10.77 GB | 119 tok/s | 12.1 tok/s | 9.6753 | 55 tok/s |
| `.escha-pro` | 11.16 GB | 113 tok/s | 10.8 tok/s | 9.6486 | 47 tok/s |

Naming: `-xt`/base/`-pro` order by size; the suffix describes the **dense**
tensors only — coded weights are byte-identical across all three builds of a
model. MQ6 dense is the default (+0.28% PPL on 27B for +17% decode). `-pro` is
a bit-exact repack of Escha's per-row int8 into per-32-block Q8_0.

Reference ladder (same slice + harness): `mq6` 21.75 GB / PPL 9.0042 /
9.5 tok/s; `escha` 10.77 GB / 9.6753 / 12.1 tok/s; `mq3` 12.62 GB / 10.0643 /
15.7 tok/s. Every escha build beats `mq3` on quality while smaller, beats
`mq6` on speed *and* size, and trades tok/s against `mq3` for quality-per-byte.

### Architecture in one paragraph

New quant types `ESCHA2T16 = 42` / `ESCHA3T16 = 43` + `RotationPlan::EschaH128`,
safetensors→`.hfq` converter (`pipeline_escha.rs`, byte-verbatim code streams,
`memcmp` post-condition), and the runtime split by model:

- **35B MoE**: codes stay verbatim; decode inside the GEMV
  (`escha_moe_gemv_native.hip`, in-register trellis decode, ~127 GB/s
  effective — the 7-ops/weight codec cost is measured at 1.83× vs no-decode).
  Q8_0-at-load was Phase 1 (69 tok/s roofline, could never beat mq4r's 71.8);
  native is Phase 2 (94 tok/s roofline). H128 pair runs at runtime around
  existing Q8 GEMVs. Plus: nt-major tile transpose at load (+24% decode GEMV),
  grouped WMMA GEMM for prefill (52 → 108 tok/s), in-trunk `mtp.*` resolver
  (commit `e1d2d55`, accept rate 0.568, 2.685 tok/window — **MTP works on
  their 27B build**).
- **27B dense**: codes stay verbatim (`Escha2T16/3T16` `WeightTensor`s);
  `escha_dense_linear_forward` = H128-in → GEMV → H128-out + bias per
  projection. Notably they did NOT go fully native-decode-in-GEMV on dense —
  the dense GEMV work is later (`1d7d1ea65` fused routed GEMV, `56aa9c874`
  native through batched prefill, `8fd307e01` dense-decode bench).

### What the PR explicitly leaves open (our attack surface)

1. **Decode trails plain MQ quants**: escha ~127 GB/s vs mq3/mq6 ~198–207 GB/s
   (memory ceiling). Their own ablation: removing decode arithmetic is 1.83× →
   would put escha at ~203 GB/s (~18 tok/s). The codec's 7 ops/weight buys the
   2-bit residency. (PR body, "Why decode trails".)
2. **Routed half doesn't amortise in prefill** (§10.5a): batched prefill
   amortises launches + activations, NOT expert weight traffic — each token's
   top-8 still read per token. That's why escha stays ~1.8× behind mq4r's
   289.6 tok/s at prefill. Fix needs sorted-expert-group grouped GEMM, which
   breaks their G4 bit-equality gate (would need restating as a bound).
3. **Prefill/decode select different experts at 2.96%/slot** (§10.5b): f16
   activation downcast for WMMA GEMMs perturbs router inputs across the f16
   rounding boundary. Final-token logit delta max 4.393e-1. Argmax stable (G6),
   logits not close.
4. **MTP on 27B**: resolver exists (`e1d2d55`), accept 0.568 measured — but the
   PR's headline 27B table (12.1 tok/s) does NOT include MTP multiplication.
   Our branch's MTP gap is a *format* gap (safetensors dir vs `.mtp`); theirs
   is solved by converting to `.hfq` with in-trunk `mtp.*`.
5. **No lossless repack into MQ** (§2 of their design doc): 10,746 distinct
   fp16 values per expert projection vs MQ2's 4/group — impossible, not lossy.
   But FOLDING (H128+diagonals baked into dense weight, then MQ6/MQ4V2) is
   implemented as `HIPFIRE_ESCHA_FOLD=mq6|mq4v2|f16` — quality-preserving path
   into containers hipfire already runs at full speed. **This is the bridge
   between their work and the owner's repack-vs-quality question (§3).**

---

## 2. Where this branch stands relative to the PR

Different architecture, not a worse one — the two approaches are complementary:

| | PR #694 (`nw_escha_w2`) | this branch (`feat/escha-w2`) |
|---|---|---|
| Input | `.hfq` via their converter (verbatim codes) | safetensors dir directly (in-kernel trellis decode) |
| 27B decode path | `EschaDenseLinear`: H128 + GEMV + H128, separate kernels | fused rotate→decode→finalize (3 launches, now measured 2% overhead — fusion refuted) |
| 27B decode | **12.1 tok/s** (their `.escha`) | **5.0 tok/s** (scalar FMA gemv @ 157 G weights/s vs ~515 G implied) |
| 27B prefill | **119 tok/s** | **~29 tok/s** (just landed: FA batching `e63add0e`) |
| Quality evidence | KLD/PPL gates G1–G6, `escha-kld.sh`, corpus slice | coherence eyeball + greedy battery; NO KLD/PPL yet |
| MTP | in-trunk resolver, accept 0.568 | format gap (safetensors dir unreadable by loader) |
| Strength | quality-gated, shipped artifacts, 2.4× our decode | live-decode path, oracles, FA-batch infra they lack |

The gap is real and quantified: **2.4× on decode, 4.1× on prefill.** Their
127 GB/s effective vs our 39 GB/s says it plainly — their kernels extract 3.3×
more of the box than ours. Scalar micro-optimisation (modulo, LDS.64) took us
3.4 → 5.0; the remaining 5.0 → 12+ is structural (WMMA/MMA decode, grouped
prefill GEMM, nt-major layout), and they have already proven each structure.

**Decision: rebase the trajectory onto their runtime, keep our verification.**
Do not re-derive the codec, the converter, the H128 pair, or the KLD harness —
all gated upstream. Our branch's durable assets are: the safetensors-direct
loader + oracles (independent verification path), the batched-FA chunk infra
(`e63add0e`, applicable to any dense FA body), the KV-index fault analysis,
and the measurement discipline (refuted premises documented, not deleted).

---

## 3. The quality experiment (owner's question)

**Question**: how does repacking/quanting affect quality vs reading the
safetensors directly?

**Design** (Astrea workflow: fingerprint → plan → calibrate → eval →
metrics → report; quality claims need measured KLD/PPL per skill rules):

- **Arm A (safetensors-direct)**: this branch @ `30ec1cb8d`, serving
  `/data/rocmfpx/Qwen3.8-27B-Escha-W2` as today. This is the "no repack" arm.
- **Arm B (verbatim `.hfq`)**: PR converter output (codes byte-identical,
  `memcmp`-gated by their G1). Quality delta A↔B isolates *our loader path*
  vs *their container path* with identical numerics — expect ~0; anything else
  is a wiring bug on one side, and that itself is the finding.
- **Arm C (fold+MQ6)**: `HIPFIRE_ESCHA_FOLD=mq6` build — escha quality baked
  into a container hipfire runs at full speed. Delta B↔C prices the fold's
  second quantisation (their table: +0.28% PPL on 27B for +17% decode).
- **Arm D (dense down-quant)**: `HIPFIRE_ESCHA_DENSE=mq6` — second quantisation
  on the non-coded tensors (their KLD table: everything 0.009785, PPL 7.6965).

**Method**: `scripts/escha-kld.sh` (theirs — teacher-forced, fixed corpus
`benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt`, f32 KV,
negative control asserting 0.000000) + local `eval_hipfire`
(`crates/hipfire-runtime/examples/eval_hipfire.rs`, `--scoring-mode
per-token` for the escha path) + `build_kld_ref_native` reference. Astrea
`fingerprint` already captured for this branch (`7fcaf904e…`, feat/escha-w2 @
`30ec1cb8d`, rope `interleaved_legacy` — note: cross-arm comparison needs the
PR arm fingerprinted too; Astrea rule 5).

**Blockers, stated upfront**: (a) disk — 22 GB free on `/`, 27 GB on `/data`;
  one 27B `.hfq` is ~11 GB, so build ONE artifact at a time and delete; (b)
  the PR branch must build on this box (101 commits over a newer master —
  expect conflicts in `prefill.rs`/`forward.rs`/`gemv.rs`, which both sides
  touched); (c) no bf16 parent on the box, so "does Escha 2-bit deliver" stays
  open — the experiment prices *repack*, not the codec (their §10.2, same
  limitation).

**Success criterion**: a table — arms A/B/C/D × (KLD vs B, PPL, decode tok/s,
resident GB) — that tells the owner exactly what each repack step costs in
quality and buys in speed. That table is the promotion evidence for §5.

---

## 4. Revised execution order (throughput-first)

DROPPED from the old plan: `lm_head` int8 (refuted — not DRAM-bound),
rotate+gemv+finalize fusion (refuted — 2% of projection), hipGraph AR capture
(demoted — launch cost small vs gemv time). These stays dropped; the PR
independently confirms launch-overhead is not the limiter (their H128 phase
analysis: 160 launches/token = 0.38 ms).

| # | work | why | beat condition |
|---|---|---|---|
| 1 | **Quality table (§3)** | sets the rules of engagement: which arms are quality-clean before perf work | arms B/C within their published PPL bands on this box |
| 2 | **Port `nt_major` tile transpose at load** (their `bb77ff87d`) | +24% decode GEMV, load-time only, gated on KLD = 0 | our 5.0 → ~6.2 tok/s; zero KLD movement |
| 3 | **WMMA/MMA coded decode** (their `escham_code_gemm` analogue for dense) | the 3× gap (157 G vs ~515 G weights/s); scalar is done | our decode → 12+ tok/s (match), → 15+ (beat) |
| 4 | **In-trunk MTP resolver** (their `e1d2d55`) or safetensors-dir loader teaching | only tg lever that doesn't need faster kernels; ref reports 1.77–1.92× | MTP engaged: accept ~0.5+, tok/s ×1.5+ |
| 5 | **Grouped prefill GEMM** (their `20707f799` lineage) | prefill 29 → 100+ tok/s; GDN half already batched here | match their 119 tok/s on 27B |
| 6 | **Fold+MQ6 shipping artifact** (`HIPFIRE_ESCHA_FOLD=mq6`) | escha quality at MQ6 speed; hedges the native-decode bet | PPL ≤ 9.70 at ≥ 14 tok/s (beats their default on both axes) |

Cumulative beat condition (27B, gfx1151): **decode ≥ 13 tok/s AND PPL ≤ 9.68**
(beats their `.escha` 12.1 tok/s / 9.6753), or **decode ≥ 16 tok/s at PPL ≤
10.07** (beats their position against `mq3` — faster than 15.7 with better
quality than 10.0643). Prefill target: ≥ 120 tok/s (match 119).

**What NOT to do**: reimplement their converter, H128 kernels, KLD harness, or
codec reference; re-litigate the refuted items; pursue Path C custom drafts
(dead); touch PFlash (legacy).

---

## 5. Promotion / merge posture

The PR is `meta:wip`, OPEN, 101 commits, mergeable. Options, in preference
order — owner's call:

1. **Rebase onto the PR runtime** (recommended): land PR #694's runtime
   (`escha.rs`, `pipeline/escha.rs`, kernels, converter, KLD harness) as the
   serving path; keep this branch's safetensors-direct loader + oracles as the
   independent verification arm and the batched-FA infra as shared prefill
   machinery. Our oracles (`check_escha_dense_batched`, `check_gdn_batched`,
   `check_escha_fa_batched`) become G7/G8/G9 alongside their G1–G6.
2. **Fold-arm ship**: if native decode stalls, ship `HIPFIRE_ESCHA_FOLD=mq6`
   artifacts through the existing MQ6 kernel fleet — no new kernels, quality
   priced by §3.
3. **Coexist**: both branches serve different inputs (safetensors-direct for
   research/fresh checkpoints, `.hfq` for production). Highest maintenance
   cost; only if (1) proves infeasible.

Registry tags stay inert until merge + daily `v1.json` republish (their Status
section) — no registry edits from this branch.

---

## 6. Open questions for the owner

1. Disk: can we free ~40 GB (one `.hfq` build + KLD scratch at a time), or
   should the quality table be measured elsewhere?
2. Merge posture: rebase-onto-PR (option 1) vs fold-arm ship (option 2)?
3. Is beating their `.escha` default (12.1 tok/s / 9.6753) the right bar, or
   should the bar be their `-xt` (12.3 tok/s, PPL 9.7242 — faster, lower
   quality)?
4. `HIPFIRE_ESCHA_FOLD=f16` attribution build (~50 GB — doesn't fit today's
   disk) to split "escha's codec" from "my re-quantisation": worth it, or is
   the mq6-fold delta sufficient?

*Drafted 2026-09-07. No code changed under this plan yet. All PR numbers are
their measured figures as of `nw_escha_w2` @ `33ba046e0`; re-verify on this
box before citing as current.*

---

## 7. Measured on this box (2026-09-07, gfx1151, feat/escha-w2 @ `30ec1cb8d`)

Coexist posture confirmed by owner: keep the branches separate; decide merge
posture after the numbers. 107 GB freed on `/`, no downloads possible (slow
link) — all artifacts converted locally from `/data/rocmfpx/Qwen3.8-27B-Escha-W2`.

### Artifacts (all local, `/home/mika/escha-hfq/`)

| file | size | recipe |
|---|---|---|
| `qwen3.8-27b.escha.hfq` (Arm B) | 11.16 GB | verbatim codes + bit-exact Q8 dense (converter default) |
| `qwen3.8-27b.dense-mq6.hfq` (Arm D) | 10.77 GB | Arm B + dense int8 → MQ6 (router/shared-expert stay Q8) |
| `qwen3.8-27b.fold-mq6-nobias.hfq` (Arm C) | 22.63 GB | fold H128+diagonals into dense + MQ6, biases DROPPED (converter refuses otherwise — see finding F2) |

### Quality (teacher-forced KLD vs weight-exact F16 reference, wikitext2 slice, n_ctx 384, 6 chunks = 1146 tokens, f32 KV, per-token scoring; oracle PPL 10.9607)

| arm | mean KLD | p99 KLD | PPL | verdict |
|---|---|---|---|---|
| B verbatim | 0.0000000 (self-control; CI 0) | 0.000 | 10.9607 | reference arm |
| D dense-mq6 | 0.000568 (CI 0.00054–0.00060) | 0.0023 | 10.9677 (+0.007) | **quality-clean** (Astrea `metrics`: `no_quality_gain` = no damage detected, not a promotion claim) |
| C fold-mq6-nobias | 0.148332 (CI 0.135–0.171) | 2.62 | 12.9947 (+2.03) | **REJECTED as built** — but confounded (see F2) |

Astrea artifacts: `kld-27b/result-data.json` (reduce), `kld-27b/astrea-armD.json`
(metrics + engine fingerprint `208d73d2…`, feat/escha-w2 @ `30ec1cb8d`).

### Findings

- **F1: dense second-quantisation is nearly free.** Arm D (MQ6 over the
  non-coded int8s) costs KLD 0.0006 / PPL +0.007 for −0.86 GB and the MQ6
  kernel fleet. France→Paris at temp 0. The PR's MoE-side KLD table said the
  same (everything 0.009785); now confirmed on dense 27B, an order of magnitude
  cleaner. Ship Arm D's recipe as the default dense build.
- **F2: the fold arm's number is confounded by dropped biases, not the fold.**
  The converter refuses fold-with-bias (`HIPFIRE_ESCHA_FOLD_DROP_BIAS=1`
  required); the 27B carries 400 bias tensors the base architecture lacks.
  KLD 0.148 / PPL +2.03 measures fold+MQ6+bias-drop jointly. Do NOT cite
  0.148 as "the fold costs 0.148" — the bias-drop isolation (fold with biases
  applied at runtime, or per-projection bias-ablation KLD) is still open.
  Coherence spot-check passes (Paris), so the damage is distributional, not
  collapse — consistent with 400 small additive corrections going missing.
- **F3: Arm B self-control is 0.0000000.** The verbatim `.hfq` served through
  the PR runtime reproduces the F16 reference bit-exactly at the KLD level —
  the repack path is verified clean end-to-end (their G1 `memcmp` + our KLD
  control agree).
- **F4: their dense linear gate FAILS on our conversion — pre-existing,
  MoE-side tooling, not our artifact.** `test_escha_dense_linear_gpu_vs_cpu`
  reports rel_rms 1.40 on `mlp.gate_proj` AND `in_proj_qkv`, but F16 store
  matches to 1e-4 (|y| 55.30 vs 56.06 — the H128 pair is right) while Native
  matches the reference EXACTLY (|y| 56.058 = |y_ref| 56.059). The gate
  compares the wrong pair (message says "F16 store" while printing Native's
  1.40). G2 (GPU decode bit-exact, 89M elements) PASSES. No action — noted so
  nobody re-investigates.
- **F5: MTP engages on every arm** (in-trunk resolver, their `e1d2d55`):
  Arm B tau=1.14, Arm C tau=1.42, Arm D tau=1.74 @ 7.0 tok/s. Our branch's MTP
  format gap is now the only thing standing between us and ×1.5+ on decode.

### Perf (same box, `escha_prefill_bench`, PR runtime)

| arm | prefill n=58 | prefill n=512 | decode (5-token tail) |
|---|---|---|---|
| B verbatim | 92.7 tok/s | 122.8 tok/s | 10.3–10.9 tok/s |
| D dense-mq6 | 60.4 tok/s | — | 10.6 tok/s |
| C fold-mq6 | 14.9 tok/s (0 H128 launches — plain MQ6 path) | — | 9.2 tok/s |

Note Arm D prefill REGRESSES vs B at n=58 (60 vs 93) — dense-MQ6 takes a
different (unfused?) prefill path; needs profiling before shipping D as
default. Decode is flat across arms (10–11 tok/s): the coded projections
dominate, dense format is second-order — consistent with the refuted
`lm_head`-int8 premise on our branch.

### Beat-condition update

Bar (their `.escha` default): 12.1 tok/s / PPL 9.6753. Our Arm B reproduces
10.3–10.9 tok/s here (vs 12.1 published — box/method delta, re-verify before
citing). Arm D is quality-clean at 10.6. Path to beat: nt_major (+24% GEMV →
~13 tok/s on the GEMV-bound portion) + MTP ×1.5 (tau 1.7 already measured on
Arm D) → ~15+ tok/s effective at PPL 10.97. That beats default on speed while
the quality story is "KLD 0.0006 vs weight-exact" rather than PPL parity —
the PPL gap (10.97 vs 9.68) is the slice/method difference (their PPL is on a
different corpus slice), so the honest claim is KLD-based, not PPL-based.
