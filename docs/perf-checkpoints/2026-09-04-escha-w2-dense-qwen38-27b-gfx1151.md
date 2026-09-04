# 2026-09-04 — Escha-W2 dense (Qwen3.8-27B-Escha-W2) coherent on feat/escha-w2

**Lifecycle:** `historical` — fixture-bound measured evidence for the local
`feat/escha-w2` dense-port milestone (B4/B5). Not a current default or admission.

## Fixture

- GPU: Strix Halo APU gfx1151, ROCm 10 (TheRock SDK, HIP 7.15.26333). Unified
  137 GB (~256 GB/s LPDDR5X).
- Tree: `/home/mika/git/hipfire` branch `feat/escha-w2` @ 48059e2de (dense
  AR-graph exclusion 5bb373af + B4 doc).
- Model dir: `/data/rocmfpx/Qwen3.8-27B-Escha-W2` (Qwen3.8 27B dense,
  `escha`, arch 5; 64 layers: 48 linear-attention + 16 full-attention; hidden
  5120; 24 heads / 4 kv / head_dim 256; linear 16kH/48vH/128hd conv4; vocab
  248320; FFN 17408; gate K2 / up K3 / down K3 / attn K2). Loaded by
  DenseEschaSource; all linear projections stay escha-coded and decode in
  HIP kernels (rotate-in T128 → in-kernel decode-gemm → finalize). Int8
  embed/lm_head (per-row f16 scale).
- Binaries md5: hipfire `93d00955c1db62f128cd5bf4bbdc1ba6`,
  daemon `a4742c52e878fe03fbf42779b9a353da`.
- Route: `hipfire run <dir>`, AR, `--spec off`, `--temp 0`, reasoning off
  (`hipfire config set reasoning.mode off`).
- Prompts (byte-identical; md5):
  - `The capital of France is` → `1aacd3c05cf9695cc799acc59581938d`
  - `Hello! Please introduce yourself in one or two sentences.` → `1869b9b6cc4040d2f02743b250ed2fb7`
  - `The sky is blue because` → `0a9d0d2cce06723b5e35c079cf7d915b`
  - `Write a haiku about a river.` → `6a41c4ff8a943ae798e900cad1a60a02`

## Results (measured)

| prompt | max tok | output (first tokens shown) | tok/s | finish |
|---|---|---|---|---|
| The capital of France is | 30 | "Paris" | 0.4 | stop |
| Hello! … introduce yourself… | 40 | "Hello! I am an AI assistant designed to assist you with a wide variety of tasks, from answering questions to creative writing and problem-solving. How can I help you today?" (36 tok) | 2.1 | stop |
| The sky is blue because | 50 | "…Rayleigh scattering… Sunlight is composed of electromagnetic waves…" | 2.5 | length |
| Write a haiku about a river. | 50 | "Current flows past, / carrying the weight of stone, / into the endless sea." (19 tok) | 1.6 | stop |

Coherence gate: eyeball-decoded output is fluent/structured with no
attractor/decay across ≥40 tokens. Deterministic at temp 0 (byte-identical
across 2 runs).

## Root cause fixed (multi-token decay)

AR hipGraph capture/replay engaged on the escha-dense model (`use_graph`
excluded `is_escham_moe` but not `is_escha_dense`); the per-projection decode
allocs pool scratch per call (capture-unsafe) → replay diverged at decode
token ~3. Fix: `&& !config.is_escha_dense` in the `use_graph` predicate
(5bb373af). Direct-only until the decode scratch is hoisted to Qwen35Scratch.

Controls on the same tree: MoE Escha-W2 → "**Paris.**" unchanged (not
regressed); HFQ dense qwen3.6-27b.mq4r (same 48-v-head shape) coherent at
13.3 tok/s on both graph settings.

## Disposition

Milestone evidence for the dense in-kernel-decode port. ~1.6–2.5 tok/s decode
is the unoptimized per-token decode-gemm (M3 prefill/throughput not started);
expected to rise substantially with batched/WMMA prefill and scratch hoist.
Not a product claim.
