# 2026-09-04 — Escha-W2 MoE (Qwen3.6-35B-A3B-Escha-W2) coherent on feat/escha-w2

**Lifecycle:** `historical` — fixture-bound measured evidence for the local
`feat/escha-w2` replay milestone (A5). Not a current default or admission.

## Fixture

- GPU: Strix Halo APU gfx1151, ROCm 10 (TheRock binary SDK, HIP 7.15.26333,
  clang 23). Unified 137 GB (LPDDR5X ~256 GB/s).
- Tree: `/home/mika/git/hipfire` branch `feat/escha-w2` @ b967a721
  (per-expert input-scale fix) + docs aad635aa/143a6ba7.
- Model dir: `/data/rocmfpx/Escha-W2` (Qwen3.6-35B-A3B-Escha-W2, eschamoe,
  arch_id=6). Raw safetensors dir → EschaSource loader → trellis/3INST decode +
  grouped-f16 FFN. Int8 dense attention (per-row f16 scales), escha-coded
  routed experts (256, top-8, mi 512), shared expert, DeltaNet hybrid 30DN/10FA.
- Binaries md5: hipfire `be74e7222dc28905940dd2d5bbd736d8`,
  daemon `d901b59967c77cafc564747c71449cc1`.
- Route: `hipfire run <dir>` AR, `--spec off`, `--temp 0`, reasoning.mode off
  (`hipfire config set reasoning.mode off` — the Qwen chat default opens a
  think block and the daemon fail-closes otherwise).
- Prompts (byte-identical, md5):
  - `Explain how a rainbow forms.` md5 `f2ced4ba3b5de6368182d2858ff9d06d`
  - `The capital of Japan is` md5 `385265ab43d31914d6098f0cb4dbe986`
  - `Write a short haiku about the ocean.` md5 `ce468bb9f3700c1967e079e15eb01e78`
  - `def is_prime(n):` md5 `74aa430d379576b0fc6af3d55d5f6748`

## Results (measured)

| prompt | max tokens | output | tok/s | finish |
|---|---|---|---|---|
| The capital of France is | 20 | "**Paris**." (9 tok) | — | stop |
| The capital of Japan is | 40 | "**Tokyo**." + fluent continuation | 6.6 | length |
| Write a short haiku about the ocean. | 40 | proper 5-7-5 haiku | 4.1 | stop |
| def is_prime(n): | 40 | clean ```python fence | 6.5 | length |
| Explain how a rainbow forms. | 128 | fluent structured 128-token answer | 10.6 | length |

Coherence gate: eyeball-decoded output is fluent, structured, factual; no
attractor/degenerate spans. Control (same tree, same config): HFQ A3B MoE
`qwen3.6-35b-a3b.mq4r` produces token-id-identical "The capital of France is
Paris." (760,6511,314,9338,369,2972,57590,159034,248046).

## Disposition

Milestone evidence for the MoE replay onto the fresh modular tree. Throughput
10.6 tok/s at temp 0 (128 tok) is decode-bandwidth-bound on Strix Halo; the
handoff's known bottlenecks (f32 dense attention dequant ~4.5 GB/token) still
apply. Not a product claim.
