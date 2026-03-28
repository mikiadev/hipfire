# hipfire Benchmarks

Hardware: AMD Radeon RX 5700 XT (8GB VRAM, RDNA1 gfx1010, 448 GB/s peak)
Branch: `web` (TurboQuant KV cache + restore baseline)
Date: 2026-03-27

## TurboQuant KV Cache — Qwen3-8B HFQ4-G256

### Speed (warm kernel cache, greedy sampling)

| Config | Short (91 tok) | Hard (128 tok) | KV Compression | KV bytes/head |
|--------|---------------|----------------|----------------|---------------|
| Q8 KV (baseline) | **59.9 tok/s** | **58.8 tok/s** | 3.88x | 132 |
| FP32 KV | 57.0 | — | 1.0x | 512 |
| turbo2 (2-bit) | 54.1 | 51.8 | **14.2x** | 36 |
| turbo2+adaptive | 54.2 | — | ~10x | 36/512 |
| turbo3 (3-bit) | 50.8 | 44.5 | 9.85x | 52 |
| turbo4 (4-bit) | 53.6 | 51.0 | 7.5x | 68 |
| turbo4+adaptive | 53.7 | 51.2 | ~5x | 68/512 |

Short = "Hello" with ChatML wrapping (9 prompt tokens, 91 generated).
Hard = "Explain the three laws of thermodynamics with mathematical formulations and real-world examples" (128 generated).
Adaptive = first and last layers use FP32 KV, middle layers use turbo.

### Quality verification

All turbo configs produce **coherent, topical output** on the hard prompt (thermodynamics explanation with correct concepts, mathematical references, multi-paragraph reasoning). No repetition loops, no garbage tokens.

Sample turbo4 output (hard prompt):
> The Zeroth Law: If two systems are each in thermal equilibrium with a third system, then they are in thermal equilibrium with each other. The first law is about energy conservation: ΔU = Q - W...

### KV cache memory at 2048 context

| Config | Qwen3-8B (32 kv_heads, 128 hd, 36 layers) |
|--------|---------------------------------------------|
| FP32 | 1.07 GB |
| Q8 | 536 MB |
| turbo4 | 278 MB |
| turbo3 | 213 MB |
| turbo2 | **147 MB** |

### Key findings

1. **TurboQuant works on RDNA1.** All three bit-widths (2/3/4) produce coherent output with FWHT + norm correction + Lloyd-Max centroids.

2. **FWHT is irreducible.** Tested 5 cheaper transforms (sign flip, permutation, sign+permute, 2-stage butterfly, none). All produce garbage at 2-bit and degrade on hard prompts at 4-bit. The full 7-stride Walsh-Hadamard butterfly is necessary for quality.

3. **Norm correction is critical.** Storing `original_norm / reconstruction_norm` per head preserves exact L2 norm through quantization. This is what makes turbo beat Q8 in quality for recurrent architectures (DeltaNet).

4. **Speed overhead is 10% vs Q8.** The FWHT costs ~5 tok/s at short context. At longer context where KV bandwidth dominates, turbo should match or beat Q8.

5. **turbo2 (2-bit) is the compression champion.** 14.2x compression vs fp32, only 10% slower than Q8, coherent output on hard prompts. Enables 3.65x more context in the same VRAM.

## Qwen3-0.6B HFQ4 (weight quantization benchmarks)

| Config | Short gen | Long gen | Prefill |
|--------|-----------|----------|---------|
| HFQ4-G256 + Q8 KV | 262.5 tok/s | 235.0 | 1263 |
| HFQ4 auto + Q8 KV | 238.3 | — | 1370 |
| Q8 weights + Q8 KV | 219.0 | 184.2 | 358 |

## Notes

- All benchmarks use greedy decoding (temp=0) with ChatML auto-detection.
- Kernel cache warm (cold compile excluded from measurements).
- turbo KV requires head_dim=128 (Qwen3-8B, Llama 8B). Qwen3-0.6B (head_dim=64) uses Q8 KV.
- Repetition penalty: 1.1, window: 64 tokens.
