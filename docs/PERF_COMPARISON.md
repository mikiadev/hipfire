# hipfire vs llama.cpp Performance Comparison

**GPU:** AMD RX 5700 XT (gfx1010, RDNA1, 8GB GDDR6, 448 GB/s peak)
**Date:** 2026-03-22
**hipfire:** phase5-hfq4 branch, HFQ4-G256 weights + Q8_0 KV cache
**llama.cpp:** build 7f8ef50cc (7209), custom ROCm build, Q8_0/Q4_K_M weights

## Results

| Benchmark | hipfire | llama.cpp | Ratio | Winner |
|-----------|---------|-----------|-------|--------|
| Qwen3-8B short gen (tok/s) | **59.3** | 44.3 | **1.34x** | hipfire |
| Qwen3-8B long gen (tok/s) | **52.7** | 42.8 | **1.23x** | hipfire |
| Qwen3-8B prefill (tok/s) | 108 | **189.2** | 0.57x | llama.cpp |
| Qwen3-0.6B short gen (tok/s) | **256.3** | 193.6 | **1.32x** | hipfire |
| Qwen3-0.6B prefill (tok/s) | 1053 | **1534** | 0.69x | llama.cpp |

hipfire wins all generation benchmarks. llama.cpp wins prefill.

## Why hipfire is faster at generation

**HFQ4-G256 occupancy advantage.** hipfire's HFQ4 weight format uses 18 VGPRs per GEMV thread vs Q4_K's 39. On RDNA1 (1024 VGPRs per SIMD), this means 20 concurrent waves vs 10 — 2x the occupancy. Higher occupancy hides memory latency better, translating to higher effective bandwidth utilization.

**Q8_0 KV cache.** KV cache quantized to Q8_0 format (136 bytes/head vs 512 FP32). Reduces attention bandwidth by 3.76x. Biggest impact on long generation: +39% tok/s at 1000+ tokens.

## Why llama.cpp is faster at prefill

llama.cpp uses **rocBLAS GEMM** for batched prompt processing — a heavily tuned matrix multiply library. hipfire uses hand-written batched GEMM kernels that achieve 1.4-2.1x throughput scaling at batch=20, while rocBLAS achieves near-theoretical bandwidth at any batch size.

hipfire's prefill improved from 56 to 108 tok/s through batched GEMM projections, batched RoPE, batched KV cache writes, and batched causal attention. The remaining gap is in the GEMM kernel efficiency.

## Key optimizations shipped

| Optimization | Impact | Description |
|-------------|--------|-------------|
| Q8_0 KV cache | +7.6% short, +39% long | KV values quantized to int8 with co-located f16 scale |
| Batched causal attention | +14% prefill | All query positions processed in one kernel launch |
| Batched GEMM projections | +59% prefill | Weight data loaded once for all prompt tokens |
| Batched RoPE | +12% prefill (0.6B) | All positions rotated in one kernel |
| Batched KV cache write | +32% prefill (0.6B) | All positions quantized and written in one launch |
| Wide GEMV (2 rows/block) | +1.3% gen | Better wave scheduling for large weight matrices |
| HFQ-native tokenizer | correctness | BPE merges from embedded tokenizer.json, no GGUF dependency |
| Repetition penalty | quality | GPU-side repeat penalty in sampling kernel |
| SIGINT handler | reliability | Graceful GPU cleanup on Ctrl-C, no zombie processes |

## Hardware context

RDNA1 (2019) — wave32, no matrix cores, no dp4a. The occupancy advantage of HFQ4 is most pronounced on this architecture where VGPR pressure directly limits wave count. On RDNA3/4 with more VGPRs and matrix cores, the relative advantage would narrow.
