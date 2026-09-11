// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

/// Un-permute the DeltaNet V-head concat from ENGINE order back to GGUF
/// order (GSQ-RCO native out_proj). The released GSQ-RCO GGUF stores
/// V-heads sequentially (head h at position h); the engine consumes
/// interleaved order (engine head e == GGUF head PERM[e]).
///
/// in_proj_qkv/z are interleaved in the .hfq file (a packed 128-row
/// chunk move), but out_proj's 128-col V blocks sit HALF INSIDE each
/// 256-element quant group, so an exact packed interleave would mix
/// per-256-block scales. out_proj is therefore passed through in GGUF
/// order and the engine permutes the ACTIVATION instead:
///     dst[row, PERM[e]*128 + j] = src[row, e*128 + j]
/// dst is then in GGUF-head order and matches the GGUF-ordered W.
extern "C" __launch_bounds__(256)
__global__ void vhead_unpermute_f32(
    const float* __restrict__ src,  // [n * 6144] engine V-head order
    float* __restrict__ dst,        // [n * 6144] GGUF V-head order
    int n
) {
    const int K = 6144;
    // Engine head e == GGUF head PERM[e] (same mapping as the f32
    // maybe_interleave_deltanet_v_heads / packed chunk-move interleave).
    static const int PERM[48] = {
        0, 16, 32, 1, 17, 33, 2, 18, 34, 3, 19, 35, 4, 20, 36, 5, 21, 37, 6, 22, 38, 7, 23, 39,
        8, 24, 40, 9, 25, 41, 10, 26, 42, 11, 27, 43, 12, 28, 44, 13, 29, 45, 14, 30, 46, 15,
        31, 47,
    };
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * K;
    if (idx >= total) return;
    int r = idx / K;
    int c = idx % K;
    int e = c / 128;
    int j = c % 128;
    int g = PERM[e];
    dst[r * K + g * 128 + j] = src[idx];
}