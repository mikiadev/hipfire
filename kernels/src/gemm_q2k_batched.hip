// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched Q2_K GEMM: Y[batch, M] = X[batch, K] @ W[M, K]^T where W is Q2_K
// quantized (84 bytes per group of 256 elements).
//
// GSQ-RCO native path: the release's Q2_K projections (attn_q, attn_gate,
// attn_qkv, ffn_down/gate — 2.7% of params) stay Q2_K on device. Decode
// serves them through the scalar `gemv_q2k`; prefill needs the batched
// equivalent or every LA/FA/FFN matcher falls through to an HFQ4-stride
// fused kernel and reads garbage.
//
// Structure mirrors `gemm_q4k_batched.hip`: one block per output row, one
// wave per row, per-batch register accumulators, warp-shuffle reduce per
// batch slice. FP-reduction order matches `gemv_q2k` group-by-group so the
// per-batch slice equals what the per-row GEMV would produce.
//
// Grid:  [M, 1, 1]    — one block per output row.
// Block: [32, 1, 1]   — one wave per row.

#define Q2K_MAX_BATCH 64

__device__ __forceinline__ float q2k_half_to_float(unsigned short h) {
    unsigned int sign = (h >> 15) & 1;
    unsigned int exp  = (h >> 10) & 0x1F;
    unsigned int frac = h & 0x3FF;
    if (exp == 0) {
        if (frac == 0) return __int_as_float(sign << 31);
        while (!(frac & 0x400)) { frac <<= 1; exp--; }
        frac &= 0x3FF;
        exp = 127 - 15 + 1 + exp;
        return __int_as_float((sign << 31) | (exp << 23) | (frac << 13));
    }
    if (exp == 31) return __int_as_float((sign << 31) | (0xFF << 23) | (frac ? (frac << 13 | 1) : 0));
    return __int_as_float((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13));
}

__launch_bounds__(32, 20)
extern "C" __global__ void gemm_q2k_batched(
    const unsigned char* __restrict__ A_q2k,  // [M, K/256 * 84] packed Q2_K
    const float* __restrict__ X,               // [batch_size, K] row-major
    float* __restrict__ Y,                     // [batch_size, M] row-major
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 256;
    const unsigned char* row_data = A_q2k + (size_t)row * blocks_per_row * 84;

    float sums[Q2K_MAX_BATCH];
    #pragma unroll
    for (int b = 0; b < Q2K_MAX_BATCH; b++) sums[b] = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 84;
        const unsigned char* scales = block;      // [16]
        const unsigned char* qs = block + 16;     // [64]
        float d = q2k_half_to_float((unsigned short)(block[80] | (block[81] << 8)));
        float dmin = q2k_half_to_float((unsigned short)(block[82] | (block[83] << 8)));

        // 8 groups of 32 elements. Thread tid covers element tid of each
        // group: qs byte (g/4)*32 + tid, scale 2g + (tid>=16), shift 2*(g%4).
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            int sc = scales[2 * g + (tid >= 16 ? 1 : 0)];
            float dl = d * (float)(sc & 0xF);
            float ml = dmin * (float)(sc >> 4);
            unsigned char qq = qs[(g / 4) * 32 + tid];
            int v = (qq >> (2 * (g % 4))) & 3;
            float w = dl * (float)v - ml;
            #pragma unroll
            for (int b = 0; b < Q2K_MAX_BATCH; b++) {
                if (b < batch_size) {
                    const float* xrow = X + (size_t)b * K;
                    sums[b] += w * xrow[bi * 256 + g * 32 + tid];
                }
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < Q2K_MAX_BATCH; b++) {
        if (b < batch_size) {
            float s = sums[b];
            for (int offset = 16; offset > 0; offset >>= 1)
                s += __shfl_down(s, offset);
            if (tid == 0) Y[(size_t)b * M + row] = s;
        }
    }
}