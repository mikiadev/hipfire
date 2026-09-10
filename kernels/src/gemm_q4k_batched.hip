// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched Q4_K GEMM: Y[batch, M] = X[batch, K] @ W[M, K]^T where W is Q4_K
// quantized (144 bytes per group of 256 elements: d/dmin fp16 + 12 scale
// bytes + 128 packed nibbles).
//
// GSQ-RCO native path: the release's lm_head + 38 projections stay Q4_K on
// device. Decode serves them through the scalar `gemv_q4k` (one row per
// block); prefill needs the batched equivalent or every LA/FA/FFN matcher
// falls through to an HFQ4-stride fused kernel and reads garbage.
//
// Structure mirrors `gemm_q8_0_batched.hip`: one block per output row, one
// wave per row, per-batch register accumulators, warp-shuffle reduce per
// batch slice. FP-reduction order matches `gemv_q4k` group-by-group (one
// accumulator FMA loop over the 8 sub-blocks) so the per-batch slice equals
// what the per-row GEMV would produce — preserves greedy parity.
//
// Grid:  [M, 1, 1]    — one block per output row.
// Block: [32, 1, 1]   — one wave per row.

#define Q4K_MAX_BATCH 64

__device__ __forceinline__ float q4k_half_to_float(unsigned short h) {
    unsigned int sign = (h >> 15) & 1;
    unsigned int exp = (h >> 10) & 0x1F;
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
extern "C" __global__ void gemm_q4k_batched(
    const unsigned char* __restrict__ A_q4k,  // [M, K/256 * 144] packed Q4_K
    const float* __restrict__ X,               // [batch_size, K] row-major
    float* __restrict__ Y,                     // [batch_size, M] row-major
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 256;
    const unsigned char* row_data = A_q4k + (size_t)row * blocks_per_row * 144;

    float sums[Q4K_MAX_BATCH];
    #pragma unroll
    for (int b = 0; b < Q4K_MAX_BATCH; b++) sums[b] = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 144;
        float d = q4k_half_to_float((unsigned short)(block[0] | (block[1] << 8)));
        float dmin = q4k_half_to_float((unsigned short)(block[2] | (block[3] << 8)));
        const unsigned char* sc = block + 4;

        // Group 0: elements 0-63 (matches gemv_q4k group 0, same FMA order).
        {
            unsigned char qbyte = block[16 + tid];
            float s0 = d * (float)(sc[0] & 63);
            float m0 = dmin * (float)(sc[4] & 63);
            float w_lo = s0 * (float)(qbyte & 0xF) - m0;
            float s1 = d * (float)(sc[1] & 63);
            float m1 = dmin * (float)(sc[5] & 63);
            float w_hi = s1 * (float)(qbyte >> 4) - m1;
            #pragma unroll
            for (int b = 0; b < Q4K_MAX_BATCH; b++) {
                if (b < batch_size) {
                    const float* xrow = X + (size_t)b * K;
                    sums[b] += w_lo * xrow[bi * 256 + tid]
                             + w_hi * xrow[bi * 256 + 32 + tid];
                }
            }
        }

        // Group 1: elements 64-127.
        {
            unsigned char qbyte = block[48 + tid];
            float s2 = d * (float)(sc[2] & 63);
            float m2 = dmin * (float)(sc[6] & 63);
            float w_lo = s2 * (float)(qbyte & 0xF) - m2;
            float s3 = d * (float)(sc[3] & 63);
            float m3 = dmin * (float)(sc[7] & 63);
            float w_hi = s3 * (float)(qbyte >> 4) - m3;
            #pragma unroll
            for (int b = 0; b < Q4K_MAX_BATCH; b++) {
                if (b < batch_size) {
                    const float* xrow = X + (size_t)b * K;
                    sums[b] += w_lo * xrow[bi * 256 + 64 + tid]
                             + w_hi * xrow[bi * 256 + 96 + tid];
                }
            }
        }

        // Group 2: elements 128-191.
        {
            unsigned char qbyte = block[80 + tid];
            float s4 = d * (float)((sc[8] & 0xF) | ((sc[0] >> 6) << 4));
            float m4 = dmin * (float)((sc[8] >> 4) | ((sc[4] >> 6) << 4));
            float w_lo = s4 * (float)(qbyte & 0xF) - m4;
            float s5 = d * (float)((sc[9] & 0xF) | ((sc[1] >> 6) << 4));
            float m5 = dmin * (float)((sc[9] >> 4) | ((sc[5] >> 6) << 4));
            float w_hi = s5 * (float)(qbyte >> 4) - m5;
            #pragma unroll
            for (int b = 0; b < Q4K_MAX_BATCH; b++) {
                if (b < batch_size) {
                    const float* xrow = X + (size_t)b * K;
                    sums[b] += w_lo * xrow[bi * 256 + 128 + tid]
                             + w_hi * xrow[bi * 256 + 160 + tid];
                }
            }
        }

        // Group 3: elements 192-255.
        {
            unsigned char qbyte = block[112 + tid];
            float s6 = d * (float)((sc[10] & 0xF) | ((sc[2] >> 6) << 4));
            float m6 = dmin * (float)((sc[10] >> 4) | ((sc[6] >> 6) << 4));
            float w_lo = s6 * (float)(qbyte & 0xF) - m6;
            float s7 = d * (float)((sc[11] & 0xF) | ((sc[3] >> 6) << 4));
            float m7 = dmin * (float)((sc[11] >> 4) | ((sc[7] >> 6) << 4));
            float w_hi = s7 * (float)(qbyte >> 4) - m7;
            #pragma unroll
            for (int b = 0; b < Q4K_MAX_BATCH; b++) {
                if (b < batch_size) {
                    const float* xrow = X + (size_t)b * K;
                    sums[b] += w_lo * xrow[bi * 256 + 192 + tid]
                             + w_hi * xrow[bi * 256 + 224 + tid];
                }
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < Q4K_MAX_BATCH; b++) {
        if (b < batch_size) {
            float s = sums[b];
            for (int offset = 16; offset > 0; offset >>= 1)
                s += __shfl_down(s, offset);
            if (tid == 0) Y[(size_t)b * M + row] = s;
        }
    }
}
