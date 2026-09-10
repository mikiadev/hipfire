// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched IQ4_XS GEMM: Y[batch, M] = X[batch, K] @ W[M, K]^T where W is
// IQ4_XS quantized (136 bytes per group of 256 elements).
//
// GSQ-RCO native path: the release's IQ4_XS projections (attn_qkv, attn_q,
// attn_k, attn_gate, ssm_out, ffn_up/down — 21% of params) stay IQ4_XS on
// device. Decode serves them through the scalar `gemv_iq4_xs`; prefill
// needs the batched equivalent or every LA/FA/FFN matcher falls through to
// an HFQ4-stride fused kernel and reads garbage.
//
// Structure mirrors `gemm_q4k_batched.hip`: one block per output row, one
// wave per row, per-batch register accumulators, warp-shuffle reduce per
// batch slice. FP-reduction order matches `gemv_iq4_xs` group-by-group so
// the per-batch slice equals what the per-row GEMV would produce.
//
// Grid:  [M, 1, 1]    — one block per output row.
// Block: [32, 1, 1]   — one wave per row.

#define IQ4XS_MAX_BATCH 64

__device__ __forceinline__ float iq4xs_half_to_float(unsigned short h) {
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

__device__ __forceinline__ float iq4xs_value(int nibble) {
    const float KV[16] = { -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f,
                           -22.0f, -10.0f, 1.0f, 13.0f, 25.0f, 38.0f,
                           53.0f, 69.0f, 89.0f, 113.0f };
    return KV[nibble & 0xF];
}

__launch_bounds__(32, 20)
extern "C" __global__ void gemm_iq4_xs_batched(
    const unsigned char* __restrict__ A_iq4xs, // [M, K/256 * 136] packed IQ4_XS
    const float* __restrict__ X,               // [batch_size, K] row-major
    float* __restrict__ Y,                     // [batch_size, M] row-major
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 256;
    const unsigned char* row_data = A_iq4xs + (size_t)row * blocks_per_row * 136;

    float sums[IQ4XS_MAX_BATCH];
    #pragma unroll
    for (int b = 0; b < IQ4XS_MAX_BATCH; b++) sums[b] = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 136;
        float d = iq4xs_half_to_float((unsigned short)(block[0] | (block[1] << 8)));
        unsigned short scales_h = (unsigned short)(block[2] | (block[3] << 8));
        const unsigned char* scales_l = block + 4;
        const unsigned char* qs = block + 8;

        // 8 groups of 32 elements. Thread tid covers element tid of each
        // group: qs byte ib*16 + (tid & 15), low nibble for tid<16 else high.
        #pragma unroll
        for (int ib = 0; ib < 8; ib++) {
            int ls = (int)((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                   | (int)(((scales_h >> (2 * ib)) & 3) << 4);
            float dl = d * (float)(ls - 32);
            unsigned char qb = qs[ib * 16 + (tid & 15)];
            int kv = (tid >= 16) ? (qb >> 4) : (qb & 0xF);
            float w = dl * iq4xs_value(kv);
            #pragma unroll
            for (int b = 0; b < IQ4XS_MAX_BATCH; b++) {
                if (b < batch_size) {
                    const float* xrow = X + (size_t)b * K;
                    sums[b] += w * xrow[bi * 256 + ib * 32 + tid];
                }
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < IQ4XS_MAX_BATCH; b++) {
        if (b < batch_size) {
            float s = sums[b];
            for (int offset = 16; offset > 0; offset >>= 1)
                s += __shfl_down(s, offset);
            if (tid == 0) Y[(size_t)b * M + row] = s;
        }
    }
}