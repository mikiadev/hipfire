// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Q2_K GEMV (GSQ-RCO native): Y[M] = A_q2k[M, K] @ x[K] where A is Q2_K
// quantized (84 bytes per group of 256 elements).
//
// Block layout (ggml-common.h block_q2_K — scales FIRST):
//   [0..16)  scales[16]
//   [16..80) qs[64]
//   [80..82) fp16 d
//   [82..84) fp16 dmin
// Two 128-element halves, each decoding 4 groups of 32 with 2-bit codes:
//   dl = d * (scales & 0xF), ml = dmin * (scales >> 4), value = dl*v - ml.
// Group g (0..7): half = g/4, shift = 2*(g%4); element e of the group reads
// qs[half*32 + e], scale scales[2g + (e>=16)].
//
// 32 threads (single warp), one element per thread per 32-group, warp
// shuffle reduce. FP-reduction order matches the CPU decoder
// (gguf_iq.rs::dequant_q2_k) group-by-group for greedy parity.
//
// Grid:  [M, 1, 1]    — one block per output row.
// Block: [32, 1, 1]   — one wave per row.

__device__ __forceinline__ float half_to_float(unsigned short h) {
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
extern "C" __global__ void gemv_q2k(
    const unsigned char* __restrict__ A_q2k,  // [M, K/256 * 84] packed Q2_K
    const float* __restrict__ x,               // [K]
    float* __restrict__ y,                     // [M]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 256;
    const unsigned char* row_data = A_q2k + (size_t)row * blocks_per_row * 84;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 84;
        const unsigned char* scales = block;      // [16]
        const unsigned char* qs = block + 16;     // [64]
        float d = half_to_float((unsigned short)(block[80] | (block[81] << 8)));
        float dmin = half_to_float((unsigned short)(block[82] | (block[83] << 8)));
        const float* xb = x + bi * 256;

        // 8 groups of 32 elements. Thread tid covers element tid of each
        // group: qs byte (g/4)*32 + tid, scale 2g + (tid>=16), shift 2*(g%4).
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            int sc = scales[2 * g + (tid >= 16 ? 1 : 0)];
            float dl = d * (float)(sc & 0xF);
            float ml = dmin * (float)(sc >> 4);
            unsigned char qq = qs[(g / 4) * 32 + tid];
            int v = (qq >> (2 * (g % 4))) & 3;
            sum += (dl * (float)v - ml) * xb[g * 32 + tid];
        }
    }

    // Warp shuffle reduction (5 steps, no shared memory)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}