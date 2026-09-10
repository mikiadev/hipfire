// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// IQ4_XS GEMV (GSQ-RCO native): Y[M] = A_iq4xs[M, K] @ x[K] where A is
// IQ4_XS quantized (136 bytes per group of 256 elements).
//
// Block layout (ggml-common.h block_iq4_xs):
//   [0..2)  fp16 d
//   [2..4)  scales_h u16
//   [4..8)  scales_l[4]
//   [8..136) qs[128] — nibbles into kvalues_iq4nl
// Each 32-element group ib: ls = (scales_l[ib/2] >> 4*(ib%2) & 0xf)
//   | ((scales_h >> 2*ib) & 3) << 4; dl = d * (ls - 32);
//   value = dl * kvalues_iq4nl[qs nibble].
//
// 32 threads (single warp), one element per thread per 32-group, warp
// shuffle reduce. FP-reduction order matches the CPU decoder
// (gguf_iq.rs::dequant_iq4_xs) group-by-group for greedy parity.
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

__device__ __forceinline__ float iq4nl_value(int nibble) {
    // kvalues_iq4nl from ggml-common.h / gguf_iq.rs.
    const float KV[16] = { -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f,
                           -22.0f, -10.0f, 1.0f, 13.0f, 25.0f, 38.0f,
                           53.0f, 69.0f, 89.0f, 113.0f };
    return KV[nibble & 0xF];
}

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_iq4_xs(
    const unsigned char* __restrict__ A_iq4xs,  // [M, K/256 * 136] packed IQ4_XS
    const float* __restrict__ x,                 // [K]
    float* __restrict__ y,                       // [M]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 256;
    const unsigned char* row_data = A_iq4xs + (size_t)row * blocks_per_row * 136;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 136;
        float d = half_to_float((unsigned short)(block[0] | (block[1] << 8)));
        unsigned short scales_h = (unsigned short)(block[2] | (block[3] << 8));
        const unsigned char* scales_l = block + 4;
        const unsigned char* qs = block + 8;
        const float* xb = x + bi * 256;

        // 8 groups of 32 elements. Thread tid covers element tid of each
        // group: qs byte ib*16 + (tid & 15), low nibble for tid<16 else high.
        #pragma unroll
        for (int ib = 0; ib < 8; ib++) {
            int ls = (int)((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                   | (int)(((scales_h >> (2 * ib)) & 3) << 4);
            float dl = d * (float)(ls - 32);
            unsigned char qb = qs[ib * 16 + (tid & 15)];
            int kv = (tid >= 16) ? (qb >> 4) : (qb & 0xF);
            sum += dl * iq4nl_value(kv) * xb[ib * 32 + tid];
        }
    }

    // Warp shuffle reduction (5 steps, no shared memory)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] = sum;
}