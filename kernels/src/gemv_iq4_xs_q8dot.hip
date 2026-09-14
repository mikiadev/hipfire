// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// q8_1 + dp4a dual-row IQ4_XS GEMV (GSQ-RCO decode-next): the IQ4_XS twin of
// gemv_iq3_s_q8dot.hip. Same dual-row mapping as gemv_iq4_xs_dualrow.hip, but
// the activation is pre-quantized to q8_1 and the dot is v_dot4_i32_i8.
//
// IQ4_XS differs from IQ3_S/IQ3_XXS: there is no separate sign array — the
// 4-bit nibble selects a signed value from the 16-entry kvalues_iq4nl table, so
// the int8 weight pack is built directly from table lookups (no sign op).
// The table is LDS-staged (per-element ds_read instead of global loads).
//
// block_iq4_xs layout (136 B / 256): d[2] scales_h[2] scales_l[4] qs[128].
// Per 8 elements per row: 8 nibble extracts, 8 kvalue lookups, pack 2x int32,
// 2 dp4a.
//
// Grid: [ceil(M/2), 1, 1]; Block: [32, 1, 1].

#include <hip/hip_runtime.h>

__device__ __forceinline__ float hipfire_iq4xs_half_to_float(unsigned short h) {
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

// kvalues_iq4nl as signed int8 (the dp4a weight pack lanes).
const signed char IQ4NL_KV8[16] = { -127, -104, -83, -65, -49, -35, -22, -10,
                                    1, 13, 25, 38, 53, 69, 89, 113 };

__device__ __forceinline__ void iq4xs_q8dot_block(
    const unsigned char* __restrict__ b0,
    const unsigned char* __restrict__ b1,
    const float* __restrict__ xscales,
    int xg_base,
    const signed char* __restrict__ xs,
    int xblk,
    const int tid, const int ib, const int qs_base, const bool high,
    const signed char* __restrict__ s_kv,
    float* __restrict__ acc0, float* __restrict__ acc1) {

    const int* xp = reinterpret_cast<const int*>(xs + xblk * 256 + 8 * tid);
    const int a_low  = xp[0];
    const int a_high = xp[1];
    const float xscale = xscales[xg_base + (tid >> 2)];

#pragma unroll
    for (int r = 0; r < 2; r++) {
        const unsigned char* b = r ? b1 : b0;

        float d = hipfire_iq4xs_half_to_float((unsigned short)(b[0] | (b[1] << 8)));
        unsigned short sh = (unsigned short)(b[2] | (b[3] << 8));
        int ls = (int)((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xf)
               | (int)(((sh >> (2 * ib)) & 3) << 4);
        float dl = d * (float)(ls - 32);

        unsigned long long q = *(const unsigned long long*)(b + 8 + qs_base);

        int w_low = 0, w_high = 0;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            int byte = (int)((q >> (8 * k)) & 0xff);
            int kv = high ? (byte >> 4) : (byte & 0xf);
            w_low |= (int)((unsigned char)s_kv[kv]) << (8 * k);
        }
#pragma unroll
        for (int k = 0; k < 4; k++) {
            int byte = (int)((q >> (8 * (4 + k))) & 0xff);
            int kv = high ? (byte >> 4) : (byte & 0xf);
            w_high |= (int)((unsigned char)s_kv[kv]) << (8 * k);
        }

        int sumi = __builtin_amdgcn_sudot4(true, w_low,  true, a_low,  0, false);
        sumi     = __builtin_amdgcn_sudot4(true, w_high, true, a_high, sumi, false);

        float acc = dl * xscale * (float)sumi;
        if (r == 0) *acc0 += acc; else *acc1 += acc;
    }
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_iq4_xs_q8dot(
    const unsigned char* __restrict__ A_iq4xs, // [M, K/256 * 136]
    const signed char* __restrict__ xs,        // [K] q8_1 int8
    const float* __restrict__ xscales,         // [K/32]
    float* __restrict__ y,                     // [M]
    int M, int K) {

    const int row0 = blockIdx.x << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 256;
    const size_t row_stride = (size_t)blocks_per_row * 136;
    const unsigned char* row0_data = A_iq4xs + (size_t)row0 * row_stride;
    const unsigned char* row1_data = A_iq4xs + (size_t)(have_row1 ? row1 : row0) * row_stride;

    __shared__ signed char s_kv[16];
    if (tid < 16) s_kv[tid] = IQ4NL_KV8[tid];
    __syncthreads();

    const int ib = tid >> 2;
    const int qs_base = ib * 16 + ((tid & 3) & 1) * 8;
    const bool high = (tid & 3) >= 2;

    float sum0 = 0.0f, sum1 = 0.0f;

    __shared__ unsigned char s_blk[544];
    int bg = 0;
    for (; bg + 2 <= blocks_per_row; bg += 2) {
        const unsigned int* src0 = (const unsigned int*)(row0_data + bg * 136);
        const unsigned int* src1 = (const unsigned int*)(row1_data + bg * 136);
        #pragma unroll
        for (int i = tid; i < 68; i += 32) {
            ((unsigned int*)s_blk)[i] = src0[i];
            ((unsigned int*)s_blk)[68 + i] = src1[i];
        }
        __syncthreads();
        #pragma unroll
        for (int bi = 0; bi < 2; bi++) {
            iq4xs_q8dot_block(
                s_blk + bi * 136, s_blk + 272 + bi * 136,
                xscales, (bg + bi) * 8, xs, bg + bi,
                tid, ib, qs_base, high, s_kv, &sum0, &sum1);
        }
        __syncthreads();
    }
    for (; bg < blocks_per_row; bg++) {
        iq4xs_q8dot_block(
            row0_data + bg * 136, row1_data + bg * 136,
            xscales, bg * 8, xs, bg,
            tid, ib, qs_base, high, s_kv, &sum0, &sum1);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
    }
    if (tid == 0) {
        y[row0] = sum0;
        if (have_row1) y[row1] = sum1;
    }
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_iq4_xs_q8dot_residual(
    const unsigned char* __restrict__ A_iq4xs,
    const signed char* __restrict__ xs,
    const float* __restrict__ xscales,
    float* __restrict__ y,
    int M, int K) {

    const int row0 = blockIdx.x << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 256;
    const size_t row_stride = (size_t)blocks_per_row * 136;
    const unsigned char* row0_data = A_iq4xs + (size_t)row0 * row_stride;
    const unsigned char* row1_data = A_iq4xs + (size_t)(have_row1 ? row1 : row0) * row_stride;

    __shared__ signed char s_kv[16];
    if (tid < 16) s_kv[tid] = IQ4NL_KV8[tid];
    __syncthreads();

    const int ib = tid >> 2;
    const int qs_base = ib * 16 + ((tid & 3) & 1) * 8;
    const bool high = (tid & 3) >= 2;

    float sum0 = 0.0f, sum1 = 0.0f;

    __shared__ unsigned char s_blk[544];
    int bg = 0;
    for (; bg + 2 <= blocks_per_row; bg += 2) {
        const unsigned int* src0 = (const unsigned int*)(row0_data + bg * 136);
        const unsigned int* src1 = (const unsigned int*)(row1_data + bg * 136);
        #pragma unroll
        for (int i = tid; i < 68; i += 32) {
            ((unsigned int*)s_blk)[i] = src0[i];
            ((unsigned int*)s_blk)[68 + i] = src1[i];
        }
        __syncthreads();
        #pragma unroll
        for (int bi = 0; bi < 2; bi++) {
            iq4xs_q8dot_block(
                s_blk + bi * 136, s_blk + 272 + bi * 136,
                xscales, (bg + bi) * 8, xs, bg + bi,
                tid, ib, qs_base, high, s_kv, &sum0, &sum1);
        }
        __syncthreads();
    }
    for (; bg < blocks_per_row; bg++) {
        iq4xs_q8dot_block(
            row0_data + bg * 136, row1_data + bg * 136,
            xscales, bg * 8, xs, bg,
            tid, ib, qs_base, high, s_kv, &sum0, &sum1);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
    }
    if (tid == 0) {
        y[row0] += sum0;
        if (have_row1) y[row1] += sum1;
    }
}