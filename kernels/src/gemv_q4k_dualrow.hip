// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Q4_K dual-row GEMV (GSQ-RCO perf item 2, experiment): two rows per
// 32-thread wave with a contiguous 8-element per-thread remap — 8 qbytes
// read as ONE contiguous u64 per thread (vs 8 byte loads in the scalar
// kernel), float4 x shared across both rows. Same block_q4_K layout
// (144 B/256).
//
// Thread mapping: thread tid handles elements e = 8*tid + k (k=0..7) of
// each 256-block. All 8 elements land in the SAME 64-element group
// g = tid>>3 and the same 32-element sub-group sg = 2*g + (t>=4), where
// t = tid&7:
//   l = 8*t + k  (k=0..7); if t<=3 the elements are the LOW nibbles of
//   qdata[16 + 8*t + k], else the HIGH nibbles of qdata[16 + 8*(t-4) + k].
//   So qbyte_base = 16 + 8*(t&3), nibble_shift = (t>=4) ? 4 : 0.
// Scale/min for sub-group sg (ggml 6-bit packing):
//   sg<4: s = d*(sc[sg]&63),        m = dmin*(sc[4+sg]&63)
//   sg>=4: s = d*((sc[8+lo]&0xf)|((sc[lo]>>6)<<4)),
//          m = dmin*((sc[8+lo]>>4)|((sc[4+lo]>>6)<<4)), lo = sg&3
//   value = s * nibble - m; sum += value * x[8*tid + k]
//
// FP-reduction order differs from the scalar kernel — greedy parity
// verified by harness.
//
// Grid: [ceil(M/2), 1, 1]; Block: [32, 1, 1].

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

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_q4k_dualrow(
    const unsigned char* __restrict__ A_q4k,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row0 = blockIdx.x << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 256;
    const size_t row_stride = (size_t)blocks_per_row * 144;
    const unsigned char* row0_data = A_q4k + (size_t)row0 * row_stride;
    const unsigned char* row1_data = A_q4k + (size_t)(have_row1 ? row1 : row0) * row_stride;

    const int g  = tid >> 3;          // 64-element group (0..3)
    const int t  = tid & 7;
    const int sg = 2 * g + (t >= 4 ? 1 : 0);
    const int lo = sg & 3;
    const bool hi_sg = sg >= 4;
    const int qbyte_base = 16 + 32 * g + 8 * (t & 3);
    const int nib_shift = (t >= 4) ? 4 : 0;
    const int x_off = 8 * tid;

    float sum0 = 0.0f, sum1 = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* b0 = row0_data + bi * 144;
        const unsigned char* b1 = row1_data + bi * 144;
        const float* xb = x + bi * 256 + x_off;

        const float4 xa = *reinterpret_cast<const float4*>(xb);
        const float4 xb4 = *reinterpret_cast<const float4*>(xb + 4);

        // Row 0
        float d0 = (float)*((const _Float16*)b0);
        float dm0 = (float)*((const _Float16*)(b0 + 2));
        const unsigned char* sc0 = b0 + 4;
        float s0, m0;
        if (!hi_sg) {
            s0 = d0 * (float)(sc0[sg] & 63);
            m0 = dm0 * (float)(sc0[4 + sg] & 63);
        } else {
            s0 = d0 * (float)((sc0[8 + lo] & 0xF) | ((sc0[lo] >> 6) << 4));
            m0 = dm0 * (float)((sc0[8 + lo] >> 4) | ((sc0[4 + lo] >> 6) << 4));
        }
        unsigned long long q0 = *(const unsigned long long*)(b0 + qbyte_base);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int nib = (int)((q0 >> (8 * k)) & 0xff) >> nib_shift & 0xF;
            float w = s0 * (float)nib - m0;
            sum0 += w * (k < 4 ? (k == 0 ? xa.x : k == 1 ? xa.y : k == 2 ? xa.z : xa.w)
                               : (k == 4 ? xb4.x : k == 5 ? xb4.y : k == 6 ? xb4.z : xb4.w));
        }

        // Row 1 (same x)
        float d1 = (float)*((const _Float16*)b1);
        float dm1 = (float)*((const _Float16*)(b1 + 2));
        const unsigned char* sc1 = b1 + 4;
        float s1, m1;
        if (!hi_sg) {
            s1 = d1 * (float)(sc1[sg] & 63);
            m1 = dm1 * (float)(sc1[4 + sg] & 63);
        } else {
            s1 = d1 * (float)((sc1[8 + lo] & 0xF) | ((sc1[lo] >> 6) << 4));
            m1 = dm1 * (float)((sc1[8 + lo] >> 4) | ((sc1[4 + lo] >> 6) << 4));
        }
        unsigned long long q1 = *(const unsigned long long*)(b1 + qbyte_base);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int nib = (int)((q1 >> (8 * k)) & 0xff) >> nib_shift & 0xF;
            float w = s1 * (float)nib - m1;
            sum1 += w * (k < 4 ? (k == 0 ? xa.x : k == 1 ? xa.y : k == 2 ? xa.z : xa.w)
                               : (k == 4 ? xb4.x : k == 5 ? xb4.y : k == 6 ? xb4.z : xb4.w));
        }
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
