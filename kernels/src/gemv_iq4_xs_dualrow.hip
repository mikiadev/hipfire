// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// IQ4_XS dual-row GEMV (GSQ-RCO perf item 2, experiment): two rows per
// 32-thread wave with a contiguous 8-element per-thread remap — 8 qs
// nibbles read as ONE contiguous u64 per thread (vs 8 byte loads in the
// scalar kernel), one scales byte + scales_h, float4 x shared across both
// rows. Same block_iq4_xs layout (136 B/256).
//
// Thread mapping: thread tid handles elements e = 8*tid + k (k=0..7) of
// each 256-block. e = ib*32 + tid' with ib = tid>>2, tid' = (tid&3)*8 + k,
// so per-thread constants are:
//   ls = (scales_l[ib/2] >> 4*(ib%2) & 0xf) | ((scales_h >> 2*ib) & 3) << 4
//   dl = d * (ls - 32)
//   qs_base = ib*16 + (tid&3)*8
//   kv = (tid&3 >= 2) ? (qs[qs_base+k] >> 4) : (qs[qs_base+k] & 0xF)
//   value = dl * kvalues_iq4nl[kv]; sum += value * x[8*tid + k]
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

// kvalues_iq4nl from ggml-common.h / gguf_iq.rs (file scope so the kernel
// can LDS-stage it: per-element lookups become ds_read, not global loads).
const float IQ4NL_KV[16] = { -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f,
                             -22.0f, -10.0f, 1.0f, 13.0f, 25.0f, 38.0f,
                             53.0f, 69.0f, 89.0f, 113.0f };
__device__ __forceinline__ float iq4nl_value(int nibble) {
    return IQ4NL_KV[nibble & 0xF];
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_iq4_xs_dualrow(
    const unsigned char* __restrict__ A_iq4xs,  // [M, K/256 * 136] packed IQ4_XS
    const float* __restrict__ x,                // [K]
    float* __restrict__ y,                      // [M]
    int M, int K
) {
    const int row0 = blockIdx.x << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 256;
    const size_t row_stride = (size_t)blocks_per_row * 136;
    const unsigned char* row0_data = A_iq4xs + (size_t)row0 * row_stride;
    const unsigned char* row1_data = A_iq4xs + (size_t)(have_row1 ? row1 : row0) * row_stride;

    // LDS-stage the 16-entry kvalues table once per kernel: per-element
    // lookups become ds_read instead of global loads (16 per iteration).
    __shared__ float s_kv[16];
    if (tid < 16) s_kv[tid] = IQ4NL_KV[tid];
    __syncthreads();

    const int ib = tid >> 2;
    // Element e = 8*tid + k has e%16 = ((tid&3)&1)*8 + k, so the 8 qs bytes
    // for a thread live at qs[ib*16 + ((tid&3)&1)*8 .. +8).
    const int qs_base = ib * 16 + ((tid & 3) & 1) * 8;
    const bool high = (tid & 3) >= 2;
    const int x_off = 8 * tid;

    float sum0 = 0.0f, sum1 = 0.0f;

    // GSQ-RCO perf item 2b: stage 4 blocks (2 rows x 4 x 136 B) into LDS
    // with COALESCED u32 loads, one sync, then decode from LDS.
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
            const unsigned char* b0 = s_blk + bi * 136;
            const unsigned char* b1 = s_blk + 272 + bi * 136;
            const float* xb = x + (bg + bi) * 256 + x_off;

            const float4 xa = *reinterpret_cast<const float4*>(xb);
            const float4 xb4 = *reinterpret_cast<const float4*>(xb + 4);

            float d0 = half_to_float((unsigned short)(b0[0] | (b0[1] << 8)));
            unsigned short sh0 = (unsigned short)(b0[2] | (b0[3] << 8));
            int ls0 = (int)((b0[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xf)
                    | (int)(((sh0 >> (2 * ib)) & 3) << 4);
            float dl0 = d0 * (float)(ls0 - 32);
            unsigned long long q0 = *(const unsigned long long*)(b0 + 8 + qs_base);
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int kv = high ? ((int)((q0 >> (8 * k)) & 0xff) >> 4) : ((int)((q0 >> (8 * k)) & 0xff) & 0xf);
                float w = dl0 * s_kv[kv];
                sum0 += w * (k < 4 ? (k == 0 ? xa.x : k == 1 ? xa.y : k == 2 ? xa.z : xa.w)
                                   : (k == 4 ? xb4.x : k == 5 ? xb4.y : k == 6 ? xb4.z : xb4.w));
            }

            float d1 = half_to_float((unsigned short)(b1[0] | (b1[1] << 8)));
            unsigned short sh1 = (unsigned short)(b1[2] | (b1[3] << 8));
            int ls1 = (int)((b1[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xf)
                    | (int)(((sh1 >> (2 * ib)) & 3) << 4);
            float dl1 = d1 * (float)(ls1 - 32);
            unsigned long long q1 = *(const unsigned long long*)(b1 + 8 + qs_base);
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int kv = high ? ((int)((q1 >> (8 * k)) & 0xff) >> 4) : ((int)((q1 >> (8 * k)) & 0xff) & 0xf);
                float w = dl1 * s_kv[kv];
                sum1 += w * (k < 4 ? (k == 0 ? xa.x : k == 1 ? xa.y : k == 2 ? xa.z : xa.w)
                                   : (k == 4 ? xb4.x : k == 5 ? xb4.y : k == 6 ? xb4.z : xb4.w));
            }
        }
        __syncthreads();
    }
    // Scalar tail for K not a multiple of 1024.
    for (; bg < blocks_per_row; bg++) {
        const unsigned char* b0 = row0_data + bg * 136;
        const unsigned char* b1 = row1_data + bg * 136;
        const float* xb = x + bg * 256 + x_off;
        const float4 xa = *reinterpret_cast<const float4*>(xb);
        const float4 xb4 = *reinterpret_cast<const float4*>(xb + 4);

        float d0 = half_to_float((unsigned short)(b0[0] | (b0[1] << 8)));
        unsigned short sh0 = (unsigned short)(b0[2] | (b0[3] << 8));
        int ls0 = (int)((b0[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xf)
                | (int)(((sh0 >> (2 * ib)) & 3) << 4);
        float dl0 = d0 * (float)(ls0 - 32);
        unsigned long long q0 = *(const unsigned long long*)(b0 + 8 + qs_base);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int kv = high ? ((int)((q0 >> (8 * k)) & 0xff) >> 4) : ((int)((q0 >> (8 * k)) & 0xff) & 0xf);
            float w = dl0 * s_kv[kv];
            sum0 += w * (k < 4 ? (k == 0 ? xa.x : k == 1 ? xa.y : k == 2 ? xa.z : xa.w)
                               : (k == 4 ? xb4.x : k == 5 ? xb4.y : k == 6 ? xb4.z : xb4.w));
        }

        float d1 = half_to_float((unsigned short)(b1[0] | (b1[1] << 8)));
        unsigned short sh1 = (unsigned short)(b1[2] | (b1[3] << 8));
        int ls1 = (int)((b1[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xf)
                | (int)(((sh1 >> (2 * ib)) & 3) << 4);
        float dl1 = d1 * (float)(ls1 - 32);
        unsigned long long q1 = *(const unsigned long long*)(b1 + 8 + qs_base);
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int kv = high ? ((int)((q1 >> (8 * k)) & 0xff) >> 4) : ((int)((q1 >> (8 * k)) & 0xff) & 0xf);
            float w = dl1 * s_kv[kv];
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
