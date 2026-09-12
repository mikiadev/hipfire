// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// IQ3_S dual-row GEMV (GSQ-RCO perf item 2): Y[M] = A_iq3s[M,K] @ x[K],
// two rows per 32-thread wave (x shared), with a contiguous 8-element
// per-thread remap that collapses the IQ3_S decode to TWO grid lookups per
// 8 elements (the scalar gemv_iq3_s did 8) and one byte each for qs/qh/
// scales/signs, plus float4 x loads. Same block_iq3_s layout (110 B/256).
//
// Thread mapping: thread tid handles elements e = 8*tid + k (k=0..7) of
// each 256-block. e = ib*32 + tid' with ib = tid>>2, tid' = (tid&3)*8 + k,
// so per-thread constants are:
//   i = ib>>1, half = ib&1, l = tid&3
//   qs_base = i*16 + half*8 + 2*l
//   shift_lo = 8-2*l (k<4), shift_hi = 7-2*l (k>=4)
//   grid_idx0 = qs[qs_base]     | ((qh[ib] << shift_lo) & 256)
//   grid_idx1 = qs[qs_base + 1] | ((qh[ib] << shift_hi) & 256)
//   grid0 = IQ3S_GRID[grid_idx0], grid1 = IQ3S_GRID[grid_idx1]
//   byte = (k<4) ? (grid0 >> 8*k) : (grid1 >> 8*(k-4)); gb = (int8)byte
//   sign = (signs[i*8 + half*4 + l] >> k) & 1
//   nib = half ? (scales[i] >> 4) : (scales[i] & 0xf)
//   db = d * (1 + 2*nib); w = db * gb * (sign ? -1 : 1)
//   sum += w * x[8*tid + k]
//
// FP-reduction order differs from the CPU decoder (per-lane partials are
// contiguous 8-element chunks, not group-strided) — greedy parity is
// verified by harness (test_outproj_decode max_abs ~1e-5), not bit-exact.
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

    const unsigned int IQ3S_GRID[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010B, 0x0101010F, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030D, 0x01010501, 0x01010503, 0x0101050B, 0x01010707, 0x01010901, 0x01010905,
    0x0101090B, 0x0101090F, 0x01010B03, 0x01010B07, 0x01010D01, 0x01010D05, 0x01010F03, 0x01010F09,
    0x01010F0F, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030B,
    0x01030501, 0x01030507, 0x0103050F, 0x01030703, 0x0103070B, 0x01030909, 0x01030D03, 0x01030D0B,
    0x01030F05, 0x01050101, 0x01050103, 0x0105010B, 0x0105010F, 0x01050301, 0x01050307, 0x0105030D,
    0x01050503, 0x0105050B, 0x01050701, 0x01050709, 0x01050905, 0x0105090B, 0x0105090F, 0x01050B03,
    0x01050B07, 0x01050F01, 0x01050F07, 0x01070107, 0x01070303, 0x0107030B, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070D, 0x01070909, 0x01070B01, 0x01070B05, 0x01070D0F, 0x01070F03,
    0x01070F0B, 0x01090101, 0x01090307, 0x0109030F, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090B03, 0x01090F01, 0x010B0105, 0x010B0109, 0x010B0501, 0x010B0505, 0x010B050D,
    0x010B0707, 0x010B0903, 0x010B090B, 0x010B090F, 0x010B0D0D, 0x010B0F07, 0x010D010D, 0x010D0303,
    0x010D0307, 0x010D0703, 0x010D0B05, 0x010D0F03, 0x010F0101, 0x010F0105, 0x010F0109, 0x010F0501,
    0x010F0505, 0x010F050D, 0x010F0707, 0x010F0B01, 0x010F0B09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030B, 0x0301030F, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070D, 0x03010B09, 0x03010B0D, 0x03010D03, 0x03010F05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010D, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030B01, 0x03030B05, 0x03030F01, 0x03030F0D, 0x03050101, 0x03050305, 0x0305030B,
    0x0305030F, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050B0B, 0x03050D01,
    0x03050F05, 0x03070103, 0x03070109, 0x0307010F, 0x03070301, 0x03070307, 0x03070503, 0x0307050F,
    0x03070701, 0x03070709, 0x03070903, 0x03070D05, 0x03070F01, 0x03090107, 0x0309010B, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090D, 0x03090B01, 0x03090B09, 0x030B0103,
    0x030B0301, 0x030B0307, 0x030B0503, 0x030B0701, 0x030B0705, 0x030B0B03, 0x030D0501, 0x030D0509,
    0x030D050F, 0x030D0909, 0x030D090D, 0x030F0103, 0x030F0107, 0x030F0301, 0x030F0305, 0x030F0503,
    0x030F070B, 0x030F0903, 0x030F0D05, 0x030F0F01, 0x05010101, 0x05010103, 0x05010107, 0x0501010B,
    0x0501010F, 0x05010301, 0x05010305, 0x05010309, 0x0501030D, 0x05010503, 0x05010507, 0x0501050F,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090B, 0x05010B01, 0x05010B05, 0x05010D0F,
    0x05010F01, 0x05010F07, 0x05010F0B, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030F,
    0x05030505, 0x0503050B, 0x05030703, 0x05030709, 0x05030905, 0x05030B03, 0x05050103, 0x05050109,
    0x0505010F, 0x05050503, 0x05050507, 0x05050701, 0x0505070F, 0x05050903, 0x05050B07, 0x05050B0F,
    0x05050F03, 0x05050F09, 0x05070101, 0x05070105, 0x0507010B, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070B01, 0x05070D0D, 0x05090103, 0x0509010F, 0x05090501,
    0x05090507, 0x05090705, 0x0509070B, 0x05090903, 0x05090F05, 0x05090F0B, 0x050B0109, 0x050B0303,
    0x050B0505, 0x050B070F, 0x050B0901, 0x050B0B07, 0x050B0F01, 0x050D0101, 0x050D0105, 0x050D010F,
    0x050D0503, 0x050D0B0B, 0x050D0D03, 0x050F010B, 0x050F0303, 0x050F050D, 0x050F0701, 0x050F0907,
    0x050F0B01, 0x07010105, 0x07010303, 0x07010307, 0x0701030B, 0x0701030F, 0x07010505, 0x07010703,
    0x07010707, 0x0701070B, 0x07010905, 0x07010909, 0x0701090F, 0x07010B03, 0x07010D07, 0x07010F03,
    0x07030103, 0x07030107, 0x0703010B, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030D01,
    0x07030F05, 0x07030F0D, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050B01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050F, 0x07070701, 0x07070903,
    0x07070907, 0x0707090F, 0x07070B0B, 0x07070F07, 0x07090107, 0x07090303, 0x0709030D, 0x07090505,
    0x07090703, 0x07090B05, 0x07090D01, 0x07090D09, 0x070B0103, 0x070B0301, 0x070B0305, 0x070B050B,
    0x070B0705, 0x070B0909, 0x070B0B0D, 0x070B0F07, 0x070D030D, 0x070D0903, 0x070F0103, 0x070F0107,
    0x070F0501, 0x070F0505, 0x070F070B, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050F, 0x09010705, 0x09010903, 0x09010B01, 0x09010F01, 0x09030105, 0x0903010F, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070B, 0x09030907, 0x09030B03, 0x09030B0B, 0x09050103,
    0x09050107, 0x09050301, 0x0905030B, 0x09050503, 0x09050707, 0x09050901, 0x09050B0F, 0x09050D05,
    0x09050F01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070B,
    0x09090101, 0x09090105, 0x09090509, 0x0909070F, 0x09090901, 0x09090F03, 0x090B010B, 0x090B010F,
    0x090B0503, 0x090B0D05, 0x090D0307, 0x090D0709, 0x090D0D01, 0x090F0301, 0x090F030B, 0x090F0701,
    0x090F0907, 0x090F0B03, 0x0B010105, 0x0B010301, 0x0B010309, 0x0B010505, 0x0B010901, 0x0B010909,
    0x0B01090F, 0x0B010B05, 0x0B010D0D, 0x0B010F09, 0x0B030103, 0x0B030107, 0x0B03010B, 0x0B030305,
    0x0B030503, 0x0B030705, 0x0B030F05, 0x0B050101, 0x0B050303, 0x0B050507, 0x0B050701, 0x0B05070D,
    0x0B050B07, 0x0B070105, 0x0B07010F, 0x0B070301, 0x0B07050F, 0x0B070909, 0x0B070B03, 0x0B070D0B,
    0x0B070F07, 0x0B090103, 0x0B090109, 0x0B090501, 0x0B090705, 0x0B09090D, 0x0B0B0305, 0x0B0B050D,
    0x0B0B0B03, 0x0B0B0B07, 0x0B0D0905, 0x0B0F0105, 0x0B0F0109, 0x0B0F0505, 0x0D010303, 0x0D010307,
    0x0D01030B, 0x0D010703, 0x0D010707, 0x0D010D01, 0x0D030101, 0x0D030501, 0x0D03050F, 0x0D030D09,
    0x0D050305, 0x0D050709, 0x0D050905, 0x0D050B0B, 0x0D050D05, 0x0D050F01, 0x0D070101, 0x0D070309,
    0x0D070503, 0x0D070901, 0x0D09050B, 0x0D090907, 0x0D090D05, 0x0D0B0101, 0x0D0B0107, 0x0D0B0709,
    0x0D0B0D01, 0x0D0D010B, 0x0D0D0901, 0x0D0F0303, 0x0D0F0307, 0x0F010101, 0x0F010109, 0x0F01010F,
    0x0F010501, 0x0F010505, 0x0F01070D, 0x0F010901, 0x0F010B09, 0x0F010D05, 0x0F030105, 0x0F030303,
    0x0F030509, 0x0F030907, 0x0F03090B, 0x0F050103, 0x0F050109, 0x0F050301, 0x0F05030D, 0x0F050503,
    0x0F050701, 0x0F050B03, 0x0F070105, 0x0F070705, 0x0F07070B, 0x0F070B07, 0x0F090103, 0x0F09010B,
    0x0F090307, 0x0F090501, 0x0F090B01, 0x0F0B0505, 0x0F0B0905, 0x0F0D0105, 0x0F0D0703, 0x0F0F0101};
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_iq3_s_dualrow(
    const unsigned char* __restrict__ A_iq3s,  // [M, K/256 * 110] packed IQ3_S
    const float* __restrict__ x,               // [K]
    float* __restrict__ y,                     // [M]
    int M, int K
) {
    const int row0 = blockIdx.x << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 256;
    const size_t row_stride = (size_t)blocks_per_row * 110;
    const unsigned char* row0_data = A_iq3s + (size_t)row0 * row_stride;
    const unsigned char* row1_data = A_iq3s + (size_t)(have_row1 ? row1 : row0) * row_stride;

    // Per-thread decode constants (same for both rows; weights differ).
    const int ib   = tid >> 2;       // 32-element group within the 256-block
    const int l    = tid & 3;        // l-block within the group
    const int i    = ib >> 1;        // scales/qh pair index
    const int half = ib & 1;
    const int qs_base   = i * 16 + half * 8 + 2 * l;
    const int shift_lo  = 8 - 2 * l;  // k < 4
    const int shift_hi  = 7 - 2 * l;  // k >= 4
    const int x_off = 8 * tid;        // contiguous 8-element chunk

    float sum0 = 0.0f, sum1 = 0.0f;

    // GSQ-RCO perf item 2b experiment: stage 4 blocks (2 rows x 4 x 110 B)
    // into LDS with COALESCED u32 loads, one sync, then decode from LDS.
    // The scalar kernel's scattered u8/u16 weight loads become ds_read.
    __shared__ unsigned char s_blk[880];
    int bg = 0;
    for (; bg + 4 <= blocks_per_row; bg += 4) {
        const unsigned int* src0 = (const unsigned int*)(row0_data + bg * 110);
        const unsigned int* src1 = (const unsigned int*)(row1_data + bg * 110);
        #pragma unroll
        for (int i = tid; i < 110; i += 32) {
            ((unsigned int*)s_blk)[i] = src0[i];
            ((unsigned int*)s_blk)[110 + i] = src1[i];
        }
        __syncthreads();
        #pragma unroll
        for (int bi = 0; bi < 4; bi++) {
            const unsigned char* b0 = s_blk + bi * 110;
            const unsigned char* b1 = s_blk + 440 + bi * 110;
            const float* xb = x + (bg + bi) * 256 + x_off;

            const float4 xa = *reinterpret_cast<const float4*>(xb);
            const float4 xb4 = *reinterpret_cast<const float4*>(xb + 4);

            float d0 = half_to_float((unsigned short)(b0[0] | (b0[1] << 8)));
            unsigned char qh0 = b0[66 + ib];
            unsigned char sc0 = b0[106 + i];
            int nib0 = half ? (sc0 >> 4) : (sc0 & 0xf);
            float db0 = d0 * (1.0f + 2.0f * (float)nib0);
            unsigned char sign0 = b0[74 + i * 8 + half * 4 + l];
            unsigned int grid0a = IQ3S_GRID[(int)b0[2 + qs_base] | ((qh0 << shift_lo) & 256)];
            unsigned int grid0b = IQ3S_GRID[(int)b0[2 + qs_base + 1] | ((qh0 << shift_hi) & 256)];
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int gb = (int)(int8_t)((k < 4 ? grid0a : grid0b) >> (8 * (k & 3)));
                float w = db0 * (float)gb * ((sign0 >> k) & 1 ? -1.0f : 1.0f);
                sum0 += w * (k < 4 ? (k == 0 ? xa.x : k == 1 ? xa.y : k == 2 ? xa.z : xa.w)
                                   : (k == 4 ? xb4.x : k == 5 ? xb4.y : k == 6 ? xb4.z : xb4.w));
            }

            float d1 = half_to_float((unsigned short)(b1[0] | (b1[1] << 8)));
            unsigned char qh1 = b1[66 + ib];
            unsigned char sc1 = b1[106 + i];
            int nib1 = half ? (sc1 >> 4) : (sc1 & 0xf);
            float db1 = d1 * (1.0f + 2.0f * (float)nib1);
            unsigned char sign1 = b1[74 + i * 8 + half * 4 + l];
            unsigned int grid1a = IQ3S_GRID[(int)b1[2 + qs_base] | ((qh1 << shift_lo) & 256)];
            unsigned int grid1b = IQ3S_GRID[(int)b1[2 + qs_base + 1] | ((qh1 << shift_hi) & 256)];
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int gb = (int)(int8_t)((k < 4 ? grid1a : grid1b) >> (8 * (k & 3)));
                float w = db1 * (float)gb * ((sign1 >> k) & 1 ? -1.0f : 1.0f);
                sum1 += w * (k < 4 ? (k == 0 ? xa.x : k == 1 ? xa.y : k == 2 ? xa.z : xa.w)
                                   : (k == 4 ? xb4.x : k == 5 ? xb4.y : k == 6 ? xb4.z : xb4.w));
            }
        }
        __syncthreads();
    }
    // Scalar tail for K not a multiple of 1024.
    for (; bg < blocks_per_row; bg++) {
        const unsigned char* b0 = row0_data + bg * 110;
        const unsigned char* b1 = row1_data + bg * 110;
        const float* xb = x + bg * 256 + x_off;
        const float4 xa = *reinterpret_cast<const float4*>(xb);
        const float4 xb4 = *reinterpret_cast<const float4*>(xb + 4);

        float d0 = half_to_float((unsigned short)(b0[0] | (b0[1] << 8)));
        unsigned char qh0 = b0[66 + ib];
        unsigned char sc0 = b0[106 + i];
        int nib0 = half ? (sc0 >> 4) : (sc0 & 0xf);
        float db0 = d0 * (1.0f + 2.0f * (float)nib0);
        unsigned char sign0 = b0[74 + i * 8 + half * 4 + l];
        unsigned int grid0a = IQ3S_GRID[(int)b0[2 + qs_base] | ((qh0 << shift_lo) & 256)];
        unsigned int grid0b = IQ3S_GRID[(int)b0[2 + qs_base + 1] | ((qh0 << shift_hi) & 256)];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int gb = (int)(int8_t)((k < 4 ? grid0a : grid0b) >> (8 * (k & 3)));
            float w = db0 * (float)gb * ((sign0 >> k) & 1 ? -1.0f : 1.0f);
            sum0 += w * (k < 4 ? (k == 0 ? xa.x : k == 1 ? xa.y : k == 2 ? xa.z : xa.w)
                               : (k == 4 ? xb4.x : k == 5 ? xb4.y : k == 6 ? xb4.z : xb4.w));
        }

        float d1 = half_to_float((unsigned short)(b1[0] | (b1[1] << 8)));
        unsigned char qh1 = b1[66 + ib];
        unsigned char sc1 = b1[106 + i];
        int nib1 = half ? (sc1 >> 4) : (sc1 & 0xf);
        float db1 = d1 * (1.0f + 2.0f * (float)nib1);
        unsigned char sign1 = b1[74 + i * 8 + half * 4 + l];
        unsigned int grid1a = IQ3S_GRID[(int)b1[2 + qs_base] | ((qh1 << shift_lo) & 256)];
        unsigned int grid1b = IQ3S_GRID[(int)b1[2 + qs_base + 1] | ((qh1 << shift_hi) & 256)];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int gb = (int)(int8_t)((k < 4 ? grid1a : grid1b) >> (8 * (k & 3)));
            float w = db1 * (float)gb * ((sign1 >> k) & 1 ? -1.0f : 1.0f);
            sum1 += w * (k < 4 ? (k == 0 ? xa.x : k == 1 ? xa.y : k == 2 ? xa.z : xa.w)
                               : (k == 4 ? xb4.x : k == 5 ? xb4.y : k == 6 ? xb4.z : xb4.w));
        }
    }

    // Warp shuffle reduction (5 steps, no shared memory)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
    }
    if (tid == 0) {
        y[row0] = sum0;
        if (have_row1) y[row1] = sum1;
    }
}
