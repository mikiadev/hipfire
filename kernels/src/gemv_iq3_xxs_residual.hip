// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Fused-residual variant of gemv_iq3_xxs: the epilogue is `y[row] += sum`
// (residual in-place; identical W·x FMA order -> greedy parity with the
// un-fused GEMV + add_inplace path).

#include <hip/hip_runtime.h>

// IQ3_XXS GEMV (GSQ-RCO native): Y[M] = A_iq3xxs[M, K] @ x[K] where A is
// IQ3_XXS quantized (98 bytes per group of 256 elements, 3.0625 bpw).
//
// Block layout (ggml-common.h block_iq3_xxs), 256 elements / 98 bytes:
//   [0..2)    d f16
//   [2..66)   qs[64] — 8-bit indices into iq3xxs_grid[256]
//   [66..98)  scales_and_signs[32] — 8 x  aux32 (per 32-element group)
// Decode per 32-element group ib32:
//   aux32 = le(scales_and_signs[4*ib32 ..])
//   db = d * (0.5 + (aux32 >> 28)) * 0.5
//   per l-block (8 elements): signs = ksigns_iq2xs[(aux32 >> 7*l) & 127],
//   grid1 = iq3xxs_grid[qs[2l]], grid2 = iq3xxs_grid[qs[2l+1]],
//   value = db * grid_byte * sign.
// Grid bytes are SIGNED-MAGNITUDE codebook values (verbatim , NOT
// two's complement) — cast via (int8).
//
// Per-element mapping (thread tid covers element e = ib32*32 + tid):
//   l = tid/8, j = tid%8
//   grid_idx = qs[ib32*8 + 2*l + (j>=4)]
//   signs = ksigns_iq2xs[(aux32 >> 7*l) & 127]
//   byte = (grid >> 8*(j&3)) & 0xff
//   sign = signs & (1 << j)
// FP-reduction order matches the CPU decoder (gguf_iq.rs
// dequant_iq3_xxs) group-by-group for greedy parity.
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

    const unsigned int IQ3XXS_GRID[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040C0C, 0x04040C1C, 0x04040C3E, 0x04041404, 0x04041414,
    0x04041C0C, 0x04042414, 0x04043E1C, 0x04043E2C, 0x040C040C, 0x040C041C, 0x040C0C04, 0x040C0C14,
    0x040C140C, 0x040C142C, 0x040C1C04, 0x040C1C14, 0x040C240C, 0x040C2C24, 0x040C3E04, 0x04140404,
    0x04140414, 0x04140424, 0x04140C0C, 0x04141404, 0x04141414, 0x04141C0C, 0x04141C1C, 0x04141C3E,
    0x04142C0C, 0x04142C3E, 0x04143E2C, 0x041C040C, 0x041C043E, 0x041C0C04, 0x041C0C14, 0x041C142C,
    0x041C3E04, 0x04240C1C, 0x04241C3E, 0x04242424, 0x04242C3E, 0x04243E1C, 0x04243E2C, 0x042C040C,
    0x042C043E, 0x042C1C14, 0x042C2C14, 0x04341C2C, 0x04343424, 0x043E0C04, 0x043E0C24, 0x043E0C34,
    0x043E241C, 0x043E340C, 0x0C04040C, 0x0C04041C, 0x0C040C04, 0x0C040C14, 0x0C04140C, 0x0C04141C,
    0x0C041C04, 0x0C041C14, 0x0C041C24, 0x0C04243E, 0x0C042C04, 0x0C0C0404, 0x0C0C0414, 0x0C0C0C0C,
    0x0C0C1404, 0x0C0C1414, 0x0C14040C, 0x0C14041C, 0x0C140C04, 0x0C140C14, 0x0C14140C, 0x0C141C04,
    0x0C143E14, 0x0C1C0404, 0x0C1C0414, 0x0C1C1404, 0x0C1C1C0C, 0x0C1C2434, 0x0C1C3434, 0x0C24040C,
    0x0C24042C, 0x0C242C04, 0x0C2C1404, 0x0C2C1424, 0x0C2C2434, 0x0C2C3E0C, 0x0C34042C, 0x0C3E1414,
    0x0C3E2404, 0x14040404, 0x14040414, 0x14040C0C, 0x14040C1C, 0x14041404, 0x14041414, 0x14041434,
    0x14041C0C, 0x14042414, 0x140C040C, 0x140C041C, 0x140C042C, 0x140C0C04, 0x140C0C14, 0x140C140C,
    0x140C1C04, 0x140C341C, 0x140C343E, 0x140C3E04, 0x14140404, 0x14140414, 0x14140C0C, 0x14140C3E,
    0x14141404, 0x14141414, 0x14141C3E, 0x14142404, 0x14142C2C, 0x141C040C, 0x141C0C04, 0x141C0C24,
    0x141C3E04, 0x141C3E24, 0x14241C2C, 0x14242C1C, 0x142C041C, 0x142C143E, 0x142C240C, 0x142C3E24,
    0x143E040C, 0x143E041C, 0x143E0C34, 0x143E242C, 0x1C04040C, 0x1C040C04, 0x1C040C14, 0x1C04140C,
    0x1C04141C, 0x1C042C04, 0x1C04342C, 0x1C043E14, 0x1C0C0404, 0x1C0C0414, 0x1C0C1404, 0x1C0C1C0C,
    0x1C0C2424, 0x1C0C2434, 0x1C14040C, 0x1C14041C, 0x1C140C04, 0x1C14142C, 0x1C142C14, 0x1C143E14,
    0x1C1C0C0C, 0x1C1C1C1C, 0x1C241C04, 0x1C24243E, 0x1C243E14, 0x1C2C0404, 0x1C2C0434, 0x1C2C1414,
    0x1C2C2C2C, 0x1C340C24, 0x1C341C34, 0x1C34341C, 0x1C3E1C1C, 0x1C3E3404, 0x24040424, 0x24040C3E,
    0x24041C2C, 0x24041C3E, 0x24042C1C, 0x24042C3E, 0x240C3E24, 0x24141404, 0x24141C3E, 0x24142404,
    0x24143404, 0x24143434, 0x241C043E, 0x241C242C, 0x24240424, 0x24242C0C, 0x24243424, 0x242C142C,
    0x242C241C, 0x242C3E04, 0x243E042C, 0x243E0C04, 0x243E0C14, 0x243E1C04, 0x2C040C14, 0x2C04240C,
    0x2C043E04, 0x2C0C0404, 0x2C0C0434, 0x2C0C1434, 0x2C0C2C2C, 0x2C140C24, 0x2C141C14, 0x2C143E14,
    0x2C1C0414, 0x2C1C2C1C, 0x2C240C04, 0x2C24141C, 0x2C24143E, 0x2C243E14, 0x2C2C0414, 0x2C2C1C0C,
    0x2C342C04, 0x2C3E1424, 0x2C3E2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340C140C,
    0x340C340C, 0x34140C3E, 0x34143424, 0x341C1C04, 0x341C1C34, 0x34242424, 0x342C042C, 0x342C2C14,
    0x34341C1C, 0x343E041C, 0x343E140C, 0x3E04041C, 0x3E04042C, 0x3E04043E, 0x3E040C04, 0x3E041C14,
    0x3E042C14, 0x3E0C1434, 0x3E0C2404, 0x3E140C14, 0x3E14242C, 0x3E142C14, 0x3E1C0404, 0x3E1C0C2C,
    0x3E1C1C1C, 0x3E1C3404, 0x3E24140C, 0x3E24240C, 0x3E2C0404, 0x3E2C0414, 0x3E2C1424, 0x3E341C04
};

    const unsigned char KSIGNS_IQ2XS[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
};

__launch_bounds__(32, 20)
extern "C" __global__ void gemv_iq3_xxs_residual(
    const unsigned char* __restrict__ A_iq3xxs, // [M, K/256 * 98] packed IQ3_XXS
    const float* __restrict__ x,                // [K]
    float* __restrict__ y,                      // [M]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int blocks_per_row = K / 256;
    const unsigned char* row_data = A_iq3xxs + (size_t)row * blocks_per_row * 98;

    float sum = 0.0f;

    for (int bi = 0; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 98;
        float d = half_to_float((unsigned short)(block[0] | (block[1] << 8)));
        const unsigned char* qs = block + 2;
        const unsigned char* aux = block + 66;
        const float* xb = x + bi * 256;

        #pragma unroll
        for (int ib32 = 0; ib32 < 8; ib32++) {
            unsigned int aux32 = (unsigned int)aux[4*ib32]
                               | ((unsigned int)aux[4*ib32+1] << 8)
                               | ((unsigned int)aux[4*ib32+2] << 16)
                               | ((unsigned int)aux[4*ib32+3] << 24);
            float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            int l = tid >> 3;
            int j = tid & 7;
            int grid_idx = (int)qs[ib32 * 8 + 2 * l + (j >= 4 ? 1 : 0)];
            unsigned int grid = IQ3XXS_GRID[grid_idx];
            int gb = (int)(int8_t)((grid >> (8 * (j & 3))) & 0xff);
            unsigned char signs = KSIGNS_IQ2XS[(aux32 >> (7 * l)) & 127];
            float w = db * (float)gb * ((signs & (1 << j)) ? -1.0f : 1.0f);
            sum += w * xb[ib32 * 32 + tid];
        }
    }

    // Warp shuffle reduction (5 steps, no shared memory)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) y[row] += sum;
}
