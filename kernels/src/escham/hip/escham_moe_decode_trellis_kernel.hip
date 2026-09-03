/*
 * HIP kernel: CORRECT ESCHAM trellis decode + 3INST codebook → F32 weights.
 *
 * This replaces the old (wrong) additive-delta codebook decode. The ESCHAM
 * format is an EXL3-style tail-biting trellis (see higgs tools/escha_ref.py,
 * exllamav3 pack.cu / codebook.cuh):
 *
 *   1. unpack_trellis: 16*K int16 codes per 16×16 tile form a circular
 *      256*K-bit stream. Thread t emits windows 2t and 2t+1 (16-bit windows
 *      at stride K, overlapping by 16-K bits).
 *   2. decode_3inst: each 16-bit window → x = (w*0xCBAC1FED & 0x8FFF8FFF) ^
 *      0x3B603B60, then the two fp16 halves are summed.
 *   3. tensor_core_perm: scatter the 256 scalars into the 16×16 tile.
 *
 * Output layout: [out_p, in_p] (transposed for direct GEMV), i.e.
 * out[out_row * in_p + in_col] = W_bare[in_col, out_row]. No Hadamard, no
 * scales — those are handled by the fold pass and the FFN gemv input scaling.
 *
 * Grid: [bi_max, bj_max, 1]. Block: [128, 1, 1] (one thread per window pair).
 */

#ifndef ESHAM_MOE_DECODE_TRELLIS_KERNEL_HIP
#define ESHAM_MOE_DECODE_TRELLIS_KERNEL_HIP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define ESCHA_TILE 16
#define MCG_MULT   0xCBAC1FEDu
#define LOP3_AND   0x8FFF8FFFu
#define LOP3_XOR   0x3B603B60u

__device__ __forceinline__ float escha_decode_3inst(uint16_t w) {
    uint32_t x = ((uint32_t)w * MCG_MULT) & LOP3_AND;
    x ^= LOP3_XOR;
    __half lo = __ushort_as_half((uint16_t)(x & 0xFFFFu));
    __half hi = __ushort_as_half((uint16_t)((x >> 16) & 0xFFFFu));
    return __half2float(__hadd(lo, hi));
}

extern "C" __global__ void escham_moe_decode_trellis_kernel(
    float* __restrict__ out,           // [out_p, in_p] f32, zero-initialized
    const int16_t* __restrict__ codes, // [bi_max, bj_max, 16*K] int16
    int k,                             // 2 or 3
    int bi_max,
    int bj_max,
    int in_p,
    int out_p
) {
    const int bi = blockIdx.x;
    const int bj = blockIdx.y;
    if (bi >= bi_max || bj >= bj_max) return;
    const int tid = threadIdx.x;   // 0..127

    const int total_codes = k * 16;
    const int n_words = (k * 256) / 32;   // uint32 words in the payload

    // Load the tile's codes as uint32 words into shared memory.
    extern __shared__ uint32_t s_pay[];
    const int16_t* tile = codes + ((size_t)(bi * bj_max + bj)) * total_codes;
    for (int i = tid; i < n_words; i += blockDim.x) {
        uint16_t lo = (uint16_t)tile[i * 2];
        uint16_t hi = (uint16_t)tile[i * 2 + 1];
        s_pay[i] = (uint32_t)lo | ((uint32_t)hi << 16);
    }
    __syncthreads();

    const int t = tid;   // window pair index, 0..127
    // Window geometry (from exllamav3 unpack_trellis_kernel):
    //   b0 = t*2k + k - 16 + 256k ; b2 = b0 + k + 16
    const int b0 = t * 2 * k + k - 16 + 256 * k;
    const int b2 = b0 + k + 16;
    const int i0 = b0 / 32;
    const int i1 = (b2 - 1) / 32;
    const int s1 = (i1 + 1) * 32 - b2;
    uint32_t a = s_pay[i0 % n_words];
    uint32_t b = s_pay[i1 % n_words];
    // full 32-bit funnel-shift result; w0 uses bits [s1+k .. s1+k+16) of the
    // 64-bit {a,b} — must NOT truncate to 16 bits before the >> k.
    uint64_t v64 = (((uint64_t)a << 32) | b) >> (uint32_t)s1;
    uint32_t w1 = (uint32_t)(v64 & 0xFFFFFFFFu);
    uint16_t w0 = (uint16_t)((w1 >> k) & 0xFFFFu);
    w1 = w1 & 0xFFFFu;

    float v0 = escha_decode_3inst(w0);
    float v1 = escha_decode_3inst(w1);

    // tensor_core_perm scatter (exl3_lib/quantize.py):
    //   for t in 0..32: r0=(t%4)*2, c0=t/4, rows={r0,r0+1,r0+8,r0+9},
    //                   cols={c0,c0+8}; perm[t*8 + j*4 + i] = rows[i]*16 + cols[j]
    // Window index within the 256: w_idx = 2t (even) / 2t+1 (odd). The perm
    // table maps w_idx -> (r, c) of the 16×16 tile.
    // We compute perm[w] directly for w in {2t, 2t+1}:
    //   from the loop: t0 = w/8, j = (w/4)%2, i = w%4
    //   r0 = (t0%4)*2, c0 = t0/4, rows={r0,r0+1,r0+8,r0+9}, cols={c0,c0+8}
    //   r = rows[i], c = cols[j]
    // Instead of recomputing, use the precomputed perm in constant memory
    // (see below) — but to keep this self-contained we recompute inline.
    for (int e = 0; e < 2; e++) {
        int w_idx = 2 * t + e;
        float v = e == 0 ? v0 : v1;
        int tq = w_idx / 8;
        int j = (w_idx / 4) % 2;
        int i = w_idx % 4;
        int r0 = (tq % 4) * 2;
        int c0 = tq / 4;
        int rows[4] = { r0, r0 + 1, r0 + 8, r0 + 9 };
        int cols[2] = { c0, c0 + 8 };
        int r = rows[i];
        int c = cols[j];
        int in_col = bi * ESCHA_TILE + r;
        int out_row = bj * ESCHA_TILE + c;
        if (in_col < in_p && out_row < out_p) {
            out[(size_t)out_row * in_p + in_col] += v;
        }
    }
}

#endif // ESHAM_MOE_DECODE_TRELLIS_KERNEL_HIP
