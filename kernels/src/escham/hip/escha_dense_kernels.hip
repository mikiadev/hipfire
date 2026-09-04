/*
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 Kaden Schutt
 * hipfire — see LICENSE and NOTICE in the project root.
 *
 * Escha-W2 code-quant DENSE decode kernels (Qwen3.8-27B-Escha-W2).
 *
 * Every linear projection stays in the int16 EXL3-trellis code; the decode-gemm
 * decodes each 16x16 tile IN-KERNEL each time it is needed (no materialized
 * weight — a full 17408x5120 down projection is ~360 MB f32, not viable
 * per-token). Port of llama.cpp's dense escha kernels
 * (`escha_rotate_in_dense` / `escha_matmul_dense` / `escha_finalize_dense`)
 * to HIP, with the export's s_in/s_out folded into the scale vectors at load:
 *
 *   in_scale[i]  = rin[i]  * s_in[i]
 *   out_scale[c] = rout[c] * s_out[c]
 *
 * The op computes, per row:
 *   u      = T128(x . in_scale)                       (rotate_in)
 *   acc    = u @ decode(code)                         (matmul)
 *   y      = T128_col(acc) . out_scale                (finalize)
 *
 * T128 is a normalized blockwise-128 Sylvester Hadamard. Only the rotation
 * directions differ from llama.cpp's names because we pre-fold the scales;
 * the decode itself is identical.
 *
 * Decode: each tile's payload is 16*K int16 = 8*K u32 words = 256*K bits of a
 * circular bit stream. The 16 bits forming the codebook index of weight (r,c)
 * are 16 cyclically-consecutive bits starting at
 *
 *   pi(r) = (r&1) | (((r>>3)&1)<<1) | (((r>>1)&3)<<3)
 *   t     = pi(r) + 32*cc + 4*(cc>>3)          (cc = c mod 16)
 *   s     = ((32-K) - K*t) mod (256K)
 *
 * (verified exact by llama.cpp against the shipped dep tables). The start word
 * is w0 = (NW - (s>>5)) mod NW (NW = 8*K words); the 16-bit window is the
 * funnel-shift of the pair (word w0, word w0-1) as the 64-bit hi:lo by (s&31),
 * low 16 bits; then escha_codebook(idx) = fp16add(lo16, hi16) of
 *   x = ((idx * 0xCBAC1FED) & 0x8FFF8FFF) ^ 0x3B603B60.
 *
 * Payload words are held in shared as OVERLAPPING pairs so one LDS.64 yields
 * the (hi,lo) pair for the funnel: pay[w].y = word w, pay[w].x = word (w+1 mod
 * NW), exactly like llama.cpp. Code is stored [in/16, out/16, 16*K] i16.
 */

#ifndef ESCHA_DENSE_KERNELS_HIP
#define ESCHA_DENSE_KERNELS_HIP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define ESCHA_DTILE     16
#define ESCHA_DNT      128     // threads per block, one per output column
#define ESCHA_DGROUPS  (ESCHA_DNT / ESCHA_DTILE)  // 8
#define ESCHA_DMAX_W    24     // uint32 words per payload (K=3: 24)
#define ESCHA_DROT_CHUNK 2048  // rotate staged u chunk (8 KB smem)

// Codebook A: escha_codebook(idx) as float. The fp16 add must stay in fp16.
__device__ __forceinline__ float escha_dense_codebook(uint32_t idx) {
    uint32_t x = idx * 0xcbac1fedu;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    __half lo = __ushort_as_half((uint16_t)(x & 0xFFFFu));
    __half hi = __ushort_as_half((uint16_t)((x >> 16) & 0xFFFFu));
    return __half2float(__hadd(lo, hi));
}

// in-place normalized Sylvester-Hadamard over each block of 128 of a float
// buffer of `n` elements. `n` must be a multiple of 128 (dense projections
// are all 128-aligned). Ported from llama.cpp escha_hadamard_128.
__device__ __forceinline__ void escha_dense_hadamard_128(
    float* v, int n, int tid, int nt) {
    for (int len = 1; len < 128; len <<= 1) {
        for (int idx = tid; idx < (n / 128) * 64; idx += nt) {
            const int blk = idx / 64;
            const int j   = idx % 64;
            const int i   = (j / len) * (2 * len) + (j % len);
            float* b = v + blk * 128 + i;
            const float a0 = b[0];
            const float a1 = b[len];
            b[0]   = a0 + a1;
            b[len] = a0 - a1;
        }
        __syncthreads();
    }
    const float scale = rsqrtf(128.0f);
    for (int i = tid; i < n; i += nt) {
        v[i] *= scale;
    }
    __syncthreads();
}

// u[row] = T128(x[row] . in_scale), chunked (IC up to 17408 exceeds the 48 KB
// default smem). One block per row, 256 threads.
extern "C" __global__ void escha_dense_rotate_in_kernel(
    const float* __restrict__ in_scale, // [IC]
    const float* __restrict__ x,        // [IC]
    float* __restrict__ u,              // [IC]
    int IC) {
    __shared__ float s_u[ESCHA_DROT_CHUNK];
    const int tid = threadIdx.x;
    for (int off = 0; off < IC; off += ESCHA_DROT_CHUNK) {
        const int n = min(ESCHA_DROT_CHUNK, IC - off);
        for (int i = tid; i < n; i += blockDim.x) {
            s_u[i] = x[off + i] * in_scale[off + i];
        }
        __syncthreads();
        escha_dense_hadamard_128(s_u, n, tid, blockDim.x);
        __syncthreads();
        for (int i = tid; i < n; i += blockDim.x) {
            u[off + i] = s_u[i];
        }
        __syncthreads();
    }
}

// analytic dep start word for a fixed column lane cc (r=0), as llama.cpp.
__device__ __forceinline__ int escha_dense_pi(int r) {
    return (r & 1) | (((r >> 3) & 1) << 1) | (((r >> 1) & 3) << 3);
}

// partial[row][ocb*128+tid] = sum over this slice's IC tiles of u . decode(code)
// One row per block; IC reduction sliced across blockIdx.z so batch-1 fills the
// device. Port of llama.cpp `escha_matmul_dense<K, R=1>` (generation path).
//
// Empirical pin (examples/pin_funnel.rs on real model tiles, both K): the exact
// codebook index of weight (r,c) is the funnel (hi = payload word w0-1, lo =
// payload word w0) >> (sp & 31), low 16 bits, where w0 = (NW - (sp>>5)) mod NW
// and the payload words are the tile's 16*K int16 as little-endian u32. We keep
// the tile's NW words in plain shared memory and do two reads per weight.
extern "C" __global__ void escha_dense_decode_gemv_kernel(
    const int16_t* __restrict__ code,    // [in/16, out/16, 16*K] i16
    const float* __restrict__ u,         // [IC] pre-rotated input
    float* __restrict__ partial,         // [n_slices * OC]
    int IC,
    int OC,
    int n_slices,
    int K) {
    extern __shared__ char s_raw[];
    uint32_t* s_pay = (uint32_t*)s_raw;               // [ESCHA_DGROUPS][NW words]
    float* s_u     = (float*)(s_pay + ESCHA_DGROUPS * (8 * K));  // [tiles_max*16]

    const int NW  = 8 * K;
    const int NB  = 32 * NW;
    const int n_wd = (16 * K) / 2;             // u32 words per tile

    const int nit = IC / ESCHA_DTILE;
    const int nct = OC / ESCHA_DTILE;

    const int tid  = threadIdx.x;
    const int ocb  = blockIdx.y;
    const int sl   = blockIdx.z;
    const int lo   = (int)(((int64_t)nit * sl) / n_slices);
    const int hi   = (int)(((int64_t)nit * (sl + 1)) / n_slices);
    const int tiles = hi - lo;
    if (tiles <= 0) return;

    const int grp = tid / ESCHA_DTILE;
    const int cc  = tid % ESCHA_DTILE;
    const int tj  = ocb * ESCHA_DGROUPS + grp;   // output tile column

    // this block's u slice is small (tiles*16); stage it once (R == 1)
    const float* u_row = u + (int64_t)lo * ESCHA_DTILE;
    for (int j = tid; j < tiles * ESCHA_DTILE; j += ESCHA_DNT) {
        s_u[j] = u_row[j];
    }

    uint32_t* pay = s_pay + grp * NW;

    // start bit for this column lane, r = 0
    int s0 = ((32 - K) - K * (32 * cc + 4 * (cc >> 3))) % NB;
    if (s0 < 0) s0 += NB;

    float acc = 0.0f;
    __syncthreads();

    for (int ti = lo; ti < hi; ++ti) {
        const int16_t* tile =
            code + ((size_t)(ti * nct + tj)) * (size_t)(16 * K);
        // load this tile's payload words (plain, little-endian u32)
        for (int w = cc; w < n_wd; w += ESCHA_DTILE) {
            pay[w] = ((const uint32_t*)tile)[w];
        }
        __syncthreads();

        const float* uu = s_u + (int64_t)(ti - lo) * ESCHA_DTILE;
        for (int r = 0; r < ESCHA_DTILE; ++r) {
            // K*pi(r) <= 81 < NB, so one conditional add restores range
            int sp = s0 - K * escha_dense_pi(r);
            if (sp < 0) sp += NB;
            const int g0 = sp >> 5;
            const int w0 = g0 ? (NW - g0) : 0;
            const int wh = (w0 + NW - 1) % NW;   // hi word = w0 - 1 (circular)

            uint32_t hi = pay[wh];
            uint32_t lo = pay[w0];
            uint32_t idx =
                ((uint64_t)hi << 32 | (uint64_t)lo) >> (sp & 31);
            idx &= 0xffffu;
            acc += uu[r] * escha_dense_codebook(idx);
        }
        __syncthreads();
    }

    partial[((int64_t)sl * 1 + 0) * OC + ocb * ESCHA_DNT + tid] = acc;
}

// finalize: sum the slices, WHT over the 128-col group, scale by out_scale.
// Works for both decode (n_rows=1, partial is [n_slices * OC]) and prefill
// (n_rows > 1, partial is [n_slices * n_rows * OC]).
extern "C" __global__ void escha_dense_finalize_kernel(
    const float* __restrict__ out_scale, // [OC]
    const float* __restrict__ partial,   // [n_slices * n_rows * OC] (or [n_slices * OC] when n_rows=1)
    float* __restrict__ y,               // [n_rows * OC] (or [OC] when n_rows=1)
    int OC,
    int n_rows,
    int n_slices) {
    __shared__ float s_acc[ESCHA_DNT];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int c   = blockIdx.y * ESCHA_DNT + tid;

    // Compute the base offset for this row in the partial array
    int64_t row_base;
    if (n_rows <= 1) {
        // Decode path: partial is [n_slices * OC], no row offset
        row_base = 0;
    } else {
        // Prefill path: partial is [n_slices * n_rows * OC]
        row_base = (int64_t)row * OC;
    }

    float sum = 0.0f;
    if (c < OC) {
        for (int s = 0; s < n_slices; ++s) {
            sum += partial[((int64_t)s) * (n_rows <= 1 ? OC : (int64_t)n_rows * OC) + row_base + c];
        }
    }
    s_acc[tid] = sum;
    __syncthreads();

    escha_dense_hadamard_128(s_acc, ESCHA_DNT, tid, ESCHA_DNT);

    if (c < OC) {
        if (n_rows <= 1) {
            y[c] = s_acc[tid] * out_scale[c];
        } else {
            y[row * OC + c] = s_acc[tid] * out_scale[c];
        }
    }
}

// ===========================================================================
// Dense prefill kernels (batched, multi-row)
//
// Same codec, same rotations, and SAME payload/funnel-shift convention as
// the decode path above — plain uint32_t payload array (not uint2 pairs),
// hi:lo assembled as ((word_w0-1) << 32 | word_w0) to match the verified
// decode kernel bit-for-bit.
//
//   escha_dense_matmul_prefill_kernel  scalar FMA, R rows per block (R=64
//                                     prefill, R=1 gen). Decoded weights go to
//                                     shared memory for reuse across rows.
//
// All write the same partial layout [n_slices][n_rows][OC] and share
// escha_dense_finalize_kernel.
// ===========================================================================

#define ESCHA_DROWS_DENSE       64   // rows per block on the prefill path
#define ESCHA_DROWS_DENSE_GEN    1   // rows per block on the gen path
#define ESCHA_DGEN_MAX_ROWS     16   // at or below this, use gen instantiation

// u[row] = T128(x[row] * in_scale) -- batched, chunked (IC up to 17408).
// One block per row, 256 threads.
extern "C" __global__ void escha_dense_rotate_in_dense_kernel(
    const float* __restrict__ in_scale, // [IC]
    const float* __restrict__ x,        // [n_rows, IC]
    float* __restrict__ u,              // [n_rows, IC]
    int IC,
    int n_rows) {
    __shared__ float s_u[ESCHA_DROT_CHUNK];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const float* x_row = x + (int64_t)row * IC;
    float* dst = u + (int64_t)row * IC;
    for (int off = 0; off < IC; off += ESCHA_DROT_CHUNK) {
        const int n = min(ESCHA_DROT_CHUNK, IC - off);
        for (int i = tid; i < n; i += blockDim.x) {
            s_u[i] = x_row[off + i] * in_scale[off + i];
        }
        __syncthreads();
        escha_dense_hadamard_128(s_u, n, tid, blockDim.x);
        __syncthreads();
        for (int i = tid; i < n; i += blockDim.x) {
            dst[off + i] = s_u[i];
        }
        __syncthreads();
    }
}

// partial[slice][row][c] = sum over this slice's input tiles of u . decode(code)
// R rows per block. Decoded weights go to shared memory for reuse across rows.
// R is a runtime parameter (1 for gen, 64 for prefill).
extern "C" __global__ void escha_dense_matmul_prefill_kernel(
    const int16_t* __restrict__ code,    // [in/16, out/16, 16*K] i16
    const float* __restrict__ u,         // [n_rows, IC]
    float* __restrict__ partial,         // [n_slices * n_rows * OC]
    int IC,
    int OC,
    int n_rows,
    int n_slices,
    int K,
    int R) {
    // Same payload convention as escha_dense_decode_gemv_kernel:
    // plain uint32_t array, funnel-shift (hi=pay[w0-1] << 32 | lo=pay[w0]) >> (sp&31).
    extern __shared__ char s_raw[];
    uint32_t* s_pay = (uint32_t*)s_raw;
    float* s_u = (float*)(s_pay + ESCHA_DGROUPS * (8 * K));

    const int NW = 8 * K;
    const int NB = 32 * NW;
    const int nit = IC / ESCHA_DTILE;
    const int nct = OC / ESCHA_DTILE;
    const int n_wd = (16 * K) / 2;

    const int tid = threadIdx.x;
    const int start = blockIdx.x * R;
    const int nrow = min(R, n_rows - start);
    const int sl = blockIdx.z;
    const int lo = (int)(((int64_t)nit * sl) / n_slices);
    const int hi = (int)(((int64_t)nit * (sl + 1)) / n_slices);

    const int grp = tid / ESCHA_DTILE;
    const int cc = tid % ESCHA_DTILE;
    const int tj = blockIdx.y * ESCHA_DGROUPS + grp;

    uint32_t* pay = s_pay + grp * NW;

    int s0 = ((32 - K) - K * (32 * cc + 4 * (cc >> 3))) % NB;
    if (s0 < 0) s0 += NB;

    float acc[64];
    for (int m = 0; m < R; ++m) acc[m] = 0.0f;

    if (R == 1) {
        const float* u_row = u + (int64_t)start * IC + (int64_t)lo * ESCHA_DTILE;
        const int n_stage = (hi - lo) * ESCHA_DTILE;
        for (int j = tid; j < n_stage; j += ESCHA_DNT) {
            s_u[j] = u_row[j];
        }
    }
    __syncthreads();

    for (int ti = lo; ti < hi; ++ti) {
        if (R != 1) {
            for (int j = tid; j < R * ESCHA_DTILE; j += ESCHA_DNT) {
                const int m = j / ESCHA_DTILE;
                const int r = j % ESCHA_DTILE;
                s_u[j] = m < nrow ? u[(int64_t)(start + m) * IC + ti * ESCHA_DTILE + r] : 0.0f;
            }
        }

        const uint32_t* src = (const uint32_t*)(code + (int64_t)(ti * nct + tj) * (16 * K));
        for (int w = cc; w < n_wd; w += ESCHA_DTILE) {
            pay[w] = src[w];
        }
        if (R == 1) { __syncwarp(); } else { __syncthreads(); }

        const float* uu = s_u + (ti - lo) * ESCHA_DTILE;

        for (int r = 0; r < ESCHA_DTILE; ++r) {
            int sp = s0 - K * escha_dense_pi(r);
            if (sp < 0) sp += NB;
            const int g0 = sp >> 5;
            const int w0 = g0 ? (NW - g0) : 0;
            const int wh = (w0 + NW - 1) % NW;

            uint32_t hi = pay[wh];
            uint32_t lo = pay[w0];
            uint32_t idx = ((uint64_t)hi << 32 | (uint64_t)lo) >> (sp & 31);
            idx &= 0xffffu;
            const float wv = escha_dense_codebook(idx);
            if (R == 1) {
                acc[0] += uu[r] * wv;
            } else {
                for (int m = 0; m < R; ++m) {
                    acc[m] += s_u[m * ESCHA_DTILE + r] * wv;
                }
            }
        }
        if (R == 1) { __syncwarp(); } else { __syncthreads(); }
    }

    for (int m = 0; m < nrow; ++m) {
        partial[((int64_t)sl * n_rows + start + m) * OC + blockIdx.y * ESCHA_DNT + tid] = acc[m];
    }
}

#endif  // ESCHA_DENSE_KERNELS_HIP
