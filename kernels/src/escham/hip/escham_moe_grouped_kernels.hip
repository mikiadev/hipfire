// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Grouped/batched escha-MoE FFN kernels for AR decode (batch=1).
//
// Instead of launching ~11 kernels per routed expert (mul + gemv + mul +
// silu + mul + gemv + mul + scaled-add), the decode path folds ALL scales
// into the cached folded weights at fill time:
//
//     W' = diag(rout) @ M_folded @ diag(s_in . rin)      (exact identity:
//          y = (M_folded @ (x . s_in . rin)) . rout  ==  W' @ x)
//
// and then runs the 8 routed experts through ONE grouped gemv per
// projection, one batched silu-mul, and one batched weighted accumulate —
// 4 launches per layer instead of ~88.
//
// Layouts (all row-major f32):
//   gate_up folded weight per expert:  [m_per=1024, k=2048]
//   down     folded weight per expert:  [m_per=2048, k=512]
//   gu_out:  [n_exp, 2*mi]   (gate half then up half per expert)
//   gated:   [n_exp, mi]
//   dn_out:  [n_exp, hidden]

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void escham_moe_grouped_gemv_f32(
    const float* const* w_ptrs,  // [n_exp] per-expert weight base pointers
    const float* __restrict__ x, // shared input (x_stride==0) or per-expert input base
    float* __restrict__ out,     // [n_exp * m_per]
    int m_per,
    int k,
    int n_exp,
    int x_stride)                // 0 = same x for every expert (gate_up);
                                 // >0 = input slice per expert at x + e*x_stride (down)
{
    // float4-vectorized GEMV: 1 block per row, 256 threads, 4 independent
    // accumulator chains (ILP) + warp-shuffle reduce (no shared-mem sync in
    // the hot loop). Assumes k % 4 == 0 and 16B-aligned base pointers;
    // misaligned views take the scalar fallback.
    int bid = blockIdx.x;
    int e = bid / m_per;
    int r = bid - e * m_per;
    const float* wrow = w_ptrs[e] + (size_t)r * (size_t)k;
    const float* xin = x + (x_stride > 0 ? e * x_stride : 0);
    if ((((uintptr_t)wrow | (uintptr_t)xin) & 0xF) != 0 || (k & 3) != 0) {
        float acc = 0.0f;
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            acc += wrow[i] * xin[i];
        }
        __shared__ float red[256];
        red[threadIdx.x] = acc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) out[(size_t)e * (size_t)m_per + (size_t)r] = red[0];
        return;
    }
    const float4* arow = (const float4*)wrow;
    const float4* xrow = (const float4*)xin;
    const int k4 = k >> 2;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int i = threadIdx.x; i < k4; i += blockDim.x) {
        float4 av = arow[i];
        float4 xv = xrow[i];
        acc0 += av.x * xv.x;
        acc1 += av.y * xv.y;
        acc2 += av.z * xv.z;
        acc3 += av.w * xv.w;
    }
    float acc = (acc0 + acc1) + (acc2 + acc3);

    // warp reduce
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    // cross-warp reduce (blockDim.x = 256 → 8 warps)
    __shared__ float sred[8];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) sred[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < (blockDim.x >> 5)) ? sred[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
        if (lane == 0) out[(size_t)e * (size_t)m_per + (size_t)r] = acc;
    }
}

// gated[e*mi + j] = silu(gu[e*2*mi + j]) * gu[e*2*mi + mi + j]
extern "C" __global__ void escham_moe_batched_silu_mul_f32(
    const float* __restrict__ gu,
    float* __restrict__ gated,
    int n_exp,
    int mi)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_exp * mi;
    if (tid >= total) return;
    int e = tid / mi;
    int j = tid - e * mi;
    const float* g = gu + (size_t)e * (size_t)(2 * mi);
    float gate = g[j];
    float up = g[mi + j];
    gated[tid] = (gate / (1.0f + __expf(-gate))) * up;
}

// expert_out[j] += sum_e weights[e] * dn[e*hidden + j]
extern "C" __global__ void escham_moe_batched_scaled_add_f32(
    float* __restrict__ expert_out,
    const float* __restrict__ dn,
    const float* __restrict__ weights,
    int n_exp,
    int hidden)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden) return;
    float acc = 0.0f;
    for (int e = 0; e < n_exp; ++e) {
        acc += weights[e] * dn[(size_t)e * (size_t)hidden + (size_t)tid];
    }
    expert_out[tid] += acc;
}

// W'[r*in_p + j] = W[r*in_p + j] * rout[r] * in_scale[j]  (in-place safe)
extern "C" __global__ void escham_apply_rowcol_scales_f32(
    float* __restrict__ w,
    const float* __restrict__ rout,
    const float* __restrict__ in_scale,
    int out_p,
    int in_p)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_p * in_p;
    if (idx >= total) return;
    int r = idx / in_p;
    int j = idx - r * in_p;
    w[idx] *= rout[r] * in_scale[j];
}

// ── FP16 folded-weight variants (halve the per-token FFN weight traffic) ──

// Convert a folded f32 [out_p, in_p] weight to f16 (after scale absorption,
// so all scales are baked in before conversion).
extern "C" __global__ void escham_f32_to_f16(
    const float* __restrict__ in,
    __half* __restrict__ out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

// Same contract as escham_moe_grouped_gemv_f32 but weights are __half (fp16).
extern "C" __global__ void escham_moe_grouped_gemv_f16(
    const __half* const* w_ptrs,
    const float* __restrict__ x,
    float* __restrict__ out,
    int m_per,
    int k,
    int n_exp,
    int x_stride)
{
    int bid = blockIdx.x;
    int e = bid / m_per;
    int r = bid - e * m_per;
    const __half2* arow = (const __half2*)(w_ptrs[e]) + (size_t)r * (size_t)(k >> 1);
    const float* xin = x + (x_stride > 0 ? e * x_stride : 0);
    const int k4 = k >> 2; // two half2 per 4-wide chunk

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int i = threadIdx.x; i < k4; i += blockDim.x) {
        __half2 ah0 = arow[2 * i];
        __half2 ah1 = arow[2 * i + 1];
        const float* xv = xin + 4 * i;
        acc0 += __half2float(ah0.x) * xv[0];
        acc1 += __half2float(ah0.y) * xv[1];
        acc2 += __half2float(ah1.x) * xv[2];
        acc3 += __half2float(ah1.y) * xv[3];
    }
    float acc = (acc0 + acc1) + (acc2 + acc3);

    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    __shared__ float sred[8];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) sred[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < (blockDim.x >> 5)) ? sred[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
        if (lane == 0) out[(size_t)e * (size_t)m_per + (size_t)r] = acc;
    }
}
