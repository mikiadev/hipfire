// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void gemv_f32(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K,
    float alpha, float beta
) {
    // float4-vectorized GEMV: 1 block per row, 256 threads, 4 independent
    // accumulator chains (ILP) + warp-shuffle reduce. Assumes K % 4 == 0 and
    // A/x 16B-aligned; misaligned views (sub-offset tensors) take the scalar
    // fallback path below.
    int row = blockIdx.x;
    if (row >= M) return;

    const float* arow_p = A + (size_t)row * (size_t)K;
    if ((((uintptr_t)arow_p | (uintptr_t)x) & 0xF) != 0 || (K & 3) != 0) {
        // scalar fallback (shared-memory block reduction)
        extern __shared__ float sdata[];
        float sum = 0.0f;
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            sum += arow_p[k] * x[k];
        }
        sdata[threadIdx.x] = sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            y[row] = (beta == 0.0f) ? alpha * sdata[0] : alpha * sdata[0] + beta * y[row];
        }
        return;
    }

    const float4* arow = (const float4*)arow_p;
    const float4* xrow = (const float4*)x;
    const int k4 = K >> 2;

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

    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    __shared__ float sred[8];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) sred[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < (blockDim.x >> 5)) ? sred[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
        if (lane == 0) {
            y[row] = (beta == 0.0f) ? alpha * acc : alpha * acc + beta * y[row];
        }
    }
}

extern "C" __global__ void gemv_f16(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ x,
    float* __restrict__ y,
    int M, int K,
    float alpha, float beta
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= M) return;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        sum += (float)A[row * K + k] * (float)x[k];
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        y[row] = (beta == 0.0f) ? alpha * sdata[0] : alpha * sdata[0] + beta * y[row];
    }
}
