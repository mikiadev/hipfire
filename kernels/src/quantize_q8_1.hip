// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// q8_1 activation quantize for the I-quant decode GEMVs (GSQ-RCO
// decode-next prototype): x[K] f32 -> xs[K] int8 + xscales[K/32] f32.
// One group = 32 contiguous elements; scale = max(|x|)/127, round-to-nearest.
// Grid = K/32 blocks of 32 threads; warp-reduce the abs-max per group.

#include <hip/hip_runtime.h>

__launch_bounds__(32, 8)
extern "C" __global__ void quantize_q8_1(
    const float* __restrict__ x,        // [K]
    signed char* __restrict__ xs,       // [K]
    float* __restrict__ xscales,        // [K/32]
    int K) {

    const int g = blockIdx.x;
    const int base = g * 32;
    const int tid = threadIdx.x;
    if (base + tid >= K) return;

    const float v = x[base + tid];
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o));

    const float inv = (a > 0.0f) ? (127.0f / a) : 0.0f;
    const float scale = (a > 0.0f) ? (a / 127.0f) : 1.0f;
    int q = (int)roundf(v * inv);
    q = q > 127 ? 127 : (q < -128 ? -128 : q);
    xs[base + tid] = (signed char)q;
    if (tid == 0) xscales[g] = scale;
}