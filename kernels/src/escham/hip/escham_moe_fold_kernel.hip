/*
 * HIP kernels: T128 fold of the transposed codebook-decode output + vec mul.
 *
 * The ESCHAM deploy chain computes, per expert:
 *     y = s_out . T128((T128(x . s_in . rin) @ W) . rout)      (T128 = 128-pt WHT)
 *
 * With the two T128s folded into the weight (T128 orthogonal, symmetric):
 *     y = s_out . (M_folded @ (x . s_in . rin))
 * where M = W^T is the DECODE KERNEL output layout [out_p, in_p] and
 *     M_folded = T128_out @ (M . rout_rows) @ T128_in
 * i.e. a block-diagonal 128-pt WHT over the OUTPUT dim (rows), a per-row
 * scale by rout, and a block-diagonal 128-pt WHT over the INPUT dim (cols).
 *
 * Both WHT passes use the same butterfly pattern as the escha EPI family:
 * 4-element local 2x2 Hadamard per thread + 32-way warp butterfly
 * (masks 1,2,4,8,16), scaled by sqrt(2)/16 = 1/sqrt(128) (normalized WHT).
 *
 * Layout: M is row-major [out_p, in_p] (row = output feature, col = input
 * feature). out_p and in_p must be multiples of 128.
 */
#ifndef ESHAM_MOE_FOLD_KERNEL_HIP
#define ESHAM_MOE_FOLD_KERNEL_HIP

#include "esham_device_utils.hip"

/* Pass 1: out1 = T128_out @ (M . rout)   (WHT over the OUTPUT dim, rows) */
extern "C" __global__
void escham_fold_t128_rows_kernel(
    float* __restrict__ out1,        // [out_p, in_p]
    const float* __restrict__ m,     // [out_p, in_p] decode output (W^T)
    const float* __restrict__ rout,  // [out_p]
    int out_p,
    int in_p)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t col = blockIdx.x;   // one input column
  const uint32_t rb  = blockIdx.y;   // 128-row block of the output dim

  uint32_t row0 = rb * 128 + tid * 4;

  float p[4];
  #pragma unroll
  for (int e = 0; e < 4; e++) {
    uint32_t row = row0 + e;
    p[e] = m[(size_t)row * (size_t)in_p + col] * rout[row];
  }

  // 2x2 Hadamard add/sub tree
  float s0p = p[0] + p[1];
  float s0m = p[0] - p[1];
  float s1p = p[2] + p[3];
  float s1m = p[2] - p[3];
  float v[4];
  v[0] = s0p + s1p;
  v[1] = s0m + s1m;
  v[2] = s0p - s1p;
  v[3] = s0m - s1m;

  // 32-way warp butterfly (5 stages)
  #pragma unroll
  for (uint32_t mask : {1u, 2u, 4u, 8u, 16u}) {
    #pragma unroll
    for (int e = 0; e < 4; e++) {
      float partner = bfly_xor_f32(v[e], mask);
      float sign    = (tid & mask) ? kNegOne : kOne;
      v[e] = v[e] * sign + partner;
    }
  }

  #pragma unroll
  for (int e = 0; e < 4; e++) {
    out1[(size_t)(row0 + e) * (size_t)in_p + col] = v[e] * kScale_had;
  }
}

/* Pass 2: out2 = out1 @ T128_in   (WHT over the INPUT dim, cols) */
extern "C" __global__
void escham_fold_t128_cols_kernel(
    float* __restrict__ out2,        // [out_p, in_p]
    const float* __restrict__ out1,  // [out_p, in_p]
    int out_p,
    int in_p)
{
  const uint32_t tid = threadIdx.x;
  const uint32_t cb  = blockIdx.x;   // 128-col block of the input dim
  const uint32_t row = blockIdx.y;

  uint32_t col = cb * 128 + tid * 4;

  float p[4];
  #pragma unroll
  for (int e = 0; e < 4; e++) {
    p[e] = out1[(size_t)row * (size_t)in_p + col + e];
  }

  float s0p = p[0] + p[1];
  float s0m = p[0] - p[1];
  float s1p = p[2] + p[3];
  float s1m = p[2] - p[3];
  float v[4];
  v[0] = s0p + s1p;
  v[1] = s0m + s1m;
  v[2] = s0p - s1p;
  v[3] = s0m - s1m;

  #pragma unroll
  for (uint32_t mask : {1u, 2u, 4u, 8u, 16u}) {
    #pragma unroll
    for (int e = 0; e < 4; e++) {
      float partner = bfly_xor_f32(v[e], mask);
      float sign    = (tid & mask) ? kNegOne : kOne;
      v[e] = v[e] * sign + partner;
    }
  }

  #pragma unroll
  for (int e = 0; e < 4; e++) {
    out2[(size_t)row * (size_t)in_p + col + e] = v[e] * kScale_had;
  }
}

/* Elementwise out = a . b (used to scale gemv inputs by s_in . rin) */
extern "C" __global__
void escham_mul_vec_f32_kernel(
    float* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int n)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (uint32_t)n) out[i] = a[i] * b[i];
}

#endif  // ESHAM_MOE_FOLD_KERNEL_HIP
