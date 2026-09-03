/*
 * Device utility functions for escham MoE kernels (HIP port).
 *
 * Mirrors the PTX-level intrinsics the original CUDA kernels relied on:
 * half<->float conversions, warp-level butterfly shuffles (shfl.bfly),
 * and the SWiGLU sigmoid approximation.
 *
 * HIP notes:
 *   - __shfl_xor(val, mask, width) implements the PTX shfl.sync.bfly.b32.
 *     bfly with width=W and mask=M == xor with mask=M (HIP uses full 32-lane
 *     warp, so width=32).
 *   - __float2half / __half2float are the half<->float intrinsics.
 *   - hexp2 is available in HIP; we use it for the SWiGLU sigmoid.
 */
#ifndef ESHAM_DEVICE_UTILS_HIP
#define ESHAM_DEVICE_UTILS_HIP

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

/* ------------------------------------------------------------------ */
/* half <-> float conversion helpers                                   */
/* PTX: cvt.f32.f16   /   cvt.rn.f16.f32                            */
/* ------------------------------------------------------------------ */

/* PTX cvt.f32.f16 %f, %rs  ->  float = half_to_float(h) */
__device__ __forceinline__ float h2f(__half h) {
  return __half2float(h);
}

/* PTX cvt.rn.f16.f32 %rs, %f  ->  half = round_to_nearest(float) */
__device__ __forceinline__ __half f2h(float f) {
  return __float2half(f);
}

/* ------------------------------------------------------------------ */
/* Warp-level butterfly shuffle (PTX: shfl.sync.bfly.b32)             */
/* ------------------------------------------------------------------ */

/* Butterfly shuffle on a 32-bit value.
 * PTX: shfl.sync.bfly.b32 %r_dst, %r_src, %mask, %width, -1
 * HIP: __shfl_xor(val, mask, 32)
 *
 * The PTX bfly with width=31 and mask=1/2/4/8/16 maps to __shfl_xor
 * with mask=1/2/4/8/16 and width=32 (HIP uses full warp width).
 */
__device__ __forceinline__ uint32_t bfly_xor_u32(uint32_t val, uint32_t mask) {
  return __shfl_xor(val, mask, 32);
}

/* f32 butterfly (reinterprets float as uint for shuffle, then back) */
__device__ __forceinline__ float bfly_xor_f32(float val, uint32_t mask) {
  uint32_t bits = __float_as_uint(val);
  uint32_t shuf = __shfl_xor(bits, mask, 32);
  return __uint_as_float(shuf);
}

/* ------------------------------------------------------------------ */
/* SWiGLU sigmoid approximation                                      */
/* PTX sequence:                                                     */
/*   mul.f32  %f445, %f437, 0fBFB8AA3B  ; x * (-1/ln2)                */
/*   ex2.approx.ftz.f32 %f446, %f445    ; exp2(x/ln2) = exp(x)        */
/*   add.f32  %f447, %f446, 1.0          ; exp(x) + 1                 */
/*   div.approx.ftz.f32 %f438, %f437, %f447 ; x / (exp(x)+1) = sigmoid */
/* 0fBFB8AA3B = -1.44269504f ≈ -1/ln(2)                              */
/* ------------------------------------------------------------------ */

__device__ __forceinline__ float sigmoid_appx(float x) {
  constexpr float NEG_LOG2_F = -1.44269504f;  // 0fBFB8AA3B
  float y = x * NEG_LOG2_F;                    // x / ln(2) negated
  float e = exp2f(y);                        // exp2(x/ln2) = exp(x)
  return x / (e + 1.0f);
}

/* ------------------------------------------------------------------ */
/* Constants extracted from PTX                                      */
/* ------------------------------------------------------------------ */
constexpr float kScale_had        = 0.0883883f;   // 0f3DB504F3 = sqrt(2)/16
constexpr float kOne              = 1.0f;          // 0f3F800000
constexpr float kNegOne           = -1.0f;         // 0fBF800000
constexpr float kZero             = 0.0f;          // 0f00000000

#endif  // ESHAM_DEVICE_UTILS_HIP
