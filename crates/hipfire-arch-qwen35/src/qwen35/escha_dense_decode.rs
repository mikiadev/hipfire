// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Escha-W2 code-quant DENSE decode engine (AR per-token path).
//!
//! Every linear projection of Qwen3.8-27B-Escha-W2 stays in the int16
//! EXL3-trellis code. This module runs one coded projection as an in-kernel
//! decode-gemm:
//!
//!   u   = T128(x . in_scale)                 (in_scale = rin . s_in)
//!   acc = u @ decode(code)                   (analytic dep + codebook)
//!   y   = T128_col(acc) . out_scale          (out_scale = rout . s_out)
//!
//! Nothing is materialized to fp16/fp32 (a 17408x5120 down proj is ~360 MB),
//! so the decode happens per token inside the kernels. The batch-1 (gen)
//! kernels slice the IC reduction across blocks to fill the GPU.

use hip_bridge::HipError;
use hip_bridge::HipResult;
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;
use rdna_compute::escha_dense;

use super::weights::EschaDenseProjWeights;

/// Run one Escha-coded dense projection: y[out] = decode(x).
/// `x` is the raw (pre-rotation) activation vector [in].
/// `y` is a caller scratch [out] that receives the result.
pub fn escha_dense_decode_proj(
    gpu: &mut Gpu,
    proj: &EschaDenseProjWeights,
    x: &GpuTensor,
    y: &GpuTensor,
) -> HipResult<()> {
    let ic = proj.in_p;
    let oc = proj.out_p;
    let nit = ic / 16;
    let n_slices = escha_dense::escha_dense_n_slices(nit, oc);

    // scratch (per-call; freed after). The GPU pool reuses the same VMM arena.
    let u = gpu.alloc_tensor(&[ic], DType::F32)?;
    let partial = gpu.alloc_tensor(&[n_slices * oc], DType::F32)?;

    let r = escha_dense::escha_dense_decode_gemv(
        gpu, &proj.code, &proj.in_scale, &proj.out_scale, x, &u, &partial, y,
    );
    let _ = gpu.free_tensor(u);
    let _ = gpu.free_tensor(partial);
    r
}

/// Host reference for one dense projection decode: y[out] computed by the
/// reference (trellis) decode with full Hadamard semantics — used by tests.
/// Mirrors llama.cpp's CPU `ggml_compute_forward_escha_mul_mat` plus the
/// s_in/s_out folds.
#[allow(clippy::too_many_arguments)]
pub fn escha_dense_decode_proj_host(
    code: &[i16],
    k: usize,
    in_p: usize,
    out_p: usize,
    in_scale: &[f32],
    out_scale: &[f32],
    x: &[f32],
) -> Vec<f32> {
    use super::escham_decode::{apply_t128_host, decode_tiles};
    // w_bare [in_p, out_p]; decode_tiles returns [in_p, out_p]
    let w_bare = decode_tiles(code, k, in_p, out_p);
    // u = T128(x . in_scale)
    let x_scaled: Vec<f32> = (0..in_p).map(|i| x[i] * in_scale[i]).collect();
    let u = apply_t128_host(&x_scaled);
    // acc = u @ w_bare
    let mut acc = vec![0.0f32; out_p];
    for o in 0..out_p {
        let mut s = 0.0f32;
        for i in 0..in_p {
            s += u[i] * w_bare[i * out_p + o];
        }
        acc[o] = s;
    }
    // y = T128_col(acc) . out_scale — per 128-col block
    let yt = apply_t128_host(&acc);
    (0..out_p).map(|o| yt[o] * out_scale[o]).collect()
}

/// GPU-side numerical check of one projection: downloads and compares against
/// the host reference. Returns (max_abs, rel) — for diagnostics/tests.
#[allow(clippy::too_many_arguments)]
pub fn escha_dense_check_proj(
    gpu: &mut Gpu,
    proj: &EschaDenseProjWeights,
    x_host: &[f32],
    code_host: &[i16],
    in_scale_host: &[f32],
    out_scale_host: &[f32],
    label: &str,
) -> HipResult<(f32, f32)> {
    let ic = proj.in_p;
    let oc = proj.out_p;
    let x = gpu.upload_f32(x_host, &[ic])?;
    let y = gpu.alloc_tensor(&[oc], DType::F32)?;
    escha_dense_decode_proj(gpu, proj, &x, &y)?;
    let got = gpu.download_f32(&y)?;
    let want = escha_dense_decode_proj_host(
        code_host,
        proj.k as usize,
        ic,
        oc,
        in_scale_host,
        out_scale_host,
        x_host,
    );
    let mut max_abs = 0.0f32;
    let mut scale = 0.0f32;
    for i in 0..oc {
        let d = (got[i] - want[i]).abs();
        max_abs = max_abs.max(d);
        scale = scale.max(want[i].abs());
    }
    eprintln!("[escha-dense] {label}: max_abs={max_abs:.6} rel={:.6}", max_abs / (scale + 1e-9));
    let _ = gpu.free_tensor(y);
    let _ = gpu.free_tensor(x);
    Ok((max_abs, max_abs / (scale + 1e-9)))
}

/// `HipError` convenience for decode-arming.
pub fn err(msg: &str) -> HipError {
    HipError::new(0, msg)
}

/// Validate that every projection dim is 16/128-aligned and consistent.
pub fn escha_dense_validate_proj(proj: &EschaDenseProjWeights, label: &str) -> HipResult<()> {
    if proj.in_p % 16 != 0 || proj.out_p % 16 != 0 {
        return Err(err(&format!(
            "{label}: dims {}x{} not 16-aligned",
            proj.in_p, proj.out_p
        )));
    }
    if proj.in_p % 128 != 0 || proj.out_p % 128 != 0 {
        return Err(err(&format!(
            "{label}: dims {}x{} not 128-aligned (WHT is blockwise-128)",
            proj.in_p, proj.out_p
        )));
    }
    Ok(())
}
