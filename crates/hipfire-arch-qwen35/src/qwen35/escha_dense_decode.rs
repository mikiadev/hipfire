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
    use crate::escham_decode::{apply_t128_host, decode_tiles};
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
    // dump first 8 values each for eyeball
    eprintln!("  got  [0..8] = {:?}", &got[..8.min(oc)]);
    eprintln!("  want [0..8] = {:?}", &want[..8.min(oc)]);
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

/// Debug: compare the in-kernel rotate `u` against the host T128 of x.in_scale.
pub fn escha_dense_check_rotate(
    gpu: &mut Gpu,
    proj: &EschaDenseProjWeights,
    x_host: &[f32],
    in_scale_host: &[f32],
) -> HipResult<f32> {
    use crate::escham_decode::apply_t128_host;
    let ic = proj.in_p;
    let x = gpu.upload_f32(x_host, &[ic])?;
    let u = gpu.alloc_tensor(&[ic], DType::F32)?;
    rdna_compute::escha_dense::escha_dense_rotate_in(gpu, &proj.in_scale, &x, &u)?;
    let got = gpu.download_f32(&u)?;
    let xs: Vec<f32> = (0..ic).map(|i| x_host[i] * in_scale_host[i]).collect();
    let want = apply_t128_host(&xs);
    let mut max_abs = 0.0f32;
    for i in 0..ic {
        max_abs = max_abs.max((got[i] - want[i]).abs());
    }
    eprintln!("[escha-dense] rotate max_abs={max_abs:.6}");
    eprintln!("  got  [0..8] = {:?}", &got[..8.min(ic)]);
    eprintln!("  want [0..8] = {:?}", &want[..8.min(ic)]);
    let _ = gpu.free_tensor(u);
    let _ = gpu.free_tensor(x);
    Ok(max_abs)
}

/// Debug: run rotate + decode-gemm only; sum slices on host; compare with
/// host `u @ W_bare` where W_bare = decode_tiles([in,out]). No finalize WHT.
#[allow(clippy::too_many_arguments)]
pub fn escha_dense_check_decode_stage(
    gpu: &mut Gpu,
    proj: &EschaDenseProjWeights,
    x_host: &[f32],
    code_host: &[i16],
    in_scale_host: &[f32],
) -> HipResult<f32> {
    use crate::escham_decode::{apply_t128_host, decode_tiles};
    let ic = proj.in_p;
    let oc = proj.out_p;
    let nit = ic / 16;
    let n_slices = rdna_compute::escha_dense::escha_dense_n_slices(nit, oc);

    let x = gpu.upload_f32(x_host, &[ic])?;
    let u = gpu.alloc_tensor(&[ic], DType::F32)?;
    let partial = gpu.alloc_tensor(&[n_slices * oc], DType::F32)?;
    rdna_compute::escha_dense::escha_dense_rotate_in(gpu, &proj.in_scale, &x, &u)?;
    // launch decode gemv only
    let r = rdna_compute::escha_dense::escha_dense_decode_gemv_stage(
        gpu, &proj.code, &u, &partial, ic, oc, n_slices, proj.k as i32,
    );
    if let Err(e) = r {
        let _ = gpu.free_tensor(x);
        let _ = gpu.free_tensor(u);
        let _ = gpu.free_tensor(partial);
        return Err(e);
    }
    let partial_host = gpu.download_f32(&partial)?;
    // host: u = T128(x.in_scale); raw = u @ W_bare^T? decide by trying both
    let xs: Vec<f32> = (0..ic).map(|i| x_host[i] * in_scale_host[i]).collect();
    let u_host = apply_t128_host(&xs);
    let w_bare = decode_tiles(code_host, proj.k as usize, ic, oc); // [in, out]
    // raw[c] = sum_i u[i] * w_bare[i][c]
    let mut raw = vec![0.0f32; oc];
    for c in 0..oc {
        let mut s = 0.0f32;
        for i in 0..ic {
            s += u_host[i] * w_bare[i * oc + c];
        }
        raw[c] = s;
    }
    let mut got = vec![0.0f32; oc];
    for s in 0..n_slices {
        for c in 0..oc {
            got[c] += partial_host[s * oc + c];
        }
    }
    let mut raw_t = vec![0.0f32; oc];
    for c in 0..oc {
        let mut s = 0.0f32;
        for i in 0..ic {
            s += u_host[i] * w_bare[c * ic + i];
        }
        raw_t[c] = s;
    }
    let mut got = vec![0.0f32; oc];
    for s in 0..n_slices {
        for c in 0..oc {
            got[c] += partial_host[s * oc + c];
        }
    }
    let mut max_abs = 0.0f32;
    let mut max_abs_t = 0.0f32;
    for c in 0..oc {
        max_abs = max_abs.max((got[c] - raw[c]).abs());
        max_abs_t = max_abs_t.max((got[c] - raw_t[c]).abs());
    }
    eprintln!("[escha-dense] decode-stage max_abs(normal)={max_abs:.6} max_abs(transposed)={max_abs_t:.6}");
    eprintln!("  got  [0..8] = {:?}", &got[..8.min(oc)]);
    eprintln!("  raw  [0..8] = {:?}", &raw[..8.min(oc)]);
    eprintln!("  rawT [0..8] = {:?}", &raw_t[..8.min(oc)]);
    let _ = gpu.free_tensor(x);
    let _ = gpu.free_tensor(u);
    let _ = gpu.free_tensor(partial);
    Ok(max_abs.min(max_abs_t))
}

/// Host check: fold-side reconstruction (the MoE-verified semantics) gemv'd on
/// raw x vs the activation-side decode (my kernel semantics). Equal iff the
/// two algebra conventions agree.
#[allow(clippy::too_many_arguments)]
pub fn escha_dense_check_fold_vs_act(
    code_host: &[i16],
    k: usize,
    in_p: usize,
    out_p: usize,
    in_scale: &[f32],
    out_scale: &[f32],
    x: &[f32],
) -> (f32, Vec<f32>, Vec<f32>) {
    use crate::escham_decode::{decode_tiles, had128_matrix};
    // fold-side: M_folded = diag(out_scale) @ H_out @ W_bare^T @ H_in @ diag(in_scale)
    let w_bare = decode_tiles(code_host, k, in_p, out_p); // [in, out]
    let mut m = vec![0.0f32; out_p * in_p]; // W_bare^T [out, in]
    for i in 0..in_p {
        for j in 0..out_p {
            m[j * in_p + i] = w_bare[i * out_p + j];
        }
    }
    let mh = had128_matrix(&m, out_p, in_p); // H_out over rows, H_in over cols
    // scale: rows by out_scale, cols by in_scale
    let mut mf = vec![0.0f32; out_p * in_p];
    for j in 0..out_p {
        for i in 0..in_p {
            mf[j * in_p + i] = mh[j * in_p + i] * out_scale[j] * in_scale[i];
        }
    }
    // y = M_folded @ x
    let mut yf = vec![0.0f32; out_p];
    for j in 0..out_p {
        let mut s = 0.0f32;
        for i in 0..in_p {
            s += mf[j * in_p + i] * x[i];
        }
        yf[j] = s;
    }
    // activation-side
    let ya = escha_dense_decode_proj_host(code_host, k, in_p, out_p, in_scale, out_scale, x);
    let mut max_abs = 0.0f32;
    for j in 0..out_p {
        max_abs = max_abs.max((yf[j] - ya[j]).abs());
    }
    eprintln!(
        "[escha-dense] fold-vs-activation max_abs={max_abs:.6}  ({})",
        if max_abs < 1e-3 { "AGREE" } else { "DISAGREE" }
    );
    (max_abs, yf, ya)
}
