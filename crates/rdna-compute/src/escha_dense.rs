// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU dispatch wrappers for the Escha-W2 code-quant DENSE decode kernels
//! (Qwen3.8-27B-Escha-W2 etc.).
//!
//! One coded projection is
//!
//!   code     I16 [in/16, out/16, 16*K]   (bitwise F16 tensor on GPU)
//!   in_scale  F32 [in]  = rin . s_in
//!   out_scale F32 [out] = rout . s_out
//!
//! and the decode-gemm computes, per token (batch-1 gen path):
//!
//!   u   = T128(x . in_scale)                 (escha_dense_rotate_in)
//!   p   = u @ decode(code)                   (escha_dense_decode_gemv,
//!                                             IC sliced across grid.z)
//!   y   = T128_col(sum_slices p) . out_scale (escha_dense_finalize)
//!
//! 16-bit codebook indices are decoded from the payload inline (analytic dep
//! + funnel shift); see the .hip file header for the formula.

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;

const NT: u32 = 128; // threads per block (one per output column)

/// Choose the IC-slice count for one decode-gemm: the natural grid is OC/128
/// blocks (output column blocks), which is 8–136 for the FFN and leaves most
/// of the GPU idle at batch 1; slicing the IC reduction multiplies the block
/// count. The u-stage per block is `ceil(nit/n_slices)*16` floats, so also cap
/// the shared memory at 48 KB.
pub fn escha_dense_n_slices(nit: usize, oc: usize) -> usize {
    let n_ocb = (oc / 128).max(1);
    // Target ~1024 blocks at batch 1 (enough to fill a gfx1151 CU array).
    let mut n_slices = (1024usize / n_ocb).max(1).min(nit);
    // smem: 8*24 uint2 pairs (1536 B) + tiles*16 floats.
    const SMEM_BYTES: usize = 48 * 1024;
    const PAY_BYTES: usize = 8 * 24 * 8;
    let max_tiles = (SMEM_BYTES - PAY_BYTES) / (16 * 4);
    while n_slices < nit && nit.div_ceil(n_slices) > max_tiles {
        n_slices *= 2;
    }
    n_slices.max(1).min(nit)
}

/// `u = T128(x . in_scale)` for one row. `ic` is the actual projection input
/// size (the caller may pass a scratch `u` sized larger than `ic` for
/// hipGraph-capture safety).
pub fn escha_dense_rotate_in(
    gpu: &mut Gpu,
    in_scale: &GpuTensor,
    x: &GpuTensor,
    u: &GpuTensor,
    ic: usize,
) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escha_dense_rotate_in",
        &kernels::escha_dense_src(),
        "escha_dense_rotate_in_kernel",
    )?;
    let sp = in_scale.buf.as_ptr();
    let xp = x.buf.as_ptr();
    let up = u.buf.as_ptr();
    let ic = ic as i32;

    let mut params: Vec<*mut c_void> = vec![
        &sp as *const _ as *mut c_void,
        &xp as *const _ as *mut c_void,
        &up as *const _ as *mut c_void,
        &ic as *const _ as *mut c_void,
    ];
    let _timer = crate::profile::begin_timer(
        &gpu.hip,
        "escha",
        "escha_dense_rotate_in",
        u.numel() * 4,
    );
    gpu.launch_maybe_blob(
        "escha_dense_rotate_in_kernel",
        [1, 1, 1],
        [256, 1, 1],
        0, // dynamic smem not used for rotate (fixed 8 KB inside kernel decl)
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(sp);
            b.push_ptr(xp);
            b.push_ptr(up);
            b.push_i32(ic);
            b
        },
    )
}

/// One decode-gemm for a single token:
///   partial[n_slices][OC] = u @ decode(code)   (IC sliced across z)
///   y = T128_col(sum_slices) . out_scale       (finalize)
///
/// `u` and `y` are caller scratch; the kernel launches allocate nothing.
#[allow(clippy::too_many_arguments)]
pub fn escha_dense_decode_gemv(
    gpu: &mut Gpu,
    code: &GpuTensor,
    in_scale: &GpuTensor,
    out_scale: &GpuTensor,
    x: &GpuTensor,
    u: &GpuTensor,
    partial: &GpuTensor,
    y: &GpuTensor,
) -> HipResult<()> {
    let ic = code.shape[0] as usize * 16;
    let oc = code.shape[1] as usize * 16;
    let k = (code.shape[2] as usize / 16) as i32;
    let nit = ic / 16;
    let n_ocb = oc / 128;
    let n_slices = escha_dense_n_slices(nit, oc);
    debug_assert!(n_slices >= 1);
    debug_assert!(n_slices * oc <= partial.numel(), "partial too small");
    debug_assert_eq!(y.numel(), oc);
    debug_assert_eq!(u.numel(), ic);

    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escha_dense_decode_gemv",
        &kernels::escha_dense_src(),
        "escha_dense_decode_gemv_kernel",
    )?;
    gpu.ensure_kernel(
        "escha_dense_finalize",
        &kernels::escha_dense_src(),
        "escha_dense_finalize_kernel",
    )?;

    let cp = code.buf.as_ptr();
    let up = u.buf.as_ptr();
    let pp = partial.buf.as_ptr();
    let ic_i = ic as i32;
    let oc_i = oc as i32;
    let ns_i = n_slices as i32;
    let mut params: Vec<*mut c_void> = vec![
        &cp as *const _ as *mut c_void,
        &up as *const _ as *mut c_void,
        &pp as *const _ as *mut c_void,
        &ic_i as *const _ as *mut c_void,
        &oc_i as *const _ as *mut c_void,
        &ns_i as *const _ as *mut c_void,
        &k as *const _ as *mut c_void,
    ];

    // 1. rotate
    escha_dense_rotate_in(gpu, in_scale, x, u, ic)?;

    // 2. decode-gemm (grid: 1 row x OC/128 col-blocks x n_slices)
    let _timer = crate::profile::begin_timer(
        &gpu.hip,
        "escha",
        "escha_dense_decode_gemv",
        n_slices * ic * oc / n_slices * 4,
    );
    // shared: 8*NW u32 payload words (K=3: 8*24*4 = 768 B) + tiles*16 floats
    let tiles_max = nit.div_ceil(n_slices);
    let nw = 8 * (k as usize);
    let smem = (8 * nw * 4 + tiles_max * 16 * 4) as u32;
    gpu.launch_maybe_blob(
        "escha_dense_decode_gemv_kernel",
        [1, n_ocb as u32, n_slices as u32],
        [NT, 1, 1],
        smem,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(cp);
            b.push_ptr(up);
            b.push_ptr(pp);
            b.push_i32(ic_i);
            b.push_i32(oc_i);
            b.push_i32(ns_i);
            b.push_i32(k);
            b
        },
    )?;

    // 3. finalize (grid: OC/128, one block per 128-col group)
    let op = out_scale.buf.as_ptr();
    let yp = y.buf.as_ptr();
    let n_rows_i = 1i32; // decode path: 1 row
    let mut params: Vec<*mut c_void> = vec![
        &op as *const _ as *mut c_void,
        &pp as *const _ as *mut c_void,
        &yp as *const _ as *mut c_void,
        &oc_i as *const _ as *mut c_void,
        &n_rows_i as *const _ as *mut c_void,
        &ns_i as *const _ as *mut c_void,
    ];
    let _timer2 = crate::profile::begin_timer(&gpu.hip, "escha", "escha_dense_finalize", oc * 4);
    gpu.launch_maybe_blob(
        "escha_dense_finalize_kernel",
        [1, n_ocb as u32, 1], // row=0 for decode
        [NT, 1, 1],
        0,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(op);
            b.push_ptr(pp);
            b.push_ptr(yp);
            b.push_i32(oc_i);
            b.push_i32(n_rows_i);
            b.push_i32(ns_i);
            b
        },
    )
}

/// Batched rotate: u[row] = T128(x[row] . in_scale) for all rows.
/// `u` is [n_rows, IC] (f32).
pub fn escha_dense_rotate_in_dense(
    gpu: &mut Gpu,
    in_scale: &GpuTensor,
    x: &GpuTensor,
    u: &GpuTensor,
    n_rows: usize,
    ic: usize,
) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escha_dense_rotate_in_dense",
        &kernels::escha_dense_src(),
        "escha_dense_rotate_in_dense_kernel",
    )?;
    let sp = in_scale.buf.as_ptr();
    let xp = x.buf.as_ptr();
    let up = u.buf.as_ptr();
    let ic_i = ic as i32;
    let n_rows_i = n_rows as i32;

    let mut params: Vec<*mut c_void> = vec![
        &sp as *const _ as *mut c_void,
        &xp as *const _ as *mut c_void,
        &up as *const _ as *mut c_void,
        &ic_i as *const _ as *mut c_void,
        &n_rows_i as *const _ as *mut c_void,
    ];
    gpu.launch_maybe_blob(
        "escha_dense_rotate_in_dense_kernel",
        [n_rows as u32, 1, 1],
        [256, 1, 1],
        0,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(sp);
            b.push_ptr(xp);
            b.push_ptr(up);
            b.push_i32(ic_i);
            b.push_i32(n_rows_i);
            b
        },
    )
}

/// Batched prefill matmul for one coded projection.
/// `u` is [n_rows, IC] pre-rotated activations.
/// `partial` is [n_slices * n_rows * OC] (f32).
/// `n_slices` is chosen to fill the device.
/// R = rows per block (1 for gen, 64 for prefill).
#[allow(clippy::too_many_arguments)]
pub fn escha_dense_matmul_prefill(
    gpu: &mut Gpu,
    code: &GpuTensor,
    u: &GpuTensor,
    partial: &GpuTensor,
    n_rows: usize,
    n_slices: usize,
    k: i32,
    r: i32,
) -> HipResult<()> {
    let ic = code.shape[0] as usize * 16;
    let oc = code.shape[1] as usize * 16;
    let nit = ic / 16;
    let n_ocb = oc / 128;

    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escha_dense_matmul_prefill",
        &kernels::escha_dense_src(),
        "escha_dense_matmul_prefill_kernel",
    )?;

    let cp = code.buf.as_ptr();
    let up = u.buf.as_ptr();
    let pp = partial.buf.as_ptr();
    let ic_i = ic as i32;
    let oc_i = oc as i32;
    let n_rows_i = n_rows as i32;
    let ns_i = n_slices as i32;

    let n_rb = (n_rows + r as usize - 1) / r as usize;

    let nw = 8 * k as usize;
    let tiles_max = nit.div_ceil(n_slices);
    let smem = (8 * nw * std::mem::size_of::<u32>() + tiles_max * 16 * std::mem::size_of::<f32>()) as u32;

    let mut params: Vec<*mut c_void> = vec![
        &cp as *const _ as *mut c_void,
        &up as *const _ as *mut c_void,
        &pp as *const _ as *mut c_void,
        &ic_i as *const _ as *mut c_void,
        &oc_i as *const _ as *mut c_void,
        &n_rows_i as *const _ as *mut c_void,
        &ns_i as *const _ as *mut c_void,
        &k as *const _ as *mut c_void,
        &r as *const _ as *mut c_void,
    ];
    gpu.launch_maybe_blob(
        "escha_dense_matmul_prefill_kernel",
        [n_rb as u32, n_ocb as u32, n_slices as u32],
        [NT, 1, 1],
        smem,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(cp);
            b.push_ptr(up);
            b.push_ptr(pp);
            b.push_i32(ic_i);
            b.push_i32(oc_i);
            b.push_i32(n_rows_i);
            b.push_i32(ns_i);
            b.push_i32(k);
            b.push_i32(r);
            b
        },
    )
}

/// Batched finalize: sum slices, WHT over 128-col group, scale by out_scale.
/// `partial` is [n_slices * n_rows * OC], `y` is [n_rows, OC].
pub fn escha_dense_finalize_dense(
    gpu: &mut Gpu,
    out_scale: &GpuTensor,
    partial: &GpuTensor,
    y: &GpuTensor,
    n_rows: usize,
    n_slices: usize,
) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escha_dense_finalize_dense",
        &kernels::escha_dense_src(),
        "escha_dense_finalize_kernel",
    )?;
    let op = out_scale.buf.as_ptr();
    let pp = partial.buf.as_ptr();
    let yp = y.buf.as_ptr();
    let oc = out_scale.numel() as i32;
    let n_rows_i = n_rows as i32;
    let ns_i = n_slices as i32;
    let n_ocb = (out_scale.numel() / 128) as u32;

    let mut params: Vec<*mut c_void> = vec![
        &op as *const _ as *mut c_void,
        &pp as *const _ as *mut c_void,
        &yp as *const _ as *mut c_void,
        &oc as *const _ as *mut c_void,
        &n_rows_i as *const _ as *mut c_void,
        &ns_i as *const _ as *mut c_void,
    ];
    gpu.launch_maybe_blob(
        "escha_dense_finalize_kernel",
        [n_rows as u32, n_ocb, 1],
        [128, 1, 1],
        0,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(op);
            b.push_ptr(pp);
            b.push_ptr(yp);
            b.push_i32(oc);
            b.push_i32(n_rows_i);
            b.push_i32(ns_i);
            b
        },
    )
}

/// Choose the IC-slice count for prefill. The matmul kernel's per-block work
/// scales as (nit / n_slices) * R (R rows accumulated per block), so to keep
/// per-block work bounded as n_rows grows, n_slices must grow with R.
/// Grid = [ceil(n_rows/R), OC/128, n_slices]. Target: per-block work ≈ constant.
pub fn escha_dense_n_slices_prefill(nit: usize, oc: usize, n_rows: usize, r: i32) -> usize {
    let n_ocb = (oc / 128).max(1);
    let n_rb = (n_rows as i32 + r - 1) / r.max(1);
    // Keep per-block work ≈ constant: each block handles (nit/n_slices) tiles * R rows.
    // Target total blocks ≈ 512-1024 for the whole grid.
    let target_blocks = if n_rows <= 4 { 1024 } else { 512 };
    let mut n_slices = target_blocks / n_rb.max(1) as usize / n_ocb.max(1);
    // Scale n_slices with R to keep per-block work bounded: more rows per block =
    // more FMAs per block, so we need more slices to compensate.
    n_slices = n_slices.max((r as usize).max(1)).min(nit);
    // Ensure nit is divisible by n_slices
    while n_slices < nit && nit % n_slices != 0 {
        n_slices += 1;
    }
    n_slices
}

/// Diagnostic stage launch: decode-gemm only (rotate + partial). Exposed for
/// the GPU-vs-host stage comparison; production callers use
/// [`escha_dense_decode_gemv`].
#[allow(clippy::too_many_arguments)]
pub fn escha_dense_decode_gemv_stage(
    gpu: &mut Gpu,
    code: &GpuTensor,
    u: &GpuTensor,
    partial: &GpuTensor,
    ic: usize,
    oc: usize,
    n_slices: usize,
    k: i32,
) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escha_dense_decode_gemv_stage",
        &kernels::escha_dense_src(),
        "escha_dense_decode_gemv_kernel",
    )?;
    let cp = code.buf.as_ptr();
    let up = u.buf.as_ptr();
    let pp = partial.buf.as_ptr();
    let ic_i = ic as i32;
    let oc_i = oc as i32;
    let ns_i = n_slices as i32;
    let nit = ic / 16;
    let n_ocb = oc / 128;
    let mut params: Vec<*mut c_void> = vec![
        &cp as *const _ as *mut c_void,
        &up as *const _ as *mut c_void,
        &pp as *const _ as *mut c_void,
        &ic_i as *const _ as *mut c_void,
        &oc_i as *const _ as *mut c_void,
        &ns_i as *const _ as *mut c_void,
        &k as *const _ as *mut c_void,
    ];
    let tiles_max = nit.div_ceil(n_slices);
    let nw = 8 * (k as usize);
    let smem = (8 * nw * 4 + tiles_max * 16 * 4) as u32;
    gpu.launch_maybe_blob(
        "escha_dense_decode_gemv_kernel",
        [1, n_ocb as u32, n_slices as u32],
        [NT, 1, 1],
        smem,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(cp);
            b.push_ptr(up);
            b.push_ptr(pp);
            b.push_i32(ic_i);
            b.push_i32(oc_i);
            b.push_i32(ns_i);
            b.push_i32(k);
            b
        },
    )
}
