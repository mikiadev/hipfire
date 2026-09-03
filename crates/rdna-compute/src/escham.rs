// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU dispatch wrappers for the 8 Escha MoE (ESCHAM code-quant) kernels.
//!
//! These kernels implement the Escha W2 / Qwen3.5-MoE ESCHAM decode path:
//!   - Hadamard input transform (shared-expert and routed-expert variants)
//!   - EPI (Expert Parallel Inference) transform with in-kernel SWiGLU
//!   - Load-balanced scatter-combine for output accumulation
//!   - Chunk building for routing table construction
//!
//! All kernels share: col = (bid_x << 7) | (tid << 2), maxntid 32,1,1,
//! 2x2 Hadamard + 32-way warp butterfly reduction (5 stages: 1,2,4,8,16),
//! sqrt(2)/16 scaling. Total: H_4 ⊗ H_32 = H_128 (Sylvester WHT).

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::HipResult;

/// Escha MoE Hadamard input transform (non-expert / shared-expert path).
///
/// Computes, per thread (4 elements):
///   p_i = h2f(in1[col+i]) * scale[col+i] * h2f(in2[col+i])
/// Then 2x2 Hadamard, 32-way butterfly (5 stages), scale by sqrt(2)/16.
///

/// CORRECT ESCHAM trellis decode (EXL3 + 3INST codebook), replacing the old
/// additive-delta decode. Produces W_bare^T [out_p, in_p] from the int16 codes.
/// No Hadamard, no scales — fold + gemv input scaling handle those.
pub fn escham_moe_decode_trellis(
    gpu: &mut Gpu,
    out: &GpuTensor,
    codes: &GpuTensor,
    k: i32,
    in_p: i32,
    out_p: i32,
) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escham_moe_decode_trellis",
        &kernels::escham_moe_decode_trellis_src(),
        "escham_moe_decode_trellis_kernel",
    )?;
    let op = out.buf.as_ptr();
    let cp = codes.buf.as_ptr();
    let bi_max = (in_p as usize / ESCHA_TILE) as i32;
    let bj_max = (out_p as usize / ESCHA_TILE) as i32;

    let mut params: Vec<*mut c_void> = vec![
        &op as *const _ as *mut c_void,
        &cp as *const _ as *mut c_void,
        &k as *const _ as *mut c_void,
        &bi_max as *const _ as *mut c_void,
        &bj_max as *const _ as *mut c_void,
        &in_p as *const _ as *mut c_void,
        &out_p as *const _ as *mut c_void,
    ];

    let _timer = crate::profile::begin_timer(
        &gpu.hip,
        "escham",
        "escham_moe_decode_trellis",
        (in_p * out_p) as usize,
    );

    // shared memory: n_words uint32 = (k*256)/32 = 16 (K=2) / 24 (K=3)
    let shmem = (((k * 256) / 32) * 4) as u32;
    gpu.launch_maybe_blob(
        "escham_moe_decode_trellis_kernel",
        [bi_max as u32, bj_max as u32, 1],
        [128, 1, 1],
        shmem,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(op);
            b.push_ptr(cp);
            b.push_i32(k);
            b.push_i32(bi_max);
            b.push_i32(bj_max);
            b.push_i32(in_p);
            b.push_i32(out_p);
            b
        },
    )
}

const ESCHA_TILE: usize = 16;

/// Fold pass 1: `out1 = T128_out @ (M . rout)` — block-diagonal 128-pt WHT over
/// the OUTPUT dim (rows) of the transposed decode output `m` [out_p, in_p],
/// with per-row scale `rout`. `out_p` and `in_p` must be multiples of 128.
pub fn escham_fold_t128_rows(
    gpu: &mut Gpu,
    out1: &GpuTensor,
    m: &GpuTensor,
    rout: &GpuTensor,
    out_p: usize,
    in_p: usize,
) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escham_fold_t128_rows",
        &kernels::escham_moe_fold_src(),
        "escham_fold_t128_rows_kernel",
    )?;
    let o1p = out1.buf.as_ptr();
    let mp = m.buf.as_ptr();
    let rp = rout.buf.as_ptr();
    let op_i32 = out_p as i32;
    let ip_i32 = in_p as i32;

    let n_blocks_x = in_p as u32;
    let n_blocks_y = (out_p / 128) as u32;

    let mut params: Vec<*mut c_void> = vec![
        &o1p as *const _ as *mut c_void,
        &mp as *const _ as *mut c_void,
        &rp as *const _ as *mut c_void,
        &op_i32 as *const _ as *mut c_void,
        &ip_i32 as *const _ as *mut c_void,
    ];

    let _timer = crate::profile::begin_timer(
        &gpu.hip,
        "escham",
        "escham_fold_t128_rows",
        out_p * in_p,
    );

    gpu.launch_maybe_blob(
        "escham_fold_t128_rows_kernel",
        [n_blocks_x, n_blocks_y, 1],
        [32, 1, 1],
        0,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(o1p);
            b.push_ptr(mp);
            b.push_ptr(rp);
            b.push_i32(op_i32);
            b.push_i32(ip_i32);
            b
        },
    )
}

/// Fold pass 2: `out2 = out1 @ T128_in` — block-diagonal 128-pt WHT over the
/// INPUT dim (cols) of `out1` [out_p, in_p].
pub fn escham_fold_t128_cols(
    gpu: &mut Gpu,
    out2: &GpuTensor,
    out1: &GpuTensor,
    out_p: usize,
    in_p: usize,
) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escham_fold_t128_cols",
        &kernels::escham_moe_fold_src(),
        "escham_fold_t128_cols_kernel",
    )?;
    let o2p = out2.buf.as_ptr();
    let o1p = out1.buf.as_ptr();
    let op_i32 = out_p as i32;
    let ip_i32 = in_p as i32;

    let n_blocks_x = (in_p / 128) as u32;
    let n_blocks_y = out_p as u32;

    let mut params: Vec<*mut c_void> = vec![
        &o2p as *const _ as *mut c_void,
        &o1p as *const _ as *mut c_void,
        &op_i32 as *const _ as *mut c_void,
        &ip_i32 as *const _ as *mut c_void,
    ];

    let _timer = crate::profile::begin_timer(
        &gpu.hip,
        "escham",
        "escham_fold_t128_cols",
        out_p * in_p,
    );

    gpu.launch_maybe_blob(
        "escham_fold_t128_cols_kernel",
        [n_blocks_x, n_blocks_y, 1],
        [32, 1, 1],
        0,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(o2p);
            b.push_ptr(o1p);
            b.push_i32(op_i32);
            b.push_i32(ip_i32);
            b
        },
    )
}

/// Elementwise `out = a . b` (F32), used to scale gemv inputs by `s_in . rin`.
pub fn escham_mul_vec_f32(
    gpu: &mut Gpu,
    out: &GpuTensor,
    a: &GpuTensor,
    b: &GpuTensor,
) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escham_mul_vec_f32",
        &kernels::escham_moe_fold_src(),
        "escham_mul_vec_f32_kernel",
    )?;
    let op = out.buf.as_ptr();
    let ap = a.buf.as_ptr();
    let bp = b.buf.as_ptr();
    let n = out.numel() as i32;

    let n_blocks = ((n as usize + 255) / 256) as u32;

    let mut params: Vec<*mut c_void> = vec![
        &op as *const _ as *mut c_void,
        &ap as *const _ as *mut c_void,
        &bp as *const _ as *mut c_void,
        &n as *const _ as *mut c_void,
    ];

    let _timer = crate::profile::begin_timer(
        &gpu.hip,
        "escham",
        "escham_mul_vec_f32",
        out.buf.size(),
    );

    gpu.launch_maybe_blob(
        "escham_mul_vec_f32_kernel",
        [n_blocks, 1, 1],
        [256, 1, 1],
        0,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(op);
            b.push_ptr(ap);
            b.push_ptr(bp);
            b.push_i32(n);
            b
        },
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Grouped / batched escha-MoE FFN kernels (AR decode, batch=1).
// Fold ALL scales into the cached folded weights at fill time
// (W' = diag(rout) @ M_folded @ diag(s_in . rin), an exact identity), then run
// the 8 routed experts through one grouped gemv per projection + one batched
// silu-mul + one batched weighted accumulate per layer: 4 launches/layer.
// ─────────────────────────────────────────────────────────────────────────────

/// W' = W ⊙ rout_row ⊙ in_scale_col (in-place safe). Row-major [out_p, in_p].
pub fn escham_apply_rowcol_scales_f32(
    gpu: &mut Gpu,
    w: &GpuTensor,
    rout: &GpuTensor,
    in_scale: &GpuTensor,
    out_p: usize,
    in_p: usize,
) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escham_apply_rowcol_scales_f32",
        &kernels::escham_moe_grouped_src(),
        "escham_apply_rowcol_scales_f32",
    )?;
    let wp = w.buf.as_ptr();
    let rp = rout.buf.as_ptr();
    let sp = in_scale.buf.as_ptr();
    let op_i32 = out_p as i32;
    let ip_i32 = in_p as i32;
    let total = (out_p * in_p) as u32;
    let n_blocks = (total + 255) / 256;
    let mut params: Vec<*mut c_void> = vec![
        &wp as *const _ as *mut c_void,
        &rp as *const _ as *mut c_void,
        &sp as *const _ as *mut c_void,
        &op_i32 as *const _ as *mut c_void,
        &ip_i32 as *const _ as *mut c_void,
    ];
    let _timer = crate::profile::begin_timer(&gpu.hip, "escham", "apply_rowcol_scales", w.buf.size());
    gpu.launch_maybe_blob(
        "escham_apply_rowcol_scales_f32",
        [n_blocks, 1, 1],
        [256, 1, 1],
        0,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(wp);
            b.push_ptr(rp);
            b.push_ptr(sp);
            b.push_i32(op_i32);
            b.push_i32(ip_i32);
            b
        },
    )
}

/// out[e*m_per + r] = Σ_j w_ptrs[e][r][j] * xin[e][j] for e in 0..n_exp.
/// `w_ptrs` is a device array of n_exp weight base pointers. `x_stride==0`
/// means every expert reads the same x (gate_up); >0 means expert e reads
/// x + e*x_stride (down, per-expert gated input).
pub fn escham_moe_grouped_gemv_f32(
    gpu: &mut Gpu,
    w_ptrs: &GpuTensor,
    x: &GpuTensor,
    out: &GpuTensor,
    m_per: usize,
    k: usize,
    n_exp: usize,
    x_stride: usize,
) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escham_moe_grouped_gemv_f32",
        &kernels::escham_moe_grouped_src(),
        "escham_moe_grouped_gemv_f32",
    )?;
    let wpp = w_ptrs.buf.as_ptr();
    let xp = x.buf.as_ptr();
    let op = out.buf.as_ptr();
    let mp = m_per as i32;
    let kp = k as i32;
    let np = n_exp as i32;
    let xsp = x_stride as i32;
    let grid = (n_exp * m_per) as u32;
    let block = 256u32.min(k as u32).max(32);
    let mut params: Vec<*mut c_void> = vec![
        &wpp as *const _ as *mut c_void,
        &xp as *const _ as *mut c_void,
        &op as *const _ as *mut c_void,
        &mp as *const _ as *mut c_void,
        &kp as *const _ as *mut c_void,
        &np as *const _ as *mut c_void,
        &xsp as *const _ as *mut c_void,
    ];
    let _timer = crate::profile::begin_timer(
        &gpu.hip,
        "escham",
        "grouped_gemv",
        n_exp * m_per * k,
    );
    gpu.launch_maybe_blob(
        "escham_moe_grouped_gemv_f32",
        [grid, 1, 1],
        [block, 1, 1],
        block * 4,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(wpp);
            b.push_ptr(xp);
            b.push_ptr(op);
            b.push_i32(mp);
            b.push_i32(kp);
            b.push_i32(np);
            b.push_i32(xsp);
            b
        },
    )
}

/// gated[e*mi + j] = silu(gu[e*2mi + j]) * gu[e*2mi + mi + j]
pub fn escham_moe_batched_silu_mul_f32(
    gpu: &mut Gpu,
    gu: &GpuTensor,
    gated: &GpuTensor,
    n_exp: usize,
    mi: usize,
) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escham_moe_batched_silu_mul_f32",
        &kernels::escham_moe_grouped_src(),
        "escham_moe_batched_silu_mul_f32",
    )?;
    let gp = gu.buf.as_ptr();
    let dp = gated.buf.as_ptr();
    let np = n_exp as i32;
    let mip = mi as i32;
    let total = (n_exp * mi) as u32;
    let n_blocks = (total + 255) / 256;
    let mut params: Vec<*mut c_void> = vec![
        &gp as *const _ as *mut c_void,
        &dp as *const _ as *mut c_void,
        &np as *const _ as *mut c_void,
        &mip as *const _ as *mut c_void,
    ];
    gpu.launch_maybe_blob(
        "escham_moe_batched_silu_mul_f32",
        [n_blocks, 1, 1],
        [256, 1, 1],
        0,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(gp);
            b.push_ptr(dp);
            b.push_i32(np);
            b.push_i32(mip);
            b
        },
    )
}

/// expert_out[j] += Σ_e weights[e] * dn[e*hidden + j]
pub fn escham_moe_batched_scaled_add_f32(
    gpu: &mut Gpu,
    expert_out: &GpuTensor,
    dn: &GpuTensor,
    weights: &GpuTensor,
    n_exp: usize,
    hidden: usize,
) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escham_moe_batched_scaled_add_f32",
        &kernels::escham_moe_grouped_src(),
        "escham_moe_batched_scaled_add_f32",
    )?;
    let ep = expert_out.buf.as_ptr();
    let dp = dn.buf.as_ptr();
    let wp = weights.buf.as_ptr();
    let np = n_exp as i32;
    let hp = hidden as i32;
    let n_blocks = ((hidden as u32) + 255) / 256;
    let mut params: Vec<*mut c_void> = vec![
        &ep as *const _ as *mut c_void,
        &dp as *const _ as *mut c_void,
        &wp as *const _ as *mut c_void,
        &np as *const _ as *mut c_void,
        &hp as *const _ as *mut c_void,
    ];
    gpu.launch_maybe_blob(
        "escham_moe_batched_scaled_add_f32",
        [n_blocks, 1, 1],
        [256, 1, 1],
        0,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(ep);
            b.push_ptr(dp);
            b.push_ptr(wp);
            b.push_i32(np);
            b.push_i32(hp);
            b
        },
    )
}

/// Convert a folded f32 [out_p, in_p] weight to f16 (after scale absorption).
pub fn escham_f32_to_f16(
    gpu: &mut Gpu,
    out: &GpuTensor,
    input: &GpuTensor,
) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escham_f32_to_f16",
        &kernels::escham_moe_grouped_src(),
        "escham_f32_to_f16",
    )?;
    let ip = input.buf.as_ptr();
    let op = out.buf.as_ptr();
    let n = input.numel() as i32;
    let n_blocks = ((n as u32) + 255) / 256;
    let mut params: Vec<*mut c_void> = vec![
        &ip as *const _ as *mut c_void,
        &op as *const _ as *mut c_void,
        &n as *const _ as *mut c_void,
    ];
    gpu.launch_maybe_blob(
        "escham_f32_to_f16",
        [n_blocks, 1, 1],
        [256, 1, 1],
        0,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(ip);
            b.push_ptr(op);
            b.push_i32(n);
            b
        },
    )
}

/// FP16-weight grouped GEMV (same contract as the f32 variant).
pub fn escham_moe_grouped_gemv_f16(
    gpu: &mut Gpu,
    w_ptrs: &GpuTensor,
    x: &GpuTensor,
    out: &GpuTensor,
    m_per: usize,
    k: usize,
    n_exp: usize,
    x_stride: usize,
) -> HipResult<()> {
    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escham_moe_grouped_gemv_f16",
        &kernels::escham_moe_grouped_src(),
        "escham_moe_grouped_gemv_f16",
    )?;
    let wpp = w_ptrs.buf.as_ptr();
    let xp = x.buf.as_ptr();
    let op = out.buf.as_ptr();
    let mp = m_per as i32;
    let kp = k as i32;
    let np = n_exp as i32;
    let xsp = x_stride as i32;
    let grid = (n_exp * m_per) as u32;
    let block = 256u32.min(k as u32).max(32);
    let mut params: Vec<*mut c_void> = vec![
        &wpp as *const _ as *mut c_void,
        &xp as *const _ as *mut c_void,
        &op as *const _ as *mut c_void,
        &mp as *const _ as *mut c_void,
        &kp as *const _ as *mut c_void,
        &np as *const _ as *mut c_void,
        &xsp as *const _ as *mut c_void,
    ];
    let _timer = crate::profile::begin_timer(
        &gpu.hip,
        "escham",
        "grouped_gemv_f16",
        n_exp * m_per * k,
    );
    gpu.launch_maybe_blob(
        "escham_moe_grouped_gemv_f16",
        [grid, 1, 1],
        [block, 1, 1],
        block * 4,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(wpp);
            b.push_ptr(xp);
            b.push_ptr(op);
            b.push_i32(mp);
            b.push_i32(kp);
            b.push_i32(np);
            b.push_i32(xsp);
            b
        },
    )
}

