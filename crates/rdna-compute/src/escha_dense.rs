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
///
/// The ~1024-block target is a measured plateau, not a guess
/// (`examples/decode_stage_probe.rs`, gfx1151, gate_proj 5120x17408 K=2,
/// 22.3 MB of code): n_slices=1 (136 blocks) takes 2.78 ms, n_slices=7 (952
/// blocks) takes 1.10 ms for the SAME bytes, and pushing further regresses
/// (ns=30 -> 1.8 ms, ns=120 -> 1.8 ms, ns=320 -> 2.9 ms). So the gemv is
/// latency/parallelism-bound below ~1000 blocks and split-K-overhead-bound
/// above it. Corollary: 22.3 MB in 1.10 ms is only 20 GB/s against the ~104
/// GB/s this device sustains (examples/mem_bw.rs), i.e. the decode gemv leaves
/// ~5x of DRAM on the table, and the ceiling is occupancy, not bandwidth.
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
    let _timer =
        crate::profile::begin_timer(&gpu.hip, "escha", "escha_dense_rotate_in", u.numel() * 4);
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
///
/// NOTE on grid order: `code.shape` is `[out/16, in/16, 16*K]` (nt-major,
/// transposed at load). IC/OC below are read SWAPPED to match (ic =
/// shape[1]*16, oc = shape[0]*16); K is order-free. The kernels index
/// `tj*nit + ti`. Reverting any one of the three (load transpose, shape
/// read, kernel index) without the other two decodes wrong tiles.
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
    // nt-major shape [out/16, in/16, 16*K]: ic = shape[1]*16, oc = shape[0]*16.
    let oc = code.shape[0] as usize * 16;
    let ic = code.shape[1] as usize * 16;
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
    // shared: 8*NW uint2 payload PAIRS (K=3: 8*24*8 = 1536 B) + tiles*16 floats.
    // Pairs, not words: see the staging comment in escha_dense_decode_gemv_kernel.
    let tiles_max = nit.div_ceil(n_slices);
    let nw = 8 * (k as usize);
    let smem = (8 * nw * 8 + tiles_max * 16 * 4) as u32;
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

/// Rows-per-block ceiling for the batched prefill matmul.
///
/// `escha_dense_matmul_prefill_kernel` is a template over `R` so that its
/// `acc[R]` accumulators stay in VGPRs. The previous runtime-`R` build spilled
/// them (gfx1151: `private_segment_fixed_size = 272`, `VGPR = 19`), which is
/// what capped row-batching at ~1.4x. `R` costs one VGPR per row, so 32 is the
/// instantiation ceiling; any remaining rows ride on `blockIdx.x`.
pub const ESCHA_DENSE_PREFILL_R_MAX: i32 = 32;

/// Pick rows-per-block: the largest power of two both `<= n_rows` and
/// `<= ESCHA_DENSE_PREFILL_R_MAX`.
pub fn escha_dense_prefill_r(n_rows: usize) -> i32 {
    if n_rows <= 1 {
        1
    } else {
        n_rows
            .next_power_of_two()
            .min(ESCHA_DENSE_PREFILL_R_MAX as usize) as i32
    }
}

/// Symbol of the instantiated `(K, R)` kernel. `K` is the trellis bit-width
/// (2 or 3 in this checkpoint) and `R` the rows per block. Anything outside
/// this table is a dispatch bug, not a runtime condition.
fn prefill_kernel_sym(k: i32, r: i32) -> Option<&'static str> {
    Some(match (k, r) {
        (2, 1) => "escha_dense_matmul_prefill_k2_r1_kernel",
        (2, 2) => "escha_dense_matmul_prefill_k2_r2_kernel",
        (2, 4) => "escha_dense_matmul_prefill_k2_r4_kernel",
        (2, 8) => "escha_dense_matmul_prefill_k2_r8_kernel",
        (2, 16) => "escha_dense_matmul_prefill_k2_r16_kernel",
        (2, 32) => "escha_dense_matmul_prefill_k2_r32_kernel",
        (3, 1) => "escha_dense_matmul_prefill_k3_r1_kernel",
        (3, 2) => "escha_dense_matmul_prefill_k3_r2_kernel",
        (3, 4) => "escha_dense_matmul_prefill_k3_r4_kernel",
        (3, 8) => "escha_dense_matmul_prefill_k3_r8_kernel",
        (3, 16) => "escha_dense_matmul_prefill_k3_r16_kernel",
        (3, 32) => "escha_dense_matmul_prefill_k3_r32_kernel",
        _ => return None,
    })
}

/// Batched prefill matmul for one coded projection.
/// `u` is [n_rows, IC] pre-rotated activations.
/// `partial` is [n_slices * n_rows * OC] (f32).
/// `n_slices` is chosen to fill the device.
/// `r` is rows per block and must be an instantiated value — use
/// [`escha_dense_prefill_r`]. `K` and `R` are baked into the kernel symbol.
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
    let sym = prefill_kernel_sym(k, r).ok_or_else(|| {
        hip_bridge::HipError::new(
            0,
            &format!(
                "escha_dense_matmul_prefill: no kernel instantiation for K={k}, R={r} \
                 (K in 2..=3, R a power of two <= {ESCHA_DENSE_PREFILL_R_MAX})"
            ),
        )
    })?;
    // nt-major shape (see decode_gemv above): shape = [out/16, in/16, 16*K].
    let oc = code.shape[0] as usize * 16;
    let ic = code.shape[1] as usize * 16;
    let nit = ic / 16;
    let n_ocb = oc / 128;

    gpu.bind_thread()?;
    gpu.ensure_kernel(
        "escha_dense_matmul_prefill",
        &kernels::escha_dense_src(),
        sym,
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
    // The kernel stages R*16 floats into s_u on the R>1 path but
    // tiles_max*16 on the R==1 gen path, so size for the larger. Before the
    // slice heuristic dropped its `n_slices >= R` floor, tiles_max could fall
    // far below R and the R>1 path overran s_u — corrupting decoded weights
    // into a verbatim attractor (9daf925bf). Keep the max: the floor is gone
    // but the sizing rule is not.
    let s_u_floats = tiles_max.max(r as usize) * 16;
    let smem =
        (8 * nw * std::mem::size_of::<u32>() + s_u_floats * std::mem::size_of::<f32>()) as u32;

    let mut params: Vec<*mut c_void> = vec![
        &cp as *const _ as *mut c_void,
        &up as *const _ as *mut c_void,
        &pp as *const _ as *mut c_void,
        &ic_i as *const _ as *mut c_void,
        &oc_i as *const _ as *mut c_void,
        &n_rows_i as *const _ as *mut c_void,
        &ns_i as *const _ as *mut c_void,
    ];
    gpu.launch_maybe_blob(
        sym,
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

/// Choose the IC-slice count for prefill. Grid = `[ceil(n_rows/R), OC/128,
/// n_slices]`; each block handles `nit / n_slices` tiles over `R` rows.
///
/// `n_slices` is a pure parallelism knob — the total FMA work is fixed, so it
/// trades occupancy against the fp32 split-K partial, which costs
/// `n_slices * n_rows * OC * 4` bytes of write AND read. It is therefore pinned
/// to the block-count target and deliberately NOT floored at `R`.
///
/// The floor it used to have (`n_slices = n_slices.max(r)`, on the theory that
/// more rows per block need more slices to bound per-block work) was wrong:
/// the grid already carries `ceil(n_rows / R)` row-blocks, so per-block work is
/// bounded by the tile count, not by `R`. All the floor did was inflate the
/// partial. At `n_rows = 58` it pinned every projection to `n_slices = 64` —
/// 259 MB for gate/up alone, ~125 GB of partial write+read per 58-token chunk.
pub fn escha_dense_n_slices_prefill(nit: usize, oc: usize, n_rows: usize, r: i32) -> usize {
    let n_ocb = (oc / 128).max(1);
    let n_rb = (n_rows as i32 + r - 1) / r.max(1);
    // Keep per-block work ≈ constant: each block handles (nit/n_slices) tiles * R rows.
    // Target total blocks ≈ 512-1024 for the whole grid.
    let target_blocks = if n_rows <= 4 { 1024 } else { 512 };
    let mut n_slices = target_blocks / n_rb.max(1) as usize / n_ocb.max(1);

    // Shared-memory bound, replacing what the deleted `>= R` floor was
    // accidentally providing. Computed against the widest payload (K=3) so it
    // holds for every projection, not just the one being sliced.
    let payload_bytes = 8 * (8 * 3) * 4;
    let max_tiles = ((ESCHA_DENSE_SMEM_BUDGET - payload_bytes) / (16 * 4)).max(1);
    n_slices = n_slices.max(nit.div_ceil(max_tiles));

    n_slices = n_slices.max(1).min(nit);
    // Ensure nit is divisible by n_slices
    while n_slices < nit && nit % n_slices != 0 {
        n_slices += 1;
    }
    n_slices
}

/// Per-block dynamic shared-memory ceiling we refuse to cross.
///
/// The prefill matmul sizes its dynamic smem as `payload + tiles_per_slice * 16
/// * 4`, because the R==1/gen path stages `u` for the WHOLE slice at once. A
/// slice that is too wide therefore asks for more shared memory than the device
/// allows and `hipModuleLaunchKernel` fails with `invalid argument` — a hard
/// crash, not a wrong answer. 64 KB is gfx1151's per-block limit. Held
/// statically rather than queried, and enforced in the *heuristic* so the two
/// cannot drift apart.
const ESCHA_DENSE_SMEM_BUDGET: usize = 64 * 1024;

/// Dynamic smem a prefill launch requests, mirroring the kernel layout so the
/// bound above can be asserted rather than assumed. Keep in sync with
/// `escha_dense_matmul_prefill`.
pub fn escha_dense_prefill_smem(nit: usize, n_slices: usize, k: i32, r: i32) -> usize {
    let nw = 8 * k as usize;
    let tiles_max = nit.div_ceil(n_slices);
    let s_u_floats = tiles_max.max(r as usize) * 16;
    8 * nw * 4 + s_u_floats * 4
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
    let smem = (8 * nw * 8 + tiles_max * 16 * 4) as u32;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Removing the `>= R` floor must not remove the shared-memory bound it
    /// was accidentally providing: an unbounded `n_slices` collapses to 1 on
    /// wide chunks and the launch asks for more dynamic smem than the device
    /// has (`hipModuleLaunchKernel: invalid argument`, reproduced on gfx1151
    /// with `down_proj` at n_rows=600).
    #[test]
    fn prefill_smem_stays_under_the_device_limit() {
        // Every projection shape in Qwen3.8-27B-Escha-W2, at every chunk size
        // the prefill driver can produce.
        let shapes: [(usize, usize); 8] = [
            (320, 10_240), // in_proj_qkv
            (320, 6_144),  // in_proj_z
            (384, 5_120),  // out_proj / o_proj
            (1088, 5_120), // down_proj  (nit is widest here)
            (320, 17_408), // gate_proj / up_proj
            (320, 12_288), // q_proj
            (320, 1_024),  // k_proj / v_proj
            (320, 5_120),
        ];
        for n_rows in [2usize, 4, 8, 17, 58, 128, 256, 600, 1024, 2048] {
            let r = escha_dense_prefill_r(n_rows);
            for (nit, oc) in shapes {
                let ns = escha_dense_n_slices_prefill(nit, oc, n_rows, r);
                for k in [2i32, 3] {
                    let smem = escha_dense_prefill_smem(nit, ns, k, r);
                    assert!(
                        smem <= ESCHA_DENSE_SMEM_BUDGET,
                        "nit={nit} oc={oc} n_rows={n_rows} R={r} K={k} n_slices={ns} \
                         asks for {smem} B of dynamic smem (limit {ESCHA_DENSE_SMEM_BUDGET})"
                    );
                }
            }
        }
    }

    /// The prefill slice heuristic must NOT floor `n_slices` at `R`.
    ///
    /// It used to (`n_slices = n_slices.max(r)`). Since the caller sizes
    /// `partial` as `n_slices * n_rows * OC` fp32, that floor pinned every
    /// projection of a 58-token chunk to `n_slices = 64` — 259 MB for gate/up
    /// and ~125 GB of split-K write+read per chunk. Total FMA work is
    /// independent of `n_slices`, so the floor bought only traffic.
    #[test]
    fn prefill_slices_are_not_floored_at_r() {
        let r = escha_dense_prefill_r(58);
        // gate_proj / up_proj: IC=5120 (nit=320), OC=17408.
        let ns = escha_dense_n_slices_prefill(320, 17_408, 58, r);
        assert!(
            ns < r as usize,
            "n_slices ({ns}) should be free to drop below R ({r}); the old floor \
             made the split-K partial ~259 MB for this projection"
        );
        let partial_bytes = ns * 58 * 17_408 * 4;
        assert!(
            partial_bytes < 64 << 20,
            "gate/up split-K partial is {} MB",
            partial_bytes / (1 << 20)
        );
    }

    /// `n_slices` must stay in `[1, nit]` and divide `nit`: the kernel
    /// partitions tiles across slices with integer bounds.
    #[test]
    fn prefill_slices_divide_tile_count() {
        for &(nit, oc, n_rows) in &[
            (320usize, 10_240usize, 58usize),
            (320, 6_144, 58),
            (384, 5_120, 58),
            (320, 12_288, 58),
            (320, 1_024, 58),
            (320, 17_408, 2),
            (1088, 5_120, 256),
        ] {
            let r = escha_dense_prefill_r(n_rows);
            let ns = escha_dense_n_slices_prefill(nit, oc, n_rows, r);
            assert!(ns >= 1 && ns <= nit, "nit={nit} oc={oc} n_slices={ns}");
            assert_eq!(
                nit % ns,
                0,
                "nit={nit} oc={oc}: n_slices={ns} must divide nit"
            );
        }
    }

    /// `R` must be a power of two within the instantiated range, or there is
    /// no kernel symbol to launch.
    #[test]
    fn prefill_r_is_always_instantiated() {
        for n_rows in [1usize, 2, 3, 5, 8, 17, 58, 64, 127, 256, 1024] {
            let r = escha_dense_prefill_r(n_rows);
            assert!(
                r > 0 && (r & (r - 1)) == 0,
                "n_rows={n_rows} -> R={r} not a power of two"
            );
            assert!(r <= ESCHA_DENSE_PREFILL_R_MAX, "R={r} exceeds the ceiling");
            for k in [2i32, 3] {
                assert!(
                    prefill_kernel_sym(k, r).is_some(),
                    "no kernel instantiated for K={k} R={r} (n_rows={n_rows})"
                );
            }
        }
        // K outside {2,3} and R outside the instantiated set are model/dispatch
        // errors, not runtime conditions — the dispatcher must refuse them.
        assert!(prefill_kernel_sym(4, 8).is_none());
        assert!(prefill_kernel_sym(2, 64).is_none());
    }
}

/// H128 input transform, single row: `out = f16round(H128(x . rin) * RS)`
/// widened to f32 (matches the batched kernel's value set). `n` must be a
/// multiple of 128. Ported entry `escha_h128_in` from PR #694
/// `kernels/src/escha_h128.hip` (verbatim); the batched forms are open work.
///
/// NOTE: `out` is an F32 tensor holding widened f16 values — do NOT pass an
/// F16 tensor here. The kernel writes `__half` (2 bytes/element); with an F16
/// `out` the wrapper's `numel()`-based size check passes but the kernel
/// writes half the bytes the caller expects. The batched kernel writes f32
/// directly; this single form inherits the F16 staging from G3 parity.
pub fn escha_h128_in(
    gpu: &mut Gpu,
    x: &GpuTensor,
    rin: &GpuTensor,
    out: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    escha_h128_single(gpu, "escha_h128_in", x, rin, out, n)
}

/// H128 output transform, single row: `out = f16round(H128(mid) * RS . rout)`
/// widened to f32. `rout` carries the per-channel prune mask (exact zeros stay
/// exact zero — see the kernel comment). Same provenance as above.
pub fn escha_h128_out(
    gpu: &mut Gpu,
    mid: &GpuTensor,
    rout: &GpuTensor,
    out: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    escha_h128_single(gpu, "escha_h128_out", mid, rout, out, n)
}

fn escha_h128_single(
    gpu: &mut Gpu,
    entry: &str,
    a: &GpuTensor,
    vec_in: &GpuTensor,
    out: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    use std::ffi::c_void;
    gpu.bind_thread()?;
    if n % 128 != 0 {
        return Err(hip_bridge::HipError::new(
            0,
            &format!("escha_h128_single: n={n} is not a multiple of 128"),
        ));
    }
    if a.numel() < n || vec_in.numel() < n || out.numel() < n {
        return Err(hip_bridge::HipError::new(
            0,
            &format!(
                "escha_h128_single: tensors too small (a={} vec={} out={} need {n})",
                a.numel(),
                vec_in.numel(),
                out.numel(),
            ),
        ));
    }
    gpu.ensure_kernel("escha_h128", &kernels::escha_h128_src(), entry)?;
    let mut a_ptr = a.buf.as_ptr();
    let mut v_ptr = vec_in.buf.as_ptr();
    let mut o_ptr = out.buf.as_ptr();
    let mut n_val = n as i32;
    let mut params: Vec<*mut c_void> = vec![
        &mut a_ptr as *mut _ as *mut c_void,
        &mut v_ptr as *mut _ as *mut c_void,
        &mut o_ptr as *mut _ as *mut c_void,
        &mut n_val as *mut _ as *mut c_void,
    ];
    gpu.launch_maybe_blob(
        entry,
        [(n / 128) as u32, 1, 1],
        [128, 1, 1],
        0,
        &mut params,
        || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(a_ptr);
            b.push_ptr(v_ptr);
            b.push_ptr(o_ptr);
            b.push_i32(n_val);
            b
        },
    )
}
