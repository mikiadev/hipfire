// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Stage-1 GSQ-RCO native parity harness: `gemv_iq4_xs` / `gemm_iq4_xs_batched`
//! / `gemv_q2k` / `gemm_q2k_batched` vs the CPU decoders in `gguf_iq.rs`.
//!
//! Reads real IQ4_XS + Q2_K tensors from the release GGUF, runs the GPU
//! kernels, and compares against a CPU dequant + dot-product reference.
//! The reference decoders below are line-ported from
//! `crates/hipfire-quantize/src/gguf_iq.rs::dequant_iq4_xs` /
//! `dequant_q2_k` (which are themselves C-verified against llama.cpp-escha
//! `ggml-quants.c`); the gguf_iq.rs versions remain the canonical oracle and
//! any drift here shows up as a parity failure.
//!
//! Pass criteria: max_abs_err < 1e-3 (fp32 accumulate vs CPU f32 accumulate
//! can differ in FMA order; 1e-3 is far tighter than the 0.05 the Q4K
//! harness uses and still loose enough for fp32 rounding).
//!
//! Usage:
//!   cargo run -p hipfire-quantize --example test_gemv_iq_native -- \
//!     /data/rocmfpx/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf

use hipfire_runtime::gguf::{GgmlType, GgufFile, TensorInfo};
use std::path::Path;

// ── CPU reference decoders (mirror gguf_iq.rs) ──────────────────────────

const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let frac = h & 0x3ff;
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits((sign as u32) << 31);
        }
        let mut f = frac;
        let mut e = exp as i32;
        while f & 0x400 == 0 {
            f <<= 1;
            e -= 1;
        }
        f &= 0x3ff;
        e = 127 - 15 + 1 + e;
        return f32::from_bits((sign as u32) << 31 | (e as u32) << 23 | (f as u32) << 13);
    }
    if exp == 31 {
        return f32::from_bits((sign as u32) << 31 | 0x7f800000u32 | (frac as u32) << 13);
    }
    f32::from_bits((sign as u32) << 31 | ((exp as u32 + 127 - 15) << 23) | (frac as u32) << 13)
}

/// Port of `gguf_iq.rs::dequant_iq4_xs` (136 B per 256, ggml type 23).
fn ref_dequant_iq4_xs(data: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 256;
    const BLK: usize = 136;
    let nblocks = n.div_ceil(QK);
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let base = b * BLK;
        if base + BLK > data.len() {
            break;
        }
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let scales_h = u16::from_le_bytes([data[base + 2], data[base + 3]]);
        let scales_l = &data[base + 4..base + 8];
        let qs = &data[base + 8..base + 136];
        let mut y = b * QK;
        let mut qs_off = 0usize;
        for ib in 0..8 {
            let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32
                | (((scales_h >> (2 * ib)) & 3) as i32) << 4;
            let dl = d * ((ls - 32) as f32);
            for j in 0..16 {
                let qb = qs[qs_off + j];
                let idx1 = y + j;
                let idx2 = y + 16 + j;
                if idx1 < n {
                    out[idx1] = dl * f32::from(KVALUES_IQ4NL[(qb & 0xf) as usize]);
                }
                if idx2 < n {
                    out[idx2] = dl * f32::from(KVALUES_IQ4NL[(qb >> 4) as usize]);
                }
            }
            y += 32;
            qs_off += 16;
        }
    }
    out
}

/// Port of `gguf_iq.rs::dequant_q2_k` (84 B per 256, ggml type 10).
fn ref_dequant_q2_k(data: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 256;
    const BLK: usize = 84;
    let nblocks = n.div_ceil(QK);
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let base = b * BLK;
        if base + BLK > data.len() {
            break;
        }
        let scales = &data[base..base + 16];
        let qs = &data[base + 16..base + 80];
        let d = f16_to_f32(u16::from_le_bytes([data[base + 80], data[base + 81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[base + 82], data[base + 83]]));
        let mut is = 0usize;
        let mut y = b * QK;
        for half in 0..2 {
            let q_off = half * 32;
            let mut shift = 0u32;
            for _j in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * (f32::from(sc & 0xF));
                let ml = dmin * (f32::from(sc >> 4));
                for l in 0..16 {
                    let v = ((qs[q_off + l] >> shift) & 3) as i8 as f32;
                    let idx = y + l;
                    if idx < n {
                        out[idx] = dl * v - ml;
                    }
                }
                y += 16;
                let sc = scales[is];
                is += 1;
                let dl = d * (f32::from(sc & 0xF));
                let ml = dmin * (f32::from(sc >> 4));
                for l in 0..16 {
                    let v = ((qs[q_off + 16 + l] >> shift) & 3) as i8 as f32;
                    let idx = y + l;
                    if idx < n {
                        out[idx] = dl * v - ml;
                    }
                }
                y += 16;
                shift += 2;
            }
        }
    }
    out
}

// ── Harness ────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path: String = if args.len() >= 2 {
        args.get(1).unwrap().to_string()
    } else {
        "/data/rocmfpx/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf".to_string()
    };
    let gguf = GgufFile::open(Path::new(&path)).expect("open GGUF failed");
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {} ({:.1} GB VRAM)", gpu.arch, {
        let (_, total) = gpu.hip.get_vram_info().unwrap_or((0, 0));
        total as f64 / 1e9
    });

    let mut passed = 0;
    let mut failed = 0;

    macro_rules! run_case {
        ($label:expr, $dtype:expr, $dequant:expr) => {{
            // Find a 2D tensor of the target dtype with K%256==0.
            let mut found: Option<&TensorInfo> = None;
            for t in &gguf.tensors {
                if t.dtype == $dtype && t.shape.len() == 2 && t.shape[1] % 256 == 0 {
                    found = Some(t);
                    break;
                }
            }
            let Some(t) = found else {
                eprintln!("  SKIP: no {} tensor in file", $label);
                return;
            };
            let m = t.shape[0];
            let k = t.shape[1];
            let raw = gguf.tensor_data(t);
            eprintln!("\n=== {}: {} [{} x {}] raw_bytes={} ===", $label, t.name, m, k, raw.len());

            // CPU reference: dequant + dot product per row.
            let a_f32 = $dequant(raw, m * k);
            let x_data: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
            let mut y_ref = vec![0.0f32; m];
            for i in 0..m {
                let mut sum = 0.0f32;
                for j in 0..k {
                    sum += a_f32[i * k + j] * x_data[j];
                }
                y_ref[i] = sum;
            }

            // GPU scalar GEMV.
            let d_raw = gpu.upload_raw(raw, &[raw.len()]).unwrap();
            let d_x = gpu.upload_f32(&x_data, &[k]).unwrap();
            let d_y = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
            match $dtype {
                GgmlType::IQ4XS => gpu.gemv_iq4_xs(&d_raw, &d_x, &d_y, m, k).unwrap(),
                GgmlType::Q2K => gpu.gemv_q2k(&d_raw, &d_x, &d_y, m, k).unwrap(),
                _ => panic!("unreachable"),
            }
            let y_gpu = gpu.download_f32(&d_y).unwrap();

            let mut max_abs_err: f32 = 0.0;
            let mut errors = 0;
            for i in 0..m {
                let abs_err = (y_gpu[i] - y_ref[i]).abs();
                max_abs_err = max_abs_err.max(abs_err);
                if abs_err > 1e-3 {
                    errors += 1;
                    if errors <= 3 {
                        eprintln!("  row {i}: gpu={:.6} ref={:.6} err={:.6}", y_gpu[i], y_ref[i], abs_err);
                    }
                }
            }
            eprintln!("  GEMV max_abs_err={max_abs_err:.8} errors={errors}/{m}");
            if max_abs_err < 1e-3 {
                eprintln!("  GEMV PASS");
                passed += 1;
            } else {
                eprintln!("  GEMV FAIL");
                failed += 1;
            }

            // GPU batched GEMM (batch sweep incl. chunk boundary) vs the
            // same reference per row.
            for &batch in &[1usize, 8usize, 19usize, 64usize, 65usize] {
            let mut x_batch = Vec::new();
            for _ in 0..batch {
                x_batch.extend_from_slice(&x_data);
            }
            let d_xb = gpu.upload_f32(&x_batch, &[batch, k]).unwrap();
            let d_yb = gpu.zeros(&[batch, m], rdna_compute::DType::F32).unwrap();
            match $dtype {
                GgmlType::IQ4XS => {
                    gpu.gemm_iq4_xs_batched(&d_raw, &d_xb, &d_yb, m, k, batch).unwrap()
                }
                GgmlType::Q2K => {
                    gpu.gemm_q2k_batched(&d_raw, &d_xb, &d_yb, m, k, batch).unwrap()
                }
                _ => panic!("unreachable"),
            }
            // Sub-view test: allocate a padded x buffer, pass a sub_offset view.
            {
                let mut padded_data = vec![0.0f32; (batch + 4) * k];
                for b in 0..batch {
                    for j in 0..k {
                        padded_data[(b + 4) * k + j] = x_data[j];
                    }
                }
                let padded = gpu.upload_f32(&padded_data, &[(batch + 4), k]).unwrap();
                let view = padded.sub_offset(4 * k, batch * k);
                let d_yv = gpu.zeros(&[batch, m], rdna_compute::DType::F32).unwrap();
                match $dtype {
                    GgmlType::IQ4XS => {
                        gpu.gemm_iq4_xs_batched(&d_raw, &view, &d_yv, m, k, batch).unwrap()
                    }
                    GgmlType::Q2K => {
                        gpu.gemm_q2k_batched(&d_raw, &view, &d_yv, m, k, batch).unwrap()
                    }
                    _ => panic!("unreachable"),
                }
                let yv = gpu.download_f32(&d_yv).unwrap();
                // Sub-view rows equal the CPU reference (same x values as x_data).
                let mut vmax: f32 = 0.0;
                for b in 0..batch {
                    for i in 0..m {
                        vmax = vmax.max((yv[b * m + i] - y_ref[i]).abs());
                    }
                }
                eprintln!("  GEMM sub-view batch={batch} max_abs_err={vmax:.8}");
                if vmax < 1e-3 {
                    eprintln!("  GEMM sub-view PASS");
                    passed += 1;
                } else {
                    eprintln!("  GEMM sub-view FAIL");
                    failed += 1;
                }
                gpu.free_tensor(padded).unwrap();
                gpu.free_tensor(d_yv).unwrap();
            }
            let y_batch = gpu.download_f32(&d_yb).unwrap();
            let mut bmax: f32 = 0.0;
            let mut berrors = 0;
            for b in 0..batch {
                for i in 0..m {
                    let abs_err = (y_batch[b * m + i] - y_ref[i]).abs();
                    bmax = bmax.max(abs_err);
                    if abs_err > 1e-3 {
                        berrors += 1;
                    }
                }
            }
            eprintln!("  GEMM batch={batch} max_abs_err={bmax:.8} errors={berrors}/{}", batch * m);
                if bmax < 1e-3 {
                    eprintln!("  GEMM batch={batch} PASS");
                    passed += 1;
                } else {
                    eprintln!("  GEMM batch={batch} FAIL");
                    failed += 1;
                }
                gpu.free_tensor(d_xb).unwrap();
                gpu.free_tensor(d_yb).unwrap();
            }

            gpu.free_tensor(d_raw).unwrap();
            gpu.free_tensor(d_x).unwrap();
            gpu.free_tensor(d_y).unwrap();
        }};
    }

    run_case!("IQ4_XS", GgmlType::IQ4XS, |d, n| ref_dequant_iq4_xs(d, n));
    run_case!("Q2_K", GgmlType::Q2K, |d, n| ref_dequant_q2_k(d, n));



    eprintln!("\n=== RESULT: {passed} passed, {failed} failed ===");
    if failed > 0 {
        std::process::exit(1);
    }
}