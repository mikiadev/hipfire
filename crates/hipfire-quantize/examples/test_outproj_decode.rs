// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GSQ-RCO Stage-4 decode-path isolation: `gemv_iq4_xs` on a REAL out_proj
//! (ssm_out) tensor with the GGUF-order un-permuted x, vs a CPU reference
//! (x_engine @ W_engine where W_engine is the f32-interleaved W).
//!
//! Pass: max_abs_err < 1e-3.
//! Usage: cargo run -p hipfire-quantize --example test_outproj_decode -- \
//!   /data/rocmfpx/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf

use hipfire_runtime::gguf::{GgmlType, GgufFile};
use std::path::Path;

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

const PERM: [usize; 48] = [
    0, 16, 32, 1, 17, 33, 2, 18, 34, 3, 19, 35, 4, 20, 36, 5, 21, 37, 6, 22, 38, 7, 23, 39, 8, 24,
    40, 9, 25, 41, 10, 26, 42, 11, 27, 43, 12, 28, 44, 13, 29, 45, 14, 30, 46, 15, 31, 47,
];

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/data/rocmfpx/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf".to_string());
    let gguf = GgufFile::open(Path::new(&path)).expect("open GGUF failed");
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");

    // Use blk.0.ssm_out.weight (IQ4_XS [5120, 6144]) — a real out_proj.
    let t = gguf
        .find_tensor("blk.0.ssm_out.weight")
        .expect("blk.0.ssm_out.weight not found");
    assert_eq!(t.dtype, GgmlType::IQ4XS, "expected IQ4_XS");
    // Engine convention: out_proj is [M=5120 output, K=6144 V-concat input].
    // (The runtime GGUF reader reports the shape reversed as [6144, 5120].)
    let m = 5120usize;
    let k = 6144usize;
    let raw = gguf.tensor_data(t);
    eprintln!("out_proj {} [{m} x {k}] raw_bytes={}", t.name, raw.len());

    // W_gguf = dequantized GGUF-order weights [m, k].
    let w_gguf = ref_dequant_iq4_xs(raw, m * k);

    // x_engine [k] in ENGINE V-head order (engine head e = GGUF head PERM[e]).
    let mut x_engine = vec![0.0f32; k];
    for e in 0..48 {
        for j in 0..128 {
            x_engine[e * 128 + j] = ((e * 128 + j) % 97) as f32 * 0.01 - 0.5;
        }
    }

    // Reference: y_ref = x_engine @ W_engine^T where W_engine[e] = W_gguf[PERM[e]].
    let mut y_ref = vec![0.0f32; m];
    for r in 0..m {
        let mut sum = 0.0f32;
        for e in 0..48 {
            let g = PERM[e];
            for j in 0..128 {
                sum += x_engine[e * 128 + j] * w_gguf[r * k + g * 128 + j];
            }
        }
        y_ref[r] = sum;
    }

    // GPU: un-permute x_engine -> x_gguf (GGUF order), then gemv_iq4_xs.
    let d_xe = gpu.upload_f32(&x_engine, &[k]).unwrap();
    let d_xg = gpu.alloc_tensor(&[k], rdna_compute::DType::F32).unwrap();
    gpu.vhead_unpermute_f32_batched(&d_xe, &d_xg, 1).unwrap();
    let d_raw = gpu.upload_raw(raw, &[raw.len()]).unwrap();
    let d_y = gpu.alloc_tensor(&[m], rdna_compute::DType::F32).unwrap();
    gpu.gemv_iq4_xs(&d_raw, &d_xg, &d_y, m, k).unwrap();
    let y_gpu = gpu.download_f32(&d_y).unwrap();

    let mut max_abs = 0.0f32;
    let mut bad = 0usize;
    for i in 0..m {
        let e = (y_gpu[i] - y_ref[i]).abs();
        if e > max_abs {
            max_abs = e;
        }
        if e > 1e-3 {
            bad += 1;
            if bad <= 5 {
                eprintln!("  row {i}: gpu={:.6} ref={:.6} err={:.6}", y_gpu[i], y_ref[i], e);
            }
        }
    }
    eprintln!("gemv_iq4_xs + unpermute: max_abs_err={max_abs:.8} bad={bad}/{m}");
    if max_abs < 1e-3 {
        println!("PASS: decode-path out_proj GEMV matches CPU reference");
    } else {
        println!("FAIL: decode-path out_proj GEMV diverges");
        std::process::exit(1);
    }
}