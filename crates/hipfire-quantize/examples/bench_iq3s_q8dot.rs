// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GSQ-RCO decode-next prototype bench: `gemv_iq3_s_q8dot` (q8_1 + dp4a)
//! vs `gemv_iq3_s_dualrow` (fp32 FMA), on a REAL IQ3_S tensor from the
//! release GGUF. Reports parity (CPU q8_1 oracle + fp32 dual-row) and
//! per-call timing for both paths, plus the q8_1 quantize cost.
//!
//! Usage:
//!   cargo run -p hipfire-quantize --example bench_iq3s_q8dot -- \
//!     /data/rocmfpx/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf [tensor-name-substring]
//!
//! Numerics note: q8_1 activations are quantized to int8 + per-32 scale, so
//! q8dot output differs from the fp32 dual-row at ~1e-2 rel. — that is the
//! expected cost of the dp4a mechanism (llama.cpp ships the same). The
//! kernel-vs-oracle check validates the kernel math itself.

use hipfire_runtime::gguf::{GgmlType, GgufFile};
use std::path::Path;
use std::time::Instant;

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

const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

const IQ3S_GRID: [u32; 512] = [
    0x01010101u32, 0x01010103u32, 0x01010105u32, 0x0101010Bu32, 0x0101010Fu32, 0x01010301u32, 0x01010303u32, 0x01010305u32,
    0x01010309u32, 0x0101030Du32, 0x01010501u32, 0x01010503u32, 0x0101050Bu32, 0x01010707u32, 0x01010901u32, 0x01010905u32,
    0x0101090Bu32, 0x0101090Fu32, 0x01010B03u32, 0x01010B07u32, 0x01010D01u32, 0x01010D05u32, 0x01010F03u32, 0x01010F09u32,
    0x01010F0Fu32, 0x01030101u32, 0x01030103u32, 0x01030105u32, 0x01030109u32, 0x01030301u32, 0x01030303u32, 0x0103030Bu32,
    0x01030501u32, 0x01030507u32, 0x0103050Fu32, 0x01030703u32, 0x0103070Bu32, 0x01030909u32, 0x01030D03u32, 0x01030D0Bu32,
    0x01030F05u32, 0x01050101u32, 0x01050103u32, 0x0105010Bu32, 0x0105010Fu32, 0x01050301u32, 0x01050307u32, 0x0105030Du32,
    0x01050503u32, 0x0105050Bu32, 0x01050701u32, 0x01050709u32, 0x01050905u32, 0x0105090Bu32, 0x0105090Fu32, 0x01050B03u32,
    0x01050B07u32, 0x01050F01u32, 0x01050F07u32, 0x01070107u32, 0x01070303u32, 0x0107030Bu32, 0x01070501u32, 0x01070505u32,
    0x01070703u32, 0x01070707u32, 0x0107070Du32, 0x01070909u32, 0x01070B01u32, 0x01070B05u32, 0x01070D0Fu32, 0x01070F03u32,
    0x01070F0Bu32, 0x01090101u32, 0x01090307u32, 0x0109030Fu32, 0x01090503u32, 0x01090509u32, 0x01090705u32, 0x01090901u32,
    0x01090907u32, 0x01090B03u32, 0x01090F01u32, 0x010B0105u32, 0x010B0109u32, 0x010B0501u32, 0x010B0505u32, 0x010B050Du32,
    0x010B0707u32, 0x010B0903u32, 0x010B090Bu32, 0x010B090Fu32, 0x010B0D0Du32, 0x010B0F07u32, 0x010D010Du32, 0x010D0303u32,
    0x010D0307u32, 0x010D0703u32, 0x010D0B05u32, 0x010D0F03u32, 0x010F0101u32, 0x010F0105u32, 0x010F0109u32, 0x010F0501u32,
    0x010F0505u32, 0x010F050Du32, 0x010F0707u32, 0x010F0B01u32, 0x010F0B09u32, 0x03010101u32, 0x03010103u32, 0x03010105u32,
    0x03010109u32, 0x03010301u32, 0x03010303u32, 0x03010307u32, 0x0301030Bu32, 0x0301030Fu32, 0x03010501u32, 0x03010505u32,
    0x03010703u32, 0x03010709u32, 0x0301070Du32, 0x03010B09u32, 0x03010B0Du32, 0x03010D03u32, 0x03010F05u32, 0x03030101u32,
    0x03030103u32, 0x03030107u32, 0x0303010Du32, 0x03030301u32, 0x03030309u32, 0x03030503u32, 0x03030701u32, 0x03030707u32,
    0x03030903u32, 0x03030B01u32, 0x03030B05u32, 0x03030F01u32, 0x03030F0Du32, 0x03050101u32, 0x03050305u32, 0x0305030Bu32,
    0x0305030Fu32, 0x03050501u32, 0x03050509u32, 0x03050705u32, 0x03050901u32, 0x03050907u32, 0x03050B0Bu32, 0x03050D01u32,
    0x03050F05u32, 0x03070103u32, 0x03070109u32, 0x0307010Fu32, 0x03070301u32, 0x03070307u32, 0x03070503u32, 0x0307050Fu32,
    0x03070701u32, 0x03070709u32, 0x03070903u32, 0x03070D05u32, 0x03070F01u32, 0x03090107u32, 0x0309010Bu32, 0x03090305u32,
    0x03090309u32, 0x03090703u32, 0x03090707u32, 0x03090905u32, 0x0309090Du32, 0x03090B01u32, 0x03090B09u32, 0x030B0103u32,
    0x030B0301u32, 0x030B0307u32, 0x030B0503u32, 0x030B0701u32, 0x030B0705u32, 0x030B0B03u32, 0x030D0501u32, 0x030D0509u32,
    0x030D050Fu32, 0x030D0909u32, 0x030D090Du32, 0x030F0103u32, 0x030F0107u32, 0x030F0301u32, 0x030F0305u32, 0x030F0503u32,
    0x030F070Bu32, 0x030F0903u32, 0x030F0D05u32, 0x030F0F01u32, 0x05010101u32, 0x05010103u32, 0x05010107u32, 0x0501010Bu32,
    0x0501010Fu32, 0x05010301u32, 0x05010305u32, 0x05010309u32, 0x0501030Du32, 0x05010503u32, 0x05010507u32, 0x0501050Fu32,
    0x05010701u32, 0x05010705u32, 0x05010903u32, 0x05010907u32, 0x0501090Bu32, 0x05010B01u32, 0x05010B05u32, 0x05010D0Fu32,
    0x05010F01u32, 0x05010F07u32, 0x05010F0Bu32, 0x05030101u32, 0x05030105u32, 0x05030301u32, 0x05030307u32, 0x0503030Fu32,
    0x05030505u32, 0x0503050Bu32, 0x05030703u32, 0x05030709u32, 0x05030905u32, 0x05030B03u32, 0x05050103u32, 0x05050109u32,
    0x0505010Fu32, 0x05050503u32, 0x05050507u32, 0x05050701u32, 0x0505070Fu32, 0x05050903u32, 0x05050B07u32, 0x05050B0Fu32,
    0x05050F03u32, 0x05050F09u32, 0x05070101u32, 0x05070105u32, 0x0507010Bu32, 0x05070303u32, 0x05070505u32, 0x05070509u32,
    0x05070703u32, 0x05070707u32, 0x05070905u32, 0x05070B01u32, 0x05070D0Du32, 0x05090103u32, 0x0509010Fu32, 0x05090501u32,
    0x05090507u32, 0x05090705u32, 0x0509070Bu32, 0x05090903u32, 0x05090F05u32, 0x05090F0Bu32, 0x050B0109u32, 0x050B0303u32,
    0x050B0505u32, 0x050B070Fu32, 0x050B0901u32, 0x050B0B07u32, 0x050B0F01u32, 0x050D0101u32, 0x050D0105u32, 0x050D010Fu32,
    0x050D0503u32, 0x050D0B0Bu32, 0x050D0D03u32, 0x050F010Bu32, 0x050F0303u32, 0x050F050Du32, 0x050F0701u32, 0x050F0907u32,
    0x050F0B01u32, 0x07010105u32, 0x07010303u32, 0x07010307u32, 0x0701030Bu32, 0x0701030Fu32, 0x07010505u32, 0x07010703u32,
    0x07010707u32, 0x0701070Bu32, 0x07010905u32, 0x07010909u32, 0x0701090Fu32, 0x07010B03u32, 0x07010D07u32, 0x07010F03u32,
    0x07030103u32, 0x07030107u32, 0x0703010Bu32, 0x07030309u32, 0x07030503u32, 0x07030507u32, 0x07030901u32, 0x07030D01u32,
    0x07030F05u32, 0x07030F0Du32, 0x07050101u32, 0x07050305u32, 0x07050501u32, 0x07050705u32, 0x07050709u32, 0x07050B01u32,
    0x07070103u32, 0x07070301u32, 0x07070309u32, 0x07070503u32, 0x07070507u32, 0x0707050Fu32, 0x07070701u32, 0x07070903u32,
    0x07070907u32, 0x0707090Fu32, 0x07070B0Bu32, 0x07070F07u32, 0x07090107u32, 0x07090303u32, 0x0709030Du32, 0x07090505u32,
    0x07090703u32, 0x07090B05u32, 0x07090D01u32, 0x07090D09u32, 0x070B0103u32, 0x070B0301u32, 0x070B0305u32, 0x070B050Bu32,
    0x070B0705u32, 0x070B0909u32, 0x070B0B0Du32, 0x070B0F07u32, 0x070D030Du32, 0x070D0903u32, 0x070F0103u32, 0x070F0107u32,
    0x070F0501u32, 0x070F0505u32, 0x070F070Bu32, 0x09010101u32, 0x09010109u32, 0x09010305u32, 0x09010501u32, 0x09010509u32,
    0x0901050Fu32, 0x09010705u32, 0x09010903u32, 0x09010B01u32, 0x09010F01u32, 0x09030105u32, 0x0903010Fu32, 0x09030303u32,
    0x09030307u32, 0x09030505u32, 0x09030701u32, 0x0903070Bu32, 0x09030907u32, 0x09030B03u32, 0x09030B0Bu32, 0x09050103u32,
    0x09050107u32, 0x09050301u32, 0x0905030Bu32, 0x09050503u32, 0x09050707u32, 0x09050901u32, 0x09050B0Fu32, 0x09050D05u32,
    0x09050F01u32, 0x09070109u32, 0x09070303u32, 0x09070307u32, 0x09070501u32, 0x09070505u32, 0x09070703u32, 0x0907070Bu32,
    0x09090101u32, 0x09090105u32, 0x09090509u32, 0x0909070Fu32, 0x09090901u32, 0x09090F03u32, 0x090B010Bu32, 0x090B010Fu32,
    0x090B0503u32, 0x090B0D05u32, 0x090D0307u32, 0x090D0709u32, 0x090D0D01u32, 0x090F0301u32, 0x090F030Bu32, 0x090F0701u32,
    0x090F0907u32, 0x090F0B03u32, 0x0B010105u32, 0x0B010301u32, 0x0B010309u32, 0x0B010505u32, 0x0B010901u32, 0x0B010909u32,
    0x0B01090Fu32, 0x0B010B05u32, 0x0B010D0Du32, 0x0B010F09u32, 0x0B030103u32, 0x0B030107u32, 0x0B03010Bu32, 0x0B030305u32,
    0x0B030503u32, 0x0B030705u32, 0x0B030F05u32, 0x0B050101u32, 0x0B050303u32, 0x0B050507u32, 0x0B050701u32, 0x0B05070Du32,
    0x0B050B07u32, 0x0B070105u32, 0x0B07010Fu32, 0x0B070301u32, 0x0B07050Fu32, 0x0B070909u32, 0x0B070B03u32, 0x0B070D0Bu32,
    0x0B070F07u32, 0x0B090103u32, 0x0B090109u32, 0x0B090501u32, 0x0B090705u32, 0x0B09090Du32, 0x0B0B0305u32, 0x0B0B050Du32,
    0x0B0B0B03u32, 0x0B0B0B07u32, 0x0B0D0905u32, 0x0B0F0105u32, 0x0B0F0109u32, 0x0B0F0505u32, 0x0D010303u32, 0x0D010307u32,
    0x0D01030Bu32, 0x0D010703u32, 0x0D010707u32, 0x0D010D01u32, 0x0D030101u32, 0x0D030501u32, 0x0D03050Fu32, 0x0D030D09u32,
    0x0D050305u32, 0x0D050709u32, 0x0D050905u32, 0x0D050B0Bu32, 0x0D050D05u32, 0x0D050F01u32, 0x0D070101u32, 0x0D070309u32,
    0x0D070503u32, 0x0D070901u32, 0x0D09050Bu32, 0x0D090907u32, 0x0D090D05u32, 0x0D0B0101u32, 0x0D0B0107u32, 0x0D0B0709u32,
    0x0D0B0D01u32, 0x0D0D010Bu32, 0x0D0D0901u32, 0x0D0F0303u32, 0x0D0F0307u32, 0x0F010101u32, 0x0F010109u32, 0x0F01010Fu32,
    0x0F010501u32, 0x0F010505u32, 0x0F01070Du32, 0x0F010901u32, 0x0F010B09u32, 0x0F010D05u32, 0x0F030105u32, 0x0F030303u32,
    0x0F030509u32, 0x0F030907u32, 0x0F03090Bu32, 0x0F050103u32, 0x0F050109u32, 0x0F050301u32, 0x0F05030Du32, 0x0F050503u32,
    0x0F050701u32, 0x0F050B03u32, 0x0F070105u32, 0x0F070705u32, 0x0F07070Bu32, 0x0F070B07u32, 0x0F090103u32, 0x0F09010Bu32,
    0x0F090307u32, 0x0F090501u32, 0x0F090B01u32, 0x0F0B0505u32, 0x0F0B0905u32, 0x0F0D0105u32, 0x0F0D0703u32, 0x0F0F0101u32,
];

/// CPU q8_1 oracle for the IQ3_S layout, mirroring gemv_iq3_s_q8dot exactly:
/// per 256-block, thread mapping tid = ib*4 + l, element e = ib*32 + l*8 + k.
fn cpu_q8dot_iq3s(data: &[u8], xs: &[i8], xscales: &[f32], m: usize, k: usize) -> Vec<f32> {
    const QK: usize = 256;
    const BLK: usize = 110;
    let blocks_per_row = k / QK;
    let mut out = vec![0.0f32; m];
    for r in 0..m {
        let row = &data[r * blocks_per_row * BLK..];
        let mut acc = 0.0f32;
        for bg in 0..blocks_per_row {
            let base = bg * BLK;
            let d = f16_to_f32(u16::from_le_bytes([row[base], row[base + 1]]));
            for tid in 0..32usize {
                let ib = tid >> 2;
                let l = tid & 3;
                let i = ib >> 1;
                let half = ib & 1;
                let qs_base = i * 16 + half * 8 + 2 * l;
                let shift_lo = 8usize.wrapping_sub(2 * l);
                let shift_hi = 7usize.wrapping_sub(2 * l);
                let qh = row[base + 66 + ib] as usize;
                let g0 = IQ3S_GRID[row[base + 2 + qs_base] as usize | ((qh << shift_lo) & 256)];
                let g1 = IQ3S_GRID[row[base + 2 + qs_base + 1] as usize | ((qh << shift_hi) & 256)];
                let s = row[base + 74 + i * 8 + half * 4 + l];
                let sc = row[base + 106 + i];
                let nib = if half == 0 { sc & 0xf } else { sc >> 4 };
                let db = d * (1.0 + 2.0 * nib as f32);
                for kk in 0..8usize {
                    let e = bg * QK + ib * 32 + l * 8 + kk;
                    let gb = if kk < 4 { g0 } else { g1 };
                    let v = (gb >> (8 * (kk & 3))) & 0xff;
                    let v = v as u8 as i8 as f32;
                    let sign = if s & KMASK_IQ2XS[kk] != 0 { -1.0 } else { 1.0 };
                    acc += db * v * sign * (xs[e] as f32) * xscales[e / 32];
                }
            }
        }
        out[r] = acc;
    }
    out
}

fn cpu_quantize_q8_1(x: &[f32], k: usize) -> (Vec<i8>, Vec<f32>) {
    let mut xs = vec![0i8; k];
    let mut scales = vec![0.0f32; k / 32];
    for g in 0..k / 32 {
        let base = g * 32;
        let mut a = 0.0f32;
        for j in 0..32 {
            a = a.max(x[base + j].abs());
        }
        let inv = if a > 0.0 { 127.0 / a } else { 0.0 };
        scales[g] = if a > 0.0 { a / 127.0 } else { 1.0 };
        for j in 0..32 {
            xs[base + j] = (x[base + j] * inv).round().clamp(-128.0, 127.0) as i8;
        }
    }
    (xs, scales)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let ma = max_abs(a, b);
    let norm = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-12);
    ma / norm
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path: String = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/data/rocmfpx/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf".to_string());
    let filter = args.get(2).cloned();
    // Optional quant selector: iq3s (default) | iq3xxs | iq4xs
    let quant = args.get(3).cloned().unwrap_or_else(|| "iq3s".to_string());
    let ggml = match quant.as_str() {
        "iq3xxs" => GgmlType::IQ3XXS,
        "iq4xs" => GgmlType::IQ4XS,
        _ => GgmlType::IQ3S,
    };

    let gguf = GgufFile::open(Path::new(&path)).expect("open GGUF failed");
    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    eprintln!("GPU: {}", gpu.arch);

    // Prefer a large IQ3_S 2D tensor; allow a name filter (e.g. in_proj_qkv).
    let mut found = None;
    for t in &gguf.tensors {
        if t.dtype == ggml && t.shape.len() == 2 && t.shape[1] % 256 == 0 {
            if let Some(f) = &filter {
                if !t.name.contains(f.as_str()) {
                    continue;
                }
            }
            found = Some(t);
            if filter.is_some() {
                break;
            }
            if t.name.contains("in_proj_qkv") {
                found = Some(t);
                break;
            }
        }
    }
    let Some(t) = found else {
        eprintln!("no matching {} 2D tensor", quant);
        std::process::exit(2);
    };
    let m = t.shape[0];
    let k = t.shape[1];
    let raw = gguf.tensor_data(t);
    eprintln!("=== {}: {} [{} x {}] ===", quant, t.name, m, k);

    // Deterministic activation with realistic magnitude (~N(0,1)-ish).
    let mut x_data: Vec<f32> = (0..k).map(|i| {
        let u = ((i as u64).wrapping_mul(2654435761) >> 33) as f32 / (1u32 << 31) as f32;
        (u * 2.0 - 1.0) * 0.5
    }).collect();
    // make it non-trivial: mix a DC and a ramp so max-abs groups vary
    for i in 0..k {
        x_data[i] += 0.25f32 * ((i % 97) as f32 / 96.0);
    }

    // CPU q8_1 oracle.
    let (xs_cpu, sc_cpu) = cpu_quantize_q8_1(&x_data, k);
    // CPU oracle only exists for IQ3_S; for the other dtypes correctness is
    // checked q8dot-vs-dualrow (within the q8_1 activation delta).
    let y_oracle = if quant == "iq3s" {
        Some(cpu_q8dot_iq3s(raw, &xs_cpu, &sc_cpu, m, k))
    } else {
        None
    };

    // Uploads.
    let d_raw = gpu.upload_raw(raw, &[raw.len()]).unwrap();
    let d_x = gpu.upload_f32(&x_data, &[k]).unwrap();
    let d_xs = gpu.upload_raw(&vec![0u8; k], &[k]).unwrap();
    let d_sc = gpu.upload_f32(&vec![0.0f32; k / 32], &[k / 32]).unwrap();
    let d_yd = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();
    let d_yq = gpu.zeros(&[m], rdna_compute::DType::F32).unwrap();

    // Run: quantize (GPU) + q8dot, and dual-row fp32.
    gpu.quantize_q8_1(&d_x, &d_xs, &d_sc, k).unwrap();
    match quant.as_str() {
        "iq3xxs" => {
            gpu.gemv_iq3_xxs_q8dot(&d_raw, &d_xs, &d_sc, &d_yq, m, k).unwrap();
            gpu.gemv_iq3_xxs_dualrow(&d_raw, &d_x, &d_yd, m, k).unwrap();
        }
        "iq4xs" => {
            gpu.gemv_iq4_xs_q8dot(&d_raw, &d_xs, &d_sc, &d_yq, m, k).unwrap();
            gpu.gemv_iq4_xs_dualrow(&d_raw, &d_x, &d_yd, m, k).unwrap();
        }
        _ => {
            gpu.gemv_iq3_s_q8dot(&d_raw, &d_xs, &d_sc, &d_yq, m, k).unwrap();
            gpu.gemv_iq3_s_dualrow(&d_raw, &d_x, &d_yd, m, k).unwrap();
        }
    }

    let y_q8 = gpu.download_f32(&d_yq).unwrap();
    let y_dual = gpu.download_f32(&d_yd).unwrap();

    // Verify the GPU q8_1 quantize against the CPU oracle inputs.
    // download_raw is not exposed; read the int8 buffer via a raw dtoh copy.
    let xs_gpu = gpu.download_raw(&d_xs).unwrap();
    let sc_gpu = gpu.download_f32(&d_sc).unwrap();
    let xs_mismatch = xs_gpu
        .iter()
        .zip(xs_cpu.iter())
        .filter(|(a, b)| **a as i8 != **b)
        .count();
    let sc_err = max_abs(&sc_gpu, &sc_cpu);
    eprintln!("  quantize: xs mismatches={xs_mismatch}/{}  sc max_abs={sc_err:.8}", k);

    let q8_vs_oracle = y_oracle.as_ref().map(|o| max_abs(&y_q8, o));
    let q8_vs_dual = max_abs(&y_q8, &y_dual);
    let dual_mag = y_dual.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    if let Some(v) = q8_vs_oracle {
        eprintln!("  q8dot  vs CPU q8_1 oracle  max_abs={v:.6}  rel={:.6}",
            v / dual_mag.max(1e-12));
    }
    eprintln!("  q8dot  vs dualrow (fp32)   max_abs={q8_vs_dual:.6}  rel={:.6}",
        q8_vs_dual / dual_mag.max(1e-12));
    eprintln!("  dualrow magnitude max={dual_mag:.4}");
    if let Some(v) = q8_vs_oracle {
        if v > 1e-3 * dual_mag.max(1e-12) {
            eprintln!("  FAIL: q8dot kernel deviates from CPU oracle beyond rounding");
        }
    }

    // Residual variant: y += A @ x_q8_1 vs dualrow residual.
    let y_init: Vec<f32> = (0..m).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
    let d_yr_q = gpu.upload_f32(&y_init, &[m]).unwrap();
    let d_yr_d = gpu.upload_f32(&y_init, &[m]).unwrap();
    match quant.as_str() {
        "iq3xxs" => {
            gpu.gemv_iq3_xxs_q8dot_residual(&d_raw, &d_xs, &d_sc, &d_yr_q, m, k).unwrap();
            gpu.gemv_iq3_xxs_dualrow_residual(&d_raw, &d_x, &d_yr_d, m, k).unwrap();
        }
        "iq4xs" => {
            gpu.gemv_iq4_xs_q8dot_residual(&d_raw, &d_xs, &d_sc, &d_yr_q, m, k).unwrap();
            gpu.gemv_iq4_xs_dualrow_residual(&d_raw, &d_x, &d_yr_d, m, k).unwrap();
        }
        _ => {
            gpu.gemv_iq3_s_q8dot_residual(&d_raw, &d_xs, &d_sc, &d_yr_q, m, k).unwrap();
            gpu.gemv_iq3_s_dualrow_residual(&d_raw, &d_x, &d_yr_d, m, k).unwrap();
        }
    }
    let yr_q = gpu.download_f32(&d_yr_q).unwrap();
    let yr_d = gpu.download_f32(&d_yr_d).unwrap();
    // oracle: y_init + oracle-dot
    if let Some(o) = y_oracle.as_ref() {
        let yr_oracle: Vec<f32> = y_init.iter().zip(o.iter()).map(|(a, b)| a + b).collect();
        eprintln!("  residual q8dot vs oracle  max_abs={:.6}", max_abs(&yr_q, &yr_oracle));
    }
    eprintln!("  residual q8dot vs dualrow max_abs={:.6}", max_abs(&yr_q, &yr_d));

    // Timing: N iterations each, sync via download.
    let iters = 200usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        match quant.as_str() {
            "iq3xxs" => gpu.gemv_iq3_xxs_dualrow(&d_raw, &d_x, &d_yd, m, k).unwrap(),
            "iq4xs" => gpu.gemv_iq4_xs_dualrow(&d_raw, &d_x, &d_yd, m, k).unwrap(),
            _ => gpu.gemv_iq3_s_dualrow(&d_raw, &d_x, &d_yd, m, k).unwrap(),
        }
    }
    let _ = gpu.download_f32(&d_yd).unwrap();
    let t_dual = t0.elapsed().as_secs_f64() / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        gpu.quantize_q8_1(&d_x, &d_xs, &d_sc, k).unwrap();
        match quant.as_str() {
            "iq3xxs" => gpu.gemv_iq3_xxs_q8dot(&d_raw, &d_xs, &d_sc, &d_yq, m, k).unwrap(),
            "iq4xs" => gpu.gemv_iq4_xs_q8dot(&d_raw, &d_xs, &d_sc, &d_yq, m, k).unwrap(),
            _ => gpu.gemv_iq3_s_q8dot(&d_raw, &d_xs, &d_sc, &d_yq, m, k).unwrap(),
        }
    }
    let _ = gpu.download_f32(&d_yq).unwrap();
    let t_q8 = t1.elapsed().as_secs_f64() / iters as f64;

    // Isolate the quantize cost.
    let t2 = Instant::now();
    for _ in 0..iters {
        gpu.quantize_q8_1(&d_x, &d_xs, &d_sc, k).unwrap();
    }
    let _ = gpu.download_f32(&d_sc).unwrap();
    let t_quant = t2.elapsed().as_secs_f64() / iters as f64;

    // Split-K experiment (IQ3_S only): time the split-K variant at 1/2/4.
    if quant == "iq3s" {
        for sp in [1usize, 2, 4] {
            let ts = Instant::now();
            for _ in 0..iters {
                gpu.gemv_iq3_s_q8dot_split(&d_raw, &d_xs, &d_sc, &d_yq, m, k, sp).unwrap();
            }
            let _ = gpu.download_f32(&d_yq).unwrap();
            eprintln!("  q8dot split={sp}: {:.1} us/call", ts.elapsed().as_secs_f64() / iters as f64 * 1e6);
        }
    }

    eprintln!("  dualrow fp32 : {:.1} us/call", t_dual * 1e6);
    eprintln!("  q8dot (incl. quantize): {:.1} us/call", t_q8 * 1e6);
    eprintln!("  quantize_q8_1: {:.1} us/call", t_quant * 1e6);
    eprintln!("  q8dot alone  : {:.1} us/call", (t_q8 - t_quant) * 1e6);
    eprintln!("  speedup (q8dot incl. quantize vs dualrow): {:.2}x", t_dual / t_q8);
    eprintln!("  speedup (q8dot alone vs dualrow): {:.2}x", t_dual / (t_q8 - t_quant));
}