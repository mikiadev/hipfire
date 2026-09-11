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
    0x0F090307u32, 0x0F090501u32, 0x0F090B01u32, 0x0F0B0505u32, 0x0F0B0905u32, 0x0F0D0105u32, 0x0F0D0703u32, 0x0F0F0101u32
];

/// Port of `gguf_input.rs::dequant_iq3_s` (110 B per 256, ggml type 21).
fn ref_dequant_iq3_s(data: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 256;
    const BLK: usize = 110;
    let nblocks = n.div_ceil(QK);
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let base = b * BLK;
        if base + BLK > data.len() {
            break;
        }
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let qs = &data[base + 2..base + 66];
        let qh = &data[base + 66..base + 74];
        let signs = &data[base + 74..base + 106];
        let scales = &data[base + 106..base + 110];
        let mut y = b * QK;
        let mut qs_off = 0usize;
        let mut signs_off = 0usize;
        let mut qh_off = 0usize;
        for ib in 0..4 {
            let sc = scales[ib];
            for half_q in 0..2 {
                let db = d
                    * (1.0
                        + 2.0
                            * (if half_q == 0 { sc & 0xf } else { sc >> 4 }) as f32);
                for l in 0..4 {
                    let qh_byte = qh[qh_off + half_q];
                    let g1 = IQ3S_GRID[(qs[qs_off + 2 * l] as usize)
                        | (((qh_byte as usize) << ((8 - 2 * l) as usize)) & 256)];
                    let g2 = IQ3S_GRID[(qs[qs_off + 2 * l + 1] as usize)
                        | (((qh_byte as usize) << ((7 - 2 * l) as usize)) & 256)];
                    let s = signs[signs_off + l];
                    for j in 0..4 {
                        let v1 = ((g1 >> (8 * j)) & 0xff) as u8 as i8 as f32;
                        let v2 = ((g2 >> (8 * j)) & 0xff) as u8 as i8 as f32;
                        let idx1 = y + j;
                        let idx2 = y + 4 + j;
                        if idx1 < n {
                            out[idx1] = db * v1 * (if s & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 });
                        }
                        if idx2 < n {
                            out[idx2] =
                                db * v2 * (if s & KMASK_IQ2XS[4 + j] != 0 { -1.0 } else { 1.0 });
                        }
                    }
                    y += 8;
                }
                qs_off += 8;
                signs_off += 4;
            }
            qh_off += 2;
        }
    }
    out
}


/// Port of `gguf_input.rs::dequant_q4_k` (144 B per 256, ggml type 12).
fn ref_dequant_q4_k(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 256;
    let block_bytes = 144;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let off = b * block_bytes;
        if off + block_bytes > data.len() {
            break;
        }
        let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[off + 2], data[off + 3]]));
        let sc_data = &data[off + 4..off + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for i in 0..4 {
            scales[i] = sc_data[i] & 63;
            mins[i] = sc_data[4 + i] & 63;
        }
        for i in 0..4 {
            scales[4 + i] = (sc_data[8 + i] & 0xF) | ((sc_data[i] >> 6) << 4);
            mins[4 + i] = (sc_data[8 + i] >> 4) | ((sc_data[4 + i] >> 6) << 4);
        }
        let qdata = &data[off + 16..off + 16 + 128];
        for group in 0..4 {
            let sb_even = group * 2;
            let sb_odd = group * 2 + 1;
            let sc_even = d * scales[sb_even] as f32;
            let m_even = dmin * mins[sb_even] as f32;
            let sc_odd = d * scales[sb_odd] as f32;
            let m_odd = dmin * mins[sb_odd] as f32;
            for l in 0..32 {
                let byte = qdata[group * 32 + l];
                let idx_even = b * block_size + group * 64 + l;
                let idx_odd = idx_even + 32;
                if idx_even < n {
                    out[idx_even] = (byte & 0x0F) as f32 * sc_even - m_even;
                }
                if idx_odd < n {
                    out[idx_odd] = ((byte >> 4) & 0x0F) as f32 * sc_odd - m_odd;
                }
            }
        }
    }
    out
}


/// Port of `gguf_iq.rs::dequant_iq3_xxs` (98 B per 256, ggml type 18).
const IQ3XXS_GRID: [u32; 256] = [
    0x04040404u32, 0x04040414u32, 0x04040424u32, 0x04040C0Cu32, 0x04040C1Cu32, 0x04040C3Eu32, 0x04041404u32, 0x04041414u32,
    0x04041C0Cu32, 0x04042414u32, 0x04043E1Cu32, 0x04043E2Cu32, 0x040C040Cu32, 0x040C041Cu32, 0x040C0C04u32, 0x040C0C14u32,
    0x040C140Cu32, 0x040C142Cu32, 0x040C1C04u32, 0x040C1C14u32, 0x040C240Cu32, 0x040C2C24u32, 0x040C3E04u32, 0x04140404u32,
    0x04140414u32, 0x04140424u32, 0x04140C0Cu32, 0x04141404u32, 0x04141414u32, 0x04141C0Cu32, 0x04141C1Cu32, 0x04141C3Eu32,
    0x04142C0Cu32, 0x04142C3Eu32, 0x04143E2Cu32, 0x041C040Cu32, 0x041C043Eu32, 0x041C0C04u32, 0x041C0C14u32, 0x041C142Cu32,
    0x041C3E04u32, 0x04240C1Cu32, 0x04241C3Eu32, 0x04242424u32, 0x04242C3Eu32, 0x04243E1Cu32, 0x04243E2Cu32, 0x042C040Cu32,
    0x042C043Eu32, 0x042C1C14u32, 0x042C2C14u32, 0x04341C2Cu32, 0x04343424u32, 0x043E0C04u32, 0x043E0C24u32, 0x043E0C34u32,
    0x043E241Cu32, 0x043E340Cu32, 0x0C04040Cu32, 0x0C04041Cu32, 0x0C040C04u32, 0x0C040C14u32, 0x0C04140Cu32, 0x0C04141Cu32,
    0x0C041C04u32, 0x0C041C14u32, 0x0C041C24u32, 0x0C04243Eu32, 0x0C042C04u32, 0x0C0C0404u32, 0x0C0C0414u32, 0x0C0C0C0Cu32,
    0x0C0C1404u32, 0x0C0C1414u32, 0x0C14040Cu32, 0x0C14041Cu32, 0x0C140C04u32, 0x0C140C14u32, 0x0C14140Cu32, 0x0C141C04u32,
    0x0C143E14u32, 0x0C1C0404u32, 0x0C1C0414u32, 0x0C1C1404u32, 0x0C1C1C0Cu32, 0x0C1C2434u32, 0x0C1C3434u32, 0x0C24040Cu32,
    0x0C24042Cu32, 0x0C242C04u32, 0x0C2C1404u32, 0x0C2C1424u32, 0x0C2C2434u32, 0x0C2C3E0Cu32, 0x0C34042Cu32, 0x0C3E1414u32,
    0x0C3E2404u32, 0x14040404u32, 0x14040414u32, 0x14040C0Cu32, 0x14040C1Cu32, 0x14041404u32, 0x14041414u32, 0x14041434u32,
    0x14041C0Cu32, 0x14042414u32, 0x140C040Cu32, 0x140C041Cu32, 0x140C042Cu32, 0x140C0C04u32, 0x140C0C14u32, 0x140C140Cu32,
    0x140C1C04u32, 0x140C341Cu32, 0x140C343Eu32, 0x140C3E04u32, 0x14140404u32, 0x14140414u32, 0x14140C0Cu32, 0x14140C3Eu32,
    0x14141404u32, 0x14141414u32, 0x14141C3Eu32, 0x14142404u32, 0x14142C2Cu32, 0x141C040Cu32, 0x141C0C04u32, 0x141C0C24u32,
    0x141C3E04u32, 0x141C3E24u32, 0x14241C2Cu32, 0x14242C1Cu32, 0x142C041Cu32, 0x142C143Eu32, 0x142C240Cu32, 0x142C3E24u32,
    0x143E040Cu32, 0x143E041Cu32, 0x143E0C34u32, 0x143E242Cu32, 0x1C04040Cu32, 0x1C040C04u32, 0x1C040C14u32, 0x1C04140Cu32,
    0x1C04141Cu32, 0x1C042C04u32, 0x1C04342Cu32, 0x1C043E14u32, 0x1C0C0404u32, 0x1C0C0414u32, 0x1C0C1404u32, 0x1C0C1C0Cu32,
    0x1C0C2424u32, 0x1C0C2434u32, 0x1C14040Cu32, 0x1C14041Cu32, 0x1C140C04u32, 0x1C14142Cu32, 0x1C142C14u32, 0x1C143E14u32,
    0x1C1C0C0Cu32, 0x1C1C1C1Cu32, 0x1C241C04u32, 0x1C24243Eu32, 0x1C243E14u32, 0x1C2C0404u32, 0x1C2C0434u32, 0x1C2C1414u32,
    0x1C2C2C2Cu32, 0x1C340C24u32, 0x1C341C34u32, 0x1C34341Cu32, 0x1C3E1C1Cu32, 0x1C3E3404u32, 0x24040424u32, 0x24040C3Eu32,
    0x24041C2Cu32, 0x24041C3Eu32, 0x24042C1Cu32, 0x24042C3Eu32, 0x240C3E24u32, 0x24141404u32, 0x24141C3Eu32, 0x24142404u32,
    0x24143404u32, 0x24143434u32, 0x241C043Eu32, 0x241C242Cu32, 0x24240424u32, 0x24242C0Cu32, 0x24243424u32, 0x242C142Cu32,
    0x242C241Cu32, 0x242C3E04u32, 0x243E042Cu32, 0x243E0C04u32, 0x243E0C14u32, 0x243E1C04u32, 0x2C040C14u32, 0x2C04240Cu32,
    0x2C043E04u32, 0x2C0C0404u32, 0x2C0C0434u32, 0x2C0C1434u32, 0x2C0C2C2Cu32, 0x2C140C24u32, 0x2C141C14u32, 0x2C143E14u32,
    0x2C1C0414u32, 0x2C1C2C1Cu32, 0x2C240C04u32, 0x2C24141Cu32, 0x2C24143Eu32, 0x2C243E14u32, 0x2C2C0414u32, 0x2C2C1C0Cu32,
    0x2C342C04u32, 0x2C3E1424u32, 0x2C3E2414u32, 0x34041424u32, 0x34042424u32, 0x34042434u32, 0x34043424u32, 0x340C140Cu32,
    0x340C340Cu32, 0x34140C3Eu32, 0x34143424u32, 0x341C1C04u32, 0x341C1C34u32, 0x34242424u32, 0x342C042Cu32, 0x342C2C14u32,
    0x34341C1Cu32, 0x343E041Cu32, 0x343E140Cu32, 0x3E04041Cu32, 0x3E04042Cu32, 0x3E04043Eu32, 0x3E040C04u32, 0x3E041C14u32,
    0x3E042C14u32, 0x3E0C1434u32, 0x3E0C2404u32, 0x3E140C14u32, 0x3E14242Cu32, 0x3E142C14u32, 0x3E1C0404u32, 0x3E1C0C2Cu32,
    0x3E1C1C1Cu32, 0x3E1C3404u32, 0x3E24140Cu32, 0x3E24240Cu32, 0x3E2C0404u32, 0x3E2C0414u32, 0x3E2C1424u32, 0x3E341C04u32
];

const KSIGNS_IQ2XS: [u8; 128] = [
    0u8, 129u8, 130u8, 3u8, 132u8, 5u8, 6u8, 135u8, 136u8, 9u8, 10u8, 139u8, 12u8, 141u8, 142u8, 15u8,
    144u8, 17u8, 18u8, 147u8, 20u8, 149u8, 150u8, 23u8, 24u8, 153u8, 154u8, 27u8, 156u8, 29u8, 30u8, 159u8,
    160u8, 33u8, 34u8, 163u8, 36u8, 165u8, 166u8, 39u8, 40u8, 169u8, 170u8, 43u8, 172u8, 45u8, 46u8, 175u8,
    48u8, 177u8, 178u8, 51u8, 180u8, 53u8, 54u8, 183u8, 184u8, 57u8, 58u8, 187u8, 60u8, 189u8, 190u8, 63u8,
    192u8, 65u8, 66u8, 195u8, 68u8, 197u8, 198u8, 71u8, 72u8, 201u8, 202u8, 75u8, 204u8, 77u8, 78u8, 207u8,
    80u8, 209u8, 210u8, 83u8, 212u8, 85u8, 86u8, 215u8, 216u8, 89u8, 90u8, 219u8, 92u8, 221u8, 222u8, 95u8,
    96u8, 225u8, 226u8, 99u8, 228u8, 101u8, 102u8, 231u8, 232u8, 105u8, 106u8, 235u8, 108u8, 237u8, 238u8, 111u8,
    240u8, 113u8, 114u8, 243u8, 116u8, 245u8, 246u8, 119u8, 120u8, 249u8, 250u8, 123u8, 252u8, 125u8, 126u8, 255u8
];

fn ref_dequant_iq3_xxs(data: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 256;
    const BLK: usize = 98;
    let nblocks = n.div_ceil(QK);
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let base = b * BLK;
        if base + BLK > data.len() {
            break;
        }
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let qs = &data[base + 2..base + 66];
        let aux = &data[base + 66..base + 98];
        let mut y = b * QK;
        let mut qs_off = 0usize;
        for ib32 in 0..8 {
            let aux32 = u32::from_le_bytes([
                aux[4 * ib32],
                aux[4 * ib32 + 1],
                aux[4 * ib32 + 2],
                aux[4 * ib32 + 3],
            ]);
            let db = d * (0.5 + ((aux32 >> 28) as f32)) * 0.5;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let g1 = IQ3XXS_GRID[qs[qs_off + 2 * l] as usize];
                let g2 = IQ3XXS_GRID[qs[qs_off + 2 * l + 1] as usize];
                let b1 = g1.to_le_bytes();
                let b2 = g2.to_le_bytes();
                for j in 0..4 {
                    let idx1 = y + j;
                    let idx2 = y + 4 + j;
                    if idx1 < n {
                        out[idx1] = db
                            * (b1[j] as i8 as f32)
                            * (if signs & (1u8 << j) != 0 { -1.0 } else { 1.0 });
                    }
                    if idx2 < n {
                        out[idx2] = db
                            * (b2[j] as i8 as f32)
                            * (if signs & (1u8 << (4 + j)) != 0 { -1.0 } else { 1.0 });
                    }
                }
                y += 8;
            }
            qs_off += 8;
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
                GgmlType::IQ3S => gpu.gemv_iq3_s(&d_raw, &d_x, &d_y, m, k).unwrap(),
                GgmlType::Q4K => gpu.gemv_q4k(&d_raw, &d_x, &d_y, m, k).unwrap(),
                GgmlType::IQ3XXS => gpu.gemv_iq3_xxs(&d_raw, &d_x, &d_y, m, k).unwrap(),
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
                GgmlType::IQ3S => {
                    gpu.gemm_iq3_s_batched(&d_raw, &d_xb, &d_yb, m, k, batch).unwrap()
                }
                GgmlType::Q4K => {
                    gpu.gemm_q4k_batched(&d_raw, &d_xb, &d_yb, m, k, batch).unwrap()
                }
                GgmlType::IQ3XXS => {
                    gpu.gemm_iq3_xxs_batched(&d_raw, &d_xb, &d_yb, m, k, batch).unwrap()
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
                    GgmlType::IQ3S => {
                        gpu.gemm_iq3_s_batched(&d_raw, &view, &d_yv, m, k, batch).unwrap()
                    }
                    GgmlType::Q4K => {
                        gpu.gemm_q4k_batched(&d_raw, &view, &d_yv, m, k, batch).unwrap()
                    }
                    GgmlType::IQ3XXS => {
                        gpu.gemm_iq3_xxs_batched(&d_raw, &view, &d_yv, m, k, batch).unwrap()
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
    run_case!("IQ3_S", GgmlType::IQ3S, |d, n| ref_dequant_iq3_s(d, n));
    run_case!("Q4_K", GgmlType::Q4K, |d, n| ref_dequant_q4_k(d, n));
    run_case!("IQ3_XXS", GgmlType::IQ3XXS, |d, n| ref_dequant_iq3_xxs(d, n));



    eprintln!("\n=== RESULT: {passed} passed, {failed} failed ===");
    if failed > 0 {
        std::process::exit(1);
    }
}