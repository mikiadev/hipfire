// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Measure the decoded weight row norms for one dense projection to check the
//! decode output scale in isolation.
//! Usage: cargo run --release -p hipfire-arch-qwen35 --example check_wrow_norm -- <layer> <proj>

use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let layer: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let proj = args.get(2).cloned().unwrap_or_else(|| "out_proj".to_string());
    let idx: serde_json::Value = serde_json::from_str(
        &fs::read_to_string("/data/rocmfpx/Qwen3.8-27B-Escha-W2/model.safetensors.index.json").unwrap(),
    )
    .unwrap();
    let wm = idx["weight_map"].as_object().unwrap();
    let base = format!("model.language_model.layers.{layer}");
    let fam = if proj == "q_proj" || proj == "k_proj" || proj == "v_proj" || proj == "o_proj" {
        "self_attn"
    } else if proj == "gate_proj" || proj == "up_proj" || proj == "down_proj" {
        "mlp"
    } else {
        "linear_attn"
    };
    let key = format!("{base}.{fam}.{proj}.escha_code");
    let shard = wm[key.as_str()].as_str().unwrap();
    let bytes = fs::read(format!("/data/rocmfpx/Qwen3.8-27B-Escha-W2/{shard}")).unwrap();
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data_start = 8 + n + ((8 - (8 + n) % 8) % 8);
    let m = &hdr[&key];
    let shape: Vec<usize> = m["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let (b, e) = (
        m["data_offsets"][0].as_u64().unwrap() as usize,
        m["data_offsets"][1].as_u64().unwrap() as usize,
    );
    let raw = &bytes[data_start + b..data_start + e];
    let code: Vec<i16> = raw.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
    let (ti, tj, _) = (shape[0], shape[1], shape[2]);
    let (in_p, out_p, k) = (ti * 16, tj * 16, shape[2] / 16);

    // in_scale/out_scale (loaded the same way as the loader)
    let load_f32 = |suffix: &str| -> Vec<f32> {
        let kk = format!("{base}.{fam}.{proj}.escha_{suffix}");
        let shard = wm[kk.as_str()].as_str().unwrap().to_string();
        let bytes = fs::read(format!("/data/rocmfpx/Qwen3.8-27B-Escha-W2/{shard}")).unwrap();
        let nn = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let hh: serde_json::Value = serde_json::from_slice(&bytes[8..8 + nn]).unwrap();
        let ds = 8 + nn + ((8 - (8 + nn) % 8) % 8);
        let mm = &hh[&kk];
        let dt = mm["dtype"].as_str().unwrap().to_string();
        let (bb, ee) = (mm["data_offsets"][0].as_u64().unwrap() as usize, mm["data_offsets"][1].as_u64().unwrap() as usize);
        let d = &bytes[ds + bb..ds + ee];
        if dt == "F16" {
            d.chunks_exact(2).map(|c| { let h = u16::from_le_bytes([c[0], c[1]]); let s = ((h >> 15) & 1) as f32; let e2 = ((h >> 10) & 0x1f) as i32; let m2 = (h & 0x3ff) as f32; if e2 == 0 { if m2 == 0.0 { if s == 1.0 { -0.0 } else { 0.0 } } else { let mut ee3 = 1i32; let mut mm3 = (h & 0x3ff) as i32; while mm3 & 0x400 == 0 { mm3 <<= 1; ee3 += 1; } mm3 &= 0x3ff; f32::from_bits(((s as u32) << 31) | (((127 - ee3 + 15) as u32) << 23) | ((mm3 as u32) << 13)) } } else if e2 == 0x1f { if m2 == 0.0 { f32::from_bits(((s as u32) << 31) | 0x7f80_0000) } else { f32::from_bits(((s as u32) << 31) | 0x7fc0_0000) } } else { f32::from_bits(((s as u32) << 31) | (((e2 + 127 - 15) as u32) << 23) | ((m2 as u32) << 13)) } }).collect()
        } else {
            d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
    };
    let rin = load_f32("rin");
    let rout = load_f32("rout");
    let sin = load_f32("s_in");
    let sout = load_f32("s_out");
    let in_scale: Vec<f32> = rin.iter().zip(sin.iter()).map(|(&r, &s)| r * s).collect();
    let out_scale: Vec<f32> = rout.iter().zip(sout.iter()).map(|(&r, &s)| r * s).collect();

    let w_bare = hipfire_arch_qwen35::escham_decode::decode_tiles(&code, k, in_p, out_p);
    // full weight W[out,in] = had128 both axes of w_bare (scales outside WHT):
    // W[c][i] = out_scale[c] * had_out_col(had_in_row(w_bare[i][c]))  -- compute via fold
    let w_full = hipfire_arch_qwen35::escham_decode::decode_and_fold_weight(
        &code, k, in_p, out_p, &in_scale, &out_scale,
    );
    let mut row_rms = vec![0.0f64; out_p];
    for c in 0..out_p {
        let mut s = 0.0f64;
        for i in 0..in_p {
            s += (w_full[c * in_p + i] as f64).powi(2);
        }
        row_rms[c] = (s / in_p as f64).sqrt();
    }
    let mean = row_rms.iter().sum::<f64>() / out_p as f64;
    let maxr = row_rms.iter().cloned().fold(0.0f64, f64::max);
    eprintln!(
        "proj {proj} L{layer} {in_p}x{out_p} K{k}: full-weight row-rms mean={mean:.4} max={maxr:.4}"
    );
    // expected output rms for unit-rms random input ≈ row_rms
    eprintln!(
        "=> wo/attn output rms for a rms=1 input ≈ {mean:.4} (range ~{}x)",
        (3.0 * mean).round()
    );
    let _ = w_bare;
}
