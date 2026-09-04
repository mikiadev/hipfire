// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Standalone GPU validation of the Escha DENSE decode-gemm against the host
//! reference for one real projection of /data/rocmfpx/Qwen3.8-27B-Escha-W2.
//!
//! Usage: cargo run --release -p hipfire-arch-qwen35 --example check_escha_dense -- <layer> <proj>
//!   layer: 0..63, proj: in_proj_qkv|in_proj_z|out_proj|q_proj|k_proj|v_proj|o_proj|
//!                  gate_proj|up_proj|down_proj
//! (layer type must match the projection family: linear_attn for the first set,
//! self_attn for q/k/v/o, mlp always.)

use std::fs;

fn read_tensor(
    wm: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> (Vec<u8>, String) {
    let shard = wm[key].as_str().unwrap().to_string();
    let bytes = fs::read(format!("/data/rocmfpx/Qwen3.8-27B-Escha-W2/{shard}")).unwrap();
    if bytes.len() < 1024 {
        eprintln!("DEBUG read_tensor({key}) shard={shard} got {} bytes!", bytes.len());
    }
    (bytes, shard)
}

fn parse_hdr(bytes: &[u8]) -> (serde_json::Value, usize) {
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    // safetensors: header padded so (8 + n + pad) % 8 == 0
    let data_start = 8 + n + ((8 - (8 + n) % 8) % 8);
    (hdr, data_start)
}

fn load_u8(wm: &serde_json::Map<String, serde_json::Value>, key: &str) -> Vec<u8> {
    let (bytes, _shard) = read_tensor(wm, key);
    eprintln!("DEBUG load_u8 {key}: file bytes {}", bytes.len());
    let (hdr, data_start) = parse_hdr(&bytes);
    eprintln!("DEBUG load_u8 {key}: data_start {data_start}");
    let meta = &hdr[key];
    let (b, e) = (
        meta["data_offsets"][0].as_u64().unwrap() as usize,
        meta["data_offsets"][1].as_u64().unwrap() as usize,
    );
    bytes[data_start + b..data_start + e].to_vec()
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        if m == 0 {
            if s == 1 { -0.0 } else { 0.0 }
        } else {
            // subnormal
            let mut ee = 1u32;
            let mut mm = m;
            while mm & 0x400 == 0 {
                mm <<= 1;
                ee += 1;
            }
            mm &= 0x3ff;
            f32::from_bits((s << 31) | ((127 - ee + 15) << 23) | (mm << 13))
        }
    } else if e == 0x1f {
        if m == 0 {
            f32::from_bits((s << 31) | 0x7f80_0000)
        } else {
            f32::from_bits((s << 31) | 0x7fc0_0000)
        }
    } else {
        f32::from_bits((s << 31) | ((e + 127 - 15) << 23) | (m << 13))
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let layer: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let proj = args.get(2).cloned().unwrap_or_else(|| "in_proj_qkv".to_string());

    let idx: serde_json::Value = serde_json::from_str(
        &fs::read_to_string("/data/rocmfpx/Qwen3.8-27B-Escha-W2/model.safetensors.index.json").unwrap(),
    )
    .unwrap();
    let wm = idx["weight_map"].as_object().unwrap().clone();

    let (family, expected_k) = if proj == "up_proj" || proj == "down_proj" {
        ("mlp", 3usize)
    } else if proj == "gate_proj" {
        ("mlp", 2usize)
    } else if proj.starts_with("in_proj") || proj == "out_proj" {
        ("linear_attn", 2usize)
    } else {
        ("self_attn", 2usize)
    };
    let base = format!("model.language_model.layers.{layer}.{family}.{proj}");

    // decode code -> i16, and read shapes from code header
    let code_key = format!("{base}.escha_code");
    let (code_bytes, _) = read_tensor(&wm, &code_key);
    let (chdr, cdata_start) = parse_hdr(&code_bytes);
    let cm = &chdr[&code_key];
    let shape: Vec<usize> = cm["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let (t_in, t_out, _last) = (shape[0], shape[1], shape[2]);
    let in_p = t_in * 16;
    let out_p = t_out * 16;
    let k = shape[2] / 16;
    assert_eq!(k, expected_k, "K mismatch for {proj}");
    let (b, e) = (
        cm["data_offsets"][0].as_u64().unwrap() as usize,
        cm["data_offsets"][1].as_u64().unwrap() as usize,
    );
    let raw = &code_bytes[cdata_start + b..cdata_start + e];
    let code: Vec<i16> = raw
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    // scales (load_u8 returns the already-sliced raw tensor bytes)
    let dtype_of = |key: &str| -> String {
        let shard = wm[key].as_str().unwrap().to_string();
        let bytes = fs::read(format!("/data/rocmfpx/Qwen3.8-27B-Escha-W2/{shard}")).unwrap();
        let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
        hdr[key]["dtype"].as_str().unwrap().to_string()
    };
    let load_f32 = |suffix: &str| -> Vec<f32> {
        let key = format!("{base}.escha_{suffix}");
        let raw = load_u8(&wm, &key);
        if dtype_of(&key) == "F16" {
            raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
        } else {
            raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
    };
    let rin = load_f32("rin");
    let rout = load_f32("rout");
    let sin = load_f32("s_in");
    let sout = load_f32("s_out");
    eprintln!(
        "projection {base}: code {shape:?} ({in_p}x{out_p}, K={k}) rin {} rout {} sin {} sout {}",
        rin.len(),
        rout.len(),
        sin.len(),
        sout.len()
    );
    let in_scale: Vec<f32> = rin.iter().zip(sin.iter()).map(|(&r, &s)| r * s).collect();
    let out_scale: Vec<f32> = rout.iter().zip(sout.iter()).map(|(&r, &s)| r * s).collect();

    // pseudo-random x
    let mut seed = (layer as u32).wrapping_mul(2654435761) ^ 0x9e3779b9 ^ 0xabcdef01;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 0.2 - 0.1
    };
    let x: Vec<f32> = (0..in_p).map(|_| next()).collect();

    let Some(mut gpu) = rdna_compute::Gpu::init().ok() else {
        eprintln!("no gpu"); return;
    };

    // upload projection
    let code_gpu = gpu.upload_raw(
        unsafe { std::slice::from_raw_parts(code.as_ptr() as *const u8, code.len() * 2) },
        &[in_p / 16, out_p / 16, k * 16],
    ).expect("code upload");
    let in_scale_gpu = gpu.upload_f32(&in_scale, &[in_p]).expect("in upload");
    let out_scale_gpu = gpu.upload_f32(&out_scale, &[out_p]).expect("out upload");
    let proj_w = hipfire_arch_qwen35::qwen35::EschaDenseProjWeights {
        code: rdna_compute::GpuTensor { buf: code_gpu.buf, shape: code_gpu.shape.clone(), dtype: code_gpu.dtype },
        in_scale: in_scale_gpu,
        out_scale: out_scale_gpu,
        in_p,
        out_p,
        k: k as u8,
    };

    let proj_name = proj; // &str label
    let _ = hipfire_arch_qwen35::qwen35::escha_dense_decode::escha_dense_check_rotate(
        &mut gpu, &proj_w, &x, &in_scale,
    )
    .expect("rotate");
    let (fa, yf, ya) =
        hipfire_arch_qwen35::qwen35::escha_dense_decode::escha_dense_check_fold_vs_act(
            &code, k, in_p, out_p, &in_scale, &out_scale, &x,
        );
    eprintln!("fold-vs-act max_abs {fa} yf[0..3]={:?} ya[0..3]={:?}", &yf[..3], &ya[..3]);
    let _ = hipfire_arch_qwen35::qwen35::escha_dense_decode::escha_dense_check_decode_stage(
        &mut gpu, &proj_w, &x, &code, &in_scale,
    )
    .expect("decode-stage");
    let (max_abs, rel) = hipfire_arch_qwen35::qwen35::escha_dense_decode::escha_dense_check_proj(
        &mut gpu, &proj_w, &x, &code, &in_scale, &out_scale, &base,
    )
    .expect("check");
    eprintln!(
        "RESULT: {proj_name} L{layer}: max_abs={max_abs:.6} rel={rel:.6}  ({})",
        if rel < 1e-3 { "PASS" } else { "FAIL" }
    );
}
