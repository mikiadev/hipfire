// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Profile the Escha-DENSE batched prefill path vs the per-token decode path
//! across batch sizes.
//!
//! Usage:
//!   cargo run --release -p hipfire-arch-qwen35 --example profile_escha_dense_prefill -- \
//!     <layer> <proj> [max_batch] [warmup] [reps]
//!   layer: 0..63
//!   proj:  in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj
//!   max_batch: largest n_rows to sweep (default 512)
//!   warmup:     warmup reps per cell (default 3)
//!   reps:       measurement reps per cell (default 10)

use std::fs;
use std::time::Instant;

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        if m == 0 {
            if s == 1 { -0.0 } else { 0.0 }
        } else {
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

fn read_tensor(
    wm: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> (Vec<u8>, String) {
    let shard = wm[key].as_str().unwrap().to_string();
    let bytes = fs::read(format!("/data/rocmfpx/Qwen3.8-27B-Escha-W2/{shard}")).unwrap();
    (bytes, shard)
}

fn parse_hdr(bytes: &[u8]) -> (serde_json::Value, usize) {
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data_start = 8 + n + ((8 - (8 + n) % 8) % 8);
    (hdr, data_start)
}

fn load_u8(wm: &serde_json::Map<String, serde_json::Value>, key: &str) -> Vec<u8> {
    let (bytes, _) = read_tensor(wm, key);
    let (hdr, data_start) = parse_hdr(&bytes);
    let meta = &hdr[key];
    let (b, e) = (
        meta["data_offsets"][0].as_u64().unwrap() as usize,
        meta["data_offsets"][1].as_u64().unwrap() as usize,
    );
    bytes[data_start + b..data_start + e].to_vec()
}

fn dtype_of(wm: &serde_json::Map<String, serde_json::Value>, key: &str) -> String {
    let (bytes, _) = read_tensor(wm, key);
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    hdr[key]["dtype"].as_str().unwrap().to_string()
}

fn load_f32(wm: &serde_json::Map<String, serde_json::Value>, base: &str, suffix: &str) -> Vec<f32> {
    let key = format!("{base}.escha_{suffix}");
    let raw = load_u8(wm, &key);
    if dtype_of(wm, &key) == "F16" {
        raw.chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    } else {
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let layer: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let proj = args.get(2).cloned().unwrap_or_else(|| "in_proj_qkv".to_string());
    let max_batch: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(512);
    let warmup: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
    let reps: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(10);

    let idx: serde_json::Value = serde_json::from_str(
        &fs::read_to_string("/data/rocmfpx/Qwen3.8-27B-Escha-W2/model.safetensors.index.json")
            .unwrap(),
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

    // Load code + scales
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

    let rin = load_f32(&wm, &base, "rin");
    let rout = load_f32(&wm, &base, "rout");
    let sin = load_f32(&wm, &base, "s_in");
    let sout = load_f32(&wm, &base, "s_out");
    let in_scale: Vec<f32> = rin.iter().zip(sin.iter()).map(|(&r, &s)| r * s).collect();
    let out_scale: Vec<f32> = rout.iter().zip(sout.iter()).map(|(&r, &s)| r * s).collect();

    eprintln!(
        "projection {base}: {in_p}x{out_p} K={k}  (code {} i16s)",
        code.len()
    );

    let Some(mut gpu) = rdna_compute::Gpu::init().ok() else {
        eprintln!("no gpu");
        return;
    };

    // Upload projection
    let code_gpu = gpu
        .upload_raw(
            unsafe { std::slice::from_raw_parts(code.as_ptr() as *const u8, code.len() * 2) },
            &[in_p / 16, out_p / 16, k * 16],
        )
        .expect("code upload");
    let in_scale_gpu = gpu.upload_f32(&in_scale, &[in_p]).expect("in upload");
    let out_scale_gpu = gpu.upload_f32(&out_scale, &[out_p]).expect("out upload");
    let proj_w = hipfire_arch_qwen35::qwen35::EschaDenseProjWeights {
        code: rdna_compute::GpuTensor {
            buf: code_gpu.buf,
            shape: code_gpu.shape.clone(),
            dtype: code_gpu.dtype,
        },
        in_scale: in_scale_gpu,
        out_scale: out_scale_gpu,
        in_p,
        out_p,
        k: k as u8,
    };

    // Pseudo-random input activations.
    let mut seed = (layer as u32).wrapping_mul(2654435761) ^ 0x9e3779b9 ^ 0xabcdef01;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 0.2 - 0.1
    };
    let mut rng_x = || (0..in_p).map(|_| next()).collect::<Vec<f32>>();

    let mut batches = vec![1usize];
    let mut b = 4;
    while b <= max_batch {
        batches.push(b);
        b *= 2;
    }

    eprintln!(
        "\n===== PER-TOKEN DECODE (n_rows=1, per-projection, with pool allocs) ====="
    );
    // Hoisted decode scratch (matches production path: no per-call pool allocs).
    let nit = in_p / 16;
    let n_slices = rdna_compute::escha_dense::escha_dense_n_slices(nit, out_p);
    let u = gpu.alloc_tensor(&[in_p], rdna_compute::DType::F32).expect("u alloc");
    let partial = gpu.alloc_tensor(&[n_slices * out_p], rdna_compute::DType::F32).expect("p alloc");
    let mut per_token_samples = Vec::new();
    for i in 0..(warmup + reps) {
        let x_host = rng_x();
        let x = gpu.upload_f32(&x_host, &[in_p]).expect("x upload");
        let y = gpu.alloc_tensor(&[out_p], rdna_compute::DType::F32).expect("y alloc");
        let t0 = Instant::now();
        hipfire_arch_qwen35::qwen35::escha_dense_decode::escha_dense_decode_proj(
            &mut gpu, &proj_w, &x, &y, &u, &partial,
        )
        .expect("per-token decode");
        let _ = gpu.download_f32(&y).expect("download sync");
        let dt = t0.elapsed();
        if i >= warmup {
            per_token_samples.push(dt);
        }
        let _ = gpu.free_tensor(y);
        let _ = gpu.free_tensor(x);
    }
    let _ = gpu.free_tensor(u);
    let _ = gpu.free_tensor(partial);
    per_token_samples.sort();
    let med = per_token_samples[per_token_samples.len() / 2];
    let min_t = *per_token_samples.iter().min().unwrap();
    let max_t = *per_token_samples.iter().max().unwrap();
    let avg: f64 = per_token_samples.iter().map(|d| d.as_secs_f64()).sum::<f64>() / reps as f64;
    eprintln!(
        "per-token decode (with alloc): reps={reps}  min={min_t:?}  med={med:?}  avg={:.3} ms  max={max_t:?}",
        avg * 1000.0
    );

    // Kernel-only time: pre-allocate u/partial, time the full decode-gemm
    // (rotate + gemv + finalize) without the per-call pool alloc/free.
    let nit = in_p / 16;
    let n_slices = rdna_compute::escha_dense::escha_dense_n_slices(nit, out_p);
    let u = gpu.alloc_tensor(&[in_p], rdna_compute::DType::F32).expect("u alloc");
    let partial = gpu.alloc_tensor(&[n_slices * out_p], rdna_compute::DType::F32).expect("p alloc");
    let mut kernel_only_samples = Vec::new();
    for i in 0..(warmup + reps) {
        let x_host = rng_x();
        let x = gpu.upload_f32(&x_host, &[in_p]).expect("x upload");
        let y = gpu.alloc_tensor(&[out_p], rdna_compute::DType::F32).expect("y alloc");
        let t0 = Instant::now();
        rdna_compute::escha_dense::escha_dense_decode_gemv(
            &mut gpu, &proj_w.code, &proj_w.in_scale, &proj_w.out_scale,
            &x, &u, &partial, &y,
        )
        .expect("decode gemv");
        let _ = gpu.download_f32(&y).expect("download sync");
        let dt = t0.elapsed();
        if i >= warmup {
            kernel_only_samples.push(dt);
        }
        let _ = gpu.free_tensor(y);
        let _ = gpu.free_tensor(x);
    }
    kernel_only_samples.sort();
    let med_k = kernel_only_samples[kernel_only_samples.len() / 2];
    let min_k = *kernel_only_samples.iter().min().unwrap();
    let max_k = *kernel_only_samples.iter().max().unwrap();
    let avg_k: f64 =
        kernel_only_samples.iter().map(|d| d.as_secs_f64()).sum::<f64>() / reps as f64;
    eprintln!(
        "per-token decode (kernel only): reps={reps}  min={min_k:?}  med={med_k:?}  avg={:.3} ms  max={max_k:?}",
        avg_k * 1000.0
    );
    let _ = gpu.free_tensor(u);
    let _ = gpu.free_tensor(partial);

    eprintln!("\n===== BATCHED PREFILL (varying n_rows) =====");
    eprintln!(
        "{:>8} | {:>10} | {:>10} | {:>10} | {:>10} | {:>8} | {:>6}",
        "n_rows", "total(ms)", "per_row(us)", "speedup/tok", "n_slices", "r", "ok"
    );
    for &n_rows in &batches {
        let r = if n_rows <= 1 { 1 } else { n_rows.next_power_of_two().min(64) };
        let n_slices = rdna_compute::escha_dense::escha_dense_n_slices_prefill(nit, out_p, n_rows, r as i32);
        let partial_len = n_slices * n_rows * out_p;
        // Build a batch of random inputs.
        let x_batch_host: Vec<f32> = (0..n_rows).flat_map(|_| rng_x()).collect();
        let x_batch = gpu
            .upload_f32(&x_batch_host, &[n_rows * in_p])
            .expect("batch upload");
        let y_batch = gpu
            .alloc_tensor(&[n_rows * out_p], rdna_compute::DType::F32)
            .expect("y batch alloc");

        // Warmup
        for _ in 0..warmup {
            hipfire_arch_qwen35::qwen35::escha_dense_forward::escha_dense_decode_proj_batch(
                &mut gpu, &proj_w, &x_batch, &y_batch, n_rows,
            )
            .expect("batch prefill");
            gpu.hip.device_synchronize().expect("sync");
        }

        // Measure
        let mut samples = Vec::new();
        for _ in 0..reps {
            let t0 = Instant::now();
            hipfire_arch_qwen35::qwen35::escha_dense_forward::escha_dense_decode_proj_batch(
                &mut gpu, &proj_w, &x_batch, &y_batch, n_rows,
            )
            .expect("batch prefill");
            gpu.hip.device_synchronize().expect("sync");
            samples.push(t0.elapsed());
        }
        samples.sort();
        let med = samples[samples.len() / 2];
        let min_t = *samples.iter().min().unwrap();
        let max_t = *samples.iter().max().unwrap();
        let avg_ms: f64 =
            samples.iter().map(|d| d.as_secs_f64()).sum::<f64>() / reps as f64 * 1000.0;
        let per_row_us = avg_ms * 1000.0 / n_rows as f64;
        let per_tok_us =
            (med.as_secs_f64() * 1e6) / n_rows as f64;
        let baseline_us =
            (med_k.as_secs_f64() * 1e6);
        let speedup = baseline_us / per_tok_us;

        // Verify correctness: download and compare first row against per-token reference.
        let y_host = gpu.download_f32(&y_batch).expect("download");
        let mut x_first_row = x_batch_host[..in_p].to_vec();
        // Apply in_scale
        for v in &mut x_first_row { *v *= 1.0; } // already scaled in batch path
        let want = hipfire_arch_qwen35::qwen35::escha_dense_decode::escha_dense_decode_proj_host(
            &code, k, in_p, out_p, &in_scale, &out_scale, &x_batch_host[..in_p],
        );
        let got_first_row = &y_host[..out_p];
        let mut max_err = 0.0f32;
        for i in 0..out_p {
            max_err = max_err.max((got_first_row[i] - want[i]).abs());
        }
        let ok = if max_err < 1e-2 { "PASS" } else { "FAIL" };

        eprintln!(
            "{:>8} | {:>10.3} | {:>10.1} | {:>10.2}x | {:>10} | {:>8} | {:>6}",
            n_rows, avg_ms, per_row_us, speedup, n_slices, r, ok
        );
        eprintln!(
            "           (min={min_t:?} med={med:?} max={max_t:?} partial_len={partial_len} max_err={max_err:.4})"
        );

        let _ = gpu.free_tensor(y_batch);
        let _ = gpu.free_tensor(x_batch);
    }
}
