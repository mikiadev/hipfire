// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! FA-layer batched oracle for the Escha DENSE path.
//!
//! `check_escha_dense_batched` validated the batched coded PROJECTIONS vs the
//! host reference; `check_gdn_batched` validated the DeltaNet carry. What is
//! NOT covered is the new `fullattn_escha_layer_prefill` body: norm → coded
//! q/k/v → deinterleave → q/k norms → RoPE → KV write + flash attention →
//! gate → coded wo + residual → FFN.
//!
//! Ground truth is the per-token `fullattn_escha_layer_forward` applied N
//! times — i.e. what decode does. Synthetic inputs (no model load), one FA
//! layer's real coded weights from the checkpoint + on-disk norms, so this
//! runs in seconds and a few hundred MB.
//!
//! Usage:
//!   cargo run --release -p hipfire-arch-qwen35 --example check_escha_fa_batched -- [layer] [rows...]
//!     layer: an FA layer index (default 3; must satisfy layer % 4 == 3)
//!     rows:  chunk rows to test (default 2 4 8 16 32 58)
//!
//! Exit nonzero on FAIL.

use std::fs;

use hipfire_arch_qwen35::qwen35::escha_dense_forward::fullattn_escha_layer_forward;
use hipfire_arch_qwen35::qwen35::BatchSemantics;
use hipfire_arch_qwen35::qwen35::EschaDenseProjWeights;
use hipfire_arch_qwen35::qwen35::FullAttnEschaLayerWeights;
use hipfire_arch_qwen35::qwen35::PrefillBatchScratch;
use hipfire_arch_qwen35::qwen35::Qwen35Scratch;

const MODEL_DIR: &str = "/data/rocmfpx/Qwen3.8-27B-Escha-W2";

// Model shape (verified from config.json; see ESCHA-W2-CONTINUE.md).
const DIM: usize = 5120;
const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const HIDDEN_DIM: usize = 17408;
const NORM_EPS: f32 = 1e-6;

fn read_tensor(
    wm: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> (Vec<u8>, String) {
    let shard = wm[key].as_str().unwrap().to_string();
    (
        fs::read(format!("{MODEL_DIR}/{shard}")).unwrap(),
        shard,
    )
}

fn parse_hdr(bytes: &[u8]) -> (serde_json::Value, usize) {
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data_start = 8 + n + ((8 - (8 + n) % 8) % 8);
    (hdr, data_start)
}

fn load_raw(wm: &serde_json::Map<String, serde_json::Value>, key: &str) -> (Vec<u8>, String) {
    let (bytes, _shard) = read_tensor(wm, key);
    let (hdr, data_start) = parse_hdr(&bytes);
    let meta = &hdr[key];
    let (b, e) = (
        meta["data_offsets"][0].as_u64().unwrap() as usize,
        meta["data_offsets"][1].as_u64().unwrap() as usize,
    );
    let dtype = meta["dtype"].as_str().unwrap().to_string();
    (bytes[data_start + b..data_start + e].to_vec(), dtype)
}

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

fn f16_slice_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

struct RawProj {
    code: Vec<i16>,
    in_scale: Vec<f32>,
    out_scale: Vec<f32>,
    in_p: usize,
    out_p: usize,
    k: usize,
}

/// Load one coded projection's raw tensors: code int16 + folded scales
/// (in_scale = rin·s_in, out_scale = rout·s_out — the same fold the loader
/// applies in `escha_load_dense_proj`; Wscale is already inside rin).
fn load_proj(
    wm: &serde_json::Map<String, serde_json::Value>,
    base: &str,
) -> Option<RawProj> {
    let code_key = format!("{base}.escha_code");
    if wm.get(&code_key).is_none() {
        return None;
    }
    // Code shape from the header: [in/16, out/16, 16*K].
    let (bytes, _) = read_tensor(wm, &code_key);
    let (hdr, data_start) = parse_hdr(&bytes);
    let cm = &hdr[&code_key];
    let shape: Vec<usize> = cm["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let (in_p, out_p, k) = (shape[0] * 16, shape[1] * 16, shape[2] / 16);
    let (b, e) = (
        cm["data_offsets"][0].as_u64().unwrap() as usize,
        cm["data_offsets"][1].as_u64().unwrap() as usize,
    );
    let code: Vec<i16> = bytes[data_start + b..data_start + e]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let load_f32 = |suffix: &str| -> Vec<f32> {
        let key = format!("{base}.escha_{suffix}");
        let (raw, dtype) = load_raw(wm, &key);
        if dtype == "F16" {
            raw.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        } else {
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
    };
    let rin = load_f32("rin");
    let rout = load_f32("rout");
    let sin = load_f32("s_in");
    let sout = load_f32("s_out");
    let in_scale: Vec<f32> = rin.iter().zip(sin.iter()).map(|(&r, &s)| r * s).collect();
    let out_scale: Vec<f32> = rout.iter().zip(sout.iter()).map(|(&r, &s)| r * s).collect();
    assert_eq!(in_scale.len(), in_p, "{base} in_scale len");
    assert_eq!(out_scale.len(), out_p, "{base} out_scale len");
    Some(RawProj { code, in_scale, out_scale, in_p, out_p, k })
}

fn upload_proj(
    gpu: &mut rdna_compute::Gpu,
    p: &RawProj,
) -> EschaDenseProjWeights {
    let code_gpu = gpu
        .upload_raw(
            unsafe { std::slice::from_raw_parts(p.code.as_ptr() as *const u8, p.code.len() * 2) },
            &[p.in_p / 16, p.out_p / 16, p.k * 16],
        )
        .expect("code upload");
    let in_scale_gpu = gpu.upload_f32(&p.in_scale, &[p.in_p]).expect("in_scale");
    let out_scale_gpu = gpu.upload_f32(&p.out_scale, &[p.out_p]).expect("out_scale");
    EschaDenseProjWeights {
        code: rdna_compute::GpuTensor {
            buf: code_gpu.buf,
            shape: code_gpu.shape.clone(),
            dtype: code_gpu.dtype,
        },
        in_scale: in_scale_gpu,
        out_scale: out_scale_gpu,
        in_p: p.in_p,
        out_p: p.out_p,
        k: p.k as u8,
    }
}

/// Non-owning view of a full projection (shares device buffers; never freed).
fn view_proj(p: &EschaDenseProjWeights) -> EschaDenseProjWeights {
    EschaDenseProjWeights {
        code: rdna_compute::GpuTensor {
            buf: unsafe { p.code.buf.alias() },
            shape: p.code.shape.clone(),
            dtype: p.code.dtype,
        },
        in_scale: rdna_compute::GpuTensor {
            buf: unsafe { p.in_scale.buf.alias() },
            shape: p.in_scale.shape.clone(),
            dtype: p.in_scale.dtype,
        },
        out_scale: rdna_compute::GpuTensor {
            buf: unsafe { p.out_scale.buf.alias() },
            shape: p.out_scale.shape.clone(),
            dtype: p.out_scale.dtype,
        },
        in_p: p.in_p,
        out_p: p.out_p,
        k: p.k,
    }
}

/// Deterministic pseudo-random block, one seed per row.
fn rand_block(seed0: u32, n: usize) -> Vec<f32> {
    let mut seed = seed0 ^ 0x9e3779b9;
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 0.2 - 0.1
        })
        .collect()
}

fn compare(got: &[f32], want: &[f32]) -> (f32, f32) {
    let mut max_abs = 0f32;
    let mut denom = 1e-12f32;
    for (g, w) in got.iter().zip(want.iter()) {
        if !g.is_finite() {
            return (f32::INFINITY, w.abs().max(denom));
        }
        let d = (g - w).abs();
        if d > max_abs {
            max_abs = d;
        }
        if w.abs() > denom {
            denom = w.abs();
        }
    }
    (max_abs, denom)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let layer: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    assert!(layer % 4 == 3, "layer {layer} is not an FA layer (need layer % 4 == 3)");
    let row_sets: Vec<usize> = if args.len() > 2 {
        args[2..].iter().filter_map(|s| s.parse().ok()).collect()
    } else {
        vec![2, 4, 8, 16, 32, 58]
    };

    let idx: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(format!("{MODEL_DIR}/model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    let wm = idx["weight_map"].as_object().unwrap().clone();

    let Some(mut gpu) = rdna_compute::Gpu::init().ok() else {
        eprintln!("no gpu: this oracle compares GPU output against the per-token path");
        return;
    };
    println!("FA batched oracle on {} — layer {layer}, ground truth = N per-token calls\n", gpu.arch.as_str());

    // Load one FA layer's coded projections + norms.
    let base = format!("model.language_model.layers.{layer}");
    let proj_names = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ];
    let mut owned: Vec<EschaDenseProjWeights> = Vec::new();
    for name in proj_names {
        match load_proj(&wm, &format!("{base}.{name}")) {
            Some(p) => owned.push(upload_proj(&mut gpu, &p)),
            None => {
                eprintln!("FAIL: missing escha_code for {base}.{name}");
                std::process::exit(1);
            }
        }
    }
    let mut norm_plus1 = |key: &str, n: usize| -> rdna_compute::GpuTensor {
        let (b, _) = load_raw(&wm, key);
        // FA q/k + input norms are gamma-1 offsets on disk (B7: bias +1).
        let v: Vec<f32> = f16_slice_to_f32(&b).iter().map(|x| x + 1.0).collect();
        assert_eq!(v.len(), n, "{key} len");
        gpu.upload_f32(&v, &[n]).unwrap_or_else(|_| panic!("norm upload {key}"))
    };
    let attn_norm = norm_plus1(&format!("{base}.input_layernorm.weight"), DIM);
    let ffn_norm = norm_plus1(&format!("{base}.post_attention_layernorm.weight"), DIM);
    let q_norm = norm_plus1(&format!("{base}.self_attn.q_norm.weight"), HEAD_DIM);
    let k_norm = norm_plus1(&format!("{base}.self_attn.k_norm.weight"), HEAD_DIM);

    // Layer struct holds non-owning views; `owned` + norms stay alive below
    // and are freed once at the end (views are never freed).
    let layer_w = FullAttnEschaLayerWeights {
        attn_norm: rdna_compute::GpuTensor {
            buf: unsafe { attn_norm.buf.alias() },
            shape: attn_norm.shape.clone(),
            dtype: attn_norm.dtype,
        },
        wq: view_proj(&owned[0]),
        wk: view_proj(&owned[1]),
        wv: view_proj(&owned[2]),
        wo: view_proj(&owned[3]),
        q_norm: rdna_compute::GpuTensor {
            buf: unsafe { q_norm.buf.alias() },
            shape: q_norm.shape.clone(),
            dtype: q_norm.dtype,
        },
        k_norm: rdna_compute::GpuTensor {
            buf: unsafe { k_norm.buf.alias() },
            shape: k_norm.shape.clone(),
            dtype: k_norm.dtype,
        },
        ffn_norm: rdna_compute::GpuTensor {
            buf: unsafe { ffn_norm.buf.alias() },
            shape: ffn_norm.shape.clone(),
            dtype: ffn_norm.dtype,
        },
        w_gate: view_proj(&owned[4]),
        w_up: view_proj(&owned[5]),
        w_down: view_proj(&owned[6]),
    };

    // Minimal config: read the real model config so every field (layer types,
    // norms, RoPE, DeltaNet geometry) matches the served model exactly.
    let loader_src = hipfire_runtime::loader_api::ModelSource::from_path(MODEL_DIR).expect("model source");
    let config = match &loader_src {
        hipfire_runtime::loader_api::ModelSource::Dir(s) => {
            hipfire_arch_qwen35::qwen35::config_from_safetensors(s).expect("config")
        }
        hipfire_runtime::loader_api::ModelSource::Hfq(h) => {
            hipfire_arch_qwen35::qwen35::config_from_hfq(h).expect("config")
        }
    };
    let max_rows: usize = *row_sets.iter().max().unwrap();
    let mut s = Qwen35Scratch::new(&mut gpu, &config, 1).expect("scratch");
    let pbs = PrefillBatchScratch::new(&mut gpu, &config, max_rows).expect("pbs");

    let mut any_fail = false;
    let mut worst_rel = 0f32;
    for &n_rows in &row_sets {
        // Same embedding row for every chunk: row t identical across n.
        let x: Vec<f32> = (0..n_rows)
            .flat_map(|r| rand_block(((layer as u32) << 16) | (r as u32).wrapping_mul(2654435761), DIM))
            .collect();
        let x_gpu = gpu.upload_f32(&x, &[n_rows * DIM]).expect("x upload");
        gpu.hip
            .memcpy_dtod(&pbs.x_batch.buf, &x_gpu.buf, n_rows * DIM * 4)
            .expect("x to pbs");
        // Positions 0..n (fresh cache).
        let pos_host: Vec<i32> = (0..n_rows as i32).collect();
        let pos_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(pos_host.as_ptr() as *const u8, n_rows * 4)
        };
        gpu.hip.memcpy_htod(&pbs.positions.buf, pos_bytes).expect("positions");

        // Fresh Q8 KV: one entry per MODEL layer would be needed by the real
        // forward; here we call the layer body directly. The per-token body
        // indexes kv_cache by model layer_idx, so size it to layer+1.
        let mut kv = hipfire_runtime::llama::KvCache::new_gpu_q8(
            &mut gpu,
            layer + 1,
            N_KV_HEADS,
            HEAD_DIM,
            128,
        )
        .expect("kv");

        hipfire_arch_qwen35::qwen35::escha_dense_forward::fullattn_escha_layer_prefill(
            &mut gpu,
            &layer_w,
            &config,
            n_rows,
            0,
            n_rows,
            layer,
            layer,
            &mut kv,
            &s,
            &pbs,
            0,
            BatchSemantics::Sequential,
            None,
        )
        .expect("batched FA");
        let got_full = gpu.download_f32(&pbs.x_batch).expect("download batched");
        let got = &got_full[..n_rows * DIM];

        // Ground truth: N per-token calls through the decode body.
        let mut want = vec![0f32; n_rows * DIM];
        {
            let mut kv2 = hipfire_runtime::llama::KvCache::new_gpu_q8(
                &mut gpu,
                layer + 1,
                N_KV_HEADS,
                HEAD_DIM,
                128,
            )
            .expect("kv2");
            for r in 0..n_rows {
                gpu.hip
                    .memcpy_dtod_at(&s.x.buf, 0, &x_gpu.buf, r * DIM * 4, DIM * 4)
                    .expect("row to s.x");
                let pos_i32 = r as i32;
                gpu.memcpy_htod_auto(&s.pos_buf, &pos_i32.to_ne_bytes()).expect("pos");
                fullattn_escha_layer_forward(
                    &mut gpu,
                    &layer_w,
                    &config,
                    r,
                    layer,
                    &mut kv2,
                    &s,
                    s.escha_dense_u.as_ref().expect("escha scratch"),
                    s.escha_dense_partial.as_ref().expect("escha scratch"),
                )
                .expect("per-token FA");
                let row = gpu.download_f32(&s.x).expect("row download");
                want[r * DIM..(r + 1) * DIM].copy_from_slice(&row);
            }
            let _ = kv2.free_gpu(&mut gpu);
        }

        let (abs, denom) = compare(got, &want);
        let rel = abs / denom;
        worst_rel = worst_rel.max(rel);
        let status = if rel <= 5e-3 { "PASS" } else { "FAIL" };
        if rel > 5e-3 {
            any_fail = true;
        }
        println!("  n_rows={n_rows:3}: rel={rel:.3e} abs={abs:.3e} [{status}]");
        let _ = gpu.free_tensor(x_gpu);
        let _ = kv.free_gpu(&mut gpu);
    }

    println!("\nworst rel = {worst_rel:.3e}");
    // Free owners (views above are non-owning aliases, never freed).
    for p in owned {
        let _ = gpu.free_tensor(p.code);
        let _ = gpu.free_tensor(p.in_scale);
        let _ = gpu.free_tensor(p.out_scale);
    }
    let _ = gpu.free_tensor(attn_norm);
    let _ = gpu.free_tensor(ffn_norm);
    let _ = gpu.free_tensor(q_norm);
    let _ = gpu.free_tensor(k_norm);
    if any_fail {
        eprintln!("FAIL: batched FA diverges from per-token beyond tolerance");
        std::process::exit(1);
    }
    println!("PASS");
}
