// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched-prefill projection oracle for the Escha DENSE path.
//!
//! `check_escha_dense` validates the single-token DECODE gemv against the host
//! reference. Nothing validated `escha_dense_decode_proj_batch`, which is why a
//! wrong batched body survived both its authoring and its "fix": re-opening the
//! batched-prefill eligibility gate measured 3.5x on prefill but changed the
//! model's greedy output from the first generated token.
//!
//! This closes that gap, and splits the remaining question by construction:
//!
//!   * R == 1 through the BATCHED path exercises `escha_dense_rotate_in_dense`
//!     + `escha_dense_finalize_dense` but the kernel's single-row branch.
//!   * R > 1 additionally exercises the multi-row staging and the `acc[R]` loop.
//!
//! So "R=1 fails too" means the shared rotate/finalize is wrong, while "R=1
//! passes and R>=2 fails" localises the defect to the multi-row path.
//!
//! Usage:
//!   cargo run --release -p hipfire-arch-qwen35 --example check_escha_dense_batched -- [layer] [rows...]
//!     layer: 0..63 (default 0)
//!     rows:  prompt rows to test (default 1 2 4 8 16 32)
//!
//! Every coded projection family present in the checkpoint is checked for that
//! layer, including both code rates: gate/out/qkv are K=2 and up/down are K=3,
//! so a K-specific defect shows up as a per-projection pass/fail split.
//!
//! NOTE: needs a GPU (it is a GPU-vs-host check), but it does NOT load the
//! model — it uploads one projection at a time, so it runs in seconds and a few
//! hundred MB instead of the ~10 GB full load.

use std::fs;

const MODEL_DIR: &str = "/data/rocmfpx/Qwen3.8-27B-Escha-W2";

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
    // safetensors: header is padded so (8 + n + pad) % 8 == 0
    let data_start = 8 + n + ((8 - (8 + n) % 8) % 8);
    (hdr, data_start)
}

/// Raw little-endian tensor bytes for `key`.
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
            if s == 1 {
                -0.0
            } else {
                0.0
            }
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

/// Load one coded projection: code + folded in/out scales.
struct Proj {
    base: String,
    code: Vec<i16>,
    k: usize,
    in_p: usize,
    out_p: usize,
    in_scale: Vec<f32>,
    out_scale: Vec<f32>,
}

fn load_proj(wm: &serde_json::Map<String, serde_json::Value>, base: &str) -> Option<Proj> {
    let code_key = format!("{base}.escha_code");
    if wm.get(&code_key).is_none() {
        return None;
    }
    let (bytes, _s) = read_tensor(wm, &code_key);
    let (hdr, data_start) = parse_hdr(&bytes);
    let cm = &hdr[&code_key];
    let shape: Vec<usize> = cm["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let (t_in, t_out, last) = (shape[0], shape[1], shape[2]);
    let (in_p, out_p, k) = (t_in * 16, t_out * 16, last / 16);
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
    // The loader folds Wscale into rin, so in_scale = rin * s_in and
    // out_scale = rout * s_out. Do NOT apply Wscale again.
    let in_scale = rin.iter().zip(sin.iter()).map(|(&r, &s)| r * s).collect();
    let out_scale = rout.iter().zip(sout.iter()).map(|(&r, &s)| r * s).collect();
    Some(Proj {
        base: base.to_string(),
        code,
        k,
        in_p,
        out_p,
        in_scale,
        out_scale,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let layer: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let row_sets: Vec<usize> = if args.len() > 2 {
        args[2..].iter().filter_map(|s| s.parse().ok()).collect()
    } else {
        vec![1, 2, 4, 8, 16, 32]
    };

    let idx: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(format!("{MODEL_DIR}/model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    let wm = idx["weight_map"].as_object().unwrap().clone();

    let is_fa = matches!(layer % 4, 3);
    let mut projs: Vec<&str> = vec![
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ];
    if layer == 0 || layer == 1 {
        // in_proj_a / in_proj_b are dense f16, not coded, so they are skipped.
        let _ = 0;
    }
    if is_fa {
        projs.extend(["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]);
    } else {
        projs.extend([
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.out_proj",
        ]);
    }

    let Some(mut gpu) = rdna_compute::Gpu::init().ok() else {
        eprintln!("no gpu: this oracle compares GPU output against the host reference");
        return;
    };

    let mut worst_rel = 0f32;
    let mut any_fail = false;

    for pname in projs {
        let base = format!("model.language_model.layers.{layer}.{pname}");
        let Some(p) = load_proj(&wm, &base) else {
            eprintln!("  (skip {pname}: no escha_code at this layer)");
            continue;
        };
        let code = p.code.clone();
        let (in_p, out_p, k) = (p.in_p, p.out_p, p.k);

        // Upload as the loader would: code as raw i16-shaped bits, scales f32.
        let code_gpu = gpu
            .upload_raw(
                unsafe {
                    std::slice::from_raw_parts(code.as_ptr() as *const u8, code.len() * 2)
                },
                &[in_p / 16, out_p / 16, k * 16],
            )
            .expect("code upload");
        let in_scale_gpu = gpu.upload_f32(&p.in_scale, &[in_p]).expect("in_scale");
        let out_scale_gpu = gpu.upload_f32(&p.out_scale, &[out_p]).expect("out_scale");
        let pw = hipfire_arch_qwen35::qwen35::EschaDenseProjWeights {
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

        println!(
            "\n=== L{layer} {pname}  {in_p}x{out_p} K={k}  ({})",
            if k == 3 { "3-bit rate" } else { "2-bit rate" }
        );

        for &n_rows in &row_sets {
            // Deterministic pseudo-random activations, one block per row so a
            // row-dependent defect cannot average away.
            let mut x = vec![0f32; n_rows * in_p];
            for r in 0..n_rows {
                let mut seed = ((layer as u32) << 16)
                    ^ ((r as u32).wrapping_mul(2654435761))
                    ^ (in_p as u32)
                    ^ 0x9e3779b9;
                let mut next = || {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    (seed as f32 / u32::MAX as f32) * 0.2 - 0.1
                };
                for v in x[r * in_p..(r + 1) * in_p].iter_mut() {
                    *v = next();
                }
            }

            let x_gpu = gpu.upload_f32(&x, &[n_rows * in_p]).expect("x upload");
            let y_gpu = gpu
                .alloc_tensor(&[n_rows * out_p], rdna_compute::DType::F32)
                .expect("y alloc");
            hipfire_arch_qwen35::qwen35::escha_dense_forward::escha_dense_decode_proj_batch(
                &mut gpu, &pw, &x_gpu, &y_gpu, n_rows,
            )
            .expect("batched proj");
            let got = gpu.download_f32(&y_gpu).expect("download");

            // Host reference, row by row.
            let r = rdna_compute::escha_dense::escha_dense_prefill_r(n_rows);
            let mut max_abs = 0f32;
            let mut denom = 1e-12f32;
            let mut bad_row = usize::MAX;
            let mut bad_rel = 0f32;
            for row in 0..n_rows {
                let want =
                    hipfire_arch_qwen35::qwen35::escha_dense_decode::escha_dense_decode_proj_host(
                        &code,
                        k,
                        in_p,
                        out_p,
                        &p.in_scale,
                        &p.out_scale,
                        &x[row * in_p..(row + 1) * in_p],
                    );
                for o in 0..out_p {
                    let g = got[row * out_p + o];
                    let d = (g - want[o]).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                    if want[o].abs() > denom {
                        denom = want[o].abs();
                    }
                }
                let rr = (0..out_p)
                    .map(|o| (got[row * out_p + o] - want[o]).abs())
                    .fold(0f32, f32::max)
                    / denom;
                if rr > bad_rel {
                    bad_rel = rr;
                    bad_row = row;
                }
            }
            let _ = gpu.free_tensor(y_gpu);
            let _ = gpu.free_tensor(x_gpu);

            let pass = bad_rel < 1e-3;
            any_fail |= !pass;
            worst_rel = worst_rel.max(bad_rel);
            println!(
                "  n_rows={n_rows:3} (R={r:2}) worst_row={bad_row:2} rel={bad_rel:.3e}  {}",
                if pass { "PASS" } else { "FAIL" }
            );
        }
    }

    println!(
        "\nORACLE: L{layer} worst rel = {worst_rel:.3e}  -> {}",
        if any_fail { "FAIL" } else { "PASS" }
    );
    if any_fail {
        std::process::exit(1);
    }
}
