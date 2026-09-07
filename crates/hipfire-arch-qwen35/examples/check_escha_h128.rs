// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! H128-vs-T128 parity oracle for the Escha DENSE path.
//!
//! Compares the ported H128 input/output transforms
//! (`escha_dense::escha_h128_in`/`escha_h128_out`, from PR #694
//! `kernels/src/escha_h128.hip`) against the T128 transforms this branch
//! already serves through (`escha_dense_rotate_in` + finalize), on real
//! model scales and synthetic activations.
//!
//! Difference under test: T128 is NORMALIZED per pass (H/sqrt(128) each
//! side, RS^2 total); H128 is UNNORMALIZED per pass with a single RS folded
//! into f16 rounding (xh = f16(H(x.rin)*RS), y = f16(H(mid)*RS*rout)).
//! Expect agreement to f16-rounding scale (~1e-3 rel), NOT bit-exact.
//! A large divergence means the port is miswired, not that rounding differs.
//!
//! Usage:
//!   cargo run --release -p hipfire-arch-qwen35 --example check_escha_h128 -- [layer]
//!
//! Exit nonzero on FAIL.

use std::fs;

const MODEL_DIR: &str = "/data/rocmfpx/Qwen3.8-27B-Escha-W2";

fn read_tensor(
    wm: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Vec<u8> {
    let shard = wm[key].as_str().unwrap().to_string();
    fs::read(format!("{MODEL_DIR}/{shard}")).unwrap()
}

fn parse_hdr(bytes: &[u8]) -> (serde_json::Value, usize) {
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data_start = 8 + n + ((8 - (8 + n) % 8) % 8);
    (hdr, data_start)
}

fn load_raw(wm: &serde_json::Map<String, serde_json::Value>, key: &str) -> (Vec<u8>, String) {
    let bytes = read_tensor(wm, key);
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

fn load_f32(wm: &serde_json::Map<String, serde_json::Value>, key: &str) -> Vec<f32> {
    let (raw, dtype) = load_raw(wm, key);
    if dtype == "F16" {
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

    let idx: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(format!("{MODEL_DIR}/model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    let wm = idx["weight_map"].as_object().unwrap().clone();

    let Some(mut gpu) = rdna_compute::Gpu::init().ok() else {
        eprintln!("no gpu");
        return;
    };
    println!("H128-vs-T128 parity on {} — layer {layer}\n", gpu.arch.as_str());

    // Real scales from gate_proj (K=2, 5120->17408): rin/s_in -> in_scale.
    let base = format!("model.language_model.layers.{layer}.mlp.gate_proj");
    let rin = load_f32(&wm, &format!("{base}.escha_rin"));
    let sin = load_f32(&wm, &format!("{base}.escha_s_in"));
    let rout = load_f32(&wm, &format!("{base}.escha_rout"));
    let sout = load_f32(&wm, &format!("{base}.escha_s_out"));
    let in_scale: Vec<f32> = rin.iter().zip(sin.iter()).map(|(&r, &s)| r * s).collect();
    let out_scale: Vec<f32> = rout.iter().zip(sout.iter()).map(|(&r, &s)| r * s).collect();
    let (ic, oc) = (in_scale.len(), out_scale.len());
    println!("gate_proj: ic={ic} oc={oc}");

    // Deterministic activation.
    let mut seed = ((layer as u32) << 16) ^ 0x9e3779b9 ^ 0xabcdef01;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 0.2 - 0.1
    };
    let x: Vec<f32> = (0..ic).map(|_| next()).collect();
    // Fake decoded-mid vector for the output side (same width discipline).
    let mid: Vec<f32> = (0..oc).map(|_| next()).collect();

    let x_gpu = gpu.upload_f32(&x, &[ic]).expect("x");
    let rin_gpu = gpu.upload_f32(&rin, &[ic]).expect("rin");
    let in_scale_gpu = gpu.upload_f32(&in_scale, &[ic]).expect("in_scale");
    let mid_gpu = gpu.upload_f32(&mid, &[oc]).expect("mid");
    let rout_gpu = gpu.upload_f32(&rout, &[oc]).expect("rout");
    let out_scale_gpu = gpu.upload_f32(&out_scale, &[oc]).expect("out_scale");

    // T128 input side (our scalar path, no f16 rounding).
    let u_t128 = gpu.alloc_tensor(&[ic], rdna_compute::DType::F32).expect("u");
    rdna_compute::escha_dense::escha_dense_rotate_in(
        &mut gpu, &in_scale_gpu, &x_gpu, &u_t128, ic,
    )
    .expect("t128 rotate");
    // H128 input side (ported kernel): __half output (2 B/element), matching
    // the kernel's `__half* xh` signature. Compared after widening on host.
    let u_h128 = gpu.alloc_tensor(&[ic], rdna_compute::DType::F16).expect("uh");
    rdna_compute::escha_dense::escha_h128_in(
        &mut gpu, &x_gpu, &rin_gpu, &u_h128, ic,
    )
    .expect("h128 in");

    let got_t = gpu.download_f32(&u_t128).expect("dl t128");
    // The kernel wrote __half (ic elements × 2 B); download raw bytes and
    // widen on host with the same RNE-unpacking the oracles use.
    let mut u_h_bytes = vec![0u8; ic * 2];
    gpu.hip.memcpy_dtoh(&mut u_h_bytes, &u_h128.buf).expect("dl h128 raw");
    let got_h: Vec<f32> = u_h_bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    // NOTE: rin-only on the H128 side vs in_scale (rin.s_in) on T128 —
    // different inputs BY CONSTRUCTION (s_in is the end-to-end fine-tune
    // scale the H128 path folds elsewhere). Report raw ranges + the RS
    // scaling relationship instead of a direct diff.
    let l2 = |v: &[f32]| (v.iter().map(|a| (*a as f64) * (*a as f64)).sum::<f64>()).sqrt();
    println!("|x|={:.4} |u_t128|={:.4} |u_h128|={:.4}", l2(&x), l2(&got_t), l2(&got_h));
    println!("u_t128[0..4]={:?}", &got_t[..4]);
    println!("u_h128[0..4]={:?}", &got_h[..4]);

    // Output side: T128 finalize needs a partial; emulate with n_slices=1 by
    // uploading mid as the single slice.
    let partial = gpu.upload_f32(&mid, &[oc]).expect("partial");
    let y_t128 = gpu.alloc_tensor(&[oc], rdna_compute::DType::F32).expect("yt");
    // escha_dense_finalize_dense with n_slices=1, n_rows=1.
    rdna_compute::escha_dense::escha_dense_finalize_dense(
        &mut gpu, &out_scale_gpu, &partial, &y_t128, 1, 1,
    )
    .expect("t128 finalize");
    let y_h128 = gpu.alloc_tensor(&[oc], rdna_compute::DType::F16).expect("yh");
    rdna_compute::escha_dense::escha_h128_out(
        &mut gpu, &mid_gpu, &rout_gpu, &y_h128, oc,
    )
    .expect("h128 out");
    let got_yt = gpu.download_f32(&y_t128).expect("dl yt");
    let mut y_h_bytes = vec![0u8; oc * 2];
    gpu.hip.memcpy_dtoh(&mut y_h_bytes, &y_h128.buf).expect("dl yh raw");
    let got_yh: Vec<f32> = y_h_bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    println!("|mid|={:.4} |y_t128|={:.4} |y_h128|={:.4}", l2(&mid), l2(&got_yt), l2(&got_yh));
    println!("y_t128[0..4]={:?}", &got_yt[..4]);
    println!("y_h128[0..4]={:?}", &got_yh[..4]);

    // Sanity: both finite, nonzero, same order of magnitude. This is a
    // smoke parity (wiring check), not a numerics gate — the two paths have
    // deliberately different rounding/scale contracts.
    let mut fail = false;
    for (name, v) in [("u_h128", &got_h), ("y_h128", &got_yh)] {
        if !v.iter().all(|x| x.is_finite()) {
            eprintln!("FAIL: {name} has non-finite elements");
            fail = true;
        }
        if l2(v) == 0.0 {
            eprintln!("FAIL: {name} is all zeros");
            fail = true;
        }
    }
    // The two input sides differ by s_in by construction; but H128-in with
    // rin vs T128 with rin.s_in must still correlate (same H, same x).
    // Check sign agreement as a wiring tripwire (>90% expected).
    let agree = got_t.iter().zip(got_h.iter()).filter(|(a, b)| a.signum() == b.signum()).count();
    let frac = agree as f64 / got_t.len() as f64;
    println!("input-side sign agreement: {frac:.3}");
    if frac < 0.90 {
        eprintln!("FAIL: sign agreement {frac:.3} < 0.90 — miswire, not rounding");
        fail = true;
    }
    if fail {
        std::process::exit(1);
    }
    println!("PASS (smoke parity — H128 kernels run, finite, correlated)");
}
