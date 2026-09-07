// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Where does Escha-dense DECODE time actually go?
//!
//! The plan told two contradictory stories about the 3.4 tok/s step and neither
//! was measured:
//!
//!   A. instruction/issue-bound -> cut VALU ops per weight, fuse the payload
//!      reads into one LDS.64 (items 6 and 7).
//!   B. bandwidth-bound -> move fewer bytes: int8 `lm_head` (item 2).
//!
//! The first clue came from `profile_escha_dense_prefill`: `gate_proj` (K=2,
//! 22.3 MB of code) and `down_proj` (K=3, 33.4 MB) — same 89.1 M weights — both
//! take ~1.0 ms per token. 50% more bytes costs 5% more time, and both land on
//! ~89 G weights/s. A DRAM-bound kernel cannot be byte-insensitive, so B is
//! wrong about the *projections* and int8 lm_head is worth much less than the
//! plan claims.
//!
//! But "not bandwidth" does not automatically mean "VALU-bound": hand arithmetic
//! puts VALU at ~10% of issue capacity and LDS at ~100 G lane-ops/s against
//! 48.7 G/s needed, which points at neither. The remaining candidates are the
//! per-tile barrier pair (two `__syncthreads()` with only 16 r-iterations of
//! work between them) and raw launch overhead (3 kernels per projection, ~400
//! projections per token, so ~1200 launches per step).
//!
//! This example separates them by timing each stage on its own, and also times
//! an empty-launch baseline so launch cost is a measured constant rather than an
//! assumption:
//!
//!   rotate_in -> u            (read IC floats, write IC floats, 2 barriers)
//!   decode_gemv -> partial    (the trellis decode + FMA loop)
//!   finalize -> y             (read n_slices*OC, WHT, scale)
//!
//! and repeats each with n_slices forced to 1 vs the heuristic, because if the
//! gemv's cost scales with the *number of tiles* rather than the number of
//! weights, the barrier is the limit and fusing/raising work-per-thread is the
//! fix — whereas if a bare launch costs as much as a whole stage, launch
//! reduction (hipGraph) is the fix.
//!
//! Usage:
//!   cargo run --release -p hipfire-arch-qwen35 --example decode_stage_probe \
//!     -- [layer] [proj] [reps]

use std::fs;
use std::time::Instant;

const MODEL_DIR: &str = "/data/rocmfpx/Qwen3.8-27B-Escha-W2";

fn f16(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 {
        if m == 0 {
            return if s == 1 { -0.0f32 } else { 0.0f32 };
        }
        let mut ee = 1u32;
        let mut mm = m;
        while mm & 0x400 == 0 {
            mm <<= 1;
            ee += 1;
        }
        mm &= 0x3ff;
        f32::from_bits((s << 31) | ((127 - ee + 15) << 23) | (mm << 13))
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

/// (tensor bytes, dtype string) for one key.
fn raw_and_hdr(
    wm: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> (Vec<u8>, serde_json::Value) {
    let bytes = fs::read(format!("{MODEL_DIR}/{}", wm[key].as_str().unwrap())).unwrap();
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let ds = 8 + n + ((8 - (8 + n) % 8) % 8);
    let meta = &hdr[key];
    let (b, e) = (
        meta["data_offsets"][0].as_u64().unwrap() as usize,
        meta["data_offsets"][1].as_u64().unwrap() as usize,
    );
    (bytes[ds + b..ds + e].to_vec(), hdr.clone())
}

fn sync(gpu: &rdna_compute::Gpu) {
    let _ = gpu.hip.device_synchronize();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let layer: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let proj = a.get(2).cloned().unwrap_or_else(|| "mlp.gate_proj".to_string());
    let reps: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(30);

    let idx: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(format!("{MODEL_DIR}/model.safetensors.index.json")).unwrap())
            .unwrap();
    let wm = idx["weight_map"].as_object().unwrap().clone();
    let base = format!("model.language_model.layers.{layer}.{proj}");
    let ck = format!("{base}.escha_code");
    if wm.get(&ck).is_none() {
        eprintln!("no escha_code at {base}");
        return;
    }
    let (cbytes, chdr) = raw_and_hdr(&wm, &ck);
    let shape: Vec<usize> = chdr[&ck]["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let (in_p, out_p, k) = (shape[0] * 16, shape[1] * 16, shape[2] / 16);
    let code: Vec<i16> = cbytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
    let code_mb = (code.len() * 2) as f64 / 1e6;

    let ld = |suf: &str| -> Vec<f32> {
        let (raw, h) = raw_and_hdr(&wm, &format!("{base}.escha_{suf}"));
        if h[&format!("{base}.escha_{suf}")]["dtype"].as_str() == Some("F16") {
            raw.chunks_exact(2).map(|c| f16(u16::from_le_bytes([c[0], c[1]]))).collect()
        } else {
            raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
    };
    let (rin, rout, sin, sout) = (ld("rin"), ld("rout"), ld("s_in"), ld("s_out"));
    let in_scale: Vec<f32> = rin.iter().zip(&sin).map(|(x, y)| x * y).collect();
    let out_scale: Vec<f32> = rout.iter().zip(&sout).map(|(x, y)| x * y).collect();

    let mut gpu = match rdna_compute::Gpu::init() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("no gpu");
            return;
        }
    };
    println!(
        "decode stage probe: {} L{layer} {proj}  {in_p}x{out_p} K={k}  code {code_mb:.1} MB",
        gpu.arch.as_str()
    );
    let nit = in_p / 16;
    let n_ocb = out_p / 128;

    let cg = gpu
        .upload_raw(unsafe { std::slice::from_raw_parts(code.as_ptr() as *const u8, code.len() * 2) },
            &[shape[0], shape[1], shape[2]])
        .unwrap();
    let ig = gpu.upload_f32(&in_scale, &[in_p]).unwrap();
    let og = gpu.upload_f32(&out_scale, &[out_p]).unwrap();
    let x = gpu.upload_f32(&vec![0.05f32; in_p], &[in_p]).unwrap();
    let u = gpu.zeros(&[in_p], rdna_compute::DType::F32).unwrap();

    // ── empty-ish launch cost: rotate_in alone (IC floats, no decode) ────
    for _ in 0..5 {
        let _ = rdna_compute::escha_dense::escha_dense_rotate_in(&mut gpu, &ig, &x, &u, in_p);
    }
    sync(&gpu);
    let t = Instant::now();
    for _ in 0..reps {
        rdna_compute::escha_dense::escha_dense_rotate_in(&mut gpu, &ig, &x, &u, in_p).unwrap();
    }
    sync(&gpu);
    let rot_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;

    // ── gemv alone, at the heuristic n_slices and at n_slices = 1 ────────
    let mut bench_gemv = |ns: usize| -> f64 {
        let part = gpu.zeros(&[ns * out_p], rdna_compute::DType::F32).unwrap();
        for _ in 0..5 {
            let _ = rdna_compute::escha_dense::escha_dense_decode_gemv_stage(
                &mut gpu, &cg, &u, &part, in_p, out_p, ns, k as i32);
        }
        sync(&gpu);
        let t = Instant::now();
        for _ in 0..reps {
            rdna_compute::escha_dense::escha_dense_decode_gemv_stage(
                &mut gpu, &cg, &u, &part, in_p, out_p, ns, k as i32).unwrap();
        }
        sync(&gpu);
        let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let _ = gpu.free_tensor(part);
        ms
    };
    let ns_h = rdna_compute::escha_dense::escha_dense_n_slices(nit, out_p);
    let gemv_h = bench_gemv(ns_h);
    let gemv_1 = bench_gemv(1);
    let ns_half = (ns_h / 2).max(1);
    let gemv_half = bench_gemv(ns_half);
    // Occupancy sweep: the gemv is latency-bound, so where the curve turns over
    // is the real target block count. Re-measure after any inner-loop change —
    // cutting per-thread work moves the optimum.
    println!("  n_slices sweep (same bytes, more/fewer blocks):");
    for ns in [1usize, 2, 4, 7, 10, 16, 20, 32, 64, 128, 160, 320] {
        if ns > nit {
            break;
        }
        let ms = bench_gemv(ns);
        println!(
            "    ns={ns:4} blocks={:<6} {:7.4} ms  {:6.1} GB/s code  {:.0} G weights/s",
            n_ocb * ns,
            ms,
            code_mb * 1e-3 / (ms / 1e3),
            (in_p * out_p) as f64 / (ms / 1e3) / 1e9
        );
    }

    // ── finalize alone ──────────────────────────────────────────────────
    let part = gpu.zeros(&[ns_h * out_p], rdna_compute::DType::F32).unwrap();
    let y = gpu.zeros(&[out_p], rdna_compute::DType::F32).unwrap();
    for _ in 0..5 {
        let _ = rdna_compute::escha_dense::escha_dense_finalize_dense(
            &mut gpu, &og, &part, &y, 1, ns_h);
    }
    sync(&gpu);
    let t = Instant::now();
    for _ in 0..reps {
        rdna_compute::escha_dense::escha_dense_finalize_dense(
            &mut gpu, &og, &part, &y, 1, ns_h).unwrap();
    }
    sync(&gpu);
    let fin_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;

    println!("  grid: n_ocb={n_ocb} blocks x n_slices   (nit={nit})");
    println!("  rotate_in  {rot_ms:8.4} ms   (1 launch, {in_p} floats in/out)");
    println!("  decode_gemv n_slices={ns_h:4}: {gemv_h:8.4} ms   -> {:.1} G weights/s",
        (in_p * out_p) as f64 / (gemv_h / 1e3) / 1e9);
    println!("  decode_gemv n_slices={ns_half:4}: {gemv_half:8.4} ms");
    println!("  decode_gemv n_slices=   1: {gemv_1:8.4} ms   ({} blocks, no slicing)", n_ocb);
    println!("  finalize   {fin_ms:8.4} ms   (1 launch)");
    println!("  ----");
    let tot = rot_ms + gemv_h + fin_ms;
    println!("  3-kernel total {tot:8.4} ms of the ~1.0 ms measured per projection");
    println!("  whole-step projection traffic implied: {:.1} GB/s (device sustains ~208 GB/s copy / ~104 GB/s read)",
        code_mb * 1e-3 * 400.0 / (tot / 1e3));
    println!("\nRead the gemv row: if n_slices=1 is ~= n_slices={ns_h}, the cost is per-BLOCK\
              \n  (barriers/launch) and fusion wins; if cost scales with tiles touched, it is\
              \n  the decode loop and op-reduction wins.");

    for t in [cg, ig, og, x, u, part, y] {
        let _ = gpu.free_tensor(t);
    }
}
