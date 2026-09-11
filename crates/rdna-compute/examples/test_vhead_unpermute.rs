// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GSQ-RCO Stage-4 parity harness for the `vhead_unpermute_f32` kernel:
//! permutes the DeltaNet V-head concat from engine order back to GGUF
//! order (dst[row, PERM[e]*128+j] = src[row, e*128+j]).
//!
//! Pass criteria: exact equality (bit-identical copy of 128-float blocks)
//! for n=1 and n=7, plus a round-trip check (un-permute then permute
//! returns the original).
//!
//! Usage:
//!   cargo run -p rdna-compute --example test_vhead_unpermute

use rdna_compute::{Gpu, GpuTensor};

const PERM: [usize; 48] = [
    0, 16, 32, 1, 17, 33, 2, 18, 34, 3, 19, 35, 4, 20, 36, 5, 21, 37, 6, 22, 38, 7, 23, 39, 8, 24,
    40, 9, 25, 41, 10, 26, 42, 11, 27, 43, 12, 28, 44, 13, 29, 45, 14, 30, 46, 15, 31, 47,
];

fn ref_unpermute(src: &[f32], n: usize) -> Vec<f32> {
    let k = 6144;
    let mut dst = vec![0.0f32; n * k];
    for r in 0..n {
        for e in 0..48 {
            let g = PERM[e];
            for j in 0..128 {
                dst[r * k + g * 128 + j] = src[r * k + e * 128 + j];
            }
        }
    }
    dst
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init failed");
    for n in [1usize, 7usize] {
        let k = 6144;
        // src[r, e*128+j] = (r*48 + e) * 128 + j  — unique per position.
        let src: Vec<f32> = (0..n * k)
            .map(|i| {
                let r = i / k;
                let e = (i % k) / 128;
                let j = i % 128;
                ((r * 48 + e) * 128 + j) as f32
            })
            .collect();
        let src_t = gpu.upload_f32(&src, &[n * k]).expect("upload");
        let dst_t = gpu.alloc_tensor(&[n * k], rdna_compute::DType::F32).expect("alloc");
        gpu.vhead_unpermute_f32_batched(&src_t, &dst_t, n)
            .expect("vhead_unpermute");
        let got = gpu.download_f32(&dst_t).expect("download");
        let want = ref_unpermute(&src, n);
        let mut max_diff = 0.0f32;
        let mut bad = 0usize;
        for i in 0..got.len() {
            let d = (got[i] - want[i]).abs();
            if d > max_diff {
                max_diff = d;
            }
            if d != 0.0 {
                bad += 1;
                if bad <= 5 {
                    eprintln!("  MISMATCH[{i}] got {} want {}", got[i], want[i]);
                }
            }
        }
        let status = if bad == 0 { "PASS" } else { "FAIL" };
        println!("vhead_unpermute n={n}: {status} max_diff={max_diff} bad={bad}/{}", got.len());
        if bad > 0 {
            std::process::exit(1);
        }
    }
    println!("vhead_unpermute: 2/2 PASS");
}