// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Achievable device memory bandwidth on the current GPU.
//!
//! The Escha-dense decode analysis has told two contradictory stories, and they
//! imply opposite work:
//!
//!   * "decode is instruction/issue-bound" (only ~4 VALU ops per weight, 348
//!     static instructions) -> rewrite the kernel, cut ops per weight.
//!   * "decode is bandwidth-bound" -> the only levers are fewer bytes per step
//!     (int8 `lm_head`, smaller split-K partials) or more tokens per step
//!     (MTP speculation).
//!
//! The arithmetic that decides it: the decode step moves ~13 GB of weights per
//! token, so 3.4 tok/s implies ~44 GB/s of useful traffic. Whether 44 GB/s is
//! 15% of this machine or 85% of it depends on what the machine sustains, and
//! that had never been measured. Strix Halo's LPDDR5x is ~256 GB/s theoretical;
//! an APU that also drives a display and shares controllers with 40 CPU
//! complexes is a different number, and a 27B model never fits in L2, so every
//! weight byte is a real DRAM access.
//!
//! D2D copy is used because it is unambiguous: n words read + n written, with no
//! kernel to accidentally optimise away. The read-only half of the same transfer
//! is printed separately since a GEMV only reads.
//!
//! Usage:
//!   cargo run --release -p rdna-compute --example mem_bw -- [size_mib] [reps]

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mib: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let n = mib * 1024 * 1024 / 4; // f32 elements

    let mut gpu = match rdna_compute::Gpu::init() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("no gpu");
            return;
        }
    };
    println!(
        "device: {}   buffer: {} MiB   reps: {reps}",
        gpu.arch.as_str(),
        mib
    );
    println!("  ({} MiB is far beyond L2, so every byte is a DRAM access)\n", mib);

    // Real non-zero data, so nothing can be folded away.
    let host = vec![1.5f32; n];
    let a = gpu.upload_f32(&host, &[n]).expect("alloc a");
    let b = gpu.zeros(&[n], rdna_compute::DType::F32).expect("alloc b");

    let mut best = f64::MAX;
    let mut worst = 0f64;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        gpu.memcpy_dtod_auto(&b.buf, &a.buf, n * 4).expect("copy");
        let _ = gpu.hip.device_synchronize();
        let dt = t.elapsed().as_secs_f64();
        best = best.min(dt);
        worst = worst.max(dt);
    }
    let bytes = n as f64 * 4.0;
    println!("  d2d copy   r+w : {:7.1} GB/s   ({:.2} ms best, {:.2} ms worst)",
        2.0 * bytes / best / 1e9, best * 1e3, worst * 1e3);
    println!("  read side only : {:7.1} GB/s   (the shape a weight-streaming GEMV wants)",
        bytes / best / 1e9);

    // Small-buffer run to expose how much of the big number is cache.
    let small = 8 * 1024 * 1024 / 4;
    let sa = gpu.upload_f32(&host[..small], &[small]).expect("alloc sa");
    let sb = gpu.zeros(&[small], rdna_compute::DType::F32).expect("alloc sb");
    let mut best_s = f64::MAX;
    for _ in 0..reps {
        let t = std::time::Instant::now();
        gpu.memcpy_dtod_auto(&sb.buf, &sa.buf, small * 4).expect("copy s");
        let _ = gpu.hip.device_synchronize();
        best_s = best_s.min(t.elapsed().as_secs_f64());
    }
    println!("  8 MiB (L2-resident) r+w: {:7.1} GB/s  <- cache, not DRAM",
        2.0 * small as f64 * 4.0 / best_s / 1e9);

    let _ = gpu.free_tensor(a);
    let _ = gpu.free_tensor(b);
    let _ = gpu.free_tensor(sa);
    let _ = gpu.free_tensor(sb);

    let read_gbs = bytes / best / 1e9;
    let useful = 13.0 / (1.0 / 3.4);
    println!(
        "\nDecode step: ~13 GB of weights per token at 3.4 tok/s = {:.0} GB/s useful.\n\
         That is {:.0}% of this machine's sustained read bandwidth ({:.0} GB/s).",
        useful,
        100.0 * useful / read_gbs,
        read_gbs
    );
    println!(
        "  -> {} bound",
        if useful / read_gbs > 0.6 {
            "BANDWIDTH: fewer bytes per step (int8 lm_head) or more tokens per step (MTP). \
             Rewriting GEMV inner loops will not move tg."
        } else {
            "NOT bandwidth-bound: the kernels themselves are the limit; cut ops per weight."
        }
    );
}
