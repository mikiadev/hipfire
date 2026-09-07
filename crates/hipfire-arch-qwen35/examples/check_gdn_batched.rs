// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched conv/GDN carry oracle for the Escha DENSE prefill path.
//!
//! Companion to `check_escha_dense_batched`, which established two things: the
//! batched coded PROJECTIONS are correct (<= 4e-4 rel vs the host reference at
//! every R), AND they are 6.8-14.2x faster per row than per-token. So the
//! projections are neither the correctness problem nor the speed problem.
//!
//! What remains in the batched arm is the recurrent carry across a chunk of
//! rows. The two halves are tested INDEPENDENTLY here, because they fail in
//! different ways and the conv output width (k_dim = 2048) is not the GDN input
//! width (n_v_heads * head_dim = 6144, post repeat_interleave) — conflating
//! them hides which one drifted.
//!
//!   conv : `conv1d_silu_split_f32` (per-token) vs `.._n(.., n_tokens = N)`.
//!          The per-token call is literally `_n(.., 1)`, so any difference is
//!          the token loop and the ring-buffer advance. That kernel's own
//!          comment says keeping the weight indexing un-hoisted is
//!          "LOAD-BEARING for byte-exact output against the single-token
//!          baseline — hoisting the loads changes the compiler's FMA fusion
//!          decisions and nudges the numerics by ~1 ULP, which cascades over
//!          2k+ greedy tokens". Byte-exactness with the per-token path is this
//!          repo's intended contract, not a standard I imposed.
//!   gdn  : `gated_delta_net_f32` vs `gated_delta_net_f32_batch_seq` —
//!          ENTIRELY DIFFERENT kernels. Prime suspect.
//!
//! Ground truth is the per-token arm applied N times, i.e. what decode does.
//! Both the per-step output and the carried state are compared, because a state
//! error is what silently poisons every later row.
//!
//! Usage:
//!   cargo run --release -p hipfire-arch-qwen35 --example check_gdn_batched \
//!     -- [n_rows...]        (default: 2 4 8 16 32 58)
//!
//! Needs a GPU. Loads no model weights (synthetic inputs), so it takes seconds
//! and a few hundred MB rather than the ~10 GB full load.

const N_V_HEADS: usize = 48; // linear_num_value_heads
const N_K_HEADS: usize = 16; // linear_num_key_heads
const HEAD_DIM: usize = 128;
const K_DIM: usize = N_K_HEADS * HEAD_DIM; // conv q/k out: 2048
const V_DIM: usize = N_V_HEADS * HEAD_DIM; // conv v out / gdn v: 6144
const QKV_DIM: usize = 2 * K_DIM + V_DIM; // conv input width: 10240
/// GDN takes q/k after repeat_interleave, so n_v_heads wide.
const GDN_QK: usize = N_V_HEADS * HEAD_DIM; // 6144
const STATE_ELEMS: usize = N_V_HEADS * HEAD_DIM * HEAD_DIM; // S-matrix: 786432

fn rand_block(seed0: u32, n: usize, scale: f32, offset: f32) -> Vec<f32> {
    let mut seed = seed0.wrapping_mul(2654435761) ^ 0x9e37_79b9;
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            offset + (seed as f32 / u32::MAX as f32) * scale
        })
        .collect()
}

/// (max abs difference, largest reference magnitude).
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

fn rel(got: &[f32], want: &[f32]) -> f32 {
    let (a, d) = compare(got, want);
    a / d
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rows: Vec<usize> = if args.len() > 1 {
        args[1..].iter().filter_map(|s| s.parse().ok()).collect()
    } else {
        vec![2, 4, 8, 16, 32, 58]
    };

    let mut gpu = match rdna_compute::Gpu::init() {
        Ok(g) => g,
        Err(_) => {
            eprintln!("no gpu");
            return;
        }
    };
    println!(
        "conv/GDN carry oracle on {} — ground truth = N sequential per-token calls\n",
        gpu.arch.as_str()
    );

    let mut worst = [0f32; 4];

    for &n in &rows {
        // Inputs keyed so that row t is the same vector for every n: a defect
        // that only appears for particular rows stays visible across the sweep.
        let conv_in = rand_block(0x6000 + n as u32, n * QKV_DIM, 2.0, -1.0);
        let conv_w = rand_block(0x7000, QKV_DIM * 4, 1.0, -0.5);
        let conv_state0 = rand_block(0x8000 + n as u32, QKV_DIM * 3, 1.0, -0.5);
        let q = rand_block(0x1000 + n as u32, n * GDN_QK, 0.2, -0.1);
        let k = rand_block(0x2000 + n as u32, n * GDN_QK, 0.2, -0.1);
        let v = rand_block(0x3000 + n as u32, n * V_DIM, 0.4, -0.2);
        let gate = rand_block(0x4000 + n as u32, n * N_V_HEADS, 4.0, -3.0);
        let beta = rand_block(0x5000 + n as u32, n * N_V_HEADS, 1.0, 0.0);
        let s0 = rand_block(0x9000 + n as u32, STATE_ELEMS, 0.2, -0.1);

        // ══ 1. conv carry ════════════════════════════════════════════════
        let cw = gpu.upload_f32(&conv_w, &[QKV_DIM * 4]).unwrap();
        let mut pt_q = vec![0f32; n * K_DIM];
        let mut pt_k = vec![0f32; n * K_DIM];
        let mut pt_v = vec![0f32; n * V_DIM];
        let mut pt_cs = gpu.upload_f32(&conv_state0, &[QKV_DIM * 3]).unwrap();
        for t in 0..n {
            let row = gpu
                .upload_f32(&conv_in[t * QKV_DIM..(t + 1) * QKV_DIM], &[QKV_DIM])
                .unwrap();
            let qo = gpu.zeros(&[K_DIM], rdna_compute::DType::F32).unwrap();
            let ko = gpu.zeros(&[K_DIM], rdna_compute::DType::F32).unwrap();
            let vo = gpu.zeros(&[V_DIM], rdna_compute::DType::F32).unwrap();
            gpu.conv1d_silu_split_f32(&qo, &ko, &vo, &row, &cw, &mut pt_cs, K_DIM, V_DIM)
                .expect("per-token conv");
            let h = gpu.download_f32(&qo).unwrap();
            pt_q[t * K_DIM..(t + 1) * K_DIM].copy_from_slice(&h);
            let h = gpu.download_f32(&ko).unwrap();
            pt_k[t * K_DIM..(t + 1) * K_DIM].copy_from_slice(&h);
            let h = gpu.download_f32(&vo).unwrap();
            pt_v[t * V_DIM..(t + 1) * V_DIM].copy_from_slice(&h);
            for x in [qo, ko, vo, row] {
                let _ = gpu.free_tensor(x);
            }
        }
        let pt_cs_end = gpu.download_f32(&pt_cs).unwrap();
        let _ = gpu.free_tensor(pt_cs);

        let mut b_cs = gpu.upload_f32(&conv_state0, &[QKV_DIM * 3]).unwrap();
        let cin = gpu.upload_f32(&conv_in, &[n * QKV_DIM]).unwrap();
        let bq = gpu.zeros(&[n * K_DIM], rdna_compute::DType::F32).unwrap();
        let bk = gpu.zeros(&[n * K_DIM], rdna_compute::DType::F32).unwrap();
        let bv = gpu.zeros(&[n * V_DIM], rdna_compute::DType::F32).unwrap();
        gpu.conv1d_silu_split_f32_n(&bq, &bk, &bv, &cin, &cw, &mut b_cs, K_DIM, V_DIM, n)
            .expect("batched conv");
        let bq_h = gpu.download_f32(&bq).unwrap();
        let bk_h = gpu.download_f32(&bk).unwrap();
        let bv_h = gpu.download_f32(&bv).unwrap();
        let b_cs_end = gpu.download_f32(&b_cs).unwrap();
        for x in [bq, bk, bv, cin, b_cs] {
            let _ = gpu.free_tensor(x);
        }

        // ══ 2. GDN carry ═════════════════════════════════════════════════
        let mut pt_gs = gpu.upload_f32(&s0, &[STATE_ELEMS]).unwrap();
        let mut pt_out = vec![0f32; n * V_DIM];
        for t in 0..n {
            let qg = gpu.upload_f32(&q[t * GDN_QK..][..GDN_QK], &[GDN_QK]).unwrap();
            let kg = gpu.upload_f32(&k[t * GDN_QK..][..GDN_QK], &[GDN_QK]).unwrap();
            let vg = gpu.upload_f32(&v[t * V_DIM..][..V_DIM], &[V_DIM]).unwrap();
            let gg = gpu
                .upload_f32(&gate[t * N_V_HEADS..][..N_V_HEADS], &[N_V_HEADS])
                .unwrap();
            let bg = gpu
                .upload_f32(&beta[t * N_V_HEADS..][..N_V_HEADS], &[N_V_HEADS])
                .unwrap();
            let og = gpu.zeros(&[V_DIM], rdna_compute::DType::F32).unwrap();
            gpu.gated_delta_net_f32(
                &qg, &kg, &vg, &gg, &bg, &mut pt_gs, &og, 1, N_V_HEADS, HEAD_DIM,
            )
            .expect("per-token gdn");
            let h = gpu.download_f32(&og).unwrap();
            pt_out[t * V_DIM..(t + 1) * V_DIM].copy_from_slice(&h);
            for x in [qg, kg, vg, gg, bg, og] {
                let _ = gpu.free_tensor(x);
            }
        }
        let pt_gs_end = gpu.download_f32(&pt_gs).unwrap();
        let _ = gpu.free_tensor(pt_gs);

        let mut b_gs = gpu.upload_f32(&s0, &[STATE_ELEMS]).unwrap();
        let qg = gpu.upload_f32(&q, &[n * GDN_QK]).unwrap();
        let kg = gpu.upload_f32(&k, &[n * GDN_QK]).unwrap();
        let vg = gpu.upload_f32(&v, &[n * V_DIM]).unwrap();
        let gg = gpu.upload_f32(&gate, &[n * N_V_HEADS]).unwrap();
        let bg = gpu.upload_f32(&beta, &[n * N_V_HEADS]).unwrap();
        let bog = gpu.zeros(&[n * V_DIM], rdna_compute::DType::F32).unwrap();
        gpu.gated_delta_net_f32_batch_seq(
            &qg, &kg, &vg, &gg, &bg, &mut b_gs, &bog, n, N_V_HEADS, HEAD_DIM,
        )
        .expect("batched gdn");
        let bog_h = gpu.download_f32(&bog).unwrap();
        let b_gs_end = gpu.download_f32(&b_gs).unwrap();
        for x in [qg, kg, vg, gg, bg, bog, b_gs] {
            let _ = gpu.free_tensor(x);
        }
        let _ = gpu.free_tensor(cw);

        // ══ report ═══════════════════════════════════════════════════════
        let conv_out_rel = rel(&bv_h, &pt_v).max(rel(&bq_h, &pt_q)).max(rel(&bk_h, &pt_k));
        let conv_state_rel = rel(&b_cs_end, &pt_cs_end);
        let gdn_out_rel = rel(&bog_h, &pt_out);
        let gdn_state_rel = rel(&b_gs_end, &pt_gs_end);
        for (i, x) in [conv_out_rel, conv_state_rel, gdn_out_rel, gdn_state_rel]
            .iter()
            .enumerate()
        {
            worst[i] = worst[i].max(*x);
        }
        // Row 0 has no carry to get wrong: it isolates arithmetic. Row n-1 has
        // the entire accumulated chain, so the pair says "arithmetic" vs "state".
        let g0 = rel(&bog_h[..V_DIM], &pt_out[..V_DIM]);
        let gn = rel(
            &bog_h[(n - 1) * V_DIM..][..V_DIM],
            &pt_out[(n - 1) * V_DIM..][..V_DIM],
        );
        println!(
            "n={n:3}   conv out {conv_out_rel:.2e}  conv state {conv_state_rel:.2e}   |   \
             gdn out {gdn_out_rel:.2e}  gdn state {gdn_state_rel:.2e}   (gdn row0 {g0:.2e} rowN {gn:.2e})"
        );
    }

    println!(
        "\nWORST: conv out {:.2e}  conv state {:.2e}  gdn out {:.2e}  gdn state {:.2e}",
        worst[0], worst[1], worst[2], worst[3]
    );
    println!(
        "  fp32 summation-order noise at these widths is ~1e-6..1e-5; at or above 1e-4 is a real arithmetic difference."
    );
    let labels = ["conv out", "conv state", "gdn out", "gdn state"];
    for (i, l) in labels.iter().enumerate() {
        if worst[i] > 1e-4 {
            println!("  DIVERGENT: {l}");
        }
    }
    let bad = worst.iter().any(|&x| x > 1e-4);
    println!("ORACLE: {}", if bad { "FAIL" } else { "PASS" });
    if bad {
        std::process::exit(1);
    }
}
