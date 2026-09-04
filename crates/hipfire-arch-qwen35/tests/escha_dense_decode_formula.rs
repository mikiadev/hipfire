// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pin the analytic (r,c) → codebook-index decode used by the Escha DENSE
//! kernels against the already-verified EXL3 trellis + tensor-core-perm
//! decode (`escham_decode::decode_tiles`).
//!
//! llama.cpp's dense kernels never read a dep table: the 16 bits forming the
//! codebook index of weight (r,c) are 16 cyclically-consecutive positions of
//! the tile's circular payload, with start bit
//!
//!   pi(r) = (r&1) | (((r>>3)&1)<<1) | (((r>>1)&3)<<3)
//!   t     = pi(r) + 32*cc + 4*(cc>>3)
//!   s     = ((32-K) - K*t) mod (256K)
//!
//! and the window is read with the payload words stored as OVERLAPPING pairs:
//! pay[w].y = word w, pay[w].x = word (w+1 mod NW); weight(r,c) =
//! codebook(funnelshift_r(hi=pay[w0].y, lo=pay[w0].x, s&31) & 0xffff) where
//! w0 = (NW - (s>>5)) mod NW.
//!
//! This test asserts the analytic formula reproduces `decode_tiles` (the
//! trellis reference verified against higgs escha_ref.py) exactly on
//! pseudo-random K=2 and K=3 tiles, and pins WHICH funnel pairing convention
//! matches (case A: pair (word w0, word w0+1); case B: (word w0-1, word w0)).

use hipfire_arch_qwen35::escham_decode::{decode_3inst, decode_tiles};

fn pi(r: usize) -> i32 {
    ((r & 1) | (((r >> 3) & 1) << 1) | (((r >> 1) & 3) << 3)) as i32
}

fn codebook(idx: u32) -> f32 {
    decode_3inst(&[idx as u16])[0]
}

/// Analytic decode of one 16x16 tile -> [16][16] f32. `swap` selects the
/// funnel pairing convention: false => (hi=word w0, lo=word w0+1), true =>
/// (hi=word w0-1, lo=word w0). Replicates the dense kernel arithmetic.
fn analytic_tile(code: &[i16], k: usize, swap: bool) -> Vec<f32> {
    let n_wd = (16 * k) / 2; // u32 words per tile
    let nw = 8 * k;
    let nb = 32 * nw;
    let words: Vec<u32> = (0..n_wd)
        .map(|i| {
            let lo = code[i * 2] as u16 as u32;
            let hi = code[i * 2 + 1] as u16 as u32;
            lo | (hi << 16)
        })
        .collect();
    let mut w = vec![0.0f32; 256];
    for r in 0..16usize {
        for cc in 0..16usize {
            let s = ((32 - k) as i32
                - (k as i32) * (pi(r) + 32 * (cc as i32) + 4 * ((cc as i32) >> 3)))
                % nb as i32;
            let sp = if s < 0 { s + nb as i32 } else { s } as usize;
            let g0 = sp >> 5;
            let w0 = if g0 != 0 { (nw - g0) % nw } else { 0 };
            let sh = sp & 31;
            let (hi, lo) = if swap {
                (words[(w0 + nw - 1) % nw], words[w0])
            } else {
                (words[w0 % nw], words[(w0 + 1) % nw])
            };
            let idx = ((((hi as u64) << 32) | lo as u64) >> sh) as u32 & 0xffff;
            w[r * 16 + cc] = codebook(idx);
        }
    }
    w
}

fn count_mismatches(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

fn run_case(k: usize, seed: u32) {
    let mut x = seed.wrapping_mul(2654435761).wrapping_add(0x9e3779b9);
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x
    };
    let code: Vec<i16> = (0..16 * k).map(|_| (next() >> 16) as u16 as i16).collect();

    let reference = decode_tiles(&code, k, 16, 16);
    let a = analytic_tile(&code, k, false);
    let b = analytic_tile(&code, k, true);
    let (ma, mb) = (count_mismatches(&a, &reference), count_mismatches(&b, &reference));
    eprintln!("K={k} seed={seed}: case A mismatches={ma}, case B mismatches={mb}");
    assert!(
        ma == 0 || mb == 0,
        "K={k} seed={seed}: neither funnel pairing matches the trellis reference (A={ma}, B={mb})"
    );
}

#[test]
fn analytic_dense_decode_matches_trellis() {
    for k in [2usize, 3] {
        for seed in [1u32, 7, 99, 12345] {
            run_case(k, seed);
        }
    }
}
