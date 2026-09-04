// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Empirical pin of the analytic dense-decode funnel ordering against the
//! verified trellis reference on REAL model tiles (not random), testing every
//! plausible pairing/shift convention so the in-kernel arithmetic is exact.

use hipfire_arch_qwen35::escham_decode::{decode_3inst, decode_tiles};

fn codebook(idx: u32) -> f32 {
    decode_3inst(&[idx as u16])[0]
}

fn pi(r: usize) -> i32 {
    ((r & 1) | (((r >> 3) & 1) << 1) | (((r >> 1) & 3) << 3)) as i32
}

/// Try to decode one tile with a given convention. Returns per-(r,c) f32.
/// `hi`/`lo` select the funnel words: pass a fn mapping (words, w0) -> (hi, lo).
fn analytic_tile<F: Fn(&[u32], usize) -> (u32, u32)>(
    code: &[i16],
    k: usize,
    pair: F,
) -> Vec<f32> {
    let nw = 8 * k;
    let nb = 32 * nw;
    let n_wd = (16 * k) / 2;
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
            let (hi, lo) = pair(&words, w0);
            let idx = ((((hi as u64) << 32) | lo as u64) >> (sp & 31)) as u32 & 0xffff;
            w[r * 16 + cc] = codebook(idx);
        }
    }
    w
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let layer: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let proj = args.get(2).cloned().unwrap_or_else(|| "in_proj_qkv".to_string());
    let (family, k) = if proj == "up_proj" || proj == "down_proj" {
        ("mlp", 3usize)
    } else {
        ("mlp", 2usize)
    };
    let _ = family;
    // read code tensor from model for the projection at layer 0 (first 16*K*... keep one tile)
    use std::fs;
    let idx: serde_json::Value = serde_json::from_str(
        &fs::read_to_string("/data/rocmfpx/Qwen3.8-27B-Escha-W2/model.safetensors.index.json").unwrap(),
    )
    .unwrap();
    let wm = idx["weight_map"].as_object().unwrap();
    // find the right family for the requested name
    let base = format!("model.language_model.layers.{layer}");
    let key = if proj.starts_with("in_proj") {
        format!("{base}.linear_attn.{proj}.escha_code")
    } else if proj.starts_with("q_") || proj.starts_with("k_") || proj.starts_with("v_") || proj.starts_with("o_") {
        format!("{base}.self_attn.{proj}.escha_code")
    } else {
        format!("{base}.mlp.{proj}.escha_code")
    };
    eprintln!("reading {key}");
    let shard = wm[key.as_str()].as_str().unwrap();
    let bytes = fs::read(format!("/data/rocmfpx/Qwen3.8-27B-Escha-W2/{shard}")).unwrap();
    // parse header
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data_start = 8 + n + ((8 - (8 + n) % 8) % 8);
    let m = &hdr[&key];
    let shape: Vec<usize> = m["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let k_shape = shape[2] / 16;
    let (b, e) = (
        m["data_offsets"][0].as_u64().unwrap() as usize,
        m["data_offsets"][1].as_u64().unwrap() as usize,
    );
    let raw = &bytes[data_start + b..data_start + e];
    let code_all: Vec<i16> = raw
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    eprintln!("code shape {shape:?}, K={k_shape}, reading tile (bi=3,bj=5)");
    // tile (bi,bj): code[(bi*bj_max+bj)*(16K)..]
    let (ti, tj) = (shape[0], shape[1]);
    let bi = 3usize % ti;
    let bj = 5usize % tj;
    let tile: Vec<i16> = code_all[(bi * tj + bj) * 16 * k_shape..(bi * tj + bj + 1) * 16 * k_shape].to_vec();

    // reference: decode_tiles over a [16,16]-sized single tile
    let reference = decode_tiles(&tile, k_shape, 16, 16);
    let ref_idx = |r: usize, c: usize| reference[r * 16 + c];

    // conventions: (hi-word, lo-word) relative to w0
    let conventions: Vec<(&str, fn(&[u32], usize) -> (u32, u32))> = vec![
        ("(w0, w0-1) llama-literal", |words, w0| {
            (words[(w0 + 7) % 8], words[w0]) // placeholder replaced below
        }),
    ];
    let _ = conventions;

    let nw = 8 * k_shape;
    let convs: Vec<(String, Box<dyn Fn(&[u32], usize) -> (u32, u32)>)> = {
        let mut v: Vec<(String, Box<dyn Fn(&[u32], usize) -> (u32, u32)>)> = Vec::new();
        // normalize wrap with nw variable
        let wrap = nw;
        v.push(("A (w0, w0+1)".into(), Box::new(move |words, w0| (words[w0 % wrap], words[(w0 + 1) % wrap]))));
        let wrap2 = nw;
        v.push(("B (w0-1, w0)".into(), Box::new(move |words, w0| (words[(w0 + wrap2 - 1) % wrap2], words[w0 % wrap2]))));
        let wrap3 = nw;
        v.push(("C (w0, w0-1) llama-literal".into(), Box::new(move |words, w0| (words[w0 % wrap3], words[(w0 + wrap3 - 1) % wrap3]))));
        v
    };

    for (name, pair) in &convs {
        let got = analytic_tile(&tile, k_shape, pair);
        let mut n_match = 0usize;
        let mut worst = 0.0f32;
        let mut nm = 0usize;
        for r in 0..16 {
            for c in 0..16 {
                if got[r * 16 + c] == ref_idx(r, c) {
                    n_match += 1;
                }
                let d = (got[r * 16 + c] - ref_idx(r, c)).abs();
                if d > worst {
                    worst = d;
                }
                if got[r * 16 + c] != ref_idx(r, c) {
                    nm += 1;
                }
            }
        }
        eprintln!("{name}: exact matches {n_match}/256, wrong {nm}, worst {worst:.4}");
    }
}
