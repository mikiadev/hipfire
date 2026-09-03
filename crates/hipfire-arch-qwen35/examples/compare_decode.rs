//! Standalone: decode one expert on GPU (trellis kernel) and compare against the
//! Rust host decode, dumping both for debugging.
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let layer: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let proj = args.get(2).cloned().unwrap_or_else(|| "gate_up".to_string());
    let expert: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);

    let Some(mut gpu) = rdna_compute::Gpu::init().ok() else {
        eprintln!("no gpu"); return;
    };
    let idx: serde_json::Value =
        serde_json::from_str(&fs::read_to_string("/data/rocmfpx/Escha-W2/model.safetensors.index.json").unwrap()).unwrap();
    let wm = idx["weight_map"].as_object().unwrap();

    let (prefix, k, in_f, out_f) = if proj == "gate_up" {
        (format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj"), 2usize, 2048usize, 1024usize)
    } else {
        (format!("model.language_model.layers.{layer}.mlp.experts.down_proj"), 3usize, 512usize, 2048usize)
    };
    let code_key = format!("{prefix}.escha_code");
    let shard = wm[&code_key].as_str().unwrap();
    let bytes = fs::read(format!("/data/rocmfpx/Escha-W2/{shard}")).unwrap();
    let code = hipfire_arch_qwen35::escham_decode::read_safetensors_expert_i16(&bytes, &code_key, expert).unwrap();
    let rin = load_f16(wm, &format!("{prefix}.escha_rin"), expert);
    let rout = load_f16(wm, &format!("{prefix}.escha_rout"), expert);

    let in_p = ((in_f + 127) / 128) * 128;
    let out_p = ((out_f + 127) / 128) * 128;

    // Host reference (correct, WITH rout)
    let host_folded = hipfire_arch_qwen35::escham_decode::decode_and_fold_weight(&code, k, in_f, out_f, &rin, &rout);
    // Host fold WITHOUT rout (pure WHT): H_out @ M @ H_in, then apply rout post-hoc
    let host_bare2 = hipfire_arch_qwen35::escham_decode::decode_tiles(&code, k, in_p, out_p);
    let mut host_m2 = vec![0.0f32; out_p * in_p];
    for i in 0..in_p { for j in 0..out_p { host_m2[j * in_p + i] = host_bare2[i * out_p + j]; } }
    let host_wh = hipfire_arch_qwen35::escham_decode::had128_matrix(&host_m2, out_p, in_p);

    // GPU: decode trellis + fold (rout=ones)
    let codes_gpu = gpu.upload_raw(unsafe { std::slice::from_raw_parts(code.as_ptr() as *const u8, code.len() * 2) }, &[code.len()]).unwrap();
    let dec = gpu.zeros(&[out_p, in_p], rdna_compute::DType::F32).unwrap();
    let mid = gpu.zeros(&[out_p, in_p], rdna_compute::DType::F32).unwrap();
    let folded = gpu.zeros(&[out_p, in_p], rdna_compute::DType::F32).unwrap();
    rdna_compute::escham::escham_moe_decode_trellis(&mut gpu, &dec, &codes_gpu, k as i32, in_p as i32, out_p as i32).unwrap();
    let dec_host = gpu.download_f32(&dec).unwrap();
    let ones = vec![1.0f32; out_p];
    let rout_ones = gpu.upload_f32(&ones, &[out_p]).unwrap();
    rdna_compute::escham::escham_fold_t128_rows(&mut gpu, &mid, &dec, &rout_ones, out_p, in_p).unwrap();
    rdna_compute::escham::escham_fold_t128_cols(&mut gpu, &folded, &mid, out_p, in_p).unwrap();
    let gpu_folded = gpu.download_f32(&folded).unwrap();

    let mut max_abs = 0.0f32;
    let mut scale = 0.0f32;
    // gpu_folded = H_out @ M @ H_in (rout=ones); host_wh = same. Compare directly.
    for i in 0..gpu_folded.len() {
        let d = (gpu_folded[i] - host_wh[i]).abs();
        max_abs = max_abs.max(d);
        scale = scale.max(host_wh[i].abs());
    }
    eprintln!("layer {layer} {proj} e{expert}: GPU-fold(no-rout) vs host-WHT max_abs={max_abs:.6} rel={:.6}",
        max_abs / (scale + 1e-9));

    // Also dump the raw decode (pre-fold) comparison: host bare weight folded manually
    let host_bare = hipfire_arch_qwen35::escham_decode::decode_tiles(&code, k, in_p, out_p);
    let mut host_m = vec![0.0f32; out_p * in_p];
    for i in 0..in_p { for j in 0..out_p { host_m[j * in_p + i] = host_bare[i * out_p + j]; } }
    let mut max_dec = 0.0f32; let mut dec_scale = 0.0f32;
    for i in 0..host_m.len() {
        let d = (dec_host[i] - host_m[i]).abs();
        max_dec = max_dec.max(d); dec_scale = dec_scale.max(host_m[i].abs());
    }
    eprintln!("  raw decode (pre-fold) max_abs={max_dec:.6} rel={:.6}", max_dec / (dec_scale + 1e-9));
    eprintln!("  dec_host[0..6] = {:?}", &dec_host[..6]);
    eprintln!("  host_m[0..6]   = {:?}", &host_m[..6]);
    // per-tile mismatch stats: which (bi, bj) tiles are wrong
    let mut wrong_tiles = 0;
    let mut total_tiles = 0;
    for bi in 0..in_p/16 {
        for bj in 0..out_p/16 {
            total_tiles += 1;
            let mut tile_max = 0.0f32;
            for r in 0..16 { for c in 0..16 {
                let o = (bj*16+c)*in_p + (bi*16+r);
                let d = (dec_host[o] - host_m[o]).abs();
                tile_max = tile_max.max(d);
            } }
            if tile_max > 1e-3 { wrong_tiles += 1; if wrong_tiles <= 4 { eprintln!("  tile bi={bi} bj={bj}: max_abs={tile_max:.4}"); } }
        }
    }
    eprintln!("  wrong tiles: {wrong_tiles}/{total_tiles}");
    // dump tile (0,0) 16x16: gpu vs host, mark diffs
    eprintln!("  tile (0,0) gpu vs host (diff>1e-3 = X):");
    for r in 0..16 {
        let mut line = String::new();
        for c in 0..16 {
            let o = (0*16+c)*in_p + (0*16+r);
            let d = (dec_host[o] - host_m[o]).abs();
            line.push(if d > 1e-3 { 'X' } else { '.' });
        }
        eprintln!("    row {r:2}: {line}");
    }
}

fn load_f16(wm: &serde_json::Map<String, serde_json::Value>, key: &str, expert: usize) -> Vec<f32> {
    let shard = wm[key].as_str().unwrap();
    let bytes = fs::read(format!("/data/rocmfpx/Escha-W2/{shard}")).unwrap();
    hipfire_arch_qwen35::escham_decode::read_safetensors_expert_f16(&bytes, key, expert).unwrap()
}
