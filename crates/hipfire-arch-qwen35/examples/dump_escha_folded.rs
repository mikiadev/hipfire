//! Decode an Escha-W2 expert with the CORRECT trellis+3INST scheme and dump
//! the folded weight for validation against the higgs reference.
use std::fs;

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut e = 1u32;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e += 1;
            }
            f &= 0x3FF;
            f32::from_bits((sign << 31) | ((127 - e + 15) << 23) | (f << 13))
        }
    } else if exp == 0x1F {
        if frac == 0 {
            f32::from_bits((sign << 31) | 0x7F80_0000)
        } else {
            f32::from_bits((sign << 31) | 0x7FC0_0000)
        }
    } else {
        f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let layer: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let proj = args.get(2).cloned().unwrap_or_else(|| "gate_up".to_string());
    let expert: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let out = args.get(4).cloned().unwrap_or_else(|| "/tmp/rust_folded.bin".to_string());

    let idx: serde_json::Value =
        serde_json::from_str(&fs::read_to_string("/data/rocmfpx/Escha-W2/model.safetensors.index.json").unwrap()).unwrap();
    let wm = idx["weight_map"].as_object().unwrap();

    let (prefix, k, in_f, out_f) = if proj == "gate_up" {
        (format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj"), 2usize, 2048usize, 1024usize)
    } else {
        (format!("model.language_model.layers.{layer}.mlp.experts.down_proj"), 3usize, 512usize, 2048usize)
    };

    // code: [E, in/16, out/16, 16*K] int16, expert along axis 0
    let code_key = format!("{prefix}.escha_code");
    let code_shard = wm[&code_key].as_str().unwrap();
    let code_bytes = fs::read(format!("/data/rocmfpx/Escha-W2/{code_shard}")).unwrap();
    let code = hipfire_arch_qwen35::escham_decode::read_safetensors_expert_i16(&code_bytes, &code_key, expert).unwrap();

    // rin / rout fp16 per expert
    let rin = load_f16(wm, &format!("{prefix}.escha_rin"), expert);
    let rout = load_f16(wm, &format!("{prefix}.escha_rout"), expert);

    let folded = hipfire_arch_qwen35::escham_decode::decode_and_fold_weight(
        &code, k, in_f, out_f, &rin, &rout,
    );
    let bytes: Vec<u8> = folded.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(&out, &bytes).unwrap();
    eprintln!(
        "dumped {out}: layer {layer} {proj} expert {expert} K={k} folded len={} mean_abs={:.6}",
        folded.len(),
        folded.iter().map(|v| v.abs()).sum::<f32>() / folded.len() as f32
    );
}

fn load_f16(wm: &serde_json::Map<String, serde_json::Value>, key: &str, expert: usize) -> Vec<f32> {
    let shard = wm[key].as_str().unwrap();
    let bytes = fs::read(format!("/data/rocmfpx/Escha-W2/{shard}")).unwrap();
    hipfire_arch_qwen35::escham_decode::read_safetensors_expert_f16(&bytes, key, expert).unwrap()
}
