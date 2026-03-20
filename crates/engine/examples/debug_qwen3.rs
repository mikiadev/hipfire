//! Debug Qwen3-0.6B: step through forward pass, print intermediate values.

use engine::gguf::GgufFile;
use engine::llama::{self, LlamaConfig, KvCache};
use std::path::Path;

fn stats(name: &str, v: &[f32]) {
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
    let var: f32 = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
    let nan_count = v.iter().filter(|x| x.is_nan()).count();
    let inf_count = v.iter().filter(|x| x.is_infinite()).count();
    eprintln!("  {name} [{} elems]: min={min:.6} max={max:.6} mean={mean:.6} std={:.6} nan={nan_count} inf={inf_count}",
        v.len(), var.sqrt());
}

fn main() {
    let path = "/home/kaden/llama.cpp/models/Qwen3-0.6B-Q8_0.gguf";
    let gguf = GgufFile::open(Path::new(path)).unwrap();
    let config = LlamaConfig::from_gguf(&gguf).unwrap();

    eprintln!("Config: dim={} hidden={} layers={} heads={} kv_heads={} head_dim={} vocab={} qk_norm={} rope_base={}",
        config.dim, config.hidden_dim, config.n_layers, config.n_heads,
        config.n_kv_heads, config.head_dim, config.vocab_size, config.has_qk_norm, config.rope_freq_base);
    eprintln!("  q_dim = {} (n_heads * head_dim)", config.n_heads * config.head_dim);
    eprintln!("  kv_dim = {} (n_kv_heads * head_dim)", config.n_kv_heads * config.head_dim);

    let mut gpu = rdna_compute::Gpu::init().unwrap();
    eprintln!("\nLoading weights...");
    let weights = llama::load_weights(&gguf, &config, &gpu).unwrap();

    // Check Q8_0 dequant by examining some weight values
    let wq_data = gpu.download_f32(&weights.layers[0].wq).unwrap();
    stats("layer0.wq", &wq_data);

    let wk_data = gpu.download_f32(&weights.layers[0].wk).unwrap();
    stats("layer0.wk", &wk_data);

    // Check QK norm weights
    if let Some(ref qn) = weights.layers[0].q_norm {
        let qn_data = gpu.download_f32(qn).unwrap();
        stats("layer0.q_norm", &qn_data);
    }
    if let Some(ref kn) = weights.layers[0].k_norm {
        let kn_data = gpu.download_f32(kn).unwrap();
        stats("layer0.k_norm", &kn_data);
    }

    // Run forward pass for BOS token
    let kv_dim = config.n_kv_heads * config.head_dim;
    let mut kv_cache = KvCache::new_gpu(&gpu, config.n_layers, config.n_kv_heads, config.head_dim, config.max_seq_len).unwrap();

    // Token 151644 = <|im_start|>
    let token = 151644u32;
    eprintln!("\n=== Forward pass: token={token} (im_start), pos=0 ===");
    let logits = llama::forward(&mut gpu, &weights, &config, token, 0, &mut kv_cache).unwrap();
    stats("logits", &logits);

    let top5: Vec<(usize, f32)> = {
        let mut indexed: Vec<_> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.into_iter().take(5).collect()
    };
    eprintln!("  top5 tokens: {:?}", top5);
    let next = llama::argmax(&logits);
    eprintln!("  argmax: {next}");

    // Second token
    eprintln!("\n=== Forward pass: token={next}, pos=1 ===");
    let logits2 = llama::forward(&mut gpu, &weights, &config, next, 1, &mut kv_cache).unwrap();
    stats("logits2", &logits2);
    let top5b: Vec<(usize, f32)> = {
        let mut indexed: Vec<_> = logits2.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.into_iter().take(5).collect()
    };
    eprintln!("  top5 tokens: {:?}", top5b);
    let next2 = llama::argmax(&logits2);
    eprintln!("  argmax: {next2}");
}
