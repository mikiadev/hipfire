# hipfire

LLM inference engine for AMD RDNA GPUs. Written from scratch in Rust + HIP. **8.5× faster than llama.cpp** on Qwen3.5 DeltaNet models.

## Quickstart

```bash
# Install (Linux, requires AMD GPU)
curl -L https://raw.githubusercontent.com/autorocm/hipfire/alpha-builds/scripts/install.sh | sh

# Or build from source
cargo build --release --features deltanet --example daemon --example infer --example infer_hfq -p engine

# Run
hipfire run models/qwen3.5-4b.q4.hfq "What is the capital of France?"
# Or directly:
./target/release/examples/infer models/qwen3.5-4b.q4.hfq "What is the capital of France?"
```

## Performance (RX 5700 XT)

| Model | tok/s | Notes |
|-------|-------|-------|
| Qwen3-8B HFQ4 | **59.9** | Standard attention, GPU sampling |
| Qwen3.5-0.8B HFQ4 | **209** | DeltaNet, tiled LDS GDN |
| Qwen3.5-4B HFQ4 | **62.5** | DeltaNet + thinking mode |
| Qwen3.5-9B HFQ4 | **44.4** | DeltaNet + VL + thinking |
| ollama Qwen3.5-9B | 4.93 | llama.cpp + ROCm (same hardware) |

## Features

- **Qwen3.5 DeltaNet**: Gated linear attention with tiled LDS kernel (32 VGPRs, 20 waves)
- **Vision-Language (VL)**: GPU vision encoder (1.3-4.6s), `--image` flag for image+text
- **TurboQuant KV**: Asymmetric q8-K + turbo4-V with 256-dim FWHT, 5.1× compression
- **Thinking mode**: `<think>` reasoning with n-gram loop prevention
- **Pre-compiled kernels**: Ship .hsaco blobs, no ROCm SDK needed at runtime
- **4 GPU arches**: gfx1010 (5700 XT), gfx1030 (6800 XT), gfx1100 (7900 XTX), gfx1200 (9070)
- **Zero VRAM leak**: Explicit GPU free + pool drain for model eviction
- **OpenAI-compatible API**: `hipfire serve` → `/v1/chat/completions` with SSE streaming
- **Agent diagnostic skill**: `.skills/hipfire-diag/` for automated GPU troubleshooting

## Supported Models

| Family | Sizes | Arch | Format |
|--------|-------|------|--------|
| Qwen3 | 0.6B, 8B | LLaMA attention | HFQ4 |
| Qwen3.5 | 0.8B, 2B, 4B, 9B | DeltaNet hybrid | HFQ4 |
| Qwen3.5-VL | 0.8B, 4B, 9B | DeltaNet + ViT | HFQ4 + F16 vision |

## CLI

```bash
hipfire serve [port]           # OpenAI-compatible HTTP server
hipfire run <model> [prompt]   # Interactive generation
hipfire run <model> --image img.png [prompt]  # Vision-language
hipfire list                   # Show local models
```

## API

```bash
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-4b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

## Architecture

```
Bun CLI (hipfire serve/run)
  └→ Rust daemon (JSON lines IPC)
       └→ GPU kernels (pre-compiled .hsaco, 94 kernels per arch)
            ├→ HFQ4 GEMV (18 VGPRs, max occupancy)
            ├→ Tiled LDS GDN (32 VGPRs, warp shuffle)
            ├→ Asymmetric turbo KV (q8-K + turbo4-V, 256-dim FWHT)
            └→ Vision encoder (GEMM, LayerNorm, ViT attention)
```

## License

MIT
