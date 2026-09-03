// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Escha-W2 (ESCHAM code-quant MoE) safetensors-directory loader.
//!
//! Escha-W2 leaves the *routed expert projections* in an int16 EXL3-trellis
//! code with fp16 `rin`/`rout` rotations and fp32 `s_in` scales; the dense
//! projections (attention, shared expert, embed, lm_head) are int8 with a
//! per-row fp16 `weight_scale`. This loader:
//!   - reads embed/lm_head int8 + per-row scales → F32 (row-major dequant),
//!   - loads the router + shared expert (dense, via `paro_load_wt` int8 path),
//!   - uploads each layer's escha codes/rin/rout and builds per-expert views,
//!   - combines `s_in · rin` into the FFN input-scale tensor.
//!
//! The trellis decode + WHT fold happens lazily on GPU at first use of each
//! expert (see `qwen35::escha_ffn::escham_moe_ffn_decode`); nothing is
//! dequantized to full fp16 at load.

use super::config::Qwen35Config;
use super::load::Layout;
use super::weights::EschaMoeFfnWeights;
use super::weights::LayerWeights;
use super::weights::Qwen35Weights;
use super::weights::SharedExpertWeights;
use hip_bridge::HipError;
use hip_bridge::HipResult;
use hipfire_runtime::llama::f16_to_f32;
use hipfire_runtime::llama::EmbeddingFormat;
use hipfire_runtime::llama::WeightTensor;
use hipfire_runtime::model_load::WeightSource;
use hipfire_runtime::model_source::ModelSource;
use hipfire_runtime::paro::load_fp16_weight_from_source;
use hipfire_runtime::paro::paro_load_wt;
use hipfire_runtime::safetensors_source::source_bytes_to_f32_vec;
use hipfire_runtime::weight_backend::ParoBackend;
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;

/// Load Qwen35 weights from a safetensors directory source, routing through
/// [`EschaSource`] when `quantization_config.quant_method == "eschamoe"` and
/// through [`ParoSource`] otherwise (the pre-existing ParoQuant dir path).
pub fn load_weights_from_safetensors(
    source: &dyn ModelSource,
    config: &Qwen35Config,
    gpu: &mut Gpu,
) -> Result<Qwen35Weights, String> {
    let devices: &mut [Gpu] = std::slice::from_mut(gpu);
    let layout = Layout::single(config.n_layers);
    if config.is_escham_moe {
        let mut escha = EschaSource::new(source, config).map_err(|e| e.to_string())?;
        crate::qwen35::load::load_weights(&mut escha, devices, &layout).map_err(|e| e.to_string())
    } else {
        let mut paro = super::load::ParoSource::new(source, config).map_err(|e| e.to_string())?;
        crate::qwen35::load::load_weights(&mut paro, devices, &layout).map_err(|e| e.to_string())
    }
}

/// WeightSource adapter for Escha code-quant MoE safetensors models. Holds the
/// source + text-tower prefix + config directly (mirrors `ParoSource`'s own
/// fields; those are private and only readable from `load.rs`).
pub struct EschaSource<'a> {
    source: &'a dyn ModelSource,
    mp: &'static str,
    c: &'a Qwen35Config,
    is_escham_moe: bool,
}

impl<'a> EschaSource<'a> {
    pub fn new(source: &'a dyn ModelSource, c: &'a Qwen35Config) -> HipResult<Self> {
        let mp = hipfire_runtime::paro::paro_text_prefix(source)?;
        Ok(Self {
            source,
            mp,
            c,
            is_escham_moe: c.is_escham_moe,
        })
    }

    /// Build a `ParoBackend` (mirrors `load.rs::qwen35_paro_backend`, which is
    /// private to that module). `QWEN35_NORM_BIAS` for qwen35 norms is 1.0.
    fn backend<'b>(&'b self, gpu: &'b mut Gpu, layer: usize) -> ParoBackend<'b> {
        ParoBackend {
            source: self.source,
            gpu,
            mp: self.mp,
            layer,
            norm_bias: 1.0,
        }
    }

    fn escha_load_moe_ffn(
        &self,
        gpu: &mut Gpu,
        layer_idx: usize,
        config: &Qwen35Config,
    ) -> HipResult<EschaMoeFfnWeights> {
        let mp = self.mp;
        let dim = config.dim;
        let n_exp = config.num_experts;
        let mi = config.moe_intermediate_size;
        let smi = config.shared_expert_intermediate_size;
        let p = format!("layers.{layer_idx}");

        // ── Router (dense f16) ──
        let router = load_fp16_weight_from_source(
            self.source,
            gpu,
            &format!("{mp}.{p}.mlp.gate.weight"),
            n_exp,
            dim,
        )?;

        // ── Shared-expert scalar gate (dense f16) ──
        let shared_expert_gate = load_fp16_weight_from_source(
            self.source,
            gpu,
            &format!("{mp}.{p}.mlp.shared_expert_gate.weight"),
            1,
            dim,
        )?;

        // ── Shared expert (int8 dense in this export; `paro_load_wt` dequant) ──
        let shared_expert = SharedExpertWeights {
            gate: paro_load_wt(
                self.source,
                gpu,
                &format!("{p}.mlp.shared_expert.gate_proj"),
                smi,
                dim,
                128,
                0,
            )?,
            up: paro_load_wt(
                self.source,
                gpu,
                &format!("{p}.mlp.shared_expert.up_proj"),
                smi,
                dim,
                128,
                0,
            )?,
            down: paro_load_wt(
                self.source,
                gpu,
                &format!("{p}.mlp.shared_expert.down_proj"),
                dim,
                smi,
                128,
                0,
            )?,
        };

        // ── Escha code-quant routed experts ──
        let (
            gate_up_codes,
            gate_up_rin,
            gate_up_rout,
            gate_up_rin_f32,
            gate_up_rout_f32,
            down_codes,
            down_rin,
            down_rout,
            down_rin_f32,
            down_rout_f32,
            gate_up_in_scale,
            down_in_scale,
        ) = self.load_escha_experts(gpu, layer_idx, config, mi, dim)?;

        Ok(EschaMoeFfnWeights {
            router,
            shared_expert_gate,
            shared_expert,
            gate_up_codes,
            down_codes,
            gate_up_rin,
            down_rin,
            gate_up_rout,
            down_rout,
            gate_up_in_scale,
            down_in_scale,
            gate_up_rin_f32_host: gate_up_rin_f32,
            down_rin_f32_host: down_rin_f32,
            gate_up_rout_f32_host: gate_up_rout_f32,
            down_rout_f32_host: down_rout_f32,
            gate_up_in: dim,
            down_in: mi,
            num_experts: n_exp,
            moe_intermediate_size: mi,
            folded_gate_up_cache: std::cell::RefCell::new((0..n_exp).map(|_| None).collect()),
            folded_down_cache: std::cell::RefCell::new((0..n_exp).map(|_| None).collect()),
        })
    }

    #[allow(clippy::type_complexity)]
    fn load_escha_experts(
        &self,
        gpu: &mut Gpu,
        layer_idx: usize,
        config: &Qwen35Config,
        mi: usize,
        dim: usize,
    ) -> HipResult<(
        Vec<GpuTensor>,
        Vec<GpuTensor>,
        Vec<GpuTensor>,
        Vec<f32>,
        Vec<f32>,
        Vec<GpuTensor>,
        Vec<GpuTensor>,
        Vec<GpuTensor>,
        Vec<f32>,
        Vec<f32>,
        GpuTensor,
        GpuTensor,
    )> {
        let mp = self.mp;
        let n_exp = config.num_experts;
        let p = format!("layers.{layer_idx}");
        let source = self.source;

        let gate_up_prefix = format!("{mp}.{p}.mlp.experts.gate_up_proj");
        let down_prefix = format!("{mp}.{p}.mlp.experts.down_proj");

        let gate_up_code_name = format!("{gate_up_prefix}.escha_code");
        let gate_up_rin_name = format!("{gate_up_prefix}.escha_rin");
        let gate_up_rout_name = format!("{gate_up_prefix}.escha_rout");
        let down_code_name = format!("{down_prefix}.escha_code");
        let down_rin_name = format!("{down_prefix}.escha_rin");
        let down_rout_name = format!("{down_prefix}.escha_rout");

        let (_, gate_up_code_data) = source.tensor_data(&gate_up_code_name).ok_or_else(|| {
            HipError::new(0, &format!("escha_code not found: {gate_up_code_name}"))
        })?;
        let gate_up_rin_info = source
            .tensor_info(&gate_up_rin_name)
            .ok_or_else(|| HipError::new(0, &format!("escha_rin not found: {gate_up_rin_name}")))?;
        let (_, gate_up_rin_data) = source.tensor_data(&gate_up_rin_name).unwrap();
        let gate_up_rout_info = source.tensor_info(&gate_up_rout_name).ok_or_else(|| {
            HipError::new(0, &format!("escha_rout not found: {gate_up_rout_name}"))
        })?;
        let (_, gate_up_rout_data) = source.tensor_data(&gate_up_rout_name).unwrap();

        let (_, down_code_data) = source
            .tensor_data(&down_code_name)
            .ok_or_else(|| HipError::new(0, &format!("escha_code not found: {down_code_name}")))?;
        let down_rin_info = source
            .tensor_info(&down_rin_name)
            .ok_or_else(|| HipError::new(0, &format!("escha_rin not found: {down_rin_name}")))?;
        let (_, down_rin_data) = source.tensor_data(&down_rin_name).unwrap();
        let down_rout_info = source
            .tensor_info(&down_rout_name)
            .ok_or_else(|| HipError::new(0, &format!("escha_rout not found: {down_rout_name}")))?;
        let (_, down_rout_data) = source.tensor_data(&down_rout_name).unwrap();

        // Per-expert activation scales (s_in), combined with rin into the FFN
        // input-scale tensor. Missing s_in → assume ones.
        let gate_up_sin_name = format!("{gate_up_prefix}.escha_s_in");
        let down_sin_name = format!("{down_prefix}.escha_s_in");
        let gate_up_sin_f32 = match source.tensor_data(&gate_up_sin_name) {
            Some((info, data)) => source_bytes_to_f32_vec(&info.dtype, data),
            None => {
                eprintln!("[Escha] WARNING: {gate_up_sin_name} missing; assuming ones");
                vec![1.0f32; n_exp * dim]
            }
        };
        let down_sin_f32 = match source.tensor_data(&down_sin_name) {
            Some((info, data)) => source_bytes_to_f32_vec(&info.dtype, data),
            None => {
                eprintln!("[Escha] WARNING: {down_sin_name} missing; assuming ones");
                vec![1.0f32; n_exp * mi]
            }
        };

        // Upload the per-projection code buffers (int16, stored as 2-byte
        // elements; typed F16 so `sub_offset` works bitwise).
        let gate_up_code_elems = gate_up_code_data.len() / 2;
        let gate_up_code_stride = gate_up_code_elems / n_exp;
        let gate_up_codes = {
            let buf = gpu.hip.malloc(gate_up_code_data.len())?;
            gpu.hip.memcpy_htod(&buf, gate_up_code_data)?;
            GpuTensor {
                buf,
                shape: vec![gate_up_code_elems],
                dtype: DType::F16, // 2 bytes/element — same bitwise as i16
            }
        };
        let down_code_elems = down_code_data.len() / 2;
        let down_code_stride = down_code_elems / n_exp;
        let down_codes = {
            let buf = gpu.hip.malloc(down_code_data.len())?;
            gpu.hip.memcpy_htod(&buf, down_code_data)?;
            GpuTensor {
                buf,
                shape: vec![down_code_elems],
                dtype: DType::F16,
            }
        };

        // f16 rin/rout → f32 host copies, then upload as per-projection f32.
        let gate_up_rin_f32 = source_bytes_to_f32_vec(&gate_up_rin_info.dtype, gate_up_rin_data);
        let gate_up_rout_f32 = source_bytes_to_f32_vec(&gate_up_rout_info.dtype, gate_up_rout_data);
        let down_rin_f32 = source_bytes_to_f32_vec(&down_rin_info.dtype, down_rin_data);
        let down_rout_f32 = source_bytes_to_f32_vec(&down_rout_info.dtype, down_rout_data);

        let gate_up_rin = gpu.upload_f32(&gate_up_rin_f32, &[n_exp, dim])?;
        let gate_up_rout = gpu.upload_f32(&gate_up_rout_f32, &[n_exp, mi * 2])?;
        let down_rin = gpu.upload_f32(&down_rin_f32, &[n_exp, mi])?;
        let down_rout = gpu.upload_f32(&down_rout_f32, &[n_exp, dim])?;

        // Per-expert views into the shared buffers.
        let gate_up_codes_views: Vec<GpuTensor> = (0..n_exp)
            .map(|e| gate_up_codes.sub_offset(e * gate_up_code_stride, gate_up_code_stride))
            .collect();
        let gate_up_rin_views: Vec<GpuTensor> = (0..n_exp)
            .map(|e| gate_up_rin.sub_offset(e * dim, dim))
            .collect();
        let gate_up_rout_views: Vec<GpuTensor> = (0..n_exp)
            .map(|e| gate_up_rout.sub_offset(e * mi * 2, mi * 2))
            .collect();
        let down_codes_views: Vec<GpuTensor> = (0..n_exp)
            .map(|e| down_codes.sub_offset(e * down_code_stride, down_code_stride))
            .collect();
        let down_rin_views: Vec<GpuTensor> = (0..n_exp)
            .map(|e| down_rin.sub_offset(e * mi, mi))
            .collect();
        let down_rout_views: Vec<GpuTensor> = (0..n_exp)
            .map(|e| down_rout.sub_offset(e * dim, dim))
            .collect();

        // Combined per-expert input scale = s_in · rin (F32), applied at the
        // GEMV input. s_in is NOT all-ones for this export, so it must apply.
        let gate_up_in_scale_f32: Vec<f32> = gate_up_rin_f32
            .iter()
            .zip(gate_up_sin_f32.iter())
            .map(|(&r, &s)| r * s)
            .collect();
        let down_in_scale_f32: Vec<f32> = down_rin_f32
            .iter()
            .zip(down_sin_f32.iter())
            .map(|(&r, &s)| r * s)
            .collect();
        let gate_up_in_scale = gpu.upload_f32(&gate_up_in_scale_f32, &[n_exp, dim])?;
        let down_in_scale = gpu.upload_f32(&down_in_scale_f32, &[n_exp, mi])?;

        Ok((
            gate_up_codes_views,
            gate_up_rin_views,
            gate_up_rout_views,
            gate_up_rin_f32,
            gate_up_rout_f32,
            down_codes_views,
            down_rin_views,
            down_rout_views,
            down_rin_f32,
            down_rout_f32,
            gate_up_in_scale,
            down_in_scale,
        ))
    }
}

impl WeightSource for EschaSource<'_> {
    type Layer = LayerWeights;

    fn n_layers(&self) -> usize {
        self.c.n_layers
    }

    fn prepare(&mut self, n_devices: usize) -> HipResult<()> {
        if n_devices > 1 {
            return Err(HipError::new(
                0,
                "Escha code-quant MoE multi-GPU loading is not supported (single-GPU only)",
            ));
        }
        Ok(())
    }

    fn read_embed(&mut self, gpu: &mut Gpu) -> HipResult<(GpuTensor, EmbeddingFormat)> {
        if self.is_escham_moe {
            // Escha embed_tokens is int8 + per-row f16 scale.
            let int8_name = format!("{}.embed_tokens.weight_int8", self.mp);
            let scale_name = format!("{}.embed_tokens.weight_scale", self.mp);
            if let (Some((_, int8_data)), Some((_, scale_data))) = (
                self.source.tensor_data(&int8_name),
                self.source.tensor_data(&scale_name),
            ) {
                let c = self.c;
                let scales: Vec<f32> = scale_data
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                let mut f32_data = vec![0.0f32; c.vocab_size * c.dim];
                for i in 0..(c.vocab_size * c.dim) {
                    let q = int8_data[i] as i8 as f32;
                    // Per-row scale: w[r][c] = int8[r][c] * scale[r].
                    f32_data[i] = q * scales[i / c.dim];
                }
                let weight = gpu.upload_f32(&f32_data, &[c.vocab_size, c.dim])?;
                return Ok((weight, EmbeddingFormat::F32));
            }
            // Fall back to the standard dense path (shouldn't happen for Escha).
        }
        let name = format!("{}.embed_tokens.weight", self.mp);
        let (_, data) = self
            .source
            .tensor_data(&name)
            .ok_or_else(|| HipError::new(0, &format!("embedding not found: {name}")))?;
        let f32_embd = hipfire_runtime::weight_backend::f16_bytes_to_f32(data);
        let token_embd = gpu.upload_f32(&f32_embd, &[self.c.vocab_size, self.c.dim])?;
        Ok((token_embd, EmbeddingFormat::F32))
    }

    fn read_final_norm(&mut self, gpu: &mut Gpu) -> HipResult<GpuTensor> {
        hipfire_runtime::paro::paro_load_norm(
            self.source,
            gpu,
            "norm.weight",
            &[self.c.dim],
            1.0,
        )
    }

    fn read_output(
        &mut self,
        gpu: &mut Gpu,
        embd: &GpuTensor,
        embd_fmt: EmbeddingFormat,
        can_alias: bool,
    ) -> HipResult<(WeightTensor, bool)> {
        if self.is_escham_moe {
            // Escha lm_head is int8 + per-row f16 scale.
            if let Some((_, int8_data)) = self.source.tensor_data("lm_head.weight_int8") {
                let (_, scale_data) = self
                    .source
                    .tensor_data("lm_head.weight_scale")
                    .ok_or_else(|| HipError::new(0, "lm_head.weight_scale not found"))?;
                let c = self.c;
                let scales: Vec<f32> = scale_data
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect();
                let mut f32_data = vec![0.0f32; c.vocab_size * c.dim];
                for i in 0..(c.vocab_size * c.dim) {
                    let q = int8_data[i] as i8 as f32;
                    f32_data[i] = q * scales[i / c.dim];
                }
                let weight = gpu.upload_f32(&f32_data, &[c.vocab_size, c.dim])?;
                let wt = WeightTensor {
                    buf: weight,
                    gpu_dtype: DType::F32,
                    m: c.vocab_size,
                    k: c.dim,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                };
                return Ok((wt, false));
            }
        }
        // Fallback: dense f16 lm_head (non-int8 export), mirroring the
        // ParoSource resolve_lm_head semantics.
        let c = self.c;
        let make_f16 = |data: &[u8]| -> Vec<f32> {
            hipfire_runtime::weight_backend::f16_bytes_to_f32(data)
        };
        if let Some((_, f16)) = self.source.tensor_data("lm_head.weight") {
            let f32 = make_f16(f16);
            let weight = gpu.upload_f32(&f32, &[c.vocab_size, c.dim])?;
            return Ok((
                WeightTensor {
                    buf: weight,
                    gpu_dtype: DType::F32,
                    m: c.vocab_size,
                    k: c.dim,
                    row_stride: 0,
                    paro: None,
                    awq_scale: None,
                },
                false,
            ));
        }
        if can_alias {
            // Tie to the (F32) embedding table already uploaded.
            let wt = WeightTensor {
                buf: embd.shallow_clone(),
                gpu_dtype: DType::F32,
                m: c.vocab_size,
                k: c.dim,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            };
            return Ok((wt, true));
        }
        // Copy the embed table as a separate lm_head.
        let name = format!("{}.embed_tokens.weight", self.mp);
        let (_, data) = self
            .source
            .tensor_data(&name)
            .ok_or_else(|| HipError::new(0, &format!("embedding not found: {name}")))?;
        let f32 = make_f16(data);
        let weight = gpu.upload_f32(&f32, &[c.vocab_size, c.dim])?;
        Ok((
            WeightTensor {
                buf: weight,
                gpu_dtype: DType::F32,
                m: c.vocab_size,
                k: c.dim,
                row_stride: 0,
                paro: None,
                awq_scale: None,
            },
            false,
        ))
    }

    fn read_layer(&mut self, gpu: &mut Gpu, layer_idx: usize) -> HipResult<LayerWeights> {
        let config = self.c;
        eprintln!(
            "  loading layer {layer_idx}/{} ({:?}, Escha)...",
            config.n_layers, config.layer_types[layer_idx]
        );
        let mut b = self.backend(gpu, layer_idx);
        let moe = |bk: &mut ParoBackend, cfg: &Qwen35Config, li: usize| {
            crate::paro_moe::paro_load_moe_ffn(
                bk.source,
                bk.gpu,
                &format!("layers.{li}"),
                cfg,
                li as u16,
            )
        };
        let moe_escha = |bk: &mut ParoBackend, cfg: &Qwen35Config, li: usize| {
            self.escha_load_moe_ffn(bk.gpu, li, cfg)
        };
        crate::layer_driver::load_layer(&mut b, config, layer_idx, moe, moe_escha)
    }
}
