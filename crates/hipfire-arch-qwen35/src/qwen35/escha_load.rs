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
use super::weights::DeltaNetEschaLayerWeights;
use super::weights::EschaDenseProjWeights;
use super::weights::EschaMoeFfnWeights;
use super::weights::FullAttnEschaLayerWeights;
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
use hipfire_runtime::paro::paro_load_norm;
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
    if config.is_escham_moe || config.is_escha_dense {
        let mut escha = EschaSource::new(source, config).map_err(|e| e.to_string())?;
        crate::qwen35::load::load_weights(&mut escha, devices, &layout).map_err(|e| e.to_string())
    } else {
        let mut paro = super::load::ParoSource::new(source, config).map_err(|e| e.to_string())?;
        crate::qwen35::load::load_weights(&mut paro, devices, &layout).map_err(|e| e.to_string())
    }
}

/// WeightSource adapter for Escha code-quant safetensors models: both the
/// ESCHAM MoE export (`quant_method == "eschamoe"`, Escha-W2) and the ESCHA
/// dense export (`quant_method == "escha"`, Qwen3.8-27B-Escha-W2). Holds the
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

    // ── Escha code-quant DENSE layer loading (quant_method == "escha") ──
    //
    // Qwen3.8-27B-Escha-W2 keeps every linear projection (attention + FFN) in
    // the int16 trellis code. Per projection the export ships:
    //   escha_code   I16 [in/16, out/16, 16*K]   (K=2 q/k/v/o/qkv/z/gate; K=3 up/down)
    //   escha_rin    F16 [in]
    //   escha_rout   F16 [out]
    //   escha_s_in   F32 [in]      (≈1, applied: MUST fold for exactness)
    //   escha_s_out  F32 [out]     (≈1)
    //   escha_config I32 [6] = [16, K, 2, 1, in, out]
    //   bias         F16 [out]     (bias-correction; NOT applied — ESCHA_APPLY_BIAS 0)
    // Small dense tensors (in_proj_a/b, A_log, conv1d, norms) stay raw F16 and
    // use the paro raw/norm loaders; embed/lm_head are int8 + per-row f16 scale.

    /// Upload one Escha-coded dense projection from `{mp}.{p}.{rel}.escha_*`.
    fn escha_load_dense_proj(
        &self,
        gpu: &mut Gpu,
        p: &str,
        rel: &str,
        expected_in: usize,
        expected_out: usize,
    ) -> HipResult<EschaDenseProjWeights> {
        let base = format!("{}.{p}.{rel}", self.mp);
        let code_name = format!("{base}.escha_code");
        let rin_name = format!("{base}.escha_rin");
        let rout_name = format!("{base}.escha_rout");
        let sin_name = format!("{base}.escha_s_in");
        let sout_name = format!("{base}.escha_s_out");
        let cfg_name = format!("{base}.escha_config");

        let (code_info, code_data) = self
            .source
            .tensor_data(&code_name)
            .ok_or_else(|| HipError::new(0, &format!("escha_code not found: {code_name}")))?;
        let code_shape = code_info.shape.clone();
        if code_shape.len() != 3 {
            return Err(HipError::new(
                0,
                &format!("{code_name}: expected [in/16, out/16, 16*K], got {code_shape:?}"),
            ));
        }
        let (ti, tout, _last) = (code_shape[0] as usize, code_shape[1] as usize, code_shape[2]);
        let in_p = ti * 16;
        let out_p = tout * 16;
        let k = (code_shape[2] / 16) as u8;
        if in_p != expected_in || out_p != expected_out {
            return Err(HipError::new(
                0,
                &format!(
                    "{code_name}: shape [{in_p}->{out_p}] != expected [{expected_in}->{expected_out}]"
                ),
            ));
        }
        // escha_config sanity: [16, K, 2, 1, in, out] (stored I32).
        if let Some((cfg_info, cfg_data)) = self.source.tensor_data(&cfg_name) {
            let cfg: Vec<i32> = match cfg_info.dtype.as_str() {
                "I32" => cfg_data
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                "F32" => cfg_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i32)
                    .collect(),
                other => {
                    return Err(HipError::new(
                        0,
                        &format!("{cfg_name}: unsupported escha_config dtype {other}"),
                    ))
                }
            };
            if cfg.len() >= 6 {
                let (tile, ck, _v, _e, cin, cout) =
                    (cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[5]);
                if tile != 16 || ck as u8 != k || cin as usize != expected_in
                    || cout as usize != expected_out
                {
                    return Err(HipError::new(
                        0,
                        &format!(
                            "{code_name}: escha_config [{tile},{ck},..,{cin},{cout}] inconsistent \
                             with code shape (K={k}, {in_p}x{out_p})"
                        ),
                    ));
                }
            }
        }
        // in_scale = rin (f16) . s_in (f32), out_scale = rout . s_out.
        let (rin_info, rin_data) = self.source.tensor_data(&rin_name).ok_or_else(|| {
            HipError::new(0, &format!("escha_rin not found: {rin_name}"))
        })?;
        let (rout_info, rout_data) = self.source.tensor_data(&rout_name).ok_or_else(|| {
            HipError::new(0, &format!("escha_rout not found: {rout_name}"))
        })?;
        let rin: Vec<f32> = source_bytes_to_f32_vec(&rin_info.dtype, rin_data);
        let rout: Vec<f32> = source_bytes_to_f32_vec(&rout_info.dtype, rout_data);
        if rin.len() != in_p || rout.len() != out_p {
            return Err(HipError::new(
                0,
                &format!(
                    "{code_name}: rin {} rout {} vs in_p {in_p} out_p {out_p}",
                    rin.len(),
                    rout.len()
                ),
            ));
        }
        let sin: Vec<f32> = match self.source.tensor_data(&sin_name) {
            Some((si, sd)) => source_bytes_to_f32_vec(&si.dtype, sd),
            None => vec![1.0f32; in_p],
        };
        let sout: Vec<f32> = match self.source.tensor_data(&sout_name) {
            Some((so, sod)) => source_bytes_to_f32_vec(&so.dtype, sod),
            None => vec![1.0f32; out_p],
        };
        if sin.len() != in_p || sout.len() != out_p {
            return Err(HipError::new(
                0,
                &format!(
                    "{code_name}: s_in {} s_out {} vs in_p {in_p} out_p {out_p}",
                    sin.len(),
                    sout.len()
                ),
            ));
        }
        let in_scale_f32: Vec<f32> = rin.iter().zip(sin.iter()).map(|(&r, &s)| r * s).collect();
        let out_scale_f32: Vec<f32> = rout.iter().zip(sout.iter()).map(|(&r, &s)| r * s).collect();

        // Code upload (bitwise F16 = i16 2-byte elements), typed F16 like the MoE path.
        let code_elems = code_data.len() / 2;
        let buf = gpu.hip.malloc(code_data.len())?;
        gpu.hip.memcpy_htod(&buf, code_data)?;
        let code = GpuTensor {
            buf,
            shape: vec![in_p / 16, out_p / 16, k as usize * 16],
            dtype: DType::F16,
        };
        let in_scale = gpu.upload_f32(&in_scale_f32, &[in_p])?;
        let out_scale = gpu.upload_f32(&out_scale_f32, &[out_p])?;
        Ok(EschaDenseProjWeights {
            code,
            in_scale,
            out_scale,
            in_p,
            out_p,
            k,
        })
    }

    /// Load one dense (non-MoE) Qwen3.5 Escha layer. Handles both DeltaNet
    /// (linear attention) and FullAttention layers by `config.layer_types`.
    pub fn escha_load_dense_layer(
        &self,
        gpu: &mut Gpu,
        layer_idx: usize,
        config: &Qwen35Config,
    ) -> HipResult<LayerWeights> {
        let p = format!("layers.{layer_idx}");
        let dim = config.dim;
        let hidden = config.hidden_dim;
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = k_dim * 2 + v_dim;
        let q_out = config.n_heads * config.head_dim * 2; // q + gate fused
        let kv_dim = config.n_kv_heads * config.head_dim;
        let o_in = config.n_heads * config.head_dim;

        // Small dense tensors via paro loaders (mp-prefixed, plain F16).
        // Closures take `gpu` as an argument so they do not hold a long-lived
        // mutable borrow across the whole layer body.
        let load_norm = |rel: &str, gpu: &mut Gpu| {
            paro_load_norm(self.source, gpu, &format!("{p}.{rel}"), &[dim], 1.0)
        };
        let load_f16_wt = |rel: &str, m: usize, k: usize, gpu: &mut Gpu| {
            let base = format!("{}.{p}.{rel}", self.mp);
            load_fp16_weight_from_source(self.source, gpu, &format!("{base}.weight"), m, k)
        };
        let load_f16_vec = |rel: &str, n: usize, gpu: &mut Gpu| {
            // paro_load_f32 reads {mp}.{name} and handles F16/BF16/F32.
            hipfire_runtime::paro::paro_load_f32(self.source, gpu, &format!("{p}.{rel}"), n)
        };

        let attn_norm = load_norm("input_layernorm.weight", gpu)?;
        let ffn_norm = load_norm("post_attention_layernorm.weight", gpu)?;

        let ffn = |this: &Self,
                   gpu: &mut Gpu|
         -> HipResult<(EschaDenseProjWeights, EschaDenseProjWeights, EschaDenseProjWeights)> {
            Ok((
                this.escha_load_dense_proj(gpu, &p, "mlp.gate_proj", dim, hidden)?,
                this.escha_load_dense_proj(gpu, &p, "mlp.up_proj", dim, hidden)?,
                this.escha_load_dense_proj(gpu, &p, "mlp.down_proj", hidden, dim)?,
            ))
        };

        match config.layer_types[layer_idx] {
            super::config::LayerType::LinearAttention => {
                let (w_gate, w_up, w_down) = ffn(self, gpu)?;
                Ok(LayerWeights::DeltaNetEscha(DeltaNetEschaLayerWeights {
                    attn_norm,
                    qkv: self.escha_load_dense_proj(gpu, &p, "linear_attn.in_proj_qkv", dim, qkv_dim)?,
                    z: self.escha_load_dense_proj(gpu, &p, "linear_attn.in_proj_z", dim, v_dim)?,
                    w_alpha: load_f16_wt(
                        "linear_attn.in_proj_a",
                        config.linear_num_value_heads,
                        dim,
                        gpu,
                    )?,
                    w_beta: load_f16_wt(
                        "linear_attn.in_proj_b",
                        config.linear_num_value_heads,
                        dim,
                        gpu,
                    )?,
                    a_log: load_f16_vec("linear_attn.A_log", config.linear_num_value_heads, gpu)?,
                    dt_bias: load_f16_vec("linear_attn.dt_bias", config.linear_num_value_heads, gpu)?,
                    conv_weight: load_f16_vec(
                        "linear_attn.conv1d.weight",
                        qkv_dim * config.conv_kernel_dim,
                        gpu,
                    )?,
                    norm_weight: load_f16_vec(
                        "linear_attn.norm.weight",
                        config.linear_value_head_dim,
                        gpu,
                    )?,
                    wo: self.escha_load_dense_proj(gpu, &p, "linear_attn.out_proj", v_dim, dim)?,
                    ffn_norm,
                    w_gate,
                    w_up,
                    w_down,
                }))
            }
            super::config::LayerType::FullAttention => {
                let (w_gate, w_up, w_down) = ffn(self, gpu)?;
                Ok(LayerWeights::FullAttnEscha(FullAttnEschaLayerWeights {
                    attn_norm,
                    wq: self.escha_load_dense_proj(gpu, &p, "self_attn.q_proj", dim, q_out)?,
                    wk: self.escha_load_dense_proj(gpu, &p, "self_attn.k_proj", dim, kv_dim)?,
                    wv: self.escha_load_dense_proj(gpu, &p, "self_attn.v_proj", dim, kv_dim)?,
                    wo: self.escha_load_dense_proj(gpu, &p, "self_attn.o_proj", o_in, dim)?,
                    q_norm: paro_load_norm(
                        self.source,
                        gpu,
                        &format!("{p}.self_attn.q_norm.weight"),
                        &[config.head_dim],
                        1.0,
                    )?,
                    k_norm: paro_load_norm(
                        self.source,
                        gpu,
                        &format!("{p}.self_attn.k_norm.weight"),
                        &[config.head_dim],
                        1.0,
                    )?,
                    ffn_norm,
                    w_gate,
                    w_up,
                    w_down,
                }))
            }
        }
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
        if self.is_escham_moe || self.c.is_escha_dense {
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
        if self.is_escham_moe || self.c.is_escha_dense {
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
        if config.is_escha_dense {
            eprintln!(
                "  loading layer {layer_idx}/{} ({:?}, Escha-dense)...",
                config.n_layers, config.layer_types[layer_idx]
            );
            return self.escha_load_dense_layer(gpu, layer_idx, config);
        }
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
