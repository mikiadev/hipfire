// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Escha-W2 code-quant DENSE layer forward bodies (AR per-token path).
//!
//! These functions implement the per-token decode for one DeltaNetEscha or
//! FullAttnEscha layer. They mirror the plain `forward_scratch_layers` arm
//! bodies (qkvza/qkv → recurrence or attention → wo residual → FFN) but every
//! coded projection runs through [`super::escha_dense_decode::escha_dense_decode_proj`]
//! instead of a WeightTensor gemv. The projection outputs land in the scratch
//! buffers the attention kernels already consume (`s.dn_qkv`, `s.dn_z`,
//! `s.fa_q_full`, `s.fa_k`, `s.fa_v`, `s.gate_ffn`, `s.up`, `s.ffn_hidden`,
//! `s.ffn_out`, `s.o`).

use hip_bridge::HipError;
use hip_bridge::HipResult;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;

use super::config::Qwen35Config;
use super::escha_dense_decode::escha_dense_decode_proj;
use super::forward::Qwen35Scratch;
use super::weights::DeltaNetEschaLayerWeights;
use super::weights::FullAttnEschaLayerWeights;

/// Decode one projection whose output is `[out_p]` into `dst` (caller scratch
/// sized `>= out_p`), allocating only the transient decode temporaries.
#[allow(clippy::too_many_arguments)]
fn decode_into(
    gpu: &mut Gpu,
    proj: &super::weights::EschaDenseProjWeights,
    x: &GpuTensor,
    dst: &GpuTensor,
) -> HipResult<()> {
    escha_dense_decode_proj(gpu, proj, x, dst)
}

/// DeltaNet (linear-attention) Escha dense layer forward.
#[allow(clippy::too_many_arguments)]
pub fn deltanet_escha_layer_forward(
    gpu: &mut Gpu,
    layer: &DeltaNetEschaLayerWeights,
    config: &Qwen35Config,
    pos: usize,
    delta_layer_idx: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    dn_state: &mut super::weights::DeltaNetState,
    s: &Qwen35Scratch,
) -> HipResult<()> {
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    // ── attention input norm + coded projections ──
    // normed x → s.tmp (decode input is the raw normed activation).
    gpu.rmsnorm_f32(&s.x, &layer.attn_norm, &s.tmp, config.norm_eps)?;
    stats(gpu, "post rmsnorm (decode input)", &s.tmp);
    decode_into(gpu, &layer.qkv, &s.tmp, &s.dn_qkv)?;
    decode_into(gpu, &layer.z, &s.tmp, &s.dn_z)?;
    stats(gpu, "post qkv/z decode", &s.dn_qkv);
    stats(gpu, "post z decode", &s.dn_z);
    // beta / alpha stay dense f16 gemvs of the normed input.
    {
        use hipfire_dispatch::context::DispatchCtx;
        use hipfire_dispatch::families::gemv::WeightRef;
        use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
        let ctx = DispatchCtx::new(gpu);
        let wr_beta = WeightRef {
            buf: &layer.w_beta.buf,
            dtype: layer.w_beta.gpu_dtype,
            m: layer.w_beta.m,
            k: layer.w_beta.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        let wr_alpha = WeightRef {
            buf: &layer.w_alpha.buf,
            dtype: layer.w_alpha.gpu_dtype,
            m: layer.w_alpha.m,
            k: layer.w_alpha.k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        };
        execute_steps(
            gpu,
            &ctx,
            &[
                Step::Gemv {
                    w: &wr_beta,
                    input: GemvInput::Raw(&s.tmp),
                    out: &s.dn_beta,
                },
                Step::Gemv {
                    w: &wr_alpha,
                    input: GemvInput::Raw(&s.tmp),
                    out: &s.dn_alpha,
                },
            ],
        )
        .map_err(|e| HipError::new(0, &e.to_string()))?;
    }

    gpu.fused_sigmoid_alpha_gate_f32(
        &s.dn_beta,
        &s.dn_alpha,
        &layer.dt_bias,
        &layer.a_log,
        n_v_heads,
    )?;
    gpu.conv1d_silu_split_f32(
        &s.dn_q_raw,
        &s.dn_k_raw,
        &s.dn_v,
        &s.dn_qkv,
        &layer.conv_weight,
        &dn_state.conv_states[delta_layer_idx],
        k_dim,
        v_dim,
    )?;
    gpu.fused_qk_l2_norm_scale_f32(
        &s.dn_q_raw,
        &s.dn_k_raw,
        config.linear_num_key_heads,
        hd,
        1.0 / (hd as f32).sqrt(),
        config.norm_eps,
    )?;
    if config.linear_num_key_heads < n_v_heads {
        let ratio = n_v_heads / config.linear_num_key_heads;
        gpu.repeat_interleave_qk_f32(
            &s.dn_q_raw,
            &s.dn_k_raw,
            &s.dn_q,
            &s.dn_k,
            config.linear_num_key_heads,
            ratio,
            hd,
        )?;
    } else {
        gpu.memcpy_dtod_auto(&s.dn_q.buf, &s.dn_q_raw.buf, k_dim * 4)?;
        gpu.memcpy_dtod_auto(&s.dn_k.buf, &s.dn_k_raw.buf, k_dim * 4)?;
    }
    match dn_state.quant {
        super::weights::StateQuant::FP32 => gpu.gated_delta_net_f32(
            &s.dn_q,
            &s.dn_k,
            &s.dn_v,
            &s.dn_alpha,
            &s.dn_beta,
            &dn_state.s_matrices[delta_layer_idx],
            &s.dn_attn_out,
            1,
            n_v_heads,
            config.linear_value_head_dim,
        )?,
        super::weights::StateQuant::Q8 => gpu.gated_delta_net_q8(
            &s.dn_q,
            &s.dn_k,
            &s.dn_v,
            &s.dn_alpha,
            &s.dn_beta,
            &dn_state.s_matrices[delta_layer_idx],
            &dn_state.s_scales[delta_layer_idx],
            &s.dn_attn_out,
            1,
            n_v_heads,
            config.linear_value_head_dim,
            dn_state.ef_residual(delta_layer_idx),
        )?,
        super::weights::StateQuant::Q4 => gpu.gated_delta_net_q4(
            &s.dn_q,
            &s.dn_k,
            &s.dn_v,
            &s.dn_alpha,
            &s.dn_beta,
            &dn_state.s_matrices[delta_layer_idx],
            &dn_state.s_scales[delta_layer_idx],
            &s.dn_attn_out,
            1,
            n_v_heads,
            config.linear_value_head_dim,
        )?,
    }
    gpu.gated_norm_f32(
        &s.dn_attn_out,
        &s.dn_z,
        &layer.norm_weight,
        &s.dn_normed,
        n_v_heads,
        config.linear_value_head_dim,
        config.norm_eps,
    )?;

    // ── wo coded projection + residual ──
    decode_into(gpu, &layer.wo, &s.dn_normed, &s.o)?;
    stats(gpu, "post wo decode", &s.o);
    gpu.add_f32(&s.x, &s.o, &s.x)?;
    stats(gpu, "post LA residual", &s.x);

    // ── FFN (gate/up/down coded) ──
    if hipfire_config::developer_var("HIPFIRE_ESCHA_DENSE_NO_FFN").ok().as_deref() == Some("1") {
        // B1 debug: skip the FFN contribution entirely.
        return Ok(());
    }
    gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
    decode_into(gpu, &layer.w_gate, &s.tmp, &s.gate_ffn)?;
    decode_into(gpu, &layer.w_up, &s.tmp, &s.up)?;
    gpu.silu_mul_f32(&s.gate_ffn, &s.up, &s.ffn_hidden)?;
    decode_into(gpu, &layer.w_down, &s.ffn_hidden, &s.o)?;
    stats(gpu, "post down decode", &s.o);
    gpu.add_f32(&s.x, &s.o, &s.x)?;
    stats(gpu, "post FFN residual", &s.x);
    let _ = pos;
    Ok(())
}

/// Full-attention Escha dense layer forward.
#[allow(clippy::too_many_arguments)]
pub fn fullattn_escha_layer_forward(
    gpu: &mut Gpu,
    layer: &FullAttnEschaLayerWeights,
    config: &Qwen35Config,
    pos: usize,
    layer_idx: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    s: &Qwen35Scratch,
) -> HipResult<()> {
    use hipfire_dispatch::context::DispatchCtx;
    // ── q/k/v coded projections from the normed input ──
    gpu.rmsnorm_f32(&s.x, &layer.attn_norm, &s.tmp, config.norm_eps)?;
    decode_into(gpu, &layer.wq, &s.tmp, &s.fa_q_full)?;
    decode_into(gpu, &layer.wk, &s.tmp, &s.fa_k)?;
    decode_into(gpu, &layer.wv, &s.tmp, &s.fa_v)?;

    gpu.deinterleave_f32(
        &s.fa_q_full,
        &s.fa_q,
        &s.fa_gate,
        config.n_heads,
        config.head_dim,
    )?;
    gpu.rmsnorm_batched(
        &s.fa_q,
        &layer.q_norm,
        &s.fa_q,
        config.n_heads,
        config.head_dim,
        config.norm_eps,
    )?;
    gpu.rmsnorm_batched(
        &s.fa_k,
        &layer.k_norm,
        &s.fa_k,
        config.n_kv_heads,
        config.head_dim,
        config.norm_eps,
    )?;

    let ctx = DispatchCtx::new(gpu);
    let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
    if kv_cache.compact_offset > 0 {
        let abs = (pos + kv_cache.compact_offset) as i32;
        gpu.memcpy_htod_auto(&s.pos_buf, &abs.to_ne_bytes())?;
    }
    gpu.rope_partial_interleaved_f32(
        &s.fa_q,
        &s.fa_k,
        &s.pos_buf,
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        n_rot,
        config.rope_theta,
    )?;
    if kv_cache.compact_offset > 0 {
        let phys = pos as i32;
        gpu.memcpy_htod_auto(&s.pos_buf, &phys.to_ne_bytes())?;
    }

    // Escha-dense FA runs UNFUSED (coded wo is not a WeightTensor), so the
    // attention kernel always leaves fa_attn_out pre-gate; apply sigmoid(gate).
    // NOTE: kv_cache is indexed by the MODEL layer index (0..63) — every
    // filtered constructor keeps one Vec entry per model layer with real
    // buffers only on full-attention layers.
    let fused_epilogue = super::forward::kv_cache_attention_dispatch(
        &ctx, gpu, kv_cache, s, config, None, layer_idx, pos,
    )?;
    debug_assert!(!fused_epilogue, "escha-dense FA must be unfused");
    gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;

    stats(gpu, "post attend out", &s.fa_attn_out);

    // ── wo coded projection + residual ──
    decode_into(gpu, &layer.wo, &s.fa_attn_out, &s.o)?;
    stats(gpu, "post wo decode (FA)", &s.o);
    gpu.add_f32(&s.x, &s.o, &s.x)?;
    stats(gpu, "post FA residual", &s.x);

    // ── FFN ──
    gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
    decode_into(gpu, &layer.w_gate, &s.tmp, &s.gate_ffn)?;
    decode_into(gpu, &layer.w_up, &s.tmp, &s.up)?;
    gpu.silu_mul_f32(&s.gate_ffn, &s.up, &s.ffn_hidden)?;
    decode_into(gpu, &layer.w_down, &s.ffn_hidden, &s.o)?;
    gpu.add_f32(&s.x, &s.o, &s.x)?;
    stats(gpu, "post FFN residual (FA)", &s.x);
    Ok(())
}

/// Debug stats (HIPFIRE_ESCHA_DENSE_TRACE=1 prints post-op ranges per layer).
fn stats(gpu: &Gpu, label: &str, t: &GpuTensor) {
    if hipfire_config::developer_var_os("HIPFIRE_ESCHA_DENSE_TRACE").is_none() {
        return;
    }
    if let Ok(vals) = gpu.download_f32(t) {
        let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
        let mut nn = 0usize;
        let mut ninf = 0usize;
        let mut n = 0usize;
        for &v in &vals {
            if v.is_nan() { nn += 1; }
            else if v.is_infinite() { ninf += 1; }
            else { n += 1; mn = mn.min(v); mx = mx.max(v); }
        }
        eprintln!("[escha-dense] {label}: n={n} nan={nn} inf={ninf} range=[{mn:.4e},{mx:.4e}] first3={:?}", &vals[..3.min(vals.len())]);
    }
}
