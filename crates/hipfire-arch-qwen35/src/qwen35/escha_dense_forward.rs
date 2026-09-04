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
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;

use super::batch::BatchSemantics;
use super::batch::PrefillBatchScratch;
use super::config::Qwen35Config;
use super::escha_dense_decode::escha_dense_decode_proj;
use super::forward::Qwen35Scratch;
use super::weights::DeltaNetEschaLayerWeights;
use super::weights::FullAttnEschaLayerWeights;

/// Decode one projection whose output is `[out_p]` into `dst` (caller scratch
/// sized `>= out_p`), allocating only the transient decode temporaries.
#[allow(clippy::too_many_arguments)]
/// Entry probe: print input hidden range for sparse layers (env-gated).
fn entry_probe(gpu: &Gpu, layer_idx: usize, s: &Qwen35Scratch, kind: &str) {
    if hipfire_config::developer_var_os("HIPFIRE_ESCHA_DENSE_TRACE").is_none() {
        return;
    }
    if hipfire_config::developer_var_os("HIPFIRE_ESCHA_DENSE_TRACE_ALL").is_none() {
        if layer_idx % 16 != 0 && layer_idx != 63 {
            return;
        }
    }
    if let Ok(v) = gpu.download_f32(&s.x) {
        let (mut mn, mut mx, mut rms) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
        for &x in &v {
            mn = mn.min(x);
            mx = mx.max(x);
            rms += (x as f64) * (x as f64) / v.len() as f64;
        }
        eprintln!("[escha-dense] {kind} L{layer_idx} x: range=[{mn:.3e},{mx:.3e}] rms={:.3e}", rms.sqrt());
    }
}

fn decode_into(
    gpu: &mut Gpu,
    proj: &super::weights::EschaDenseProjWeights,
    x: &GpuTensor,
    dst: &GpuTensor,
) -> HipResult<()> {
    escha_dense_decode_proj(gpu, proj, x, dst)
}

/// Live-decode audit (HIPFIRE_ESCHA_DENSE_AUDIT=1): compare the GPU in-kernel
/// decode of `proj` against the host reference on THIS token's REAL input `x`.
/// The static oracle uses random x; this checks real structured activations at
/// every decode position. Reports max_abs/rel to stderr. Audits at every layer
/// divisible by 16 unless HIPFIRE_ESCHA_DENSE_AUDIT_LAYER overrides to one layer.
#[allow(clippy::too_many_arguments)]
fn audit_decode(
    gpu: &Gpu,
    proj: &super::weights::EschaDenseProjWeights,
    x: &GpuTensor,
    got: &GpuTensor,
    layer_idx: usize,
    pos: usize,
    label: &str,
) {
    if hipfire_config::developer_var_os("HIPFIRE_ESCHA_DENSE_AUDIT").is_none() {
        return;
    }
    let layer_filter = hipfire_config::developer_var("HIPFIRE_ESCHA_DENSE_AUDIT_LAYER")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    if !layer_filter.map(|lf| lf == layer_idx).unwrap_or(layer_idx % 16 == 0) {
        return;
    }
    let pos_filter = hipfire_config::developer_var("HIPFIRE_ESCHA_DENSE_AUDIT_POS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    if let Some(pf) = pos_filter {
        if pf != pos {
            return;
        }
    }
    use super::escha_dense_decode::escha_dense_decode_proj_host;
    let (x_h, in_scale, out_scale) = match (
        gpu.download_f32(x),
        gpu.download_f32(&proj.in_scale),
        gpu.download_f32(&proj.out_scale),
    ) {
        (Ok(x), Ok(i), Ok(o)) => (x, i, o),
        _ => return,
    };
    let n_bytes = proj.code.buf.size();
    let mut code_bytes = vec![0u8; n_bytes];
    if gpu.hip.memcpy_dtoh(&mut code_bytes, &proj.code.buf).is_err() {
        return;
    }
    let code_i16: Vec<i16> = code_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let got_v = gpu.download_f32(got).unwrap_or_default();
    let want = escha_dense_decode_proj_host(
        &code_i16,
        proj.k as usize,
        proj.in_p,
        proj.out_p,
        &in_scale,
        &out_scale,
        &x_h,
    );
    if got_v.len() != want.len() || got_v.is_empty() {
        return;
    }
    let mut max_abs = 0.0f32;
    let mut scale = 0.0f32;
    for i in 0..got_v.len() {
        max_abs = max_abs.max((got_v[i] - want[i]).abs());
        scale = scale.max(want[i].abs());
    }
    eprintln!(
        "[escha-dense-audit] L{layer_idx} pos {pos} {label} live-decode: max_abs={max_abs:.5} rel={:.5}",
        max_abs / (scale + 1e-9)
    );
}

/// DeltaNet (linear-attention) Escha dense layer forward.
#[allow(clippy::too_many_arguments)]
pub fn deltanet_escha_layer_forward(
    gpu: &mut Gpu,
    layer: &DeltaNetEschaLayerWeights,
    config: &Qwen35Config,
    pos: usize,
    layer_idx: usize,
    delta_layer_idx: usize,
    kv_cache: &mut hipfire_runtime::llama::KvCache,
    dn_state: &mut super::weights::DeltaNetState,
    s: &Qwen35Scratch,
) -> HipResult<()> {
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;

    entry_probe(gpu, layer_idx, s, "LA");

    // ── attention input norm + coded projections ──
    // normed x → s.tmp (decode input is the raw normed activation).
    gpu.rmsnorm_f32(&s.x, &layer.attn_norm, &s.tmp, config.norm_eps)?;
    stats(gpu, "post rmsnorm (decode input)", &s.tmp);
    decode_into(gpu, &layer.qkv, &s.tmp, &s.dn_qkv)?;
    decode_into(gpu, &layer.z, &s.tmp, &s.dn_z)?;
    audit_decode(gpu, &layer.qkv, &s.tmp, &s.dn_qkv, layer_idx, pos, "qkv");
    audit_decode(gpu, &layer.z, &s.tmp, &s.dn_z, layer_idx, pos, "z");
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
    small_probe(gpu, "alpha", &s.dn_alpha);
    small_probe(gpu, "beta", &s.dn_beta);
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
    qkv_probe(gpu, "dn_q_raw", &s.dn_q_raw);
    qkv_probe(gpu, "dn_v", &s.dn_v);
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
        super::weights::StateQuant::FP32 => {
            let r = gpu.gated_delta_net_f32(
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
            );
            state_probe(gpu, dn_state, delta_layer_idx, "post-gdn-f32");
            r?
        }
        super::weights::StateQuant::Q8 => {
            let r = gpu.gated_delta_net_q8(
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
            );
            state_probe(gpu, dn_state, delta_layer_idx, "post-gdn-q8");
            r?
        }
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
    stats(gpu, "post gated-norm", &s.dn_normed);

    // ── wo coded projection + residual ──
    decode_into(gpu, &layer.wo, &s.dn_normed, &s.o)?;
    audit_decode(gpu, &layer.wo, &s.dn_normed, &s.o, layer_idx, pos, "wo");
    stats(gpu, "post wo decode", &s.o);
    stats(gpu, "pre-add x (LA)", &s.x);
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
    audit_decode(gpu, &layer.w_gate, &s.tmp, &s.gate_ffn, layer_idx, pos, "gate");
    audit_decode(gpu, &layer.w_up, &s.tmp, &s.up, layer_idx, pos, "up");
    gpu.silu_mul_f32(&s.gate_ffn, &s.up, &s.ffn_hidden)?;
    decode_into(gpu, &layer.w_down, &s.ffn_hidden, &s.o)?;
    audit_decode(gpu, &layer.w_down, &s.ffn_hidden, &s.o, layer_idx, pos, "down");
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
    entry_probe(gpu, layer_idx, s, "FA");
    if hipfire_config::developer_var("HIPFIRE_ESCHA_DENSE_NO_ATTN").ok().as_deref() == Some("1") {
        // B1 debug: full-attention passthrough (only FFN acts).
        gpu.rmsnorm_f32(&s.x, &layer.ffn_norm, &s.tmp, config.norm_eps)?;
        decode_into(gpu, &layer.w_gate, &s.tmp, &s.gate_ffn)?;
        decode_into(gpu, &layer.w_up, &s.tmp, &s.up)?;
        gpu.silu_mul_f32(&s.gate_ffn, &s.up, &s.ffn_hidden)?;
        decode_into(gpu, &layer.w_down, &s.ffn_hidden, &s.o)?;
        gpu.add_f32(&s.x, &s.o, &s.x)?;
        return Ok(());
    }
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
    if hipfire_config::developer_var("HIPFIRE_ESCHA_DENSE_NO_FA_GATE").ok().as_deref() != Some("1") {
        gpu.sigmoid_mul_f32(&s.fa_attn_out, &s.fa_gate)?;
    }

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

/// Batched prefill for one Escha-coded dense projection.
/// `x_batch` is [N, IC], `y_batch` is [N, OC].
/// Uses the prefill kernel (R=64 rows per block, decoded weights to shared).
pub fn escha_dense_decode_proj_batch(
    gpu: &mut Gpu,
    proj: &super::weights::EschaDenseProjWeights,
    x_batch: &GpuTensor,
    y_batch: &GpuTensor,
    n_rows: usize,
) -> HipResult<()> {
    let ic = proj.in_p;
    let oc = proj.out_p;
    let k = proj.k as i32;
    let nit = ic / 16;
    let r = if n_rows <= 1 { 1 } else { 64 };
    let n_slices = rdna_compute::escha_dense::escha_dense_n_slices_prefill(nit, oc, n_rows);
    debug_assert!(n_slices >= 1);

    // Rotate: u = T128(x . in_scale)
    let u = gpu.alloc_tensor(&[n_rows * ic], DType::F32)?;
    rdna_compute::escha_dense::escha_dense_rotate_in_dense(
        gpu, &proj.in_scale, x_batch, &u, n_rows,
    )?;

    // Matmul: partial = u @ decode(code)
    let partial = gpu.alloc_tensor(&[n_slices * n_rows * oc], DType::F32)?;
    rdna_compute::escha_dense::escha_dense_matmul_prefill(
        gpu, &proj.code, &u, &partial, n_rows, n_slices, k, r,
    )?;

    // Finalize: y = T128_col(sum_slices) . out_scale
    rdna_compute::escha_dense::escha_dense_finalize_dense(
        gpu, &proj.out_scale, &partial, y_batch, n_rows, n_slices,
    )?;

    let _ = gpu.free_tensor(u);
    let _ = gpu.free_tensor(partial);
    Ok(())
}

/// Batched prefill forward for one DeltaNetEscha layer.
/// Mirrors `deltanet_escha_layer_forward` but uses batched prefill projections.
#[allow(clippy::too_many_arguments)]
pub fn deltanet_escha_layer_prefill(
    gpu: &mut Gpu,
    layer: &DeltaNetEschaLayerWeights,
    config: &Qwen35Config,
    n_rows: usize,
    delta_layer_idx: usize,
    dn_state: &mut super::weights::DeltaNetState,
    pbs: &super::batch::PrefillBatchScratch,
) -> HipResult<()> {
    let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
    let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let n_v_heads = config.linear_num_value_heads;
    let hd = config.linear_key_head_dim;
    let qkv_dim = k_dim * 2 + v_dim;

    // ── attention input norm + coded projections ──
    gpu.rmsnorm_batched(&pbs.x_batch, &layer.attn_norm, &pbs.x_rot_batch, n_rows, config.dim, config.norm_eps)?;
    escha_dense_decode_proj_batch(gpu, &layer.qkv, &pbs.x_rot_batch, &pbs.dn_qkv_batch, n_rows)?;
    escha_dense_decode_proj_batch(gpu, &layer.z, &pbs.x_rot_batch, &pbs.dn_z_batch, n_rows)?;

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
                    input: GemvInput::Raw(&pbs.x_rot_batch),
                    out: &pbs.dn_beta_batch,
                },
                Step::Gemv {
                    w: &wr_alpha,
                    input: GemvInput::Raw(&pbs.x_rot_batch),
                    out: &pbs.dn_alpha_batch,
                },
            ],
        )
        .map_err(|e| HipError::new(0, &e.to_string()))?;
    }

    gpu.fused_sigmoid_alpha_gate_f32_batched(
        &pbs.dn_beta_batch,
        &pbs.dn_alpha_batch,
        &layer.dt_bias,
        &layer.a_log,
        n_v_heads,
        n_rows,
    )?;

    gpu.conv1d_silu_split_f32_n(
        &pbs.dn_q_raw_batch,
        &pbs.dn_k_raw_batch,
        &pbs.dn_v_batch,
        &pbs.dn_qkv_batch,
        &layer.conv_weight,
        &dn_state.conv_states[delta_layer_idx],
        k_dim,
        v_dim,
        n_rows,
    )?;
    gpu.fused_qk_l2_norm_scale_f32_batched(
        &pbs.dn_q_raw_batch,
        &pbs.dn_k_raw_batch,
        config.linear_num_key_heads,
        hd,
        1.0 / (hd as f32).sqrt(),
        config.norm_eps,
        n_rows,
    )?;
    if config.linear_num_key_heads < n_v_heads {
        let ratio = n_v_heads / config.linear_num_key_heads;
        gpu.repeat_interleave_qk_f32_batched(
            &pbs.dn_q_raw_batch,
            &pbs.dn_k_raw_batch,
            &pbs.dn_q_batch,
            &pbs.dn_k_batch,
            config.linear_num_key_heads,
            ratio,
            hd,
            n_rows,
        )?;
    } else {
        gpu.memcpy_dtod_auto(&pbs.dn_q_batch.buf, &pbs.dn_q_raw_batch.buf, n_rows * k_dim * 4)?;
        gpu.memcpy_dtod_auto(&pbs.dn_k_batch.buf, &pbs.dn_k_raw_batch.buf, n_rows * k_dim * 4)?;
    }
    match dn_state.quant {
        super::weights::StateQuant::FP32 => {
            gpu.gated_delta_net_f32_batch_seq(
                &pbs.dn_q_batch,
                &pbs.dn_k_batch,
                &pbs.dn_v_batch,
                &pbs.dn_alpha_batch,
                &pbs.dn_beta_batch,
                &dn_state.s_matrices[delta_layer_idx],
                &pbs.dn_attn_out_batch,
                n_rows,
                n_v_heads,
                config.linear_value_head_dim,
            )?;
        }
        super::weights::StateQuant::Q8 => {
            gpu.gated_delta_net_q8_batch_seq(
                &pbs.dn_q_batch,
                &pbs.dn_k_batch,
                &pbs.dn_v_batch,
                &pbs.dn_alpha_batch,
                &pbs.dn_beta_batch,
                &dn_state.s_matrices[delta_layer_idx],
                &dn_state.s_scales[delta_layer_idx],
                &pbs.dn_attn_out_batch,
                n_rows,
                n_v_heads,
                config.linear_value_head_dim,
                dn_state.ef_residual(delta_layer_idx),
            )?;
        }
        super::weights::StateQuant::Q4 => {
            return Err(HipError::new(0, "escha-dense Q4 state not yet batched"));
        }
    }
    gpu.gated_norm_f32_batched(
        &pbs.dn_attn_out_batch,
        &pbs.dn_z_batch,
        &layer.norm_weight,
        &pbs.dn_normed_batch,
        n_v_heads,
        config.linear_value_head_dim,
        config.norm_eps,
        n_rows,
    )?;

    // ── wo coded projection + residual ──
    escha_dense_decode_proj_batch(gpu, &layer.wo, &pbs.dn_normed_batch, &pbs.x_rot_batch, n_rows)?;
    gpu.add_f32(&pbs.x_batch, &pbs.x_rot_batch, &pbs.x_batch)?;

    // ── FFN (gate/up/down coded) ──
    gpu.rmsnorm_batched(&pbs.x_batch, &layer.ffn_norm, &pbs.x_rot_batch, n_rows, config.dim, config.norm_eps)?;
    escha_dense_decode_proj_batch(gpu, &layer.w_gate, &pbs.x_rot_batch, &pbs.gate_ffn_batch, n_rows)?;
    escha_dense_decode_proj_batch(gpu, &layer.w_up, &pbs.x_rot_batch, &pbs.up_batch, n_rows)?;
    gpu.silu_mul_f32(&pbs.gate_ffn_batch, &pbs.up_batch, &pbs.ffn_hidden_batch)?;
    escha_dense_decode_proj_batch(gpu, &layer.w_down, &pbs.ffn_hidden_batch, &pbs.x_rot_batch, n_rows)?;
    gpu.add_f32(&pbs.x_batch, &pbs.x_rot_batch, &pbs.x_batch)?;

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

/// Probe the DeltaNet recurrent state (S-matrix + conv state) magnitudes after
/// a gated-delta-net update for one layer. Gated on HIPFIRE_ESCHA_DENSE_TRACE.
fn state_probe(
    gpu: &Gpu,
    dn_state: &super::weights::DeltaNetState,
    delta_layer_idx: usize,
    label: &str,
) {
    if hipfire_config::developer_var_os("HIPFIRE_ESCHA_DENSE_TRACE").is_none() {
        return;
    }
    // NOTE: Q8 S-matrices are BYTE buffers (shape = byte count), so they cannot
    // be downloaded as f32; only probe the FP32 S-matrix.
    if dn_state.quant == super::weights::StateQuant::FP32 {
        if let Some(t) = dn_state.s_matrices.get(delta_layer_idx) {
            if let Ok(v) = gpu.download_f32(t) {
                let mut s = 0.0f64;
                let mut mx = 0.0f64;
                for &x in &v {
                    s += (x as f64) * (x as f64);
                    mx = mx.max((x as f64).abs());
                }
                eprintln!(
                    "[escha-dense] {label} S[{delta_layer_idx}]: rms={:.4e} absmax={:.4e}",
                    (s / v.len() as f64).sqrt(),
                    mx
                );
            }
        }
    }
    if let Some(t) = dn_state.conv_states.get(delta_layer_idx) {
        if let Ok(v) = gpu.download_f32(t) {
            let mut s = 0.0f64;
            for &x in &v {
                s += (x as f64) * (x as f64);
            }
            eprintln!(
                "[escha-dense] {label} conv[{delta_layer_idx}]: rms={:.4e}",
                (s / v.len() as f64).sqrt()
            );
        }
    }
}

/// Probe a small per-head vector (alpha/beta gates) — env-gated.
fn small_probe(gpu: &Gpu, label: &str, t: &GpuTensor) {
    if hipfire_config::developer_var_os("HIPFIRE_ESCHA_DENSE_TRACE").is_none() {
        return;
    }
    if let Ok(v) = gpu.download_f32(t) {
        let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
        for &x in &v {
            mn = mn.min(x);
            mx = mx.max(x);
        }
        eprintln!("[escha-dense] {label}: n={} range=[{mn:.3e},{mx:.3e}] first4={:?}", v.len(), &v[..4.min(v.len())]);
    }
}

/// Probe whether conv outputs are token-differentiating (env-gated).
fn qkv_probe(gpu: &Gpu, label: &str, t: &GpuTensor) {
    if hipfire_config::developer_var_os("HIPFIRE_ESCHA_DENSE_TRACE").is_none() {
        return;
    }
    if let Ok(v) = gpu.download_f32(t) {
        let (mut mn, mut mx, mut rms) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64);
        for &x in &v {
            mn = mn.min(x);
            mx = mx.max(x);
            rms += (x as f64) * (x as f64) / v.len() as f64;
        }
        eprintln!(
            "[escha-dense] {label}: n={} rms={:.4e} range=[{mn:.3e},{mx:.3e}] first2={:?}",
            v.len(),
            rms.sqrt(),
            &v[..2.min(v.len())]
        );
    }
}
