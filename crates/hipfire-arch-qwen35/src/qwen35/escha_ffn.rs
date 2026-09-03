// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Escha-W2 code-quant MoE FFN decode engine (AR per-token path).
//!
//! Per routed expert, the projection is stored as int16 EXL3-trellis codes.
//! This module:
//!   1. lazily decodes an expert's gate_up/down on GPU — trellis unpack +
//!      3INST codebook + tensor-core perm → W_bareᵀ, then folds the two
//!      blockwise-128 WHT passes (`M_folded = H_out @ W_bareᵀ @ H_in`), then
//!      absorbs `diag(rout) · diag(s_in·rin)` into the weight and stores fp16.
//!   2. runs the top-8 routed experts through ONE grouped f16 gemv per
//!      projection + a batched SiLU·mul + a batched scaled add (4 launches),
//!      then the shared expert (dense) and a single residual add.

use super::config::Qwen35Config;
use super::weights::EschaMoeFfnWeights;
use hip_bridge::HipError;
use hip_bridge::HipResult;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
use rdna_compute::DType;
use rdna_compute::Gpu;
use rdna_compute::GpuTensor;

/// Lazily decode + fold one expert projection on GPU and return the folded
/// f32 weight [out_p, in_p] (row-major), ready for the grouped gemv. The
/// per-projection scale absorption + fp16 conversion happen in the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_and_fold_expert_gpu(
    gpu: &mut Gpu,
    codes_view: &GpuTensor,
    k: usize,
    in_p: usize,
    out_p: usize,
) -> HipResult<GpuTensor> {
    // EXL3 trellis + 3INST decode → W_bare^T [out_p, in_p]; pure WHT fold
    // (rout applied post-GEMV, since diag(rout)@T_out != T_out@diag(rout)).
    let decode_scratch = gpu.zeros(&[out_p, in_p], DType::F32)?;
    let mid = gpu.zeros(&[out_p, in_p], DType::F32)?;
    let folded = gpu.zeros(&[out_p, in_p], DType::F32)?;

    rdna_compute::escham::escham_moe_decode_trellis(
        gpu,
        &decode_scratch,
        codes_view,
        k as i32,
        in_p as i32,
        out_p as i32,
    )?;
    // Fold with rout = ones: pure WHT on both axes.
    let ones = vec![1.0f32; out_p];
    let rout_ones = gpu.upload_f32(&ones, &[out_p])?;
    rdna_compute::escham::escham_fold_t128_rows(
        gpu,
        &mid,
        &decode_scratch,
        &rout_ones,
        out_p,
        in_p,
    )?;
    rdna_compute::escham::escham_fold_t128_cols(gpu, &folded, &mid, out_p, in_p)?;
    gpu.free_tensor(rout_ones)?;
    gpu.free_tensor(decode_scratch)?;
    gpu.free_tensor(mid)?;
    Ok(folded)
}

/// Run one token through an Escha code-quant MoE FFN.
///
/// `x_norm` is the post-RMSNorm hidden state; the routed + shared expert
/// output is added into `x_residual` in place.
#[allow(clippy::too_many_arguments)]
pub(crate) fn escham_moe_ffn_decode(
    gpu: &mut Gpu,
    ffn: &EschaMoeFfnWeights,
    x_norm: &GpuTensor,
    x_residual: &GpuTensor,
    config: &Qwen35Config,
    layer_idx: usize,
) -> HipResult<()> {
    let hidden = config.dim;
    let mi = config.moe_intermediate_size;
    let k = config.num_experts_per_tok;
    let n_exp = config.num_experts;
    let smi = config.shared_expert_intermediate_size;

    // ── Router logits (dense f16 gemv) + host top-k ──
    let router_logits = gpu.alloc_tensor(&[n_exp], DType::F32)?;
    {
        let ctx = DispatchCtx::new(gpu);
        let wr = ffn.router.dispatch_ref();
        let step = Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(x_norm),
            out: &router_logits,
        };
        execute_steps(gpu, &ctx, &[step]).map_err(|e| HipError::new(0, &e.to_string()))?;
    }
    let logits_host = gpu.download_f32(&router_logits)?;
    gpu.free_tensor(router_logits)?;

    let topk: Vec<(usize, f32)> = {
        let mut idx: Vec<usize> = (0..n_exp).collect();
        idx.sort_by(|&a, &b| {
            logits_host[b]
                .partial_cmp(&logits_host[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(k);
        idx.into_iter().map(|i| (i, logits_host[i])).collect()
    };
    // softmax over the top-k logits (Qwen convention: exp(v)/Σexp).
    let topk_weights: Vec<f32> = {
        let z: f32 = topk.iter().map(|(_, v)| v.exp()).sum::<f32>().max(1e-30);
        topk.iter().map(|(_, v)| (v.exp() / z) as f32).collect()
    };

    // Padded dims are /128 aligned for this model (dim/mi/hidden are).
    let gu_in_p = hidden;
    let gu_out_p = ((mi * 2 + 127) / 128) * 128;
    let down_in_p = mi;
    let down_out_p = hidden;

    let expert_out = gpu.zeros(&[hidden], DType::F32)?;
    let gate_up_out = gpu.zeros(&[mi * 2], DType::F32)?;
    let down_out = gpu.zeros(&[hidden], DType::F32)?;
    let gated = gpu.zeros(&[mi], DType::F32)?;
    let gu_out_all = gpu.zeros(&[k * gu_out_p], DType::F32)?;
    let gated_all = gpu.zeros(&[k * mi], DType::F32)?;
    let dn_out_all = gpu.zeros(&[k * down_out_p], DType::F32)?;
    let w_dev = gpu.upload_f32(&topk_weights, &[k])?;
    let shared_scalar = gpu.zeros(&[1], DType::F32)?;

    // ── Routed experts: lazy per-expert decode + fold → fp16 cache ──
    let mut cache_gu = ffn.folded_gate_up_cache.borrow_mut();
    let mut cache_down = ffn.folded_down_cache.borrow_mut();
    let mut gu_ptrs: Vec<u64> = Vec::with_capacity(k);
    let mut dn_ptrs: Vec<u64> = Vec::with_capacity(k);

    for &(exp_idx, _logit) in topk.iter() {
        let (gu_codes, gu_in_scale) = (&ffn.gate_up_codes[exp_idx], &ffn.gate_up_in_scale);
        let (dn_codes, dn_in_scale) = (&ffn.down_codes[exp_idx], &ffn.down_in_scale);
        let (gu_rout, dn_rout) = (&ffn.gate_up_rout[exp_idx], &ffn.down_rout[exp_idx]);

        let folded_gu_gpu = if let Some(ref cached) = cache_gu[exp_idx] {
            cached.shallow_clone()
        } else {
            let folded_f32 = decode_and_fold_expert_gpu(gpu, gu_codes, 2, gu_in_p, gu_out_p)?;
            // Absorb rout (rows) + s_in·rin (cols): W' = diag(rout) @ M @ diag(scale).
            rdna_compute::escham::escham_apply_rowcol_scales_f32(
                gpu,
                &folded_f32,
                gu_rout,
                gu_in_scale,
                gu_out_p,
                gu_in_p,
            )?;
            let folded_f16 = gpu.zeros(&[gu_out_p, gu_in_p], DType::F16)?;
            rdna_compute::escham::escham_f32_to_f16(gpu, &folded_f16, &folded_f32)?;
            gpu.free_tensor(folded_f32)?;
            cache_gu[exp_idx] = Some(folded_f16.shallow_clone());
            folded_f16
        };

        let folded_down_gpu = if let Some(ref cached) = cache_down[exp_idx] {
            cached.shallow_clone()
        } else {
            let folded_f32 = decode_and_fold_expert_gpu(gpu, dn_codes, 3, down_in_p, down_out_p)?;
            rdna_compute::escham::escham_apply_rowcol_scales_f32(
                gpu,
                &folded_f32,
                dn_rout,
                dn_in_scale,
                down_out_p,
                down_in_p,
            )?;
            let folded_f16 = gpu.zeros(&[down_out_p, down_in_p], DType::F16)?;
            rdna_compute::escham::escham_f32_to_f16(gpu, &folded_f16, &folded_f32)?;
            gpu.free_tensor(folded_f32)?;
            cache_down[exp_idx] = Some(folded_f16.shallow_clone());
            folded_f16
        };

        gu_ptrs.push(folded_gu_gpu.buf.as_ptr() as u64);
        dn_ptrs.push(folded_down_gpu.buf.as_ptr() as u64);
    }
    drop(cache_gu);
    drop(cache_down);

    let gu_bytes: Vec<u8> = gu_ptrs.iter().flat_map(|q| q.to_ne_bytes()).collect();
    let dn_bytes: Vec<u8> = dn_ptrs.iter().flat_map(|q| q.to_ne_bytes()).collect();
    let gu_ptrs_t = gpu.upload_raw(&gu_bytes, &[2 * k])?;
    let dn_ptrs_t = gpu.upload_raw(&dn_bytes, &[2 * k])?;

    // One grouped gemv per projection over the k routed experts.
    rdna_compute::escham::escham_moe_grouped_gemv_f16(
        gpu, &gu_ptrs_t, x_norm, &gu_out_all, gu_out_p, gu_in_p, k, 0,
    )?;
    rdna_compute::escham::escham_moe_batched_silu_mul_f32(gpu, &gu_out_all, &gated_all, k, mi)?;
    rdna_compute::escham::escham_moe_grouped_gemv_f16(
        gpu, &dn_ptrs_t, &gated_all, &dn_out_all, down_out_p, down_in_p, k, mi,
    )?;
    rdna_compute::escham::escham_moe_batched_scaled_add_f32(
        gpu, &expert_out, &dn_out_all, &w_dev, k, hidden,
    )?;

    // ── Shared expert (dense) ──
    let shared = &ffn.shared_expert;
    {
        let ctx = DispatchCtx::new(gpu);
        let wr = ffn.shared_expert_gate.dispatch_ref();
        let step = Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(x_norm),
            out: &shared_scalar,
        };
        execute_steps(gpu, &ctx, &[step]).map_err(|e| HipError::new(0, &e.to_string()))?;
    }
    gpu.sigmoid_f32(&shared_scalar)?;
    {
        let ctx = DispatchCtx::new(gpu);
        let wr = shared.gate.dispatch_ref();
        let out_view = gate_up_out.sub_offset(0, smi);
        execute_steps(gpu, &ctx, &[Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(x_norm),
            out: &out_view,
        }])
        .map_err(|e| HipError::new(0, &e.to_string()))?;
    }
    {
        let ctx = DispatchCtx::new(gpu);
        let wr = shared.up.dispatch_ref();
        let out_view = down_out.sub_offset(0, smi);
        execute_steps(gpu, &ctx, &[Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(x_norm),
            out: &out_view,
        }])
        .map_err(|e| HipError::new(0, &e.to_string()))?;
    }
    let gate_view = gate_up_out.sub_offset(0, smi);
    let up_view = down_out.sub_offset(0, smi);
    gpu.silu_mul_f32(&gate_view, &up_view, &gated)?;
    {
        let ctx = DispatchCtx::new(gpu);
        let wr = shared.down.dispatch_ref();
        let gated_view = gated.sub_offset(0, smi);
        let out_view = down_out.sub_offset(0, hidden);
        execute_steps(gpu, &ctx, &[Step::Gemv {
            w: &wr,
            input: GemvInput::Raw(&gated_view),
            out: &out_view,
        }])
        .map_err(|e| HipError::new(0, &e.to_string()))?;
    }
    gpu.scaled_add_inplace_gpu_scalar_f32(&expert_out, &down_out, &shared_scalar)?;

    // Single residual add after routed + shared.
    gpu.add_f32(x_residual, &expert_out, x_residual)?;

    gpu.free_tensor(expert_out)?;
    gpu.free_tensor(gate_up_out)?;
    gpu.free_tensor(down_out)?;
    gpu.free_tensor(gated)?;
    gpu.free_tensor(gu_out_all)?;
    gpu.free_tensor(gated_all)?;
    gpu.free_tensor(dn_out_all)?;
    gpu.free_tensor(w_dev)?;
    gpu.free_tensor(gu_ptrs_t)?;
    gpu.free_tensor(dn_ptrs_t)?;
    gpu.free_tensor(shared_scalar)?;

    let _ = layer_idx;
    Ok(())
}
