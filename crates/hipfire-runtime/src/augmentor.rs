// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! WeightAugmentor — a plugin interface for transparently transforming weight
//! tensors at load time. Arch crates call `load_weight()` and augmentors run
//! automatically based on the model's QuantConfig.

use crate::llama::WeightTensor;
use crate::model_source::{ModelSource, QuantConfig};
use hip_bridge::HipResult;
use rdna_compute::Gpu;

// ── Trait ──────────────────────────────────────────────────────────────────────

/// A plugin that may replace or post-process a weight tensor at load time.
///
/// Implementors are registered in `DEFAULT_AUGMENTORS`. `load_weight()` iterates
/// the list; the first active augmentor whose `try_load` returns `Some` wins.
/// If no augmentor fires, the caller must use its own base-loading fallback.
pub trait WeightAugmentor: Send + Sync {
    fn name(&self) -> &'static str;

    /// True if this augmentor applies to models with the given QuantConfig.
    fn is_active_for(&self, qc: &QuantConfig) -> bool;

    /// True if this augmentor applies to the given source (delegates to
    /// is_active_for if quant_config is present, otherwise false).
    fn is_active(&self, source: &dyn ModelSource) -> bool {
        source
            .quant_config()
            .map(|qc| self.is_active_for(qc))
            .unwrap_or(false)
    }

    /// Attempt to fully load the weight tensor named `base_name` (no extension).
    /// Returns `Ok(Some(t))` if this augmentor handles it (e.g. PaRo: reads
    /// `.qweight`, `.qzeros`, etc.), `Ok(None)` to pass to the next augmentor
    /// or to the base loader.
    fn try_load(
        &self,
        source: &dyn ModelSource,
        base_name: &str,
        out_dim: usize,
        in_dim: usize,
        gpu: &mut Gpu,
    ) -> HipResult<Option<WeightTensor>>;
}

// ── Dispatch helper ────────────────────────────────────────────────────────────

/// Try every active augmentor in order. Returns the first `Some(WeightTensor)`
/// found, or `None` if no augmentor handled the tensor.
///
/// The caller is responsible for providing a fallback (standard HFQ loading or
/// error) when `None` is returned.
pub fn try_augmentors(
    source: &dyn ModelSource,
    base_name: &str,
    out_dim: usize,
    in_dim: usize,
    gpu: &mut Gpu,
    augmentors: &[&'static dyn WeightAugmentor],
) -> HipResult<Option<WeightTensor>> {
    for a in augmentors {
        if a.is_active(source) {
            if let Some(t) = a.try_load(source, base_name, out_dim, in_dim, gpu)? {
                return Ok(Some(t));
            }
        }
    }
    Ok(None)
}

// ── ParoAugmentor ──────────────────────────────────────────────────────────────

pub struct ParoAugmentor;

impl ParoAugmentor {
    pub fn is_active_for(qc: &QuantConfig) -> bool {
        qc.method == "paroquant" && qc.krot > 0
    }
}

impl WeightAugmentor for ParoAugmentor {
    fn name(&self) -> &'static str {
        "paroquant"
    }

    fn is_active_for(&self, qc: &QuantConfig) -> bool {
        ParoAugmentor::is_active_for(qc)
    }

    fn try_load(
        &self,
        source: &dyn ModelSource,
        base_name: &str,
        out_dim: usize,
        in_dim: usize,
        gpu: &mut Gpu,
    ) -> HipResult<Option<WeightTensor>> {
        // Only fires if the quantized tensors actually exist for this weight.
        // Some tensors are excluded from quantization (router, embeddings) and
        // have no .qweight — paro_load_wt falls back to .weight for those.
        if source
            .tensor_info(&format!("{base_name}.qweight"))
            .is_none()
        {
            return Ok(None);
        }
        let qc = source
            .quant_config()
            .expect("ParoAugmentor: quant_config required");
        let t = crate::paro::load_paro_weight(
            source,
            gpu,
            base_name,
            out_dim,
            in_dim,
            qc.group_size,
            qc.krot,
        )?;
        Ok(Some(t))
    }
}

// ── Int8Augmentor (Escha-W2 / Qwen3.5-MoE int8 dense export) ───────────────

/// Loads `.weight_int8` + `.weight_scale` (per-row f16 scales) weights for the
/// Escha-W2 code-quant export, where the dense projections (attention + shared
/// expert) are int8. The scale is converted with its STORED dtype (F16 here);
/// misreading it as BF16 zeroes every weight.
pub struct Int8Augmentor;

impl WeightAugmentor for Int8Augmentor {
    fn name(&self) -> &'static str {
        "int8"
    }

    fn is_active_for(&self, qc: &QuantConfig) -> bool {
        qc.method == "escha" || qc.method == "eschamoe"
    }

    fn try_load(
        &self,
        source: &dyn ModelSource,
        base_name: &str,
        out_dim: usize,
        in_dim: usize,
        gpu: &mut Gpu,
    ) -> HipResult<Option<WeightTensor>> {
        let int8_name = format!("{base_name}.weight_int8");
        let scale_name = format!("{base_name}.weight_scale");
        if source.tensor_info(&int8_name).is_none() || source.tensor_info(&scale_name).is_none() {
            return Ok(None);
        }
        let (_, int8_data) = source.tensor_data(&int8_name).unwrap();
        let (scale_info, scale_data) = source.tensor_data(&scale_name).unwrap();
        let scales = crate::safetensors_source::source_bytes_to_f32_vec(&scale_info.dtype, scale_data);
        let mut f32_data = vec![0.0f32; out_dim * in_dim];
        for i in 0..(out_dim * in_dim) {
            let q = int8_data[i] as i8 as f32;
            // Per-row scale: w[r][c] = int8[r][c] * scale[r].
            f32_data[i] = q * scales[i / in_dim];
        }
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4)
        };
        let buf = gpu.upload_raw(bytes, &[out_dim, in_dim])?;
        Ok(Some(WeightTensor {
            buf,
            gpu_dtype: rdna_compute::DType::F32,
            m: out_dim,
            k: in_dim,
            row_stride: 0,
            paro: None,
            awq_scale: None,
        }))
    }
}

// ── Default registry ───────────────────────────────────────────────────────────

static PARO: ParoAugmentor = ParoAugmentor;
static INT8: Int8Augmentor = Int8Augmentor;

/// Default augmentor set used by all arch crates. Extend per-arch by building
/// a custom slice: `&[DEFAULT_AUGMENTORS, &[&MyAugmentor]].concat()`.
pub static DEFAULT_AUGMENTORS: &[&dyn WeightAugmentor] = &[&PARO, &INT8];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_source::QuantConfig;

    fn make_quant_config(krot: u8) -> QuantConfig {
        QuantConfig {
            method: "paroquant".into(),
            bits: 4,
            group_size: 128,
            krot,
            dynamic_excludes: vec![],
        }
    }

    // Tests call the free function `ParoAugmentor::is_active_for(&QuantConfig)`
    // which does not need a ModelSource — no mock needed for these three tests.

    #[test]
    fn paro_augmentor_active_when_krot_positive() {
        let qc = make_quant_config(8);
        assert!(ParoAugmentor::is_active_for(&qc));
    }

    #[test]
    fn paro_augmentor_inactive_when_krot_zero() {
        let qc = make_quant_config(0);
        assert!(!ParoAugmentor::is_active_for(&qc));
    }

    #[test]
    fn paro_augmentor_inactive_for_non_paro_method() {
        let mut qc = make_quant_config(8);
        qc.method = "awq".into();
        assert!(!ParoAugmentor::is_active_for(&qc));
    }
}
