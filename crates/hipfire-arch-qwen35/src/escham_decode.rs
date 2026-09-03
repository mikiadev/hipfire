// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! ESCHAM codebook decode and reference forward implementation.
//!
//! The ESCHAM weight format is an EXL3-style **trellis-coded** quantization
//! (see higgs `tools/escha_ref.py` and exllamav3 `pack.cu`/`codebook.cuh`):
//!
//! - `escha_code` I16 [E, in/16, out/16, 16*K] is a **tail-biting trellis**
//!   stream. Each 16-bit code is a window into a circular 256*K-bit stream at
//!   stride K; consecutive codes overlap by 16-K bits (`unpack_trellis`).
//! - Each unpacked 16-bit code is decoded through the **3INST codebook**:
//!   `x = (code * 0xCBAC1FED & 0x8FFF8FFF) ^ 0x3B603B60`, then the two fp16
//!   halves are summed (`decode_3inst`). This is the same formula llama.cpp's
//!   escha-moe kernel computes inline.
//! - The 256 decoded scalars are scattered into the 16×16 tile via a fixed
//!   `tensor_core_perm` (exl3_lib/quantize.py).
//! - The full expert weight is `W = had128(had128(w_bare * rin * rout))` with
//!   blockwise-128 Sylvester Hadamard on both axes (scales OUTSIDE the WHT).
//!
//! **CRITICAL**: the earlier "sparse-compact delta table" decode
//! (`compact.pkl` / `compact.bin` — per-slot (65536, n_nz) additive deltas
//! indexed by the raw 16-bit code) was WRONG. It matched single-slot probes
//! of the op but not the full trellis decode: decoded weights were ~1000× too
//! large and orthogonal (cosine ~0) to the true weights. The correct decode
//! needs no codebook at all — it is a pure function of the codes.
//!
//! The forward chain (matching higgs `escha_ref.reconstruct`, variant
//! "both_outside"):
//! ```text
//! w_bare = decode_tiles(code)                     [in, out]
//! W      = had128(had128(w_bare * rin * rout))    [in, out]
//! y      = W.T @ x                                [out]   (or folded GEMV)
//! ```
//! Folding for the GEMV path (M = W^T [out, in]):
//! ```text
//! M_folded = diag(rout) @ H_out @ W_bare^T @ H_in
//! y        = M_folded @ (x . s_in . rin)
//! ```
//! (rout OUTSIDE the output WHT — applying it inside is NOT equivalent.)

use std::path::Path;
use hip_bridge::{HipError, HipResult};

/// MCG multiplier for the 3INST codebook (same as llama.cpp / higgs).
const MCG_MULT: u32 = 0xCBAC_1FED;
/// LOP3 AND mask (LOP3.LUT 0x6a = (a & b) ^ c).
const LOP3_AND: u32 = 0x8FFF_8FFF;
/// LOP3 XOR constant.
const LOP3_XOR: u32 = 0x3B60_3B60;
/// Hadamard block size.
const HAD_BLOCK: usize = 128;

/// Unpack the tail-biting trellis codes: (..., 16*K) int16 -> (..., 256) uint16.
///
/// Each code is a 16-bit window into a circular 256*K-bit stream at stride K;
/// consecutive codes overlap by 16-K bits. Ported from higgs
/// `escha_ref.unpack_trellis` / exllamav3 `unpack_trellis_kernel`.
pub fn unpack_trellis(packed: &[i16], k: usize) -> Vec<u16> {
    let n_codes = packed.len() / (16 * k);
    let mut out = vec![0u16; n_codes * 256];
    let n_words = (k * 256) / 32; // uint32 words in the payload

    for c in 0..n_codes {
        // Reinterpret the 16*k int16 as 8*k uint32 (little-endian host order,
        // matching the CUDA kernel's u32 view of the same memory).
        let mut u32buf = [0u32; 24]; // max 8*3 = 24 words (K=3)
        for (i, w) in u32buf.iter_mut().enumerate().take(8 * k) {
            let lo = packed[c * 16 * k + i * 2] as u16 as u32;
            let hi = packed[c * 16 * k + i * 2 + 1] as u16 as u32;
            *w = lo | (hi << 16);
        }
        for t in 0..128usize {
            // NOTE: keep `- 16` last — evaluated left-to-right, `k - 16` would
            // underflow usize in debug builds for small t (release wraps to the
            // same final value, so release-only testing missed this).
            let b0 = t * 2 * k + k + 256 * k - 16;
            let b2 = b0 + k + 16;
            let i0 = b0 / 32;
            let i1 = (b2 - 1) / 32;
            let s1 = (i1 + 1) * 32 - b2;

            let a = u32buf[i0 % n_words] as u64;
            let b = u32buf[i1 % n_words] as u64;
            // __funnelshift_r(lo=b, hi=a, s1): bits [s1+31:s1] of the 64-bit {a,b}
            let w1 = (((a << 32) | b) >> s1) as u32;
            let w0 = (w1 >> k) & 0xFFFF;
            let w1 = w1 & 0xFFFF;
            out[c * 256 + t * 2] = w0 as u16;
            out[c * 256 + t * 2 + 1] = w1 as u16;
        }
    }
    out
}

/// The 256-entry 16×16 tile permutation (exl3_lib/quantize.py `tensor_core_perm`).
pub fn tensor_core_perm() -> [usize; 256] {
    let mut perm = [0usize; 256];
    for t in 0..32usize {
        let r0 = (t % 4) * 2;
        let c0 = t / 4;
        let rows = [r0, r0 + 1, r0 + 8, r0 + 9];
        for (j, c) in [c0, c0 + 8].iter().enumerate() {
            for (i, r) in rows.iter().enumerate() {
                perm[t * 8 + j * 4 + i] = r * 16 + c;
            }
        }
    }
    perm
}

/// Decode 3INST codebook scalars: `x = (code*MCG & AND) ^ XOR`, then sum the
/// two fp16 halves. Ported from higgs `escha_ref.decode_3inst` (codebook.cuh
/// `decode_3inst<1>`) and llama.cpp `escha_codebook`.
pub fn decode_3inst(codes: &[u16]) -> Vec<f32> {
    codes
        .iter()
        .map(|&c| {
            let x = (c as u32).wrapping_mul(MCG_MULT) & LOP3_AND ^ LOP3_XOR;
            let lo = (x & 0xFFFF) as u16;
            let hi = ((x >> 16) & 0xFFFF) as u16;
            let lo_f = f16_to_f32(lo);
            let hi_f = f16_to_f32(hi);
            lo_f + hi_f
        })
        .collect()
}

/// Decode one expert's codes to its bare (pre-Hadamard, pre-scale) weight.
/// `code` is [bi_max, bj_max, 16*K] int16 (row-major), returns [in_p, out_p] f32.
pub fn decode_tiles(code: &[i16], k: usize, in_p: usize, out_p: usize) -> Vec<f32> {
    let bi_max = in_p / 16;
    let bj_max = out_p / 16;
    let perm = tensor_core_perm();
    let n_tiles = bi_max * bj_max;
    // unpack per tile: each tile's codes are [16*K] at (bi, bj)
    let mut vals = vec![0.0f32; n_tiles * 256];
    for bi in 0..bi_max {
        for bj in 0..bj_max {
            let tile_codes: Vec<i16> = (0..16 * k)
                .map(|kk| code[(bi * bj_max + bj) * (16 * k) + kk])
                .collect();
            let unpacked = unpack_trellis(&tile_codes, k);
            let decoded = decode_3inst(&unpacked);
            let base = (bi * bj_max + bj) * 256;
            for (i, &d) in decoded.iter().enumerate() {
                vals[base + i] = d;
            }
        }
    }
    // scatter via perm into [bi_max, bj_max, 16, 16], then reshape [in_p, out_p]
    let mut w = vec![0.0f32; in_p * out_p];
    for bi in 0..bi_max {
        for bj in 0..bj_max {
            let base = (bi * bj_max + bj) * 256;
            for (i, &v) in vals[base..base + 256].iter().enumerate() {
                let p = perm[i];
                let r = p / 16;
                let c = p % 16;
                w[(bi * 16 + r) * out_p + (bj * 16 + c)] += v;
            }
        }
    }
    w
}

/// Public wrapper: blockwise-128 Sylvester Hadamard on both axes of a row-major
/// [rows, cols] matrix (in-axis 0, then out-axis 1). Matches the GPU fold.
pub fn had128_matrix(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let m1 = had128_2d(m, rows, cols, 1);
    had128_2d(&m1, rows, cols, 0)
}

/// Blockwise-128 Sylvester Hadamard applied along `axis` of a row-major [rows, cols]
/// matrix. `axis=0` transforms rows (length `rows`), `axis=1` transforms columns.
/// Returns a new matrix.
fn had128_2d(m: &[f32], rows: usize, cols: usize, axis: usize) -> Vec<f32> {    let mut out = m.to_vec();
    if axis == 0 {
        // WHT over rows in 128-blocks: each column independently
        for c in 0..cols {
            for b in 0..rows / HAD_BLOCK {
                let base = b * HAD_BLOCK * cols + c;
                let mut block: Vec<f32> = (0..HAD_BLOCK).map(|r| m[base + r * cols]).collect();
                had128_inplace(&mut block);
                for r in 0..HAD_BLOCK {
                    out[base + r * cols] = block[r];
                }
            }
        }
    } else {
        // WHT over cols in 128-blocks: each row independently
        for r in 0..rows {
            for b in 0..cols / HAD_BLOCK {
                let base = r * cols + b * HAD_BLOCK;
                let mut block: Vec<f32> = m[base..base + HAD_BLOCK].to_vec();
                had128_inplace(&mut block);
                out[base..base + HAD_BLOCK].copy_from_slice(&block);
            }
        }
    }
    out
}

/// In-place normalized Sylvester Hadamard over a 128-element block.
fn had128_inplace(x: &mut [f32]) {
    debug_assert_eq!(x.len(), HAD_BLOCK);
    let mut h = 1usize;
    while h < HAD_BLOCK {
        for i in (0..HAD_BLOCK).step_by(2 * h) {
            for j in 0..h {
                let a = x[i + j];
                let b = x[i + j + h];
                x[i + j] = a + b;
                x[i + j + h] = a - b;
            }
        }
        h *= 2;
    }
    let scale = 1.0 / (HAD_BLOCK as f32).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// f16 bit pattern -> f32 (IEEE half, round-to-nearest-even via f32 conversion).
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;
    let f = if exp == 0 {
        if frac == 0 {
            (sign << 31) as f32
        } else {
            // subnormal
            let mut e = 1u32;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e += 1;
            }
            f &= 0x3FF;
            let bits = (sign << 31) | ((127 - e + 15) << 23) | (f << 13);
            f32::from_bits(bits)
        }
    } else if exp == 0x1F {
        if frac == 0 {
            f32::from_bits((sign << 31) | 0x7F80_0000)
        } else {
            f32::from_bits((sign << 31) | 0x7FC0_0000)
        }
    } else {
        let bits = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
        f32::from_bits(bits)
    };
    f
}

/// Decode one expert's gate_up or down projection to its FULL weight [out, in]
/// (Hadamard + scales applied, matching higgs `escha_ref.reconstruct` variant
/// "both_outside": both scales OUTSIDE the WHT).
///
/// Returns row-major [out_p, in_p] f32 (the GEMV-ready orientation).
pub fn decode_to_full_weight(
    code: &[i16],
    k: usize,
    in_features: usize,
    out_features: usize,
    rin: &[f32],
    rout: &[f32],
) -> Vec<f32> {
    let in_p = ((in_features + 127) / 128) * 128;
    let out_p = ((out_features + 127) / 128) * 128;
    // w_bare [in_p, out_p]
    let mut w_bare = decode_tiles(code, k, in_p, out_p);
    // apply rin (per in col) and rout (per out col) — outside the WHT
    for i in 0..in_p {
        for j in 0..out_p {
            let si = if i < in_features { rin[i] } else { 1.0 };
            let so = if j < out_features { rout[j] } else { 1.0 };
            w_bare[i * out_p + j] *= si * so;
        }
    }
    // had over in-axis (0) then out-axis (1)
    let w_h = had128_2d(&w_bare, in_p, out_p, 0);
    let w_hh = had128_2d(&w_h, in_p, out_p, 1);
    // transpose to [out_p, in_p]
    let mut w = vec![0.0f32; out_p * in_p];
    for i in 0..in_p {
        for j in 0..out_p {
            w[j * in_p + i] = w_hh[i * out_p + j];
        }
    }
    w
}

/// The GEMV-ready FOLDED weight: `M_folded = diag(rout) @ H_out @ W_bare^T @ H_in`
/// so that `y = M_folded @ (x . s_in . rin)` reproduces the reference forward
/// `y = rout . H_out( H_in(x . rin) @ W_bare )`. rout is OUTSIDE the out WHT.
///
/// Returns [out_p, in_p] f32 row-major.
pub fn decode_and_fold_weight(
    code: &[i16],
    k: usize,
    in_features: usize,
    out_features: usize,
    rin: &[f32],
    rout: &[f32],
) -> Vec<f32> {
    let in_p = ((in_features + 127) / 128) * 128;
    let out_p = ((out_features + 127) / 128) * 128;
    let w_bare = decode_tiles(code, k, in_p, out_p); // [in_p, out_p]
    // M = W_bare^T [out_p, in_p]
    let mut m = vec![0.0f32; out_p * in_p];
    for i in 0..in_p {
        for j in 0..out_p {
            m[j * in_p + i] = w_bare[i * out_p + j];
        }
    }
    // H_in over cols (in axis), H_out over rows (out axis), then scale rows by rout
    let m1 = had128_2d(&m, out_p, in_p, 1); // cols (in)
    let mut m2 = had128_2d(&m1, out_p, in_p, 0); // rows (out)
    for j in 0..out_p {
        let s = if j < out_features { rout[j] } else { 1.0 };
        for i in 0..in_p {
            m2[j * in_p + i] *= s;
        }
    }
    m2
}

/// Read one expert's int16 code tensor from a safetensors buffer.
/// `key` names the tensor; the expert index slices the leading axis.
pub fn read_safetensors_expert_i16(
    shard_bytes: &[u8],
    key: &str,
    expert: usize,
) -> HipResult<Vec<i16>> {
    let (header, data_start) = parse_safetensors_header(shard_bytes)?;
    let meta = header
        .get(key)
        .ok_or_else(|| HipError::new(0, &format!("safetensors tensor not found: {key}")))?;
    let dtype = meta
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| HipError::new(0, "missing dtype"))?;
    if dtype != "I16" {
        return Err(HipError::new(
            0,
            &format!("expected I16 for {key}, got {dtype}"),
        ));
    }
    let shape: Vec<usize> = meta
        .get("shape")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
        .ok_or_else(|| HipError::new(0, "missing shape"))?;
    let offsets = meta
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect::<Vec<_>>())
        .ok_or_else(|| HipError::new(0, "missing data_offsets"))?;
    let stride = shape.iter().skip(1).product::<usize>();
    let begin = data_start + offsets[0] + expert * stride * 2;
    let end = begin + stride * 2;
    let slice = shard_bytes
        .get(begin..end)
        .ok_or_else(|| HipError::new(0, "safetensors slice out of bounds"))?;
    Ok(slice
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Read one expert's fp16 scale tensor (rin/rout) from a safetensors buffer.
pub fn read_safetensors_expert_f16(
    shard_bytes: &[u8],
    key: &str,
    expert: usize,
) -> HipResult<Vec<f32>> {
    let (header, data_start) = parse_safetensors_header(shard_bytes)?;
    let meta = header
        .get(key)
        .ok_or_else(|| HipError::new(0, &format!("safetensors tensor not found: {key}")))?;
    let dtype = meta
        .get("dtype")
        .and_then(|v| v.as_str())
        .ok_or_else(|| HipError::new(0, "missing dtype"))?;
    if dtype != "F16" {
        return Err(HipError::new(
            0,
            &format!("expected F16 for {key}, got {dtype}"),
        ));
    }
    let shape: Vec<usize> = meta
        .get("shape")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
        .ok_or_else(|| HipError::new(0, "missing shape"))?;
    let offsets = meta
        .get("data_offsets")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect::<Vec<_>>())
        .ok_or_else(|| HipError::new(0, "missing data_offsets"))?;
    let stride = shape.iter().skip(1).product::<usize>();
    let begin = data_start + offsets[0] + expert * stride * 2;
    let end = begin + stride * 2;
    let slice = shard_bytes
        .get(begin..end)
        .ok_or_else(|| HipError::new(0, "safetensors slice out of bounds"))?;
    Ok(slice
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

/// Parse a safetensors header, returning (metadata map, data_start offset).
fn parse_safetensors_header(
    bytes: &[u8],
) -> HipResult<(serde_json::Map<String, serde_json::Value>, usize)> {
    if bytes.len() < 8 {
        return Err(HipError::new(0, "safetensors too short"));
    }
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n])
        .map_err(|e| HipError::new(0, &format!("bad safetensors header JSON: {e}")))?;
    let map = header
        .as_object()
        .cloned()
        .ok_or_else(|| HipError::new(0, "safetensors header not an object"))?;
    Ok((map, 8 + n))
}

/// Sparse-compact codebook for ESCHAM quantization.
///
/// Loaded from `compact.bin` (~244 MB) and held in CPU memory.
/// Used to decode int16 codes to fp32 weight matrices on-the-fly.
///
/// Data layout (matches Python pickle format):
/// - positions: list of arrays, each of shape (n_nz, 2) stored as flat Vec<i8>
/// - values: list of arrays, each of shape (65536, n_nz) stored as Vec<f32>
///
/// Held behind `Arc` for cheap cloning across layers (244 MB loaded once,
/// shared by all 40 MoE layers).
#[derive(Debug)]
pub struct Codebook {
    /// K=2 positions: list of flat Vec<i8> arrays (n_nz*2 elements each)
    pub k2_positions: Vec<Vec<i8>>,
    /// K=2 values: list of Vec<f32> arrays (65536*n_nz elements each)
    pub k2_values: Vec<Vec<f32>>,
    /// K=3 positions: list of flat Vec<i8> arrays (n_nz*2 elements each)
    pub k3_positions: Vec<Vec<i8>>,
    /// K=3 values: list of Vec<f32> arrays (65536*n_nz elements each)
    pub k3_values: Vec<Vec<f32>>,
}

/// GPU-resident codebook for on-device decode.
///
/// Contains the same data as `Codebook` but uploaded to GPU memory so that
/// the decode kernel can read slots without host-GPU transfers.
pub struct CodebookGpu {
    pub k2_positions: rdna_compute::GpuTensor,   // flat i8 array (padded, uniform stride)
    pub k2_values: rdna_compute::GpuTensor,       // flat f32 array (padded, uniform stride)
    pub k3_positions: rdna_compute::GpuTensor,    // flat i8 array (padded, uniform stride)
    pub k3_values: rdna_compute::GpuTensor,       // flat f32 array (padded, uniform stride)
    pub k2_pos_len: usize,
    pub k2_val_len: usize,
    pub k3_pos_len: usize,
    pub k3_val_len: usize,
    /// Elements per slot (positions: n_nz*2; values: 65536*n_nz) after padding.
    /// The decode kernel indexes with these uniform strides.
    pub k2_pos_stride: usize,
    pub k2_val_stride: usize,
    pub k3_pos_stride: usize,
    pub k3_val_stride: usize,
}

impl Codebook {
    /// Load codebook from compact.bin file (converted from compact.pkl).
    ///
    /// Binary format:
    ///   MAGIC: b"ESCHACB"  (7 bytes)
    ///   VERSION: u32 LE (1)
    ///   K2_NUM_SLOTS: u32 LE
    ///   K3_NUM_SLOTS: u32 LE
    ///   For each K2 slot: n_nz(u32 LE), positions(n_nz*2 i8 bytes), values(n_nz*65536 f32 LE)
    ///   For each K3 slot: n_nz(u32 LE), positions(n_nz*2 i8 bytes), values(n_nz*65536 f32 LE)
    pub fn load(path: &Path) -> HipResult<Self> {
        use std::io::Read;

        let mut file = std::fs::File::open(path)
            .map_err(|e| HipError::new(0, &format!("Failed to open codebook: {}", e)))?;

        // Read magic
        let mut magic = [0u8; 7];
        file.read_exact(&mut magic)
            .map_err(|e| HipError::new(0, &format!("Failed to read magic: {}", e)))?;
        if magic != *b"ESCHACB" {
            return Err(HipError::new(
                0,
                &format!(
                    "Invalid codebook magic: {:?}, expected ESCHACB",
                    std::str::from_utf8(&magic)
                ),
            ));
        }

        // Read version
        let mut version_buf = [0u8; 4];
        file.read_exact(&mut version_buf)
            .map_err(|e| HipError::new(0, &format!("Failed to read version: {}", e)))?;
        let version = u32::from_le_bytes(version_buf);
        if version != 1 {
            return Err(HipError::new(
                0,
                &format!("Unsupported codebook version: {}", version),
            ));
        }

        // Read slot counts
        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)
            .map_err(|e| HipError::new(0, &format!("Failed to read K2 num slots: {}", e)))?;
        let k2_num = u32::from_le_bytes(buf4) as usize;

        file.read_exact(&mut buf4)
            .map_err(|e| HipError::new(0, &format!("Failed to read K3 num slots: {}", e)))?;
        let k3_num = u32::from_le_bytes(buf4) as usize;

        // Read K2 slots
        let mut k2_positions = Vec::with_capacity(k2_num);
        let mut k2_values = Vec::with_capacity(k2_num);

        for _ in 0..k2_num {
            file.read_exact(&mut buf4)
                .map_err(|e| HipError::new(0, &format!("Failed to read K2 n_nz: {}", e)))?;
            let n_nz = u32::from_le_bytes(buf4) as usize;

            let pos_size = n_nz * 2;
            let mut pos_buf = vec![0u8; pos_size];
            file.read_exact(&mut pos_buf)
                .map_err(|e| HipError::new(0, &format!("Failed to read K2 positions: {}", e)))?;
            // Cast u8 -> i8 (same size/alignment)
            let pos_i8: Vec<i8> = unsafe {
                std::slice::from_raw_parts(pos_buf.as_ptr() as *const i8, pos_size)
                    .to_vec()
            };
            k2_positions.push(pos_i8);

            let val_size = n_nz * 65536;
            let mut val_buf = vec![0.0f32; val_size];
            file.read_exact(unsafe {
                std::slice::from_raw_parts_mut(val_buf.as_mut_ptr() as *mut u8, val_size * 4)
            })
            .map_err(|e| HipError::new(0, &format!("Failed to read K2 values: {}", e)))?;
            k2_values.push(val_buf);
        }

        // Read K3 slots
        let mut k3_positions = Vec::with_capacity(k3_num);
        let mut k3_values = Vec::with_capacity(k3_num);

        for _ in 0..k3_num {
            file.read_exact(&mut buf4)
                .map_err(|e| HipError::new(0, &format!("Failed to read K3 n_nz: {}", e)))?;
            let n_nz = u32::from_le_bytes(buf4) as usize;

            let pos_size = n_nz * 2;
            let mut pos_buf = vec![0u8; pos_size];
            file.read_exact(&mut pos_buf)
                .map_err(|e| HipError::new(0, &format!("Failed to read K3 positions: {}", e)))?;
            let pos_i8: Vec<i8> = unsafe {
                std::slice::from_raw_parts(pos_buf.as_ptr() as *const i8, pos_size)
                    .to_vec()
            };
            k3_positions.push(pos_i8);

            let val_size = n_nz * 65536;
            let mut val_buf = vec![0.0f32; val_size];
            file.read_exact(unsafe {
                std::slice::from_raw_parts_mut(val_buf.as_mut_ptr() as *mut u8, val_size * 4)
            })
            .map_err(|e| HipError::new(0, &format!("Failed to read K3 values: {}", e)))?;
            k3_values.push(val_buf);
        }

        eprintln!(
            "[Escha Codebook] Loaded {} K2 slots, {} K3 slots from {:?}",
            k2_num,
            k3_num,
            path
        );

        Ok(Self {
            k2_positions,
            k2_values,
            k3_positions,
            k3_values,
        })
    }

    /// Upload this codebook to GPU for on-device decode.
    pub fn upload_to_gpu(&self, gpu: &mut rdna_compute::Gpu) -> HipResult<CodebookGpu> {
        // The ESCHAM codebook slots are NOT uniform: K3 positions have n_nz
        // either 10 or 11 in this export. The GPU decode kernel indexes slots
        // with a fixed stride, so pad every slot to the max n_nz of its K
        // family (zero fill: a padded (0,0) position with a 0 value adds 0).
        // Positions are padded to a uniform per-slot stride (the decode kernel
        // indexes slots as `k3_pos + k_idx * pos_stride`); the real positions of
        // a short slot sit at the front, trailing zeros add nothing.
        //
        // Values are REPACKED to a uniform per-code row width of `max_nz`: each
        // slot's [65536, n_nz_k] array becomes [65536, max_nz] with the real
        // entries at the front of each code row and zero fill after. This keeps
        // the kernel's `val_ptr[code_u * val_stride + i]` indexing valid even
        // though different slots have different original n_nz (K3 mixes 10/11).
        let repack_family = |positions: &[Vec<i8>], values: &[Vec<f32>]| {
            let max_nz = positions
                .iter()
                .map(|p| p.len() / 2)
                .max()
                .unwrap_or(0);
            let pos_stride = max_nz * 2;
            let val_stride = 65536 * max_nz;
            let mut pos_out = Vec::with_capacity(positions.len() * pos_stride);
            for p in positions.iter() {
                pos_out.extend_from_slice(p);
                pos_out.resize(pos_out.len() + (pos_stride - p.len()), 0i8);
            }
            let mut val_out = vec![0.0f32; values.len() * val_stride];
            for (s, v) in values.iter().enumerate() {
                let nz = v.len() / 65536;
                debug_assert!(nz <= max_nz && nz * 65536 == v.len());
                for c in 0..65536 {
                    let src = &v[c * nz..(c + 1) * nz];
                    let base = s * val_stride + c * max_nz;
                    val_out[base..base + nz].copy_from_slice(src);
                }
            }
            (pos_out, val_out, pos_stride, val_stride)
        };

        let (k2_pos, k2_vals, k2_pos_stride, k2_val_stride) =
            repack_family(&self.k2_positions, &self.k2_values);
        let (k3_pos, k3_vals, k3_pos_stride, k3_val_stride) =
            repack_family(&self.k3_positions, &self.k3_values);

        let k2_pos_len = k2_pos.len();
        let k2_pos_bytes = unsafe {
            std::slice::from_raw_parts(k2_pos.as_ptr() as *const u8, k2_pos_len * std::mem::size_of::<i8>())
        };
        let k2_pos_gpu = gpu.upload_raw(k2_pos_bytes, &[k2_pos_len])?;
        let k2_val = gpu.upload_f32(&k2_vals, &[k2_vals.len()])?;

        let k3_pos_len = k3_pos.len();
        let k3_pos_bytes = unsafe {
            std::slice::from_raw_parts(k3_pos.as_ptr() as *const u8, k3_pos_len * std::mem::size_of::<i8>())
        };
        let k3_pos_gpu = gpu.upload_raw(k3_pos_bytes, &[k3_pos_len])?;
        let k3_val = gpu.upload_f32(&k3_vals, &[k3_vals.len()])?;

        eprintln!("[Escha] Codebook uploaded to GPU: K2 pos={}, K2 val={}, K3 pos={}, K3 val={} (padded strides pos {}x{} val {}x{})",
            k2_pos_len, k2_vals.len(), k3_pos_len, k3_vals.len(),
            k2_pos_stride, k3_pos_stride, k2_val_stride, k3_val_stride);

        Ok(CodebookGpu {
            k2_positions: k2_pos_gpu,
            k2_values: k2_val,
            k3_positions: k3_pos_gpu,
            k3_values: k3_val,
            k2_pos_len: k2_pos_len,
            k2_val_len: k2_vals.len(),
            k3_pos_len: k3_pos_len,
            k3_val_len: k3_vals.len(),
            k2_pos_stride,
            k2_val_stride,
            k3_pos_stride,
            k3_val_stride,
        })
    }

    /// Check codebook slot lengths
    pub fn check_slot_lengths(&self) {
        if !self.k2_positions.is_empty() {
            let first_pos = &self.k2_positions[0];
            let first_val = &self.k2_values[0];
            eprintln!("[Codebook Debug] K2 first slot: pos len={}, val len={}", first_pos.len(), first_val.len());
        }
        if !self.k3_positions.is_empty() {
            let first_pos = &self.k3_positions[0];
            let first_val = &self.k3_values[0];
            eprintln!("[Codebook Debug] K3 first slot: pos len={}, val len={}", first_pos.len(), first_val.len());
        }
    }

    /// Create an empty codebook (for testing).
    pub fn empty() -> Self {
        Self {
            k2_positions: Vec::new(),
            k2_values: Vec::new(),
            k3_positions: Vec::new(),
            k3_values: Vec::new(),
        }
    }
}

/// Decode int16 codes to fp32 weight matrix using the codebook.
///
/// Implements the reconstruction formula from the codebook README.
///
/// # Arguments
/// * `code` - flat int16 codes, length = in_p//16 * out_p//16 * 16 * K
/// * `codebook` - reference to the codebook
/// * `k` - codebook parameter (2 for gate_up, 3 for down)
/// * `in_features` - actual input dimension
/// * `out_features` - actual output dimension
///
/// # Returns
/// Dense fp32 weight matrix of shape (in_features, out_features) in row-major
pub fn decode_codes_to_weights(
    code: &[i16],
    codebook: &Codebook,
    k: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let in_p = ((in_features + 127) / 128) * 128; // padded to 128
    let out_p = ((out_features + 127) / 128) * 128;

    let bi_max = in_p / 16;
    let bj_max = out_p / 16;

    // Get positions and values for this K
    let (positions, values) = if k == 2 {
        (&codebook.k2_positions, &codebook.k2_values)
    } else {
        (&codebook.k3_positions, &codebook.k3_values)
    };

    let k_max = positions.len();

    // Decode codes block by block
    // NOTE: k_idx is the slot index directly (0..total_codes), not k_idx/16.
    // The codebook has `total_codes` slots (32 for K=2, 48 for K=3), and
    // code[code_offset + k_idx] indexes into slot k_idx.
    let mut w = vec![0.0f32; in_features * out_features];

    let total_codes = k * 16;  // slots per expert: K*16 (e.g. 32 for K=2, 48 for K=3)

    for bi in 0..bi_max {
        for bj in 0..bj_max {
            for k_idx in 0..total_codes {
                let code_offset = (bi * bj_max + bj) * total_codes + k_idx;
                let code_val = code[code_offset] as u16 as usize;

                if code_val >= 65536 {
                    eprintln!("[Escha Debug] code_val {} >= 65536 at code_offset {}", code_val, code_offset);
                }

                if k_idx >= k_max {
                    continue;
                }

                let pos = &positions[k_idx];
                let val = &values[k_idx];

                // Positions are flat: [row0, col0, row1, col1, ...]
                // Values are flat: [code0_row0, code0_row1, ..., code1_row0, ...]
                let n_nz = pos.len() / 2;
                for i in 0..n_nz {
                    let row = pos[i * 2] as usize;
                    let col = pos[i * 2 + 1] as usize;

                    // Check bounds
                    let w_row = bi * 16 + row;
                    let w_col = bj * 16 + col;

                    if w_row < in_features && w_col < out_features {
                        // val is shape (65536, n_nz), flat: val[code_val * n_nz + i]
                        w[w_row * out_features + w_col] += val[code_val * n_nz + i];
                    }
                }
            }
        }
    }

    w
}

/// Decode codes and apply scale for ESCHAM kernel pipeline.
/// Returns f32 weights scaled by rin ⊗ rout, ready for hadamard transform.
///
/// # Arguments
/// * `code` - flat int16 codes
/// * `codebook` - reference to the codebook
/// * `k` - codebook parameter (2 for gate_up, 3 for down)
/// * `in_features` - actual input dimension
/// * `out_features` - actual output dimension
/// * `rin` - per-input feature scale [in_features]
/// * `rout` - per-output feature scale [out_features]
///
/// # Returns
/// Dense fp32 weight matrix of shape (in_features, out_features) with scales applied
pub fn decode_and_scale_for_kernel(
    code: &[i16],
    codebook: &Codebook,
    k: usize,
    in_features: usize,
    out_features: usize,
    rin: &[f32],
    rout: &[f32],
) -> Vec<f32> {
    // Decode codes to f32 weights
    let w = decode_codes_to_weights(code, codebook, k, in_features, out_features);
    
    // Scale: w * rin ⊗ rout
    // rin has shape [in_features], rout has shape [out_features]
    let w_scaled: Vec<f32> = w
        .iter()
        .enumerate()
        .map(|(i, &val)| {
            let in_idx = i / out_features;
            let out_idx = i % out_features;
            val * rin[in_idx] * rout[out_idx]
        })
        .collect();
    
    w_scaled
}

/// Build the T128 matrix (128x128 normalized Walsh-Hadamard).
///
/// Uses Sylvester construction: H_1 = [1], H_{2n} = [H_n H_n; H_n -H_n].
/// Returns a flat vector of size 128*128 in row-major order.
/// Normalized by 1/sqrt(128).
fn build_t128() -> Vec<f32> {
    // Build 2D matrix using Sylvester construction
    let mut h: Vec<Vec<f32>> = vec![vec![1.0f32]];
    while h.len() < 128 {
        let n = h.len();
        let mut new_h = vec![vec![0.0f32; n * 2]; n * 2];
        for i in 0..n {
            for j in 0..n {
                new_h[i][j] = h[i][j];           // top-left
                new_h[i][j + n] = h[i][j];        // top-right
                new_h[i + n][j] = h[i][j];        // bottom-left
                new_h[i + n][j + n] = -h[i][j];   // bottom-right
            }
        }
        h = new_h;
    }
    
    // Flatten to 1D (row-major)
    let n = h.len();
    let mut flat = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            flat[i * n + j] = h[i][j];
        }
    }
    
    // Normalize
    let scale = 1.0 / (n as f32).sqrt();
    for v in flat.iter_mut() {
        *v *= scale;
    }
    
    flat
}

/// Build T128 scaled by rin: T128_rin = diag(rin) @ T128.
///
/// This allows us to compute T128(x * rin) as a single GEMV: x @ T128_rin.
///
/// # Arguments
/// * `rin` - input scales, length 128
///
/// # Returns
/// T128_rin matrix of shape (128, 128) in row-major
pub fn build_t128_rin(rin: &[f32]) -> Vec<f32> {
    let t128 = build_t128();
    let mut result = vec![0.0f32; 128 * 128];

    for i in 0..128 {
        for j in 0..128 {
            // T128_rin[i, j] = rin[i] * T128[i, j]
            result[i * 128 + j] = rin[i] * t128[i * 128 + j];
        }
    }

    result
}

/// Build T128 scaled by rout: T128_rout = diag(rout) @ T128.
///
/// This allows us to compute T128(y * rout) as a single GEMV: y @ T128_rout.
///
/// # Arguments
/// * `rout` - output scales, length 128
///
/// # Returns
/// T128_rout matrix of shape (128, 128) in row-major
pub fn build_t128_rout(rout: &[f32]) -> Vec<f32> {
    build_t128_rin(rout) // Same formula
}

/// Apply T128 transform to input (host-side, for verification).
///
/// # Arguments
/// * `x` - input array, length must be divisible by 128
///
/// # Returns
/// Transformed array of same length
pub fn apply_t128_host(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    assert!(n % 128 == 0, "Input length must be divisible by 128");

    let t128 = build_t128();
    let mut y = vec![0.0f32; n];
    let n_blocks = n / 128;

    for b in 0..n_blocks {
        let x_block = &x[b * 128..(b + 1) * 128];
        let y_block = &mut y[b * 128..(b + 1) * 128];

        for i in 0..128 {
            let mut sum = 0.0f32;
            for j in 0..128 {
                sum += t128[i * 128 + j] * x_block[j];
            }
            y_block[i] = sum;
        }
    }

    y
}

/// Apply SiLU activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
fn silu(x: f32) -> f32 {
    if x >= 0.0 {
        x / (1.0 + (-x).exp())
    } else {
        // For negative x, use numerically stable form: x * exp(x) / (1 + exp(x))
        // = x * (1 - 1/(1+exp(x)))
        // For large negative x, exp(x) ≈ 0, so silu(x) ≈ x * exp(x) ≈ 0
        let exp_x = x.exp();
        x * exp_x / (1.0 + exp_x)
    }
}

/// Full ESCHAM forward pass for one expert (Python reference algorithm).
///
/// Implements:
/// ```python
/// x_scaled = x * rin * s_in
/// xh = T128 @ x_scaled
/// gu = xh @ W_gate_up    # [2*mi,]
/// gate, up = gu.chunk(2)
/// gated = silu(gate) * up
/// y = gated @ W_down     # [dim,]
/// y = T128 @ (y * rout)
/// y = y * s_out
/// ```
///
/// # Arguments
/// * `x` - input activations, shape `(dim,)`
/// * `gate_up_code` - int16 codes for gate_up, flat array
/// * `down_code` - int16 codes for down, flat array
/// * `rin` - input scales, shape `(dim,)`
/// * `rout` - output scales, shape `(dim,)`
/// * `s_in` - per-feature input scales, shape `(dim,)`
/// * `s_out` - per-feature output scales, shape `(dim,)`
/// * `codebook` - reference to the codebook
/// * `dim` - hidden dimension
/// * `mi` - moe intermediate size
///
/// # Returns
/// Output tensor of shape `(dim,)`
pub fn escham_expert_forward(
    x: &[f32],
    gate_up_code: &[i16],
    down_code: &[i16],
    gu_rin: &[f32],
    gu_rout: &[f32],
    d_rin: &[f32],
    d_rout: &[f32],
    s_in: &[f32],
    s_out: &[f32],
    codebook: &Codebook,
    dim: usize,
    mi: usize,
) -> Vec<f32> {
    // gate_up: in=dim, out=2*mi, K=2
    let gate_up_out = decode_and_forward(
        x, gate_up_code, gu_rin, gu_rout, s_in, s_out,
        codebook, dim, mi * 2, 2,
    );

    // SiLU gate * up
    let gate = &gate_up_out[..mi];
    let up = &gate_up_out[mi..];
    let mut gated = vec![0.0f32; mi];
    for j in 0..mi {
        gated[j] = silu(gate[j]) * up[j];
    }

    // down: in=mi, out=dim, K=3
    let y = decode_and_forward(
        &gated, down_code, d_rin, d_rout, s_in, s_out,
        codebook, mi, dim, 3,
    );

    y
}

/// Decode codes and apply forward pass for one projection.
///
/// Implements: y = T128((x * rin * s_in) @ W) * rout * s_out
fn decode_and_forward(
    x: &[f32],
    code: &[i16],
    rin: &[f32],
    rout: &[f32],
    s_in: &[f32],
    s_out: &[f32],
    codebook: &Codebook,
    in_features: usize,
    out_features: usize,
    k: usize,
) -> Vec<f32> {
    // Decode codes -> fp32 weights
    let w = decode_codes_to_weights(code, codebook, k, in_features, out_features);

    // Scale input: x_scaled = x * rin * s_in
    let x_scaled: Vec<f32> = x
        .iter()
        .zip(rin.iter())
        .zip(s_in.iter())
        .map(|((xi, ri), si)| xi * ri * si)
        .collect();

    // Hadamard transform: xh = T128 @ x_scaled
    let xh = apply_t128_host(&x_scaled);

    // GEMV: y = xh @ W
    let mut y = vec![0.0f32; out_features];
    for o in 0..out_features {
        let mut sum = 0.0f32;
        for i in 0..in_features {
            sum += xh[i] * w[i * out_features + o];
        }
        y[o] = sum;
    }

    // Scale output: y = y * rout * s_out
    let y_final: Vec<f32> = y
        .iter()
        .zip(rout.iter())
        .zip(s_out.iter())
        .map(|((yi, ro), so)| yi * ro * so)
        .collect();

    y_final
}

/// Decode a single expert's gate_up or down projection from int16 codes.
///
/// # Arguments
/// * `code` - int16 codes, flat array of length (in_p//16) * (out_p//16) * 16 * K
/// * `codebook` - reference to the codebook
/// * `k` - codebook parameter (2 for gate_up, 3 for down)
/// * `in_features` - actual input dimension
/// * `out_features` - actual output dimension
///
/// # Returns
/// Dense fp32 weight matrix of shape (in_features, out_features) in row-major
#[allow(dead_code)]
pub fn decode_weights(
    code: &[i16],
    codebook: &Codebook,
    k: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    decode_codes_to_weights(code, codebook, k, in_features, out_features)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_trellis_k2_all_ones() {
        // A 16*K=32-element int16 payload of all zeros → every 16-bit window is 0.
        let packed = vec![0i16; 32];
        let out = unpack_trellis(&packed, 2);
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_unpack_trellis_k2_single_window() {
        // One tile, K=2, every int16 = 0x0001 → the circular 512-bit stream has
        // ones at positions ≡ 0 (mod 16). Each 16-bit window [b0, b0+15] at
        // stride 2 then contains exactly one set bit; its phase rotates with t:
        // windows are 2^((2 + 2i) mod 16) = 4,16,64,256,1024,4096,16384,1, …
        // (period 8, so each of the 8 values appears 32 times).
        let packed = vec![1i16; 32];
        let out = unpack_trellis(&packed, 2);
        assert_eq!(out.len(), 256);
        for (i, &v) in out.iter().enumerate() {
            let expect = 1u16 << ((2 + 2 * i) % 16);
            assert_eq!(v, expect, "window {i}: got {v:#06x}, want {expect:#06x}");
        }
    }

    #[test]
    fn test_decode_3inst_matches_python() {
        // Values from the higgs reference decode_3inst (formula):
        //   code 0 → 1.84375, code 1 → 0.134521484375, code 2 → -0.75927734375
        let codes = [0u16, 1, 2, 7];
        let out = decode_3inst(&codes);
        assert!((out[0] - 1.84375).abs() < 1e-4, "got {}", out[0]);
        assert!((out[1] - 0.134521484375).abs() < 1e-4, "got {}", out[1]);
        assert!((out[2] + 0.75927734375).abs() < 1e-4, "got {}", out[2]);
    }

    #[test]
    fn test_f16_to_f32() {
        // 1.0 = 0x3C00, -1.0 = 0xBC00, 0.5 = 0x3800
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
    }

    #[test]
    fn test_build_t128_orthogonality() {
        let t128 = build_t128();

        // Verify T @ T^T = I
        let mut identity = vec![0.0f32; 128 * 128];
        for i in 0..128 {
            for j in 0..128 {
                let mut sum = 0.0f32;
                for k in 0..128 {
                    sum += t128[i * 128 + k] * t128[j * 128 + k];
                }
                identity[i * 128 + j] = sum;
            }
        }

        for i in 0..128 {
            for j in 0..128 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (identity[i * 128 + j] - expected).abs() < 1e-5,
                    "T@T^T({},{}) = {}, expected {}",
                    i,
                    j,
                    identity[i * 128 + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_apply_t128_roundtrip() {
        let t128 = build_t128();

        // Create random input
        let x: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();

        // Apply T128
        let xh = apply_t128_host(&x);

        // Apply inverse T128 (same as T128 since orthogonal)
        let x_back = apply_t128_host(&xh);

        // Should recover original (up to numerical error)
        for i in 0..128 {
            assert!(
                (x[i] - x_back[i]).abs() < 1e-4,
                "Roundtrip failed at {}: x={}, x_back={}",
                i,
                x[i],
                x_back[i]
            );
        }
    }

    #[test]
    fn test_build_t128_rin() {
        let rin: Vec<f32> = (0..128).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let t128_rin = build_t128_rin(&rin);

        // Verify: T128_rin[i, j] = rin[i] * T128[i, j]
        let t128 = build_t128();
        for i in 0..128 {
            for j in 0..128 {
                let expected = rin[i] * t128[i * 128 + j];
                assert!(
                    (t128_rin[i * 128 + j] - expected).abs() < 1e-6,
                    "T128_rin({},{}) = {}, expected {}",
                    i,
                    j,
                    t128_rin[i * 128 + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_silu() {
        // silu(0) = 0
        assert!((silu(0.0)).abs() < 1e-6);
        // silu(x) -> x as x -> inf
        assert!((silu(100.0) - 100.0).abs() < 1e-3);
        // silu(x) -> 0 as x -> -inf
        assert!(silu(-100.0).abs() < 1e-6);
        // silu is smooth at 0
        assert!((silu(0.001) - 0.001 / 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_codebook_load_binary() {
        let path = std::path::Path::new("/home/mika/home-arch/codebook/compact.bin");
        if !path.exists() {
            eprintln!("SKIP: compact.bin not found at {:?}", path);
            return;
        }
        let cb = Codebook::load(path).expect("Failed to load codebook");
        assert_eq!(cb.k2_positions.len(), 32, "Expected 32 K2 slots");
        assert_eq!(cb.k2_values.len(), 32, "Expected 32 K2 value arrays");
        assert_eq!(cb.k3_positions.len(), 48, "Expected 48 K3 slots");
        assert_eq!(cb.k3_values.len(), 48, "Expected 48 K3 value arrays");
        // Verify K2 slot 0: n_nz=15 positions, shape (65536, 15) values
        assert_eq!(cb.k2_positions[0].len(), 30, "K2[0] positions should have 30 elements (15*2)");
        assert_eq!(cb.k2_values[0].len(), 65536 * 15, "K2[0] values should have 65536*15 elements");
        // Verify K3 slot 0: n_nz=10
        assert_eq!(cb.k3_positions[0].len(), 20, "K3[0] positions should have 20 elements (10*2)");
        assert_eq!(cb.k3_values[0].len(), 65536 * 10, "K3[0] values should have 65536*10 elements");
        // Verify values are reasonable (fp16 -> f32 range: roughly -65504 to 65504)
        let k2_max = cb.k2_values[0].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(k2_max < 70000.0, "K2 values out of fp16 range: max={}", k2_max);
        println!("Codebook loaded successfully: 32 K2 slots, 48 K3 slots");
    }
}
