// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Minimal GGUF reader + dequant copied from `crates/engine/src/{gguf.rs,llama.rs}`.
//! Self-contained so hipfire-quantize doesn't pull engine's GPU dependency tree.
//! TODO: factor into a shared `gguf-codec` crate.

use byteorder::{LittleEndian, ReadBytesExt};
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Cursor, Read};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x46554747;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    IQ1M = 29,
    BF16 = 30,
    Q1_0 = 41,
    Q2_0 = 42,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::IQ2XXS),
            17 => Some(Self::IQ2XS),
            18 => Some(Self::IQ3XXS),
            19 => Some(Self::IQ1S),
            20 => Some(Self::IQ4NL),
            21 => Some(Self::IQ3S),
            22 => Some(Self::IQ2S),
            23 => Some(Self::IQ4XS),
            29 => Some(Self::IQ1M),
            30 => Some(Self::BF16),
            41 => Some(Self::Q1_0),
            42 => Some(Self::Q2_0),
            _ => None,
        }
    }

    pub fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            Self::Q1_0 | Self::Q2_0 => 128,
            // I-quants: all QK_K = 256 except IQ4_NL (QK4_NL = 32).
            Self::IQ4NL => 32,
            Self::IQ2XXS | Self::IQ2XS | Self::IQ3XXS | Self::IQ1S | Self::IQ3S
            | Self::IQ2S | Self::IQ4XS | Self::IQ1M => 256,
        }
    }

    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 290,
            Self::Q1_0 => 18,
            Self::Q2_0 => 34,
            // From ggml-common.h static_asserts (QK_K = 256):
            // IQ2_XXS = 2 + 64 = 66; IQ2_XS = 2 + 64 + 8 = 74;
            // IQ3_XXS = 2 + 96 = 98; IQ3_S = 2 + 64 + 8 + 32 + 4 = 110;
            // IQ2_S = 2 + 64 + 8 + 8 = 82; IQ4_XS = 2 + 2 + 4 + 128 = 136.
            // IQ4_NL uses QK4_NL = 32: 2 + 16 = 18.
            Self::IQ2XXS => 66,
            Self::IQ2XS => 74,
            Self::IQ3XXS => 98,
            Self::IQ3S => 110,
            Self::IQ2S => 82,
            Self::IQ4XS => 136,
            Self::IQ4NL => 18,
            // IQ1_M (type 29): no per-block d; 32 + 16 + 8 = 56 B per 256
            // elements (1.75 bpw). The super-scale is reassembled from the
            // scales nibbles (see gguf_iq::dequant_iq1_m).
            Self::IQ1M => 56,
            // IQ1_S (type 19, 2 + 32 + 16 = 50) has no ported decoder yet —
            // fail closed with a named panic in tensor_to_f32 rather than a
            // wrong byte size here. (This file carries no IQ1_S tensors, but
            // the arm must stay explicit so a future file fails loudly.)
            Self::IQ1S => {
                panic!("GGUF type {:?} not yet ported (no decoder)", self)
            }
        }
    }

    pub fn tensor_bytes(self, n: usize) -> usize {
        let bs = self.block_size();
        let nblocks = (n + bs - 1) / bs;
        nblocks * self.block_bytes()
    }

    /// Human label for logs and error messages.
    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ3XXS => "IQ3_XXS",
            Self::IQ1S => "IQ1_S",
            Self::IQ4NL => "IQ4_NL",
            Self::IQ3S => "IQ3_S",
            Self::IQ2S => "IQ2_S",
            Self::IQ4XS => "IQ4_XS",
            Self::IQ1M => "IQ1_M",
            Self::BF16 => "BF16",
            Self::Q1_0 => "Q1_0",
            Self::Q2_0 => "Q2_0",
        }
    }
}

#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetaValue::U32(v) => Some(*v),
            MetaValue::I32(v) => Some(*v as u32),
            MetaValue::U64(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetaValue::F32(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::String(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: GgmlType,
    pub offset: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn byte_size(&self) -> usize {
        self.dtype.tensor_bytes(self.numel())
    }
}

pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_offset: usize,
    mmap: Mmap,
}

impl GgufFile {
    /// Test-only: metadata + tensor-table view with no backing file. The
    /// mmap is a zero-length map of an empty temp file (memmap2 handles
    /// zero-length files); config translation reads only `metadata` and
    /// `tensors`.
    #[cfg(test)]
    pub fn for_tests(
        metadata: HashMap<String, MetaValue>,
        tensors: Vec<TensorInfo>,
    ) -> io::Result<Self> {
        let file = tempfile::tempfile()?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self {
            version: 3,
            metadata,
            tensors,
            tensor_data_offset: 0,
            mmap,
        })
    }

    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut cursor = Cursor::new(&mmap[..]);

        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid GGUF magic: 0x{magic:08x}"),
            ));
        }

        let version = cursor.read_u32::<LittleEndian>()?;
        if version < 2 || version > 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported GGUF version: {version}"),
            ));
        }

        let tensor_count = cursor.read_u64::<LittleEndian>()? as usize;
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()? as usize;

        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = read_string(&mut cursor)?;
            let value = read_meta_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(cursor.read_u64::<LittleEndian>()? as usize);
            }
            let dtype_raw = cursor.read_u32::<LittleEndian>()?;
            let dtype = GgmlType::from_u32(dtype_raw).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown GGML type: {dtype_raw}"),
                )
            })?;
            let offset = cursor.read_u64::<LittleEndian>()? as usize;
            tensors.push(TensorInfo {
                name,
                shape,
                dtype,
                offset,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as usize;

        let pos = cursor.position() as usize;
        let tensor_data_offset = (pos + alignment - 1) / alignment * alignment;

        Ok(GgufFile {
            version,
            metadata,
            tensors,
            tensor_data_offset,
            mmap,
        })
    }

    pub fn tensor_data(&self, info: &TensorInfo) -> &[u8] {
        let start = self.tensor_data_offset + info.offset;
        let end = start + info.byte_size();
        &self.mmap[start..end]
    }

    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.meta(key).and_then(|v| v.as_u32())
    }
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.meta(key).and_then(|v| v.as_f32())
    }
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.meta(key).and_then(|v| v.as_str())
    }
}

fn read_string(cursor: &mut Cursor<&[u8]>) -> io::Result<String> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    String::from_utf8(buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("invalid UTF-8: {e}")))
}

fn read_meta_value(cursor: &mut Cursor<&[u8]>) -> io::Result<MetaValue> {
    let vtype = cursor.read_u32::<LittleEndian>()?;
    read_typed_value(cursor, vtype)
}

fn read_typed_value(cursor: &mut Cursor<&[u8]>, vtype: u32) -> io::Result<MetaValue> {
    match vtype {
        0 => Ok(MetaValue::U8(cursor.read_u8()?)),
        1 => Ok(MetaValue::I8(cursor.read_i8()?)),
        2 => Ok(MetaValue::U16(cursor.read_u16::<LittleEndian>()?)),
        3 => Ok(MetaValue::I16(cursor.read_i16::<LittleEndian>()?)),
        4 => Ok(MetaValue::U32(cursor.read_u32::<LittleEndian>()?)),
        5 => Ok(MetaValue::I32(cursor.read_i32::<LittleEndian>()?)),
        6 => Ok(MetaValue::F32(cursor.read_f32::<LittleEndian>()?)),
        7 => Ok(MetaValue::Bool(cursor.read_u8()? != 0)),
        8 => Ok(MetaValue::String(read_string(cursor)?)),
        9 => {
            let elem_type = cursor.read_u32::<LittleEndian>()?;
            let count = cursor.read_u64::<LittleEndian>()? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count {
                arr.push(read_typed_value(cursor, elem_type)?);
            }
            Ok(MetaValue::Array(arr))
        }
        10 => Ok(MetaValue::U64(cursor.read_u64::<LittleEndian>()?)),
        11 => Ok(MetaValue::I64(cursor.read_i64::<LittleEndian>()?)),
        12 => Ok(MetaValue::F64(cursor.read_f64::<LittleEndian>()?)),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown metadata value type: {vtype}"),
        )),
    }
}

// ─── Dequant (copied from engine/src/llama.rs) ────────────────────────────

/// `iq3s_grid[512]` from llama.cpp-escha `ggml/src/ggml-common.h` — the
/// IQ3_S 4-value codebook grid. Each u32 packs 4 signed-magnitude grid
/// values as bytes; the dequant indexes `[qs | qh-bit]` and multiplies by
/// the sign from `signs & kmask_iq2xs[j]`. Verbatim copy; do not hand-edit.
const IQ3S_GRID: [u32; 512] = [
    0x01010101u32, 0x01010103u32, 0x01010105u32, 0x0101010Bu32, 0x0101010Fu32, 0x01010301u32, 0x01010303u32, 0x01010305u32,
    0x01010309u32, 0x0101030Du32, 0x01010501u32, 0x01010503u32, 0x0101050Bu32, 0x01010707u32, 0x01010901u32, 0x01010905u32,
    0x0101090Bu32, 0x0101090Fu32, 0x01010B03u32, 0x01010B07u32, 0x01010D01u32, 0x01010D05u32, 0x01010F03u32, 0x01010F09u32,
    0x01010F0Fu32, 0x01030101u32, 0x01030103u32, 0x01030105u32, 0x01030109u32, 0x01030301u32, 0x01030303u32, 0x0103030Bu32,
    0x01030501u32, 0x01030507u32, 0x0103050Fu32, 0x01030703u32, 0x0103070Bu32, 0x01030909u32, 0x01030D03u32, 0x01030D0Bu32,
    0x01030F05u32, 0x01050101u32, 0x01050103u32, 0x0105010Bu32, 0x0105010Fu32, 0x01050301u32, 0x01050307u32, 0x0105030Du32,
    0x01050503u32, 0x0105050Bu32, 0x01050701u32, 0x01050709u32, 0x01050905u32, 0x0105090Bu32, 0x0105090Fu32, 0x01050B03u32,
    0x01050B07u32, 0x01050F01u32, 0x01050F07u32, 0x01070107u32, 0x01070303u32, 0x0107030Bu32, 0x01070501u32, 0x01070505u32,
    0x01070703u32, 0x01070707u32, 0x0107070Du32, 0x01070909u32, 0x01070B01u32, 0x01070B05u32, 0x01070D0Fu32, 0x01070F03u32,
    0x01070F0Bu32, 0x01090101u32, 0x01090307u32, 0x0109030Fu32, 0x01090503u32, 0x01090509u32, 0x01090705u32, 0x01090901u32,
    0x01090907u32, 0x01090B03u32, 0x01090F01u32, 0x010B0105u32, 0x010B0109u32, 0x010B0501u32, 0x010B0505u32, 0x010B050Du32,
    0x010B0707u32, 0x010B0903u32, 0x010B090Bu32, 0x010B090Fu32, 0x010B0D0Du32, 0x010B0F07u32, 0x010D010Du32, 0x010D0303u32,
    0x010D0307u32, 0x010D0703u32, 0x010D0B05u32, 0x010D0F03u32, 0x010F0101u32, 0x010F0105u32, 0x010F0109u32, 0x010F0501u32,
    0x010F0505u32, 0x010F050Du32, 0x010F0707u32, 0x010F0B01u32, 0x010F0B09u32, 0x03010101u32, 0x03010103u32, 0x03010105u32,
    0x03010109u32, 0x03010301u32, 0x03010303u32, 0x03010307u32, 0x0301030Bu32, 0x0301030Fu32, 0x03010501u32, 0x03010505u32,
    0x03010703u32, 0x03010709u32, 0x0301070Du32, 0x03010B09u32, 0x03010B0Du32, 0x03010D03u32, 0x03010F05u32, 0x03030101u32,
    0x03030103u32, 0x03030107u32, 0x0303010Du32, 0x03030301u32, 0x03030309u32, 0x03030503u32, 0x03030701u32, 0x03030707u32,
    0x03030903u32, 0x03030B01u32, 0x03030B05u32, 0x03030F01u32, 0x03030F0Du32, 0x03050101u32, 0x03050305u32, 0x0305030Bu32,
    0x0305030Fu32, 0x03050501u32, 0x03050509u32, 0x03050705u32, 0x03050901u32, 0x03050907u32, 0x03050B0Bu32, 0x03050D01u32,
    0x03050F05u32, 0x03070103u32, 0x03070109u32, 0x0307010Fu32, 0x03070301u32, 0x03070307u32, 0x03070503u32, 0x0307050Fu32,
    0x03070701u32, 0x03070709u32, 0x03070903u32, 0x03070D05u32, 0x03070F01u32, 0x03090107u32, 0x0309010Bu32, 0x03090305u32,
    0x03090309u32, 0x03090703u32, 0x03090707u32, 0x03090905u32, 0x0309090Du32, 0x03090B01u32, 0x03090B09u32, 0x030B0103u32,
    0x030B0301u32, 0x030B0307u32, 0x030B0503u32, 0x030B0701u32, 0x030B0705u32, 0x030B0B03u32, 0x030D0501u32, 0x030D0509u32,
    0x030D050Fu32, 0x030D0909u32, 0x030D090Du32, 0x030F0103u32, 0x030F0107u32, 0x030F0301u32, 0x030F0305u32, 0x030F0503u32,
    0x030F070Bu32, 0x030F0903u32, 0x030F0D05u32, 0x030F0F01u32, 0x05010101u32, 0x05010103u32, 0x05010107u32, 0x0501010Bu32,
    0x0501010Fu32, 0x05010301u32, 0x05010305u32, 0x05010309u32, 0x0501030Du32, 0x05010503u32, 0x05010507u32, 0x0501050Fu32,
    0x05010701u32, 0x05010705u32, 0x05010903u32, 0x05010907u32, 0x0501090Bu32, 0x05010B01u32, 0x05010B05u32, 0x05010D0Fu32,
    0x05010F01u32, 0x05010F07u32, 0x05010F0Bu32, 0x05030101u32, 0x05030105u32, 0x05030301u32, 0x05030307u32, 0x0503030Fu32,
    0x05030505u32, 0x0503050Bu32, 0x05030703u32, 0x05030709u32, 0x05030905u32, 0x05030B03u32, 0x05050103u32, 0x05050109u32,
    0x0505010Fu32, 0x05050503u32, 0x05050507u32, 0x05050701u32, 0x0505070Fu32, 0x05050903u32, 0x05050B07u32, 0x05050B0Fu32,
    0x05050F03u32, 0x05050F09u32, 0x05070101u32, 0x05070105u32, 0x0507010Bu32, 0x05070303u32, 0x05070505u32, 0x05070509u32,
    0x05070703u32, 0x05070707u32, 0x05070905u32, 0x05070B01u32, 0x05070D0Du32, 0x05090103u32, 0x0509010Fu32, 0x05090501u32,
    0x05090507u32, 0x05090705u32, 0x0509070Bu32, 0x05090903u32, 0x05090F05u32, 0x05090F0Bu32, 0x050B0109u32, 0x050B0303u32,
    0x050B0505u32, 0x050B070Fu32, 0x050B0901u32, 0x050B0B07u32, 0x050B0F01u32, 0x050D0101u32, 0x050D0105u32, 0x050D010Fu32,
    0x050D0503u32, 0x050D0B0Bu32, 0x050D0D03u32, 0x050F010Bu32, 0x050F0303u32, 0x050F050Du32, 0x050F0701u32, 0x050F0907u32,
    0x050F0B01u32, 0x07010105u32, 0x07010303u32, 0x07010307u32, 0x0701030Bu32, 0x0701030Fu32, 0x07010505u32, 0x07010703u32,
    0x07010707u32, 0x0701070Bu32, 0x07010905u32, 0x07010909u32, 0x0701090Fu32, 0x07010B03u32, 0x07010D07u32, 0x07010F03u32,
    0x07030103u32, 0x07030107u32, 0x0703010Bu32, 0x07030309u32, 0x07030503u32, 0x07030507u32, 0x07030901u32, 0x07030D01u32,
    0x07030F05u32, 0x07030F0Du32, 0x07050101u32, 0x07050305u32, 0x07050501u32, 0x07050705u32, 0x07050709u32, 0x07050B01u32,
    0x07070103u32, 0x07070301u32, 0x07070309u32, 0x07070503u32, 0x07070507u32, 0x0707050Fu32, 0x07070701u32, 0x07070903u32,
    0x07070907u32, 0x0707090Fu32, 0x07070B0Bu32, 0x07070F07u32, 0x07090107u32, 0x07090303u32, 0x0709030Du32, 0x07090505u32,
    0x07090703u32, 0x07090B05u32, 0x07090D01u32, 0x07090D09u32, 0x070B0103u32, 0x070B0301u32, 0x070B0305u32, 0x070B050Bu32,
    0x070B0705u32, 0x070B0909u32, 0x070B0B0Du32, 0x070B0F07u32, 0x070D030Du32, 0x070D0903u32, 0x070F0103u32, 0x070F0107u32,
    0x070F0501u32, 0x070F0505u32, 0x070F070Bu32, 0x09010101u32, 0x09010109u32, 0x09010305u32, 0x09010501u32, 0x09010509u32,
    0x0901050Fu32, 0x09010705u32, 0x09010903u32, 0x09010B01u32, 0x09010F01u32, 0x09030105u32, 0x0903010Fu32, 0x09030303u32,
    0x09030307u32, 0x09030505u32, 0x09030701u32, 0x0903070Bu32, 0x09030907u32, 0x09030B03u32, 0x09030B0Bu32, 0x09050103u32,
    0x09050107u32, 0x09050301u32, 0x0905030Bu32, 0x09050503u32, 0x09050707u32, 0x09050901u32, 0x09050B0Fu32, 0x09050D05u32,
    0x09050F01u32, 0x09070109u32, 0x09070303u32, 0x09070307u32, 0x09070501u32, 0x09070505u32, 0x09070703u32, 0x0907070Bu32,
    0x09090101u32, 0x09090105u32, 0x09090509u32, 0x0909070Fu32, 0x09090901u32, 0x09090F03u32, 0x090B010Bu32, 0x090B010Fu32,
    0x090B0503u32, 0x090B0D05u32, 0x090D0307u32, 0x090D0709u32, 0x090D0D01u32, 0x090F0301u32, 0x090F030Bu32, 0x090F0701u32,
    0x090F0907u32, 0x090F0B03u32, 0x0B010105u32, 0x0B010301u32, 0x0B010309u32, 0x0B010505u32, 0x0B010901u32, 0x0B010909u32,
    0x0B01090Fu32, 0x0B010B05u32, 0x0B010D0Du32, 0x0B010F09u32, 0x0B030103u32, 0x0B030107u32, 0x0B03010Bu32, 0x0B030305u32,
    0x0B030503u32, 0x0B030705u32, 0x0B030F05u32, 0x0B050101u32, 0x0B050303u32, 0x0B050507u32, 0x0B050701u32, 0x0B05070Du32,
    0x0B050B07u32, 0x0B070105u32, 0x0B07010Fu32, 0x0B070301u32, 0x0B07050Fu32, 0x0B070909u32, 0x0B070B03u32, 0x0B070D0Bu32,
    0x0B070F07u32, 0x0B090103u32, 0x0B090109u32, 0x0B090501u32, 0x0B090705u32, 0x0B09090Du32, 0x0B0B0305u32, 0x0B0B050Du32,
    0x0B0B0B03u32, 0x0B0B0B07u32, 0x0B0D0905u32, 0x0B0F0105u32, 0x0B0F0109u32, 0x0B0F0505u32, 0x0D010303u32, 0x0D010307u32,
    0x0D01030Bu32, 0x0D010703u32, 0x0D010707u32, 0x0D010D01u32, 0x0D030101u32, 0x0D030501u32, 0x0D03050Fu32, 0x0D030D09u32,
    0x0D050305u32, 0x0D050709u32, 0x0D050905u32, 0x0D050B0Bu32, 0x0D050D05u32, 0x0D050F01u32, 0x0D070101u32, 0x0D070309u32,
    0x0D070503u32, 0x0D070901u32, 0x0D09050Bu32, 0x0D090907u32, 0x0D090D05u32, 0x0D0B0101u32, 0x0D0B0107u32, 0x0D0B0709u32,
    0x0D0B0D01u32, 0x0D0D010Bu32, 0x0D0D0901u32, 0x0D0F0303u32, 0x0D0F0307u32, 0x0F010101u32, 0x0F010109u32, 0x0F01010Fu32,
    0x0F010501u32, 0x0F010505u32, 0x0F01070Du32, 0x0F010901u32, 0x0F010B09u32, 0x0F010D05u32, 0x0F030105u32, 0x0F030303u32,
    0x0F030509u32, 0x0F030907u32, 0x0F03090Bu32, 0x0F050103u32, 0x0F050109u32, 0x0F050301u32, 0x0F05030Du32, 0x0F050503u32,
    0x0F050701u32, 0x0F050B03u32, 0x0F070105u32, 0x0F070705u32, 0x0F07070Bu32, 0x0F070B07u32, 0x0F090103u32, 0x0F09010Bu32,
    0x0F090307u32, 0x0F090501u32, 0x0F090B01u32, 0x0F0B0505u32, 0x0F0B0905u32, 0x0F0D0105u32, 0x0F0D0703u32, 0x0F0F0101u32,
];

/// Sign-bit mask for one byte of IQ2/IQ3 signs: bit j selects element j.
const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// Dequantize IQ3_S (ggml type 21, 3.4375 bpw) to f32.
///
/// Port of `dequantize_row_iq3_s` from llama.cpp-escha
/// `ggml/src/ggml-quants.c`. Block layout for 256 elements (110 bytes):
/// `d` f16, `qs[64]`, `qh[8]`, `signs[32]`, `scales[4]` (nibbles, two db
/// scales per byte: `db = d * (1 + 2*nibble)`). Each 32-element half-block
/// decodes 8 grid lookups × 4 values: `grid = iq3s_grid[qs | qh-bit]`,
/// value = `db * grid_byte * sign`. Grid bytes are SIGNED-MAGNITUDE
/// codebook values (verbatim u8, NOT two's complement) — cast via `as i8.
///
/// Whole tiles move: this is a line-for-line port, not a re-derivation. Any
/// divergence from the C is a bug; verify with the unit test below against
/// vectors produced by the C decoder.
fn dequant_iq3_s(data: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 256;
    const BLK: usize = 110;
    let nblocks = n.div_ceil(QK);
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let base = b * BLK;
        if base + BLK > data.len() {
            break;
        }
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let qs = &data[base + 2..base + 66];
        let qh = &data[base + 66..base + 74];
        let signs = &data[base + 74..base + 106];
        let scales = &data[base + 106..base + 110];
        // Output cursor for this block (clamped at n by the idx guards).
        let mut y = b * QK;
        let mut qs_off = 0usize;
        let mut signs_off = 0usize;
        let mut qh_off = 0usize;
        // Mirrors the C exactly: ib32 = 0,2,4,6 over 8 sub-blocks of 32.
        // Each ib32 step consumes one scales byte (db1 = low nibble, db2 =
        // high nibble), 16 qs bytes, 8 signs bytes, 2 qh bytes, and emits 64
        // outputs (two l-blocks of 32). qs/signs advance 8/4 per l-block;
        // qh advances 2 per ib32 step.
        for ib in 0..4 {
            let sc = scales[ib];
            for half_q in 0..2 {
                let db = d
                    * (1.0
                        + 2.0
                            * (if half_q == 0 { sc & 0xf } else { sc >> 4 }) as f32);
                for l in 0..4 {
                    // qh byte selection matches the C: first l-block reads
                    // qh[0] of the pair, second reads qh[1].
                    let qh_byte = qh[qh_off + half_q];
                    let g1 = IQ3S_GRID[(qs[qs_off + 2 * l] as usize)
                        | (((qh_byte as usize) << ((8 - 2 * l) as usize)) & 256)];
                    let g2 = IQ3S_GRID[(qs[qs_off + 2 * l + 1] as usize)
                        | (((qh_byte as usize) << ((7 - 2 * l) as usize)) & 256)];
                    let s = signs[signs_off + l];
                    for j in 0..4 {
                        let v1 = ((g1 >> (8 * j)) & 0xff) as u8 as i8 as f32;
                        let v2 = ((g2 >> (8 * j)) & 0xff) as u8 as i8 as f32;
                        let idx1 = y + j;
                        let idx2 = y + 4 + j;
                        if idx1 < n {
                            out[idx1] = db * v1 * (if s & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 });
                        }
                        if idx2 < n {
                            out[idx2] =
                                db * v2 * (if s & KMASK_IQ2XS[4 + j] != 0 { -1.0 } else { 1.0 });
                        }
                    }
                    y += 8;
                }
                qs_off += 8;
                signs_off += 4;
            }
            qh_off += 2;
        }
    }
    out
}

fn dequant_q4_0(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 32;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let off = b * 18;
        if off + 18 > data.len() {
            break;
        }
        let scale = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        for j in 0..16 {
            let byte = data[off + 2 + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            let idx = b * block_size + j * 2;
            if idx < n {
                out[idx] = lo as f32 * scale;
            }
            if idx + 1 < n {
                out[idx + 1] = hi as f32 * scale;
            }
        }
    }
    out
}

fn dequant_q8_0(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 32;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let off = b * 34;
        if off + 34 > data.len() {
            break;
        }
        let scale = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        for j in 0..32 {
            let q = data[off + 2 + j] as i8 as f32;
            let idx = b * block_size + j;
            if idx < n {
                out[idx] = q * scale;
            }
        }
    }
    out
}

fn dequant_q2_0(data: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 128;
    const BLK: usize = 34;
    let mut out = Vec::with_capacity(n);
    let nblocks = n / QK;
    for b in 0..nblocks {
        let base = b * BLK;
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let qs = &data[base + 2..base + BLK];
        for j in 0..QK {
            let code = (qs[j / 4] >> ((j % 4) * 2)) & 0x03;
            out.push((code as i32 - 1) as f32 * d);
        }
    }
    out
}

fn dequant_q1_0(data: &[u8], n: usize) -> Vec<f32> {
    const BLK: usize = 18;
    const QK: usize = 128;
    let nblocks = (n + QK - 1) / QK;
    let mut out = Vec::with_capacity(n);
    for b in 0..nblocks {
        let base = b * BLK;
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let neg_d = -d;
        for j in 0..QK {
            if out.len() == n {
                break;
            }
            let byte = data[base + 2 + (j >> 3)];
            let bit = (byte >> (j & 7)) & 1;
            out.push(if bit == 1 { d } else { neg_d });
        }
    }
    out
}

fn dequant_q4_k(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 256;
    let block_bytes = 144;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let off = b * block_bytes;
        if off + block_bytes > data.len() {
            break;
        }
        let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[off + 2], data[off + 3]]));

        let sc_data = &data[off + 4..off + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for i in 0..4 {
            scales[i] = sc_data[i] & 63;
            mins[i] = sc_data[4 + i] & 63;
        }
        for i in 0..4 {
            scales[4 + i] = (sc_data[8 + i] & 0xF) | ((sc_data[i] >> 6) << 4);
            mins[4 + i] = (sc_data[8 + i] >> 4) | ((sc_data[4 + i] >> 6) << 4);
        }

        let qdata = &data[off + 16..off + 16 + 128];
        for group in 0..4 {
            let sb_even = group * 2;
            let sb_odd = group * 2 + 1;
            let sc_even = d * scales[sb_even] as f32;
            let m_even = dmin * mins[sb_even] as f32;
            let sc_odd = d * scales[sb_odd] as f32;
            let m_odd = dmin * mins[sb_odd] as f32;
            for l in 0..32 {
                let byte = qdata[group * 32 + l];
                let idx_even = b * block_size + group * 64 + l;
                let idx_odd = idx_even + 32;
                if idx_even < n {
                    out[idx_even] = (byte & 0x0F) as f32 * sc_even - m_even;
                }
                if idx_odd < n {
                    out[idx_odd] = ((byte >> 4) & 0x0F) as f32 * sc_odd - m_odd;
                }
            }
        }
    }
    out
}

fn dequant_q5_k(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 256;
    let block_bytes = 176;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let off = b * block_bytes;
        if off + block_bytes > data.len() {
            break;
        }
        let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[off + 2], data[off + 3]]));

        // 12-byte packed scales/mins — same layout as Q4_K
        let sc_data = &data[off + 4..off + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for i in 0..4 {
            scales[i] = sc_data[i] & 63;
            mins[i] = sc_data[4 + i] & 63;
        }
        for i in 0..4 {
            scales[4 + i] = (sc_data[8 + i] & 0xF) | ((sc_data[i] >> 6) << 4);
            mins[4 + i] = (sc_data[8 + i] >> 4) | ((sc_data[4 + i] >> 6) << 4);
        }

        // 32 bytes of high bits (1 bit per element), then 128 bytes of low nibbles
        let qh = &data[off + 16..off + 48];
        let ql = &data[off + 48..off + 176];

        for group in 0..4 {
            let sb_even = group * 2;
            let sb_odd = group * 2 + 1;
            let sc_even = d * scales[sb_even] as f32;
            let m_even = dmin * mins[sb_even] as f32;
            let sc_odd = d * scales[sb_odd] as f32;
            let m_odd = dmin * mins[sb_odd] as f32;
            for l in 0..32 {
                let byte = ql[group * 32 + l];
                let hbit = ((qh[l] >> group) & 1) as u8;
                let hbit2 = ((qh[l] >> (group + 4)) & 1) as u8;
                let idx_even = b * block_size + group * 64 + l;
                let idx_odd = idx_even + 32;
                if idx_even < n {
                    let q = ((byte & 0x0F) | (hbit << 4)) as f32;
                    out[idx_even] = q * sc_even - m_even;
                }
                if idx_odd < n {
                    let q = (((byte >> 4) & 0x0F) | (hbit2 << 4)) as f32;
                    out[idx_odd] = q * sc_odd - m_odd;
                }
            }
        }
    }
    out
}

fn dequant_q6_k(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 256;
    let block_bytes = 210;
    let nblocks = (n + block_size - 1) / block_size;
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let off = b * block_bytes;
        if off + block_bytes > data.len() {
            break;
        }
        let mut ql = &data[off..off + 128];
        let mut qh = &data[off + 128..off + 192];
        let mut sc = &data[off + 192..off + 208];
        let d = f16_to_f32(u16::from_le_bytes([data[off + 208], data[off + 209]]));
        let base = b * block_size;
        for group in 0..2 {
            let y_off = base + group * 128;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                let idx0 = y_off + l;
                let idx1 = y_off + l + 32;
                let idx2 = y_off + l + 64;
                let idx3 = y_off + l + 96;
                if idx0 < n {
                    out[idx0] = d * sc[is] as i8 as f32 * q1 as f32;
                }
                if idx1 < n {
                    out[idx1] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                }
                if idx2 < n {
                    out[idx2] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                }
                if idx3 < n {
                    out[idx3] = d * sc[is + 6] as i8 as f32 * q4 as f32;
                }
            }
            ql = &ql[64..];
            qh = &qh[32..];
            sc = &sc[8..];
        }
    }
    out
}

/// Dispatcher: dequantize any supported tensor to f32. Panics on unsupported types.
pub fn tensor_to_f32(info: &TensorInfo, data: &[u8]) -> Vec<f32> {
    let n = info.numel();
    match info.dtype {
        GgmlType::F32 => {
            let mut out = vec![0.0f32; n];
            for (i, chunk) in data.chunks_exact(4).enumerate().take(n) {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            out
        }
        GgmlType::F16 => {
            let mut out = vec![0.0f32; n];
            for (i, chunk) in data.chunks_exact(2).enumerate().take(n) {
                out[i] = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            out
        }
        GgmlType::BF16 => {
            let mut out = vec![0.0f32; n];
            for (i, chunk) in data.chunks_exact(2).enumerate().take(n) {
                out[i] = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            out
        }
        GgmlType::Q4_0 => dequant_q4_0(data, n),
        GgmlType::Q8_0 => dequant_q8_0(data, n),
        GgmlType::Q2_0 => dequant_q2_0(data, n),
        GgmlType::Q1_0 => dequant_q1_0(data, n),
        GgmlType::Q4K => dequant_q4_k(data, n),
        GgmlType::Q5K => dequant_q5_k(data, n),
        GgmlType::Q6K => dequant_q6_k(data, n),
        GgmlType::Q2K => crate::gguf_iq::dequant_q2_k(data, n),
        GgmlType::IQ3S => dequant_iq3_s(data, n),
        GgmlType::IQ2XXS => crate::gguf_iq::dequant_iq2_xxs(data, n),
        GgmlType::IQ2XS => crate::gguf_iq::dequant_iq2_xs(data, n),
        GgmlType::IQ3XXS => crate::gguf_iq::dequant_iq3_xxs(data, n),
        GgmlType::IQ2S => crate::gguf_iq::dequant_iq2_s(data, n),
        GgmlType::IQ4XS => crate::gguf_iq::dequant_iq4_xs(data, n),
        GgmlType::IQ1M => crate::gguf_iq::dequant_iq1_m(data, n),
        other => panic!(
            "GGUF tensor type {:?} not implemented (tensor: {})",
            other, info.name
        ),
    }
}

#[cfg(test)]
mod q2_0_tests {
    use super::*;
    #[test]
    fn q2_0_type_params() {
        let t = GgmlType::from_u32(42).expect("ggml_type 42 = Q2_0");
        assert_eq!(t, GgmlType::Q2_0);
        assert_eq!(t.block_size(), 128);
        assert_eq!(t.block_bytes(), 34); // 2 (FP16 d) + 32 (2-bit x 128)
    }

    #[test]
    fn q2_0_dequant_matches_formula() {
        let mut block = vec![0u8; 34];
        block[0] = 0x00;
        block[1] = 0x40; // FP16 2.0 little-endian
        block[2] = 0xE4; // codes 0,1,2,3 (LSB-first)
        let out = dequant_q2_0(&block, 128);
        assert_eq!(&out[0..4], &[-2.0, 0.0, 2.0, 4.0]); // (code-1)*d
        assert!(out[4..128].iter().all(|&x| x == -2.0)); // code 0 -> -d
        assert_eq!(out.len(), 128);
    }
}

#[cfg(test)]
mod q1_0_tests {
    use super::*;

    #[test]
    fn q1_0_ggml_type_reads() {
        let t = GgmlType::from_u32(41).expect("ggml_type 41 = Q1_0");
        assert_eq!(t, GgmlType::Q1_0);
        assert_eq!(t.block_size(), 128);
        assert_eq!(t.block_bytes(), 18);
    }

    #[test]
    fn dequant_q1_0_sign_only() {
        // One block: d=0.5, all-ones bits -> all +0.5; first bit cleared -> element 0 = -0.5.
        // FP16 0.5 = 0x3800 little-endian ([0x00, 0x38]); no `half` crate dep in this
        // workspace, so the bit pattern is constructed directly (same style as the
        // q2_0_dequant_matches_formula test above).
        let mut blk = vec![0u8; 18];
        blk[0] = 0x00;
        blk[1] = 0x38; // FP16 0.5, little-endian
        for b in blk[2..18].iter_mut() {
            *b = 0xFF; // all bits set -> all +d
        }
        let all_pos = dequant_q1_0(&blk, 128);
        assert_eq!(all_pos.len(), 128);
        assert!(all_pos.iter().all(|&v| (v - 0.5).abs() < 1e-3));

        blk[2] &= !1u8; // clear bit 0 of first qs byte -> element 0 = -d
        let mixed = dequant_q1_0(&blk, 128);
        assert!((mixed[0] + 0.5).abs() < 1e-3);
        assert!((mixed[1] - 0.5).abs() < 1e-3);
    }
}

#[cfg(test)]
mod iq3_s_tests {
    use super::*;

    #[test]
    fn iq3_s_type_params() {
        let t = GgmlType::from_u32(21).expect("ggml_type 21 = IQ3_S");
        assert_eq!(t, GgmlType::IQ3S);
        assert_eq!(t.block_size(), 256);
        // d(2) + qs(64) + qh(8) + signs(32) + scales(4) = 110.
        assert_eq!(t.block_bytes(), 110);
        assert_eq!(t.label(), "IQ3_S");
    }

    /// Hand-traceable block: d = 1.0, all scales nibbles = 0 (db = 1.0),
    /// all qs/qh = 0 (grid[0] = 0x01010101 -> bytes [1,1,1,1]), all signs
    /// clear (positive). Every output must be exactly +1.0.
    #[test]
    fn iq3_s_dequant_all_ones_grid() {
        let mut blk = vec![0u8; 110];
        blk[0] = 0x00;
        blk[1] = 0x3C; // FP16 1.0 little-endian
                        // qs/qh/signs/scales all zero by construction.
        let out = dequant_iq3_s(&blk, 256);
        assert_eq!(out.len(), 256);
        assert!(
            out.iter().all(|&v| (v - 1.0).abs() < 1e-6),
            "first values: {:?}",
            &out[..8]
        );
    }

    /// Sign application: same block but signs[0] = 0xFF flips the first
    /// eight outputs (elements 0..8 read signs[0] bits 0..7).
    #[test]
    fn iq3_s_dequant_sign_flip() {
        let mut blk = vec![0u8; 110];
        blk[0] = 0x00;
        blk[1] = 0x3C; // FP16 1.0
        blk[74] = 0xFF; // signs[0]: flip elements 0..7
        let out = dequant_iq3_s(&blk, 256);
        assert!(out[0..8].iter().all(|&v| (v + 1.0).abs() < 1e-6));
        assert!(out[8..16].iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    /// Scale nibbles: scales[0] = 0x10 -> db1 = d*(1+0) = 1.0 for the first
    /// 32 outputs, db2 = d*(1+2) = 3.0 for outputs 32..64.
    #[test]
    fn iq3_s_dequant_scale_nibbles() {
        let mut blk = vec![0u8; 110];
        blk[0] = 0x00;
        blk[1] = 0x3C; // FP16 1.0
        blk[106] = 0x10; // low nibble 0, high nibble 1
        let out = dequant_iq3_s(&blk, 256);
        assert!((out[0] - 1.0).abs() < 1e-6, "db1 path: {}", out[0]);
        assert!((out[32] - 3.0).abs() < 1e-6, "db2 path: {}", out[32]);
    }

    /// Grid indexing: qs[0] = 1 selects grid[1] = 0x01010103 -> first value
    /// byte 0x03 = 3. Output[0] must be 3.0 (db = 1.0, sign +).
    #[test]
    fn iq3_s_dequant_grid_index() {
        let mut blk = vec![0u8; 110];
        blk[0] = 0x00;
        blk[1] = 0x3C; // FP16 1.0
        blk[2] = 0x01; // qs[0] = 1 -> grid1 = iq3s_grid[1], byte0 = 0x03
        let out = dequant_iq3_s(&blk, 256);
        assert!((out[0] - 3.0).abs() < 1e-6, "got {}", out[0]);
    }
}
