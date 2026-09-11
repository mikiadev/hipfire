// Standalone: verify gemv_iq2_xxs on the engine's exact [M,K] orientation.
// Loads the .hfq, finds an IQ2XXS 2D tensor, dequantizes with the .hfq
// shape, compares GPU GEMV + batched GEMM vs CPU.
use hipfire_runtime::hfq::HfqFile;
use std::path::Path;
use rdna_compute::DType;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args[1].clone();
    let hfq = HfqFile::open(Path::new(&path)).expect("open hfq");
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    eprintln!("GPU {}", gpu.arch);

    let mut tested = 0;
    for t in hfq.tensors() {
        if t.quant_type != 55 { continue; } // IQ2XXS
        if t.shape.len() != 2 { continue; }
        let m = t.shape[0] as usize; let k = t.shape[1] as usize;
        if k % 256 != 0 { continue; }
        let Some((_, raw)) = hfq.tensor_data(&t.name) else { continue; };
        eprintln!("tensor {} [{} x {}] bytes={}", t.name, m, k, raw.len());
        // CPU dequant with .hfq [M,K] orientation (row-major M rows of K)
        let a_f32 = dequant_iq2_xxs(raw, m * k);
        let x_data: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
        let mut y_ref = vec![0.0f32; m];
        for i in 0..m {
            let mut sum = 0.0f32;
            for j in 0..k { sum += a_f32[i * k + j] * x_data[j]; }
            y_ref[i] = sum;
        }
        let d_raw = gpu.upload_raw(raw, &[raw.len()]).unwrap();
        let d_x = gpu.upload_f32(&x_data, &[k]).unwrap();
        let d_y = gpu.zeros(&[m], DType::F32).unwrap();
        gpu.gemv_iq2_xxs(&d_raw, &d_x, &d_y, m, k).unwrap();
        let y_gpu = gpu.download_f32(&d_y).unwrap();
        let mut max_err = 0.0f32; let mut errs = 0;
        for i in 0..m {
            let e = (y_gpu[i] - y_ref[i]).abs();
            max_err = max_err.max(e);
            if e > 1e-3 { errs += 1; if errs <= 3 { eprintln!("  row {} gpu={} ref={}", i, y_gpu[i], y_ref[i]); } }
        }
        eprintln!("  GEMV [{} x {}] max_err={} errs={}/{}", m, k, max_err, errs, m);
        // batched
        let d_xb = gpu.upload_f32(&x_data, &[1, k]).unwrap();
        let d_yb = gpu.zeros(&[1, m], DType::F32).unwrap();
        gpu.gemm_iq2_xxs_batched(&d_raw, &d_xb, &d_yb, m, k, 1).unwrap();
        let yb = gpu.download_f32(&d_yb).unwrap();
        let mut bmax = 0.0f32;
        for i in 0..m { bmax = bmax.max((yb[i] - y_ref[i]).abs()); }
        eprintln!("  GEMM batch1 max_err={}", bmax);
        gpu.free_tensor(d_raw).unwrap(); gpu.free_tensor(d_x).unwrap(); gpu.free_tensor(d_y).unwrap();
        tested += 1;
        if tested >= 20 { break; }
    }
    eprintln!("tested {}", tested);
}

// IQ2_XXS decoder (from gguf_input.rs dequant_iq2_xxs)
const IQ2XXS_GRID: [u64; 256] = [
    0x0808080808080808u64, 0x080808080808082Bu64, 0x0808080808081919u64, 0x0808080808082B08u64,
    0x0808080808082B2Bu64, 0x0808080808190819u64, 0x0808080808191908u64, 0x08080808082B0808u64,
    0x08080808082B082Bu64, 0x08080808082B2B08u64, 0x08080808082B2B2Bu64, 0x0808080819080819u64,
    0x0808080819081908u64, 0x0808080819190808u64, 0x0808080819192B08u64, 0x08080808192B0819u64,
    0x08080808192B1908u64, 0x080808082B080808u64, 0x080808082B08082Bu64, 0x080808082B082B2Bu64,
    0x080808082B2B082Bu64, 0x0808081908080819u64, 0x0808081908081908u64, 0x0808081908190808u64,
    0x0808081908191919u64, 0x0808081919080808u64, 0x080808192B081908u64, 0x080808192B192B08u64,
    0x0808082B08080808u64, 0x0808082B0808082Bu64, 0x0808082B082B082Bu64, 0x0808082B2B08082Bu64,
    0x0808190808080819u64, 0x0808190808081908u64, 0x0808190808190808u64, 0x08081908082B0819u64,
    0x08081908082B1908u64, 0x0808190819080808u64, 0x080819081908082Bu64, 0x0808190819082B08u64,
    0x08081908192B0808u64, 0x080819082B080819u64, 0x080819082B081908u64, 0x080819082B190808u64,
    0x080819082B2B1908u64, 0x0808191908080808u64, 0x080819190808082Bu64, 0x0808191908082B08u64,
    0x08081919082B0808u64, 0x080819191908192Bu64, 0x08081919192B2B19u64, 0x080819192B080808u64,
    0x080819192B190819u64, 0x0808192B08082B19u64, 0x0808192B08190808u64, 0x0808192B19080808u64,
    0x0808192B2B081908u64, 0x0808192B2B2B1908u64, 0x08082B0808080808u64, 0x08082B0808081919u64,
    0x08082B0808082B08u64, 0x08082B0808191908u64, 0x08082B08082B2B08u64, 0x08082B0819080819u64,
    0x08082B0819081908u64, 0x08082B0819190808u64, 0x08082B081919082Bu64, 0x08082B082B082B08u64,
    0x08082B1908081908u64, 0x08082B1919080808u64, 0x08082B2B0808082Bu64, 0x08082B2B08191908u64,
    0x0819080808080819u64, 0x0819080808081908u64, 0x0819080808190808u64, 0x08190808082B0819u64,
    0x0819080819080808u64, 0x08190808192B0808u64, 0x081908082B081908u64, 0x081908082B190808u64,
    0x081908082B191919u64, 0x0819081908080808u64, 0x0819081908082B08u64, 0x08190819082B0808u64,
    0x0819081919190808u64, 0x0819081919192B2Bu64, 0x081908192B080808u64, 0x0819082B082B1908u64,
    0x0819082B19081919u64, 0x0819190808080808u64, 0x0819190808082B08u64, 0x08191908082B0808u64,
    0x08191908082B1919u64, 0x0819190819082B19u64, 0x081919082B080808u64, 0x0819191908192B08u64,
    0x08191919192B082Bu64, 0x0819192B08080808u64, 0x0819192B0819192Bu64, 0x08192B0808080819u64,
    0x08192B0808081908u64, 0x08192B0808190808u64, 0x08192B0819080808u64, 0x08192B082B080819u64,
    0x08192B1908080808u64, 0x08192B1908081919u64, 0x08192B192B2B0808u64, 0x08192B2B19190819u64,
    0x082B080808080808u64, 0x082B08080808082Bu64, 0x082B080808082B2Bu64, 0x082B080819081908u64,
    0x082B0808192B0819u64, 0x082B08082B080808u64, 0x082B08082B08082Bu64, 0x082B0819082B2B19u64,
    0x082B081919082B08u64, 0x082B082B08080808u64, 0x082B082B0808082Bu64, 0x082B190808080819u64,
    0x082B190808081908u64, 0x082B190808190808u64, 0x082B190819080808u64, 0x082B19081919192Bu64,
    0x082B191908080808u64, 0x082B191919080819u64, 0x082B1919192B1908u64, 0x082B192B2B190808u64,
    0x082B2B0808082B08u64, 0x082B2B08082B0808u64, 0x082B2B082B191908u64, 0x082B2B2B19081908u64,
    0x1908080808080819u64, 0x1908080808081908u64, 0x1908080808190808u64, 0x1908080808192B08u64,
    0x19080808082B0819u64, 0x19080808082B1908u64, 0x1908080819080808u64, 0x1908080819082B08u64,
    0x190808081919192Bu64, 0x19080808192B0808u64, 0x190808082B080819u64, 0x190808082B081908u64,
    0x190808082B190808u64, 0x1908081908080808u64, 0x19080819082B0808u64, 0x19080819192B0819u64,
    0x190808192B080808u64, 0x190808192B081919u64, 0x1908082B08080819u64, 0x1908082B08190808u64,
    0x1908082B19082B08u64, 0x1908082B1919192Bu64, 0x1908082B192B2B08u64, 0x1908190808080808u64,
    0x1908190808082B08u64, 0x19081908082B0808u64, 0x190819082B080808u64, 0x190819082B192B19u64,
    0x190819190819082Bu64, 0x19081919082B1908u64, 0x1908192B08080808u64, 0x19082B0808080819u64,
    0x19082B0808081908u64, 0x19082B0808190808u64, 0x19082B0819080808u64, 0x19082B0819081919u64,
    0x19082B1908080808u64, 0x19082B1919192B08u64, 0x19082B19192B0819u64, 0x19082B192B08082Bu64,
    0x19082B2B19081919u64, 0x19082B2B2B190808u64, 0x1919080808080808u64, 0x1919080808082B08u64,
    0x1919080808190819u64, 0x1919080808192B19u64, 0x19190808082B0808u64, 0x191908082B080808u64,
    0x191908082B082B08u64, 0x1919081908081908u64, 0x191908191908082Bu64, 0x191908192B2B1908u64,
    0x1919082B2B190819u64, 0x191919082B190808u64, 0x191919082B19082Bu64, 0x1919191908082B2Bu64,
    0x1919192B08080819u64, 0x1919192B19191908u64, 0x19192B0808080808u64, 0x19192B0808190819u64,
    0x19192B0808192B19u64, 0x19192B08192B1908u64, 0x19192B1919080808u64, 0x19192B2B08082B08u64,
    0x192B080808081908u64, 0x192B080808190808u64, 0x192B080819080808u64, 0x192B0808192B2B08u64,
    0x192B081908080808u64, 0x192B081919191919u64, 0x192B082B08192B08u64, 0x192B082B192B0808u64,
    0x192B190808080808u64, 0x192B190808081919u64, 0x192B191908190808u64, 0x192B19190819082Bu64,
    0x192B19192B081908u64, 0x192B2B081908082Bu64, 0x2B08080808080808u64, 0x2B0808080808082Bu64,
    0x2B08080808082B2Bu64, 0x2B08080819080819u64, 0x2B0808082B08082Bu64, 0x2B08081908081908u64,
    0x2B08081908192B08u64, 0x2B08081919080808u64, 0x2B08082B08190819u64, 0x2B08190808080819u64,
    0x2B08190808081908u64, 0x2B08190808190808u64, 0x2B08190808191919u64, 0x2B08190819080808u64,
    0x2B081908192B0808u64, 0x2B08191908080808u64, 0x2B0819191908192Bu64, 0x2B0819192B191908u64,
    0x2B08192B08082B19u64, 0x2B08192B19080808u64, 0x2B08192B192B0808u64, 0x2B082B080808082Bu64,
    0x2B082B1908081908u64, 0x2B082B2B08190819u64, 0x2B19080808081908u64, 0x2B19080808190808u64,
    0x2B190808082B1908u64, 0x2B19080819080808u64, 0x2B1908082B2B0819u64, 0x2B1908190819192Bu64,
    0x2B1908192B080808u64, 0x2B19082B19081919u64, 0x2B19190808080808u64, 0x2B191908082B082Bu64,
    0x2B19190819081908u64, 0x2B19191919190819u64, 0x2B192B082B080819u64, 0x2B192B19082B0808u64,
    0x2B2B08080808082Bu64, 0x2B2B080819190808u64, 0x2B2B08082B081919u64, 0x2B2B081908082B19u64,
    0x2B2B082B08080808u64, 0x2B2B190808192B08u64, 0x2B2B2B0819190808u64, 0x2B2B2B1908081908u64
];

const KSIGNS_IQ2XS: [u8; 128] = [
    0u8, 129u8, 130u8, 3u8, 132u8, 5u8, 6u8, 135u8, 136u8, 9u8, 10u8, 139u8, 12u8, 141u8, 142u8, 15u8,
    144u8, 17u8, 18u8, 147u8, 20u8, 149u8, 150u8, 23u8, 24u8, 153u8, 154u8, 27u8, 156u8, 29u8, 30u8, 159u8,
    160u8, 33u8, 34u8, 163u8, 36u8, 165u8, 166u8, 39u8, 40u8, 169u8, 170u8, 43u8, 172u8, 45u8, 46u8, 175u8,
    48u8, 177u8, 178u8, 51u8, 180u8, 53u8, 54u8, 183u8, 184u8, 57u8, 58u8, 187u8, 60u8, 189u8, 190u8, 63u8,
    192u8, 65u8, 66u8, 195u8, 68u8, 197u8, 198u8, 71u8, 72u8, 201u8, 202u8, 75u8, 204u8, 77u8, 78u8, 207u8,
    80u8, 209u8, 210u8, 83u8, 212u8, 85u8, 86u8, 215u8, 216u8, 89u8, 90u8, 219u8, 92u8, 221u8, 222u8, 95u8,
    96u8, 225u8, 226u8, 99u8, 228u8, 101u8, 102u8, 231u8, 232u8, 105u8, 106u8, 235u8, 108u8, 237u8, 238u8, 111u8,
    240u8, 113u8, 114u8, 243u8, 116u8, 245u8, 246u8, 119u8, 120u8, 249u8, 250u8, 123u8, 252u8, 125u8, 126u8, 255u8
];

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let frac = h & 0x3ff;
    if exp == 0 {
        if frac == 0 { return f32::from_bits((sign as u32) << 31); }
        let mut f = frac;
        let mut e = exp as i32;
        while f & 0x400 == 0 { f <<= 1; e -= 1; }
        f &= 0x3ff;
        e = 127 - 15 + 1 + e;
        return f32::from_bits((sign as u32) << 31 | (e as u32) << 23 | (f as u32) << 13);
    }
    if exp == 31 { return f32::from_bits((sign as u32) << 31 | 0x7f800000u32 | (frac as u32) << 13); }
    f32::from_bits((sign as u32) << 31 | ((exp as u32 + 127 - 15) << 23) | (frac as u32) << 13)
}

fn dequant_iq2_xxs(data: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 256;
    const BLK: usize = 66;
    let nblocks = n.div_ceil(QK);
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let base = b * BLK;
        if base + BLK > data.len() { break; }
        let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
        let qs = &data[base + 2..base + 66];
        let mut y = b * QK;
        for ib32 in 0..8 {
            let off = 8 * ib32;
            let a1 = u32::from_le_bytes([qs[off + 4], qs[off + 5], qs[off + 6], qs[off + 7]]);
            let aux8 = [qs[off], qs[off + 1], qs[off + 2], qs[off + 3], qs[off + 4], qs[off + 5], qs[off + 6], qs[off + 7]];
            let db = d * (0.5 + ((a1 >> 28) as f32)) * 0.25;
            for l in 0..4 {
                let grid = IQ2XXS_GRID[aux8[l] as usize];
                let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
                let gb = grid.to_le_bytes();
                for j in 0..8 {
                    let idx = y + j;
                    if idx < n {
                        out[idx] = db * (gb[j] as i8 as f32) * (if signs & (1u8 << j) != 0 { -1.0 } else { 1.0 });
                    }
                }
                y += 8;
            }
        }
    }
    out
}
