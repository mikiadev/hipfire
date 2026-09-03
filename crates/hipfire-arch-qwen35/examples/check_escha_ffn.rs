//! Throwaway GPU check: production escha FFN cache path (decode_and_fold +
//! scale absorb + f16) vs host reference for one (layer, expert, proj).
//! usage: check_escha_ffn <layer> <expert>
use hipfire_arch_qwen35::escham_decode as escham;
use rdna_compute::DType;
use rdna_compute::Gpu;

fn read_i16(src: &dyn hipfire_runtime::model_source::ModelSource, name: &str) -> Vec<i16> {
    let (info, data) = src.tensor_data(name).expect(name);
    assert_eq!(info.dtype, "I16");
    data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
}
fn read_f16(src: &dyn hipfire_runtime::model_source::ModelSource, name: &str) -> Vec<f32> {
    let (info, data) = src.tensor_data(name).expect(name);
    assert_eq!(info.dtype, "F16");
    data.chunks_exact(2).map(|c| hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
}
fn read_f32(src: &dyn hipfire_runtime::model_source::ModelSource, name: &str) -> Vec<f32> {
    let (info, data) = src.tensor_data(name).expect(name);
    assert_eq!(info.dtype, "F32");
    data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let lyr: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let exp: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let Some(mut gpu) = Gpu::init().ok() else { eprintln!("no gpu"); return; };
    let src = hipfire_runtime::safetensors_source::SafetensorsSource::open(std::path::Path::new("/data/rocmfpx/Escha-W2")).unwrap();
    let mp = "model.language_model";
    let dim = 2048usize;
    let mi = 512usize;

    let gu_prefix = format!("{mp}.layers.{lyr}.mlp.experts.gate_up_proj");
    let dn_prefix = format!("{mp}.layers.{lyr}.mlp.experts.down_proj");

    // gate_up: codes [128,64,32], k=2
    #[allow(dead_code)]
let _ = dn_prefix;
let gcode_all = read_i16(&src, &format!("{gu_prefix}.escha_code"));
    let gcode_e = gcode_all[exp*128*64*32..(exp+1)*128*64*32].to_vec();
    let grin_all = read_f16(&src, &format!("{gu_prefix}.escha_rin"));
    let grout_all = read_f16(&src, &format!("{gu_prefix}.escha_rout"));
    let gsin_all = read_f32(&src, &format!("{gu_prefix}.escha_s_in"));
    let grin_e = grin_all[exp*dim..(exp+1)*dim].to_vec();
    let grout_e = grout_all[exp*mi*2..(exp+1)*mi*2].to_vec();
    let gsin_e = gsin_all[exp*dim..(exp+1)*dim].to_vec();

    // GPU production path: decode+fold(rout=ones) -> absorb rout*in_scale -> f16
    let codes_gpu = gpu.upload_raw(unsafe{std::slice::from_raw_parts(gcode_e.as_ptr() as *const u8, gcode_e.len()*2)}, &[gcode_e.len()]).unwrap();
    let folded = hipfire_arch_qwen35::qwen35::escha_ffn::decode_and_fold_expert_gpu(&mut gpu, &codes_gpu, 2, dim, mi*2).unwrap();
    let in_scale: Vec<f32> = grin_e.iter().zip(gsin_e.iter()).map(|(r,s)| r*s).collect();
    let rout_g = gpu.upload_f32(&grout_e, &[mi*2]).unwrap();
    let inscale_g = gpu.upload_f32(&in_scale, &[dim]).unwrap();
    rdna_compute::escham::escham_apply_rowcol_scales_f32(&mut gpu, &folded, &rout_g, &inscale_g, mi*2, dim).unwrap();
    let folded_f16 = gpu.zeros(&[mi*2, dim], DType::F16).unwrap();
    rdna_compute::escham::escham_f32_to_f16(&mut gpu, &folded_f16, &folded).unwrap();
    // gemv on x ~ N(0,1)
    let mut rng = 777u64;
    let mut rand = move || { rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((rng>>33) as f32)/(1u64<<31) as f32 - 1.0 };
    let x: Vec<f32> = (0..dim).map(|_| rand()).collect();
    let x_g = gpu.upload_f32(&x, &[dim]).unwrap();
    let y_g = gpu.zeros(&[mi*2], DType::F32).unwrap();
    // grouped gemv single-expert path via f16 weight ptr table
    let wptr = folded_f16.buf.as_ptr() as u64;
    let wb: Vec<u8> = wptr.to_ne_bytes().to_vec();
    let wpt = gpu.upload_raw(&wb, &[2]).unwrap();
    rdna_compute::escham::escham_moe_grouped_gemv_f16(&mut gpu, &wpt, &x_g, &y_g, mi*2, dim, 1, 0).unwrap();
    let y_gpu = gpu.download_f32(&y_g).unwrap();

    // Host reference (correct): fold WITH rout=ones, absorb in_scale on the
    // input and rout post (exact identity with the GPU weight-absorbed form).
    let ones_gu: Vec<f32> = vec![1.0f32; mi*2];
    // decode_and_fold_weight applies rout rows — pass ones, then apply rout
    // post-GEMV (GPU absorbed it into the weight; the math is the same).
    let mf = escham::decode_and_fold_weight(&gcode_e, 2, dim, mi*2, &ones_gu, &ones_gu);
    let mut y_ref = vec![0.0f32; mi*2];
    for o in 0..mi*2 {
        let mut s = 0.0f32;
        for i in 0..dim { s += mf[o*dim + i] * x[i] * in_scale[i]; }
        y_ref[o] = s * grout_e[o];
    }
    // NOTE: decode_and_fold_weight folds WITH rout per beta signature. Compare raw.
    let scl: f32 = y_ref.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1e-3);
    let mut maxabs = 0f32;
    for i in 0..mi*2 { maxabs = maxabs.max((y_gpu[i]-y_ref[i]).abs()); }
    eprintln!("layer {lyr} exp {exp} gate_up: y_gpu[0..4]={:?}", &y_gpu[..4]);
    eprintln!("  y_ref[0..4]={:?}", &y_ref[..4]);
    eprintln!("  maxabs={maxabs:.4} scale={scl:.4} rel={:.4}", maxabs/scl);
}
