//! Compile HIP kernels to code objects (.hsaco) via hipcc.

use hip_bridge::HipResult;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Compiles HIP kernel sources to code objects, with caching.
pub struct KernelCompiler {
    cache_dir: PathBuf,
    arch: String,
    compiled: HashMap<String, PathBuf>,
}

impl KernelCompiler {
    pub fn new(arch: &str) -> HipResult<Self> {
        let cache_dir = std::env::temp_dir().join("hipfire_kernels");
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            hip_bridge::HipError::new(0, &format!("failed to create cache dir: {e}"))
        })?;
        Ok(Self {
            cache_dir,
            arch: arch.to_string(),
            compiled: HashMap::new(),
        })
    }

    /// Compile a HIP kernel source string. Returns path to .hsaco file.
    /// Caches by kernel name + source hash — recompiles if source changes.
    pub fn compile(&mut self, name: &str, source: &str) -> HipResult<&Path> {
        if self.compiled.contains_key(name) {
            return Ok(&self.compiled[name]);
        }

        let src_path = self.cache_dir.join(format!("{name}.hip"));
        let obj_path = self.cache_dir.join(format!("{name}.hsaco"));
        let hash_path = self.cache_dir.join(format!("{name}.hash"));

        // Check if cached .hsaco matches current source
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        self.arch.hash(&mut hasher);
        let src_hash = format!("{:016x}", hasher.finish());

        let cache_valid = obj_path.exists() && hash_path.exists()
            && std::fs::read_to_string(&hash_path).unwrap_or_default() == src_hash;

        if !cache_valid {
            std::fs::write(&src_path, source).map_err(|e| {
                hip_bridge::HipError::new(0, &format!("failed to write kernel source: {e}"))
            })?;

            // Remove stale .hsaco before compiling
            let _ = std::fs::remove_file(&obj_path);

            let output = Command::new("hipcc")
                .args([
                    "--genco",
                    &format!("--offload-arch={}", self.arch),
                    "-O3",
                    "-o",
                    obj_path.to_str().unwrap(),
                    src_path.to_str().unwrap(),
                ])
                .output()
                .map_err(|e| {
                    hip_bridge::HipError::new(0, &format!("failed to run hipcc: {e}"))
                })?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(hip_bridge::HipError::new(
                    0,
                    &format!("hipcc compilation failed for {name}:\n{stderr}"),
                ));
            }

            // Write hash only after successful compilation
            let _ = std::fs::write(&hash_path, &src_hash);
        }

        self.compiled.insert(name.to_string(), obj_path);
        Ok(&self.compiled[name])
    }
}
