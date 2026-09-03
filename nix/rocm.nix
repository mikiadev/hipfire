# nix/rocm.nix — ROCm provider abstraction for hipfire's Nix packaging.
#
# hipfire never links ROCm at build time: hip-bridge dlopens libamdhip64 at
# runtime and rdna-compute JIT-compiles kernels via hipcc. So a "ROCm
# provider" only has to answer four questions for the dev shell, the package
# wrappers, and the kernel precompiler:
#
#   * which store dirs go on LD_LIBRARY_PATH (dlopen'd runtime libs)
#   * which environment the runtime/JIT needs (ROCM_PATH / HIP_PATH /
#     HIP_CLANG_PATH / device-lib bitcode)
#   * which derivation(s) provide the device compiler and tools
#   * a human label + version for diagnostics
#
# Two providers are shipped:
#
#   * nixpkgs        — the classic `pkgs.rocmPackages` set (ROCm 7.x). Default;
#                      byte-for-byte the behaviour previous nix files had.
#   * therock        — AMD's binary TheRock SDK (ROCm 10.x / HIP 7.15+), fetched
#                      from stable.repo.amd.com by ./therock-sdk.nix. Opt-in.
#
# Consumers (package.nix, dev-shell.nix, kernels.nix, module.nix) take a
# provider attrset instead of hardcoding `rocmPackages.<component>` paths, so
# switching a whole build/run to a different ROCm is a one-line flake change
# instead of a per-component rewrite.
{ lib }:

let
  # nixpkgs ROCm packages live under pkgs.rocmPackages (ROCm 7.x on current
  # nixpkgs). Some attributes (rocm-core) carry the version string.
  nixpkgs = { rocmPackages }:
    let
      version = rocmPackages.rocm-core.version or "unknown";
      runtimeLibs = [
        rocmPackages.clr
        rocmPackages.rocm-runtime
        rocmPackages.rocm-comgr
        rocmPackages.rocprofiler-register
      ];
    in
    {
      flavor = "nixpkgs";
      inherit version rocmPackages;
      label = "nixpkgs-rocm-${version}";
      # Derivation dirs whose `lib/` holds the dlopen'd runtime (libamdhip64,
      # libhsa-runtime64, libamd_comgr, librocprofiler-register). These are the
      # same four components package.nix/dev-shell.nix/module.nix always put on
      # LD_LIBRARY_PATH.
      libDirs = runtimeLibs;
      # Env the runtime + JIT need. nixpkgs' clr ships hipcc but NOT the ROCm
      # device-libs bitcode (separate package), so the device-lib path must be
      # handed to hipcc explicitly via HIPFIRE_HIPCC_EXTRA_FLAGS.
      env = {
        HIP_PATH = "${rocmPackages.clr}";
        HIPFIRE_HIPCC_EXTRA_FLAGS =
          "--rocm-device-lib-path=${rocmPackages.rocm-device-libs}/amdgcn/bitcode";
      };
      # Full derivations for a dev shell (hipcc + tools). rocprofiler-sdk is
      # the rocprofv3 ground truth used by scripts/rocprof-wrap.sh.
      shellInputs = [
        rocmPackages.clr
        rocmPackages.rocm-smi
        rocmPackages.rocminfo
        rocmPackages.rocprofiler-sdk
      ];
      # Device compiler entry point (used by the kernel precompiler).
      hipcc = "${rocmPackages.clr}/bin/hipcc";
    };

  # AMD TheRock binary SDK. `version` pins the release (e.g. "10.0.0") and
  # `target` the GPU suffix of the tarball (gfx1151 = Strix Halo). The SDK is
  # self-contained: hipcc, amdclang/LLVM, device-libs bitcode, and every
  # runtime library live under one root, so the env only needs ROCM_PATH /
  # HIP_PATH plus the HIP_CLANG_PATH shim that makes clang find Nix's C++
  # headers (see ./therock-sdk.nix).
  therock = { pkgs, target ? "gfx1151", version ? "10.0.0" }:
    let
      sdk = pkgs.callPackage ./therock-sdk.nix { inherit target version; };
      # hipcc resolves its device compiler via $HIP_CLANG_PATH/<clang++>. The
      # SDK's bin/therock-hip-clang++ is the Nix-aware wrapper; expose a dir
      # whose clang++/clang are symlinks to it.
      clangShim = pkgs.runCommand "therock-hip-clang-shim-${target}-${version}" { } ''
        mkdir -p $out/bin
        ln -s ${sdk}/bin/therock-hip-clang++ $out/bin/clang++
        ln -s ${sdk}/bin/therock-hip-clang++ $out/bin/clang
      '';
    in
    {
      flavor = "therock";
      inherit version;
      label = "therock-${target}-${version}";
      libDirs = [ sdk ];
      env = {
        ROCM_PATH = "${sdk}";
        HIP_PATH = "${sdk}";
        # Raw SDK clang++ cannot find Nix's libstdc++ headers; hipcc looks for
        # clang++ under HIP_CLANG_PATH, so point it at a dir whose clang++ is
        # the Nix-aware wrapper produced by the SDK derivation.
        HIP_CLANG_PATH = "${clangShim}/bin";
        HIP_PLATFORM = "amd";
      };
      shellInputs = [ sdk ];
      hipcc = "${sdk}/bin/hipcc";
      inherit sdk clangShim;
    };
in
{
  inherit nixpkgs therock;
}
