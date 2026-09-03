# nix/dev-shell.nix — hipfire development shell.
#
# The default shell keeps nixpkgs' split rocmPackages (ROCm 7.x). A `therock`
# shell against the AMD TheRock binary SDK (ROCm 10.x) is provided via the
# ./rocm.nix provider abstraction:
#
#   nix develop .#therock      # or: nix develop .#devShells.x86_64-linux.therock
#
# The provider supplies:
#   * shellInputs   — derivations exposing hipcc + tools on PATH
#   * libDirs       — derivations whose lib/ must be on LD_LIBRARY_PATH so the
#                     daemon can dlopen libamdhip64 etc.
#   * env           — extra env (ROCM_PATH / HIP_PATH / HIP_CLANG_PATH /
#                     HIPFIRE_HIPCC_EXTRA_FLAGS for the device-lib bitcode)
#   * label         — human diagnostics string
#
# The shell also needs the rust toolchain from rust-overlay. Everything else
# mirrors the old nixpkgs-only file's behaviour byte-for-byte when the default
# provider is selected.
{ lib
, mkShell
, rust-bin
, rocmPackages ? null
, pkg-config
, rocm ? null
, rocmSupport ? true
}:

let
  rocmLib = import ./rocm.nix { inherit lib; };
  provider =
    if rocm != null then rocm
    else if rocmPackages != null then rocmLib.nixpkgs { inherit rocmPackages; }
    else null;
  useRocm = rocmSupport && provider != null;
  label = if provider != null then provider.label else "no-rocm";
in
mkShell {
  name = "hipfire-dev-${label}";

  nativeBuildInputs = [
    (rust-bin.stable.latest.default.override {
      extensions = [ "rust-src" "rust-analyzer" ];
    })
    pkg-config
  ] ++ lib.optionals useRocm provider.shellInputs;

  # Match the runtime closure used by package.nix/module.nix: the daemon
  # dlopens libamdhip64 / rocm-runtime / rocm-comgr / rocprofiler-register at
  # startup. Without the full set, `nix develop` cannot run the daemon
  # ("no ROCm-capable device detected" at initialization).
  LD_LIBRARY_PATH = lib.optionalString useRocm (lib.makeLibraryPath provider.libDirs);

  # Provider env (ROCM_PATH/HIP_PATH/HIP_CLANG_PATH/device-lib flags). Each
  # attr is exported as-is.
  env = lib.optionalAttrs useRocm provider.env;

  shellHook = ''
    echo "hipfire dev shell (${label})"
    echo "  rust: $(rustc --version)"
    ${lib.optionalString useRocm ''
      echo "  hip:  $(hipcc --version 2>&1 | head -1)"
    ''}
  '';
}
