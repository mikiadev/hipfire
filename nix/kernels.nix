# nix/kernels.nix — pre-compiled GPU kernels for hipfire.
#
# Compiles kernels/src/*.hip to kernels/compiled/<arch> for the requested GPU
# targets via scripts/compile-kernels.sh. The script resolves hipcc through
# hipfire's own ROCm resolver (crates/hipfire-config hipfire-rocm-resolve), so
# it honors ROCM_PATH / HIP_PATH / HIP_CLANG_PATH from the environment. That
# means this derivation must run inside the provider env: `rocm` selects the
# ROCm provider (./rocm.nix), whose env is exported around the build.
#
# Default target list is empty (daemon JIT-compiles on first use); override
# via `gpuTargets` or the NixOS module's services.hipfire.gpuTargets.
{ lib
, stdenv
, rocmPackages ? null
, rocm ? null
, gpuTargets ? [ ]
}:

let
  rocmLib = import ./rocm.nix { inherit lib; };
  provider =
    if rocm != null then rocm
    else if rocmPackages != null then rocmLib.nixpkgs { inherit rocmPackages; }
    else null;
  src = lib.cleanSource ./..;
  cargoToml = builtins.fromTOML (builtins.readFile (src + "/Cargo.toml"));

  # Provider env exported around the build: nixpkgs → HIP_PATH +
  # HIPFIRE_HIPCC_EXTRA_FLAGS (device libs live in a separate package);
  # therock → ROCM_PATH/HIP_PATH + HIP_CLANG_PATH (device libs are inside the
  # SDK root, hipcc is Nix-aware via the shim).
  providerEnvExport = lib.optionalString (provider != null)
    (lib.concatStringsSep "\n" (
      lib.mapAttrsToList (k: v: "export ${k}=${lib.escapeShellArg v}") provider.env
    ));
in
stdenv.mkDerivation {
  pname = "hipfire-kernels";
  version = cargoToml.workspace.package.version or cargoToml.package.version;

  inherit src;

  nativeBuildInputs = lib.optionals (provider != null) provider.shellInputs;

  buildPhase = ''
    runHook preBuild
    export HOME=$TMPDIR
    ${providerEnvExport}
    # Allow partial failures — some kernels are arch-specific and won't
    # compile for every target. The daemon JIT-compiles missing kernels.
    bash scripts/compile-kernels.sh ${lib.concatStringsSep " " gpuTargets} || {
      echo "WARNING: some kernels failed to compile (see above). Daemon will JIT-compile them on first use."
    }
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out/kernels/compiled
    for arch in ${lib.concatStringsSep " " gpuTargets}; do
      if [ -d "kernels/compiled/$arch" ]; then
        cp -r "kernels/compiled/$arch" "$out/kernels/compiled/"
      fi
    done
    runHook postInstall
  '';

  meta = with lib; {
    description = "Pre-compiled GPU kernels for hipfire";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
