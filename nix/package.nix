# nix/package.nix — hipfire package derivation.
#
# Layout notes (post-modular repo):
#   * The daemon is a [[bin]] named `daemon` in crates/hipfire-daemon (it used
#     to be a hipfire-runtime [[example]]). Build with
#     `cargo build --release -p hipfire-daemon`.
#   * The CLI is a [[bin]] named `hipfire` in crates/hipfire-cli.
#   * `infer`/`infer_hfq` were hipfire-runtime [[example]]s and are now
#     archived research harnesses in crates/saddle-lab, which is not part of
#     the default build graph — the product installs daemon + CLI only.
#
# ROCm: hipfire dlopens libamdhip64 at runtime and JIT-compiles kernels via
# hipcc, so nothing links ROCm at build time. Two ways to select the ROCm:
#
#   * `rocmPackages` — the classic pkgs.rocmPackages (ROCm 7.x). Auto-filled
#     by callPackage; kept for backwards compatibility (flake default and the
#     NixOS module both go through it).
#   * `rocm` — a *provider* attrset from ./rocm.nix (nixpkgs or therock).
#     Passing one takes precedence over rocmPackages and lets a caller select
#     the AMD TheRock binary SDK (ROCm 10.x) without touching this file.
#
# The provider's `env` attrs (ROCM_PATH / HIP_PATH / HIP_CLANG_PATH /
# HIPFIRE_HIPCC_EXTRA_FLAGS) are exported to the wrapped binaries so the
# daemon's runtime dlopen and JIT compile resolve to the same ROCm. Its
# `libDirs` go on LD_LIBRARY_PATH.
{ lib
, rustPlatform
, rocmPackages ? null
, makeWrapper
, rocm ? null
, rocmSupport ? true
, src ? lib.cleanSource ./..
, cargoLockFile ? ../Cargo.lock
}:

let
  cargoToml = builtins.fromTOML (builtins.readFile (src + "/Cargo.toml"));
  rocmLib = import ./rocm.nix { inherit lib; };

  # Provider resolution: explicit `rocm` wins; else classic rocmPackages; else
  # empty (rocmSupport=false case, caller supplies libamdhip64 itself).
  provider =
    if rocm != null then rocm
    else if rocmPackages != null then rocmLib.nixpkgs { inherit rocmPackages; }
    else null;

  envExport = lib.optionalString (rocmSupport && provider != null && provider ? env)
    (lib.concatStringsSep " " (
      lib.mapAttrsToList (k: v: "--set ${k} ${lib.escapeShellArg v}") provider.env
    ));
  ldLibraryPath = lib.optionalString (rocmSupport && provider != null && provider ? libDirs)
    (lib.makeLibraryPath provider.libDirs);
in
rustPlatform.buildRustPackage {
  pname = "hipfire";
  version = cargoToml.workspace.package.version or cargoToml.package.version;

  inherit src;
  cargoLock.lockFile = cargoLockFile;
  doCheck = false;  # tests require GPU

  buildPhase = ''
    runHook preBuild
    # Daemon is a [[bin]] in crates/hipfire-daemon (deltanet is its default
    # feature; --features deltanet is explicit for clarity).
    cargo build --release -p hipfire-daemon --features deltanet
    cargo build --release -p hipfire-cli
    runHook postBuild
  '';

  dontCargoInstall = true;

  nativeBuildInputs = [ makeWrapper ];

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin

    # Install and wrap daemon binary with LD_LIBRARY_PATH for libamdhip64.so
    # dlopen plus the provider env (ROCM_PATH/HIP_PATH/HIP_CLANG_PATH/...).
    cp target/release/daemon $out/bin/hipfire-daemon-unwrapped
    makeWrapper $out/bin/hipfire-daemon-unwrapped $out/bin/hipfire-daemon \
      --prefix LD_LIBRARY_PATH : "${ldLibraryPath}" \
      ${envExport}

    # Install the native Rust control plane. HIPFIRE_DAEMON_BIN points it at
    # the ROCm-wrapped daemon rather than relying on a source-tree layout.
    cp target/release/hipfire $out/bin/hipfire-unwrapped
    makeWrapper $out/bin/hipfire-unwrapped $out/bin/hipfire \
      --set HIPFIRE_DAEMON_BIN $out/bin/hipfire-daemon \
      --prefix LD_LIBRARY_PATH : "${ldLibraryPath}" \
      ${envExport}

    runHook postInstall
  '';

  meta = with lib; {
    description = "LLM inference for AMD RDNA GPUs";
    homepage = "https://github.com/warpfront/hipfire";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
    mainProgram = "hipfire";
  };
}
