# nix/therock-sdk.nix — self-contained AMD TheRock binary ROCm SDK.
#
# AMD publishes relocatable binary ROCm SDK tarballs for TheRock under
# https://stable.repo.amd.com/rocm/core/tarball/. Unlike nixpkgs' split
# rocmPackages (clr + rocm-runtime + rocm-comgr + rocprofiler-register +
# rocm-device-libs as separate derivations), a TheRock tarball is one flat
# root holding hipcc, the amdclang/LLVM toolchain, the amdgcn bitcode
# device-libs, and every runtime .so — which is exactly the layout hipfire's
# ROCm resolver treats as a "complete root" (include/hip/hip_runtime.h,
# lib/libamdhip64.so.N, bin/hipcc).
#
# The tarball is built for a concrete GPU target (gfx1151 = Strix Halo). Each
# (target, version) pair has its own tarball URL and hash; update the map at
# the bottom of this file when bumping.
#
# Nix-awareness: the bundled clang++ is a generic upstream clang that does not
# know about Nix's libstdc++/libc, so compiling any HIP kernel fails with
# "cstdlib file not found" unless clang is told where the toolchain is. The
# SDK derivation therefore also installs:
#
#   * bin/therock-hip-clang++ — a wrapper around lib/llvm/bin/clang++ that
#     adds --gcc-toolchain and the Nix libc/libstdc++ include+link flags
#     (mirrors hellas-ai/nix-strix-halo's rocm-sdk derivation).
#   * passthru.clangShim — a dir whose `clang++`/`clang` are symlinks to that
#     wrapper. hipcc (the python script in bin/) resolves its device compiler
#     through $HIP_CLANG_PATH, so exporting
#     HIP_CLANG_PATH=${sdk.clangShim}/bin makes hipfire's JIT kernel compiles
#     work against a Nix GCC toolchain.
#
# hipfire itself does not link this derivation at build time (dlopen + JIT),
# so this is a lightweight stdenvNoCC unpack + wrapper, no cmake.

{ lib
, stdenvNoCC
, stdenv
, fetchurl
, patchelf
, libdrm
, numactl
, rdma-core
, target ? "gfx1151"
, version ? "10.0.0"
}:

let
  # (target, version) -> { url, hash }. Keep the current pins here so the map
  # is readable; nightlies/stable move, so check the resolved store path or
  # `nix-prefetch-url` when bumping.
  pins = {
    # 10.0.0 (HIP 7.15.26333, AMD clang 23), Strix Halo / gfx1151.
    "gfx1151-10.0.0" = {
      url = "https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx1151-10.0.0.tar.gz";
      hash = "sha256-T+q9ny2nI1LfN/bXFKVIR9P+kTwDQfviplQsEWQCS68=";
    };
  };
  pin = pins."${target}-${version}" or (throw
    "nix/therock-sdk.nix: no pin for target '${target}' version '${version}'. Add it to the pins map.");
in
stdenvNoCC.mkDerivation {
  pname = "therock-rocm-sdk-${target}";
  inherit version;

  src = fetchurl {
    inherit (pin) url hash;
  };

  nativeBuildInputs = [ patchelf ];

  propagatedBuildInputs = [ libdrm numactl rdma-core ];

  dontConfigure = true;
  dontBuild = true;
  dontPatchELF = true;
  dontStrip = true;

  unpackPhase = ''
    runHook preUnpack
    tar -xzf "$src"
    runHook postUnpack
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p "$out"
    shopt -s dotglob nullglob

    if [ -d install ]; then
      cp -R install/* "$out/"
    else
      entries=(*/)
      if [ "''${#entries[@]}" -eq 1 ] && [ -d "''${entries[0]}install" ]; then
        cp -R "''${entries[0]}install/"* "$out/"
      else
        cp -R ./* "$out/"
      fi
    fi

    chmod -R u+w "$out"
    find "$out" -type f -name "*.so*" -exec chmod 755 {} \; 2>/dev/null || true
    find "$out/bin" "$out/llvm/bin" -type f -exec chmod 755 {} \; 2>/dev/null || true

    # TheRock binary SDKs are built as relocatable Linux tarballs, not Nix
    # derivations. Make the shipped ELF binaries/libraries usable in pure Nix
    # build sandboxes so hipcc/hipconfig/rocminfo actually run.
    gcc_runtime=${lib.escapeShellArg "${stdenv.cc.cc.lib}/lib"}
    libdrm_runtime=${lib.escapeShellArg "${lib.getLib libdrm}/lib"}
    numa_runtime=${lib.escapeShellArg "${lib.getLib numactl}/lib"}
    rdma_runtime=${lib.escapeShellArg "${lib.getLib rdma-core}/lib"}
    dynamic_linker=${lib.escapeShellArg stdenv.cc.bintools.dynamicLinker}
    runtime_paths=(
      "$gcc_runtime"
      "$libdrm_runtime"
      "$numa_runtime"
      "$rdma_runtime"
    )

    patch_elf() {
      local elf="$1"
      if old_rpath=$(patchelf --print-rpath "$elf" 2>/dev/null); then
        new_rpath="$old_rpath"
        for runtime_path in "''${runtime_paths[@]}"; do
          case ":$new_rpath:" in
            *":$runtime_path:"*) ;;
            *) new_rpath="''${new_rpath:+$new_rpath:}$runtime_path" ;;
          esac
        done
        patchelf --set-rpath "$new_rpath" "$elf" 2>/dev/null || true
      fi

      if patchelf --print-interpreter "$elf" >/dev/null 2>&1; then
        patchelf --set-interpreter "$dynamic_linker" "$elf" 2>/dev/null || true
      fi
    }

    for dir in "$out/bin" "$out/llvm/bin" "$out/lib/llvm/bin"; do
      [ -d "$dir" ] || continue
      for tool in \
        hipcc hipconfig hipcc_cmake_linker_helper \
        rocminfo rocm_agent_enumerator rocm-smi amd-smi \
        amdclang amdclang++ clang clang++ clang-offload-bundler \
        lld ld.lld llvm-config llvm-ar llvm-ranlib llvm-link llvm-objcopy; do
        [ -f "$dir/$tool" ] || continue
        patch_elf "$dir/$tool"
      done
    done

    for dir in "$out/lib"; do
      [ -d "$dir" ] || continue
      while IFS= read -r -d "" elf; do
        patch_elf "$elf"
      done < <(find "$dir" -maxdepth 1 -type f \( -name "*.so" -o -name "*.so.*" \) -print0)
    done

    # Nix-aware device compiler wrapper: the bundled clang++ cannot find Nix's
    # libstdc++/libc headers on its own. This wrapper is what hipcc must use
    # (via HIP_CLANG_PATH) for JIT kernel compiles to succeed.
    cc_cflags_before="$(cat ${stdenv.cc}/nix-support/cc-cflags-before)"
    cc_cflags="$(cat ${stdenv.cc}/nix-support/cc-cflags)"
    libc_cflags="$(cat ${stdenv.cc}/nix-support/libc-cflags)"
    libc_crt1_cflags="$(cat ${stdenv.cc}/nix-support/libc-crt1-cflags)"
    libc_ldflags="$(cat ${stdenv.cc}/nix-support/libc-ldflags)"
    cc_ldflags="$(cat ${stdenv.cc}/nix-support/cc-ldflags)"
    libc="$(cat ${stdenv.cc}/nix-support/orig-libc)"
    dynamic_linker="$(cat ${stdenv.cc}/nix-support/dynamic-linker)"

    cat > "$out/bin/therock-hip-clang++" <<EOF
    #!/bin/sh
    exec "$out/lib/llvm/bin/clang++" \\
      --gcc-toolchain=${stdenv.cc.cc} \\
      --rocm-path="$out" \\
      $cc_cflags_before \\
      $cc_cflags \\
      $libc_cflags \\
      $libc_crt1_cflags \\
      "\$@" \\
      -L$libc/lib \\
      -Wl,--dynamic-linker=$dynamic_linker \\
      -Wl,-rpath,${stdenv.cc.cc.lib}/lib \\
      -Wl,-rpath,$libc/lib \\
      $libc_ldflags \\
      $cc_ldflags
EOF
    sed -i 's/^    //' "$out/bin/therock-hip-clang++"
    chmod 755 "$out/bin/therock-hip-clang++"

    # TheRock's CMake HIP helper normally re-enters hipcc, which then invokes
    # the raw bundled clang++ during link steps. That loses Nix's libc/libstdc++
    # search paths. Keep the CMake contract, but route the actual link through
    # the Nix-aware compiler wrapper above.
    cat > "$out/bin/hipcc_cmake_linker_helper" <<EOF
    #!/bin/sh
    [ "\$#" -gt 0 ] && shift
    exec "$out/bin/therock-hip-clang++" "\$@"
EOF
    sed -i 's/^    //' "$out/bin/hipcc_cmake_linker_helper"
    chmod 755 "$out/bin/hipcc_cmake_linker_helper"

    runHook postInstall
  '';

  passthru = {
    inherit target;
  };

  meta = with lib; {
    description = "AMD TheRock ROCm ${version} binary SDK (HIP ${version}, target ${target})";
    homepage = "https://github.com/ROCm/TheRock";
    license = licenses.mit;
    maintainers = [ ];
    platforms = [ "x86_64-linux" ];
  };
}
