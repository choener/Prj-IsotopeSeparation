{ stdenv, lib
, fetchFromGitHub, fetchurl, autoPatchelfHook
, cmake, conan, dpkg
, zstd, hdf5, ... }: let

# build the ont vbz plugin; using the precompiled plugin

in stdenv.mkDerivation rec {
  pname = "vbz_compression";
  version = "1.0.2";
  src = fetchurl {
    url = "https://github.com/nanoporetech/vbz_compression/releases/download/1.0.2/ont-vbz-hdf-plugin_1.0.2-1.bionic_amd64.deb";
    sha256 = "sha256-Ipy9fOIJ+keve8t+4XMa/vd0YPmTzGoAddhollv1xZQ=";
  };
  unpackPhase = ''
    ${dpkg}/bin/dpkg -x ${src} .
  '';
  installPhase = ''
    mkdir -p $out/lib
    mv usr/local/hdf5/lib/plugin/libvbz_hdf_plugin.so $out/lib
  '';
  buildInputs = [ zstd stdenv.cc.cc.lib ];
  nativeBuildInputs = [ autoPatchelfHook ];
}

