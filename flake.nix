{
  description = "RNAmodBayes: Bayesian analysis of RNA modifications";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/20.09";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: let

    # each system
    eachSystem = system: let
      config = { allowUnfree = true;};
      pkgs = import nixpkgs {
        inherit system;
        inherit config;
        overlays = [ self.overlay ];
      };
      pyenv = pkgs.python37.withPackages (p: [
        #p.arviz
        p.h5py
        p.matplotlib
        p.numpy
        p.pandas
        p.pymc3
        p.scikitlearn
        p.seaborn
      ]);

    in rec {
      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [ pyenv hdf5 ont_vbz_compression ];
        HDF5_PLUGIN_PATH="${pkgs.hdf5}/lib:${pkgs.ont_vbz_compression}/lib";
      };
      #devShells."vbz" = pkgs.mkShell {
      #  buildInputs = with pkgs; [ cmake ]
      #};
      packages."vbz" = pkgs.ont_vbz_compression;
    }; # eachSystem

    # python3.7 is needed for pymc3. However, do *not* override "python", otherwise we rebuild have
    # the world!
    overlay = final: prev: {
      ont_vbz_compression = final.callPackage ./pkgs/vbz {};
      #python37 = prev.python37.override {
      #  #packageOverrides = pself: psuper: {
      #  #  tensorflow = pself.tensorflow_2;
      #  #}; # package overrdise
      #}; # python3 override
    };

  in
    flake-utils.lib.eachDefaultSystem eachSystem // { inherit overlay; };
}
