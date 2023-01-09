{
  description = "Isotope Separation: determine if we can separate out deuterium in ONT data";

  # This uses the new pymc 4.0.0! Be very careful updating the inputs!

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/master";
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
      pyenv = pkgs.python3.withPackages (p: [
        p.arviz
        p.h5py
        p.joblib
        p.matplotlib
        p.numpy
        p.pandas
        p.pymc
        p.scikitlearn
        p.scipy
        p.seaborn
        #p.jax
        #p.jaxlib
      ]);

    in rec {
      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [ pyenv hdf5 ont_vbz_compression nodejs ]; # cudatoolkit
        HDF5_PLUGIN_PATH = "${pkgs.hdf5}/lib:${pkgs.ont_vbz_compression}/lib";
        PYTHONPATH = "./IsotopeSep";
        # disable *within-process* multithreading, because nuts tends to hang
        MKL_NUM_THREADS = 1;
        OMP_NUM_THREADS = 1;
        name = "dev-shell";
      };
      packages."vbz" = pkgs.ont_vbz_compression;
    }; # eachSystem

    overlay = final: prev: {
      ont_vbz_compression = final.callPackage ./IsotopeSep/pkgs/vbz {};
    };

  in
    flake-utils.lib.eachDefaultSystem eachSystem // { inherit overlay; };
}
