{
  description = "Isotope Separation: determine if we can separate out deuterium in ONT data";

  # This uses the new pymc 4.0.0! Be very careful updating the inputs!

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
    jupyterWith.url = "github:tweag/jupyterWith";
  };

  outputs = { self, nixpkgs, flake-utils, jupyterWith }: let

    # each system
    eachSystem = system: let
      config = { allowUnfree = true;};
      pkgs = import nixpkgs {
        inherit system;
        inherit config;
        overlays = nixpkgs.lib.attrValues jupyterWith.overlays ++ [ self.overlay ];
      };
      pyenv = pkgs.python3.withPackages (p: [
        p.arviz
        p.h5py
        p.matplotlib
        p.numpy
        p.pandas
        p.pymc3
        p.scikitlearn
        p.seaborn
        p.scipy
      ]);

    in rec {
      devShell = pkgs.mkShell {
        buildInputs = with pkgs; [ pyenv hdf5 ont_vbz_compression nodejs ];
        HDF5_PLUGIN_PATH="${pkgs.hdf5}/lib:${pkgs.ont_vbz_compression}/lib";
      };
      packages."vbz" = pkgs.ont_vbz_compression;
    }; # eachSystem

    # python3.7 is needed for pymc3. However, do *not* override "python", otherwise we rebuild have
    # the world!
    overlay = final: prev: {
      ont_vbz_compression = final.callPackage ./pkgs/vbz {};
    };

  in
    flake-utils.lib.eachDefaultSystem eachSystem // { inherit overlay; };
}
