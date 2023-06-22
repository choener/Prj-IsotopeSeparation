{
  description = "Isotope Separation: determine if we can separate out deuterium in ONT data";

  # This uses the new pymc 4.0.0! Be very careful updating the inputs!

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/23.05";
    flake-utils.url = "github:numtide/flake-utils";
    devshell.url = "github:numtide/devshell";
  };

  outputs = { self, nixpkgs, flake-utils, devshell }: let

    # each system
    eachSystem = system: let
      config = { allowUnfree = true;};
      pkgs = import nixpkgs {
        inherit system;
        inherit config;
        overlays = [ self.overlay devshell.overlays.default ];
      };
      pyenv = pkgs.python310.withPackages (p: [
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
        p.zstandard
        p.jax
        #p.jaxlibWithCuda
        p.jaxlib
        p.numpyro
        p.tkinter
      ]);

    in rec {
      devShell = let
        # pkgs = import nixpkgs { inherit system; overlays = [ devshell.overlay ]; };
      in pkgs.devshell.mkShell {
        devshell.packages = with pkgs; [ pyenv hdf5 ont_vbz_compression nodejs tk ]; # cudatoolkit
        env = [
          { name = "HDF5_PLUGIN_PATH"; value = "${pkgs.hdf5}/lib:${pkgs.ont_vbz_compression}/lib"; }
          { name = "PYTHONPATH"; eval = "$PRJ_ROOT/ONTlib:$PRJ_ROOT/IsotopeSep"; }
          { name = "MKL_NUM_THREADS"; value = 1; }
          { name = "OMP_NUM_THREADS"; value = 1; }
          { name = "TK_LIBRARY"; value = "${pkgs.tk}/lib/${pkgs.tk.libPrefix}"; }
        ];
        imports = [ (pkgs.devshell.importTOML ./devshell.toml) ];
      };
      packages."vbz" = pkgs.ont_vbz_compression;
      # TODO Generate a docker container suitable for podman. Should be built with cuda support.
    }; # eachSystem

    overlay = final: prev: rec {
      ont_vbz_compression = final.callPackage ./IsotopeSep/pkgs/vbz {};
      #python310 = prev.python310.override {
      #  packageOverrides = self: super: {
      #    #numpyro = self.callPackage ./numpyro.nix {};
      #    jaxlib = self.jaxlibWithCuda;

      #  };
      #};
      #python310Packages = python310.pkgs;
    };

  in
    flake-utils.lib.eachDefaultSystem eachSystem // { inherit overlay; };
}
