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
        #p.matplotlib
        #p.numpy
        #p.pandas
        p.pymc3
        #p.seaborn
        p.scikitlearn
        p.h5py
      ]);

    in rec {
      devShell = pkgs.mkShell {
        buildInputs = [ pyenv ];
      };
    }; # eachSystem

    # python3.7 is needed for pymc3. However, do *not* override "python", otherwise we rebuild have
    # the world!
    overlay = final: prev: {
      #python37 = prev.python37.override {
      #  #packageOverrides = pself: psuper: {
      #  #  tensorflow = pself.tensorflow_2;
      #  #}; # package overrdise
      #}; # python3 override
    };

  in
    flake-utils.lib.eachDefaultSystem eachSystem // { inherit overlay; };
}
