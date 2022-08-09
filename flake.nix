{
  description = "Isotope Separation: determine if we can separate out deuterium in ONT data";

  # This uses the new pymc 4.0.0! Be very careful updating the inputs!

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
    jupyterWith.url = "github:tweag/jupyterWith";
    IsotopeSep = { url = "./IsotopeSep"; inputs = { nixpkgs.follows = "nixpkgs"; flake-utils.follows = "flake-utils"; jupyterWith.follows = "jupyterWith"; }; };
  };

  outputs = { self, nixpkgs, flake-utils, jupyterWith, IsotopeSep }: let

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
      # TODO run the data link scripts automatically?
      devShell = IsotopeSep.devShell.${system};
    }; # eachSystem

  in
    flake-utils.lib.eachDefaultSystem eachSystem;
}
