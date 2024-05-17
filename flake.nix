{
  description = "JIT compilation with type-checking, shape-checking, & runtime error handling built in.";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixfmt = {
      inputs.flake-utils.follows = "flake-utils";
      url = "github:serokell/nixfmt";
    };
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs =
    {
      flake-utils,
      nixfmt,
      nixpkgs,
      self,
    }:
    let
      pname = "check-and-compile";
      pyname = "check_and_compile";
      version = "0.0.1";
      src = ./.;
      default-pkgs =
        p:
        with p;
        [
          beartype
          jaxtyping
        ]
        ++ [
          (jax.overridePythonAttrs (
            old:
            old
            // {
              doCheck = false;
              propagatedBuildInputs = old.propagatedBuildInputs ++ [ p.jaxlib-bin ];
            }
          ))
        ];
      ci-pkgs =
        p: with p; [
          black
          mypy
        ];
      dev-pkgs = p: with p; [ python-lsp-server ];
      lookup-pkg-sets = ps: p: builtins.concatMap (f: f p) ps;
    in
    {
      lib.with-pkgs =
        pkgs: pypkgs:
        pypkgs.buildPythonPackage {
          inherit pname version src;
          pyproject = true;
          dependencies = lookup-pkg-sets [ default-pkgs ] pypkgs;
        };
    }
    // flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        pypkgs = pkgs.python311Packages;
        python-with = ps: "${pypkgs.python.withPackages (lookup-pkg-sets ps)}/bin/python";
      in
      {
        apps.ci = {
          type = "app";
          program = "${
            let
              pname = "ci";
              python = python-with [
                default-pkgs
                ci-pkgs
              ];
              find = "${pkgs.findutils}/bin/find";
              nixfmt-bin = "${nixfmt.packages.${system}.default}/bin/nixfmt";
              rm = "${pkgs.coreutils}/bin/rm";
              xargs = "${pkgs.findutils}/bin/xargs";
              exec = ''
                #!${pkgs.bash}/bin/bash

                set -eu

                export JAX_ENABLE_X64=1

                ${rm} -fr result
                ${find} . -name '*.nix' | ${xargs} ${nixfmt-bin} --check
                ${python} -m black --check .
                ${python} -m mypy .
              '';
            in
            pkgs.stdenv.mkDerivation {
              inherit pname version src;
              buildPhase = ":";
              installPhase = ''
                mkdir -p $out/bin
                echo "${exec}" > $out/bin/${pname}
                chmod +x $out/bin/${pname}
              '';
            }
          }/bin/ci";
        };
        devShells.default = pkgs.mkShell {
          JAX_ENABLE_X64 = "1";
          packages = (
            lookup-pkg-sets [
              default-pkgs
              ci-pkgs
              dev-pkgs
            ] pypkgs
          );
        };
        packages.default = self.lib.with-pkgs pkgs pypkgs;
      }
    );
}
