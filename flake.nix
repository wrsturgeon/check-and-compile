{
  description = "JIT compilation with type-checking, shape-checking, & runtime error handling built in.";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };
  outputs =
    {
      flake-utils,
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
      ci-pkgs = p: with p; [ black ];
      dev-pkgs = p: with p; [ python-lsp-server ];
      lookup-pkg-sets = ps: p: builtins.concatMap (f: f p) ps;
    in
    {
      lib.with-pkgs =
        pkgs: pypkgs:
        pkgs.stdenv.mkDerivation {
          inherit pname version src;
          propagatedBuildInputs = lookup-pkg-sets [ default-pkgs ] pypkgs;
          buildPhase = ":";
          installPhase = ''
            mkdir -p $out/${pypkgs.python.sitePackages}
            mv ./${pyname} $out/${pypkgs.python.sitePackages}/${pyname}
          '';
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
        packages.ci =
          let
            pname = "ci";
            python = python-with [
              default-pkgs
              ci-pkgs
            ];
            exec = ''
              #!${pkgs.bash}/bin/bash

              set -eu

              ${python} -m black --check .
            '';
          in
          pkgs.stdenv.mkDerivation {
            inherit pname version src;
            buildPhase = ":";
            installPhase = ''
              mkdir -p $out/${pypkgs.python.sitePackages}
              mv ./${pyname} $out/${pypkgs.python.sitePackages}/${pyname}

              mkdir -p $out/bin
              echo "${exec}" > $out/bin/${pname}
              chmod +x $out/bin/${pname}
            '';
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
      }
    );
}
