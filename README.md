# `check_and_compile`

This is a thin wrapper around [JAX](https://github.com/google/jax)'s `jit` and `checkify`,
as well as [`jaxtyping`](https://github.com/patrick-kidger/jaxtyping)'s `jaxtyped`.

If this works fantastically, all credit is theirs for building these fantastic tools!

There's not much _new_ here; it's just a useful tool that I find makes life significantly easier.

## Use

Like this:

```python
from check_and_compile import check_and_compile

from jax import numpy as jnp
from jax.experimental.checkify import check
from jaxtyping import Array, Float

@check_and_compile(1, 3)  # (1, 3, ...) are the indices of static arguments!
def f(
    x: Float[Array, "a b c"],
    static_option: bool,
    z: Float[Array, "a b c"],
    second_option: bool,
) -> Float[Array, "a b c"]:

    if static_option:
        check(jnp.all(x == 0), "`x` should be a zero matrix!")

    if second_option:
        check(jnp.all(z == 0), "`z` should be a zero matrix!")

    return x + z


f(jnp.zeros([1, 2, 3]), True, jnp.zeros([1, 2, 3]), True)  # good!
# >>> Compiling f...

f(jnp.zeros([1, 2, 3]), True, jnp.zeros([1, 2, 3]), True)  # same call again...
# no terminal output, since `f` has already been compiled
# note that this called the *compiled* version of `f`

f(jnp.zeros([1, 2, 3]), True, jnp.zeros([42, 42, 42]), True)  # wrong shapes!
# >>> jaxtyping.TypeCheckError: Type-check error whilst checking the parameters of f.
# >>> The problem arose whilst typechecking parameter 'z'.
# >>> Actual value: Traced<ShapedArray(float64[42,42,42])>with<DynamicJaxprTrace(level=1/0)>
# >>> Expected type: <class 'Float[Array, 'a b c']'>.
# >>> ----------------------
# >>> Called with parameters: { 'second_option': True,
# >>>   'static_option': True,
# >>>   'x': Traced<ShapedArray(float64[1,2,3])>with<DynamicJaxprTrace(level=1/0)>,
# >>>   'z': Traced<ShapedArray(float64[42,42,42])>with<DynamicJaxprTrace(level=1/0)>}
# >>> Parameter annotations: (x: Float[Array, 'a b c'], static_option: bool, z: Float[Array, 'a b c'], second_option: bool).
# >>> The current values for each jaxtyping axis annotation are as follows.
# >>> a=1
# >>> b=2
# >>> c=3

f(jnp.ones([1, 2, 3]), True, jnp.zeros([1, 2, 3]), True)  # first check fails
# >>> jax._src.checkify.JaxRuntimeError: `x` should be a zero matrix! (`check` failed)

f(jnp.zeros([1, 2, 3]), True, jnp.ones([1, 2, 3]), True)  # second check fails
# >>> jax._src.checkify.JaxRuntimeError: `z` should be a zero matrix! (`check` failed)
```

## Packaging

This package uses [Nix](https://github.com/nixos/nix) for fully reproducible builds.

It works will all versions of Python, so instead of providing `packages.${system}.default`,
it provides `lib.with-pkgs`, which you can use like this:

```nix
{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    check-and-compile.url = "github:wrsturgeon/check-and-compile";
  };
  outputs =
    { flake-utils, nixpkgs, check-and-compile, self }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = import nixpkgs { inherit system; }; in {
        devShells.default = pkgs.mkShell {
          packages = [
            (pkgs.python311.withPackages (pypkgs: [
              # ... some Python dependencies ...
              (check-and-compile.lib.with-pkgs pkgs pypkgs)
              # ... more Python dependencies ...
            ]))
          ];
        };
      }
    );
}
```
