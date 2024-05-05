from check_and_compile import check_and_compile

from jax import grad, jit, numpy as jnp
from jax.experimental.checkify import check, checkify
from jaxtyping import Array, Float32


print("Hi")


@check_and_compile()
def a():
    print("Running `a` for the first time")
    check(True, "check failed in `a`!")


print("Defined `a`")


@check_and_compile()
def b():
    print("Running `b` for the first time")
    a()
    check(True, "check failed in `b`!")


print("Defined `b`")


b()


print("Ran `b`")


b()


print("Ran `b` again")


@check_and_compile()
def c(x: Float32[Array, ""]) -> Float32[Array, ""]:
    return x + 1


x = jnp.array(1, dtype=jnp.float32)
c(x)
jit(checkify(grad(c)))(x)


print()

print("Cool, that should have printed exactly the following:")
print(">>> Hi")
print(">>> Defined `a`")
print(">>> Defined `b`")
print(">>> Running `b` for the first time")
print(">>> Running `a` for the first time")
print(">>> Compiling b...")
print(">>> Ran `b`")
print(">>> Ran `b` again")
print()
print("Note that `a` should *not* have been compiled,")
print("since it was already part of `b` when `b` was compiled,")
print("and that `b` should not have printed the second time,")
print("since we should have called the compiled version.")
