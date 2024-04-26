from beartype import beartype
from beartype.typing import Callable
from jaxtyping import jaxtyped
import os


if os.getenv("NO_JIT") == "1":  # pragma: no cover

    print(
        "*** NOTE: `NO_JIT` environment variable is `1`, so "
        "functions marked `@check_and_compile(...)` will *not* be compiled!"
    )

    # If we're not JIT-compiling, just typecheck:
    def check_and_compile(*static_argnums) -> Callable[[Callable], Callable]:
        return jaxtyped(typechecker=beartype)  # itself a function

else:  # pragma: no cover

    from jax import core, jit, numpy as jnp
    from jax.experimental.checkify import checkify, all_checks

    # There's lots of functions returning functions here,
    # so I've left copious documentation to clarify what everything does:
    def check_and_compile(*static_argnums) -> Callable[[Callable], Callable]:

        def partially_applied(f: Callable) -> Callable:

            # Define a wrapped function that we'll compile below:
            def checkified(*args, **kwargs):

                # Get this function's name:
                name: str = getattr(f, "__qualname__", "an unnamed function")

                # Give a disclaimer on keyword arguments:
                if kwargs and (os.getenv("CHECK_AND_COMPILE_SILENT") != "1"):
                    print(
                        f"*** NOTE: {name} received keyword arguments {kwargs.keys()}."
                        " Keyword arguments don't often work well with JAX's JIT static arguments."
                        " It might work, but if you can, please try to make them positional instead!"
                    )

                # Type-check the function:
                g = jaxtyped(f, typechecker=beartype)

                # Functionalize runtime checks:
                g = checkify(g, errors=all_checks)

                # Call it:
                y = g(*args, **kwargs)

                # Functions are JIT-compiled the first time they return,
                # so print a message s.t. we can visualize compilation times:
                if os.getenv("CHECK_AND_COMPILE_SILENT") != "1":
                    print(f"Compiling {name}...")

                # Return the output (and, implicitly, kickstart JIT-compilation):
                return y

            f_not_compiled = jaxtyped(typechecker=beartype)(f)

            # The above function returns a (possible) error as well as its input,
            # so we need to make sure we don't forget to check the error & blow past it:
            def f_compiled(*args, **kwargs):

                # JIT-compile the above wrapped function:
                g = jit(checkified, static_argnums=static_argnums)

                # Call it:
                err, y = g(*args, **kwargs)

                # If it wasn't successful, throw the error:
                err.throw()

                # Otherwise, return the successful value:
                return y

            # Check if we're already inside a compiled function;
            # in that case, since JAX inlines everything,
            # don't waste time compiling this as well.
            # Check strategy taken from <https://github.com/google/jax/discussions/9241>
            return lambda *args, **kwargs: (
                f_not_compiled if isinstance(jnp.empty([]), core.Tracer) else f_compiled
            )(*args, **kwargs)

        # Return a decorator that does all of the above to a function:
        return partially_applied
