from beartype import beartype
from beartype.typing import Callable
from jaxtyping import jaxtyped
import os


if os.getenv("NO_JIT") == "1":  # pragma: no cover

    print(
        "*** NOTE: `NO_JIT` environment variable is `1`, "
        "so no functions marked `@check_and_compile(...)` will be compiled!"
    )

    # If we're not JIT-compiling, just typecheck:
    def jit(*static_argnums) -> Callable[[Callable], Callable]:
        return jaxtyped(typechecker=beartype)  # itself a function

else:  # pragma: no cover

    from jax import jit as jax_jit
    from jax.experimental.checkify import checkify, all_checks

    # There's lots of functions returning functions here,
    # so I've left copious documentation to clarify what everything does:
    def jit(*static_argnums) -> Callable[[Callable], Callable]:

        def partially_applied(f: Callable) -> Callable:

            # Define a wrapped function that we'll compile below:
            def checkified(*args, **kwargs):

                # Type-check the function:
                g = jaxtyped(f, typechecker=beartype)

                # Functionalize runtime checks:
                g = checkify(g, errors=all_checks)

                # Call it:
                y = g(*args, **kwargs)

                # Functions are JIT-compiled the first time they return,
                # so print a message so we can visualize compilation times:
                if os.getenv("CHECK_AND_COMPILE_SILENT") != "1":
                    name: str = getattr(f, "__qualname__", "an unnamed function")
                    print(f"Compiling {name}...")

                # Return the output (and, implicitly, kickstart JIT-compilation):
                return y

            # The above function returns a (possible) error as well as its input,
            # so we need to make sure we don't forget to check the error & blow past it:
            def handle_err(*args, **kwargs):

                # JIT-compile the above wrapped function:
                g = jax_jit(checkified, static_argnums=static_argnums)

                # Call it:
                err, y = g(*args, **kwargs)

                # If it wasn't successful, throw the error:
                err.throw()

                # Otherwise, return the successful value:
                return y

            # Return the above function, batteries included:
            return handle_err

        # Return a decorator that does all of the above to a function:
        return partially_applied
