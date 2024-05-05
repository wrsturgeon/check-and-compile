from beartype import beartype
from beartype.typing import Callable
from jax import core, jit, numpy as jnp
from jax.experimental.checkify import checkify  # all_checks
from jaxtyping import jaxtyped
import os


should_jit = os.getenv("NO_JIT") != "1"
if not should_jit:
    print(
        "*** NOTE: `NO_JIT` environment variable is `1`, so "
        "functions marked `@check_and_compile(...)` will *not* be compiled!"
    )


# Check if we're already inside a compiled function.
# In that case, since JAX inlines everything,
# don't waste time compiling an internal function as well.
# Check strategy taken from <https://github.com/google/jax/discussions/9241>
def being_traced() -> bool:
    return isinstance(jnp.empty([]), core.Tracer)


# There's lots of functions-returning-functions here,
# so I've left copious documentation to clarify what everything does:
def check_and_compile(*static_argnums) -> Callable[[Callable], Callable]:

    def partially_applied(f: Callable) -> Callable:

        # Get this function's name:
        name: str = getattr(f, "__qualname__", "an unnamed function")

        # Define a wrapped function that we'll compile below:
        def checkified(*args, **kwargs):

            # Type-check the function:
            g = jaxtyped(f, typechecker=beartype)

            # Functionalize runtime checks:
            g = checkify(g)  # errors=all_checks)

            # Call it:
            y = g(*args, **kwargs)

            # Return the output (and, implicitly, kickstart JIT-compilation):
            return y

        # Define both functions now, since their exact IDs apparently matter to JAX:

        # The above function returns a (possible) error as well as its input,
        # so we need to make sure we don't forget to check the error & blow past it:
        def f_not_compiled(*args, **kwargs):

            # Call it:
            err, y = checkified(*args, **kwargs)

            # If it wasn't successful, throw the error:
            err.throw()

            # Otherwise, return the successful value:
            return y

        # Functions are JIT-compiled the first time they return,
        # so print a message s.t. we can visualize compilation times:
        def with_message(*args, **kwargs):

            if os.getenv("CHECK_AND_COMPILE_SILENT") != "1":
                print(f"Calling {name}...")

            y = checkified(*args, **kwargs)

            if os.getenv("CHECK_AND_COMPILE_SILENT") != "1":
                print(f"Compiling {name}...")

            return y

        # Same comment as the above function
        def f_compiled(*args, **kwargs):

            # JIT-compile the above wrapped function:
            g = jit(with_message, static_argnums=static_argnums)

            # Call it:
            err, y = g(*args, **kwargs)

            # If it wasn't successful, throw the error:
            err.throw()

            # Otherwise, return the successful value:
            return y

        def choose_at_runtime(*args, **kwargs):
            compile_toplevel = should_jit and not being_traced()
            call = f_compiled if compile_toplevel else f_not_compiled
            return call(*args, **kwargs)

        return choose_at_runtime

    # Return a decorator that does all of the above to a function:
    return partially_applied
