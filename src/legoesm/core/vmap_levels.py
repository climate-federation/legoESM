"""Generic vmap-over-levels wrapper for 2D operators."""

import jax
import jax.numpy as jnp
from functools import wraps


def vmap_over_levels(fn_2d):
    """Lift a 2D operator to 3D by vmapping over the last axis.

    The wrapped function accepts arrays whose last axis is the level
    dimension and returns arrays with the same trailing axis.

    Convention:
    - All positional arguments are array-like with shape (..., nlev).
    - All keyword arguments are non-array (grid, config, coefficients).
    - fn_2d returns a single array or a tuple of arrays.

    Usage::

        grad_x_3d = vmap_over_levels(fc_gradient_x)
        result = grad_x_3d(field_3d, grid=grid, fc_config=fc_config)
    """
    @wraps(fn_2d)
    def wrapper(*args, **kwargs):
        args_t = tuple(jnp.moveaxis(a, -1, 0) for a in args)

        def single_level(*a_k):
            return fn_2d(*a_k, **kwargs)

        result = jax.vmap(single_level)(*args_t)

        if isinstance(result, tuple):
            return tuple(jnp.moveaxis(r, 0, -1) for r in result)
        return jnp.moveaxis(result, 0, -1)

    return wrapper
