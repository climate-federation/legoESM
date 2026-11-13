"""Dtype fixed-point helper retained after retiring tiled cube atmosphere."""
import jax.numpy as jnp


def test_unroll_to_dtype_fixed_point_public():
    """Direct unit for the PROMOTED helper: a mixed-dtype carry unrolls
    exactly until the step's output dtypes are a fixed point; a stable
    carry unrolls nothing."""
    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
        unroll_to_dtype_fixed_point,
    )

    def step1(s):
        # First application promotes 'a' f32->f64 (mixing the f64 'b');
        # then stable.
        return {"a": s["a"] + 0.0 * s["b"], "b": s["b"]}

    mixed = {"a": jnp.zeros(3, jnp.float32), "b": jnp.zeros(3, jnp.float64)}
    out, n_left = unroll_to_dtype_fixed_point(step1, mixed, 5)
    assert n_left == 4
    assert out["a"].dtype == jnp.float64

    stable = {"a": jnp.zeros(3, jnp.float64), "b": jnp.zeros(3, jnp.float64)}
    out, n_left = unroll_to_dtype_fixed_point(step1, stable, 5)
    assert n_left == 5
    assert out is stable
