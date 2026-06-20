"""Regression: WENO-Z float32 common-shift conditioning (large-mean/small-fluctuation).

A field with a large mean and small fluctuations (e.g. θ≈265 K with O(0.1 K) eddies)
makes the WENO-Z smoothness indicators β — formed from raw-value squared differences —
catastrophically cancel in float32, producing NaN. The reconstruction is shift-invariant,
so float32 is common-shifted by the central stencil value (see core.weno._weno_z_shifted);
float64 is left bit-identical to the unshifted core.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core import weno


def _stencil(mean, fluct, dtype, n=6):
    key = jax.random.PRNGKey(0)
    return [(mean + fluct * jax.random.normal(jax.random.fold_in(key, k), (32,))).astype(dtype)
            for k in range(n)]


def test_weno5_f32_large_mean_is_finite_and_matches_f64():
    """float32 weno5 on θ≈265±0.1 is finite (no β cancellation) and tracks float64."""
    f32 = _stencil(265.0, 0.1, jnp.float32)
    fp32_p, fp32_m = weno.weno5_z(f32)
    assert np.all(np.isfinite(np.asarray(fp32_p)))
    assert np.all(np.isfinite(np.asarray(fp32_m)))
    # same stencil in float64 (reference); the shifted f32 should track it closely
    jax.config.update("jax_enable_x64", True)
    f64 = [x.astype(jnp.float64) for x in f32]
    fp64_p, fp64_m = weno.weno5_z(f64)
    np.testing.assert_allclose(np.asarray(fp32_p), np.asarray(fp64_p), rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(np.asarray(fp32_m), np.asarray(fp64_m), rtol=2e-3, atol=2e-3)


def test_weno5_f64_bit_identical_to_unshifted_core():
    """The float64 path must be UNCHANGED (bit-identical to the raw _weno_z_core) —
    the shift is float32-only, so no float64 regression anywhere it is used."""
    jax.config.update("jax_enable_x64", True)
    f = _stencil(265.0, 0.5, jnp.float64)
    eps = weno._default_eps(f[0])
    shifted = weno.weno5_z(f)
    raw = weno._weno_z_core(list(f), 5, eps)
    for a, b in zip(shifted, raw):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_weno5_constant_field_recovers_constant():
    """Constant stencil ⇒ reconstruction equals the constant (shift-exactness)."""
    for dtype in (jnp.float32,):
        c = jnp.full((16,), 300.0, dtype)
        fp, fm = weno.weno5_z([c] * 6)
        np.testing.assert_allclose(np.asarray(fp), 300.0, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(fm), 300.0, rtol=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
