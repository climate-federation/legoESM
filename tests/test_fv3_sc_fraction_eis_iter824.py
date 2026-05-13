"""FV3_3D iter 824: sc_fraction_eis_fv3 (Wood-Bretherton 2006 fit).

f_low = clip(slope·EIS + intercept, 0, 1).
Default slope=0.06, intercept=0.41.

Tests
-----

1. ``test_sc_zero_eis``: EIS=0 → f_low=0.41.
2. ``test_sc_strong_inversion``: EIS=10 → clipped to 1.
3. ``test_sc_unstable_clipped_zero``: EIS=-10 → clipped to 0.
4. ``test_sc_monotone``: ↑EIS → ↑f_low.
5. ``test_sc_custom_coeffs``: custom slope/intercept overrides.
6. ``test_sc_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import sc_fraction_eis_fv3


def test_sc_zero_eis():
    """EIS=0 → f_low = intercept (0.41)."""
    eis = jnp.array([0.0])
    f = sc_fraction_eis_fv3(eis)
    np.testing.assert_allclose(np.asarray(f), [0.41], rtol=1e-12)


def test_sc_strong_inversion():
    """EIS=10 → 0.06·10+0.41 = 1.01 → clipped to 1."""
    eis = jnp.array([10.0])
    f = sc_fraction_eis_fv3(eis)
    np.testing.assert_allclose(np.asarray(f), [1.0], rtol=1e-12)


def test_sc_unstable_clipped_zero():
    """EIS=-10 → 0.06·(-10)+0.41 = -0.19 → clipped to 0."""
    eis = jnp.array([-10.0])
    f = sc_fraction_eis_fv3(eis)
    np.testing.assert_allclose(np.asarray(f), [0.0], atol=1e-12)


def test_sc_monotone():
    """Monotone non-decreasing in EIS."""
    eis = jnp.array([-5.0, -2.0, 0.0, 2.0, 4.0, 6.0, 10.0])
    f = sc_fraction_eis_fv3(eis)
    diffs = jnp.diff(f)
    assert jnp.all(diffs >= 0.0)


def test_sc_custom_coeffs():
    """Custom slope/intercept override."""
    eis = jnp.array([5.0])
    # slope=0.1, intercept=0.3 → 0.1·5+0.3 = 0.8
    f = sc_fraction_eis_fv3(eis, slope=0.1, intercept=0.3)
    np.testing.assert_allclose(np.asarray(f), [0.8], rtol=1e-12)


def test_sc_shapes_finite():
    """3-D shapes preserved, finite, in [0, 1]."""
    rng = np.random.default_rng(seed=824)
    n_x, n_y = 6, 8
    eis = jnp.asarray(rng.uniform(-15.0, 15.0, size=(n_x, n_y)))
    f = sc_fraction_eis_fv3(eis)
    assert f.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(f))
    assert jnp.all(f >= 0.0)
    assert jnp.all(f <= 1.0)
