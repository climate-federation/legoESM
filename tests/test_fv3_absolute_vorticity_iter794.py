"""FV3_3D iter 794: absolute_vorticity_fv3 (η = ζ + f).

Composes iter-778 ``coriolis_parameter_fv3``.

Tests
-----

1. ``test_eta_pure_planetary``: ζ=0 → η=f.
2. ``test_eta_pure_relative``: f=0 → η=ζ.
3. ``test_eta_cancellation``: ζ=-f → η=0.
4. ``test_eta_composes_iter778``: (ζ, lat) → f → η.
5. ``test_eta_anticyclonic_negative``: TC eyewall |ζ|>|f|, sign matters.
6. ``test_eta_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    absolute_vorticity_fv3,
    coriolis_parameter_fv3,
)


def test_eta_pure_planetary():
    """ζ=0 → η=f."""
    zeta = jnp.zeros((3,))
    f = jnp.array([1e-4, -1e-4, 5e-5])
    eta = absolute_vorticity_fv3(zeta, f)
    np.testing.assert_allclose(np.asarray(eta), np.asarray(f), atol=1e-15)


def test_eta_pure_relative():
    """f=0 → η=ζ."""
    zeta = jnp.array([1e-4, -2e-4, 5e-5])
    f = jnp.zeros((3,))
    eta = absolute_vorticity_fv3(zeta, f)
    np.testing.assert_allclose(np.asarray(eta), np.asarray(zeta), atol=1e-15)


def test_eta_cancellation():
    """ζ=-f → η=0."""
    f = jnp.array([1e-4, -1e-4, 5e-5])
    zeta = -f
    eta = absolute_vorticity_fv3(zeta, f)
    np.testing.assert_allclose(np.asarray(eta), jnp.zeros((3,)), atol=1e-15)


def test_eta_composes_iter778():
    """Pipeline (ζ, lat) → f → η."""
    zeta = jnp.array([1.0e-5])
    f = coriolis_parameter_fv3(jnp.array([45.0]), units="deg")
    eta = absolute_vorticity_fv3(zeta, f)
    # |f|=1.03e-4, η = 1.03e-4 + 1e-5 = 1.13e-4
    expected = float(f[0]) + 1.0e-5
    np.testing.assert_allclose(np.asarray(eta), [expected], rtol=1e-12)


def test_eta_anticyclonic_negative():
    """Strong anticyclone (ζ<0, |ζ|>|f|): η changes sign."""
    f = jnp.array([1.0e-4])
    zeta_strong_anti = jnp.array([-3.0e-4])
    eta = absolute_vorticity_fv3(zeta_strong_anti, f)
    # η = 1e-4 + (-3e-4) = -2e-4 (sign flipped)
    assert float(eta[0]) < 0.0


def test_eta_shapes_3d_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=794)
    n_x, n_y, km = 4, 5, 20
    zeta = jnp.asarray(rng.uniform(-3e-4, 3e-4, size=(n_x, n_y, km)))
    f = jnp.asarray(rng.uniform(-1.5e-4, 1.5e-4, size=(n_x, n_y, km)))
    eta = absolute_vorticity_fv3(zeta, f)
    assert eta.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(eta))
