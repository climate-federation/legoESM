"""FV3_3D iter 829: effective_radiating_temperature_fv3.

T_eff = (OLR / (ε·σ))^(1/4).

Tests
-----

1. ``test_earth_mean``: OLR=240 → T_eff ≈ 255 K.
2. ``test_planck_inverse``: σT⁴ → T (round trip).
3. ``test_lower_emissivity``: ε<1 → higher T_eff for same OLR.
4. ``test_olr_floored``: OLR=0 → finite via floor.
5. ``test_monotone_olr``: ↑OLR → ↑T_eff.
6. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import effective_radiating_temperature_fv3


def test_earth_mean():
    """Earth global-mean OLR=240 W/m², ε=1 → T_eff ≈ 255 K."""
    olr = jnp.array([240.0])
    t_eff = effective_radiating_temperature_fv3(olr)
    expected = (240.0 / constants.sigma_sb) ** 0.25
    np.testing.assert_allclose(np.asarray(t_eff), [expected], rtol=1e-12)
    assert 254.0 < float(t_eff[0]) < 256.0


def test_planck_inverse():
    """Round trip: σ·T^4 → T."""
    t_known = jnp.array([255.0, 288.0, 300.0])
    olr = constants.sigma_sb * t_known ** 4
    t_recovered = effective_radiating_temperature_fv3(olr)
    np.testing.assert_allclose(
        np.asarray(t_recovered), np.asarray(t_known), rtol=1e-12
    )


def test_lower_emissivity():
    """ε=0.5 → T_eff higher than ε=1 at same OLR (must compensate)."""
    olr = jnp.array([240.0])
    t_unit = effective_radiating_temperature_fv3(olr, emissivity=1.0)
    t_half = effective_radiating_temperature_fv3(olr, emissivity=0.5)
    assert float(t_half[0]) > float(t_unit[0])


def test_olr_floored():
    """OLR=0 → finite via floor."""
    olr = jnp.array([0.0])
    t_eff = effective_radiating_temperature_fv3(olr, olr_floor=1e-6)
    assert jnp.all(jnp.isfinite(t_eff))


def test_monotone_olr():
    """↑OLR → ↑T_eff."""
    olr = jnp.array([100.0, 200.0, 300.0, 400.0])
    t = effective_radiating_temperature_fv3(olr)
    assert jnp.all(jnp.diff(t) > 0.0)


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=829)
    n_x, n_y = 6, 8
    olr = jnp.asarray(rng.uniform(50.0, 400.0, size=(n_x, n_y)))
    eps = jnp.asarray(rng.uniform(0.5, 1.0, size=(n_x, n_y)))
    t = effective_radiating_temperature_fv3(olr, eps)
    assert t.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(t))
    assert jnp.all(t > 0.0)
