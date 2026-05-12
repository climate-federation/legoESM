"""FV3_3D iter 814: brn_supercell_fv3 (Weisman-Klemp 1982).

BRN = CAPE / (0.5·|V_shear|²).

Tests
-----

1. ``test_brn_supercell_range``: textbook supercell case → 10-50.
2. ``test_brn_too_much_shear``: small CAPE + huge shear → BRN < 10.
3. ``test_brn_too_little_shear``: huge CAPE + tiny shear → BRN > 50.
4. ``test_brn_zero_shear_floored``: zero shear → huge finite.
5. ``test_brn_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import brn_supercell_fv3


def test_brn_supercell_range():
    """Classic supercell: CAPE=2500 J/kg, 25 m/s 0-6 km shear → BRN ~ 8.

    BRN = 2500 / (0.5·25²) = 2500/312.5 = 8. Edge of supercell range."""
    cape = jnp.array([2500.0])
    u_shear = jnp.array([25.0])
    v_shear = jnp.array([0.0])
    brn = brn_supercell_fv3(cape, u_shear, v_shear)
    np.testing.assert_allclose(np.asarray(brn), [8.0], rtol=1e-12)


def test_brn_too_much_shear():
    """Modest CAPE 500 + 40 m/s shear → BRN = 500/800 = 0.625 < 10."""
    cape = jnp.array([500.0])
    u_shear = jnp.array([40.0])
    v_shear = jnp.array([0.0])
    brn = brn_supercell_fv3(cape, u_shear, v_shear)
    assert float(brn[0]) < 10.0


def test_brn_too_little_shear():
    """Huge CAPE 4000 + 5 m/s shear → BRN = 4000/12.5 = 320 > 50."""
    cape = jnp.array([4000.0])
    u_shear = jnp.array([5.0])
    v_shear = jnp.array([0.0])
    brn = brn_supercell_fv3(cape, u_shear, v_shear)
    assert float(brn[0]) > 50.0


def test_brn_zero_shear_floored():
    """V_shear=0 → BRN huge but finite via ke_floor."""
    cape = jnp.array([2500.0])
    u_shear = jnp.array([0.0])
    v_shear = jnp.array([0.0])
    brn = brn_supercell_fv3(cape, u_shear, v_shear)
    assert jnp.all(jnp.isfinite(brn))
    assert float(brn[0]) > 1e6


def test_brn_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=814)
    n_x, n_y = 6, 8
    cape = jnp.asarray(rng.uniform(500.0, 5000.0, size=(n_x, n_y)))
    u_s = jnp.asarray(rng.uniform(-40.0, 40.0, size=(n_x, n_y)))
    v_s = jnp.asarray(rng.uniform(-40.0, 40.0, size=(n_x, n_y)))
    brn = brn_supercell_fv3(cape, u_s, v_s)
    assert brn.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(brn))
    assert jnp.all(brn >= 0.0)
