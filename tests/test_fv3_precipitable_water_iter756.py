"""FV3_3D iter 756: precipitable_water_fv3 port.

PWV = column water vapor in kg/m^2 (numerically = mm).

Tests
-----

1. ``test_pwv_zero_q``.
2. ``test_pwv_known_tropical``.
3. ``test_pwv_units_kg_per_m2_equals_mm``.
4. ``test_pwv_shapes_3d``.
5. ``test_pwv_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import precipitable_water_fv3


def test_pwv_zero_q():
    """q=0 → PWV=0."""
    km = 5
    q = jnp.zeros((km,))
    delp = jnp.full((km,), 1.0e4)
    pwv = precipitable_water_fv3(q, delp)
    assert abs(float(pwv)) < 1e-15


def test_pwv_known_tropical():
    """Tropical q=0.015, p_s=1e5 → PWV ~ 153 kg/m² (huge, but uniform
    column unrealistic — useful as analytic check).

    PWV = q · p_s / g = 0.015 · 1e5 / 9.81 ≈ 152.9 kg/m².
    """
    km = 10
    q = jnp.full((km,), 0.015)
    delp = jnp.full((km,), 1.0e4)
    pwv = precipitable_water_fv3(q, delp)
    expected = 0.015 * 1.0e5 / constants.g
    assert abs(float(pwv) - expected) / expected < 1e-10


def test_pwv_units_kg_per_m2_equals_mm():
    """1 kg/m² of water ≡ 1 mm depth at ρ_water = 1000 kg/m³.

    Verify the unit identity (sanity, no actual unit conversion
    in code; just documents the equivalence)."""
    rho_water = 1000.0
    one_kg_per_m2_depth_m = 1.0 / rho_water    # = 1 mm
    assert abs(one_kg_per_m2_depth_m * 1000.0 - 1.0) < 1e-12   # = 1 mm


def test_pwv_shapes_3d():
    """3-D → 2-D output."""
    rng = np.random.default_rng(seed=756)
    n_x, n_y, km = 4, 5, 20
    q = jnp.asarray(rng.uniform(0.0, 0.020, size=(n_x, n_y, km)))
    delp = jnp.full((n_x, n_y, km), 5000.0)
    pwv = precipitable_water_fv3(q, delp)
    assert pwv.shape == (n_x, n_y)


def test_pwv_finite():
    """No NaN/Inf."""
    rng = np.random.default_rng(seed=757)
    q = jnp.asarray(rng.uniform(0.0, 0.025, size=(4, 4, 30)))
    delp = jnp.asarray(rng.uniform(500.0, 2000.0, size=(4, 4, 30)))
    pwv = precipitable_water_fv3(q, delp)
    assert jnp.all(jnp.isfinite(pwv))
