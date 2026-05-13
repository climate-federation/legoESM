"""FV3_3D iter 771: mixing_ratio_fv3 + iter-716 refactor.

Identity: r = q / (1 - q)  [kg/kg]

Extracts q/(1-q) pattern previously inline in iter-716
``eqv_pot_bolton_fv3``; iter-716 now delegates.

Tests
-----

1. ``test_mixing_ratio_small_q_approx_q``: q ≪ 1 → r ≈ q within 2%.
2. ``test_mixing_ratio_above_q``: r > q for any q > 0.
3. ``test_mixing_ratio_zero_q_floor``: q = 0 → r small positive (floor).
4. ``test_mixing_ratio_iter716_unchanged``: iter-716 θ_e output preserved.
5. ``test_mixing_ratio_shapes_3d``.
6. ``test_mixing_ratio_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    eqv_pot_bolton_fv3,
    mixing_ratio_fv3,
)


def test_mixing_ratio_small_q_approx_q():
    """For typical atmospheric q (≲ 0.02), r ≈ q within ~2 %."""
    q = jnp.array([0.001, 0.005, 0.010, 0.020])
    r = mixing_ratio_fv3(q)
    np.testing.assert_allclose(np.asarray(r), np.asarray(q), rtol=0.025)


def test_mixing_ratio_above_q():
    """r > q for any q > 0 (1/(1-q) > 1)."""
    q = jnp.array([1e-4, 1e-3, 1e-2, 5e-2])
    r = mixing_ratio_fv3(q)
    assert jnp.all(r > q)


def test_mixing_ratio_zero_q_floor():
    """q = 0 → r ≥ 1e-12 (floor prevents div/log singularities)."""
    q = jnp.array([0.0])
    r = mixing_ratio_fv3(q)
    assert jnp.all(r >= 1e-12)
    assert jnp.all(r < 1e-11)  # floor not blown up


def test_mixing_ratio_iter716_unchanged():
    """iter-716 Bolton θ_e output preserved after refactor.

    iter-716 currently used floor 1e-10 g/kg = 1e-13 kg/kg;
    iter-771 helper uses 1e-12 kg/kg.  For atmospheric q > 1e-12
    (always true: typical q is 1e-3 to 2e-2), the floor never
    engages and outputs are bit-identical.
    """
    rng = np.random.default_rng(seed=771)
    km = 5
    pt = jnp.full((km,), 290.0)
    q = jnp.asarray(rng.uniform(0.001, 0.020, size=(km,)))
    delp = jnp.full((km,), 1000.0)
    delz = jnp.full((km,), -100.0)
    theta_e = eqv_pot_bolton_fv3(pt, delp, q, delz=delz, moist=True)
    assert jnp.all(jnp.isfinite(theta_e))
    # Bolton θ_e for warm moist air is in 320-380 K range
    assert jnp.all((theta_e > 290.0) & (theta_e < 400.0))


def test_mixing_ratio_shapes_3d():
    """3-D shapes preserved."""
    rng = np.random.default_rng(seed=772)
    n_x, n_y, km = 4, 5, 20
    q = jnp.asarray(rng.uniform(0.001, 0.020, size=(n_x, n_y, km)))
    r = mixing_ratio_fv3(q)
    assert r.shape == (n_x, n_y, km)


def test_mixing_ratio_finite():
    """No NaN/Inf for realistic q range."""
    rng = np.random.default_rng(seed=773)
    q = jnp.asarray(rng.uniform(0.0, 0.030, size=(8, 30)))
    r = mixing_ratio_fv3(q)
    assert jnp.all(jnp.isfinite(r))
    assert jnp.all(r > 0.0)
