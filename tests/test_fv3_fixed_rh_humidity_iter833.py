"""FV3_3D iter 833: fixed_rh_humidity_change_fv3.

q_new = q_old · q_sat(T_new, p) / q_sat(T_old, p).

Tests
-----

1. ``test_identity``: T_new = T_old → q_new = q_old.
2. ``test_rh_preserved``: RH = q/q_sat invariant before/after.
3. ``test_small_dt_matches_cc``: small ΔT matches iter-832 linear
   CC scaling to <1% error.
4. ``test_finite_warming_amplifies``: ΔT=3 K → q increases ~22%.
5. ``test_cooling_dries``: T_new < T_old → q_new < q_old.
6. ``test_cold_q_sat_floor``: T_old=0 floored, no NaN.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants, thermo
from legoesm.grids.cubed_sphere import (
    clausius_clapeyron_dqdt_fv3,
    fixed_rh_humidity_change_fv3,
)


def test_identity():
    """T_new = T_old → q_new = q_old."""
    q = jnp.array([0.01])
    t = jnp.array([288.0])
    p = jnp.array([1.0e5])
    q_new = fixed_rh_humidity_change_fv3(q, t, t, p)
    np.testing.assert_allclose(np.asarray(q_new), np.asarray(q), rtol=1e-12)


def test_rh_preserved():
    """RH invariant before/after at fixed p."""
    q_old = jnp.array([0.008])
    t_old = jnp.array([288.0])
    t_new = jnp.array([291.0])
    p = jnp.array([1.0e5])
    rh_old = q_old / thermo.saturation_mixing_ratio(t_old, p)
    q_new = fixed_rh_humidity_change_fv3(q_old, t_old, t_new, p)
    rh_new = q_new / thermo.saturation_mixing_ratio(t_new, p)
    np.testing.assert_allclose(np.asarray(rh_new), np.asarray(rh_old), rtol=1e-12)


def test_small_dt_matches_cc():
    """Small ΔT: finite-diff vs iter-832 analytic CC < 1% error."""
    t_old = jnp.array([288.0])
    p = jnp.array([1.0e5])
    q_sat_old = thermo.saturation_mixing_ratio(t_old, p)
    q_old = 0.5 * q_sat_old  # RH=0.5
    dt = 0.01
    q_new = fixed_rh_humidity_change_fv3(q_old, t_old, t_old + dt, p)
    dq_finite = (q_new - q_old) / dt
    dq_analytic = clausius_clapeyron_dqdt_fv3(t_old, p, rh=0.5)
    rel_err = float(
        jnp.abs((dq_finite[0] - dq_analytic[0]) / dq_analytic[0])
    )
    assert rel_err < 0.01, f"finite/CC diverge {rel_err*100:.2f}%"


def test_finite_warming_amplifies():
    """ΔT=3 K at T=288 K → q amplified ~20-23% (Earth-mean CC)."""
    q_old = jnp.array([0.008])
    t_old = jnp.array([288.0])
    t_new = jnp.array([291.0])
    p = jnp.array([1.0e5])
    q_new = fixed_rh_humidity_change_fv3(q_old, t_old, t_new, p)
    ratio = float(q_new[0] / q_old[0])
    assert 1.18 < ratio < 1.25, f"3K amplification ratio {ratio} off CC band"


def test_cooling_dries():
    """T_new < T_old → q_new < q_old (fixed-RH cooling)."""
    q_old = jnp.array([0.01])
    t_old = jnp.array([290.0])
    t_new = jnp.array([280.0])
    p = jnp.array([1.0e5])
    q_new = fixed_rh_humidity_change_fv3(q_old, t_old, t_new, p)
    assert float(q_new[0]) < float(q_old[0])


def test_cold_q_sat_floor():
    """T_old=0 floored, no NaN."""
    q_old = jnp.array([1.0e-10])
    t_old = jnp.array([0.0])
    t_new = jnp.array([250.0])
    p = jnp.array([1.0e5])
    q_new = fixed_rh_humidity_change_fv3(q_old, t_old, t_new, p)
    assert jnp.all(jnp.isfinite(q_new))


def test_shapes_finite():
    """3-D shapes preserved, finite, non-negative."""
    rng = np.random.default_rng(seed=833)
    n_x, n_y, n_z = 4, 5, 6
    t_old = jnp.asarray(rng.uniform(240.0, 310.0, size=(n_x, n_y, n_z)))
    t_new = t_old + jnp.asarray(
        rng.uniform(-5.0, 5.0, size=(n_x, n_y, n_z))
    )
    p = jnp.asarray(rng.uniform(2.0e4, 1.0e5, size=(n_x, n_y, n_z)))
    q_sat_old = thermo.saturation_mixing_ratio(t_old, p)
    q_old = 0.6 * q_sat_old
    q_new = fixed_rh_humidity_change_fv3(q_old, t_old, t_new, p)
    assert q_new.shape == (n_x, n_y, n_z)
    assert jnp.all(jnp.isfinite(q_new))
    assert jnp.all(q_new >= 0.0)
