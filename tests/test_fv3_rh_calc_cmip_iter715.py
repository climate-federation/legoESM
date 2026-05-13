"""FV3_3D iter 715: rh_calc_fv3 do_cmip upgrade.

iter-695 used saturation over liquid only.  iter-715 adds the
``do_cmip=True`` flag that switches to FV3-faithful
``compute_qs(... es_over_liq_and_ice=.true.)`` path — pure liquid
above T_freeze, pure ice below T_freeze − 20 K, linear blend in
between.

Tests
-----

1. ``test_rh_cmip_above_freeze_matches_liquid``.
2. ``test_rh_cmip_far_below_freeze_matches_ice``.
3. ``test_rh_cmip_blend_at_T_minus_10``.
4. ``test_rh_cmip_default_matches_iter695``.
5. ``test_rh_cmip_shapes_3d``.
6. ``test_rh_cmip_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants, thermo
from legoesm.grids.cubed_sphere import rh_calc_fv3


def test_rh_cmip_above_freeze_matches_liquid():
    """T = 290 K (above freeze) → CMIP RH == liquid-only RH."""
    p = jnp.full((5,), 1.0e5)
    t = jnp.full((5,), 290.0)
    qv = jnp.full((5,), 0.005)
    rh_liq = rh_calc_fv3(p, t, qv, do_cmip=False)
    rh_cmip = rh_calc_fv3(p, t, qv, do_cmip=True)
    assert jnp.all(jnp.abs(rh_liq - rh_cmip) < 1e-10)


def test_rh_cmip_far_below_freeze_matches_ice():
    """T < T_freeze − 20 → CMIP RH uses pure ice saturation."""
    p = jnp.full((5,), 5.0e4)
    t = jnp.full((5,), constants.T_freeze - 30.0)
    qv = jnp.full((5,), 1.0e-4)
    qs_ice = thermo.saturation_mixing_ratio_ice(t, p)
    expected_rh = 100.0 * qv / qs_ice
    rh_cmip = rh_calc_fv3(p, t, qv, do_cmip=True)
    assert jnp.all(jnp.abs(rh_cmip - expected_rh) < 1e-10)


def test_rh_cmip_blend_at_T_minus_10():
    """At T = T_freeze − 10 K, blend weight w_liq = 0.5."""
    p = jnp.full((5,), 5.0e4)
    t = jnp.full((5,), constants.T_freeze - 10.0)
    qv = jnp.full((5,), 1.0e-4)
    qs_liq = thermo.saturation_mixing_ratio(t, p)
    qs_ice = thermo.saturation_mixing_ratio_ice(t, p)
    qs_blend = 0.5 * qs_liq + 0.5 * qs_ice
    expected_rh = 100.0 * qv / qs_blend
    rh_cmip = rh_calc_fv3(p, t, qv, do_cmip=True)
    assert jnp.all(jnp.abs(rh_cmip - expected_rh) < 1e-10)


def test_rh_cmip_default_matches_iter695():
    """Default do_cmip=False keeps iter-695 (liquid-only) behavior."""
    rng = np.random.default_rng(seed=715)
    p = jnp.asarray(rng.uniform(1.0e4, 1.0e5, size=(5,)))
    t = jnp.asarray(rng.uniform(220.0, 300.0, size=(5,)))
    qv = jnp.asarray(rng.uniform(1e-6, 0.01, size=(5,)))
    rh_default = rh_calc_fv3(p, t, qv)
    rh_explicit = rh_calc_fv3(p, t, qv, do_cmip=False)
    assert jnp.allclose(rh_default, rh_explicit, atol=1e-15)


def test_rh_cmip_shapes_3d():
    """3-D input → 3-D output."""
    rng = np.random.default_rng(seed=716)
    n_x, n_y, km = 4, 5, 20
    p = jnp.asarray(rng.uniform(1.0e4, 1.0e5, size=(n_x, n_y, km)))
    t = jnp.asarray(rng.uniform(220.0, 300.0, size=(n_x, n_y, km)))
    qv = jnp.asarray(rng.uniform(0.0, 0.02, size=(n_x, n_y, km)))
    rh = rh_calc_fv3(p, t, qv, do_cmip=True)
    assert rh.shape == (n_x, n_y, km)


def test_rh_cmip_finite():
    """No NaN/Inf in CMIP path."""
    rng = np.random.default_rng(seed=717)
    n_x, n_y, km = 4, 4, 30
    p = jnp.asarray(rng.uniform(1.0e4, 1.0e5, size=(n_x, n_y, km)))
    t = jnp.asarray(rng.uniform(200.0, 320.0, size=(n_x, n_y, km)))
    qv = jnp.asarray(rng.uniform(1e-8, 0.025, size=(n_x, n_y, km)))
    rh = rh_calc_fv3(p, t, qv, do_cmip=True)
    assert jnp.all(jnp.isfinite(rh))
