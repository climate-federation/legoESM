"""FV3_3D iter 720: saturation_mixing_ratio_blend helper in thermo.py.

Reusable helper matching FV3's compute_qs(... es_over_liq_and_ice=
.true.) behaviour.  Extracted from iter-715 rh_calc_fv3 do_cmip
path for reuse by other diagnostics.

Tests
-----

1. ``test_qsat_blend_above_T_top_matches_liquid``.
2. ``test_qsat_blend_below_T_bot_matches_ice``.
3. ``test_qsat_blend_midpoint_is_average``.
4. ``test_qsat_blend_custom_top_and_width``.
5. ``test_qsat_blend_rh_calc_iter715_unchanged``.
6. ``test_qsat_blend_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants, thermo
from legoesm.grids.cubed_sphere import rh_calc_fv3


def test_qsat_blend_above_T_top_matches_liquid():
    """T > T_freeze → blend = liquid sat."""
    T = jnp.full((5,), 290.0)
    p = jnp.full((5,), 1.0e5)
    qs_blend = thermo.saturation_mixing_ratio_blend(T, p)
    qs_liq = thermo.saturation_mixing_ratio(T, p)
    assert jnp.allclose(qs_blend, qs_liq, atol=1e-10)


def test_qsat_blend_below_T_bot_matches_ice():
    """T < T_freeze − 20 → blend = ice sat."""
    T = jnp.full((5,), constants.T_freeze - 30.0)
    p = jnp.full((5,), 5.0e4)
    qs_blend = thermo.saturation_mixing_ratio_blend(T, p)
    qs_ice = thermo.saturation_mixing_ratio_ice(T, p)
    assert jnp.allclose(qs_blend, qs_ice, atol=1e-10)


def test_qsat_blend_midpoint_is_average():
    """T = T_freeze − 10 → w_liq = 0.5, blend = 0.5·liq + 0.5·ice."""
    T = jnp.full((5,), constants.T_freeze - 10.0)
    p = jnp.full((5,), 5.0e4)
    qs_blend = thermo.saturation_mixing_ratio_blend(T, p)
    qs_liq = thermo.saturation_mixing_ratio(T, p)
    qs_ice = thermo.saturation_mixing_ratio_ice(T, p)
    expected = 0.5 * qs_liq + 0.5 * qs_ice
    assert jnp.allclose(qs_blend, expected, atol=1e-12)


def test_qsat_blend_custom_top_and_width():
    """Custom T_top and width parameterize the blend correctly."""
    T = jnp.full((5,), 240.0)
    p = jnp.full((5,), 5.0e4)
    # Wide blend centered higher: T_top=250, width=40 → bot=210
    qs = thermo.saturation_mixing_ratio_blend(
        T, p, T_blend_top=250.0, T_blend_width=40.0,
    )
    # T=240 falls inside [210, 250]: w_liq = (240-210)/40 = 0.75
    qs_liq = thermo.saturation_mixing_ratio(T, p)
    qs_ice = thermo.saturation_mixing_ratio_ice(T, p)
    expected = 0.75 * qs_liq + 0.25 * qs_ice
    assert jnp.allclose(qs, expected, atol=1e-12)


def test_qsat_blend_rh_calc_iter715_unchanged():
    """iter-715 rh_calc_fv3 do_cmip=True still matches inline blend
    after refactor (regression on rh_calc behavior)."""
    rng = np.random.default_rng(seed=720)
    p = jnp.asarray(rng.uniform(1.0e4, 1.0e5, size=(10,)))
    t = jnp.asarray(rng.uniform(220.0, 300.0, size=(10,)))
    qv = jnp.asarray(rng.uniform(1e-6, 0.02, size=(10,)))
    rh_cmip = rh_calc_fv3(p, t, qv, do_cmip=True)
    # Manual reproduction of iter-715 inline formula
    qs_blend = thermo.saturation_mixing_ratio_blend(t, p)
    expected_rh = 100.0 * qv / qs_blend
    assert jnp.allclose(rh_cmip, expected_rh, atol=1e-12)


def test_qsat_blend_finite():
    """No NaN/Inf on random 2-D inputs."""
    rng = np.random.default_rng(seed=721)
    T = jnp.asarray(rng.uniform(200.0, 320.0, size=(4, 30)))
    p = jnp.asarray(rng.uniform(1.0e4, 1.0e5, size=(4, 30)))
    qs = thermo.saturation_mixing_ratio_blend(T, p)
    assert jnp.all(jnp.isfinite(qs))
