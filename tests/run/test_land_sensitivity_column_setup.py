"""Direct tests for ``scripts/run/land_sensitivity/column_setup.py``.

Covers the dispatch (unknown texture / biome raise), the physical ordering of
the texture thresholds, the forcing perturbation knobs, rollout finiteness and
eager-vs-jit parity, and the gradient-check helpers (including that a failing
check really raises).
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from scripts.run.land_sensitivity import column_setup as cs

jax.config.update("jax_enable_x64", True)

_LAT = float(np.deg2rad(40.0))


def test_unknown_texture_raises():
    with pytest.raises(ValueError, match="Unknown texture"):
        cs.make_column_config("silt")


def test_unknown_biome_raises():
    with pytest.raises(ValueError, match="Unknown biome"):
        cs.make_canopy_params(1, biome="not-a-biome")


@pytest.mark.parametrize("texture", sorted(cs.TEXTURES))
def test_thresholds_ordered(texture):
    cfg = cs.make_column_config(texture)
    assert 0.0 < cfg.theta_wp < cfg.theta_fc < float(cfg.hydraulics.theta_sat)


def test_sand_holds_less_plant_available_water_than_clay():
    sand, clay = cs.make_column_config("sand"), cs.make_column_config("clay")
    assert sand.theta_wp < clay.theta_wp
    assert sand.theta_fc < clay.theta_fc


def test_canopy_params_vegetated():
    p = cs.make_canopy_params(3, lai=2.5)
    assert p.LAI.shape == (3,)
    np.testing.assert_allclose(np.asarray(p.LAI), 2.5)
    np.testing.assert_allclose(np.asarray(p.FNonVeg), 0.0)
    assert float(p.Vcmax25_C3_leaf[0]) > 0.0 and float(p.hc[0]) > 0.0


def test_time_axis_wraps_doy_and_hour():
    axis = cs.make_time_axis(48, 3600.0, start_doy=364.5)
    assert axis.doy.shape == (48,)
    assert np.all((axis.doy >= 0.0) & (axis.doy < 365.0))
    assert np.isclose(axis.hour[12], 12.0) and np.isclose(axis.hour[24], 0.0)
    assert np.isclose(axis.doy[12], 0.0)          # 364.5 + 0.5 day wraps to 0
    with pytest.raises(ValueError):
        cs.make_time_axis(0, 3600.0, start_doy=1.0)


def test_forcing_knobs_touch_only_air_temperature_and_humidity():
    f0 = cs.column_forcing(_LAT, 0.0, 180.0, 12.0, 1e-5)
    f1 = cs.column_forcing(_LAT, 0.0, 180.0, 12.0, 1e-5, t_air_offset_k=2.0, q_air_scale=1.1)
    np.testing.assert_allclose(np.asarray(f1.T_lowest - f0.T_lowest), 2.0)
    np.testing.assert_allclose(np.asarray(f1.q_lowest / f0.q_lowest), 1.1)
    for name in ("sw_down", "lw_down", "precip_total", "u_lowest", "p_surface"):
        np.testing.assert_array_equal(np.asarray(getattr(f1, name)), np.asarray(getattr(f0, name)))


def test_rollout_finite_and_jit_parity():
    cfg = cs.make_column_config("loam")
    params = cs.make_canopy_params(1)
    state0 = cs.initial_state(cfg, t_init_k=288.0)
    axis = cs.make_time_axis(6, 3600.0, start_doy=180.0, start_hour=9.0)
    kw = dict(lat_rad=_LAT, lon_rad=0.0, time_axis=axis, dt_s=3600.0, precip=1e-5)
    final_e, hist_e, states = cs.rollout(state0, cfg, params, return_states=True, **kw)
    final_j, hist_j = jax.jit(lambda s: cs.rollout(s, cfg, params, **kw))(state0)
    assert hist_e.theta.shape == (6, 8) and hist_e.lhflx.shape == (6,)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in hist_e)
    np.testing.assert_allclose(np.asarray(hist_e.lhflx), np.asarray(hist_j.lhflx), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(final_e.theta_soil),
                               np.asarray(final_j.theta_soil), rtol=1e-10)
    # the stacked state trajectory ends at the final state
    np.testing.assert_array_equal(np.asarray(cs.state_at(states, 5).theta_soil),
                                  np.asarray(final_e.theta_soil))
    dz, z_node, z_iface = cs.soil_layers(cfg)
    assert dz.shape == (8,) and z_iface.shape == (9,)
    w = cs.column_water_kg_m2(final_e.theta_soil[0], dz)
    assert float(w) > 0.0


def test_central_difference_and_checks():
    def f(x):
        return jnp.sum(x ** 2)

    x = jnp.asarray([1.0, 2.0, 3.0])
    fd = cs.central_difference(f, x, 1, 1e-3)
    assert abs(fd - 4.0) < 1e-6
    fd1 = cs.central_difference(f, x, 1, 1e-3, one_sided=True)   # (x+e)^2 - x^2 = 4e + e^2
    assert abs(fd1 - 4.001) < 1e-6
    fd_s, smooth = cs.finite_difference(f, x, 1, 1e-3, atol=1e-9)
    assert smooth and abs(fd_s - 4.0) < 1e-6
    good = cs.gradient_check("ok", 4.0, 4.0 + 1e-6, rtol=1e-3, atol=0.0)
    bad = cs.gradient_check("bad", 4.0, 4.1, rtol=1e-3, atol=0.0)
    nan = cs.gradient_check("nan", float("nan"), 4.0, rtol=1e-3, atol=1.0)
    assert good.passed and not bad.passed and not nan.passed
    zero = cs.gradient_check("zero", 0.0, 0.0, rtol=1e-3, atol=1e-9)
    assert zero.passed
    cs.require_all_pass([good, zero])
    with pytest.raises(RuntimeError, match="gradient check FAILED"):
        cs.require_all_pass([good, bad])
    assert "FAIL" in cs.format_checks([bad])


def test_finite_difference_flags_a_jump_as_nonsmooth():
    def step(x):
        return jnp.where(x[1] > 2.0, 1.0, 0.0)

    x = jnp.asarray([1.0, 2.0, 3.0])
    fd, smooth = cs.finite_difference(step, x, 1, 1e-3, atol=1e-9)
    assert not smooth                       # 1/(2 eps) doubles when eps halves
    ns = cs.gradient_check("jump", 0.0, fd, rtol=1e-3, atol=1e-9, fd_consistent=smooth)
    assert not ns.passed and not ns.fd_consistent
    assert "NONSMOOTH" in cs.format_checks([ns])
    good = cs.gradient_check("ok", 4.0, 4.0, rtol=1e-3, atol=0.0)
    cs.require_all_pass([ns, good])         # a non-smooth point cannot arbitrate
    with pytest.raises(RuntimeError, match="unverified"):
        cs.require_all_pass([ns])           # ... but something smooth must have passed


def test_provenance_is_json_serialisable():
    p = cs.provenance({"a": 1, "path": __import__("pathlib").Path("x")}, extra=[1, 2])
    assert {"git_sha", "jax_version", "jax_enable_x64", "timestamp_utc", "args", "extra"} <= set(p)
    json.dumps(p)


def test_column_storage_counts_every_water_store():
    config = cs.make_column_config("loam")
    dz, _, _ = cs.soil_layers(config)
    state = cs.initial_state(config, t_init_k=290.0)
    soil = float(cs.column_water_kg_m2(state.theta_soil[0], dz))
    assert state.W_canopy is None                       # interception off in the demo column
    base = float(cs.column_storage_kg_m2(state, dz))
    np.testing.assert_allclose(base, soil + float(state.snow_depth[0])
                               + float(state.surface_water[0]) * constants.rho_water,
                               rtol=1e-12)
    wetter = state._replace(surface_water=state.surface_water + 0.002,   # 2 mm pond
                            snow_depth=state.snow_depth + 3.0,
                            W_canopy=jnp.asarray([0.5]))
    np.testing.assert_allclose(float(cs.column_storage_kg_m2(wetter, dz)) - base,
                               2.0 + 3.0 + 0.5, rtol=1e-12)
    no_pond = state._replace(surface_water=None)
    np.testing.assert_allclose(float(cs.column_storage_kg_m2(no_pond, dz)),
                               soil + float(state.snow_depth[0]), rtol=1e-12)
