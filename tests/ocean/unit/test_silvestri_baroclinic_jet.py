"""Silvestri et al. 2024 §5 baroclinic-jet recipe: the buoyancy front, thermal-
wind IC, zonal-mean restoring, and the per-scheme builder."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
from legoesm.ocean.experiments import silvestri_baroclinic_jet as SJ
from legoesm.ocean.experiments.silvestri_schemes import SILVESTRI_JET_MAIN


@pytest.fixture(autouse=True)
def _fp64():
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def test_front_profile_shape():
    """B(φ): 1 at the south edge, 0 at the north edge, 1/2 at the centre,
    monotonically decreasing northward."""
    cfg = SJ.SilvestriJetConfig()
    lat = np.radians(np.linspace(cfg.lat_south, cfg.lat_north, 41))
    B = SJ.silvestri_front_B(lat, cfg)
    assert np.isclose(B[0], 1.0) and np.isclose(B[-1], 0.0)
    mid = SJ.silvestri_front_B(np.radians([cfg.lat_center]), cfg)[0]
    assert np.isclose(mid, 0.5, atol=1e-12)
    assert np.all(np.diff(B) <= 1e-12)                 # non-increasing northward
    assert np.all((B >= 0.0) & (B <= 1.0))


def test_initial_state_front_and_thermal_wind():
    """The IC carries the front in T (warmer south) and a thermal-wind u that
    vanishes at the bottom (purely baroclinic shear)."""
    from legoesm.grids.latlon import create_regional_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    cfg = SJ.SilvestriJetConfig()
    grid, _ = create_regional_latlon_grid(
        24, 16, cfg.lat_south, cfg.lat_north,
        lon_west=cfg.lon_west, lon_east=cfg.lon_east, periodic_x=True)
    z = create_ocean_z_star(n_levels=12, H_max=cfg.H_max)
    state, _ = SJ._build_initial_state(grid, z, cfg)
    T = np.asarray(state.T.data)
    u = np.asarray(state.u.data)
    # Front: zonal-mean surface T warmer in the south than the north (B: 1→0).
    Tzm_surf = np.nanmean(np.where(T[..., 0] != 0.0, T[..., 0], np.nan), axis=1)
    assert Tzm_surf[2] > Tzm_surf[-3], (Tzm_surf[2], Tzm_surf[-3])
    # Thermal wind: |u| at the surface (k=0) ≫ |u| at the bottom (≈0).
    surf = np.max(np.abs(u[:, :, 0]))
    bott = np.max(np.abs(u[:, :, -1]))
    assert surf > 10 * bott + 1e-12, (surf, bott)
    assert np.all(np.isfinite(T)) and np.all(np.isfinite(u))


def test_uniform_vertical_grid():
    """Paper §5 uses fixed dz=20 m (50 levels over 1 km) — the builder makes the
    vertical spacing uniform."""
    r = SJ.build_silvestri_baroclinic_jet_setup(n_lat=20, n_lon=16, nlev=50)
    dz = np.asarray(r.z_coord.dz_ref)
    assert np.allclose(dz, 20.0, atol=1e-9), (dz.min(), dz.max())


def test_deformation_radius_order():
    """L_d = N·H/(π|f|) ≈ 5–7 km for the paper config (initial ~5.5 km)."""
    cfg = SJ.SilvestriJetConfig()
    N = np.sqrt(cfg.N2)
    f0 = abs(2 * constants.Omega * np.sin(np.radians(cfg.lat_center)))
    Ld_km = N * cfg.H_max / (np.pi * f0) / 1000.0
    assert 4.5 < Ld_km < 7.5, Ld_km


def test_zonal_mean_restoring_relaxes_mean_not_eddy():
    """The restoring nudges the ZONAL-MEAN toward the target but leaves the eddy
    (zero-zonal-mean) part untouched."""
    n_lat, n_lon, nlev = 8, 16, 3
    rng = np.random.default_rng(1)
    ref_zm = jnp.asarray(rng.standard_normal((n_lat, nlev)))
    # Field = (target zonal mean) + a known offset on the mean + an eddy.
    offset = 0.5
    lon_wave = np.cos(2 * np.pi * np.arange(n_lon) / n_lon)[None, :, None]
    eddy = jnp.asarray(0.3 * lon_wave * np.ones((n_lat, 1, nlev)))
    field = ref_zm[:, None, :] + offset + eddy
    gamma, dt = 1.0 / (50 * 86400.0), 900.0
    out = SJ.apply_zonal_mean_restoring(field, ref_zm, gamma, dt)
    # Eddy part unchanged (correction is constant in lon).
    out_eddy = out - jnp.mean(out, axis=1, keepdims=True)
    assert np.allclose(np.asarray(out_eddy), np.asarray(eddy), atol=1e-12)
    # Zonal mean moved toward the target by exactly γ·dt·offset.
    zm_before = float(jnp.mean(field, axis=1)[0, 0]) - float(ref_zm[0, 0])  # = offset
    zm_after = float(jnp.mean(out, axis=1)[0, 0]) - float(ref_zm[0, 0])
    assert np.isclose(zm_after, offset * (1 - gamma * dt), rtol=1e-10)
    assert zm_after < zm_before                         # relaxed toward target


def test_stabilize_backstop_applied_to_noclosure_only():
    """stabilize=True applies the A_h=1000+C_smag=0.1 backstop to the no-closure
    WENO/flux schemes, but NOT to SM2/QG2 (which keep their own closure — adding
    A_h would trip the double-friction guard)."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    # No-closure scheme → backstop applied.
    rw = SJ.build_silvestri_baroclinic_jet_setup(
        n_lat=16, n_lon=12, scheme="W9V", nlev=8, stabilize=True)
    assert rw.model_config.A_h == 1000.0 and rw.model_config.C_smag == 0.1
    assert rw.model_config.smag_cfl_safety == 0.5
    LatLonCGridOceanModel(rw.grid, rw.z_coord, rw.model_config)   # builds (no guard trip)
    # Explicit-closure scheme → backstop NOT applied (guard would otherwise trip).
    rq = SJ.build_silvestri_baroclinic_jet_setup(
        n_lat=16, n_lon=12, scheme="QG2", nlev=8, stabilize=True)
    assert rq.model_config.A_h == 0.0 and rq.model_config.C_smag == 0.0
    assert rq.model_config.lateral_friction_scheme == "qg_leith"
    LatLonCGridOceanModel(rq.grid, rq.z_coord, rq.model_config)   # builds (no double friction)
    # Default (stabilize=False) leaves the no-closure scheme un-damped (faithful).
    r0 = SJ.build_silvestri_baroclinic_jet_setup(n_lat=16, n_lon=12, scheme="W9V", nlev=8)
    assert r0.model_config.A_h == 0.0 and r0.model_config.C_smag == 0.0


@pytest.mark.parametrize("scheme", SILVESTRI_JET_MAIN)
def test_builds_and_steps_each_scheme(scheme):
    """The recipe builds for each scheme, validates, and a full step + restoring
    stays finite."""
    from legoesm.core.field import Field
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    r = SJ.build_silvestri_baroclinic_jet_setup(
        n_lat=20, n_lon=16, scheme=scheme, nlev=8)
    assert isinstance(r, SJ.SilvestriJetRecipe)
    model = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config)
    s = r.initial_state

    def _z(d):
        return Field(data=jnp.zeros_like(d.data), name=d.name + "_incr_prev",
                     dims=d.dims, units=d.units)
    s = s._replace(T_incr_prev=_z(s.T), S_incr_prev=_z(s.S),
                   u_incr_prev=_z(s.u), v_incr_prev=_z(s.v))
    nxt = model.step(s, 600.0)
    nxt = SJ.restore_state(nxt, r.restoring, 600.0)
    assert bool(jnp.all(jnp.isfinite(nxt.u.data))), scheme
    assert bool(jnp.all(jnp.isfinite(nxt.T.data))), scheme
