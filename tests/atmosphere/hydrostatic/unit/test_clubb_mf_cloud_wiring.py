"""CLUBB-MF plume cloud wiring: the turbulence lane publishes the plume cloud
into the carry, both radiation lanes hand the lagged carry to the cloud call
(only when CloudConfig.clubb_mf_cloud is on), and the factory refuses a
producer/consumer mismatch."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
from legoesm.atmosphere.physics.physics_state import init_physics_state
from legoesm.atmosphere.physics.radiation import integration as radint
from legoesm.atmosphere.physics.radiation.config import RadiationConfig, RRTMGPConfig
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

jax.config.update("jax_enable_x64", True)

NLEV = 6


class _Captured(Exception):
    pass


def _rad(mf_cloud, rad="rrtmgp"):
    return RadiationConfig(scheme=rad, cloud_scheme="cam6_clubb",
                           cloud_config=CloudConfig(scheme="cam6_clubb", clubb_mf_cloud=mf_cloud),
                           use_clubb_cloud_fraction=True,
                           rrtmgp=RRTMGPConfig(include_clouds=True), diurnal_cycle=False)


def _carry(ncol):
    return init_physics_state(ncol, NLEV, PhysicsConfig())._replace(
        cloud_fraction=jnp.full((ncol, NLEV), 0.37),
        conv_mass_flux_up=jnp.full((ncol, NLEV + 1), 0.02),
        conv_icwmr=jnp.full((ncol, NLEV), 2.0e-4),
        mf_cloudfrac=jnp.full((ncol, NLEV), 0.03),
        mf_qc=jnp.full((ncol, NLEV), 1.0e-3))


def _spy(monkeypatch):
    seen = {}

    def spy(*a, **kw):
        seen.update(kw)
        raise _Captured()
    monkeypatch.setattr(radint, "compute_cloud_properties", spy)
    monkeypatch.setattr(radint.jax, "checkpoint", lambda f, **kw: f)
    return seen


def _lane_inputs(lane):
    if lane == "mpas":
        from tests.unit.test_physics_smoke import make_hydrostatic_setup
        state, grid, sigma = make_hydrostatic_setup(n=2, nlev=NLEV)
        return state, grid, sigma, 6 * 2 * 2
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import isothermal_rest_state_spectral
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    grid = create_gaussian_grid(8, dealiasing="quadratic")
    sigma = create_sigma_coordinate(NLEV)
    shp = (grid.n_lat, grid.n_lon, NLEV)
    tracers = {"q_v": jnp.full(shp, 5.0e-3), "q_c": jnp.full(shp, 1.0e-4),
               "q_i": jnp.full(shp, 2.0e-5)}
    state = isothermal_rest_state_spectral(grid, sigma, T_init=280.0, p_s_init=1.0e5,
                                           tracers=tracers)
    return state, grid, sigma, int(grid.n_lat) * int(grid.n_lon)


@pytest.mark.parametrize("lane, model_type", [("mpas", "mpas"), ("spectral", "spectral_pe")])
def test_radiation_hands_the_lagged_plume_cloud_to_the_cloud_call(lane, model_type, monkeypatch):
    state, grid, sigma, ncol = _lane_inputs(lane)
    seen = _spy(monkeypatch)
    for on in (True, False):
        seen.clear()
        fn = radint.make_radiation_physics(_rad(on), model_type=model_type,
                                           use_clubb_cloud_fraction=True)
        with pytest.raises(_Captured):
            fn(state, grid, sigma, phys_state=_carry(ncol))
        if on:
            assert np.allclose(np.asarray(seen["mf_cloudfrac"]), 0.03)
            assert np.allclose(np.asarray(seen["mf_qc"]), 1.0e-3)
        else:
            assert seen["mf_cloudfrac"] is None and seen["mf_qc"] is None


def test_factory_refuses_a_producer_consumer_mismatch():
    turb = lambda mf: TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(prognostic=True),
                                       clubb_mf=mf)
    for t, r in ((turb(True), _rad(False, "gray")), (turb(False), _rad(True, "gray"))):
        with pytest.raises(ValueError, match="clubb_mf"):
            make_physics(PhysicsConfig(radiation=r, turbulence=t), model_type="mpas")
    with pytest.raises(ValueError, match="clubb_mf"):  # un-nested cam6_clubb fallback
        make_physics(PhysicsConfig(radiation=_rad(False, "gray")._replace(cloud_config=None),
                                   turbulence=turb(True)), model_type="mpas")
    assert callable(make_physics(PhysicsConfig(radiation=_rad(True, "gray"),
                                               turbulence=turb(True)), model_type="mpas"))


def test_scm_publishes_the_plume_cloud_only_with_the_mass_flux():
    from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
    from tests.atmosphere.hydrostatic.unit.test_scm import (
        _RCE_DT, _RCE_NLEV, _baseline_cfg, _tropical_qv_profile, _tropical_T_profile)

    got = {}
    for on in (False, True):
        cfg = _baseline_cfg(turbulence=TurbulenceConfig(
            scheme="clubb", clubb=CLUBBConfig(prognostic=True), clubb_mf=on))
        scm = SingleColumnModel.create(
            physics_config=cfg, nlev=_RCE_NLEV, dt=_RCE_DT, T_profile=_tropical_T_profile(),
            q_v_profile=_tropical_qv_profile(), latitude_deg=0.0)
        scm.run(nsteps=4)
        ps = scm.phys_state
        got[on] = (np.asarray(ps.mf_cloudfrac), np.asarray(ps.mf_qc))
    assert np.all(got[False][0] == 0.0) and np.all(got[False][1] == 0.0)
    assert np.max(got[True][0]) > 0.0 and np.max(got[True][1]) > 0.0


def test_cam6_cloud_call_adds_the_capped_plume_cloud():
    """The cam6_clubb cloud call combines the plume cloud with ZM's (larger
    of two) and adds the 0.8-capped share to the stratus; refuses a missing
    plume carry when the hookup is on."""
    from legoesm.atmosphere.physics.clouds.cloud_fraction import compute_cloud_properties
    from legoesm.atmosphere.physics.turbulence.clubb_mf import mf_concld, mf_deep_cloud
    from tests.unit.test_cam6_cloud_fraction import CLUBB, _column_inputs

    T, p, q_v, dp, q_i, lat, p_half = _column_inputs()
    alst = jnp.full(T.shape, 0.05)
    mfcf = jnp.full(T.shape, 0.03).at[:, :2].set(0.2)
    mfqc = jnp.full(T.shape, 1.0e-3)
    kw = dict(q_cloud=jnp.full(T.shape, 1.0e-5), q_ice=q_i, cloud_fraction_override=alst,
              lat=lat, p_half=p_half, clubb_config=CLUBB)
    on = CloudConfig(scheme="cam6_clubb", clubb_mf_cloud=True)
    off = compute_cloud_properties(T, p, q_v, dp, CloudConfig(scheme="cam6_clubb"), **kw)
    got = compute_cloud_properties(T, p, q_v, dp, on, mf_cloudfrac=mfcf, mf_qc=mfqc, **kw)
    deep, _ = mf_deep_cloud(jnp.zeros_like(T), jnp.zeros_like(T), mfcf, mfqc, on)
    want = jnp.minimum(off.cloud_fraction + mf_concld(deep, on), 1.0)
    np.testing.assert_allclose(np.asarray(got.cloud_fraction), np.asarray(want), rtol=1e-12)
    assert float(jnp.max(got.cloud_fraction - off.cloud_fraction)) > 0.25
    with pytest.raises(ValueError, match="plume cloud"):
        compute_cloud_properties(T, p, q_v, dp, on, **kw)
