"""Tests for the ARM obs-vs-SCM scoring assembly (les_suite.arm_obs_score)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from legoesm.atmosphere.forcing.scm.sccm_arm import ARMObsReference
from legoesm.atmosphere.les_suite.arm_obs_score import (
    ARMComparables,
    build_arm_comparables,
    score_arm_obs,
)
from legoesm.atmosphere.physics import (
    ConvectionConfig,
    GravityWaveDragConfig,
    MicrophysicsConfig,
    PhysicsConfig,
    RadiationConfig,
    TurbulenceConfig,
)
from legoesm.atmosphere.physics._shared import exner_function

_PRESS = np.array([100000.0, 85000.0, 70000.0, 50000.0])
_TIMES = np.array([0.0, 3600.0, 7200.0])
_EXNER = np.asarray(exner_function(_PRESS), dtype=np.float64)


def _obs(**overrides) -> ARMObsReference:
    """A minimal ARMObsReference with everything absent unless overridden."""
    base = dict(
        base_date=970618, latitude_deg=36.6, longitude_deg=-97.5,
        time_seconds=_TIMES, pressure_pa=_PRESS,
        T=None, q=None, u=None, v=None,
        surface_pressure_pa=None, surface_air_temperature_k=None,
        surface_rh_percent=None, surface_wind_speed=None,
        sensible_heat_flux_wm2=None, latent_heat_flux_wm2=None, precip_m_s=None,
        total_cloud_fraction=None, low_cloud_fraction=None, mid_cloud_fraction=None,
        high_cloud_fraction=None, liquid_water_path_kg_m2=None,
        cloud_top_height_m=None, cloud_thickness_m=None,
        surface_net_down_radiation_wm2=None, toa_lw_up_wm2=None, toa_sw_dn_wm2=None,
    )
    base.update(overrides)
    return ARMObsReference(**base)


def _theta_true() -> np.ndarray:
    z = np.array([300.0, 305.0, 315.0, 330.0])           # increasing theta with height
    return np.broadcast_to(z, (_TIMES.size, _PRESS.size)).copy()


def test_perfect_match_scores_zero():
    theta = _theta_true()
    q = np.broadcast_to(np.array([0.016, 0.010, 0.005, 0.001]), theta.shape).copy()
    obs = _obs(T=theta * _EXNER[None, :], q=q, u=np.full_like(theta, 5.0),
               v=np.zeros_like(theta), liquid_water_path_kg_m2=np.array([0.0, 0.1, 0.05]))
    comp = ARMComparables(
        time_seconds=_TIMES, pressure_pa=_PRESS, theta=theta, q=q,
        u=np.full_like(theta, 5.0), v=np.zeros_like(theta),
        lwp_kg_m2=np.array([0.0, 0.1, 0.05]))
    s = score_arm_obs(obs, comp)
    assert set(s.channels) == {"theta", "q", "u", "v", "lwp"}
    for r in (s.theta_rmse, s.q_rmse, s.u_rmse, s.v_rmse, s.lwp_rmse):
        assert r == pytest.approx(0.0, abs=1e-9)
    assert s.combined == pytest.approx(0.0, abs=1e-9)


def test_theta_offset_is_normalized_by_obs_spread():
    theta = _theta_true()
    obs = _obs(T=theta * _EXNER[None, :])
    delta = 2.0
    comp = ARMComparables(time_seconds=_TIMES, pressure_pa=_PRESS, theta=theta + delta)
    s = score_arm_obs(obs, comp)
    # normalized RMSE = |delta| / mass-weighted std of the obs theta profile.
    from legoesm.atmosphere.les_suite.arm_obs_score import (
        _FLOOR_THETA_K,
        _pressure_thickness_weights,
        _profile_scale,
    )
    w = _pressure_thickness_weights(_PRESS)
    scale = _profile_scale(theta, w, _FLOOR_THETA_K)
    assert s.theta_rmse == pytest.approx(delta / scale, rel=1e-6)
    assert s.channels == ("theta",)
    assert s.combined == pytest.approx(s.theta_rmse)


def test_missing_obs_are_masked_not_zero_filled():
    theta = _theta_true()
    obs_T = theta * _EXNER[None, :]
    obs_T[1, 2] = np.nan                       # one missing sounding level
    obs = _obs(T=obs_T, q=np.full_like(theta, np.nan))  # q entirely missing
    comp = ARMComparables(time_seconds=_TIMES, pressure_pa=_PRESS,
                          theta=theta, q=np.full_like(theta, 0.01))
    s = score_arm_obs(obs, comp)
    assert s.theta_rmse == pytest.approx(0.0, abs=1e-9)  # masked NaN, rest perfect
    assert s.q_rmse is None                              # all-NaN obs -> not scored
    assert "q" not in s.channels


def test_all_nan_model_channel_raises_not_dropped():
    # obs present but the MODEL is all-NaN for that channel -> must raise, not
    # silently drop (dropping would let producing NaNs improve the score).
    theta = _theta_true()
    obs = _obs(T=theta * _EXNER[None, :])
    comp = ARMComparables(time_seconds=_TIMES, pressure_pa=_PRESS,
                          theta=np.full_like(theta, np.nan))
    with pytest.raises(ValueError, match="no finite profile values"):
        score_arm_obs(obs, comp)

    obs2 = _obs(liquid_water_path_kg_m2=np.array([0.1, 0.2, 0.05]))
    comp2 = ARMComparables(time_seconds=_TIMES, pressure_pa=_PRESS,
                           lwp_kg_m2=np.full(3, np.nan))
    with pytest.raises(ValueError, match="no finite series values"):
        score_arm_obs(obs2, comp2)


def test_pressure_grid_mismatch_raises():
    theta = _theta_true()
    obs = _obs(T=theta * _EXNER[None, :])
    comp = ARMComparables(time_seconds=_TIMES, pressure_pa=_PRESS + 500.0, theta=theta)
    with pytest.raises(ValueError, match="must match obs.pressure_pa"):
        score_arm_obs(obs, comp)


def test_time_misalignment_raises():
    theta = _theta_true()
    obs = _obs(T=theta * _EXNER[None, :])
    comp = ARMComparables(time_seconds=_TIMES + 500.0, pressure_pa=_PRESS, theta=theta)
    with pytest.raises(ValueError, match="align with observation times"):
        score_arm_obs(obs, comp)


def test_cloud_and_precip_scored_when_supplied():
    obs = _obs(total_cloud_fraction=np.array([0.1, 0.8, 0.4]),
               precip_m_s=np.array([0.0, 1.0e-4, 0.0]))
    comp = ARMComparables(
        time_seconds=_TIMES, pressure_pa=_PRESS,
        total_cloud_fraction=np.array([0.1, 0.8, 0.4]),
        precip_mm_day=obs.precip_mm_day.copy())
    s = score_arm_obs(obs, comp)
    assert s.cloud_rmse == pytest.approx(0.0, abs=1e-9)
    assert s.precip_rmse == pytest.approx(0.0, abs=1e-9)
    assert set(s.channels) == {"cloud", "precip"}


# --- end-to-end: run the ARM-forced SCM and score it vs the (synthetic) obs ---
def _moist_cfg() -> PhysicsConfig:
    # The ARM forcing prescribes surface fluxes, so the turbulence surface bulk
    # exchange must be OFF (Ch_neutral=0) or the SCM rejects the config as a
    # double-count. Keep Cd_neutral for momentum drag.
    turb = TurbulenceConfig(scheme="holtslag_boville")
    hb = turb.holtslag_boville
    turb = turb._replace(
        holtslag_boville=hb._replace(surface=hb.surface._replace(Ch_neutral=0.0)))
    return PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=turb,
        microphysics=MicrophysicsConfig(scheme="sundqvist"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )


def test_build_and_score_end_to_end(tmp_path):
    # Reuse the SCCM fixture writer from the loader test.
    import importlib.util

    loader_test = (
        Path(__file__).resolve().parents[2]
        / "atmosphere/hydrostatic/unit/test_sccm_arm_loader.py"
    )
    spec = importlib.util.spec_from_file_location("_sccm_loader_test", loader_test)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from legoesm.atmosphere.forcing.scm.sccm_arm import load_arm_sccm_case

    path = mod._write_sccm_fixture(tmp_path / "arm_fixture.nc")
    case = load_arm_sccm_case(path, nlev=20, dt=200.0)

    comp = build_arm_comparables(case, _moist_cfg(), save_every=2)
    # comparables land on the obs times + obs pressure levels.
    assert np.allclose(comp.time_seconds, case.obs.time_seconds)
    assert comp.theta.shape == (case.obs.time_seconds.size, case.obs.pressure_pa.size)
    assert np.all(np.isfinite(comp.theta))
    assert np.all(np.isfinite(comp.u)) and np.all(np.isfinite(comp.v))
    assert comp.lwp_kg_m2 is not None and np.all(np.isfinite(comp.lwp_kg_m2))
    assert np.all(comp.lwp_kg_m2 >= -1e-12)

    s = score_arm_obs(case.obs, comp)
    assert "theta" in s.channels and "q" in s.channels
    assert "lwp" in s.channels
    assert s.combined is not None and np.isfinite(s.combined) and s.combined >= 0.0


def test_dt_and_t0_overrides_are_rejected(tmp_path):
    import importlib.util

    loader_test = (
        Path(__file__).resolve().parents[2]
        / "atmosphere/hydrostatic/unit/test_sccm_arm_loader.py"
    )
    spec = importlib.util.spec_from_file_location("_sccm_loader_test2", loader_test)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from legoesm.atmosphere.forcing.scm.sccm_arm import load_arm_sccm_case

    case = load_arm_sccm_case(mod._write_sccm_fixture(tmp_path / "f.nc"), nlev=16)
    with pytest.raises(ValueError, match="cannot be passed via create_overrides"):
        build_arm_comparables(case, _moist_cfg(), dt=200.0, t0_seconds=0.0)
