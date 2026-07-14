"""Tests for the DEPHY-SCM atmospheric case loader."""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from legoesm import constants
from legoesm.atmosphere.forcing.scm.dephy_scm import load_dephy_scm_case
from legoesm.atmosphere.physics import (
    ConvectionConfig,
    GravityWaveDragConfig,
    MicrophysicsConfig,
    PhysicsConfig,
    RadiationConfig,
    TurbulenceConfig,
)
from legoesm.thermo import (
    mixing_ratio_to_specific_humidity,
    specific_humidity_tendency_to_mixing_ratio_tendency,
    specific_humidity_to_mixing_ratio,
)


_SECONDS_PER_DAY = 24.0 * 3600.0


def _no_physics_cfg() -> PhysicsConfig:
    return PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )


def _write_case(
    path: Path,
    *,
    case_id: str,
    lat: float,
    lon: float,
    orog: float,
    surface_type: str,
    radiation: str,
    time: np.ndarray,
    theta_tendency: np.ndarray | None,
    qv_tendency: np.ndarray,
    wa: np.ndarray | None,
    hfss: np.ndarray,
    hfls: np.ndarray,
    tskin: np.ndarray,
    include_unsupported_wind_adv: bool = False,
) -> Path:
    lev = np.asarray([0.0, 500.0, 1500.0, 3000.0, 6000.0, 12000.0, 20000.0])
    pa = np.asarray([101500.0, 95000.0, 85000.0, 70000.0, 50000.0, 20000.0, 5200.0])
    ta = np.asarray([300.0, 296.0, 286.0, 272.0, 248.0, 220.0, 205.0])
    rv = np.asarray([0.0170, 0.0160, 0.0110, 0.0050, 0.0020, 0.0005, 0.0])
    qv = np.asarray(mixing_ratio_to_specific_humidity(jnp.asarray(rv)))
    ua = np.asarray([-8.75, -8.75, -7.5, -6.0, -4.6, -4.6, -4.6])
    va = np.zeros_like(ua)
    ug = np.tile(-10.0 + 0.0018 * lev, (time.size, 1))
    vg = np.zeros_like(ug)
    pa_forc = np.tile(pa, (time.size, 1))
    zh_forc = np.tile(lev, (time.size, 1))

    data_vars = {
        "ps": (("t0",), np.asarray([pa[0]])),
        "zh": (("t0", "lev"), lev.reshape(1, -1)),
        "pa": (("t0", "lev"), pa.reshape(1, -1)),
        "ta": (("t0", "lev"), ta.reshape(1, -1)),
        "rv": (("t0", "lev"), rv.reshape(1, -1)),
        "qv": (("t0", "lev"), qv.reshape(1, -1)),
        "ua": (("t0", "lev"), ua.reshape(1, -1)),
        "va": (("t0", "lev"), va.reshape(1, -1)),
        "zh_forc": (("time", "lev"), zh_forc),
        "pa_forc": (("time", "lev"), pa_forc),
        "ug": (("time", "lev"), ug),
        "vg": (("time", "lev"), vg),
        "tnrv_adv": (("time", "lev"), qv_tendency),
        "hfss": (("time",), hfss),
        "hfls": (("time",), hfls),
        "tskin": (("time",), tskin),
        "lat": (("time",), np.full(time.size, lat)),
        "lon": (("time",), np.full(time.size, lon)),
        "orog": (("time",), np.full(time.size, orog)),
    }
    if theta_tendency is not None:
        data_vars["tntheta_adv"] = (("time", "lev"), theta_tendency)
    else:
        rad = np.zeros((time.size, lev.size))
        rad[:, lev <= 1500.0] = -2.0 / _SECONDS_PER_DAY
        data_vars["tntheta_rad"] = (("time", "lev"), rad)
    if wa is not None:
        data_vars["wa"] = (("time", "lev"), wa)
    if include_unsupported_wind_adv:
        data_vars["tnua_adv"] = (("time", "lev"), np.full_like(qv_tendency, 1e-5))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "t0": np.asarray([0.0]),
            "time": time,
            "lev": lev,
        },
        attrs={
            "case": case_id,
            "title": f"{case_id} SCM fixture",
            "start_date": "2000-01-01 00:00:00",
            "end_date": "2000-01-01 01:00:00",
            "surface_type": surface_type,
            "surface_forcing_temp": "surface_flux",
            "surface_forcing_moisture": "surface_flux",
            "surface_forcing_wind": "ustar" if surface_type == "ocean" else "z0",
            "radiation": radiation,
            "forc_geo": 1,
            "forc_wa": int(wa is not None),
            "adv_rv": 1,
        },
    )
    ds.to_netcdf(path)
    return path


def _write_bomex_fixture(path: Path, *, unsupported: bool = False) -> Path:
    time = np.asarray([0.0, 1800.0, 3600.0])
    lev_size = 7
    qv_adv = np.zeros((time.size, lev_size))
    qv_adv[:, :3] = -1.2e-8
    wa = np.zeros((time.size, lev_size))
    wa[:, 1:4] = -0.0065
    return _write_case(
        path,
        case_id="BOMEX/REF",
        lat=15.0,
        lon=-56.5,
        orog=0.0,
        surface_type="ocean",
        radiation="tend",
        time=time,
        theta_tendency=None,
        qv_tendency=qv_adv,
        wa=wa,
        hfss=np.full(time.size, 8.0e-3 * constants.c_pd),
        hfls=np.full(time.size, 5.2e-5 * constants.L_v),
        tskin=np.full(time.size, 300.4),
        include_unsupported_wind_adv=unsupported,
    )


def _write_armcu_fixture(path: Path) -> Path:
    time = np.asarray([0.0, 1200.0, 2400.0, 3600.0])
    lev_size = 7
    theta_adv = np.zeros((time.size, lev_size))
    qv_adv = np.zeros_like(theta_adv)
    for it, scale in enumerate([0.0, 0.25, 0.75, 1.0]):
        theta_adv[it, :4] = -7.2e-5 * scale
        qv_adv[it, :4] = (2.2e-8 - 1.0e-7 * scale)
    return _write_case(
        path,
        case_id="ARMCU/REF",
        lat=36.0,
        lon=-97.5,
        orog=314.0,
        surface_type="land",
        radiation="off",
        time=time,
        theta_tendency=theta_adv,
        qv_tendency=qv_adv,
        wa=None,
        hfss=np.asarray([-30.0, 90.0, 140.0, 100.0]),
        hfls=np.asarray([5.0, 250.0, 500.0, 180.0]),
        tskin=np.asarray([295.4, 303.0, 313.0, 300.0]),
    )


def test_humidity_conversion_helpers_roundtrip_and_tendency():
    q = jnp.asarray([0.0, 0.01, 0.02])
    r = specific_humidity_to_mixing_ratio(q)
    np.testing.assert_allclose(mixing_ratio_to_specific_humidity(r), q)

    dq_dt = jnp.asarray([1.0e-8, -2.0e-8, 3.0e-8])
    dr_dt = specific_humidity_tendency_to_mixing_ratio_tendency(q, dq_dt)
    expected = dq_dt / (1.0 - q) ** 2
    np.testing.assert_allclose(dr_dt, expected)


def test_load_bomex_dephy_scm_case_maps_forcing_and_runs(tmp_path):
    path = _write_bomex_fixture(tmp_path / "BOMEX_REF_SCM_driver.nc")
    case = load_dephy_scm_case(path, nlev=18, dt=300.0, dtype=jnp.float64)

    assert case.case_id == "BOMEX/REF"
    assert case.surface_type == "ocean"
    assert case.forcing.prescribe == "fluxes"
    assert case.forcing.subsidence_w is not None
    assert case.forcing.theta_adv is not None
    assert case.forcing.qv_adv is not None
    assert case.forcing.f_c == pytest.approx(
        2.0 * constants.Omega * np.sin(np.deg2rad(15.0))
    )
    assert case.T_profile.shape == (18,)
    assert float(case.T_profile[0]) < float(case.T_profile[-1])
    assert not np.allclose(np.asarray(case.u_profile), float(case.u_profile[0]))
    assert "prescribed radiative tendency" in " ".join(case.notes)

    w = np.asarray(case.forcing.subsidence_w(0.0))
    theta_adv = np.asarray(case.forcing.theta_adv(0.0))
    assert w.min() < 0.0
    assert theta_adv.min() < 0.0
    assert float(case.forcing.w_th_s(0.0)) > 0.0
    assert float(case.forcing.w_qv_s(0.0)) > 0.0

    scm = case.create_scm(physics_config=_no_physics_cfg())
    final, history = scm.run(nsteps=12, save_every=6)
    T = np.asarray(final.T.data)
    qv = np.asarray(final.tracers["q_v"].data)
    assert np.all(np.isfinite(T))
    assert np.all(np.isfinite(qv))
    assert 150.0 < float(T.min()) < float(T.max()) < 330.0
    assert float(qv.min()) > -1.0e-5
    assert history.T.shape[0] >= 2


def test_load_armcu_dephy_scm_case_maps_time_varying_fluxes_and_runs(tmp_path):
    path = _write_armcu_fixture(tmp_path / "ARMCU_REF_SCM_driver.nc")
    case = load_dephy_scm_case(path, nlev=20, dt=300.0, dtype=jnp.float64)

    assert case.case_id == "ARMCU/REF"
    assert case.surface_type == "land"
    assert case.forcing.subsidence_w is None
    assert case.phis == pytest.approx(constants.g * 314.0)
    assert case.forcing.theta_adv is not None
    assert case.forcing.qv_adv is not None
    assert float(case.forcing.w_th_s(0.0)) < 0.0
    assert float(case.forcing.w_th_s(2400.0)) > 0.0
    assert "surface wind forcing metadata" in " ".join(case.notes)

    scm = case.create_scm(physics_config=_no_physics_cfg())
    final, history = scm.run(nsteps=10, save_every=5)
    assert np.all(np.isfinite(np.asarray(final.T.data)))
    assert np.all(np.isfinite(np.asarray(final.tracers["q_v"].data)))
    assert 180.0 < float(np.min(history.T)) < float(np.max(history.T)) < 325.0


def test_load_dephy_scm_strict_rejects_unsupported_nonzero_forcing(tmp_path):
    path = _write_bomex_fixture(
        tmp_path / "BOMEX_REF_SCM_driver.nc",
        unsupported=True,
    )
    case = load_dephy_scm_case(path, nlev=12, dt=300.0, strict=False)
    assert any("tnua_adv" in item for item in case.unsupported)

    with pytest.raises(ValueError, match="unsupported forcing"):
        load_dephy_scm_case(path, nlev=12, dt=300.0, strict=True)


@pytest.mark.parametrize(
    "relative_path",
    [
        "BOMEX/REF/BOMEX_REF_SCM_driver.nc",
        "ARMCU/REF/ARMCU_REF_SCM_driver.nc",
    ],
)
def test_external_standard_dephy_case_runs_when_available(relative_path):
    root = os.environ.get("DEPHY_SCM_ROOT")
    if not root:
        pytest.skip("Set DEPHY_SCM_ROOT to run against the external DEPHY checkout.")
    path = Path(root) / relative_path
    if not path.exists():
        pytest.skip(f"DEPHY case file not found: {path}")

    case = load_dephy_scm_case(path, nlev=24, dt=600.0, dtype=jnp.float64)
    scm = case.create_scm(physics_config=_no_physics_cfg())
    final, _ = scm.run(nsteps=12, save_every=6)
    assert np.all(np.isfinite(np.asarray(final.T.data)))
    assert np.all(np.isfinite(np.asarray(final.tracers["q_v"].data)))
    assert 120.0 < float(np.min(final.T.data)) < float(np.max(final.T.data)) < 340.0
