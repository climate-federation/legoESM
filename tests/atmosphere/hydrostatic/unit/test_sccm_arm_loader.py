"""Tests for the NCAR-SCCM (ARM SGP) obs-forcing loader."""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from legoesm.atmosphere.forcing.scm.sccm_arm import load_arm_sccm_case
from legoesm.atmosphere.physics import (
    ConvectionConfig,
    GravityWaveDragConfig,
    MicrophysicsConfig,
    PhysicsConfig,
    RadiationConfig,
    TurbulenceConfig,
)

from legoesm import constants

_ARM9707 = Path("data/les_cases/ARM9707/arm9707.nc")


def _no_physics_cfg() -> PhysicsConfig:
    return PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="none"),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )


def _write_sccm_fixture(path: Path) -> Path:
    """A tiny SCCM-format file with the singleton lat/lon axes SCCM carries."""
    lev = np.array([20000.0, 50000.0, 70000.0, 85000.0, 92500.0, 100000.0])  # Pa
    time = np.array([0.0, 3600.0, 7200.0, 10800.0, 14400.0])  # 1-h cadence
    nt, nz = time.size, lev.size
    T = np.array([220.0, 250.0, 272.0, 285.0, 295.0, 300.0])   # warm at surface
    q = np.array([0.0005, 0.002, 0.005, 0.010, 0.014, 0.016])
    u = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0])
    v = np.zeros(nz)

    def prof(base):
        return np.tile(base, (nt, 1)).reshape(nt, nz, 1, 1)

    def sfc(series):
        return np.asarray(series, dtype=float).reshape(nt, 1, 1)

    ds = xr.Dataset(
        data_vars={
            "bdate": (("ncl_scalar",), np.array([970618.0])),
            "tsec": (("time",), time),  # seconds after 0Z (SCCM stores time here)
            "Ps": (("time", "lat", "lon"), sfc(np.full(nt, 100000.0))),
            "T": (("time", "lev", "lat", "lon"), prof(T)),
            "q": (("time", "lev", "lat", "lon"), prof(q)),
            "u": (("time", "lev", "lat", "lon"), prof(u)),
            "v": (("time", "lev", "lat", "lon"), prof(v)),
            # omega > 0 everywhere (subsidence, positive DOWN)
            "omega": (("time", "lev", "lat", "lon"), prof(np.full(nz, 0.02))),
            # cooling by horizontal advection; moistening
            "divT": (("time", "lev", "lat", "lon"), prof(np.full(nz, -1.0e-5))),
            "divq": (("time", "lev", "lat", "lon"), prof(np.full(nz, 1.0e-8))),
            "shflx": (("time", "lat", "lon"), sfc([100.0, -20.0, 50.0, 80.0, 120.0])),
            "lhflx": (("time", "lat", "lon"), sfc([200.0, 150.0, 180.0, 220.0, 250.0])),
            "Prec": (("time", "lat", "lon"), sfc([0.0, 1.0e-4, 0.0, 2.0e-4, 0.0])),
            "totcld": (("time", "lat", "lon"), sfc([0.0, 50.0, 80.0, 30.0, 0.0])),
            "cldliq": (("time", "lat", "lon"), sfc([0.0, 1.0e-4, 2.0e-4, 5.0e-5, 0.0])),
        },
        coords={
            "lev": lev,
            "time": time,
            "lat": np.array([36.6]),
            "lon": np.array([-97.5]),
        },
    )
    ds.to_netcdf(path)
    return path


def test_load_sccm_fixture_conversions_signs_and_obs(tmp_path):
    path = _write_sccm_fixture(tmp_path / "arm_fixture.nc")
    case = load_arm_sccm_case(path, nlev=24, dtype=jnp.float64)

    # --- geometry / initial state ---
    assert case.nlev == 24
    assert case.base_date == 970618
    assert case.dt <= 300.0  # capped below the 1-h forcing cadence
    assert case.forcing.f_c == pytest.approx(
        2.0 * constants.Omega * np.sin(np.deg2rad(36.6))
    )
    # top→bottom ordering: cold top, warm surface.
    assert float(case.T_profile[0]) < float(case.T_profile[-1])
    assert float(case.q_v_profile[-1]) > float(case.q_v_profile[0])

    # --- forcing signs (CLAUDE.md sign convention) ---
    sub = np.asarray(case.forcing.subsidence_w(0.0))
    assert np.all(sub < 0.0)  # omega>0 (down) => w<0 (SCM positive up)
    theta_adv = np.asarray(case.forcing.theta_adv(0.0))
    assert np.all(theta_adv < 0.0)  # divT<0 cooling
    # theta_adv = divT / exner; the surface full level sits just below p_s so
    # exner is slightly <1 and |theta_adv| slightly exceeds |divT| (=1e-5).
    assert theta_adv[-1] == pytest.approx(-1.0e-5, rel=2e-2)
    assert abs(theta_adv[-1]) >= 1.0e-5
    assert np.all(np.asarray(case.forcing.qv_adv(0.0)) > 0.0)

    assert case.forcing.prescribe == "fluxes"
    assert float(case.forcing.w_th_s(0.0)) > 0.0        # shflx[0]=+100
    assert float(case.forcing.w_th_s(3600.0)) < 0.0     # shflx[1]=-20
    assert float(case.forcing.w_qv_s(0.0)) > 0.0

    # --- observation reference ---
    obs = case.obs
    assert obs.total_cloud_fraction == pytest.approx([0.0, 0.5, 0.8, 0.3, 0.0])
    assert obs.liquid_water_path_kg_m2 == pytest.approx(
        np.array([0.0, 1.0e-4, 2.0e-4, 5.0e-5, 0.0]) * constants.rho_water
    )
    assert np.nanmax(obs.precip_mm_day) == pytest.approx(2.0e-4 * 1000.0 * 86400.0)
    assert obs.T.shape == (5, 6)
    assert obs.latitude_deg == pytest.approx(36.6)


def test_sccm_fixture_scm_round_trip_is_finite(tmp_path):
    path = _write_sccm_fixture(tmp_path / "arm_fixture.nc")
    case = load_arm_sccm_case(path, nlev=20, dt=120.0, dtype=jnp.float64)
    scm = case.create_scm(physics_config=_no_physics_cfg())
    final, history = scm.run(nsteps=10, save_every=5)
    T = np.asarray(final.T.data)
    qv = np.asarray(final.tracers["q_v"].data)
    assert np.all(np.isfinite(T))
    assert np.all(np.isfinite(qv))
    assert 150.0 < float(T.min()) < float(T.max()) < 340.0
    assert float(qv.min()) > -1.0e-5
    assert history.T.shape[0] >= 2


def test_disable_channels():
    """Loader honours the include_* switches (uses the synthetic fixture)."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = _write_sccm_fixture(Path(d) / "arm_fixture.nc")
        case = load_arm_sccm_case(
            path, nlev=16, include_vertical_velocity=False,
            include_advective_tendencies=False, include_surface_fluxes=False,
        )
        assert case.forcing.subsidence_w is None
        assert case.forcing.theta_adv is None
        assert case.forcing.qv_adv is None
        assert case.forcing.prescribe == "none"
        # obs reference is still fully populated regardless of forcing switches.
        assert case.obs.total_cloud_fraction is not None


@pytest.mark.skipif(not _ARM9707.exists(), reason="ARM9707 obs file not present")
def test_load_real_arm9707_forcing_and_obs():
    case = load_arm_sccm_case(_ARM9707, nlev=48, dtype=jnp.float64)
    # ARM SGP Central Facility (Lamont, OK), summer-1997 IOP.
    assert case.base_date == 970618
    assert case.latitude_deg == pytest.approx(36.6, abs=0.2)
    assert case.longitude_deg == pytest.approx(-97.5, abs=0.3)
    assert case.duration_seconds > 20 * 86400.0  # ~29-day IOP
    assert case.forcing.prescribe == "fluxes"
    assert case.forcing.subsidence_w is not None
    assert case.forcing.theta_adv is not None

    # physical surface state
    assert 290.0 < float(case.T_profile[-1]) < 310.0
    assert 0.005 < float(case.q_v_profile[-1]) < 0.025
    # ARM SGP Central Facility sits at ~318 m: phis = g*z is read, not dropped.
    assert 2500.0 < case.phis < 3600.0

    obs = case.obs
    assert obs.total_cloud_fraction is not None
    assert 0.0 <= np.nanmin(obs.total_cloud_fraction)
    assert np.nanmax(obs.total_cloud_fraction) <= 1.0001
    assert np.nanmax(obs.liquid_water_path_kg_m2) > 0.0
    # diurnal sensible heat flux swings negative (night) to strongly positive.
    assert np.nanmin(obs.sensible_heat_flux_wm2) < 0.0
    assert np.nanmax(obs.sensible_heat_flux_wm2) > 80.0
