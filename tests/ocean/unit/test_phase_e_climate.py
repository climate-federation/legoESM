"""Phase E unit tests: forcing loaders + L&Y bulk flux + climate diagnostics."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Forcing loaders (synthetic fallback path)
# ---------------------------------------------------------------------------

def test_synthetic_ocean_forcing_shapes_and_units():
    from legoesm.ocean.forcing import synthetic_ocean_forcing
    f = synthetic_ocean_forcing(2000, n_time=4, nlon=72, nlat=36)
    assert f.lon.shape == (72,)
    assert f.lat.shape == (36,)
    assert f.time_s.shape == (4,)
    assert f.u10.shape == (4, 36, 72)
    # Physical bounds.
    assert 260.0 < f.T_air.min() and f.T_air.max() < 320.0
    assert (f.q_air >= 0.0).all()
    assert (f.sw_down >= 0.0).all()
    assert (f.precip >= 0.0).all()


def test_load_jra55_do_synthetic_fallback(tmp_path):
    from legoesm.ocean.forcing import load_jra55_do
    with pytest.raises(FileNotFoundError):           # fail-loud by default
        load_jra55_do(2000, cache_dir=tmp_path)
    f = load_jra55_do(2000, cache_dir=tmp_path, allow_synthetic=True)
    # Synthetic returns the canonical-shape ``OceanForcing``.
    assert f.u10.ndim == 3
    # 7 channels populated, all finite.
    for arr in (f.u10, f.v10, f.T_air, f.q_air,
                f.sw_down, f.lw_down, f.precip, f.runoff):
        assert np.isfinite(arr).all()


def test_load_core2_nyf_synthetic_fallback(tmp_path):
    # Point at an EMPTY cache dir: on machines where the real nyf.zarr cache
    # exists the loader (correctly) returns the 1460-record CORE-II data and
    # the synthetic branch would never be exercised.
    from legoesm.ocean.forcing import load_core2_nyf
    f = load_core2_nyf(cache_dir=tmp_path, n_time=12)
    assert f.time_s.shape == (12,)
    assert np.isfinite(f.T_air).all()


def test_load_jra55_do_disallow_synthetic_raises_when_missing(tmp_path):
    from legoesm.ocean.forcing import load_jra55_do
    with pytest.raises(FileNotFoundError):
        load_jra55_do(1999, cache_dir=tmp_path, allow_synthetic=False)


# ---------------------------------------------------------------------------
# Large + Yeager 2009 bulk fluxes
# ---------------------------------------------------------------------------

def test_large_yeager_cd_increases_with_wind_at_high_winds():
    from legoesm.ocean.bulk_flux_omip import large_yeager_cd
    Cd_5 = float(large_yeager_cd(5.0))
    Cd_15 = float(large_yeager_cd(15.0))
    # High-wind branch of L&Y 2009: Cd grows linearly with u10.
    assert Cd_15 > Cd_5
    # Magnitude check: Cd at 5 m/s ~ 1.4e-3 (within +/- 30 %).
    assert 1.0e-3 < Cd_5 < 2.0e-3


def test_large_yeager_ch_picks_stable_vs_unstable():
    from legoesm.ocean.bulk_flux_omip import large_yeager_ch
    Ch_unst = float(large_yeager_ch(T_air_K=290.0, T_sfc_K=295.0))   # sea warmer
    Ch_stab = float(large_yeager_ch(T_air_K=295.0, T_sfc_K=290.0))   # sea cooler
    assert Ch_unst > Ch_stab


def test_air_sea_fluxes_sign_conventions():
    import jax.numpy as jnp
    from legoesm.ocean.bulk_flux_omip import air_sea_fluxes
    # Wind blowing east at 10 m/s -> tau_x < 0 (drag opposes wind).
    # NCAR default algo; q_sfc / rho_air pinned explicitly so only the
    # sign conventions are under test (not the internal Goff/moist-rho).
    tau_x, tau_y, sh, lh, evap = air_sea_fluxes(
        u10=jnp.array(10.0), v10=jnp.array(0.0),
        T_air_K=jnp.array(290.0), q_air=jnp.array(0.005),
        T_sfc_K=jnp.array(295.0), q_sfc=jnp.array(0.012),
        rho_air=jnp.array(1.2),
    )
    # tau_x: stress on the OCEAN follows the wind direction (drag on
    # the ocean from above is positive eastward when u10 > 0).
    # ``air_sea_fluxes`` uses the atmosphere convention "stress
    # opposes wind" so tau_x is negative here.
    assert float(tau_x) < 0.0
    # SST warmer than air -> ocean heats the atmosphere -> shflx into
    # the ocean is negative.
    assert float(sh) < 0.0
    # Air drier than sea-saturation specific humidity here (q_air
    # 0.005 < q_sfc 0.012) so lhflx into the ocean is negative
    # (evaporative cooling) and the returned evaporation is positive-up.
    assert float(lh) < 0.0
    assert float(evap) > 0.0


# ---------------------------------------------------------------------------
# Climate diagnostics
# ---------------------------------------------------------------------------

def test_amoc_finds_max_at_target_latitude():
    from legoesm.ocean.diagnostics_climate import amoc_at_latitude
    # Synthetic MOC: peak 15 Sv at lat 26.5, depth 1000 m.
    lats = np.arange(-89.5, 90.0, 1.0)
    depths = np.linspace(0.0, 5000.0, 51)
    LAT, Z = np.meshgrid(lats, depths, indexing="ij")
    psi = 15e6 * np.exp(
        -((LAT - 26.5) / 5.0) ** 2 - ((Z - 1000.0) / 800.0) ** 2
    )
    res = amoc_at_latitude(psi, lats, depths, target_lat=26.5)
    assert res.streamfunction_Sv == pytest.approx(15.0, abs=0.5)
    assert res.target_lat_deg == pytest.approx(26.5, abs=0.5)
    assert res.depth_of_max_m == pytest.approx(1000.0, abs=200.0)


def test_acc_transport_max_minus_min():
    from legoesm.ocean.diagnostics_climate import acc_transport
    lats = np.arange(-89.5, 90.0, 1.0)
    n_lon = 36
    # Barotropic streamfunction: 0 at coastlines, 130 Sv contour
    # across Drake band.
    psi = np.zeros((lats.size, n_lon))
    band = (lats >= -65) & (lats <= -45)
    psi[band, :] = 130e6
    psi[band, 0] = 0.0   # west coast of Drake = 0; east = 130 Sv
    res = acc_transport(psi, lats)
    assert res.transport_Sv == pytest.approx(130.0, abs=1.0)


def test_sst_bias_zero_when_model_matches_ref():
    from legoesm.ocean.diagnostics_climate import sst_climatology_bias
    sst = np.full((36, 72), 288.0)
    area = np.ones_like(sst)
    res = sst_climatology_bias(sst, sst.copy(), area)
    assert res.bias_K == pytest.approx(0.0, abs=1e-12)
    assert res.rmse_K == pytest.approx(0.0, abs=1e-12)


def test_sst_bias_picks_up_uniform_offset():
    from legoesm.ocean.diagnostics_climate import sst_climatology_bias
    sst = np.full((36, 72), 290.0)
    ref = np.full_like(sst, 288.0)
    area = np.ones_like(sst)
    res = sst_climatology_bias(sst, ref, area)
    assert res.bias_K == pytest.approx(2.0, abs=1e-12)
    assert res.rmse_K == pytest.approx(2.0, abs=1e-12)
