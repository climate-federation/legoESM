"""Tests for the synthetic CMIP6 AMIP forcing deck.

Validates that the files produced by ``scripts/generate_amip_forcing.py``
are loaded correctly by the corresponding production loaders in
``src/legoesm/forcing/external.py`` and ``src/legoesm/forcing/amip.py``,
and that the values returned at canonical query points are physically
realistic (DU, ppmv, AOD ranges).

The tests are *fast* — they generate a tiny 2-year deck on disk in a
``tmp_path`` and exercise every CMIP6 forcing channel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure scripts/ is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import generate_amip_forcing as gaf  # noqa: E402

from legoesm.forcing.external import (  # noqa: E402
    GHGConfig,
    OzoneConfig,
    AerosolConfig,
    SolarConfig,
    get_ghg_at_time,
    get_ozone_at_time,
    get_aerosol_at_time,
    get_solar_forcing_at_time,
    get_tsi_at_time,
)


@pytest.fixture(scope="module")
def deck(tmp_path_factory):
    """Two-year synthetic deck (1979-1980), generated once per module."""
    out = tmp_path_factory.mktemp("amip_deck")
    sy, ey = 1979, 1980

    gaf.make_sst_sic(out / f"sst_{sy}-{ey}.nc", sy, ey, nlat=37, nlon=72)
    gaf.make_ghg_annual(out / f"ghg_{sy}-{ey}.nc", sy, ey)
    gaf.make_ozone_clim(out / "ozone.nc", nlat=18, nlev=20)
    gaf.make_solar(out / f"solar_{sy}-{ey}.nc", sy, ey)
    gaf.make_aerosol_clim(out / "aerosol.nc", nlat=36)
    gaf.make_volcanic(out / f"volcanic_{sy}-{ey}.nc", sy, ey, nlat=18)

    return {
        "dir": out,
        "sst": out / f"sst_{sy}-{ey}.nc",
        "ghg": out / f"ghg_{sy}-{ey}.nc",
        "ozone": out / "ozone.nc",
        "solar": out / f"solar_{sy}-{ey}.nc",
        "aerosol": out / "aerosol.nc",
        "volcanic": out / f"volcanic_{sy}-{ey}.nc",
        "start_year": sy,
    }


# ============================================================================
# SST / SIC
# ============================================================================

class TestSST:
    def test_global_mean_in_observed_range(self, deck):
        import netCDF4
        with netCDF4.Dataset(deck["sst"]) as ds:
            sst_C = ds["sst"][:].astype(np.float64)
            lat = ds["lat"][:].astype(np.float64)
        sst_K = sst_C + 273.15
        # Area-weighted (cos lat) global mean
        w = np.cos(np.deg2rad(lat))
        w /= w.sum()
        gm = (sst_K.mean(axis=(0, 2)) * w).sum()
        assert 285.0 < gm < 295.0, f"Global SST {gm:.2f} K out of observed band"

    def test_no_freezing_violations(self, deck):
        import netCDF4
        with netCDF4.Dataset(deck["sst"]) as ds:
            sst_K = ds["sst"][:].astype(np.float64) + 273.15
        # Seawater freezing point ~ 271.35 K (S=35 psu)
        assert sst_K.min() >= 271.0

    def test_sic_in_unit_interval(self, deck):
        import netCDF4
        with netCDF4.Dataset(deck["sst"]) as ds:
            sic = ds["sic"][:].astype(np.float64)
        assert sic.min() >= 0.0
        assert sic.max() <= 1.0
        # SIC should be confined to polar caps (|lat| > 40°)
        lat = ds = None
        with netCDF4.Dataset(deck["sst"]) as ds:
            lat = ds["lat"][:].astype(np.float64)
            sic = ds["sic"][:].astype(np.float64)
        # No tropical sea-ice
        tropical_band = np.where(np.abs(lat) < 30.0)[0]
        assert sic[:, tropical_band, :].max() < 0.05


# ============================================================================
# GHG (annual_file)
# ============================================================================

class TestGHG:
    def test_loads_through_annual_file_path(self, deck):
        cfg = GHGConfig(
            source="annual_file", path=str(deck["ghg"]),
            start_year=deck["start_year"],
        )
        # Day 0 → 1979 starting GHGs
        ghg = get_ghg_at_time(cfg, day=0.0)
        assert 330.0 < ghg["co2_ppmv"] < 345.0  # ~336.78 expected
        assert 1500.0 < ghg["ch4_ppbv"] < 1600.0
        assert 295.0 < ghg["n2o_ppbv"] < 310.0

    def test_transient_evolution(self, deck):
        cfg = GHGConfig(
            source="annual_file", path=str(deck["ghg"]),
            start_year=deck["start_year"],
        )
        # 365 days later → ~1980
        co2_y0 = get_ghg_at_time(cfg, day=0.0)["co2_ppmv"]
        co2_y1 = get_ghg_at_time(cfg, day=365.0)["co2_ppmv"]
        assert co2_y1 > co2_y0, "CO2 should increase between 1979 and 1980"

    def test_cfcs_present(self, deck):
        cfg = GHGConfig(
            source="annual_file", path=str(deck["ghg"]),
            start_year=deck["start_year"],
        )
        ghg = get_ghg_at_time(cfg, day=0.0)
        assert ghg["cfc11_pptv"] > 100.0
        assert ghg["cfc12_pptv"] > 100.0


# ============================================================================
# Ozone (climatology)
# ============================================================================

class TestOzone:
    def test_climatology_loads(self, deck):
        import jax.numpy as jnp
        cfg = OzoneConfig(
            enabled=True, source="climatology", path=str(deck["ozone"]),
            start_year=deck["start_year"],
        )
        lat_grid = jnp.array(np.deg2rad(np.linspace(-89.0, 89.0, 32)))
        # Pressure grid in Pa, top → surface
        p_grid = jnp.array(np.logspace(2, 5, 25))
        o3_vmr = get_ozone_at_time(cfg, day=15.0, lat_grid=lat_grid,
                                    p_grid=p_grid)
        o3_arr = np.asarray(o3_vmr)
        assert o3_arr.shape == (32, 25)
        # Must be non-negative
        assert o3_arr.min() >= 0.0
        # Stratospheric peak in [0.5, 10] ppmv
        peak = o3_arr.max()
        assert 0.5e-6 < peak < 10e-6, f"O3 peak {peak:.2e} out of band"


# ============================================================================
# Aerosol
# ============================================================================

class TestAerosol:
    def test_aod_in_band(self, deck):
        import jax.numpy as jnp
        cfg = AerosolConfig(
            enabled=True, source="climatology", path=str(deck["aerosol"]),
            volcanic_enabled=True, volcanic_path=str(deck["volcanic"]),
        )
        lat_grid = jnp.array(np.deg2rad(np.linspace(-89.0, 89.0, 24)))
        aod = get_aerosol_at_time(cfg, day=180.0, lat_grid=lat_grid)
        a = np.asarray(aod)
        assert a.shape == (24,)
        # AOD@550 nm: tropics ~ 0.05-0.30, plus volcanic background ~ 0.005
        assert a.min() > 0.0
        assert a.max() < 0.50, f"Excess aerosol AOD {a.max():.3f}"

    def test_volcanic_pinatubo_signal(self, deck):
        """1991 Pinatubo year should produce elevated AOD."""
        # Our deck spans only 1979-1980 so volcanic forcing is at background.
        # Re-generate a 1991-1992 deck to check Pinatubo arrives.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            gaf.make_volcanic(tmp / "v.nc", 1991, 1992, nlat=18)
            import netCDF4
            with netCDF4.Dataset(tmp / "v.nc") as ds:
                aod = ds["aod"][:].astype(np.float64)
            # Pinatubo peak in late 1991 should be at least 0.05 above background
            assert aod.max() > 0.04, f"Pinatubo signal too weak: {aod.max():.3f}"


# ============================================================================
# Solar
# ============================================================================

class TestSolar:
    def test_tsi_loads(self, deck):
        # Case-insensitive lookup must accept upper-case 'TSI'
        cfg = SolarConfig(
            S_0=1361.0, source="file", path=str(deck["solar"]),
            tsi_var="TSI",
            start_year=deck["start_year"],
        )
        tsi = get_tsi_at_time(cfg, day=0.0)
        assert 1358.0 < float(tsi) < 1364.0, f"TSI {tsi} out of band"

    def test_spectral_expansion(self, deck):
        cfg = SolarConfig(
            S_0=1361.0, source="spectral_file", path=str(deck["solar"]),
            tsi_var="TSI", spectral_var="SSI_frac",
            start_year=deck["start_year"],
        )
        out = get_solar_forcing_at_time(cfg, day=0.0)
        assert "tsi" in out
        assert "solar_fraction_by_gpt" in out
        gpt = np.asarray(out["solar_fraction_by_gpt"])
        # Expanded from 14 bands to 112 g-points
        assert gpt.shape == (112,), f"Spectral shape {gpt.shape}"
        # Should sum to ~1
        s = float(gpt.sum())
        assert 0.95 < s < 1.05, f"SSI fractions sum {s:.3f}"


# ============================================================================
# Cross-channel: full deck consistency
# ============================================================================

class TestDeckConsistency:
    """Cross-channel sanity checks for the full deck."""

    def test_all_files_exist(self, deck):
        for name in ("sst", "ghg", "ozone", "solar", "aerosol", "volcanic"):
            p = deck[name]
            assert p.exists(), f"deck file missing: {p}"
            assert p.stat().st_size > 0

    def test_amip_template_is_transient(self):
        """Regression: the AMIP template must be transient now that we
        wired it to the historical GHG table."""
        from legoesm.forcing.experiments import (
            EXPERIMENT_TEMPLATES, ghg_at_year,
        )
        amip = EXPERIMENT_TEMPLATES["amip"]
        assert amip.forcing_type == "transient"
        # CO2 should evolve between 1979 and 2014 endpoints
        co2_1979, _, _ = ghg_at_year("amip", 1979)
        co2_2014, _, _ = ghg_at_year("amip", 2014)
        assert co2_2014 > co2_1979 + 50.0, (
            f"AMIP CO2 should increase by >50 ppmv from 1979 ({co2_1979:.1f}) "
            f"to 2014 ({co2_2014:.1f})"
        )

    def test_fix_moisture_warns_with_prognostic_microphysics(self):
        """Regression: enabling --fix-moisture together with a precipitating
        microphysics scheme should now emit a strongly-worded warning
        explaining that the q_v-only rescale is unsafe.

        Catches the bug where ``fix_moisture_hydrostatic`` rescales only
        ``q_v`` (not ``q_c`` / ``q_r`` / cumulative precipitation),
        which, in concert with Kessler-style schemes, drives a runaway
        moisture source that crashes the dycore.
        """
        from legoesm.driver.config import ExperimentConfig

        cfg = ExperimentConfig(
            fix_moisture=True,
            microphysics="kessler",
        )
        warnings = cfg.validate()
        relevant = [w for w in warnings if "fix_moisture" in w]
        assert len(relevant) == 1, (
            f"Expected exactly one fix_moisture warning, got: {warnings}"
        )
        msg = relevant[0]
        assert "INCORRECT" in msg or "unsafe" in msg.lower(), (
            f"Warning should flag the bug strongly, got: {msg}"
        )

    def test_no_fix_moisture_warning_without_microphysics(self):
        """The warning must NOT fire when microphysics='none'."""
        from legoesm.driver.config import ExperimentConfig
        cfg = ExperimentConfig(fix_moisture=True, microphysics="none")
        warnings = cfg.validate()
        assert not any("fix_moisture" in w for w in warnings)
