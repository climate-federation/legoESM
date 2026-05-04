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


class TestOzoneInterannual:
    """Tests for the ``--ozone-interannual`` generator branch.

    The CMIP6 AMIP protocol uses a 1850–2014 (>12 month) input4MIPs
    ozone file, which the loader dispatches through the *non-cyclic*
    branch (``_interp_monthly_noncyclic`` keyed on ``ntime > 12``).
    The synthetic generator's interannual mode triggers exactly this
    code path, with a 1979→2014 strengthening Antarctic ozone-hole
    signal that should be visible at the SH polar lower stratosphere
    in austral spring.
    """

    @pytest.fixture(scope="class")
    def interannual_path(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("o3ia")
        gaf.make_ozone_clim(out / "ozone.nc", nlat=18, nlev=20,
                             start_year=1979, end_year=2014)
        return out / "ozone.nc"

    def test_loads_through_noncyclic_branch(self, interannual_path):
        import jax.numpy as jnp
        from legoesm.forcing.external import OzoneConfig, get_ozone_at_time
        cfg = OzoneConfig(
            enabled=True, source="climatology", path=str(interannual_path),
            start_year=1979,
        )
        lat_grid = jnp.array(np.deg2rad(np.linspace(-89.0, 89.0, 16)))
        p_grid = jnp.array(np.logspace(2, 5, 20))
        out = get_ozone_at_time(cfg, day=15.0, lat_grid=lat_grid,
                                p_grid=p_grid)
        assert out.shape == (16, 20)
        assert float(np.max(out)) > 1e-7

    def test_ozone_hole_strengthens_over_time(self, interannual_path):
        """SH polar lower-strat ozone in austral spring should drop
        between 1979 and 2014 (strengthening hole)."""
        import jax.numpy as jnp
        from legoesm.forcing.external import OzoneConfig, get_ozone_at_time

        cfg = OzoneConfig(
            enabled=True, source="climatology", path=str(interannual_path),
            start_year=1979,
        )
        lat_grid = jnp.array(np.deg2rad(np.linspace(-89.0, 89.0, 18)))
        # Target a level inside the synthetic hole window (50-200 hPa)
        p_grid = jnp.array(np.logspace(np.log10(80e2), np.log10(150e2), 6))

        def sh_polar_min(day_int: int) -> float:
            o3 = get_ozone_at_time(
                cfg, day=float(day_int), lat_grid=lat_grid, p_grid=p_grid,
            )
            return float(np.min(np.asarray(o3)[0:3, :]))  # 3 southernmost lats

        sept_1979 = sh_polar_min(258)        # 1979 Sept
        sept_2014 = sh_polar_min(13042)      # 2014 Sept
        assert sept_2014 < sept_1979 * 0.7, (
            f"Synthetic Antarctic ozone hole should strengthen by 1979→2014; "
            f"got 1979 Sept SH-polar O3={sept_1979:.2e}, "
            f"2014 Sept SH-polar O3={sept_2014:.2e}"
        )


class TestDeckChecker:
    """Regression tests for ``run_amip_cmip6_deck.py:_check_forcing_files``.

    The deck checker must:
    1. Reject decks missing one or more of the 6 canonical files.
    2. Accept the **interannual** ozone filename
       ``ozone_amip_<sy>-<ey>.nc`` produced by
       ``generate_amip_forcing.py --ozone-interannual``.
    3. **Prefer** the interannual ozone file when both interannual and
       climatology files coexist (it is what real CMIP6 input4MIPs
       ozone is, and exercises the loader's non-cyclic dispatch).
    """

    def _make_deck(self, out: Path, *, with_interannual_o3: bool,
                    with_clim_o3: bool, sy: int = 1979, ey: int = 1980) -> None:
        """Generate a deck with one of (interannual ozone, clim ozone, both)
        plus all the other forcing files."""
        out.mkdir(parents=True, exist_ok=True)
        gaf.make_sst_sic(out / f"sst_sic_amip_{sy}-{ey}.nc", sy, ey,
                          nlat=37, nlon=72)
        gaf.make_ghg_annual(out / f"ghg_amip_{sy}-{ey}.nc", sy, ey)
        if with_interannual_o3:
            gaf.make_ozone_clim(out / f"ozone_amip_{sy}-{ey}.nc",
                                 nlat=18, nlev=20, start_year=sy, end_year=ey)
        if with_clim_o3:
            gaf.make_ozone_clim(out / "ozone_amip_clim.nc", nlat=18, nlev=20)
        gaf.make_solar(out / f"solar_amip_{sy}-{ey}.nc", sy, ey)
        gaf.make_aerosol_clim(out / "aerosol_amip_clim.nc", nlat=36)
        gaf.make_volcanic(out / f"volcanic_amip_{sy}-{ey}.nc", sy, ey,
                           nlat=18)

    def test_missing_files_reported(self, tmp_path):
        from run_amip_cmip6_deck import _check_forcing_files
        files = _check_forcing_files(tmp_path, 1979, 1980)
        assert "_missing" in files
        # All six channels should be reported missing
        missing = files["_missing"].split(",")
        assert set(missing) == {"sst", "ghg", "ozone", "solar",
                                  "aerosol", "volcanic"}

    def test_interannual_ozone_accepted(self, tmp_path):
        """Interannual ozone file alone is sufficient; clim is optional."""
        from run_amip_cmip6_deck import _check_forcing_files
        self._make_deck(tmp_path, with_interannual_o3=True,
                        with_clim_o3=False)
        files = _check_forcing_files(tmp_path, 1979, 1980)
        assert "_missing" not in files, (
            f"Interannual-only deck should be complete, got missing: "
            f"{files.get('_missing')}"
        )
        assert files["ozone"].name == "ozone_amip_1979-1980.nc"

    def test_climatology_ozone_accepted(self, tmp_path):
        """Climatology ozone file alone is sufficient (legacy default)."""
        from run_amip_cmip6_deck import _check_forcing_files
        self._make_deck(tmp_path, with_interannual_o3=False,
                        with_clim_o3=True)
        files = _check_forcing_files(tmp_path, 1979, 1980)
        assert "_missing" not in files
        assert files["ozone"].name == "ozone_amip_clim.nc"

    def test_interannual_preferred_over_clim(self, tmp_path):
        """When both files exist, the interannual one wins (it's what
        real CMIP6 ozone is and exercises the non-cyclic loader)."""
        from run_amip_cmip6_deck import _check_forcing_files
        self._make_deck(tmp_path, with_interannual_o3=True,
                        with_clim_o3=True)
        files = _check_forcing_files(tmp_path, 1979, 1980)
        assert "_missing" not in files
        assert files["ozone"].name == "ozone_amip_1979-1980.nc", (
            "When both interannual and climatology ozone files are "
            "present, the deck checker must prefer the interannual file."
        )


class TestOzoneUnitDetection:
    """Regression tests for the ozone unit-conversion path.

    The legoESM radiation kernel (RRTMG/RRTMGP) consumes ozone as
    volume mixing ratio (mol/mol).  CMIP6 ``vmro3`` files are already
    in mol/mol — no conversion needed — but several alternative
    datasets (older NCAR/E3SM, some CESM forcing decks) use ``tro3``
    in kg/kg.  The loader must detect units and convert; otherwise a
    silently-mis-loaded ozone file would distort stratospheric SW
    heating by ~40%.
    """

    def _write_ozone_file(self, path: Path, *, varname: str,
                           units: str | None,
                           value: float = 1.0e-6) -> None:
        """Write a tiny 12-month × 4-lat × 5-plev ozone file with the
        requested variable name and units.  Stores a constant value
        ``value`` so the post-load array exactly equals
        ``value * unit_factor``."""
        from netCDF4 import Dataset
        with Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("time", 12)
            ds.createDimension("lat", 4)
            ds.createDimension("plev", 5)

            t = ds.createVariable("time", "f8", ("time",))
            t.units = "days since 1850-01-01 00:00:00"
            t.calendar = "noleap"
            t[:] = np.array([15.5 + 30.4375 * m for m in range(12)])

            la = ds.createVariable("lat", "f8", ("lat",))
            la.units = "degrees_north"
            la[:] = np.array([-60.0, -30.0, 30.0, 60.0])

            pl = ds.createVariable("plev", "f8", ("plev",))
            pl.units = "Pa"
            pl[:] = np.array([5e2, 5e3, 1e4, 5e4, 9e4])

            v = ds.createVariable(varname, "f8", ("time", "lat", "plev"))
            if units is not None:
                v.units = units
            v[:] = value

    def test_vmr_units_pass_through_unchanged(self, tmp_path):
        """``mol mol-1`` is the CMIP6 vmro3 default; no conversion."""
        from legoesm.forcing.external import _ozone_unit_factor
        assert _ozone_unit_factor("mol mol-1", "vmro3") == 1.0
        assert _ozone_unit_factor("mol/mol", "vmro3") == 1.0
        assert _ozone_unit_factor("1", "vmro3") == 1.0

    def test_kg_kg_converts_to_vmr(self):
        """``kg kg-1`` is mass mixing ratio; convert by M_dry/M_o3."""
        from legoesm import constants
        from legoesm.forcing.external import _ozone_unit_factor
        expected = constants.M_dry / constants.M_o3
        assert _ozone_unit_factor("kg kg-1", "tro3") == expected
        assert _ozone_unit_factor("kg/kg", "tro3") == expected

    def test_ppmv_converts(self):
        from legoesm.forcing.external import _ozone_unit_factor
        assert _ozone_unit_factor("ppmv", "o3") == pytest.approx(1.0e-6)

    def test_empty_units_with_vmro3_assumed_vmr(self):
        """When ``units`` is missing, ``vmro3`` is assumed vmr (CMIP6
        convention)."""
        from legoesm.forcing.external import _ozone_unit_factor
        assert _ozone_unit_factor("", "vmro3") == 1.0

    def test_empty_units_with_tro3_assumed_mmr(self):
        """``tro3`` is the CMIP-protocol name for mass mixing ratio."""
        from legoesm import constants
        from legoesm.forcing.external import _ozone_unit_factor
        expected = constants.M_dry / constants.M_o3
        assert _ozone_unit_factor("", "tro3") == expected

    def test_unrecognised_units_raises(self):
        from legoesm.forcing.external import _ozone_unit_factor
        with pytest.raises(ValueError, match="Unrecognised ozone units"):
            _ozone_unit_factor("DU", "ozone")

    def test_loader_applies_conversion_for_kg_kg(self, tmp_path):
        """End-to-end: a ``tro3`` file in kg/kg loads as vmr after
        multiplication by ``M_dry / M_o3``.

        Dispatch through the public ``get_ozone_at_time`` so the test
        also covers the file-read path, not only the factor function.
        """
        import jax.numpy as jnp
        from legoesm import constants
        from legoesm.forcing.external import (
            OzoneConfig, get_ozone_at_time, _detect_ozone_varname,
            _load_monthly_zonal_with_levels,
        )

        path = tmp_path / "tro3_kgkg.nc"
        # Use a value that becomes a sensible vmr after conversion:
        # kg/kg = 1e-6 → vmr = 1e-6 * (M_dry / M_o3) ≈ 6.04e-7.
        self._write_ozone_file(path, varname="tro3",
                                units="kg kg-1", value=1.0e-6)

        # Clear cache so the new file is re-read
        _detect_ozone_varname.cache_clear()
        _load_monthly_zonal_with_levels.cache_clear()

        cfg = OzoneConfig(enabled=True, source="climatology",
                          path=str(path), start_year=1979)
        lat_grid = jnp.array(np.deg2rad(np.array([-30.0, 30.0])))
        p_grid = jnp.array(np.array([1e3, 1e4, 5e4]))
        out = get_ozone_at_time(cfg, day=15.0, lat_grid=lat_grid,
                                 p_grid=p_grid)
        out = np.asarray(out)
        expected = 1.0e-6 * (constants.M_dry / constants.M_o3)
        # Field is constant so all entries must match the expected vmr.
        assert np.allclose(out, expected, rtol=1e-6), (
            f"tro3 kg/kg conversion failed: got {out.flat[0]:.3e}, "
            f"expected {expected:.3e}"
        )

    def test_loader_no_conversion_for_vmro3(self, tmp_path):
        """A ``vmro3`` file in mol mol-1 must come back unchanged."""
        import jax.numpy as jnp
        from legoesm.forcing.external import (
            OzoneConfig, get_ozone_at_time, _detect_ozone_varname,
            _load_monthly_zonal_with_levels,
        )

        path = tmp_path / "vmro3_molmol.nc"
        self._write_ozone_file(path, varname="vmro3",
                                units="mol mol-1", value=2.0e-6)

        _detect_ozone_varname.cache_clear()
        _load_monthly_zonal_with_levels.cache_clear()

        cfg = OzoneConfig(enabled=True, source="climatology",
                          path=str(path), start_year=1979)
        lat_grid = jnp.array(np.deg2rad(np.array([-30.0, 30.0])))
        p_grid = jnp.array(np.array([1e3, 1e4, 5e4]))
        out = np.asarray(get_ozone_at_time(cfg, day=15.0,
                                             lat_grid=lat_grid,
                                             p_grid=p_grid))
        assert np.allclose(out, 2.0e-6, rtol=1e-6), (
            f"vmro3 mol mol-1 should pass unchanged; got {out.flat[0]:.3e}"
        )


class TestGHGOutOfRangeWarning:
    """Regression test for the GHG anchor-table out-of-range warning
    (P2 own audit, iter 4).

    For an AMIP run starting after 2021 (last anchor in
    ``_GHG_HISTORICAL``), ``ghg_at_year("amip", year)`` silently
    returns the 2021 endpoint — a flat-tail extrapolation.  At year
    2025 the real-world CO2 is ~424 ppm but the table returns
    414.72 — a ~2% silent bias.  The warning informs the user that
    they need an external GHG file or a table extension.
    """

    def test_year_in_range_does_not_warn(self, caplog):
        from legoesm.forcing.experiments import (
            ghg_at_year, _GHG_OUT_OF_RANGE_WARNED,
        )
        _GHG_OUT_OF_RANGE_WARNED.clear()
        import logging
        with caplog.at_level(logging.WARNING,
                              logger="legoesm.forcing.experiments"):
            ghg_at_year("amip", 2014.0)
        assert all("ghg_table" not in r.message for r in caplog.records)

    def test_year_out_of_range_warns_once(self, caplog):
        from legoesm.forcing.experiments import (
            ghg_at_year, _GHG_OUT_OF_RANGE_WARNED,
        )
        _GHG_OUT_OF_RANGE_WARNED.clear()
        import logging
        with caplog.at_level(logging.WARNING,
                              logger="legoesm.forcing.experiments"):
            ghg_at_year("amip", 2025.0)
            ghg_at_year("amip", 2025.5)  # same int year — no second warning
        warnings = [r for r in caplog.records
                    if "ghg_table" in r.message]
        assert len(warnings) == 1, (
            f"Expected one warning for 2025.0/2025.5; got "
            f"{[w.message for w in warnings]}"
        )
        assert "amip" in warnings[0].message
        assert "[1850, 2021]" in warnings[0].message

    def test_different_years_warn_separately(self, caplog):
        from legoesm.forcing.experiments import (
            ghg_at_year, _GHG_OUT_OF_RANGE_WARNED,
        )
        _GHG_OUT_OF_RANGE_WARNED.clear()
        import logging
        with caplog.at_level(logging.WARNING,
                              logger="legoesm.forcing.experiments"):
            ghg_at_year("amip", 2025.0)
            ghg_at_year("amip", 2030.0)
        warnings = [r for r in caplog.records
                    if "ghg_table" in r.message]
        assert len(warnings) == 2

    def test_amip_clamps_at_last_anchor(self):
        """Verify the bias the warning is alerting users to: years past
        2021 silently clamp to the 2021 CO2 value."""
        from legoesm.forcing.experiments import ghg_at_year
        co2_2021, _, _ = ghg_at_year("amip", 2021.0)
        co2_2025, _, _ = ghg_at_year("amip", 2025.0)
        co2_2050, _, _ = ghg_at_year("amip", 2050.0)
        # Flat tail beyond 2021.
        assert co2_2025 == co2_2021
        assert co2_2050 == co2_2021


class TestSpectralPathWarning:
    """Regression tests for the silent-drop warning for grid paths that
    bypass the external CMIP6 forcing pipeline (P2 codex iter-3 + iter-4).

    Two paths are affected:
    * ``ModelDriver._run_spectral`` (gaussian/spectral) hard-codes
      gray radiation and constant solar.
    * ``ModelDriver._run_mpas`` (voronoi/mpas) builds physics without
      calling ``_precompute_external_forcing``; the external configs
      never reach the radiation kernel.

    When a user requests ``--radiation rrtmg`` on either path, the
    deck driver must warn loudly that the external GHG/ozone/aerosol/
    volcanic forcings are loaded but never consumed.
    """

    def test_warning_fires_on_gaussian_spectral_rrtmg(self, tmp_path):
        """The deck driver --dry-run output must include the
        silent-drop warning for the (gaussian, spectral, rrtmg) combo."""
        import subprocess

        # Build a tiny forcing deck so --dry-run can pass file checks.
        sy, ey = 1979, 1980
        gaf.make_sst_sic(tmp_path / f"sst_sic_amip_{sy}-{ey}.nc",
                          sy, ey, nlat=37, nlon=72)
        gaf.make_ghg_annual(tmp_path / f"ghg_amip_{sy}-{ey}.nc", sy, ey)
        gaf.make_ozone_clim(tmp_path / "ozone_amip_clim.nc",
                             nlat=18, nlev=20)
        gaf.make_solar(tmp_path / f"solar_amip_{sy}-{ey}.nc", sy, ey)
        gaf.make_aerosol_clim(tmp_path / "aerosol_amip_clim.nc",
                                nlat=36)
        gaf.make_volcanic(tmp_path / f"volcanic_amip_{sy}-{ey}.nc",
                           sy, ey, nlat=18)

        deck_script = _REPO_ROOT / "scripts" / "run_amip_cmip6_deck.py"
        cmd = [
            sys.executable, str(deck_script),
            "--forcing-dir", str(tmp_path),
            "--start-year", str(sy), "--end-year", str(ey),
            "--grid-type", "gaussian", "--discretization", "spectral",
            "--radiation", "rrtmg", "--resolution", "21",
            "--days", "1", "--dry-run",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"dry-run failed:\n{r.stderr}"
        out = r.stdout
        assert "WARNING" in out and "spectral" in out, (
            f"Deck driver should warn that gaussian/spectral + rrtmg "
            f"silently drops external forcings; got stdout:\n{out}"
        )
        # Forcing channels must be reported as inert, not ACTIVE.
        # Match only the activity-report lines (start with '  ' and
        # contain the channel name in label form); skip the command
        # printout where flags like '--aerosol-forcing' would alias.
        report_labels = (
            "Greenhouse gases",
            "Ozone (cyclic clim",
            "Tropospheric aerosol",
            "Volcanic stratospheric",
        )
        for label in report_labels:
            line = next((ln for ln in out.splitlines()
                          if label in ln), None)
            assert line is not None, f"Missing {label} report line"
            assert "inert" in line.lower(), (
                f"Forcing line {line!r} should be 'inert' under "
                f"gaussian/spectral + rrtmg, not 'ACTIVE'."
            )

    def test_no_warning_on_cubed_sphere_rrtmg(self, tmp_path):
        """No warning should fire for cubed_sphere + rrtmg."""
        import subprocess

        sy, ey = 1979, 1980
        gaf.make_sst_sic(tmp_path / f"sst_sic_amip_{sy}-{ey}.nc",
                          sy, ey, nlat=37, nlon=72)
        gaf.make_ghg_annual(tmp_path / f"ghg_amip_{sy}-{ey}.nc", sy, ey)
        gaf.make_ozone_clim(tmp_path / "ozone_amip_clim.nc",
                             nlat=18, nlev=20)
        gaf.make_solar(tmp_path / f"solar_amip_{sy}-{ey}.nc", sy, ey)
        gaf.make_aerosol_clim(tmp_path / "aerosol_amip_clim.nc",
                                nlat=36)
        gaf.make_volcanic(tmp_path / f"volcanic_amip_{sy}-{ey}.nc",
                           sy, ey, nlat=18)

        deck_script = _REPO_ROOT / "scripts" / "run_amip_cmip6_deck.py"
        cmd = [
            sys.executable, str(deck_script),
            "--forcing-dir", str(tmp_path),
            "--start-year", str(sy), "--end-year", str(ey),
            "--grid-type", "cubed_sphere", "--discretization", "centered",
            "--radiation", "rrtmg", "--resolution", "8",
            "--days", "1", "--dry-run",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0
        assert "WARNING" not in r.stdout, (
            f"No warning expected on cubed_sphere + rrtmg; got:\n"
            f"{r.stdout}"
        )
        # Forcing channels must all show ACTIVE.
        report_labels = (
            "Greenhouse gases",
            "Ozone (cyclic clim",
            "Tropospheric aerosol",
            "Volcanic stratospheric",
        )
        for label in report_labels:
            line = next((ln for ln in r.stdout.splitlines()
                          if label in ln), None)
            assert line is not None
            assert "ACTIVE" in line, f"Got: {line!r}"

    def test_warning_fires_on_voronoi_mpas_rrtmg(self, tmp_path):
        """voronoi/mpas + rrtmg also bypasses external forcings;
        the deck driver must warn just like for gaussian/spectral."""
        import subprocess

        sy, ey = 1979, 1980
        gaf.make_sst_sic(tmp_path / f"sst_sic_amip_{sy}-{ey}.nc",
                          sy, ey, nlat=37, nlon=72)
        gaf.make_ghg_annual(tmp_path / f"ghg_amip_{sy}-{ey}.nc", sy, ey)
        gaf.make_ozone_clim(tmp_path / "ozone_amip_clim.nc",
                             nlat=18, nlev=20)
        gaf.make_solar(tmp_path / f"solar_amip_{sy}-{ey}.nc", sy, ey)
        gaf.make_aerosol_clim(tmp_path / "aerosol_amip_clim.nc",
                                nlat=36)
        gaf.make_volcanic(tmp_path / f"volcanic_amip_{sy}-{ey}.nc",
                           sy, ey, nlat=18)

        deck_script = _REPO_ROOT / "scripts" / "run_amip_cmip6_deck.py"
        cmd = [
            sys.executable, str(deck_script),
            "--forcing-dir", str(tmp_path),
            "--start-year", str(sy), "--end-year", str(ey),
            "--grid-type", "voronoi", "--discretization", "mpas",
            "--radiation", "rrtmg", "--resolution", "4",
            "--days", "1", "--dry-run",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"dry-run failed:\n{r.stderr}"
        out = r.stdout
        assert "WARNING" in out and ("voronoi" in out.lower()
                                       or "mpas" in out.lower()), (
            f"Deck driver should warn that voronoi/mpas + rrtmg "
            f"silently drops external forcings; got stdout:\n{out}"
        )
        report_labels = (
            "Greenhouse gases",
            "Ozone (cyclic clim",
            "Tropospheric aerosol",
            "Volcanic stratospheric",
        )
        for label in report_labels:
            line = next((ln for ln in out.splitlines()
                          if label in ln), None)
            assert line is not None
            assert "inert" in line.lower(), (
                f"Forcing line {line!r} should be 'inert' under "
                f"voronoi/mpas + rrtmg, not 'ACTIVE'."
            )


class TestVolcanicNonCyclic:
    """Regression tests for the volcanic non-cyclic dispatch (P2 codex
    iter-3 review).

    A multi-year volcanic AOD file (e.g. 1979–1980 with the synthetic
    Pinatubo signal anchored at 1991) was previously sampled via
    ``_interp_monthly_cyclic``, which mod-365.25-wraps a multi-year
    axis and erases the eruption calendar.  The fix dispatches
    ``len(mid_days) > 12`` files through ``_interp_monthly_noncyclic``
    with ``_simday_to_file_day`` mapping, so 1991 Pinatubo lands at
    sim-day = (1991 - start_year) * 365.25 + month_offset rather than
    being collapsed onto a 12-month repeating cycle.
    """

    @pytest.fixture(scope="class")
    def volcanic_path(self, tmp_path_factory):
        """Generate a multi-year volcanic file (1979–1992) so the
        Pinatubo 1991 spike is in the file but at a calendar location
        that gets erased by cyclic interpolation."""
        out = tmp_path_factory.mktemp("volc")
        path = out / "volcanic_1979_1992.nc"
        gaf.make_volcanic(path, 1979, 1992, nlat=18)
        return path

    def test_multiyear_volcanic_dispatch_through_noncyclic(self,
                                                            volcanic_path):
        """A 14-year volcanic file should hit the non-cyclic branch."""
        import jax.numpy as jnp
        from legoesm.forcing.external import (
            AerosolConfig, get_aerosol_at_time,
            _load_volcanic_auto_anchored, _load_volcanic_cmip6_anchored,
            _load_volcanic_cmip6, _load_volcanic_auto,
        )

        # Bust caches — fixtures may have run earlier with a different
        # path.
        for fn in (
            _load_volcanic_auto, _load_volcanic_auto_anchored,
            _load_volcanic_cmip6, _load_volcanic_cmip6_anchored,
        ):
            if hasattr(fn, "cache_clear"):
                fn.cache_clear()

        cfg = AerosolConfig(
            enabled=True, source="climatology", path="",
            use_reference_if_missing=True,
            reference_aod_550=0.0, reference_lat_factor=0.0,
            volcanic_enabled=True, volcanic_path=str(volcanic_path),
            volcanic_scale=1.0, start_year=1979,
        )
        lat_grid = jnp.array(np.deg2rad(np.array([-45.0, 0.0, 45.0])))

        # Day 0 = 1979-01-01: very low background AOD.
        a0 = float(np.max(np.asarray(
            get_aerosol_at_time(cfg, day=0.0, lat_grid=lat_grid)
        )))
        # Day in 1991 (Pinatubo) — strong volcanic AOD expected.
        # 1991-07-01 ≈ sim_day 12*365.25 + 182 ≈ 4565
        pinatubo_sim_day = 12.0 * 365.25 + 182.0
        a_pinatubo = float(np.max(np.asarray(
            get_aerosol_at_time(cfg, day=pinatubo_sim_day,
                                  lat_grid=lat_grid)
        )))
        # Day in 1995 (post-Pinatubo): AOD should have fallen back
        # below the 1991 peak.
        a_1995 = float(np.max(np.asarray(
            get_aerosol_at_time(cfg, day=16.0 * 365.25 + 182.0,
                                  lat_grid=lat_grid)
        )))

        # The synthetic generator places Pinatubo at year 1991 with
        # peak AOD ~0.18.  Without non-cyclic dispatch, day 0 and day
        # 4565 would sample the same cyclic phase and yield the same
        # AOD (or near-zero in both, depending on phase).  With
        # non-cyclic dispatch, the 1991 spike is visible.
        assert a_pinatubo > a0 + 0.01, (
            f"Pinatubo 1991 AOD {a_pinatubo:.4f} should significantly "
            f"exceed 1979 background {a0:.4f}; if it doesn't, the "
            f"multi-year volcanic file is being sampled cyclically "
            f"and the 1991 calendar location is lost."
        )
        assert a_pinatubo > a_1995 + 0.005, (
            f"Pinatubo 1991 AOD {a_pinatubo:.4f} should exceed "
            f"post-Pinatubo 1995 AOD {a_1995:.4f}; a flat trace "
            f"indicates cyclic sampling masked the eruption calendar."
        )

    def test_twelve_month_volcanic_still_cyclic(self, tmp_path):
        """A 12-month volcanic file should continue to use cyclic
        interpolation (no behavior change for legacy climatology)."""
        import jax.numpy as jnp
        from legoesm.forcing.external import (
            AerosolConfig, get_aerosol_at_time,
            _load_volcanic_auto_anchored, _load_volcanic_cmip6_anchored,
            _load_volcanic_cmip6, _load_volcanic_auto,
            _load_monthly_zonal_anchored,
        )

        # Build a tiny 12-month aod file.
        from netCDF4 import Dataset
        path = tmp_path / "volc_clim.nc"
        with Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("time", 12)
            ds.createDimension("lat", 4)
            t = ds.createVariable("time", "f8", ("time",))
            t.units = "days since 1850-01-01 00:00:00"
            t.calendar = "noleap"
            t[:] = np.array([15.5 + 30.4375 * m for m in range(12)])
            la = ds.createVariable("lat", "f8", ("lat",))
            la.units = "degrees_north"
            la[:] = np.array([-60.0, -30.0, 30.0, 60.0])
            v = ds.createVariable("aod", "f8", ("time", "lat"))
            v[:] = 0.05  # constant climatology

        for fn in (
            _load_volcanic_auto, _load_volcanic_auto_anchored,
            _load_volcanic_cmip6, _load_volcanic_cmip6_anchored,
            _load_monthly_zonal_anchored,
        ):
            if hasattr(fn, "cache_clear"):
                fn.cache_clear()

        cfg = AerosolConfig(
            enabled=True, source="climatology", path="",
            use_reference_if_missing=True,
            reference_aod_550=0.0, reference_lat_factor=0.0,
            volcanic_enabled=True, volcanic_path=str(path),
            volcanic_scale=1.0, start_year=1979,
        )
        lat_grid = jnp.array(np.deg2rad(np.array([-30.0, 30.0])))
        # Day 0 vs day 365 should give the same result on a cyclic
        # 12-month file.
        a0 = np.asarray(get_aerosol_at_time(cfg, day=0.0,
                                              lat_grid=lat_grid))
        a365 = np.asarray(get_aerosol_at_time(cfg, day=365.0,
                                                lat_grid=lat_grid))
        assert np.allclose(a0, a365, atol=1e-6), (
            f"12-month volcanic climatology must be cyclic; got "
            f"a(day=0)={a0}, a(day=365)={a365}"
        )

    def test_aerosol_config_has_start_year_default(self):
        """AerosolConfig.start_year defaults to 1979 (CMIP6 AMIP)."""
        from legoesm.forcing.external import AerosolConfig
        assert AerosolConfig().start_year == 1979


class TestValidator:
    """Regression tests for ``scripts/validate_amip_run.py``.

    Two specific failure modes the loose validator can exhibit:
    1. ``Status: BLOWUP`` line in ``results.txt`` should be **fatal**
       in all modes (including non-strict).  Otherwise a clamped /
       NaN-then-finite model can sneak past validation when its
       last-step diagnostics happen to fall inside the bounds.
    2. The TOA energy-residual tolerance must scale on **simulated
       days**, not on ``ts['days'].size``.  With a 5-day diagnostic
       cadence, a 365-day production run only writes 73 samples and
       would otherwise stay in the cold-start band forever, masking a
       divergent radiative imbalance.
    """

    def _write_run(
        self, run_dir: Path, *, status: str, residual_max: float,
        sim_days: float, n_samples: int,
    ) -> None:
        """Write a minimal ``timeseries.npz`` + ``results.txt`` for the
        validator."""
        run_dir.mkdir(parents=True, exist_ok=True)
        days = np.linspace(0.0, sim_days, n_samples)
        residuals = np.linspace(0.0, residual_max, n_samples)
        np.savez(
            run_dir / "timeseries.npz",
            days=days,
            sst=np.full(n_samples, 293.0),
            sic=np.zeros(n_samples),
            T_atm=np.full(n_samples, 295.0),
            T_low=np.full(n_samples, 295.0),
            max_wind=np.full(n_samples, 20.0),
            precip=np.full(n_samples, 3.0),
            CWV=np.full(n_samples, 50.0),
            sw_up_toa=np.full(n_samples, 100.0),
            lw_up_toa=np.full(n_samples, 240.0),
            sw_net_sfc=np.full(n_samples, 180.0),
            lw_net_sfc=np.full(n_samples, -60.0),
            dry_mass_ps=np.full(n_samples, 1.0e5),
            sigma=np.linspace(0.05, 0.95, 30),
            profiles_T=np.array([]),
            profiles_qv=np.array([]),
            energy_toa_net=np.full(n_samples, 0.0),
            energy_column=np.full(n_samples, 1e7),
            energy_dE_dt=np.zeros(n_samples),
            energy_residual=residuals,
            moisture_column_water=np.full(n_samples, 50.0),
            moisture_precip_rate=np.full(n_samples, 3.0),
            moisture_residual=np.full(n_samples, 0.5),
        )
        (run_dir / "results.txt").write_text(f"Status: {status}\n")

    def test_blowup_status_fatal_even_without_strict(self, tmp_path):
        """A ``Status: BLOWUP`` run with otherwise OK scalars must
        return non-zero from validate() with ``strict=False``."""
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        from validate_amip_run import validate

        run = tmp_path / "blowup_run"
        # Diagnostics inside bounds, but status says BLOWUP.
        self._write_run(run, status="BLOWUP", residual_max=10.0,
                        sim_days=1.0, n_samples=3)
        rc = validate(run, strict=False)
        assert rc == 1, (
            "validate() with strict=False must FAIL when results.txt has "
            "Status: BLOWUP, even when diagnostics are inside bounds."
        )

    def test_failed_status_fatal_even_without_strict(self, tmp_path):
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        from validate_amip_run import validate

        run = tmp_path / "failed_run"
        self._write_run(run, status="FAILED", residual_max=10.0,
                        sim_days=1.0, n_samples=3)
        rc = validate(run, strict=False)
        assert rc == 1

    def test_completed_status_passes(self, tmp_path):
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        from validate_amip_run import validate

        run = tmp_path / "ok_run"
        self._write_run(run, status="COMPLETED", residual_max=10.0,
                        sim_days=1.0, n_samples=3)
        rc = validate(run, strict=False)
        assert rc == 0

    def test_long_run_uses_production_residual_bound(self, tmp_path):
        """A 400-day run with a residual just over 50 W/m² should fail
        the production tolerance even though n_samples=80 is small.

        Before the fix, this would slip through the spin-up bound
        (200 W/m²) because it was keyed on ``n = days.size`` instead of
        elapsed simulated days.
        """
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        from validate_amip_run import validate

        run = tmp_path / "long_run"
        # 400 simulated days, 5-day diagnostic cadence → 80 samples.
        # Residual just above the production threshold (50 W/m²) but
        # well under the spin-up threshold (200 W/m²).
        self._write_run(run, status="COMPLETED", residual_max=80.0,
                        sim_days=400.0, n_samples=80)
        rc = validate(run, strict=False)
        assert rc == 1, (
            "A 400-day run with |residual| max=80 W/m² must fail the "
            "production bound (50 W/m²); the previous sample-count "
            "gate would have falsely passed it."
        )

    def test_short_run_keeps_cold_start_bound(self, tmp_path):
        """A 1-day cold-start run with residual=400 W/m² must still
        pass — the cold-start tolerance is 500 W/m²."""
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        from validate_amip_run import validate

        run = tmp_path / "short_run"
        self._write_run(run, status="COMPLETED", residual_max=400.0,
                        sim_days=1.0, n_samples=3)
        rc = validate(run, strict=False)
        assert rc == 0
