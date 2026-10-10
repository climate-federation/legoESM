"""Tests for the synthetic CMIP6 AMIP forcing deck.

Validates that the files produced by ``scripts/data/generate_amip_forcing.py``
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
sys.path.insert(0, str(_REPO_ROOT))           # for `from scripts.data import ...`
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from scripts.data import generate_amip_forcing as gaf  # noqa: E402

from legoesm import constants  # noqa: E402
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
        sst_K = sst_C + constants.T_freeze
        # Area-weighted (cos lat) global mean
        w = np.cos(np.deg2rad(lat))
        w /= w.sum()
        gm = (sst_K.mean(axis=(0, 2)) * w).sum()
        assert 285.0 < gm < 295.0, f"Global SST {gm:.2f} K out of observed band"

    def test_no_freezing_violations(self, deck):
        import netCDF4
        with netCDF4.Dataset(deck["sst"]) as ds:
            sst_K = ds["sst"][:].astype(np.float64) + constants.T_freeze
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
            S_0=constants.S_0, source="file", path=str(deck["solar"]),
            tsi_var="TSI",
            start_year=deck["start_year"],
        )
        tsi = get_tsi_at_time(cfg, day=0.0)
        assert 1358.0 < float(tsi) < 1364.0, f"TSI {tsi} out of band"

    def test_spectral_expansion(self, deck):
        cfg = SolarConfig(
            S_0=constants.S_0, source="spectral_file", path=str(deck["solar"]),
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
    """Regression tests for ``run_amip_smoke_deck.py:_check_forcing_files``.

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
        from scripts.run.run_amip_smoke_deck import _check_forcing_files
        files = _check_forcing_files(tmp_path, 1979, 1980)
        assert "_missing" in files
        # All six channels should be reported missing
        missing = files["_missing"].split(",")
        assert set(missing) == {"sst", "ghg", "ozone", "solar",
                                  "aerosol", "volcanic"}

    def test_interannual_ozone_accepted(self, tmp_path):
        """Interannual ozone file alone is sufficient; clim is optional."""
        from scripts.run.run_amip_smoke_deck import _check_forcing_files
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
        from scripts.run.run_amip_smoke_deck import _check_forcing_files
        self._make_deck(tmp_path, with_interannual_o3=False,
                        with_clim_o3=True)
        files = _check_forcing_files(tmp_path, 1979, 1980)
        assert "_missing" not in files
        assert files["ozone"].name == "ozone_amip_clim.nc"

    def test_interannual_preferred_over_clim(self, tmp_path):
        """When both files exist, the interannual one wins (it's what
        real CMIP6 ozone is and exercises the non-cyclic loader)."""
        from scripts.run.run_amip_smoke_deck import _check_forcing_files
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


class TestSolarBandExpansion:
    """Regression tests for the band → g-point expansion fix
    (P2 codex iter-5 review).

    Previously, ``_expand_bands_to_gpoints`` *copied* each band's
    fraction to every g-point in the band, then a downstream
    sum-to-1 normalize ran.  The result inflated bands with more
    g-points and shrunk bands with fewer, distorting the per-band
    proportions.  The fix divides each band's fraction by the
    number of g-points in that band so the sum-over-band-g-points
    equals the band fraction, and a sum-to-1 normalize preserves the
    per-band integrals.
    """

    def _make_fake_rrtmg_lookup(self, path: Path,
                                  *, bnd_limits: list[tuple[int, int]]) -> None:
        """Create a synthetic netCDF that mimics the bnd_limits_gpt
        layout of the real RRTMG-SW lookup table."""
        from netCDF4 import Dataset
        n_bands = len(bnd_limits)
        with Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("band", n_bands)
            ds.createDimension("two", 2)
            v = ds.createVariable("bnd_limits_gpt", "i4", ("band", "two"))
            for i, (lo, hi) in enumerate(bnd_limits):
                v[i, 0] = lo
                v[i, 1] = hi

    def test_band_integral_preserved_uniform_bands(self, tmp_path):
        """Equal-width bands: each band 4 g-points; expansion +
        sum-to-1 normalize should give a uniform per-g-point profile
        (1/n_gpt) for a uniform per-band input (1/n_bands)."""
        from legoesm.forcing.external import _expand_bands_to_gpoints
        path = tmp_path / "rrtmg_uniform.nc"
        n_bands = 4
        n_gpt_per_band = 4
        bnd_limits = [(i * n_gpt_per_band + 1,
                        (i + 1) * n_gpt_per_band)
                       for i in range(n_bands)]
        self._make_fake_rrtmg_lookup(path, bnd_limits=bnd_limits)

        # Uniform per-band input
        spec = np.full(n_bands, 1.0 / n_bands)
        out = _expand_bands_to_gpoints(spec, str(path))
        assert out.shape == (n_bands * n_gpt_per_band,)
        # Each g-point should have value 1 / (n_bands * n_gpt_per_band)
        expected = np.full(n_bands * n_gpt_per_band,
                             1.0 / (n_bands * n_gpt_per_band))
        assert np.allclose(out, expected, rtol=1e-12), (
            f"Uniform expansion broken: got {out}, expected {expected}"
        )
        # Sum-over-g-points-in-band-i must equal spec[i].
        for i, (lo, hi) in enumerate(bnd_limits):
            band_sum = np.sum(out[lo - 1:hi])
            assert np.isclose(band_sum, spec[i]), (
                f"Band {i} integral broken: got {band_sum}, "
                f"expected {spec[i]}"
            )

    def test_band_integral_preserved_unequal_bands(self, tmp_path):
        """Unequal-width bands (the realistic CMIP6 case): expansion
        must preserve each band's integral so the per-band SW power
        is not silently rescaled."""
        from legoesm.forcing.external import _expand_bands_to_gpoints
        # Two bands: band 0 has 8 g-points, band 1 has 2 g-points.
        # With the *old* (broken) expansion, the band-0 integral
        # would be 8x its band fraction (due to copy-to-all-gpoints).
        bnd_limits = [(1, 8), (9, 10)]
        path = tmp_path / "rrtmg_unequal.nc"
        self._make_fake_rrtmg_lookup(path, bnd_limits=bnd_limits)

        spec = np.array([0.7, 0.3])  # CMIP6-like band fractions
        out = _expand_bands_to_gpoints(spec, str(path))
        # Sum over g-points in each band must equal the band fraction
        assert np.isclose(np.sum(out[0:8]), 0.7), (
            f"Band 0 integral broken: got {np.sum(out[0:8]):.4f}, "
            f"expected 0.7"
        )
        assert np.isclose(np.sum(out[8:10]), 0.3), (
            f"Band 1 integral broken: got {np.sum(out[8:10]):.4f}, "
            f"expected 0.3"
        )
        # Total sum must equal sum-of-band-fractions == 1.0
        assert np.isclose(np.sum(out), 1.0)

    def _make_fake_rrtmg_lookup_with_source(
        self, path: Path, *, bnd_limits: list[tuple[int, int]],
        solar_source: list[float],
    ) -> None:
        """Synthetic RRTMG-SW table carrying ``solar_source_quiet`` (the
        per-g-point solar source) in addition to ``bnd_limits_gpt``."""
        from netCDF4 import Dataset
        n_bands = len(bnd_limits)
        n_gpt = max(hi for _, hi in bnd_limits)
        with Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("band", n_bands)
            ds.createDimension("two", 2)
            ds.createDimension("gpt", n_gpt)
            v = ds.createVariable("bnd_limits_gpt", "i4", ("band", "two"))
            for i, (lo, hi) in enumerate(bnd_limits):
                v[i, 0] = lo
                v[i, 1] = hi
            s = ds.createVariable("solar_source_quiet", "f8", ("gpt",))
            s[:] = np.asarray(solar_source, dtype=float)

    def test_within_band_split_follows_solar_source(self, tmp_path):
        """With ``solar_source_quiet`` present, a band's fraction is
        distributed over its g-points PROPORTIONAL to the per-g-point solar
        source (not split equally) — while the band integral is preserved.

        Pins the within-band source-shape fix: RRTMGP g-points within a band
        carry very unequal solar weight, so a uniform split dumps flux into
        strongly-absorbing g-points and ~doubles clear-sky SW absorption.
        """
        from legoesm.forcing.external import _expand_bands_to_gpoints
        # One 4-g-point band; source weights 0.4/0.3/0.2/0.1.
        bnd_limits = [(1, 4)]
        solar_source = [4.0, 3.0, 2.0, 1.0]
        path = tmp_path / "rrtmg_source.nc"
        self._make_fake_rrtmg_lookup_with_source(
            path, bnd_limits=bnd_limits, solar_source=solar_source)

        spec = np.array([0.8])  # single-band fraction
        out = _expand_bands_to_gpoints(spec, str(path))
        assert out.shape == (4,)
        # Within-band split ∝ source shape (NOT uniform 0.2 each).
        expected = 0.8 * np.array([0.4, 0.3, 0.2, 0.1])
        assert np.allclose(out, expected, rtol=1e-12), (
            f"Within-band split not source-shaped: got {out}, "
            f"expected {expected}")
        assert not np.allclose(out, np.full(4, 0.8 / 4)), (
            "Split is uniform — source shape ignored")
        # Band integral preserved.
        assert np.isclose(np.sum(out), 0.8)

    def test_zero_source_band_falls_back_to_uniform(self, tmp_path):
        """A band whose solar source sums to zero falls back to the uniform
        split (no divide-by-zero), still preserving the band integral."""
        from legoesm.forcing.external import _expand_bands_to_gpoints
        bnd_limits = [(1, 4)]
        path = tmp_path / "rrtmg_zerosrc.nc"
        self._make_fake_rrtmg_lookup_with_source(
            path, bnd_limits=bnd_limits, solar_source=[0.0, 0.0, 0.0, 0.0])
        spec = np.array([0.5])
        out = _expand_bands_to_gpoints(spec, str(path))
        assert np.allclose(out, np.full(4, 0.5 / 4), rtol=1e-12)
        assert np.isclose(np.sum(out), 0.5)


class TestCMIPBandOrderRemap:
    """Issue #322: the MPI-M CMIP6 14-band ``SSI_frac`` file is in
    RRTMG-SW band order, which differs from the RRTMGP-SW g-point-table
    order by a one-band cyclic rotation (the 820-2680 cm^-1 overlap band
    is last in the CMIP file but first in RRTMGP).  Expanding by array
    index without re-ordering shifts the whole solar spectrum one band
    toward longer wavelengths, dumping UV flux into the near-IR
    water-vapour band (~2x clear-sky SW absorption).
    """

    def _rrtmgp_bands(self):
        import xarray as xr
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
            DEFAULT_SW_GAS,
        )
        ds = (xr.open_dataset(DEFAULT_SW_GAS)
              if not DEFAULT_SW_GAS.endswith(".zarr")
              else xr.open_zarr(DEFAULT_SW_GAS))
        wn = ds["bnd_limits_wavenumber"].values          # (14, 2) cm^-1
        gpt = ds["bnd_limits_gpt"].values.astype(int)    # (14, 2) 1-indexed
        ds.close()
        return wn, gpt, DEFAULT_SW_GAS

    def test_roll_aligns_every_band_by_wavelength(self):
        """Each CMIP band's flux must be re-ordered into the RRTMGP band
        covering the SAME wavenumber interval."""
        from legoesm.forcing.external import _cmip_sw_band_order_to_rrtmgp
        wn_rrtmgp, _, _ = self._rrtmgp_bands()
        n = wn_rrtmgp.shape[0]
        # CMIP order = RRTMGP order rotated the other way (820 band last).
        wn_cmip = np.roll(wn_rrtmgp, -1, axis=0)
        for k in range(n):
            e_k = np.zeros(n)
            e_k[k] = 1.0
            out = _cmip_sw_band_order_to_rrtmgp(e_k)
            assert np.isclose(out.sum(), 1.0)  # permutation: no flux lost
            j = int(np.argmax(out))
            assert np.isclose(out[j], 1.0)
            assert np.allclose(wn_rrtmgp[j], wn_cmip[k]), (
                f"CMIP band {k} ({wn_cmip[k]} cm^-1) re-ordered to RRTMGP "
                f"band {j} ({wn_rrtmgp[j]} cm^-1) — wavelength mismatch"
            )

    def test_uv_flux_lands_on_uv_gpoints(self):
        """All-UV input must end on the RRTMGP UV g-points (highest
        wavenumber band), not the near-IR."""
        from legoesm.forcing.external import (
            _cmip_sw_band_order_to_rrtmgp, _expand_bands_to_gpoints,
        )
        wn_rrtmgp, gpt, path = self._rrtmgp_bands()
        n = wn_rrtmgp.shape[0]
        wn_cmip = np.roll(wn_rrtmgp, -1, axis=0)
        uv_rrtmgp = int(np.argmax(wn_rrtmgp[:, 1]))  # highest-wavenumber band
        uv_cmip = int(np.argmax(wn_cmip[:, 1]))
        spec = np.zeros(n)
        spec[uv_cmip] = 1.0
        out = _expand_bands_to_gpoints(
            _cmip_sw_band_order_to_rrtmgp(spec), str(path),
        )
        lo, hi = gpt[uv_rrtmgp]  # 1-indexed g-point range of the UV band
        assert np.isclose(out[lo - 1:hi].sum(), 1.0), (
            "UV flux did not land on the UV g-points"
        )
        mask = np.ones(out.shape[0], dtype=bool)
        mask[lo - 1:hi] = False
        assert np.allclose(out[mask], 0.0), "UV flux leaked outside UV band"

    def test_control_without_remap_uv_misplaced(self):
        """Control documenting the #322 bug: the bare index-map (no
        re-order) puts UV flux in a lower-wavenumber RRTMGP band."""
        from legoesm.forcing.external import _expand_bands_to_gpoints
        wn_rrtmgp, gpt, path = self._rrtmgp_bands()
        n = wn_rrtmgp.shape[0]
        wn_cmip = np.roll(wn_rrtmgp, -1, axis=0)
        uv_cmip = int(np.argmax(wn_cmip[:, 1]))
        spec = np.zeros(n)
        spec[uv_cmip] = 1.0
        out = _expand_bands_to_gpoints(spec, str(path))  # NO re-order
        lo, hi = gpt[uv_cmip]
        # Index-map keeps the flux at band `uv_cmip`, whose wavenumber
        # range is NOT the UV (max-wavenumber) band — the bug.
        assert wn_rrtmgp[uv_cmip, 1] < wn_rrtmgp[:, 1].max()
        assert np.isclose(out[lo - 1:hi].sum(), 1.0)

    def test_end_to_end_spectral_file_preserves_uv(self, tmp_path):
        """Through the full ``get_solar_forcing_at_time`` spectral_file
        path: a CMIP-order file with all flux in the UV band yields a
        per-g-point profile concentrated on the RRTMGP UV g-points."""
        import netCDF4
        from legoesm.forcing.external import (
            SolarConfig, get_solar_forcing_at_time,
        )
        wn_rrtmgp, gpt, _ = self._rrtmgp_bands()
        n = wn_rrtmgp.shape[0]
        wn_cmip = np.roll(wn_rrtmgp, -1, axis=0)
        uv_rrtmgp = int(np.argmax(wn_rrtmgp[:, 1]))
        uv_cmip = int(np.argmax(wn_cmip[:, 1]))

        path = tmp_path / "solar_uv.nc"
        with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("time", 2)
            ds.createDimension("numwl", n)
            t = ds.createVariable("time", "f8", ("time",))
            t[:] = [0.0, 365.0]
            t.units = "days since 1850-01-01 00:00:00"
            t.calendar = "noleap"
            tsi = ds.createVariable("TSI", "f8", ("time",))
            tsi[:] = [constants.S_0, constants.S_0]
            ssi = ds.createVariable("SSI_frac", "f8", ("time", "numwl"))
            band = np.zeros(n)
            band[uv_cmip] = 1.0
            ssi[:] = np.broadcast_to(band, (2, n))

        cfg = SolarConfig(
            S_0=constants.S_0, source="spectral_file", path=str(path),
            tsi_var="TSI", spectral_var="SSI_frac",
            spectral_band_order="rrtmg_sw",  # MPI-M CMIP order -> reorder
            normalize_spectral=False,
        )
        out = get_solar_forcing_at_time(cfg, day=0.0)
        gpt_frac = np.asarray(out["solar_fraction_by_gpt"])
        assert gpt_frac.shape == (112,)
        lo, hi = gpt[uv_rrtmgp]
        assert np.isclose(gpt_frac[lo - 1:hi].sum(), 1.0), (
            "spectral_file path did not place UV flux on UV g-points"
        )

    def test_explicit_as_is_never_rotates(self, tmp_path):
        """Codex #322 review: ``spectral_band_order="as_is"`` is the
        explicit escape hatch — even for an ``SSI_frac``-named file it
        expands by index with NO rotation, so a file already in RRTMGP
        order is never corrupted (overrides ``auto`` detection)."""
        import netCDF4
        from legoesm.forcing.external import (
            SolarConfig, get_solar_forcing_at_time,
        )
        wn_rrtmgp, gpt, _ = self._rrtmgp_bands()
        n = wn_rrtmgp.shape[0]
        # File already in RRTMGP order: put unit flux directly in the UV
        # (max-wavenumber) RRTMGP band.
        uv_rrtmgp = int(np.argmax(wn_rrtmgp[:, 1]))
        path = tmp_path / "solar_rrtmgp_order.nc"
        with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("time", 2)
            ds.createDimension("numwl", n)
            t = ds.createVariable("time", "f8", ("time",))
            t[:] = [0.0, 365.0]
            t.units = "days since 1850-01-01 00:00:00"
            t.calendar = "noleap"
            tsi = ds.createVariable("TSI", "f8", ("time",))
            tsi[:] = [constants.S_0, constants.S_0]
            ssi = ds.createVariable("SSI_frac", "f8", ("time", "numwl"))
            band = np.zeros(n)
            band[uv_rrtmgp] = 1.0
            ssi[:] = np.broadcast_to(band, (2, n))
        cfg = SolarConfig(
            S_0=constants.S_0, source="spectral_file", path=str(path),
            tsi_var="TSI", spectral_var="SSI_frac",
            spectral_band_order="as_is",  # explicit: never rotate
            normalize_spectral=False,
        )
        out = get_solar_forcing_at_time(cfg, day=0.0)
        gpt_frac = np.asarray(out["solar_fraction_by_gpt"])
        lo, hi = gpt[uv_rrtmgp]
        # No rotation: the already-RRTMGP UV band stays on the UV g-points.
        assert np.isclose(gpt_frac[lo - 1:hi].sum(), 1.0), (
            "explicit as_is must NOT reorder an already-RRTMGP-order file"
        )

    def test_auto_leaves_generic_14band_untouched(self, tmp_path):
        """A generic 14-band file with no MPI-M signature (variable not
        ``SSI_frac``, filename not ``swflux_14band``) must NOT be rotated
        under the default ``auto`` — only files with the CMIP6 signature
        are auto-remapped (codex #322 round-3 concern)."""
        import netCDF4
        from legoesm.forcing.external import (
            SolarConfig, get_solar_forcing_at_time,
        )
        wn_rrtmgp, gpt, _ = self._rrtmgp_bands()
        n = wn_rrtmgp.shape[0]
        uv_rrtmgp = int(np.argmax(wn_rrtmgp[:, 1]))
        path = tmp_path / "generic_spectrum.nc"
        with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("time", 2)
            ds.createDimension("numwl", n)
            t = ds.createVariable("time", "f8", ("time",))
            t[:] = [0.0, 365.0]
            t.units = "days since 1850-01-01 00:00:00"
            t.calendar = "noleap"
            tsi = ds.createVariable("TSI", "f8", ("time",))
            tsi[:] = [constants.S_0, constants.S_0]
            spv = ds.createVariable("spec14", "f8", ("time", "numwl"))
            band = np.zeros(n)
            band[uv_rrtmgp] = 1.0
            spv[:] = np.broadcast_to(band, (2, n))
        cfg = SolarConfig(
            S_0=constants.S_0, source="spectral_file", path=str(path),
            tsi_var="TSI", spectral_var="spec14",  # no MPI-M signature
            normalize_spectral=False,              # default order "auto"
        )
        out = get_solar_forcing_at_time(cfg, day=0.0)
        gpt_frac = np.asarray(out["solar_fraction_by_gpt"])
        lo, hi = gpt[uv_rrtmgp]
        assert np.isclose(gpt_frac[lo - 1:hi].sum(), 1.0), (
            "auto must NOT rotate a generic (non-SSI_frac) 14-band file"
        )

    def test_invalid_band_order_raises(self, tmp_path):
        """Unknown ``spectral_band_order`` must raise (no silent default)."""
        import netCDF4
        import pytest
        from legoesm.forcing.external import (
            SolarConfig, get_solar_forcing_at_time,
        )
        path = tmp_path / "solar_bad.nc"
        with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("time", 2)
            ds.createDimension("numwl", 14)
            t = ds.createVariable("time", "f8", ("time",))
            t[:] = [0.0, 365.0]
            t.units = "days since 1850-01-01 00:00:00"
            t.calendar = "noleap"
            tsi = ds.createVariable("TSI", "f8", ("time",))
            tsi[:] = [constants.S_0, constants.S_0]
            ssi = ds.createVariable("SSI_frac", "f8", ("time", "numwl"))
            ssi[:] = np.full((2, 14), 1.0 / 14.0)
        cfg = SolarConfig(
            S_0=constants.S_0, source="spectral_file", path=str(path),
            tsi_var="TSI", spectral_var="SSI_frac",
            spectral_band_order="bogus",
        )
        with pytest.raises(ValueError, match="spectral_band_order"):
            get_solar_forcing_at_time(cfg, day=0.0)

    def test_cli_threads_band_order_into_experiment_config(self):
        """Codex #322 follow-up: the run_amip CLI must thread
        --solar-spectral-band-order all the way into ExperimentConfig
        (which ModelDriver passes to SolarConfig), defaulting to
        'auto'."""
        import sys
        from pathlib import Path
        repo = Path(__file__).resolve().parents[2]
        if str(repo / "scripts") not in sys.path:
            sys.path.insert(0, str(repo / "scripts"))
        from scripts.run import run_amip
        parser = run_amip.build_arg_parser()
        cfg_default = run_amip.build_config_from_args(parser.parse_args([]))
        assert cfg_default.solar_spectral_band_order == "auto"
        cfg_rrtmg = run_amip.build_config_from_args(
            parser.parse_args(["--solar-spectral-band-order", "rrtmg_sw"]),
        )
        assert cfg_rrtmg.solar_spectral_band_order == "rrtmg_sw"

    def test_cmip6_deck_selects_rrtmg_sw(self):
        """The production CMIP6 deck launcher consumes the MPI-M
        SSI_frac file, so it must forward --solar-spectral-band-order
        rrtmg_sw to run_amip (issue #322)."""
        from pathlib import Path
        deck = (Path(__file__).resolve().parents[2]
                / "scripts" / "run" / "run_amip_smoke_deck.py").read_text()
        assert '"--solar-spectral-band-order", "rrtmg_sw"' in deck, (
            "CMIP6 deck must forward --solar-spectral-band-order rrtmg_sw "
            "for the MPI-M SSI_frac file"
        )

    def test_legacy_amip_config_dict_defaults_to_auto(self):
        """Codex #322 follow-up: a legacy AMIP config dict that predates
        the ``solar_spectral_band_order`` field backfills to ``auto``,
        which only rotates files carrying the MPI-M CMIP6 signature and
        leaves generic files untouched — so legacy replay is never
        silently corrupted."""
        from legoesm.forcing.amip_config import config_from_dict
        from legoesm.driver.config import ExperimentConfig
        legacy = {
            "solar_source": "spectral_file",
            "solar_file": "old.nc",
            "solar_spectral_var": "SSI_frac",
            # no solar_spectral_band_order key (predates the field)
        }
        amip_cfg = config_from_dict(legacy)
        assert amip_cfg.solar_spectral_band_order == "auto"
        exp = ExperimentConfig.from_amip_config(amip_cfg)
        assert exp.solar_spectral_band_order == "auto"

    def test_auto_detects_ssi_frac_and_rotates(self, tmp_path, caplog):
        """Codex #322 round-5: a canonical MPI-M ``SSI_frac`` 14-band
        file consumed with the default ``auto`` ordering must NOT run
        with the known-wrong order — it is auto-detected and rotated to
        RRTMGP order (UV flux lands on the UV g-points), with an INFO log
        recording the auto-selection.  This is the fail-safe behaviour:
        the canonical CMIP6 file can never silently use the wrong band
        order on any entry point."""
        import logging
        import netCDF4
        from legoesm.forcing.external import (
            SolarConfig, get_solar_forcing_at_time,
        )
        wn_rrtmgp, gpt, _ = self._rrtmgp_bands()
        n = wn_rrtmgp.shape[0]
        wn_cmip = np.roll(wn_rrtmgp, -1, axis=0)
        uv_rrtmgp = int(np.argmax(wn_rrtmgp[:, 1]))
        uv_cmip = int(np.argmax(wn_cmip[:, 1]))
        path = tmp_path / "swflux_14band_cmip6.nc"
        with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("time", 2)
            ds.createDimension("numwl", n)
            t = ds.createVariable("time", "f8", ("time",))
            t[:] = [0.0, 365.0]
            t.units = "days since 1850-01-01 00:00:00"
            t.calendar = "noleap"
            tsi = ds.createVariable("TSI", "f8", ("time",))
            tsi[:] = [constants.S_0, constants.S_0]
            ssi = ds.createVariable("SSI_frac", "f8", ("time", "numwl"))
            band = np.zeros(n)
            band[uv_cmip] = 1.0   # UV in CMIP order
            ssi[:] = np.broadcast_to(band, (2, n))
        cfg = SolarConfig(
            S_0=constants.S_0, source="spectral_file", path=str(path),
            tsi_var="TSI", spectral_var="SSI_frac",  # default order "auto"
            normalize_spectral=False,
        )
        with caplog.at_level(logging.INFO, logger="legoesm.forcing.external"):
            out = get_solar_forcing_at_time(cfg, day=0.0)
        gpt_frac = np.asarray(out["solar_fraction_by_gpt"])
        lo, hi = gpt[uv_rrtmgp]
        # Auto-rotation places the UV flux on the RRTMGP UV g-points.
        assert np.isclose(gpt_frac[lo - 1:hi].sum(), 1.0), (
            "auto must rotate the detected SSI_frac file to RRTMGP order"
        )
        assert any(
            "rrtmg_sw" in r.message or "#322" in r.message
            for r in caplog.records
        ), "expected an INFO log recording the auto-selected band order"


class TestFirstDateCalendarPreservation:
    """Regression test for the iter-6 calendar-preservation fix in
    ``_extract_first_date``.

    A CMIP6 file with ``time.units="months since 1850-01-01"`` and
    ``calendar="noleap"`` previously produced a ``DatetimeGregorian``
    anchor.  Combined with the iter-5 calendar-aware
    ``_simday_to_file_day`` (which picks the constructor based on
    ``first_date.calendar``), a Gregorian first_date would silently
    use Gregorian arithmetic against a noleap file, producing a ~31
    day offset by 1979 (and much larger for ``360_day``).
    """

    def test_first_date_noleap_calendar_preserved(self):
        import cftime
        from legoesm.forcing.external import _extract_first_date
        # Numeric mid_days = 0.5 month past 1850-01-01 (mid-January 1850).
        time_values = np.array([0.5, 1.5, 2.5])
        first = _extract_first_date(
            time_values, "months since 1850-01-01", "noleap",
        )
        assert isinstance(first, cftime.DatetimeNoLeap), (
            f"Noleap calendar lost: got {type(first).__name__}"
        )
        # 0.5 month past Jan 1 = mid-January 1850.
        assert first.year == 1850 and first.month == 1

    def test_first_date_360day_calendar_preserved(self):
        import cftime
        from legoesm.forcing.external import _extract_first_date
        time_values = np.array([15.0, 45.0, 75.0])
        first = _extract_first_date(
            time_values, "months since 1850-01-01", "360_day",
        )
        assert isinstance(first, cftime.Datetime360Day)

    def test_first_date_gregorian_calendar_preserved(self):
        import cftime
        from legoesm.forcing.external import _extract_first_date
        time_values = np.array([15.5, 45.5, 75.5])
        first = _extract_first_date(
            time_values, "months since 1850-01-01", "gregorian",
        )
        assert isinstance(first, cftime.DatetimeGregorian)

    def test_extract_to_simday_noleap_end_to_end(self):
        """Combined extract + sim_day pipeline: noleap file with mid-Jan
        1850 anchor and 1979 sim epoch must give 129 * 365 - 15 days
        (Jan 1 1979 is 47070 days past mid-Jan 1850 in a noleap
        calendar)."""
        from legoesm.forcing.external import (
            _extract_first_date, _simday_to_file_day,
        )
        # 0.5 month past Jan 1 1850 (mid-January 1850 noleap).
        time_values = np.array([0.5, 1.5])
        first = _extract_first_date(
            time_values, "months since 1850-01-01", "noleap",
        )
        file_day = _simday_to_file_day(0.0, 1979, first)
        # 1850-Jan-1 → 1979-Jan-1 in noleap = 129 * 365 = 47085 days.
        # First record is mid-Jan 1850 (≈ 15 days past Jan 1).
        # So Jan 1 1979 lands at file_day ≈ 47085 - 15 = 47070.
        # Allow ±2 days for noleap-month roundoff (365.25/12 vs 30).
        expected = 129 * 365 - 15
        assert abs(file_day - expected) < 3.0, (
            f"NoLeap end-to-end: expected ~{expected}, got {file_day}"
        )


class TestSimDayCalendar:
    """Regression test for the calendar-aware sim-epoch in
    ``_simday_to_file_day`` (P2 codex iter-5 review).

    When a real CMIP6 file uses ``calendar='noleap'`` (most
    input4MIPs ozone, aerosol, GHG annual files do), the sim epoch
    must be constructed in the *same* calendar so the day-difference
    counts no leap days.  The previous fallback through
    ``numpy.datetime64`` is Gregorian-only and would shift the file
    date by ~30 days for a 1979 sim_day with 1850 noleap anchor.
    """

    def test_noleap_anchor_does_not_count_gregorian_leaps(self):
        import cftime
        from legoesm.forcing.external import _simday_to_file_day
        # Anchor: 1850-01-01 in NOLEAP calendar.  Simulation epoch:
        # 1979-01-01.  Expected delta = (1979-1850) * 365 = 47085 days.
        # Gregorian arithmetic would introduce ~32 leap days.
        anchor = cftime.DatetimeNoLeap(1850, 1, 1)
        file_day = _simday_to_file_day(0.0, 1979, anchor)
        expected = (1979 - 1850) * 365
        assert int(round(file_day)) == expected, (
            f"NOLEAP anchor: expected file_day={expected}, "
            f"got {file_day:.1f} — mismatch implies the sim epoch "
            f"was constructed in a different calendar."
        )

    def test_360day_anchor_uses_360day_year(self):
        import cftime
        from legoesm.forcing.external import _simday_to_file_day
        anchor = cftime.Datetime360Day(1850, 1, 1)
        file_day = _simday_to_file_day(0.0, 1979, anchor)
        expected = (1979 - 1850) * 360
        assert int(round(file_day)) == expected, (
            f"360_day anchor: expected file_day={expected}, "
            f"got {file_day:.1f}"
        )

    def test_gregorian_anchor_unchanged(self):
        """Gregorian/standard anchor — original behavior preserved.

        1850 → 1979 spans 31 leap years in the Gregorian calendar:
        every fourth year from 1852 to 1976 inclusive is 32 years,
        minus 1900 (a centurial year not divisible by 400) gives 31.
        Expected delta = 129 * 365 + 31 = 47116 days.
        """
        import cftime
        from legoesm.forcing.external import _simday_to_file_day
        anchor = cftime.DatetimeGregorian(1850, 1, 1)
        file_day = _simday_to_file_day(0.0, 1979, anchor)
        assert int(round(file_day)) == 47116


class TestNoAerosolNoVolcanicFlags:
    """Regression tests for the iter-4 codex review's three deck-driver
    fixes around disabled aerosol/volcanic channels:

    1. ``_check_forcing_files`` skips aerosol/volcanic from the
       missing-file check when ``require_aerosol``/``require_volcanic``
       is False.  This lets users run ``--no-aerosol --no-volcanic``
       on a partial deck without spurious "missing files" errors.
    2. ``--no-aerosol`` (without ``--no-volcanic``) implies the
       volcanic channel is also disabled — ``ModelDriver`` gates
       volcanic on ``aerosol_forcing == "external"``, so passing
       ``--volcanic-aerosol-file`` without ``--aerosol-forcing
       external`` is silently ignored.  The deck driver now prints a
       NOTE at startup and does not pass the volcanic flag.
    3. The activity report now uses the same ``aerosol_active`` /
       ``volcanic_active`` state, so the report cannot lie when the
       channels are disabled by either flag.
    """

    def _make_partial_deck(self, out: Path, *, sy: int = 1979,
                            ey: int = 1980,
                            include_aerosol: bool = False,
                            include_volcanic: bool = False) -> None:
        out.mkdir(parents=True, exist_ok=True)
        gaf.make_sst_sic(out / f"sst_sic_amip_{sy}-{ey}.nc",
                          sy, ey, nlat=37, nlon=72)
        gaf.make_ghg_annual(out / f"ghg_amip_{sy}-{ey}.nc", sy, ey)
        gaf.make_ozone_clim(out / "ozone_amip_clim.nc", nlat=18, nlev=20)
        gaf.make_solar(out / f"solar_amip_{sy}-{ey}.nc", sy, ey)
        if include_aerosol:
            gaf.make_aerosol_clim(out / "aerosol_amip_clim.nc", nlat=36)
        if include_volcanic:
            gaf.make_volcanic(out / f"volcanic_amip_{sy}-{ey}.nc",
                               sy, ey, nlat=18)

    def test_check_files_skips_aerosol_when_disabled(self, tmp_path):
        from scripts.run.run_amip_smoke_deck import _check_forcing_files
        self._make_partial_deck(tmp_path,
                                 include_aerosol=False,
                                 include_volcanic=False)
        # With aerosol/volcanic NOT required, the partial deck must
        # be considered complete.
        files = _check_forcing_files(
            tmp_path, 1979, 1980,
            require_aerosol=False, require_volcanic=False,
        )
        assert "_missing" not in files, (
            f"Partial deck should be complete with aerosol/volcanic "
            f"disabled, got missing: {files.get('_missing')}"
        )
        # Default require=True still reports missing.
        files_required = _check_forcing_files(tmp_path, 1979, 1980)
        assert "_missing" in files_required
        assert "aerosol" in files_required["_missing"]
        assert "volcanic" in files_required["_missing"]

    def test_check_files_skips_volcanic_only(self, tmp_path):
        from scripts.run.run_amip_smoke_deck import _check_forcing_files
        self._make_partial_deck(tmp_path,
                                 include_aerosol=True,
                                 include_volcanic=False)
        files = _check_forcing_files(
            tmp_path, 1979, 1980,
            require_aerosol=True, require_volcanic=False,
        )
        assert "_missing" not in files

    def test_no_aerosol_implies_no_volcanic_in_deck_driver(self, tmp_path):
        """End-to-end: --no-aerosol disables volcanic too (with NOTE)."""
        import subprocess
        self._make_partial_deck(tmp_path,
                                 include_aerosol=True,
                                 include_volcanic=True)
        deck_script = _REPO_ROOT / "scripts" / "run" / "run_amip_smoke_deck.py"
        cmd = [
            sys.executable, str(deck_script),
            "--forcing-dir", str(tmp_path),
            "--start-year", "1979", "--end-year", "1980",
            "--grid-type", "cubed_sphere", "--discretization", "centered",
            "--radiation", "rrtmg", "--resolution", "8",
            # Deck default IC is now era5 (needs --ic-path); this test
            # exercises the aerosol/volcanic flag coupling, so use the
            # cheap uniform IC to keep the dry-run self-contained.
            "--ic", "default",
            "--days", "1", "--no-aerosol",  # volcanic *not* disabled
            "--dry-run",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"dry-run failed: {r.stderr}"
        out = r.stdout
        # NOTE explaining the implicit no-volcanic
        assert "NOTE" in out and "aerosol" in out and "volcanic" in out, (
            f"Expected NOTE about --no-aerosol implying no-volcanic; "
            f"got:\n{out}"
        )
        # Activity report must not show Aerosol / Volcanic as ACTIVE.
        for label in ("Tropospheric aerosol", "Volcanic stratospheric"):
            line = next((ln for ln in out.splitlines() if label in ln), None)
            assert line is None or "ACTIVE" not in line, (
                f"Activity report shows {label} as ACTIVE under "
                f"--no-aerosol; got: {line!r}"
            )
        # Command line must NOT include --volcanic-aerosol-file
        # (passing it without --aerosol-forcing external would silently
        # drop it; the iter-4 fix prevents the lie).
        cmd_block = out.split("[deck] Command:", 1)[1].split(
            "[deck] Forcing-channel activity"
        )[0]
        assert "--volcanic-aerosol-file" not in cmd_block, (
            "Volcanic file should not be passed when --no-aerosol "
            "implicitly disables volcanic; cmd block:\n"
            f"{cmd_block}"
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


class TestSpectralMpasForcingActive:
    """Spectral (gaussian) and MPAS (voronoi) paths now run the unified
    physics pipeline and CONSUME external CMIP6 forcing (PR #395), so the
    deck no longer prints the old 'silent-drop' warning and the activity
    report shows GHG/ozone/aerosol/volcanic as ACTIVE.  Only the solar
    FILE (TSI + 14-band) remains inert on these paths (constant S_0).

    The deck default IC is --ic era5; these dry-run tests pass --ic
    default so no ERA5 zarr is required.
    """

    def _make_deck(self, tmp_path):
        sy, ey = 1979, 1980
        gaf.make_sst_sic(tmp_path / f"sst_sic_amip_{sy}-{ey}.nc",
                          sy, ey, nlat=37, nlon=72)
        gaf.make_ghg_annual(tmp_path / f"ghg_amip_{sy}-{ey}.nc", sy, ey)
        gaf.make_ozone_clim(tmp_path / "ozone_amip_clim.nc",
                             nlat=18, nlev=20)
        gaf.make_solar(tmp_path / f"solar_amip_{sy}-{ey}.nc", sy, ey)
        gaf.make_aerosol_clim(tmp_path / "aerosol_amip_clim.nc", nlat=36)
        gaf.make_volcanic(tmp_path / f"volcanic_amip_{sy}-{ey}.nc",
                          sy, ey, nlat=18)
        return sy, ey

    def _run(self, tmp_path, grid_type, disc, res):
        import subprocess
        sy, ey = self._make_deck(tmp_path)
        deck_script = _REPO_ROOT / "scripts" / "run" / "run_amip_smoke_deck.py"
        cmd = [
            sys.executable, str(deck_script),
            "--forcing-dir", str(tmp_path),
            "--start-year", str(sy), "--end-year", str(ey),
            "--grid-type", grid_type, "--discretization", disc,
            "--radiation", "rrtmg", "--resolution", str(res),
            "--ic", "default",
            "--days", "1", "--dry-run",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"dry-run failed:\n{r.stderr}\n{r.stdout}"
        return r.stdout

    _RAD_LABELS = (
        "Greenhouse gases",
        "Ozone (cyclic clim",
        "Tropospheric aerosol",
        "Volcanic stratospheric",
    )

    def _assert_rad_active(self, out):
        for label in self._RAD_LABELS:
            line = next((ln for ln in out.splitlines() if label in ln), None)
            assert line is not None, f"Missing {label} report line"
            assert "ACTIVE" in line, (
                f"{label!r} should be ACTIVE (forcing now consumed): {line!r}")

    def test_gaussian_spectral_rrtmg_forcing_active(self, tmp_path):
        out = self._run(tmp_path, "gaussian", "spectral", 21)
        # No silent-drop warning; spectral-solar remains inert (constant S_0).
        assert "does NOT consume external CMIP6 forcings" not in out
        self._assert_rad_active(out)
        solar = next((ln for ln in out.splitlines()
                      if "Solar TSI" in ln), None)
        assert solar is not None and "inert" in solar.lower(), (
            f"spectral solar FILE should still be inert: {solar!r}")

    def test_cubed_sphere_rrtmg_forcing_active(self, tmp_path):
        out = self._run(tmp_path, "cubed_sphere", "centered", 8)
        assert "does NOT consume external CMIP6 forcings" not in out
        self._assert_rad_active(out)
        solar = next((ln for ln in out.splitlines()
                      if "Solar TSI" in ln), None)
        assert solar is not None and "ACTIVE" in solar, (
            f"cubed_sphere solar should be ACTIVE: {solar!r}")

    def test_voronoi_mpas_rrtmg_forcing_active(self, tmp_path):
        out = self._run(tmp_path, "voronoi", "mpas", 4)
        assert "does NOT consume external CMIP6 forcings" not in out
        self._assert_rad_active(out)
        solar = next((ln for ln in out.splitlines()
                      if "Solar TSI" in ln), None)
        assert solar is not None and "inert" in solar.lower(), (
            f"MPAS solar FILE should still be inert: {solar!r}")




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
        """Generate a multi-year volcanic file (1979–1996) so the
        Pinatubo 1991 spike is in the file but at a calendar location
        that gets erased by cyclic interpolation.  It reaches 1996 because
        the 1995 sample below must lie inside the file: a day outside a
        non-cyclic file raises (F37) instead of holding the last record."""
        out = tmp_path_factory.mktemp("volc")
        path = out / "volcanic_1979_1996.nc"
        gaf.make_volcanic(path, 1979, 1996, nlat=18)
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
    """Regression tests for ``scripts/validate/validate_amip_run.py``.

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
        from scripts.validate.validate_amip_run import validate

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
        from scripts.validate.validate_amip_run import validate

        run = tmp_path / "failed_run"
        self._write_run(run, status="FAILED", residual_max=10.0,
                        sim_days=1.0, n_samples=3)
        rc = validate(run, strict=False)
        assert rc == 1

    def test_completed_status_passes(self, tmp_path):
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        from scripts.validate.validate_amip_run import validate

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
        from scripts.validate.validate_amip_run import validate

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
        from scripts.validate.validate_amip_run import validate

        run = tmp_path / "short_run"
        self._write_run(run, status="COMPLETED", residual_max=400.0,
                        sim_days=1.0, n_samples=3)
        rc = validate(run, strict=False)
        assert rc == 0


class TestClearSkyDiagToggle:
    """Issue #434 cheap lever: ``--clear-sky-diag`` / ``--no-clear-sky-diag``.

    The deck always forwarded ``--clear-sky-diag`` (a SECOND clear-sky RRTMG
    pass, ≈2× the radiation cost — needed for CRE diagnostics, pure overhead
    otherwise). The toggle lets a perf / spin-up run drop the second pass while
    keeping it ON by default for the production CRE-scoring deck.
    """

    _DECK = (
        Path(__file__).resolve().parents[2]
        / "scripts" / "run" / "run_amip_smoke_deck.py"
    )

    def test_no_clear_sky_diag_argument_declared(self):
        src = self._DECK.read_text()
        assert '"--no-clear-sky-diag"' in src, (
            "deck must expose --no-clear-sky-diag to drop the 2x radiation pass"
        )
        assert 'dest="clear_sky_diag"' in src

    def test_clear_sky_diag_forwarded_conditionally(self):
        src = self._DECK.read_text()
        # The flag is included only when args.clear_sky_diag is set (default on),
        # never unconditionally.
        assert '["--clear-sky-diag"] if args.clear_sky_diag else []' in src
        assert '\n        "--clear-sky-diag",\n' not in src, (
            "deck must NOT forward --clear-sky-diag unconditionally any more"
        )

    def test_default_is_clear_sky_on(self):
        """Default (no flag) keeps the clear-sky pass — the production deck
        reports CRE, so the behaviour is unchanged unless --no-clear-sky-diag."""
        import argparse
        # Mirror the deck's declaration (BooleanOptional-style pair).
        p = argparse.ArgumentParser()
        p.add_argument("--clear-sky-diag", dest="clear_sky_diag",
                       action="store_true", default=True)
        p.add_argument("--no-clear-sky-diag", dest="clear_sky_diag",
                       action="store_false")
        assert p.parse_args([]).clear_sky_diag is True
        assert p.parse_args(["--no-clear-sky-diag"]).clear_sky_diag is False


class TestDeckPhysicsDefaultsDrift:
    """The deck driver forwards its physics defaults to run_amip, whose
    _require_full_physics_for_amip rejects 'none'. A deck default of 'none'
    therefore kills every deck launch (and the all-grids smoke) at parse time
    — the exact drift the 2026-07-17 audit smoke run caught for
    --gravity-wave-drag. AST tripwire: no full-physics scheme flag in the deck
    script may default to 'none'."""

    _FULL_PHYSICS_FLAGS = {
        "--microphysics", "--convection", "--turbulence", "--gravity-wave-drag",
    }

    def test_deck_defaults_pass_full_physics_guard(self):
        import ast

        deck_path = (
            Path(__file__).resolve().parents[2]
            / "scripts" / "run" / "run_amip_smoke_deck.py"
        )
        tree = ast.parse(deck_path.read_text())
        found = {}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in self._FULL_PHYSICS_FLAGS):
                continue
            default = next(
                (kw.value.value for kw in node.keywords
                 if kw.arg == "default"
                 and isinstance(kw.value, ast.Constant)),
                None,
            )
            found[node.args[0].value] = default
        assert set(found) == self._FULL_PHYSICS_FLAGS, (
            f"deck flags moved/renamed; update this tripwire: {found}"
        )
        offenders = {f: d for f, d in found.items() if d == "none"}
        assert not offenders, (
            f"deck defaults rejected by _require_full_physics_for_amip: "
            f"{offenders} — every deck launch dies at parse time"
        )
