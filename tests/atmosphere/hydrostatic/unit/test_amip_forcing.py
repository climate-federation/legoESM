"""Tests for the AMIP forcing loader.

Uses synthetic NetCDF datasets to verify unit conversion, SIC clamping,
time interpolation, and missing-value fill.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.forcing.amip import (
    AMIPForcingConfig,
    load_amip_forcing,
    get_forcing_at_time,
    get_amip_preset,
    _fill_nan_nearest,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere


def _write_synthetic_forcing(path, n_time=4, sst_celsius=True, sic_percent=True,
                              add_nans=False):
    """Create a minimal synthetic AMIP forcing NetCDF file."""
    import xarray as xr

    n_lat, n_lon = 18, 36
    lat = np.linspace(-85, 85, n_lat)
    lon = np.linspace(5, 355, n_lon)
    times = np.arange(n_time) * 30.0  # days 0, 30, 60, 90

    sst = np.full((n_time, n_lat, n_lon), 20.0)  # 20 C or K (depends on offset)
    sic = np.full((n_time, n_lat, n_lon), 10.0 if sic_percent else 0.1)

    # Add latitude gradient to SST
    sst += lat[None, :, None] * 0.1

    # Add time variation
    sst += np.arange(n_time)[:, None, None] * 0.5

    # Polar ice only
    sic[:, :2, :] = 80.0 if sic_percent else 0.8  # high-lat ice
    sic[:, -2:, :] = 60.0 if sic_percent else 0.6

    if add_nans:
        # Simulate land mask
        sst[:, 5:8, 10:15] = np.nan
        sic[:, 5:8, 10:15] = np.nan

    time_coord = np.array([np.datetime64("2000-01-01") + np.timedelta64(int(t), "D")
                           for t in times])

    ds = xr.Dataset(
        {
            "sst": (("time", "lat", "lon"), sst.astype(np.float32)),
            "sic": (("time", "lat", "lon"), sic.astype(np.float32)),
        },
        coords={
            "time": time_coord,
            "lat": lat,
            "lon": lon,
        },
    )
    ds.to_netcdf(path)
    return ds


def _write_split_custom_forcing(sst_path, sic_path):
    """Create separate SST/SIC files with custom coord names and cftime."""
    import cftime
    import xarray as xr

    n_lat, n_lon, n_time = 18, 36, 4
    lat = np.linspace(-85, 85, n_lat)
    lon = np.linspace(5, 355, n_lon)
    times = [cftime.DatetimeNoLeap(2000, month, 1) for month in range(1, n_time + 1)]

    sst = np.full((n_time, n_lat, n_lon), 20.0, dtype=np.float32)
    sic = np.full((n_time, n_lat, n_lon), 0.1, dtype=np.float32)
    sst += lat[None, :, None] * 0.1
    sst += np.arange(n_time, dtype=np.float32)[:, None, None] * 0.5
    sic[:, :2, :] = 0.8
    sic[:, -2:, :] = 0.6

    coords = {
        "time": times,
        "ylat": lat,
        "xlon": lon,
    }
    xr.Dataset(
        {"SST": (("time", "ylat", "xlon"), sst)},
        coords=coords,
    ).to_netcdf(sst_path)
    xr.Dataset(
        {"iceFrac": (("time", "ylat", "xlon"), sic)},
        coords=coords,
    ).to_netcdf(sic_path)


@pytest.fixture(scope="module")
def grid():
    return create_cubed_sphere(4)


@pytest.fixture(scope="module")
def forcing_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("amip") / "forcing.nc"
    _write_synthetic_forcing(str(path), sst_celsius=True, sic_percent=True)
    return str(path)


@pytest.fixture(scope="module")
def forcing_path_nans(tmp_path_factory):
    path = tmp_path_factory.mktemp("amip_nans") / "forcing_nans.nc"
    _write_synthetic_forcing(str(path), sst_celsius=True, sic_percent=True, add_nans=True)
    return str(path)


@pytest.fixture(scope="module")
def split_custom_forcing_paths(tmp_path_factory):
    root = tmp_path_factory.mktemp("amip_split")
    sst_path = root / "sst.nc"
    sic_path = root / "sic.nc"
    _write_split_custom_forcing(str(sst_path), str(sic_path))
    return str(sst_path), str(sic_path)


class TestAMIPPresets:
    def test_cobe_preset(self):
        cfg = get_amip_preset("cobe")
        assert cfg.sst_var == "SST_cpl"
        assert cfg.sic_var == "ice_cov"
        assert cfg.sst_offset == 0.0
        assert cfg.sic_scale == 0.01

    def test_hadisst_preset(self):
        cfg = get_amip_preset("hadisst")
        assert cfg.sst_var == "sst"
        assert cfg.sst_offset == constants.T_freeze

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_amip_preset("nonexistent")


class TestLoadForcing:
    def test_load_basic(self, grid, forcing_path):
        config = AMIPForcingConfig(
            path=forcing_path,
            sst_var="sst",
            sic_var="sic",
            sst_offset=constants.T_freeze,  # C -> K
            sic_scale=0.01,     # percent -> fraction
        )
        forcing = load_amip_forcing(config, grid)
        assert forcing.sst.shape[0] == 4  # n_time
        assert forcing.sst.shape[1:] == (6, 4, 4)
        assert forcing.sic.shape == forcing.sst.shape
        assert forcing.times.shape == (4,)

    def test_sst_in_kelvin(self, grid, forcing_path):
        """SST should be converted to Kelvin (offset applied)."""
        config = AMIPForcingConfig(
            path=forcing_path,
            sst_var="sst",
            sic_var="sic",
            sst_offset=constants.T_freeze,
            sic_scale=0.01,
        )
        forcing = load_amip_forcing(config, grid)
        # Raw data is ~20 C, so after +273.15 offset should be ~293 K
        mean_sst = float(jnp.mean(forcing.sst[0]))
        assert 280 < mean_sst < 310, f"SST mean {mean_sst} not in expected Kelvin range"

    def test_sic_clamped(self, grid, forcing_path):
        """SIC should be in [0, 1] after scaling."""
        config = AMIPForcingConfig(
            path=forcing_path,
            sst_var="sst",
            sic_var="sic",
            sst_offset=constants.T_freeze,
            sic_scale=0.01,
        )
        forcing = load_amip_forcing(config, grid)
        assert float(jnp.min(forcing.sic)) >= 0.0
        assert float(jnp.max(forcing.sic)) <= 1.0

    def test_nan_fill(self, grid, forcing_path_nans):
        """NaN values over land should be filled."""
        config = AMIPForcingConfig(
            path=forcing_path_nans,
            sst_var="sst",
            sic_var="sic",
            sst_offset=constants.T_freeze,
            sic_scale=0.01,
        )
        forcing = load_amip_forcing(config, grid)
        assert jnp.all(jnp.isfinite(forcing.sst))
        assert jnp.all(jnp.isfinite(forcing.sic))

    def test_missing_variable_raises(self, grid, forcing_path):
        """Missing variable should give a clear error."""
        config = AMIPForcingConfig(
            path=forcing_path,
            sst_var="nonexistent_sst",
            sic_var="sic",
        )
        with pytest.raises(KeyError, match="Missing variables"):
            load_amip_forcing(config, grid)

    def test_load_split_custom_with_cftime(self, grid, split_custom_forcing_paths):
        """Custom forcing should support separate SIC files and cftime axes."""
        sst_path, sic_path = split_custom_forcing_paths
        config = AMIPForcingConfig(
            dataset="custom",
            path=sst_path,
            sic_path=sic_path,
            sst_var="SST",
            sic_var="iceFrac",
            time_var="time",
            lat_var="ylat",
            lon_var="xlon",
            sst_offset=constants.T_freeze,
            sic_scale=1.0,
        )
        forcing = load_amip_forcing(config, grid)
        assert forcing.sst.shape[0] == 4
        assert forcing.sic.shape == forcing.sst.shape
        assert float(forcing.times[0]) == 0.0
        assert float(forcing.times[1]) > 0.0
        assert jnp.all(jnp.isfinite(forcing.sst))
        assert jnp.all(jnp.isfinite(forcing.sic))


class TestTimeInterpolation:
    def test_at_record_time(self, grid, forcing_path):
        """Interpolation at exact record time should return that record."""
        config = AMIPForcingConfig(
            path=forcing_path,
            sst_var="sst",
            sic_var="sic",
            sst_offset=constants.T_freeze,
            sic_scale=0.01,
        )
        forcing = load_amip_forcing(config, grid)
        sst_0, sic_0 = get_forcing_at_time(forcing, 0.0)
        assert jnp.allclose(sst_0, forcing.sst[0], atol=1e-5)

    def test_midpoint_interpolation(self, grid, forcing_path):
        """Midpoint between two records should be the average."""
        config = AMIPForcingConfig(
            path=forcing_path,
            sst_var="sst",
            sic_var="sic",
            sst_offset=constants.T_freeze,
            sic_scale=0.01,
        )
        forcing = load_amip_forcing(config, grid)
        sst_mid, _ = get_forcing_at_time(forcing, 15.0)  # midpoint of 0 and 30
        expected = 0.5 * (forcing.sst[0] + forcing.sst[1])
        assert jnp.allclose(sst_mid, expected, atol=1e-4)

    def test_cyclic_outside_range(self, grid, forcing_path):
        """A climatology (span < 1 yr, loaded without start_year) repeats on a
        365-day annual period — NOT the record span."""
        config = AMIPForcingConfig(
            path=forcing_path,
            sst_var="sst",
            sic_var="sic",
            sst_offset=constants.T_freeze,
            sic_scale=0.01,
        )
        forcing = load_amip_forcing(config, grid)
        # One full year (365 d) later returns the same SST as the start.
        t0 = float(forcing.times[0])
        sst_wrap, _ = get_forcing_at_time(forcing, t0 + 365.0)
        sst_first, _ = get_forcing_at_time(forcing, t0)
        assert jnp.allclose(sst_wrap, sst_first, atol=1e-5)

    def test_climatology_dec_jan_seam_is_continuous(self, grid, tmp_path):
        """The Dec->Jan seam of a 12-month mid-month climatology INTERPOLATES
        (Taylor 2000 cyclic bcs), not holds December then jumps at the year
        boundary. Records Jan16..Dec16 sit at rel-days 0..~334 with SST tag
        301..312; the wrap anchor after December is January (301) at day 365."""
        p = tmp_path / "clim12.nc"
        _write_multiyear_forcing(str(p), start=2000, nyears=1, midmonth=True)
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic")
        f = load_amip_forcing(cfg, grid)   # no start_year -> 12-month climatology
        dec_day = float(f.times[-1])       # last mid-month anchor (December)
        dec = float(jnp.mean(get_forcing_at_time(f, dec_day)[0]))
        jan = float(jnp.mean(get_forcing_at_time(f, 365.0)[0]))   # wraps to January
        mid = float(jnp.mean(get_forcing_at_time(f, 0.5 * (dec_day + 365.0))[0]))
        assert abs(dec - 312.0) < 1e-3 and abs(jan - 301.0) < 1e-3
        # Strictly between Dec and Jan, ~linear across the seam (fails on the old
        # clip-to-ntime-2 path, which holds December -> mid == 312).
        assert min(dec, jan) + 0.1 < mid < max(dec, jan) - 0.1
        assert abs(mid - 0.5 * (dec + jan)) < 0.6
        # Continuous across the boundary: just-before-wrap ~ the January anchor.
        pre = float(jnp.mean(get_forcing_at_time(f, 364.999)[0]))
        assert abs(pre - jan) < 0.6
        # JIT-safe (ntime-static branch + modular index math): compiles and
        # matches eager — this runs inside the jitted forcing update.
        import jax
        seam_day = 0.5 * (dec_day + 365.0)
        j = jax.jit(lambda d: get_forcing_at_time(f, d)[0])
        assert abs(float(jnp.mean(j(seam_day))) - mid) < 1e-6


class TestFillNanNearest:
    def test_no_nans_passthrough(self):
        data = np.ones((2, 5, 10))
        lat = np.linspace(-90, 90, 5)
        lon = np.linspace(0, 350, 10)
        result = _fill_nan_nearest(data, lat, lon)
        np.testing.assert_array_equal(result, data)

    def test_fills_nans(self):
        data = np.ones((1, 5, 10))
        data[0, 2, 5] = np.nan
        lat = np.linspace(-90, 90, 5)
        lon = np.linspace(0, 350, 10)
        result = _fill_nan_nearest(data, lat, lon)
        assert not np.any(np.isnan(result))
        assert result[0, 2, 5] == 1.0  # nearest neighbor should be 1.0


def _write_multiyear_forcing(path, start=2000, nyears=3, midmonth=False,
                             use_cftime=False):
    """Monthly SST/SIC file spanning [start, start+nyears). SST encodes the
    absolute year+month (300 + (year-start)*10 + month [K]) so each record is
    identifiable; SIC constant. ``midmonth`` puts records on the 16th (bcs
    convention) instead of the 1st; ``use_cftime`` writes a NoLeap cftime axis
    (the real input4MIPs format) instead of datetime64."""
    import xarray as xr

    n_lat, n_lon = 8, 16
    lat = np.linspace(-80, 80, n_lat)
    lon = np.linspace(10, 350, n_lon)
    day = 16 if midmonth else 1
    dates, tags = [], []
    for y in range(start, start + nyears):
        for m in range(1, 13):
            if use_cftime:
                import cftime
                dates.append(cftime.DatetimeNoLeap(y, m, day))
            else:
                dates.append(np.datetime64(f"{y:04d}-{m:02d}-{day:02d}"))
            tags.append(300.0 + (y - start) * 10.0 + m)
    n_time = len(dates)
    sst = np.zeros((n_time, n_lat, n_lon), np.float32)
    for t, v in enumerate(tags):
        sst[t] = v
    sic = np.full((n_time, n_lat, n_lon), 0.2, np.float32)
    xr.Dataset(
        {"sst": (("time", "lat", "lon"), sst), "sic": (("time", "lat", "lon"), sic)},
        coords={"time": np.array(dates), "lat": lat, "lon": lon},
    ).to_netcdf(path)


class TestStartYearAnchoring:
    """AMIP-II date anchoring: a model day indexes the file by real calendar
    date, so a run reads the SST/SIC of its simulated year (issue: SST/SIC were
    indexed relative to the file's first record → a 1979 run used 1870 ocean)."""

    def test_reads_run_year_not_file_start(self, grid, tmp_path):
        p = tmp_path / "multiyear.nc"
        _write_multiyear_forcing(str(p), start=2000, nyears=3)
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic")
        f_anchored = load_amip_forcing(cfg, grid, start_year=2001)
        f_legacy = load_amip_forcing(cfg, grid)  # relative (pre-fix) indexing
        sst0_anchored = float(jnp.mean(get_forcing_at_time(f_anchored, 0.0)[0]))
        sst0_legacy = float(jnp.mean(get_forcing_at_time(f_legacy, 0.0)[0]))
        assert abs(sst0_anchored - 311.0) < 1.0   # 2001-Jan tag (300+10+1)
        assert abs(sst0_legacy - 301.0) < 1.0      # 2000-Jan tag — documents old bug
        assert float(f_anchored.times[0]) < 0.0    # 2000 precedes the 2001 epoch

    def test_midmonth_offset_preserved(self, grid, tmp_path):
        # Mid-month bcs anchor sits ~15.5 days after Jan 1, not on day 0.
        p = tmp_path / "midmonth.nc"
        _write_multiyear_forcing(str(p), start=2001, nyears=2, midmonth=True)
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic")
        f = load_amip_forcing(cfg, grid, start_year=2001)
        assert 14.0 < float(f.times[0]) < 17.0      # 2001-01-16 -> ~15.5, not 0
        sst_midjan = float(jnp.mean(get_forcing_at_time(f, 15.5)[0]))
        assert abs(sst_midjan - 301.0) < 1.0        # 2001-Jan tag (start year -> 300+0+1)

    def test_coverage_guard_raises_when_year_outside_file(self, grid, tmp_path):
        p = tmp_path / "multiyear.nc"
        _write_multiyear_forcing(str(p), start=2000, nyears=3)
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic")
        with pytest.raises(ValueError, match="does not cover start_year"):
            load_amip_forcing(cfg, grid, start_year=1990)

    def test_transient_past_end_holds_last_record(self, grid, tmp_path):
        # A transient (multi-year) file uses no cyclic wrap; a query past the
        # last record must HOLD it, not linearly extrapolate off the clamped
        # index (unclamped weight -> physically unbounded SST).
        p = tmp_path / "multiyear.nc"
        _write_multiyear_forcing(str(p), start=2000, nyears=3)
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic")
        f = load_amip_forcing(cfg, grid, start_year=2001)
        last_day = float(f.times[-1])
        sst_end, sic_end = get_forcing_at_time(f, last_day)
        sst_past, sic_past = get_forcing_at_time(f, last_day + 1500.0)  # far past
        # last record = 2002-12 tag = 300 + (2002-2000)*10 + 12 = 332
        assert abs(float(jnp.mean(sst_past)) - 332.0) < 1.0
        assert jnp.allclose(sst_past, sst_end, atol=1e-4)   # held, not extrapolated
        assert jnp.allclose(sic_past, sic_end, atol=1e-6)

    def test_thirteen_record_file_is_transient(self, grid, tmp_path):
        # A 13-month NoLeap series spans exactly 365 days (< 366) but is
        # multi-year — must be classified transient (anchored + coverage-guarded
        # via ntime>12), NOT a climatology. A start_year outside the file then
        # trips the coverage guard (which only runs on the transient path).
        import cftime
        import xarray as xr
        n_lat, n_lon = 8, 16
        lat = np.linspace(-80, 80, n_lat)
        lon = np.linspace(10, 350, n_lon)
        dates = [cftime.DatetimeNoLeap(2001 + (m // 12), (m % 12) + 1, 1)
                 for m in range(13)]
        sst = np.stack([np.full((n_lat, n_lon), 290.0 + m, np.float32)
                        for m in range(13)])
        sic = np.full((13, n_lat, n_lon), 0.2, np.float32)
        p = tmp_path / "thirteen.nc"
        xr.Dataset(
            {"sst": (("time", "lat", "lon"), sst), "sic": (("time", "lat", "lon"), sic)},
            coords={"time": np.array(dates), "lat": lat, "lon": lon},
        ).to_netcdf(str(p))
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic")
        with pytest.raises(ValueError, match="does not cover start_year"):
            load_amip_forcing(cfg, grid, start_year=1990)

    def test_thirteen_record_interp_holds_not_wraps(self, grid, tmp_path):
        # get_forcing_at_time must honor the SAME ntime>12 transient criterion as
        # the loader: a 13-record file spans 365 d (<366) yet must NOT cyclically
        # wrap. A query past the last record holds it, not replays record 0.
        # (Regression for the interp using only ``span`` to decide the period.)
        import cftime
        import xarray as xr
        n_lat, n_lon = 8, 16
        lat = np.linspace(-80, 80, n_lat)
        lon = np.linspace(10, 350, n_lon)
        dates = [cftime.DatetimeNoLeap(2001 + (m // 12), (m % 12) + 1, 1)
                 for m in range(13)]
        sst = np.stack([np.full((n_lat, n_lon), 290.0 + m, np.float32)
                        for m in range(13)])
        sic = np.full((13, n_lat, n_lon), 0.2, np.float32)
        p = tmp_path / "thirteen_wrap.nc"
        xr.Dataset(
            {"sst": (("time", "lat", "lon"), sst), "sic": (("time", "lat", "lon"), sic)},
            coords={"time": np.array(dates), "lat": lat, "lon": lon},
        ).to_netcdf(str(p))
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic")
        f = load_amip_forcing(cfg, grid, start_year=2001)
        last = float(f.times[-1])
        sst_past = float(jnp.mean(get_forcing_at_time(f, last + 400.0)[0]))
        # last record = 2002-01 tag = 290 + 12 = 302; held, not wrapped to 290.
        assert abs(sst_past - 302.0) < 1.0
        # The period==0 modulo in the discarded jnp.where branch must NOT emit a
        # NaN: grad w.r.t. day stays finite (a `% 0` rem-NaN would poison the VJP
        # through both where-branches, and trip JAX_DEBUG_NANS).
        import jax
        g = jax.grad(lambda d: jnp.mean(get_forcing_at_time(f, d)[0]))(last + 400.0)
        assert jnp.isfinite(g)

    def test_bare_numeric_transient_axis_raises(self, grid, tmp_path):
        # A metadata-less numeric time axis cannot be calendar-anchored (its
        # reference year is unknown); a transient one must fail loudly, not
        # silently serve the wrong era. A <=12-record climatology still loads.
        import xarray as xr
        n_lat, n_lon = 8, 16
        lat = np.linspace(-80, 80, n_lat)
        lon = np.linspace(10, 350, n_lon)

        def _write(path, ntime):
            t = np.arange(ntime, dtype=np.float64) * 30.0  # plain days, no units
            xr.Dataset(
                {"sst": (("time", "lat", "lon"),
                         np.full((ntime, n_lat, n_lon), 290.0, np.float32)),
                 "sic": (("time", "lat", "lon"),
                         np.full((ntime, n_lat, n_lon), 0.2, np.float32))},
                coords={"time": t, "lat": lat, "lon": lon},
            ).to_netcdf(path)

        p_t = tmp_path / "numeric_transient.nc"
        _write(str(p_t), 24)  # span 690 d -> transient
        cfg_t = AMIPForcingConfig(path=str(p_t), sst_var="sst", sic_var="sic")
        with pytest.raises(ValueError, match="bare-numeric"):
            load_amip_forcing(cfg_t, grid, start_year=1979)

        p_c = tmp_path / "numeric_clim.nc"
        _write(str(p_c), 12)  # climatology -> relative, no anchoring, loads fine
        cfg_c = AMIPForcingConfig(path=str(p_c), sst_var="sst", sic_var="sic")
        assert load_amip_forcing(cfg_c, grid, start_year=1979).times.shape == (12,)

    def test_cftime_axis_anchors(self, grid, tmp_path):
        # The real input4MIPs format is a NoLeap cftime axis — exercise the
        # cftime.date2num(units="days since <start_year>-01-01") anchoring path.
        p = tmp_path / "cftime_my.nc"
        _write_multiyear_forcing(str(p), start=2000, nyears=3, use_cftime=True)
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic")
        f = load_amip_forcing(cfg, grid, start_year=2001)
        sst0 = float(jnp.mean(get_forcing_at_time(f, 0.0)[0]))
        assert abs(sst0 - 311.0) < 1.0        # 2001-Jan tag (300+10+1)

    def test_climatology_phase_independent_of_start_year(self, grid, tmp_path):
        # REGRESSION: a 12-month climatology (span < 1 yr) must be indexed by
        # day-of-year, NOT anchored to start_year — else its seasonal phase
        # shifts arbitrarily with start_year. A single-year file dated 2000
        # loaded for 1979 vs 2005 must give identical SST for every day.
        p = tmp_path / "clim.nc"
        _write_multiyear_forcing(str(p), start=2000, nyears=1, midmonth=True)
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic")
        f_79 = load_amip_forcing(cfg, grid, start_year=1979)
        f_05 = load_amip_forcing(cfg, grid, start_year=2005)
        for day in (0.0, 100.0, 300.0):
            s79 = float(jnp.mean(get_forcing_at_time(f_79, day)[0]))
            s05 = float(jnp.mean(get_forcing_at_time(f_05, day)[0]))
            assert abs(s79 - s05) < 1e-6     # phase independent of start_year
        # Day 0 is the January record (tag 301), not a shifted month.
        assert abs(float(jnp.mean(get_forcing_at_time(f_79, 0.0)[0])) - 301.0) < 1.0

    @staticmethod
    def _write_icon_multiyear(path, start=2000, nyears=3, midmonth=True):
        """ICON-unstructured (cell/clon/clat) analogue of
        ``_write_multiyear_forcing``: SST tags each record with
        300 + (year-start)*10 + month [K] so the served era is identifiable."""
        import xarray as xr
        n_cell = 12
        day = 16 if midmonth else 1
        dates, tags = [], []
        for y in range(start, start + nyears):
            for m in range(1, 13):
                dates.append(np.datetime64(f"{y:04d}-{m:02d}-{day:02d}"))
                tags.append(300.0 + (y - start) * 10.0 + m)
        n_time = len(dates)
        sst = np.zeros((n_time, n_cell), np.float32)
        for t, v in enumerate(tags):
            sst[t] = v
        xr.Dataset(
            {
                "sst": (("time", "cell"), sst),
                "sic": (("time", "cell"), np.full((n_time, n_cell), 0.2, np.float32)),
            },
            coords={
                "time": np.array(dates),
                "clon": ("cell", np.linspace(0.0, 6.0, n_cell)),
                "clat": ("cell", np.linspace(-1.0, 1.0, n_cell)),
            },
        ).to_netcdf(str(path))

    def test_icon_transient_anchors_to_run_year(self, grid, tmp_path):
        # The ICON path anchors a multi-year file exactly like lat-lon
        # (pre-fix it indexed relative to the first record — a 1979 run on
        # the pool bc_sst_1979_2016.nc, which STARTS 1978, read 1978 ocean).
        p = tmp_path / "icon_my.nc"
        self._write_icon_multiyear(str(p), start=2000, nyears=3)
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic")
        f_anchored = load_amip_forcing(cfg, grid, start_year=2001)
        f_legacy = load_amip_forcing(cfg, grid)   # relative (pre-fix) indexing
        sst0_anchored = float(jnp.mean(get_forcing_at_time(f_anchored, 15.5)[0]))
        sst0_legacy = float(jnp.mean(get_forcing_at_time(f_legacy, 0.0)[0]))
        assert abs(sst0_anchored - 311.0) < 1.0   # 2001-Jan tag (300+10+1)
        assert abs(sst0_legacy - 301.0) < 1.0     # 2000-Jan tag — documents old bug
        assert float(f_anchored.times[0]) < 0.0   # 2000 precedes the 2001 epoch

    def test_icon_coverage_guard_raises_when_year_outside_file(self, grid, tmp_path):
        p = tmp_path / "icon_my.nc"
        self._write_icon_multiyear(str(p), start=2000, nyears=3)
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic")
        with pytest.raises(ValueError, match="does not cover start_year"):
            load_amip_forcing(cfg, grid, start_year=1990)

    def test_icon_climatology_phase_independent_of_start_year(self, grid, tmp_path):
        # A 12-month ICON climatology stays first-record-relative (day-of-year
        # phase), independent of start_year — same contract as lat-lon.
        p = tmp_path / "icon_clim.nc"
        self._write_icon_multiyear(str(p), start=2000, nyears=1)
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic")
        f_79 = load_amip_forcing(cfg, grid, start_year=1979)
        f_05 = load_amip_forcing(cfg, grid, start_year=2005)
        for day in (0.0, 100.0, 300.0):
            s79 = float(jnp.mean(get_forcing_at_time(f_79, day)[0]))
            s05 = float(jnp.mean(get_forcing_at_time(f_05, day)[0]))
            assert abs(s79 - s05) < 1e-6


class TestNoleapCalendarDateAnchoring:
    """Audit F1: the model day advances on a strict noleap (365-day) clock,
    so an anchored file axis must count days BY CALENDAR DATE on that clock.
    True Gregorian day counts ran ~1 d per 4 yr ahead of the model's nominal
    date (~9 days over 1979-2014) while insolation stayed 365-periodic."""

    def test_time_coord_to_days_maps_by_calendar_date(self):
        from legoesm.forcing.amip import _time_coord_to_days
        t = np.array([
            np.datetime64("1987-07-16"),
            np.datetime64("1980-02-29"),
            np.datetime64("1980-03-01"),
        ])
        days = _time_coord_to_days(t, epoch_year=1979)
        # 1987-07-16 -> (1987-1979)*365 + noleap doy(Jul 16)=197 - 1 = 3116,
        # NOT the true Gregorian count 3118 (leap days 1980, 1984).
        greg = float((t[0] - np.datetime64("1979-01-01")) / np.timedelta64(1, "D"))
        assert greg == 3118.0          # documents the old (drifting) mapping
        assert days[0] == 8 * 365 + 196.0
        assert days[1] == 1 * 365 + 58.0    # Feb 29 collapsed onto Feb 28
        assert days[2] == 1 * 365 + 59.0    # Mar 1
        # cftime Gregorian axis maps identically (incl. intra-day fraction).
        import cftime
        tg = np.array([cftime.DatetimeGregorian(1987, 7, 16, 12)], dtype=object)
        days_g = _time_coord_to_days(tg, epoch_year=1979)
        assert days_g[0] == 8 * 365 + 196.5
        # cftime NoLeap axis: unchanged exact mapping.
        tn = np.array([cftime.DatetimeNoLeap(1987, 7, 16)], dtype=object)
        days_n = _time_coord_to_days(tn, epoch_year=1979)
        assert days_n[0] == 8 * 365 + 196.0

    def test_midjuly_year8_read_at_model_midjuly(self, grid, tmp_path):
        # End-to-end: a 1979-1988 Gregorian-dated monthly file read at model
        # day (N+8)*365 + 196 (= model 1987-07-16) must return the 1987-07
        # record EXACTLY (weight 0), not a leap-drifted mix.  Tags encode
        # year+month as 280 + (year-1979)*2 + month*0.1 [K] (kept inside the
        # 240-340 K physical-sanity guard over 10 years).
        import xarray as xr
        n_lat, n_lon = 8, 16
        lat = np.linspace(-80, 80, n_lat)
        lon = np.linspace(10, 350, n_lon)
        dates, tags = [], []
        for y in range(1979, 1989):
            for m in range(1, 13):
                dates.append(np.datetime64(f"{y:04d}-{m:02d}-16"))
                tags.append(280.0 + (y - 1979) * 2.0 + m * 0.1)
        sst = np.stack([np.full((n_lat, n_lon), v, np.float32) for v in tags])
        sic = np.full((len(tags), n_lat, n_lon), 0.2, np.float32)
        p = tmp_path / "greg_decade.nc"
        xr.Dataset(
            {"sst": (("time", "lat", "lon"), sst),
             "sic": (("time", "lat", "lon"), sic)},
            coords={"time": np.array(dates), "lat": lat, "lon": lon},
        ).to_netcdf(str(p))
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic")
        f = load_amip_forcing(cfg, grid, start_year=1979)
        day = 8 * 365 + 196.0
        sst_t = float(jnp.mean(get_forcing_at_time(f, day)[0]))
        assert abs(sst_t - (280.0 + 8 * 2.0 + 0.7)) < 1e-3  # 1987-07 tag = 296.7
        # The anchored axis itself carries the noleap day count for that
        # record (times are exact, no ~2-day Gregorian excess by 1987).
        idx = 8 * 12 + 6                                    # 1987-07 record
        assert float(f.times[idx]) == day

    def test_daily_axis_feb29_record_dropped(self, grid, tmp_path):
        # A DAILY Gregorian axis maps Feb 29 onto the same noleap coordinate
        # as Feb 28; the duplicate record must be DROPPED (OMIP leap-day
        # filtering) so the anchored axis stays strictly increasing for
        # searchsorted (codex review).
        import xarray as xr
        n_lat, n_lon = 8, 16
        lat = np.linspace(-80, 80, n_lat)
        lon = np.linspace(10, 350, n_lon)
        n_days = 62   # 1980-01-01 .. 1980-03-02 (incl. Feb 29) -> transient
        dates = np.array([np.datetime64("1980-01-01") + np.timedelta64(t, "D")
                          for t in range(n_days)])
        sst = np.stack([np.full((n_lat, n_lon), 285.0 + 0.1 * t, np.float32)
                        for t in range(n_days)])
        sic = np.full((n_days, n_lat, n_lon), 0.2, np.float32)
        p = tmp_path / "daily_leap.nc"
        xr.Dataset(
            {"sst": (("time", "lat", "lon"), sst),
             "sic": (("time", "lat", "lon"), sic)},
            coords={"time": dates, "lat": lat, "lon": lon},
        ).to_netcdf(str(p))
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic")
        f = load_amip_forcing(cfg, grid, start_year=1980)
        # Feb 29 (record 59) dropped: 61 records remain, axis strictly up.
        assert f.times.shape[0] == n_days - 1
        assert bool(jnp.all(jnp.diff(f.times) > 0))
        # Model day 58 (= 1980-02-28 noleap) returns the FEB 28 record.
        sst_t = float(jnp.mean(get_forcing_at_time(f, 58.0)[0]))
        assert abs(sst_t - (285.0 + 0.1 * 58)) < 1e-3
        # Model day 59 (= 1980-03-01 noleap) returns the MAR 1 record (the
        # Feb-29 value, 285.0+5.9, is gone).
        sst_t = float(jnp.mean(get_forcing_at_time(f, 59.0)[0]))
        assert abs(sst_t - (285.0 + 0.1 * 60)) < 1e-3


class TestBcsClipAfterInterp:
    """PCMDI mid-month bcs anchors overshoot [0,1]; clip AFTER interpolation."""

    def _write(self, path, sic_vals):
        import xarray as xr
        n_lat, n_lon = 8, 16
        lat = np.linspace(-80, 80, n_lat)
        lon = np.linspace(10, 350, n_lon)
        times = np.array([np.datetime64("2001-01-16"), np.datetime64("2001-02-16")])
        sst = np.full((2, n_lat, n_lon), 290.0, np.float32)
        sic = np.zeros((2, n_lat, n_lon), np.float32)
        sic[0] = sic_vals[0]
        sic[1] = sic_vals[1]
        xr.Dataset(
            {"sst": (("time", "lat", "lon"), sst), "sic": (("time", "lat", "lon"), sic)},
            coords={"time": times, "lat": lat, "lon": lon},
        ).to_netcdf(path)

    def test_overshoot_loads_and_clips(self, grid, tmp_path):
        p = tmp_path / "bcs.nc"
        self._write(str(p), sic_vals=(2.5, -0.5))  # raw ~+250% / -50%
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic", sic_scale=1.0)
        f = load_amip_forcing(cfg, grid, start_year=2001)   # must NOT raise
        assert float(jnp.max(f.sic)) > 1.0                  # anchors kept unclamped
        _, sic_t = get_forcing_at_time(f, 25.0)             # between the two records
        assert float(jnp.min(sic_t)) >= 0.0
        assert float(jnp.max(sic_t)) <= 1.0

    def test_wrong_scale_still_rejected(self, grid, tmp_path):
        # A genuine 100x scale error (percent file at scale 1.0) still trips.
        p = tmp_path / "pct.nc"
        self._write(str(p), sic_vals=(80.0, 80.0))
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic", sic_scale=1.0)
        with pytest.raises(ValueError):
            load_amip_forcing(cfg, grid, start_year=2001)

    def test_sic_scale_zero_no_ice_run(self, grid, tmp_path):
        # sic_scale=0 is an explicit no-sea-ice sensitivity run: it must load
        # (exempt from the units/scale guard, which otherwise rejects a fraction
        # file at any scale != 1) and yield zero ice.
        import xarray as xr
        n_lat, n_lon = 8, 16
        lat = np.linspace(-80, 80, n_lat)
        lon = np.linspace(10, 350, n_lon)
        times = np.array([np.datetime64("2001-01-16"), np.datetime64("2001-02-16")])
        sic = xr.DataArray(
            np.full((2, n_lat, n_lon), 0.8, np.float32),
            dims=("time", "lat", "lon"), attrs={"units": "1"},   # fraction
        )
        sst = xr.DataArray(np.full((2, n_lat, n_lon), 290.0, np.float32),
                           dims=("time", "lat", "lon"))
        p = tmp_path / "noice.nc"
        xr.Dataset({"sst": sst, "sic": sic},
                   coords={"time": times, "lat": lat, "lon": lon}).to_netcdf(str(p))
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic", sic_scale=0.0)
        f = load_amip_forcing(cfg, grid, start_year=2001)   # must NOT raise
        _, sic_t = get_forcing_at_time(f, 20.0)
        assert float(jnp.max(sic_t)) == 0.0                  # no ice anywhere


def _write_icon_file(path, sst_vals, sic_vals, sst_units=None, sic_units=None):
    """ICON-unstructured (cell/clon/clat) file with explicit per-record cell
    values.  ``sst_vals``/``sic_vals`` have shape (ntime, ncell); optional
    ``units`` attrs exercise the shared units guard."""
    import xarray as xr
    sst_vals = np.asarray(sst_vals, np.float32)
    sic_vals = np.asarray(sic_vals, np.float32)
    n_time, n_cell = sst_vals.shape
    times = np.array([np.datetime64("2001-01-16") + np.timedelta64(31 * t, "D")
                      for t in range(n_time)])
    sst = xr.DataArray(sst_vals, dims=("time", "cell"))
    sic = xr.DataArray(sic_vals, dims=("time", "cell"))
    if sst_units is not None:
        sst.attrs["units"] = sst_units
    if sic_units is not None:
        sic.attrs["units"] = sic_units
    xr.Dataset(
        {"sst": sst, "sic": sic},
        coords={
            "time": times,
            "clon": ("cell", np.linspace(0.0, 6.0, n_cell)),
            "clat": ("cell", np.linspace(-1.0, 1.0, n_cell)),
        },
    ).to_netcdf(str(path))


class TestIconDataQuality:
    """Audits F2 + F3: the ICON unstructured path must keep bcs anchors
    unclamped (clip AFTER interpolation, Taylor 2000) and must run the same
    NaN-fill + units/sanity guards as the lat-lon path."""

    def test_icon_anchors_unclamped_clip_after_interp(self, grid, tmp_path):
        # F2: SIC anchors overshooting [0,1] and a sub-freezing SST anchor
        # survive LOAD unclamped; the interpolated value is clipped/floored.
        p = tmp_path / "icon_bcs.nc"
        n_cell = 12
        sst = np.stack([np.full(n_cell, 250.0), np.full(n_cell, 300.0)])
        sic = np.stack([np.full(n_cell, 1.5), np.full(n_cell, -0.5)])
        _write_icon_file(p, sst, sic)
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic",
                                sic_scale=1.0)
        f = load_amip_forcing(cfg, grid)
        assert float(jnp.max(f.sic)) > 1.0          # anchors kept raw
        assert float(jnp.min(f.sic)) < 0.0
        assert float(jnp.min(f.sst)) < float(constants.T_freeze_ocean)
        t0 = float(f.times[0])
        for day in (t0, t0 + 15.5, t0 + 31.0):
            sst_t, sic_t = get_forcing_at_time(f, day)
            assert float(jnp.min(sic_t)) >= 0.0
            assert float(jnp.max(sic_t)) <= 1.0
            assert float(jnp.min(sst_t)) >= float(cfg.T_ice) - 1e-5

    def test_icon_nan_cells_filled_before_regrid(self, grid, tmp_path):
        # F3: NaN (land-masked) ICON cells must be nearest-filled BEFORE the
        # KD-tree regrid — previously they propagated to the target grid.
        p = tmp_path / "icon_nan.nc"
        n_cell = 12
        sst = np.full((2, n_cell), 290.0)
        sic = np.full((2, n_cell), 0.3)
        sst[:, 3:6] = np.nan
        sic[:, 3:6] = np.nan
        _write_icon_file(p, sst, sic)
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic",
                                sic_scale=1.0)
        f = load_amip_forcing(cfg, grid)
        assert bool(jnp.all(jnp.isfinite(f.sst)))
        assert bool(jnp.all(jnp.isfinite(f.sic)))
        # Filled from the nearest VALID cell (all-constant valid field).
        np.testing.assert_allclose(np.asarray(f.sst), 290.0, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(f.sic), 0.3, rtol=1e-6)

    def test_icon_percent_sic_wrong_scale_rejected(self, grid, tmp_path):
        # F3: percent SIC at sic_scale=1.0 must fail loudly (same guard as
        # the lat-lon path), and load fine at the correct 0.01 scale.
        p = tmp_path / "icon_pct.nc"
        n_cell = 12
        sst = np.full((2, n_cell), 290.0)
        sic = np.full((2, n_cell), 80.0)          # percent
        _write_icon_file(p, sst, sic, sic_units="%")
        cfg_bad = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic",
                                    sic_scale=1.0)
        with pytest.raises(ValueError, match="(?i)percent"):
            load_amip_forcing(cfg_bad, grid)
        cfg_ok = cfg_bad._replace(sic_scale=0.01)
        f = load_amip_forcing(cfg_ok, grid)
        np.testing.assert_allclose(np.asarray(f.sic), 0.8, rtol=1e-6)

    def test_icon_kelvin_sst_spurious_offset_rejected(self, grid, tmp_path):
        # F3: a Kelvin ICON file with the Celsius offset must be rejected
        # (would land at ~575 K), same cross-check as the lat-lon path.
        p = tmp_path / "icon_kelvin.nc"
        n_cell = 12
        sst = np.full((2, n_cell), 290.0)
        sic = np.full((2, n_cell), 0.3)
        _write_icon_file(p, sst, sic, sst_units="K")
        cfg = AMIPForcingConfig(path=str(p), sst_var="sst", sic_var="sic",
                                sst_offset=constants.T_freeze, sic_scale=1.0)
        with pytest.raises(ValueError, match="(?i)kelvin|spurious"):
            load_amip_forcing(cfg, grid)


def test_sst_floor_applied_after_interp():
    """SST anchors below the seawater freezing point survive load (bcs) but the
    interpolated value is floored to T_ice."""
    from legoesm.forcing.amip import AMIPForcing
    cfg = AMIPForcingConfig()  # T_ice = constants.T_freeze_ocean
    sst = jnp.stack([jnp.full((4, 8), 250.0), jnp.full((4, 8), 300.0)])  # below/above
    sic = jnp.full((2, 4, 8), 0.2)
    forcing = AMIPForcing(times=jnp.array([0.0, 30.0]), sst=sst, sic=sic, config=cfg)
    sst_t, _ = get_forcing_at_time(forcing, 0.0)  # the 250 K anchor
    assert float(jnp.min(sst_t)) >= float(cfg.T_ice) - 1e-6


def test_configured_t_ice_is_the_floor():
    """The SST floor honors the CONFIGURED T_ice (e.g. from --t-ice-k), not the
    hardcoded default — get_forcing_at_time reads forcing.config.T_ice."""
    from legoesm.forcing.amip import AMIPForcing
    cfg = AMIPForcingConfig(T_ice=275.0)          # a raised freezing floor
    sst = jnp.stack([jnp.full((4, 8), 273.0), jnp.full((4, 8), 273.0)])  # below 275
    sic = jnp.full((2, 4, 8), 0.2)
    forcing = AMIPForcing(times=jnp.array([0.0, 30.0]), sst=sst, sic=sic, config=cfg)
    sst_t, _ = get_forcing_at_time(forcing, 0.0)
    assert float(jnp.min(sst_t)) >= 275.0 - 1e-6   # floored at the configured value


def test_get_forcing_at_time_single_record():
    """PR D: a single-record (ntime==1) forcing — e.g. a climatological mean or
    a 2D file promoted to shape (1, ...) — must return the single SST/SIC field
    for ANY day. The cyclic-wrap + searchsorted clip(0, ntime-2)=clip(0, -1)
    path produced idx=-1, dt=0, a blown-up weight, and ~0 K SST for day !=
    times[0]."""
    from legoesm.forcing.amip import AMIPForcing

    sst = jnp.full((1, 4, 8), 290.0)
    sic = jnp.full((1, 4, 8), 0.3)
    forcing = AMIPForcing(
        times=jnp.array([0.0]), sst=sst, sic=sic, config=AMIPForcingConfig(),
    )
    for day in (0.0, 100.0, 365.0):
        s, i = get_forcing_at_time(forcing, float(day))
        np.testing.assert_allclose(np.asarray(s), 290.0, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(i), 0.3, rtol=1e-6)


def test_get_forcing_at_time_preserves_field_dtype():
    """The float64 time axis (needed for multi-decade days-since-epoch
    arithmetic) must not promote a float32-policy SST/SIC field to float64
    through the interpolation weight — get_forcing_at_time returns the STORED
    field dtype. (Non-vacuous only under JAX x64, which the suite enables.)"""
    from legoesm.forcing.amip import AMIPForcing

    times = jnp.asarray([0.0, 30.0, 60.0], dtype=jnp.float64)   # float64 axis
    sst = jnp.full((3, 2, 2), 290.0, dtype=jnp.float32)         # float32 field
    sic = jnp.full((3, 2, 2), 0.1, dtype=jnp.float32)
    forcing = AMIPForcing(
        times=times, sst=sst, sic=sic, config=AMIPForcingConfig(T_ice=271.35),
    )
    s, i = get_forcing_at_time(forcing, 15.0)   # interior interp (float64 weight)
    assert s.dtype == sst.dtype and i.dtype == sic.dtype
