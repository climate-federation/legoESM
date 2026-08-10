"""Tests for OMIP-2 protocol-compliance I/O additions.

Covers:
  * WOA18 IC loader — bilinear interp, monthly climatology, bathymetry
    mask, and named-constant fill values.
  * CMOR NetCDF writer — expanded ``Omon`` and new ``SImon`` tables for
    OMIP-2 core ocean and sea-ice prognostic variables.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Skip the whole module when xarray / netCDF4 are unavailable.  These I/O
# pieces have no meaningful tests without on-disk NetCDF round-trips.
# ---------------------------------------------------------------------------
xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")


# ===========================================================================
# WOA bilinear horizontal interpolation
# ===========================================================================

class TestWOABilinearInterp:
    def test_constant_field_recovered(self):
        from legoesm.ocean.init_woa import _bilinear_2d
        src_lat = np.linspace(-89.5, 89.5, 180)
        src_lon = np.linspace(0.5, 359.5, 360)
        field = np.full((180, 360), 5.0)
        tlat = np.array([10.0, -45.0, 0.0])
        tlon = np.array([200.0, 150.0, 0.0])
        out = _bilinear_2d(tlat, tlon, src_lat, src_lon, field)
        assert np.allclose(out, 5.0)

    def test_linear_field_matches_analytic(self):
        """Source = T(lat, lon) = lat + 0.1·lon.  Bilinear of a linear
        function is exact away from the wrap edge."""
        from legoesm.ocean.init_woa import _bilinear_2d
        src_lat = np.linspace(-89.5, 89.5, 180)
        src_lon = np.linspace(0.5, 359.5, 360)
        LAT, LON = np.meshgrid(src_lat, src_lon, indexing="ij")
        field = LAT + 0.1 * LON
        # Pick targets well inside the lat-lon source range so no wrap
        tlat = np.array([12.3, -34.6, 5.7])
        tlon = np.array([200.4, 150.7, 89.9])
        out = _bilinear_2d(tlat, tlon, src_lat, src_lon, field)
        expected = tlat + 0.1 * tlon
        assert np.allclose(out, expected, rtol=1e-10, atol=1e-8)

    def test_periodic_wrap(self):
        """Target between src_lon[-1] and src_lon[0]+360 must use the
        periodic wrap."""
        from legoesm.ocean.init_woa import _bilinear_2d
        src_lat = np.linspace(-89.5, 89.5, 180)
        src_lon = np.linspace(0.5, 359.5, 360)  # last cell at 359.5
        field = np.zeros((180, 360))
        # Make a step at the wrap so we can see the interpolation
        field[:, 0] = 10.0      # at lon=0.5
        field[:, -1] = 20.0     # at lon=359.5
        # Target at lon=0.0 — exactly between 359.5 (=20) and 0.5 (=10)
        out = _bilinear_2d(
            np.array([0.0]),
            np.array([0.0]),
            src_lat, src_lon, field,
        )
        # Midpoint → 15.0 ± rounding
        assert out.shape == (1,)
        assert 13.0 < float(out[0]) < 17.0

    def test_nan_inputs_reweighted(self):
        """One NaN corner: bilinear should renormalise over remaining."""
        from legoesm.ocean.init_woa import _bilinear_2d
        src_lat = np.array([0.0, 1.0])
        src_lon = np.array([0.0, 1.0])
        field = np.array([[1.0, 2.0], [np.nan, 4.0]])
        # Midpoint
        out = _bilinear_2d(
            np.array([0.5]),
            np.array([0.5]),
            src_lat, src_lon, field,
        )
        # Equal weights 0.25 each.  NaN at (1, 0) drops out → weighted
        # average over remaining 3 corners: (1 + 2 + 4) / 3 = 7/3
        assert float(out[0]) == pytest.approx(7.0 / 3.0, rel=1e-10)

    def test_all_nan_returns_nan(self):
        from legoesm.ocean.init_woa import _bilinear_2d
        src_lat = np.array([0.0, 1.0])
        src_lon = np.array([0.0, 1.0])
        field = np.full((2, 2), np.nan)
        out = _bilinear_2d(
            np.array([0.5]),
            np.array([0.5]),
            src_lat, src_lon, field,
        )
        assert np.isnan(out[0])

    def test_non_monotonic_raises(self):
        from legoesm.ocean.init_woa import _bilinear_2d
        src_lat = np.array([0.0, -1.0, 1.0])  # not monotonic
        src_lon = np.array([0.0, 1.0])
        field = np.zeros((3, 2))
        with pytest.raises(ValueError, match="monotonic"):
            _bilinear_2d(
                np.array([0.5]), np.array([0.5]),
                src_lat, src_lon, field,
            )


# ===========================================================================
# WOA monthly climatology + bathymetry mask
# ===========================================================================

def _write_synthetic_woa(tmp_path, t_time=1, s_time=None):
    """Write minimal WOA-style T/S NetCDF files to ``tmp_path``.

    Variables: ``t_an`` / ``s_an`` with shape ``(time, depth, lat, lon)``.
    ``s_time`` defaults to ``t_time`` to keep T and S consistent.
    """
    if s_time is None:
        s_time = t_time
    lat = np.linspace(-89.5, 89.5, 18)   # coarse 10-deg
    lon = np.linspace(0.5, 359.5, 36)
    depth = np.array([0.0, 100.0, 500.0, 1000.0, 2000.0])

    def _make(time):
        T = np.zeros((time, depth.size, lat.size, lon.size))
        S = np.full((time, depth.size, lat.size, lon.size), 35.0)
        for t in range(time):
            # Temperature: warm at surface, cold at depth, lat-dependent
            for k, z in enumerate(depth):
                T[t, k, :, :] = (
                    25.0 * np.cos(np.radians(lat))[:, None]
                    * np.exp(-z / 1000.0)
                    + 0.5 * t   # monthly drift
                )
            # Salinity: simple latitudinal pattern
            S[t, :, :, :] = (
                35.0 + 0.5 * np.sin(np.radians(lat))[None, :, None]
            )
        return T, S

    T_data, _ = _make(t_time)
    _, S_data = _make(s_time)

    T_path = tmp_path / "woa_T.nc"
    S_path = tmp_path / "woa_S.nc"

    xr.Dataset(
        {"t_an": (("time", "depth", "lat", "lon"), T_data)},
        coords={
            "time": np.arange(t_time, dtype=np.float64),
            "depth": depth,
            "lat": lat,
            "lon": lon,
        },
    ).to_netcdf(T_path)
    xr.Dataset(
        {"s_an": (("time", "depth", "lat", "lon"), S_data)},
        coords={
            "time": np.arange(s_time, dtype=np.float64),
            "depth": depth,
            "lat": lat,
            "lon": lon,
        },
    ).to_netcdf(S_path)
    return T_path, S_path, lat, lon, depth


class TestWOAMonthlyAndBathymetry:
    def test_load_woa18_annual(self, tmp_path):
        from legoesm.ocean.init_woa import load_woa18
        T_path, S_path, lat, lon, depth = _write_synthetic_woa(tmp_path)
        T, S, lat_w, lon_w, depth_w = load_woa18(T_path, S_path)
        assert T.shape == (lat.size, lon.size, depth.size)
        # The file's OWN depth axis must come back, not WOA_DEPTHS:
        # that is what lets a non-WOA climatology be read correctly.
        assert np.allclose(depth_w, depth)
        assert S.shape == (lat.size, lon.size, depth.size)
        assert np.allclose(lat_w, lat)
        assert np.allclose(lon_w, lon)

    def test_load_woa18_monthly_index(self, tmp_path):
        from legoesm.ocean.init_woa import load_woa18
        T_path, S_path, *_ = _write_synthetic_woa(
            tmp_path, t_time=12, s_time=12,
        )
        T_jan, _, _, _, _ = load_woa18(T_path, S_path, month=1)
        T_jul, _, _, _, _ = load_woa18(T_path, S_path, month=7)
        # Monthly drift built into synthetic data: t=0 vs t=6 → +3 K
        assert float(np.mean(T_jul - T_jan)) == pytest.approx(3.0, rel=1e-10)

    def test_load_woa18_single_file_per_month(self, tmp_path):
        """Per-month layout: caller pre-selects file matching the
        target month; loader reads index 0 with ``month=None``."""
        from legoesm.ocean.init_woa import load_woa18
        T_path, S_path, *_ = _write_synthetic_woa(tmp_path, t_time=1)
        T_data, _, _, _, _ = load_woa18(
            T_path, S_path,
            monthly_layout="single_file_per_month",
            month=None,
        )
        assert T_data.shape[-1] > 0  # smoke

    def test_load_woa18_single_file_per_month_rejects_month(self, tmp_path):
        """Per-month layout with month=N must raise (silent-mismatch
        guard added in codex iter)."""
        from legoesm.ocean.init_woa import load_woa18
        T_path, S_path, *_ = _write_synthetic_woa(tmp_path, t_time=1)
        with pytest.raises(ValueError, match="single_file_per_month"):
            load_woa18(
                T_path, S_path,
                monthly_layout="single_file_per_month",
                month=3,
            )

    def test_load_woa18_concatenated_short_raises(self, tmp_path):
        """Concatenated layout with month past axis length must raise
        (no silent fallback to index 0)."""
        from legoesm.ocean.init_woa import load_woa18
        T_path, S_path, *_ = _write_synthetic_woa(tmp_path, t_time=1)
        with pytest.raises(IndexError, match="time length"):
            load_woa18(T_path, S_path, month=7)

    def test_load_woa18_multimonth_index_out_of_range(self, tmp_path):
        """Multi-month file shorter than 12 should error if the
        requested month index is past the end."""
        from legoesm.ocean.init_woa import load_woa18
        T_path, S_path, *_ = _write_synthetic_woa(tmp_path, t_time=3)
        with pytest.raises(IndexError, match="time length"):
            load_woa18(T_path, S_path, month=5)

    def test_load_woa18_bad_month(self, tmp_path):
        from legoesm.ocean.init_woa import load_woa18
        T_path, S_path, *_ = _write_synthetic_woa(tmp_path)
        with pytest.raises(ValueError, match="month must be"):
            load_woa18(T_path, S_path, month=0)
        with pytest.raises(ValueError, match="month must be"):
            load_woa18(T_path, S_path, month=13)

    def test_init_ocean_bathymetry_invalid(self, tmp_path):
        from legoesm.ocean.init_woa import init_ocean_from_woa
        from legoesm.ocean.vertical import create_ocean_z_star

        T_path, S_path, *_ = _write_synthetic_woa(tmp_path)
        z_coord = create_ocean_z_star(n_levels=20, H_max=4000.0)

        class _G:
            lat = np.radians(np.array([[0.0, 0.0], [10.0, 10.0]]))
            lon = np.radians(np.array([[10.0, 20.0], [10.0, 20.0]]))
        grid = _G()

        # Negative depth → ValueError
        with pytest.raises(ValueError, match="strictly positive"):
            init_ocean_from_woa(
                grid, z_coord, T_path, S_path,
                bathymetry_depth=np.full((2, 2), -100.0),
            )
        # NaN → ValueError
        with pytest.raises(ValueError, match="non-finite"):
            init_ocean_from_woa(
                grid, z_coord, T_path, S_path,
                bathymetry_depth=np.array([[1000.0, np.nan], [1000.0, 1000.0]]),
            )

    def test_woa_negative_lon_axis_supported(self):
        """`[-180, 180)` source longitudes should be sortable, not
        rejected by the monotonicity check."""
        from legoesm.ocean.init_woa import _bilinear_2d
        src_lat = np.linspace(-89.5, 89.5, 18)
        src_lon = np.linspace(-179.5, 179.5, 36)  # spans 0
        LAT, LON = np.meshgrid(src_lat, src_lon, indexing="ij")
        field = LAT + 0.1 * LON  # smooth linear
        # Target in the eastern hemisphere
        out = _bilinear_2d(
            np.array([0.0]), np.array([45.0]),
            src_lat, src_lon, field,
        )
        # Linear field: T(0, 45) = 0 + 4.5 = 4.5
        assert float(out[0]) == pytest.approx(4.5, abs=0.5)

    def test_init_ocean_uses_the_files_own_depth_axis(self, tmp_path):
        """A climatology on NON-WOA levels must be read at ITS depths.

        The loader used to take the source depth axis from
        ``WOA_DEPTHS[:n_depth]`` regardless of what the file said, so a file
        on e.g. PHC3's 33 levels had its abyssal values read as if they sat in
        the top few hundred metres — a silently wrong ocean, never an error.

        The discriminator here is a source whose levels are FAR from
        ``WOA_DEPTHS[:5]`` (0, 5, 10, 15, 20 m): a two-level file at 0 m and
        3000 m, warm on top and cold at the bottom.  Read correctly, the model
        column is warm through the upper ocean and only cools towards 3000 m.
        Read against ``WOA_DEPTHS``, the whole column below 20 m would be the
        3000 m value.
        """
        from legoesm.ocean.init_woa import WOA_DEPTHS, init_ocean_from_woa
        from legoesm.ocean.vertical import create_ocean_z_star

        lat = np.linspace(-45.0, 45.0, 4)
        lon = np.linspace(0.0, 270.0, 4)
        depth = np.array([0.0, 3000.0])
        T = np.zeros((1, depth.size, lat.size, lon.size))
        T[0, 0] = 20.0          # surface
        T[0, 1] = 0.0           # 3000 m
        S = np.full((1, depth.size, lat.size, lon.size), 35.0)
        coords = {"time": np.array([0.0]), "depth": depth,
                  "lat": lat, "lon": lon}
        T_path = tmp_path / "src_T.nc"
        S_path = tmp_path / "src_S.nc"
        xr.Dataset({"t_an": (("time", "depth", "lat", "lon"), T)},
                   coords=coords).to_netcdf(T_path)
        xr.Dataset({"s_an": (("time", "depth", "lat", "lon"), S)},
                   coords=coords).to_netcdf(S_path)

        # Sanity: the discriminator only works if the file's axis and the
        # WOA prefix really disagree.
        assert not np.allclose(depth, WOA_DEPTHS[:depth.size])

        z_coord = create_ocean_z_star(n_levels=20, H_max=3000.0)

        class _G:
            lat = np.radians(np.array([[0.0, 0.0], [10.0, 10.0]]))
            lon = np.radians(np.array([[10.0, 20.0], [10.0, 20.0]]))

        T_out, _ = init_ocean_from_woa(_G(), z_coord, T_path, S_path,
                                       interp="bilinear")
        col = np.asarray(T_out)[0, 0]
        model_depths = np.abs(np.asarray(z_coord.z_full_ref))

        # Linear in depth between the two source levels, so the model level
        # nearest 1500 m must sit near 10 degC — the old behaviour put it at 0.
        k_mid = int(np.argmin(np.abs(model_depths - 1500.0)))
        expected = 20.0 * (1.0 - model_depths[k_mid] / 3000.0)
        assert abs(col[k_mid] - expected) < 0.5, (
            f"depth {model_depths[k_mid]:.0f} m -> {col[k_mid]:.3f} degC, "
            f"expected ~{expected:.3f}")
        # And the top of the column must still be the surface value.
        assert abs(col[0] - 20.0) < 0.5

    def test_init_ocean_bathymetry_mask(self, tmp_path):
        """Cells deeper than bathymetry get the named-constant fill."""
        from legoesm import constants
        from legoesm.ocean.init_woa import init_ocean_from_woa
        from legoesm.ocean.vertical import create_ocean_z_star

        T_path, S_path, lat_w, lon_w, _ = _write_synthetic_woa(tmp_path)
        z_coord = create_ocean_z_star(n_levels=20, H_max=4000.0)

        # Mock a tiny lat-lon grid via a duck-typed namespace
        class _G:
            lat = np.radians(np.array([[0.0, 0.0], [10.0, 10.0]]))
            lon = np.radians(np.array([[10.0, 20.0], [10.0, 20.0]]))
        grid = _G()

        # Bathymetry: 500 m at point [0, 0], 4000 m elsewhere
        bath = np.array([[500.0, 4000.0], [4000.0, 4000.0]])
        T, S = init_ocean_from_woa(
            grid, z_coord, T_path, S_path,
            interp="bilinear", bathymetry_depth=bath,
        )
        T_np = np.asarray(T)
        S_np = np.asarray(S)
        model_depths = np.abs(np.asarray(z_coord.z_full_ref))

        # At the shallow point, all model levels below 500 m must equal
        # the named-constant fill.
        below = model_depths > 500.0
        if below.any():
            assert np.allclose(
                T_np[0, 0, below], constants.T_deep_ocean_ref_C, atol=1e-10,
            )
            assert np.allclose(
                S_np[0, 0, below], constants.S_deep_ocean_ref_psu, atol=1e-10,
            )

        # At deep points the surface cell should NOT equal the fill
        # (the synthetic WOA T is positive at surface, so 1.5°C fill
        # would only appear by accident).
        assert not np.isclose(
            T_np[1, 1, 0], constants.T_deep_ocean_ref_C, atol=0.01,
        )


# ===========================================================================
# CMOR Omon expansion + SImon table
# ===========================================================================

class TestCMORTablesOMIP2:
    def test_omon_core_vars_registered(self):
        from legoesm.io.cmor_output import CMOR_TABLES
        omon = CMOR_TABLES["Omon"]
        # OMIP-2 core variables (Griffies+ 2016 GMD)
        required = {
            "tos", "sos", "zos", "mlotst",
            "hfds", "wfo", "tauuo", "tauvo",
            "thetao", "so", "uo", "vo", "wo",
        }
        missing = required - set(omon.keys())
        assert not missing, f"Omon missing OMIP-2 vars: {missing}"

    def test_omon_3d_vars_use_depth(self):
        from legoesm.io.cmor_output import CMOR_TABLES
        omon = CMOR_TABLES["Omon"]
        for v in ("thetao", "so", "uo", "vo", "wo"):
            assert "depth" in omon[v]["dimensions"], v

    def test_simon_table_registered(self):
        from legoesm.io.cmor_output import CMOR_TABLES
        assert "SImon" in CMOR_TABLES
        simon = CMOR_TABLES["SImon"]
        required = {"siconc", "sithick", "siu", "siv", "sitemptop"}
        missing = required - set(simon.keys())
        assert not missing, f"SImon missing vars: {missing}"

    def test_simon_realm_is_seaice(self):
        from legoesm.io.cmor_output import table_realm
        assert table_realm("SImon") == "seaIce"

    def test_lookup_cmor_entry_finds_thetao(self):
        from legoesm.io.cmor_output import lookup_cmor_entry
        table, entry = lookup_cmor_entry("thetao")
        assert table == "Omon"
        assert entry["standard_name"] == "sea_water_potential_temperature"


# ===========================================================================
# CMOR depth-axis ocean-vs-soil dispatch
# ===========================================================================

class TestCMORDepthAxis:
    def test_make_depth_ocean_attrs(self):
        from legoesm.io.cmor_output import _make_depth_da
        da = _make_depth_da(np.array([10.0, 100.0]), kind="ocean")
        assert da.attrs["long_name"] == "Ocean Depth"
        assert da.attrs["positive"] == "down"
        assert da.attrs["standard_name"] == "depth"

    def test_make_depth_soil_attrs(self):
        from legoesm.io.cmor_output import _make_depth_da
        da = _make_depth_da(np.array([0.1, 0.5]), kind="soil")
        assert "Land Surface" in da.attrs["long_name"]

    def test_make_depth_invalid_kind(self):
        from legoesm.io.cmor_output import _make_depth_da
        with pytest.raises(ValueError, match="kind must be"):
            _make_depth_da(np.array([1.0]), kind="atmos")


# ===========================================================================
# CFWriter end-to-end: 3-D ``thetao`` write produces a CF/NetCDF file
# with the expected ocean-depth coord
# ===========================================================================

class TestCFWriterOmonThetao:
    def test_writes_thetao_file(self, tmp_path):
        from legoesm.io.cmor_output import CFWriter
        writer = CFWriter(
            output_dir=str(tmp_path),
            experiment_id="omip2",
            model_id="legoESM-test",
            freq="mon",
            calendar="noleap",
            ref_date="0001-01-01",
        )
        nlat, nlon, nlev = 4, 8, 3
        lat = np.linspace(-60, 60, nlat)
        lon = np.linspace(0, 315, nlon)
        depth = np.array([10.0, 100.0, 1000.0])
        data = 285.0 + 0.01 * np.arange(nlev * nlat * nlon).reshape(nlev, nlat, nlon)
        path = writer.write_field(
            var_name="thetao",
            data=data,
            time=15.0,
            time_bounds=(0.0, 30.0),
            lat=lat,
            lon=lon,
            depth=depth,
        )
        writer.close()
        assert path.exists()
        ds = xr.open_dataset(path)
        try:
            assert "thetao" in ds.data_vars
            assert "depth" in ds.coords
            assert ds["depth"].attrs["positive"] == "down"
            assert "Ocean Depth" in ds["depth"].attrs["long_name"]
            # CMIP6 Omon thetao is in degC (not K) — codex review fix.
            assert ds["thetao"].attrs["units"] == "degC"
            assert (
                ds["thetao"].attrs["standard_name"]
                == "sea_water_potential_temperature"
            )
            # Shape: (time=1, depth=3, lat=4, lon=8)
            assert ds["thetao"].shape == (1, nlev, nlat, nlon)
        finally:
            ds.close()


class TestClimatologyDepthAxis:
    """Codex round-4: the SALINITY file's depth axis must be checked too."""

    @staticmethod
    def _write(tmp_path, t_depth, s_depth):
        lat = np.linspace(-45.0, 45.0, 3)
        lon = np.linspace(0.0, 240.0, 3)

        def _one(path, var, depth):
            data = np.zeros((1, depth.size, lat.size, lon.size))
            xr.Dataset(
                {var: (("time", "depth", "lat", "lon"), data)},
                coords={"time": np.array([0.0]), "depth": depth,
                        "lat": lat, "lon": lon},
            ).to_netcdf(path)

        tp, sp = tmp_path / "t.nc", tmp_path / "s.nc"
        _one(tp, "t_an", t_depth)
        _one(sp, "s_an", s_depth)
        return tp, sp

    def test_mismatched_t_and_s_depth_axes_raise(self, tmp_path):
        """Equal level COUNTS with different depths would place salinity at
        the temperature file's depths without a word."""
        from legoesm.ocean.init_woa import load_woa18

        tp, sp = self._write(tmp_path, np.array([0.0, 100.0, 1000.0]),
                             np.array([0.0, 200.0, 3000.0]))
        with pytest.raises(ValueError, match="DIFFERENT"):
            load_woa18(tp, sp)

    def test_matching_axes_are_accepted(self, tmp_path):
        from legoesm.ocean.init_woa import load_woa18

        depth = np.array([0.0, 100.0, 1000.0])
        tp, sp = self._write(tmp_path, depth, depth)
        *_, depth_w = load_woa18(tp, sp)
        assert np.allclose(depth_w, depth)
