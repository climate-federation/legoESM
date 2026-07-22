"""Tests for scripts/data/build_sftlf_from_surfdata.py.

Covers: 1-D extraction from a synthetic regular CLM-style surfdata file,
clipping of the CLM float overshoot, irregular-grid rejection, and the
FULL consumer path — the built file round-trips through
``legoesm.grids.topography._load_land_fraction_file`` and a synthetic
below-sea-level inland-sea cell (the Caspian scenario) lands as LAND,
while the elevation>0 rule on the same synthetic topography classifies it
as ocean (the refuting fixture that motivated the mask).

A Levante-only regression on the REAL staged CLM surfdata asserts
Caspian/Aral f_land = 1.0 end-to-end (skipped where the asset is absent).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "build_sftlf_from_surfdata",
    REPO / "scripts" / "data" / "build_sftlf_from_surfdata.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_sftlf = _mod.build_sftlf

REAL_SURFDATA = REPO / "data" / "clm" / "surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc"


def _write_synthetic_surfdata(path, frac, lat, lon, irregular=False):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("lsmlat", lat.size)
        ds.createDimension("lsmlon", lon.size)
        lat2d = np.broadcast_to(lat[:, None], (lat.size, lon.size)).copy()
        lon2d = np.broadcast_to(lon[None, :], (lat.size, lon.size)).copy()
        if irregular:
            lat2d[0, 0] += 1.0  # break regularity
        v = ds.createVariable("LATIXY", "f8", ("lsmlat", "lsmlon")); v[:] = lat2d
        v = ds.createVariable("LONGXY", "f8", ("lsmlat", "lsmlon")); v[:] = lon2d
        v = ds.createVariable("LANDFRAC_PFT", "f8", ("lsmlat", "lsmlon"))
        v[:] = frac


class TestBuildSftlf:

    def _synthetic(self, tmp_path, irregular=False):
        # 5x8 regular grid; a "Caspian" land cell at (40N, 50E) with
        # fraction slightly OVER 1 (the CLM overshoot), ocean elsewhere
        # in the middle band.
        lat = np.array([-60.0, -20.0, 0.0, 40.0, 60.0])
        lon = np.arange(0.0, 360.0, 45.0)
        frac = np.zeros((5, 8))
        frac[3, 1] = 1.0000000193      # "Caspian" (40N, 45E col) — land
        frac[4, :] = 1.0               # a land band
        src = tmp_path / "surfdata.nc"
        _write_synthetic_surfdata(src, frac, lat, lon, irregular=irregular)
        return src, lat, lon

    def test_extract_and_clip(self, tmp_path):
        src, lat, lon = self._synthetic(tmp_path)
        out = tmp_path / "sftlf.nc"
        build_sftlf(str(src), str(out))
        with netCDF4.Dataset(out) as ds:
            sftlf = np.asarray(ds.variables["sftlf"][:])
            assert np.asarray(ds.variables["lat"][:]).shape == lat.shape
            assert np.asarray(ds.variables["lon"][:]).shape == lon.shape
        assert sftlf.max() <= 1.0          # overshoot clipped
        assert sftlf[3, 1] == 1.0
        assert sftlf[1, 4] == 0.0

    def test_irregular_grid_rejected(self, tmp_path):
        src, _, _ = self._synthetic(tmp_path, irregular=True)
        with pytest.raises(ValueError, match="regular"):
            build_sftlf(str(src), str(tmp_path / "out.nc"))

    def test_missing_variable_raises(self, tmp_path):
        src, _, _ = self._synthetic(tmp_path)
        with pytest.raises(KeyError, match="NOPE"):
            build_sftlf(str(src), str(tmp_path / "out.nc"), var_name="NOPE")

    def test_roundtrip_through_loader_fixes_inland_sea(self, tmp_path):
        """Full consumer path: the built sftlf feeds
        _load_land_fraction_file and the synthetic Caspian cell is LAND,
        while the elevation>0 rule on matching synthetic topography says
        ocean (the refuting fixture)."""
        from legoesm.grids.topography import (
            _load_land_fraction_file, land_mask_from_topography)

        src, lat, lon = self._synthetic(tmp_path)
        out = tmp_path / "sftlf.nc"
        build_sftlf(str(src), str(out))

        # Target grid = the source points themselves (loader interpolates).
        tgt_lat = np.broadcast_to(lat[:, None], (5, 8))
        tgt_lon = np.broadcast_to(lon[None, :], (5, 8))
        f_land = _load_land_fraction_file(str(out), "", tgt_lat, tgt_lon)
        assert f_land.shape == tgt_lat.shape
        assert f_land[3, 1] >= 0.5, "Caspian-like cell must be land via mask"

        # Refuting fixture: below-sea-level basin under the elevation rule.
        z_s = np.full((5, 8), 300.0)
        z_s[3, 1] = -28.0              # Caspian surface elevation
        mask_elev = np.asarray(land_mask_from_topography(z_s))
        assert mask_elev[3, 1] < 0.5, (
            "elevation>0 rule classifies the below-sea-level basin as "
            "ocean — the bug this mask fixes; if this starts passing, "
            "the rule changed and the mask may be redundant")

    @pytest.mark.skipif(not REAL_SURFDATA.exists(),
                        reason="staged CLM surfdata not present")
    def test_real_surfdata_caspian_aral_land(self, tmp_path):
        out = tmp_path / "sftlf_real.nc"
        info = build_sftlf(str(REAL_SURFDATA), str(out))
        assert info["caspian_42N_51E"] >= 0.99
        assert info["aral_45N_60E"] >= 0.99
        assert info["mid_atlantic_30N_40W"] == 0.0

        # End-to-end through the loader on a C24-like 1° probe grid
        # around the Caspian box.
        from legoesm.grids.topography import _load_land_fraction_file
        glat, glon = np.meshgrid(np.arange(36.0, 52.0, 1.0),
                                 np.arange(46.0, 62.0, 1.0), indexing="ij")
        f_land = _load_land_fraction_file(str(out), "", glat, glon)
        assert float(f_land.mean()) > 0.9
