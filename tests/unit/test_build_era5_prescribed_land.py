"""scripts/data/build_era5_prescribed_land.py: ERA5 sign/unit conversion and
the file contract the prescribed-land reader expects (cdo stubbed out)."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from scripts.data import build_era5_prescribed_land as b

_LAT = np.arange(-89.5, 90.0, 1.0)
_LON = np.arange(0.5, 360.0, 1.0)
# ERA5-convention January-like values: accumulations positive DOWNWARD [J m-2/day]
_ERA5 = {235: 286.0, 243: 0.17, 146: -15.0 * 86400.0, 147: -80.0 * 86400.0}


def _fake_ymonmean(sign=1.0):
    def f(kind, code, years, grid, out):
        assert grid == "r360x180" and list(years) == list(range(1979, 1981))
        v = _ERA5[code] * (sign if code in (146, 147) else 1.0)
        arr = np.full((12, _LAT.size, _LON.size), v)
        xr.Dataset({f"var{code}": (("time", "lat", "lon"), arr)},
                   coords={"time": np.arange(12), "lat": _LAT, "lon": _LON}).to_netcdf(out)
    return f


def test_build_writes_the_reader_contract_with_fluxes_positive_up(tmp_path, monkeypatch):
    from legoesm.forcing.prescribed_land import load_prescribed_land
    monkeypatch.setattr(b, "_ymonmean", _fake_ymonmean())
    out = tmp_path / "presc.nc"
    b.main(["--years", "1979-1980", "--out", str(out)])
    with xr.open_dataset(out, decode_times=False) as ds:
        np.testing.assert_allclose(ds["hfss"].values, 15.0)   # up, W m-2
        np.testing.assert_allclose(ds["hfls"].values, 80.0)
        np.testing.assert_allclose(ds["ts"].values, 286.0)
        np.testing.assert_allclose(ds["time"].values[:3], [15.5, 45.0, 74.5])
        assert ds["hfss"].attrs["positive"] == "up"

    class _G:
        grid_lat = np.deg2rad(np.array([0.0, 45.0]))
        grid_lon = np.deg2rad(np.array([0.0, 180.0]))
    pl = load_prescribed_land(str(out), _G())
    np.testing.assert_allclose(pl.hfls, 80.0)
    np.testing.assert_allclose(pl.albedo, 0.17)


def test_build_refuses_an_unflipped_sign(tmp_path, monkeypatch):
    """If the accumulations were already upward, the conversion would make the
    fluxes downward: the plausibility window refuses the file."""
    monkeypatch.setattr(b, "_ymonmean", _fake_ymonmean(sign=-1.0))
    with pytest.raises(ValueError, match="plausible range"):
        b.main(["--years", "1979-1980", "--out", str(tmp_path / "x.nc")])
    assert not (tmp_path / "x.nc").exists()


def test_era5_paths_follow_the_pool_layout():
    p = b._era5_file("fc", 146, 1979)
    assert str(p).endswith("sf/fc/1M/146/E5sf12_1M_1979_146.grb")
    assert str(b._era5_file("an", 235, 1990)).endswith("sf/an/1M/235/E5sf00_1M_1990_235.grb")
