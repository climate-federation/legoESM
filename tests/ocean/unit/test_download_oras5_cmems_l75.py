"""Direct test of ``scripts/data/download_oras5_cmems_l75.py`` (no network).

Synthetic 0.25-degree month through the real S2S remap: channel order, the model-grid alignment
(row 0 = 89.5S, column i centred on i degrees E), NaN on land, and the day-file validator.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
REPO = Path(__file__).resolve().parents[3]
S2S = Path("/work/bd1083/b309178/diffESM/S2S_model")
if not (S2S / "s2s" / "cmems.py").exists():
    pytest.skip("S2S_model checkout (s2s.cmems remap) not available", allow_module_level=True)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(S2S))
from s2s.cmems import _block_index, _remap  # noqa: E402

from scripts.data import download_oras5_cmems_l75 as dl  # noqa: E402

LAT = np.arange(-80.0, 90.001, 0.25)
LON = np.arange(-180.0, 180.0, 0.25)


def _month():
    nz = dl.NLEV
    lon2 = np.broadcast_to(LON, (LAT.size, LON.size))
    lat2 = np.broadcast_to(LAT[:, None], (LAT.size, LON.size))
    land = (lat2 > 10) & (lat2 < 20) & (lon2 > 30) & (lon2 < 40)       # a 10x10 deg island
    ds3, ds2 = {}, {}
    for n, v in enumerate(dl.VARS_3D):
        base = {"thetao": np.mod(lon2, 360.0), "so": lat2}.get(v, np.full(lon2.shape, float(n)))
        x = base[None, None] + 1000.0 * np.arange(nz)[None, :, None, None] + np.zeros((2, nz, 1, 1))
        x = np.where(land, np.nan, x)
        ds3[v] = xr.Dataset({f"{v}_oras": (("time", "depth", "latitude", "longitude"), x)},
                            coords={"time": np.array(["2000-01-01", "2000-01-02"], "M8[ns]"),
                                    "depth": np.arange(nz, dtype=float), "latitude": LAT,
                                    "longitude": LON})
    for n, v in enumerate(dl.VARS_2D):
        x = np.where(land, np.nan, np.full((2,) + lon2.shape, 10.0 + n))
        ds2[v] = xr.Dataset({f"{v}_oras": (("time", "latitude", "longitude"), x)},
                            coords={"time": ds3["thetao"].time, "latitude": LAT, "longitude": LON})
    return ds3, ds2


def test_day_stack_layout_and_alignment():
    ds3, ds2 = _month()
    idx, w = _block_index(LAT, LON, dl.NLAT, dl.NLON, dl.LAT0, dl.LON0)
    a = dl.day_stack(ds3, ds2, 1, idx, w, _remap)
    assert a.shape == (304, 180, 360) and a.dtype == np.float32
    t = a[:75]                                    # thetao block: lon + 1000*k
    # column i collects the points at i-0.5, i-0.25, i, i+0.25 (east longitude)
    assert np.allclose(t[0, 100, 200], 200 - 0.125, atol=1e-4)
    assert np.allclose(t[7, 100, 200], 7000 + 200 - 0.125, atol=1e-3)
    s = a[75:150]                                 # so block: lat; row 100 = 10.5N
    assert np.allclose(s[0, 100, 200], 10.5 - 0.125, atol=1e-3)
    assert np.allclose(a[150, 100, 200], 2.0) and np.allclose(a[225, 100, 200], 3.0)
    assert np.allclose(a[300:, 100, 200], [10, 11, 12, 13])
    assert np.isnan(a[:, 105, 35]).all()          # inside the island
    assert np.isnan(a[:, 5, 200]).all()           # south of 80S: no source
    assert np.isfinite(a[:, 95, 200]).all()
    assert np.allclose(a[150, 100, [0, 359]], 2.0)          # seam columns wrap, both filled
    assert np.allclose(s[0, 100, [0, 359]], 10.5 - 0.125, atol=1e-3)


def test_month_span_and_validator(tmp_path):
    assert dl.month_span(0) == ("1993-01-01", "1993-01-31")
    assert dl.month_span(361) == ("2023-02-01", "2023-02-28")
    assert dl.month_span(37) == ("1996-02-01", "1996-02-29")
    good, bad = tmp_path / "g.npy", tmp_path / "b.npy"
    np.save(good, np.zeros((dl.NCHAN, 180, 360), np.float32))
    np.save(bad, np.zeros((dl.NCHAN, 180, 360), np.float64))
    nan = tmp_path / "n.npy"
    np.save(nan, np.full((dl.NCHAN, 180, 360), np.nan, np.float32))
    assert dl.done_ok(good) and not dl.done_ok(bad) and not dl.done_ok(tmp_path / "none.npy")
    assert not dl.done_ok(nan)
    assert len(dl.channel_names(np.arange(75.0))) == dl.NCHAN
