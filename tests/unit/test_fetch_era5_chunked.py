"""Network-free test of the chunked ERA5 fetch's merge assembly.

The fetch (network) path streams ARCO-ERA5; the mergeable/testable part is
``merge_partials`` — it must stack 12 monthly (nh, nlat, nlon) partials onto a leading
month axis and carry the static geopotential/land-sea-mask + lat/lon/hours, producing the
exact key set the model loader (`load_training_data`) expects, with NO network."""
import importlib.util
import pathlib

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "fetch_era5_chunked",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "data" / "fetch_era5_chunked.py")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def _write_partials(dir_, nh=4, nlat=6, nlon=8):
    """12 synthetic month partials; each month's fields are filled with the month index
    so the merge's month ordering is verifiable."""
    lat = np.linspace(90, -90, nlat); lon = np.linspace(0, 360, nlon, endpoint=False)
    lsm = (np.arange(nlat * nlon).reshape(nlat, nlon) % 2).astype(float)
    elev = np.ones((nlat, nlon)) * 100.0
    hours = np.arange(nh, dtype=float)
    for m in range(1, 13):
        out = {v: np.full((nh, nlat, nlon), float(m)) for v in _MOD._OUT_VARS}
        out.update(elev_m=elev, lsm=lsm, lat=lat, lon=lon, hours=hours)
        np.savez(f"{dir_}/m{m:02d}.npz", **out)


def test_merge_shapes_and_keys(tmp_path):
    _write_partials(tmp_path, nh=4, nlat=6, nlon=8)
    out = _MOD.merge_partials(str(tmp_path))
    # every model-forcing var is (12, nh, nlat, nlon)
    for v in _MOD._OUT_VARS:
        assert out[v].shape == (12, 4, 6, 8), v
    # static fields carried from month 01, correct shapes
    assert out["lsm"].shape == (6, 8) and out["elev_m"].shape == (6, 8)
    assert out["lat"].shape == (6,) and out["lon"].shape == (8,)
    assert out["hours"].tolist() == [0, 1, 2, 3]
    # month axis is ordered 1..12 (each month filled with its index)
    for m in range(12):
        assert np.all(out["2m_temperature"][m] == float(m + 1))


def test_merge_missing_month_raises(tmp_path):
    _write_partials(tmp_path)
    (tmp_path / "m07.npz").unlink()
    with pytest.raises(FileNotFoundError):
        _MOD.merge_partials(str(tmp_path))
