"""Direct tests for scripts/data/prep_era5_ic_from_zarr.py (offline)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "data" / "prep_era5_ic_from_zarr.py")
_spec = importlib.util.spec_from_file_location("prep_era5_ic_from_zarr", _SCRIPT)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def _synthetic_era5(*, short_names=False, drop_required=False, n_t=4, n_lev=3, n_lat=6, n_lon=8):
    time = pd.date_range("1979-01-01", periods=n_t, freq="6h")
    lev = np.array([250.0, 500.0, 850.0])[:n_lev]
    lat = np.linspace(-80, 80, n_lat)
    lon = np.linspace(0, 315, n_lon)
    pl = ("time", "level", "lat", "lon")
    sf = ("time", "lat", "lon")
    def pa(v): return (pl, np.full((n_t, n_lev, n_lat, n_lon), v, dtype="float32"))
    def sa(v): return (sf, np.full((n_t, n_lat, n_lon), v, dtype="float32"))
    names = (dict(t=pa(250.0), u=pa(10.0), v=pa(0.0), q=pa(1e-3), sp=sa(101325.0))
             if short_names else
             dict(temperature=pa(250.0), u_component_of_wind=pa(10.0),
                  v_component_of_wind=pa(0.0), specific_humidity=pa(1e-3),
                  surface_pressure=sa(101325.0), skin_temperature=sa(290.0)))
    if drop_required:
        names.pop("specific_humidity", None)
        names.pop("q", None)
    return xr.Dataset(names, coords={"time": time, "level": lev, "lat": lat, "lon": lon})


def test_subsets_one_time_with_required_vars():
    snap = prep.subset_era5_ic_snapshot(_synthetic_era5(), year=1979, month=1, day=1, hour=0)
    assert snap.sizes["time"] == 1                              # length-1 time kept
    for v in ("temperature", "u_component_of_wind", "v_component_of_wind",
              "specific_humidity", "surface_pressure"):
        assert v in snap.data_vars
    assert "skin_temperature" in snap.data_vars                 # optional, present -> kept


def test_nearest_time_selection():
    snap = prep.subset_era5_ic_snapshot(_synthetic_era5(), year=1979, month=1, day=1, hour=11)
    # 11:00 is nearest the 12:00 (3rd) step
    assert pd.Timestamp(snap["time"].values[0]) == pd.Timestamp("1979-01-01T12:00:00")


def test_short_ecmwf_aliases_resolve():
    snap = prep.subset_era5_ic_snapshot(_synthetic_era5(short_names=True),
                                        year=1979, month=1, day=1)
    # short-named store still selected via alias resolution (keys are the store's short names)
    assert {"t", "u", "v", "q", "sp"}.issubset(set(snap.data_vars))


def test_missing_required_var_raises():
    with pytest.raises(KeyError):
        prep.subset_era5_ic_snapshot(_synthetic_era5(drop_required=True),
                                     year=1979, month=1, day=1)


def test_no_time_coord_raises():
    ds = xr.Dataset({"temperature": (("lat", "lon"), np.zeros((4, 8)))},
                    coords={"lat": np.linspace(-80, 80, 4), "lon": np.linspace(0, 350, 8)})
    with pytest.raises(KeyError):
        prep.subset_era5_ic_snapshot(ds, year=1979, month=1, day=1)


def test_build_writes_local_zarr_loadable_by_load_era5_ic(tmp_path, monkeypatch):
    """End-to-end: build a local IC zarr (mocked source) and load it via the real
    load_era5_ic, proving the cached snapshot is run-ready.

    The mocked source carries the cloud store's zarr-v2 Blosc ENCODING, so this
    also guards the encoding-strip fix: without it, re-emitting that codec raises
    'Expected a BytesBytesCodec' on write."""
    def _source_with_v2_encoding(_path):
        ds = _synthetic_era5()
        try:
            import numcodecs
            for v in ds.data_vars:
                ds[v].encoding = {"compressor": numcodecs.Blosc()}   # zarr-v2 codec
        except Exception:
            pass
        return ds
    monkeypatch.setattr(prep, "open_era5_store", _source_with_v2_encoding)
    out = tmp_path / "ic.zarr"
    ret = prep.build_era5_ic("ignored://store", str(out), year=1979, month=1, day=1)
    assert ret == str(out) and out.exists()

    from legoesm.training.era5_to_state import load_era5_ic
    # the synthetic source carries no surface geopotential (optional in the
    # builder); an idealized store must say so to load (decision C)
    sl = load_era5_ic(str(out), year=1979, month=1, day=1, allow_flat_phis=True)
    assert np.all(np.isfinite(np.asarray(sl.T)))               # real ERA5Slice produced
    assert np.asarray(sl.p_s).size > 0
