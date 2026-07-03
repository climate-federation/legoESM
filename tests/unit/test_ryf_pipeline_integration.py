"""End-to-end integration test: synthetic IAF -> make_ryf -> cache builder.

The make_ryf and prepare_omip_forcing modules were designed in
isolation. This test catches the join: that the Zarr make_ryf writes
is exactly what build_jra55_cache reads.

Without this, the first time we run the chain on real data we'd find
out (e.g.) that make_ryf's time-axis units string disagrees with what
the cache builder's regex expects.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")
pd = pytest.importorskip("pandas")
# Both integration tests below run the full make_ryf -> build_jra55_cache
# chain; the cache builder writes via ``.chunk(...).to_zarr``, which needs the
# optional dask chunk manager. Skip (don't error) when dask is absent.
pytest.importorskip("dask")


# Load make_ryf as a module (it lives in scripts/, not in the package).
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "data" / "make_ryf.py"
_spec = importlib.util.spec_from_file_location("make_ryf", _SCRIPT)
make_ryf_mod = importlib.util.module_from_spec(_spec)
sys.modules["make_ryf"] = make_ryf_mod
_spec.loader.exec_module(make_ryf_mod)

from legoesm.forcing.jra55_do import (
    JRA55DoConfig,
    build_jra55_cache,
    load_jra55_slice,
)


# ============================================================================
# Synthetic IAF generator (matches make_ryf's expected layout)
# ============================================================================

def _write_synthetic_iaf(iaf_dir: Path, year: int, base_value: float):
    """Write all 10 variables for one calendar year into iaf_dir."""
    for v in make_ryf_mod.VARS:
        if v.ts_scheme == "instant":
            n_per_day = 8
            first_h, first_m = 0, 0
            freq = "3h"
        elif v.ts_scheme == "mean3h":
            n_per_day = 8
            first_h, first_m = 1, 30
            freq = "3h"
        else:
            n_per_day = 1
            first_h, first_m = 12, 0
            freq = "1D"
        n = 365 * n_per_day  # use non-leap convention always
        if freq == "3h":
            times = pd.date_range(
                f"{year}-01-01 {first_h:02d}:{first_m:02d}",
                periods=n, freq="3h",
            )
        else:
            times = pd.date_range(f"{year}-01-01 12:00", periods=n, freq="1D")

        n_lat, n_lon = 6, 12
        # Per-record, per-cell distinguishable values.
        idx = np.arange(n, dtype=np.float64)[:, None, None]
        data = base_value + idx * np.ones((1, n_lat, n_lon))

        # Latitude centres at half-cell offsets so cell edges land on poles.
        half_lat = 90.0 / n_lat
        lat = np.linspace(-90.0 + half_lat, 90.0 - half_lat, n_lat)
        half_lon = 180.0 / n_lon
        lon = np.linspace(half_lon, 360.0 - half_lon, n_lon)

        ds = xr.Dataset(
            {v.name: (("time", "lat", "lon"), data)},
            coords={"time": times, "lat": lat, "lon": lon},
        )
        path = make_ryf_mod._local_path(iaf_dir, v, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(str(path))


# ============================================================================
# End-to-end pipeline test
# ============================================================================

def test_make_ryf_output_is_consumable_by_cache_builder(tmp_path):
    """The Zarr make_ryf writes must be readable by build_jra55_cache
    end-to-end without manual intervention."""
    # Step 1: synthetic 2-year IAF
    iaf = tmp_path / "iaf"
    _write_synthetic_iaf(iaf, 1990, base_value=10000.0)
    _write_synthetic_iaf(iaf, 1991, base_value=20000.0)

    # Step 2: build RYF
    ryf_path = tmp_path / "RYF9091.zarr"
    make_ryf_mod.make_ryf(iaf, 1990, 1991, ryf_path, progress=False)
    assert ryf_path.exists()

    # Step 2.5: inspect what the time-axis units look like — this is
    # exactly what the cache builder will see.
    ds_raw = xr.open_zarr(str(ryf_path), decode_times=False)
    units = ds_raw["time"].attrs.get("units", "")
    # The cache builder accepts days|hours|minutes|seconds since YYYY-MM-DD;
    # whichever xarray happens to choose, the regex must match.
    import re
    m = re.match(
        r"^\s*(days|hours|minutes|seconds)\s+since\s+(\d{4})", units,
    )
    assert m, f"can't parse year from time units: {units!r}"
    ref_year = int(m.group(2))

    # Step 3: build the model-grid cache.
    target_lat_edges = np.deg2rad(np.linspace(-90.0, 90.0, 5))   # 4 cells
    target_lon_edges = np.deg2rad(np.linspace(0.0, 360.0, 9))    # 8 cells
    cfg = JRA55DoConfig(
        source_path=str(ryf_path),
        years=(ref_year, ref_year),
        target_lat_edges=target_lat_edges,
        target_lon_edges=target_lon_edges,
        cache_dir=tmp_path / "cache",
        ref_year=ref_year,
    )
    cache_path = build_jra55_cache(cfg, overwrite=True, progress=False)
    assert cache_path.exists()

    # Step 4: load a slice from the cache — verify the runtime path works.
    slc = load_jra55_slice(cache_path, day=0.0, ref_year=ref_year)
    for v in make_ryf_mod.VARS:
        arr = np.asarray(getattr(slc, v.name))
        # Shape matches target grid
        assert arr.shape == (4, 8), f"{v.name} got shape {arr.shape}"
        # Values are finite
        assert np.all(np.isfinite(arr))


def test_make_ryf_output_supports_cycle_mode_via_cache(tmp_path):
    """After the chain make_ryf -> build_jra55_cache, the runtime
    cycle=True flag works on the resulting cache (single-year cache
    drives multi-year runs)."""
    iaf = tmp_path / "iaf"
    _write_synthetic_iaf(iaf, 1990, base_value=10000.0)
    _write_synthetic_iaf(iaf, 1991, base_value=20000.0)
    ryf_path = tmp_path / "RYF9091.zarr"
    make_ryf_mod.make_ryf(iaf, 1990, 1991, ryf_path, progress=False)

    ds_raw = xr.open_zarr(str(ryf_path), decode_times=False)
    units = ds_raw["time"].attrs.get("units", "")
    import re
    m = re.match(
        r"^\s*(days|hours|minutes|seconds)\s+since\s+(\d{4})", units,
    )
    ref_year = int(m.group(2))

    cfg = JRA55DoConfig(
        source_path=str(ryf_path),
        years=(ref_year, ref_year),
        target_lat_edges=np.deg2rad(np.linspace(-90.0, 90.0, 5)),
        target_lon_edges=np.deg2rad(np.linspace(0.0, 360.0, 9)),
        cache_dir=tmp_path / "cache",
        ref_year=ref_year,
    )
    cache_path = build_jra55_cache(cfg, overwrite=True, progress=False)

    # In cycle mode, day 0 == day 365 == day 730.
    s0 = load_jra55_slice(cache_path, 0.0, ref_year=ref_year, cycle=True)
    s1 = load_jra55_slice(cache_path, 365.0, ref_year=ref_year, cycle=True)
    s2 = load_jra55_slice(cache_path, 730.0, ref_year=ref_year, cycle=True)
    for v in make_ryf_mod.VARS:
        np.testing.assert_array_equal(
            np.asarray(getattr(s0, v.name)),
            np.asarray(getattr(s1, v.name)),
        )
        np.testing.assert_array_equal(
            np.asarray(getattr(s0, v.name)),
            np.asarray(getattr(s2, v.name)),
        )
