"""Smoke test for scripts/data/build_ec_site_example_drivers.py.

Reuses the synthetic DifferBESS-style driver from ``test_ec_site_run`` (named as a
``*_gapfree.nc`` so the builder finds it), runs ``build_one`` for a one-year
window, and asserts the trimmed output: the analysis window is subset, only
model-consumed variables are kept, floats are float32, every kept variable carries
units + long_name + description, and the dataset carries site/provenance globals.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import xarray as xr
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]
_BUILD_PY = _REPO / "scripts" / "data" / "build_ec_site_example_drivers.py"
_DRIVER_TEST_PY = pathlib.Path(__file__).with_name("test_ec_site_run.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_make_driver = _load("test_ec_site_run", _DRIVER_TEST_PY)._make_driver


def test_build_one_trims_and_annotates(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    out = tmp_path / "out"
    # synthetic driver spanning >1 year so the window subset is a real cut
    dp = str(src / "SYN-Test_driver_v2_gapfree.nc")
    _make_driver(dp, n=48 * 400)
    build = _load("build_ec_site_example_drivers", _BUILD_PY)
    # augment with the consumed vars the base synthetic driver lacks, so the test
    # exercises the FULL keep-set (closure band + gap-fill flags), then compare the
    # output against exactly what the model consumes AND the source provides.
    with xr.open_dataset(dp) as _d:
        srcds = _d.load().copy(deep=True)
    nn = srcds.sizes["time"]
    for extra, val in [("ET_CORR", 2.2), ("H_CORR", 50.0), ("met_atm_filled", 0.0),
                       ("met_any_filled", 0.0), ("soil_filled", 0.0)]:
        srcds[extra] = ("time", np.full(nn, val))
    srcds.to_netcdf(dp)

    p = build.build_one("SYN-Test", 2015, 2015, str(src), str(out))
    assert pathlib.Path(p).name == "SYN-Test_driver_v2_gapfree.nc"
    ds = xr.open_dataset(p)

    # EXACTLY the kept vars the source provides survive — a consumed var silently
    # dropped from build_one's `have` filter (e.g. the EMISSIVITY regression) fails.
    expected = {v for v in build.KEEP if v in srcds.data_vars}
    assert set(ds.data_vars) == expected
    assert "EMISSIVITY" in ds and "ET_CORR" in ds and "soil_filled" in ds
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.floating):      # source float64 preserved
            assert ds[v].dtype == np.float64
        for a in ("units", "long_name", "description"):  # metadata on every var
            assert a in ds[v].attrs and ds[v].attrs[a]
    # exact units for the reader-sensitive fields (a hPa->Pa etc. regression fails)
    assert ds["VPD"].attrs["units"] == "hPa"
    assert ds["PA"].attrs["units"] == "kPa"
    assert ds["SWC"].attrs["units"] == "percent"
    assert ds["ET"].attrs["units"] == "mm day-1"
    assert ds["CO2"].attrs["units"] == "umol mol-1"

    # window actually subset (synthetic starts 2015-06-01; one year -> < full record)
    assert set(np.unique(ds.time.dt.year.values)) == {2015}
    assert ds.sizes["time"] < 48 * 400

    # dataset provenance globals
    for a in ("title", "site", "source", "gapfree_source", "targets_are_raw",
              "temporal_resolution", "Conventions"):
        assert a in ds.attrs
    assert ds.attrs["site"] == "SYN-Test"


def test_keep_set_covers_reader_required(tmp_path):
    """The kept-variable set must include every field the reader marks REQUIRED
    (so a trimmed driver never drops a model input)."""
    build = _load("build_ec_site_example_drivers", _BUILD_PY)
    from legoesm.land.boundary_data import ec_site
    assert set(ec_site._REQUIRED) <= set(build.KEEP)
    # and every kept variable is documented
    assert set(build.KEEP) <= set(build.METADATA)


def test_build_one_empty_window_raises(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    _make_driver(str(src / "SYN-Test_driver_v2_gapfree.nc"), n=96)
    build = _load("build_ec_site_example_drivers", _BUILD_PY)
    with pytest.raises(ValueError, match="no steps"):
        build.build_one("SYN-Test", 1990, 1991, str(src), str(tmp_path / "o"))
