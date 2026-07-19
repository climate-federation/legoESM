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
    _make_driver(str(src / "SYN-Test_driver_v2_gapfree.nc"), n=48 * 400)
    build = _load("build_ec_site_example_drivers", _BUILD_PY)

    p = build.build_one("SYN-Test", 2015, 2015, str(src), str(out))
    assert pathlib.Path(p).name == "SYN-Test_driver_v2_gapfree.nc"
    ds = xr.open_dataset(p)

    # every kept variable is one the model consumes; source float64 is preserved
    # (no downcast — prognostic soil moisture drifts under float32).
    assert set(ds.data_vars) <= set(build.KEEP)
    assert "SW_IN" in ds and "GPP_DT" in ds            # forcing + a target survived
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.floating):
            assert ds[v].dtype == np.float64
        # metadata present on every kept variable
        for a in ("units", "long_name", "description"):
            assert a in ds[v].attrs and ds[v].attrs[a]

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
