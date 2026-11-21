"""``--sw-skin-fraction`` (jerlov_2band) and ``ohlmann_2003`` are read only on the tripole lane; elsewhere refused."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_sea_ice_scan_skin", _HERE / "test_run_omip_sea_ice_scan.py")
_scan = importlib.util.module_from_spec(_spec)
sys.modules["_sea_ice_scan_skin"] = _scan
_spec.loader.exec_module(_scan)
run_omip = _scan.run_omip


@pytest.mark.parametrize("sw,skin,match", [
    ("auto", 0.0, "--sw-skin-fraction"),
    ("jerlov_2band", 0.0, "--sw-skin-fraction"),      # the plain lat-lon lane never reads the scheme
    ("ohlmann_2003", None, "implemented on the tripole"),
])
def test_unread_shortwave_options_are_refused_off_tripole(tmp_path, sw, skin, match):
    cache = _scan._make_synthetic_cache(tmp_path, n_lat=8, n_lon=16, n_records=8)
    grid, z_coord, _, _, _ = _scan._make_tiny_latlon_setup(n_lat=8, n_lon=16)
    T_woa, S_woa = _scan._make_woa_like_targets(grid, nlev=4)
    args = _scan._argparse_namespace(jra55_cache=str(cache))
    args.sw_penetration = sw
    args.sw_skin_fraction = skin
    with pytest.raises(ValueError, match=match):
        run_omip._setup_jra55_forcing_state(args, grid, "latlon", z_coord=z_coord, T_woa=T_woa, S_woa=S_woa)
