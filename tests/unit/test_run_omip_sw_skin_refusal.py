"""``--sw-skin-fraction`` is read only by jerlov_2band on the lat-lon / tripole lanes; elsewhere it is refused."""
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


@pytest.mark.parametrize("sw", ["auto", "sweeney_2band"])
def test_skin_fraction_without_jerlov_is_refused(tmp_path, sw):
    cache = _scan._make_synthetic_cache(tmp_path, n_lat=8, n_lon=16, n_records=8)
    grid, z_coord, _, _, _ = _scan._make_tiny_latlon_setup(n_lat=8, n_lon=16)
    T_woa, S_woa = _scan._make_woa_like_targets(grid, nlev=4)
    args = _scan._argparse_namespace(jra55_cache=str(cache))
    args.sw_penetration = sw
    args.sw_skin_fraction = 0.0
    with pytest.raises(ValueError, match="--sw-skin-fraction"):
        run_omip._setup_jra55_forcing_state(args, grid, "latlon", z_coord=z_coord, T_woa=T_woa, S_woa=S_woa)
