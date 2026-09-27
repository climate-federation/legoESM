"""NEMO viscosity file -> MPAS/FESOM points: nearest WET NEMO point per level."""
from __future__ import annotations

import netCDF4 as nc4
import numpy as np

from scripts.run.run_omip_core2 import (
    _FESOM_WIRED_DESTS,
    _build_arg_parser,
    nemo_ldf_nearest_wet,
    validate_fesom_stage,
)


def _files(tmp_path):
    lat = np.repeat(np.linspace(-80, 80, 331)[:, None], 360, 1)
    lon = np.repeat(np.linspace(0.5, 359.5, 360)[None, :], 331, 0)
    ahm = np.zeros((1, 2, 331, 360))
    ahm[0, 0, :, 180:] = 5000.0            # level 0: east half wet
    ahm[0, 1, :, 180:] = 700.0             # level 1: different value
    ldf, dom = tmp_path / "ldf.nc", tmp_path / "dom.nc"
    with nc4.Dataset(ldf, "w") as d:
        for n, s in (("t", 1), ("z", 2), ("y", 331), ("x", 360)):
            d.createDimension(n, s)
        for v in ("ahmt_3d", "ahmf_3d"):
            d.createVariable(v, "f8", ("t", "z", "y", "x"))[:] = ahm
        d.createVariable("nav_lat", "f8", ("y", "x"))[:] = lat
        d.createVariable("nav_lon", "f8", ("y", "x"))[:] = lon
    with nc4.Dataset(dom, "w") as d:
        d.createDimension("y", 331); d.createDimension("x", 360)
        d.createVariable("gphif", "f8", ("y", "x"))[:] = lat
        d.createVariable("glamf", "f8", ("y", "x"))[:] = lon
    return str(ldf), str(dom)


def test_coastal_target_takes_wet_value_not_land_zero(tmp_path):
    ldf, dom = _files(tmp_path)
    # Target at lon 170: its all-points nearest is dry (value 0); wet starts at 180.5.
    lat = np.radians([10.0, 10.0]); lon = np.radians([170.0, 250.0])
    ahmt, ahmf = nemo_ldf_nearest_wet(ldf, dom, lat, lon, [0, 1, 1])
    assert ahmt.shape == (2, 3)
    np.testing.assert_array_equal(ahmt[0], [5000.0, 700.0, 700.0])
    np.testing.assert_array_equal(ahmf[1], [5000.0, 700.0, 700.0])


def test_fesom_stage_allows_nemo_ldf_flags():
    assert {"nemo_ldf_file", "lateral_side_bc"} <= _FESOM_WIRED_DESTS
    p = _build_arg_parser()
    args = p.parse_args(["--grid", "fesom", "--fesom-mesh-dir", "/m", "--output", "/o",
                         "--no-emp", "--nemo-ldf-file", "/f.nc",
                         "--lateral-side-bc", "no_slip"])
    validate_fesom_stage(args, p)
