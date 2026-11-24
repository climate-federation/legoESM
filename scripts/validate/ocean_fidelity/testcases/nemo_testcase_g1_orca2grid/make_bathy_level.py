#!/usr/bin/env python3
"""Write DOMAINcfg's level-bathymetry input for ORCA2G1 from ORCA2's own mask.

``Bathy_level`` is 1 where the shipped ORCA2 ``bottom_level`` is wet and 0 on
land: ORCA2's coast over a one-level column.  tools/DOMAINcfg reads it at
domzgr.F90:625-629 (``ln_zco``, ``nn_bathy = 1``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import netCDF4
import numpy as np


def bathy_level(bottom_level: np.ndarray) -> np.ndarray:
    return (np.asarray(bottom_level) > 0).astype(np.float64)


def write_bathy_level(source: Path, target: Path) -> int:
    with netCDF4.Dataset(source, "r") as src:
        level = bathy_level(src.variables["bottom_level"][:])
    with netCDF4.Dataset(target, "w", format="NETCDF4_CLASSIC") as dst:
        dst.createDimension("y", level.shape[0])
        dst.createDimension("x", level.shape[1])
        dst.createVariable("Bathy_level", "f8", ("y", "x"))[:] = level
        dst.source = f"bottom_level > 0 of {source.name}"
    return int(level.sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print("G1_BATHY_LEVEL_WET", write_bathy_level(args.source, args.target))
