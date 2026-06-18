"""Shared VEROS→legoESM layout bridges for the global veros-faithful recipes.

The 1° and flexible global recipes carried byte-identical shape-generic
layout bridges (their ``veros_*_to_legoesm_{1deg,flex}`` helpers); the 1°
recipe's module DEDUP NOTE asked that whichever landed second factor them
into one shared fidelity helper.  This is that module.

These are pure NumPy layout transforms (no model state, no autodiff) that
re-order a VEROS array into legoESM grid convention — harness glue, not
model numerics, per ``docs/ocean_fidelity/oracle_recipe_strategy.md``.

The 4° recipe keeps its own ``veros_xyz_to_legoesm``/``veros_xy_to_legoesm``
which are pinned to that recipe's fixed ``NX`` and take an explicit
``n_lat_grid`` override, so they are a distinct variant (a different goal
would select them) and are deliberately not folded in here.
"""
from __future__ import annotations

import numpy as np

__all__ = ["veros_xyz_to_legoesm", "veros_xy_to_legoesm"]


def veros_xyz_to_legoesm(arr_xyz: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """(x, y, z) VEROS z-order (k=0 deepest) → legoESM (lat, lon, z) with
    k=0 SURFACE, plus the two wall rows.  Sizes are derived from the input
    (shape-generic)."""
    nx, ny, nz = arr_xyz.shape
    out = np.full((ny + 2, nx, nz), fill, dtype=np.float64)
    out[1:-1, :, :] = np.transpose(arr_xyz, (1, 0, 2))[:, :, ::-1]
    return out


def veros_xy_to_legoesm(arr_xy: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """(x, y) → legoESM (lat, lon) with wall rows (2-D forcing fields)."""
    nx, ny = arr_xy.shape
    out = np.full((ny + 2, nx), fill, dtype=np.float64)
    out[1:-1, :] = arr_xy.T
    return out
