"""Shared Veros↔legoESM data-prep glue for the global free-run recipes (#433).

Mimicry-only HARNESS glue shared by the global_flexible and global_1deg recipes
(and their free-run drivers): the (x,y,z)/(x,y) axis-transpose + wall-row layout
bridges, the Veros ``set_forcing_kernel`` MIT-grid wind-stress index shift, and
the Veros T-cell area weights.  Per the oracle-recipe doctrine this is harness
glue (axis transpose / time-level / forcing placement mimicry), NOT model code —
it lives in the fidelity harness, never in a shippable model path.

Factored from the two recipes' previously copy-pasted ``*_flex`` / ``*_1deg``
twins (the DEDUP NOTE in ``veros_global_1deg_recipe``).  The layout bridges and
the area formula were byte-identical; the tau shift differed ONLY in its x
boundary (cyclic-x roll vs zero-ghost), captured here by the ``x_cyclic`` flag —
a genuine grid property, not indexing-only copy-paste.
"""

from __future__ import annotations

import numpy as np
from legoesm.ocean.constants_config import VEROS_CONSTANTS_CONFIG


def veros_xyz_to_legoesm(arr_xyz: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """(x, y, z) VEROS z-order (k=0 deepest) → legoESM (lat, lon, z) with k=0
    SURFACE, plus the two wall rows (shape-generic; sizes from the input)."""
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


def veros_mit_tau_shift(
    taux_xym: np.ndarray,
    tauy_xym: np.ndarray,
    *,
    x_cyclic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """The Veros ``set_forcing_kernel`` MIT-grid index shift: ``surface_taux[i]
    = taux[i+1]`` (one cell in x) and ``surface_tauy[j] = tauy[j+1]`` (one cell
    in y), applied ONCE at data-prep time on the interior monthly stacks
    (x, y, 12).

    The y-shift always pulls the zero NORTH GHOST into the last interior row
    (Veros never fills ``tauy``'s y ghosts).  The x-shift depends on the grid's
    x boundary:

    * ``x_cyclic=True`` (global_flexible, cyclic-x ghosts ``taux[nx+2]=taux[2]``):
      a roll by −1.
    * ``x_cyclic=False`` (global_1deg): Veros never fills the custom ``taux`` x
      ghosts either, so the x-shift pulls a ZERO ghost into the last interior
      column (a cyclic roll fails the harness gate by 0.2 N/m²).
    """
    if x_cyclic:
        taux_shift = np.roll(taux_xym, -1, axis=0)
    else:
        taux_shift = np.concatenate(
            [taux_xym[1:, :, :], np.zeros_like(taux_xym[:1, :, :])], axis=0)
    tauy_shift = np.concatenate(
        [tauy_xym[:, 1:, :], np.zeros_like(tauy_xym[:, :1, :])], axis=1)
    return taux_shift, tauy_shift


def veros_area_t(
    yt_deg: np.ndarray,
    *,
    dx_deg: float,
    dyt_deg,
    r_earth: float = VEROS_CONSTANTS_CONFIG.R_earth,
) -> np.ndarray:
    """Veros T-cell area column weights ``dxt·dyt·cost`` [m²] (per-latitude row;
    broadcast over x).  ``dyt_deg`` may be a scalar (uniform grid) or a per-row
    array (stretched grid); ``dx_deg`` is the uniform zonal spacing (360/nx)."""
    degtom = r_earth * np.pi / 180.0
    return (dx_deg * degtom) * (np.asarray(dyt_deg) * degtom) * np.cos(
        np.deg2rad(np.asarray(yt_deg)))
