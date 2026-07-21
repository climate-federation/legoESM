"""Majorant (null-collision) extinction for Woodcock delta tracking.

Phase 1 uses a single **global scalar majorant** ``k_null = max(k_ext)`` (floored),
which is unconditionally correct Woodcock tracking. A coarse per-block majorant
grid (microhh ``create_knull_grid``) is a pure efficiency optimisation for
optically clustered fields (cloud k_ext >> clear-sky) and is a documented
Phase-1.1 follow-up — it changes runtime, never the answer.
"""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array


class MajorantGrid(NamedTuple):
  """Coarse null-collision (Woodcock) majorant grid for delta tracking.

  ``grid[cix,ciy,ciz]`` is the per-coarse-cell majorant extinction (max over the
  fine cells it covers, floored). ``bx``/``by``/``bz`` are the block sizes in
  fine cells; horizontal blocks MUST divide ``nx``/``ny`` (periodic faces stay on
  the domain grid). ``zc_faces`` are the coarse vertical interfaces (ascending,
  ``[0]``=surface, ``[-1]``=TOD). A single coarse cell (bx=nx,by=ny,bz=nz)
  reproduces the global-scalar majorant exactly: with one cell per horizontal
  axis the walk takes no interior horizontal stops (uniform majorant).
  """

  grid: Array      # (ncx, ncy, ncz) majorant extinction [1/m]
  zc_faces: Array  # (ncz+1,) ascending coarse z interfaces [m]
  bx: int
  by: int
  bz: int
  ncx: int
  ncy: int
  ncz: int


def build_majorant_grid(
    k_ext: Array,
    z_faces: Array,
    knull_floor: float,
    coarsen_xy: int = 0,
    coarsen_z: int = 0,
) -> MajorantGrid:
  """Build a coarse majorant grid for delta tracking.

  ``coarsen_xy`` / ``coarsen_z`` are block sizes in fine cells; ``<=0`` means "no
  coarsening" (one coarse cell spanning the whole axis -> global scalar majorant
  on that axis). A finer grid (smaller blocks) gives tighter majorants in clear
  air next to optically thick clouds, so far fewer null-collision steps — pure
  speedup, the answer (delta tracking) is unbiased for any valid majorant.

  Args:
    k_ext: extinction field ``(nx, ny, nz)`` [1/m], non-negative.
    z_faces: ``(nz+1,)`` ascending interfaces [m] ([0]=surface, [-1]=TOD).
    knull_floor: majorant floor [1/m].
    coarsen_xy: horizontal block size (fine cells); MUST divide nx and ny.
    coarsen_z: vertical block size (fine cells); last block may be shorter.
  """
  nx, ny, nz = k_ext.shape
  bx = nx if coarsen_xy <= 0 else int(coarsen_xy)
  by = ny if coarsen_xy <= 0 else int(coarsen_xy)
  bz = nz if coarsen_z <= 0 else int(coarsen_z)
  if nx % bx != 0 or ny % by != 0:
    raise ValueError(
        f"coarsen_xy={coarsen_xy} must divide both nx={nx} and ny={ny} "
        "(periodic horizontal majorant faces must stay on the domain grid)."
    )
  ncx, ncy = nx // bx, ny // by
  ncz = -(-nz // bz)  # ceil; last vertical block may be shorter (z not periodic)

  # Block-reduce by max. Horizontal divides exactly; pad z up to ncz*bz with 0
  # (k_ext >= 0, so the pad never raises the max of a real block).
  pad_z = ncz * bz - nz
  k_pad = jnp.pad(k_ext, ((0, 0), (0, 0), (0, pad_z)))
  blocks = k_pad.reshape(ncx, bx, ncy, by, ncz, bz)
  grid = jnp.max(blocks, axis=(1, 3, 5))
  grid = jnp.maximum(grid, knull_floor).astype(k_ext.dtype)

  # Coarse z interfaces at block edges: z_faces[min(k*bz, nz)].
  zc_idx = jnp.minimum(jnp.arange(ncz + 1) * bz, nz)
  zc_faces = z_faces[zc_idx].astype(k_ext.dtype)

  return MajorantGrid(grid=grid, zc_faces=zc_faces, bx=bx, by=by, bz=bz,
                      ncx=ncx, ncy=ncy, ncz=ncz)
