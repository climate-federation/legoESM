"""Core single-photon Woodcock (null-collision) transport for the 3D MC tracer.

Geometry: doubly-periodic plane, ``z`` increasing upward, ``z_faces[0]`` the
surface and ``z_faces[nz]`` the top-of-domain (TOD). One photon is walked by a
``lax.while_loop``; ``trace_batch`` ``vmap``s it over a photon batch.

**Analog Monte-Carlo** (no weight reduction): at a real collision the photon is
absorbed with probability ``1-ssa`` (deposited in exactly one cell) or scatters;
at the surface it is absorbed with probability ``1-albedo`` or reflects
(Lambertian). Each photon therefore terminates in exactly one bin
(volume-absorbed / surface-absorbed / escaped-TOD), which makes energy
conservation exact photon-by-photon and lets the batch be tallied by a single
``segment_sum`` (no in-loop scatter-add, so the walk stays ``vmap``-clean).

See ``docs/specs/mc3d_raytracer.md``.
"""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.radiation.mc3d import knull_grid
from legoesm.atmosphere.physics.radiation.mc3d import sampling

Array: TypeAlias = jax.Array

# Terminal status codes (photon ends in exactly one).
STATUS_ALIVE = 0
STATUS_VOL_ABS = 1  # absorbed in a volume cell
STATUS_SFC_ABS = 2  # absorbed at the surface
STATUS_TOD_UP = 3   # escaped through the top of the domain
STATUS_MAXITER = 4  # hit the tracking-step cap (logged; should be ~0 in tests)


class PlaneRTGeometry(NamedTuple):
  """Doubly-periodic plane ray-tracing geometry."""

  dx: float          # horizontal spacing x [m]
  dy: float          # horizontal spacing y [m]
  Lx: float          # domain length x [m]
  Ly: float          # domain length y [m]
  z_faces: Array     # (nz+1,) ascending layer interfaces [m], [0]=sfc [nz]=TOD
  nx: int
  ny: int
  nz: int


class _PhotonState(NamedTuple):
  key: Array
  pos: Array       # (3,) [m]
  dir: Array       # (3,) unit
  cix: Array       # int scalar coarse-cell x index (carried; DDA-stepped)
  ciy: Array       # int scalar coarse-cell y index
  ciz: Array       # int scalar coarse-cell z index
  alive: Array     # bool scalar
  status: Array    # int scalar
  vol_idx: Array   # int scalar flat (ix*ny+iy)*nz+iz of absorption cell, else -1
  sfc_idx: Array   # int scalar flat ix*ny+iy of surface-absorption col, else -1
  n_iter: Array    # int scalar


def _cell_indices(pos: Array, geom: PlaneRTGeometry):
  """Wrapped horizontal + searchsorted vertical cell indices at ``pos``."""
  x = jnp.mod(pos[0], geom.Lx)
  y = jnp.mod(pos[1], geom.Ly)
  ix = jnp.clip((x / geom.dx).astype(jnp.int32), 0, geom.nx - 1)
  iy = jnp.clip((y / geom.dy).astype(jnp.int32), 0, geom.ny - 1)
  iz = jnp.searchsorted(geom.z_faces, pos[2], side="right") - 1
  iz = jnp.clip(iz.astype(jnp.int32), 0, geom.nz - 1)
  return ix, iy, iz


def trace_one(
    key: Array,
    pos0: Array,
    dir0: Array,
    k_ext: Array,
    ssa: Array,
    g: Array,
    maj: "knull_grid.MajorantGrid",
    albedo: Array,
    geom: PlaneRTGeometry,
    max_iterations: int,
    rayleigh_frac: Array | None = None,
    mie_cdf: Array | None = None,
    mie_ang: Array | None = None,
    r_eff: Array | None = None,
) -> _PhotonState:
  """Walk one photon to termination via coarse-grid (decomposition) Woodcock
  tracking; return its terminal ``_PhotonState``.

  The free path is sampled with the LOCAL coarse-cell majorant. The flight stops
  at the first of: a real/null collision, a coarse-cell face (resample majorant),
  or a domain z edge (surface / TOD). Delta tracking is unbiased for any
  per-region majorant >= local k_ext, so this is a pure speedup over a single
  global majorant. ``maj`` block sizes are static Python ints (closed over vmap).
  """
  z_sfc = maj.zc_faces[0]
  z_tod = maj.zc_faces[-1]
  dtype = k_ext.dtype
  dxc = maj.bx * geom.dx
  dyc = maj.by * geom.dy
  bx, by, bz = maj.bx, maj.by, maj.bz
  ncx, ncy, ncz = maj.ncx, maj.ncy, maj.ncz

  # Initial coarse-cell indices (carried + DDA-stepped thereafter; never
  # re-floored from a possibly-on-face position -> no boundary aliasing/bias).
  xw0 = jnp.mod(pos0[0], geom.Lx)
  yw0 = jnp.mod(pos0[1], geom.Ly)
  cix0 = jnp.clip((xw0 / dxc).astype(jnp.int32), 0, ncx - 1)
  ciy0 = jnp.clip((yw0 / dyc).astype(jnp.int32), 0, ncy - 1)
  ciz0 = jnp.clip(
      (jnp.searchsorted(maj.zc_faces, pos0[2], side="right") - 1).astype(
          jnp.int32), 0, ncz - 1)

  init = _PhotonState(
      key=key,
      pos=pos0.astype(dtype),
      dir=dir0.astype(dtype),
      cix=cix0, ciy=ciy0, ciz=ciz0,
      alive=jnp.array(True),
      status=jnp.array(STATUS_ALIVE, jnp.int32),
      vol_idx=jnp.array(-1, jnp.int32),
      sfc_idx=jnp.array(-1, jnp.int32),
      n_iter=jnp.array(0, jnp.int32),
  )

  def cond(s: _PhotonState) -> Array:
    return s.alive

  def body(s: _PhotonState) -> _PhotonState:
    key, k_path, k_real, k_abs, k_scat, k_sfc, k_refl = jax.random.split(s.key, 7)
    x, y, z = s.pos[0], s.pos[1], s.pos[2]
    ux, uy, uz = s.dir[0], s.dir[1], s.dir[2]
    cix, ciy, ciz = s.cix, s.ciy, s.ciz
    k_maj = maj.grid[cix, ciy, ciz]

    t_coll = sampling.free_path(k_path, k_maj)

    # Distance to the CURRENT coarse cell's faces (derived from carried indices,
    # not a position re-floor). Horizontal axes with one coarse cell have no
    # interior faces (uniform majorant) -> never stop (== global-scalar walk).
    # Sign-based face distance: a tiny-but-nonzero component gives a LARGE FINITE
    # distance (never silently dropped to inf), and an EXACTLY-zero component
    # gives +inf (no crossing on that axis). hi-x / lo-x are > 0 / < 0 by the
    # face-on-entry invariant, so no 0/0. A near-axis-aligned ray is handled
    # exactly (large finite distance), never silently dropped.
    # An EXACTLY-zero component means no motion on that axis -> no crossing
    # (inf). The denominator is replaced by 1.0 in the zero branch so the
    # (masked-out) division never evaluates 0/0 -- jnp.where computes BOTH
    # branches, so a raw /ux would emit a dead NaN that trips jax_debug_nans and
    # could poison reverse-mode AD (Phase 5 emulator). safe_u keeps it AD-clean.
    safe_ux = jnp.where(ux == 0.0, 1.0, ux)
    safe_uy = jnp.where(uy == 0.0, 1.0, uy)
    safe_uz = jnp.where(uz == 0.0, 1.0, uz)
    if ncx > 1:
      lo_x, hi_x = cix.astype(dtype) * dxc, (cix + 1).astype(dtype) * dxc
      tx = jnp.where(ux == 0.0, jnp.inf,
                     jnp.where(ux > 0.0, (hi_x - x) / safe_ux,
                               (lo_x - x) / safe_ux))
    else:
      tx = jnp.inf
    if ncy > 1:
      lo_y, hi_y = ciy.astype(dtype) * dyc, (ciy + 1).astype(dtype) * dyc
      ty = jnp.where(uy == 0.0, jnp.inf,
                     jnp.where(uy > 0.0, (hi_y - y) / safe_uy,
                               (lo_y - y) / safe_uy))
    else:
      ty = jnp.inf
    z_up = maj.zc_faces[ciz + 1]
    z_dn = maj.zc_faces[ciz]
    tz = jnp.where(uz == 0.0, jnp.inf,
                   jnp.where(uz > 0.0, (z_up - z) / safe_uz,
                             (z_dn - z) / safe_uz))

    t_bound = jnp.minimum(jnp.minimum(tx, ty), tz)
    collide_first = t_coll < t_bound
    dist = jnp.where(collide_first, t_coll, t_bound)
    pos = s.pos + dist * s.dir

    boundary = ~collide_first
    # Which axis/axes bound the flight. SYMMETRIC masks: an exact corner tie
    # (e.g. tx==ty) marks BOTH axes so they step in the SAME iteration, leaving
    # pos on no residual face (no zero-distance follow-up sub-step).
    x_min = boundary & (tx <= ty) & (tx <= tz)
    y_min = boundary & (ty <= tx) & (ty <= tz)
    z_min = boundary & (tz <= tx) & (tz <= ty)

    # Domain z edges (only z faces can be the domain surface / TOD). uz sign is
    # consistent with the tz convention above (uz==0 -> tz=inf -> z_min False).
    at_tod = z_min & (uz > 0.0) & (ciz == ncz - 1)
    at_sfc = z_min & (uz < 0.0) & (ciz == 0)
    domain_edge = at_tod | at_sfc

    # DDA step of the coarse-cell index along the crossed axis (periodic x,y;
    # bounded z). No position re-floor, so on-face round-off cannot alias cells.
    sx = jnp.where(ux > 0, 1, -1).astype(jnp.int32)
    sy = jnp.where(uy > 0, 1, -1).astype(jnp.int32)
    sz = jnp.where(uz > 0, 1, -1).astype(jnp.int32)
    cix_n = jnp.where(x_min, jnp.mod(cix + sx, ncx), cix)
    ciy_n = jnp.where(y_min, jnp.mod(ciy + sy, ncy), ciy)
    # Interior z crossing steps ciz; at a domain edge it stays (reflect) or the
    # photon escapes (TOD).
    ciz_n = jnp.where(z_min & (~domain_edge), ciz + sz, ciz)

    # Place pos EXACTLY on the entered face in the NEW cell's frame so the
    # carried index and the stored position stay consistent across the periodic
    # seam (jnp.mod-ing the position would put it in cell 0's frame while the
    # index says ncx-1 -> negative next-step face distance). z domain edges are
    # pinned; interior z crossings sit on the shared coarse face.
    new_z_int = jnp.where(uz > 0, maj.zc_faces[ciz_n], maj.zc_faces[ciz_n + 1])
    pos = pos.at[2].set(jnp.where(
        at_tod, z_tod,
        jnp.where(at_sfc, z_sfc,
                  jnp.where(z_min, new_z_int, pos[2]))))
    if ncx > 1:
      new_x = jnp.where(sx > 0, cix_n.astype(dtype) * dxc,
                        (cix_n + 1).astype(dtype) * dxc)
      pos = pos.at[0].set(jnp.where(x_min, new_x, pos[0]))
    else:
      # Single coarse cell: no interior faces; wrap to bound float drift.
      pos = pos.at[0].set(jnp.mod(pos[0], geom.Lx))
    if ncy > 1:
      new_y = jnp.where(sy > 0, ciy_n.astype(dtype) * dyc,
                        (ciy_n + 1).astype(dtype) * dyc)
      pos = pos.at[1].set(jnp.where(y_min, new_y, pos[1]))
    else:
      pos = pos.at[1].set(jnp.mod(pos[1], geom.Ly))

    # --- collision branch (delta-tracking accept/reject, local majorant) ---
    # On a collision the photon stayed in the CURRENT coarse cell, so the fine
    # cell is clamped to that coarse cell's fine-index range -> k_ext <= k_maj
    # exactly (unbiased), immune to the on-face rounding of the moved position.
    ix0, iy0, iz0 = _cell_indices(pos, geom)
    iz_hi = jnp.minimum((ciz + 1) * bz, geom.nz) - 1
    ix = jnp.clip(ix0, cix * bx, (cix + 1) * bx - 1)
    iy = jnp.clip(iy0, ciy * by, (ciy + 1) * by - 1)
    iz = jnp.clip(iz0, ciz * bz, iz_hi)
    cell_flat = (ix * geom.ny + iy) * geom.nz + iz
    col_flat = ix * geom.ny + iy

    ke = k_ext[ix, iy, iz]
    w = ssa[ix, iy, iz]
    gg = g[ix, iy, iz]
    real = jax.random.uniform(k_real, dtype=dtype) < (ke / k_maj)
    absorbed = jax.random.uniform(k_abs, dtype=dtype) < (1.0 - w)
    vol_abs = collide_first & real & absorbed
    scatter = collide_first & real & (~absorbed)
    if rayleigh_frac is None:
      scat_dir = sampling.scatter_direction(k_scat, s.dir, gg)
    else:
      # Explicit Rayleigh (gas) + cloud (Mie LUT, else HG) mixture per the local
      # Rayleigh fraction (oracle-faithful gas/cloud phase split).
      re_cell = r_eff[ix, iy, iz] if r_eff is not None else None
      scat_dir = sampling.scatter_direction_mixed(
          k_scat, s.dir, gg, rayleigh_frac[ix, iy, iz],
          mie_cdf=mie_cdf, mie_ang=mie_ang, r_eff=re_cell)

    # --- surface branch ---
    reflect = at_sfc & (jax.random.uniform(k_sfc, dtype=dtype) < albedo)
    sfc_abs = at_sfc & (~reflect)
    refl_dir = sampling.lambertian_reflect(k_refl, dtype)

    # New direction (default unchanged: null collision or interior face cross).
    new_dir = jnp.where(scatter, scat_dir, s.dir)
    new_dir = jnp.where(reflect, refl_dir, new_dir)

    n_iter = s.n_iter + 1
    maxed = n_iter >= max_iterations
    dead = vol_abs | at_tod | sfc_abs
    alive = ~(dead | maxed)

    status = jnp.where(vol_abs, STATUS_VOL_ABS, STATUS_ALIVE)
    status = jnp.where(at_tod, STATUS_TOD_UP, status)
    status = jnp.where(sfc_abs, STATUS_SFC_ABS, status)
    status = jnp.where(maxed & (~dead), STATUS_MAXITER, status)

    vol_idx = jnp.where(vol_abs, cell_flat, s.vol_idx)
    sfc_idx = jnp.where(sfc_abs, col_flat, s.sfc_idx)

    return _PhotonState(
        key=key,
        pos=pos,
        dir=new_dir,
        cix=cix_n.astype(jnp.int32),
        ciy=ciy_n.astype(jnp.int32),
        ciz=ciz_n.astype(jnp.int32),
        alive=alive,
        status=status.astype(jnp.int32),
        vol_idx=vol_idx.astype(jnp.int32),
        sfc_idx=sfc_idx.astype(jnp.int32),
        n_iter=n_iter.astype(jnp.int32),
    )

  return jax.lax.while_loop(cond, body, init)


def trace_batch(
    keys: Array,
    pos0: Array,
    dir0: Array,
    k_ext: Array,
    ssa: Array,
    g: Array,
    maj: "knull_grid.MajorantGrid",
    albedo: Array,
    geom: PlaneRTGeometry,
    max_iterations: int,
    rayleigh_frac: Array | None = None,
    mie_cdf: Array | None = None,
    mie_ang: Array | None = None,
    r_eff: Array | None = None,
) -> _PhotonState:
  """``vmap`` ``trace_one`` over a photon batch (leading axis on
  ``keys``/``pos0``/``dir0``). ``maj``/fields/geom are CLOSED OVER (not mapped)
  so the majorant block sizes stay static Python ints inside ``trace_one``."""
  def _one(key, pos, dir_):
    return trace_one(key, pos, dir_, k_ext, ssa, g, maj, albedo, geom,
                     max_iterations, rayleigh_frac=rayleigh_frac,
                     mie_cdf=mie_cdf, mie_ang=mie_ang, r_eff=r_eff)

  return jax.vmap(_one)(keys, pos0, dir0)
