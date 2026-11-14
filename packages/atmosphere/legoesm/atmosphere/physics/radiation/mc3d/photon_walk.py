"""Woodcock (null-collision) photon transport for the 3D MC tracer.

Geometry: doubly-periodic plane, ``z`` increasing upward, ``z_faces[0]`` the
surface and ``z_faces[nz]`` the top-of-domain (TOD). Both walks share one
flight step (``_flight``: coarse-majorant free flight + exact face DDA).

``trace_batch_weighted`` (shortwave) is microhh's weighted estimator; see its
docstring for the one declared departure (weight floor instead of Russian
roulette).

``trace_one`` / ``trace_batch`` (longwave) are **analog**: at a real collision
the photon is absorbed with probability ``1-ssa`` (deposited in exactly one
cell) or scatters; at the surface it is absorbed with probability ``1-albedo``
or reflects (Lambertian). Each photon terminates in exactly one bin, so energy
conservation is exact photon-by-photon.

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


class PhotonState(NamedTuple):
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


def _coarse_indices(pos0, maj, geom):
  """Initial coarse-cell indices (carried + DDA-stepped thereafter; never
  re-floored from a possibly-on-face position -> no boundary aliasing/bias)."""
  xw0 = jnp.mod(pos0[..., 0], geom.Lx)
  yw0 = jnp.mod(pos0[..., 1], geom.Ly)
  cix0 = jnp.clip((xw0 / (maj.bx * geom.dx)).astype(jnp.int32), 0, maj.ncx - 1)
  ciy0 = jnp.clip((yw0 / (maj.by * geom.dy)).astype(jnp.int32), 0, maj.ncy - 1)
  ciz0 = jnp.clip(
      (jnp.searchsorted(maj.zc_faces, pos0[..., 2], side="right") - 1).astype(
          jnp.int32), 0, maj.ncz - 1)
  return cix0, ciy0, ciz0


def _flight(pos_in, dir_in, cix, ciy, ciz, t_coll, maj, geom):
  """One Woodcock flight from ``pos_in`` along ``dir_in``: stop at the sampled
  collision distance ``t_coll`` or the current coarse cell's face, whichever is
  first. Returns the new position, DDA-stepped coarse indices, the
  collision/surface/TOD masks and the fine collision cell (clamped to the coarse
  cell so ``k_ext <= k_maj``)."""
  dtype = pos_in.dtype
  z_sfc = maj.zc_faces[0]
  z_tod = maj.zc_faces[-1]
  dxc = maj.bx * geom.dx
  dyc = maj.by * geom.dy
  bx, by, bz = maj.bx, maj.by, maj.bz
  ncx, ncy, ncz = maj.ncx, maj.ncy, maj.ncz
  x, y, z = pos_in[0], pos_in[1], pos_in[2]
  ux, uy, uz = dir_in[0], dir_in[1], dir_in[2]
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
  pos = pos_in + dist * dir_in

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
  return pos, cix_n, ciy_n, ciz_n, collide_first, at_tod, at_sfc, ix, iy, iz


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
    *,
    free_path_floor_sharpness: float,
) -> PhotonState:
  """Walk one photon to termination via coarse-grid (decomposition) Woodcock
  tracking; return its terminal ``PhotonState``.

  The free path is sampled with the LOCAL coarse-cell majorant. The flight stops
  at the first of: a real/null collision, a coarse-cell face (resample majorant),
  or a domain z edge (surface / TOD). Delta tracking is unbiased for any
  per-region majorant >= local k_ext, so this is a pure speedup over a single
  global majorant. ``maj`` block sizes are static Python ints (closed over vmap).
  """
  dtype = k_ext.dtype
  cix0, ciy0, ciz0 = _coarse_indices(pos0, maj, geom)

  init = PhotonState(
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

  def cond(s: PhotonState) -> Array:
    return s.alive

  def body(s: PhotonState) -> PhotonState:
    key, k_path, k_real, k_abs, k_scat, k_sfc, k_refl = jax.random.split(s.key, 7)
    cix, ciy, ciz = s.cix, s.ciy, s.ciz
    k_maj = maj.grid[cix, ciy, ciz]

    t_coll = sampling.free_path(k_path, k_maj, floor_sharpness=free_path_floor_sharpness)
    (pos, cix_n, ciy_n, ciz_n, collide_first, at_tod, at_sfc,
     ix, iy, iz) = _flight(s.pos, s.dir, cix, ciy, ciz, t_coll, maj, geom)
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

    return PhotonState(
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
    *,
    free_path_floor_sharpness: float,
) -> PhotonState:
  """``vmap`` ``trace_one`` over a photon batch (leading axis on
  ``keys``/``pos0``/``dir0``). ``maj``/fields/geom are CLOSED OVER (not mapped)
  so the majorant block sizes stay static Python ints inside ``trace_one``."""
  def _one(key, pos, dir_):
    return trace_one(key, pos, dir_, k_ext, ssa, g, maj, albedo, geom,
                     max_iterations, rayleigh_frac=rayleigh_frac,
                     mie_cdf=mie_cdf, mie_ang=mie_ang, r_eff=r_eff,
                     free_path_floor_sharpness=free_path_floor_sharpness)

  return jax.vmap(_one)(keys, pos0, dir0)


class WeightedTally(NamedTuple):
  """Weighted-photon tallies of one batch (units: launched-photon weight).

  microhh ``ray_tracer_kernel`` count arrays. ``abs_*`` are flat cells
  ``(ix*ny+iy)*nz+iz``; the rest flat columns ``ix*ny+iy``. ``residual`` is the
  weight still in flight at the step cap (diagnostic; 0 in a converged walk).
  Per realization: ``sum(abs_dir+abs_dif) + sum(sfc_dir+sfc_dif-sfc_up)
  + sum(tod_up) + residual == number of photons launched``.
  """

  abs_dir: Array
  abs_dif: Array
  sfc_dir: Array
  sfc_dif: Array
  sfc_up: Array
  tod_up: Array
  residual: Array
  n_capped: Array


def trace_batch_weighted(
    key: Array,
    pos0: Array,
    dir0: Array,
    direct0: Array,
    k_ext: Array,
    ssa: Array,
    g: Array,
    maj: "knull_grid.MajorantGrid",
    albedo: Array,
    geom: PlaneRTGeometry,
    max_iterations: int,
    weight_floor: float,
    rayleigh_frac: Array | None = None,
    mie_cdf: Array | None = None,
    mie_ang: Array | None = None,
    r_eff: Array | None = None,
    *,
    free_path_floor_sharpness: float,
) -> WeightedTally:
  """Walk a photon batch with microhh's weighted estimator (raytracer_kernels.cu
  ``ray_tracer_kernel``, v0.3 fc291af) and return its tallies.

  Batch-synchronous: one ``while_loop`` iteration advances every live photon by
  one flight and scatter-adds its deposits into the grid tallies.

  At every (real or null) collision a photon of weight ``w`` deposits
  ``w (1 - f)`` with ``f = 1 - (1 - ssa) k_ext / k_null`` and keeps ``w f``;
  it then scatters with probability ``ssa k_ext / (k_null f)``, else it is a
  null collision. At the surface it deposits ``w`` (direct or diffuse), keeps
  ``w albedo``, adds that to the upward surface flux and reflects
  Lambertian. Leaving through the top adds ``w`` to ``tod_up``.

  DEPARTURE: microhh's Russian roulette (``w < 0.5`` -> ``w`` becomes 0 or 1)
  conserves energy only in expectation. Here a photon whose weight drops below
  ``weight_floor`` instead deposits its remainder where it is (volume cell, or
  the surface for a surface hit) and stops, so the budget closes per
  realization to round-off; the bias is at most ``weight_floor`` of a photon.
  Gas scattering is Rayleigh, cloud scattering Mie (LUT) or HG; microhh's
  aerosol arm is not ported (the plane path carries no aerosol optics).
  """
  dtype = k_ext.dtype
  nb = pos0.shape[0]
  ncells = geom.nx * geom.ny * geom.nz
  ncols = geom.nx * geom.ny
  cix0, ciy0, ciz0 = _coarse_indices(pos0, maj, geom)
  free_path = jax.vmap(
      lambda k, m: sampling.free_path(
          k, m, floor_sharpness=free_path_floor_sharpness))
  if rayleigh_frac is None:
    scatter = jax.vmap(sampling.scatter_direction)
  else:
    scatter = jax.vmap(
        lambda k, d, gg, rf, re: sampling.scatter_direction_mixed(
            k, d, gg, rf, mie_cdf=mie_cdf, mie_ang=mie_ang, r_eff=re))
  reflect_dir = jax.vmap(lambda k: sampling.lambertian_reflect(k, dtype))
  flight = jax.vmap(
      lambda p, d, a, b, c, t: _flight(p, d, a, b, c, t, maj, geom))
  zeros_c = jnp.zeros((ncells,), dtype)
  zeros_s = jnp.zeros((ncols,), dtype)

  def cond(c):
    return jnp.any(c[5]) & (c[6] < max_iterations)

  def body(c):
    (key, pos, dir_, ci, w, alive, n, direct, t) = c
    key, k_path, k_real, k_scat, k_refl = jax.random.split(key, 5)
    k_maj = maj.grid[ci[0], ci[1], ci[2]]
    t_coll = free_path(jax.random.split(k_path, nb), k_maj)
    (pos_n, cix_n, ciy_n, ciz_n, coll, at_tod, at_sfc,
     ix, iy, iz) = flight(pos, dir_, ci[0], ci[1], ci[2], t_coll)
    cell = (ix * geom.ny + iy) * geom.nz + iz
    col = ix * geom.ny + iy
    coll = alive & coll
    at_tod = alive & at_tod
    at_sfc = alive & at_sfc

    # --- collision: weighted absorption (Iwabuchi 2006, as microhh) ---
    ke = k_ext[ix, iy, iz]
    om = ssa[ix, iy, iz]
    # a = 1 - f_no_abs, computed directly: microhh's 1 - (1 - a) cancels in fp32.
    a = (1.0 - om) * (ke / k_maj)
    f = 1.0 - a
    w_c = w * f
    floor_c = coll & (w_c < weight_floor)
    dep = jnp.where(coll, w * a + jnp.where(floor_c, w_c, 0.0), 0.0)
    safe_f = jnp.where(f > 0.0, f, 1.0)
    p_scat = om * (ke / k_maj) / safe_f
    u_real = jax.random.uniform(k_real, (nb,), dtype=dtype)
    scat = coll & (~floor_c) & (u_real < p_scat)
    if rayleigh_frac is None:
      s_dir = scatter(jax.random.split(k_scat, nb), dir_, g[ix, iy, iz])
    else:
      re = (r_eff[ix, iy, iz] if r_eff is not None
            else jnp.zeros((nb,), dtype))
      s_dir = scatter(jax.random.split(k_scat, nb), dir_, g[ix, iy, iz],
                      rayleigh_frac[ix, iy, iz], re)

    # --- surface: deposit, keep w*albedo, Lambertian reflection ---
    w_s = w * albedo
    floor_s = at_sfc & (w_s < weight_floor)
    w_up = jnp.where(at_sfc & (~floor_s), w_s, 0.0)
    r_dir = reflect_dir(jax.random.split(k_refl, nb))

    w_sfc = jnp.where(at_sfc, w, 0.0)
    t = WeightedTally(
        abs_dir=t.abs_dir.at[cell].add(jnp.where(direct, dep, 0.0)),
        abs_dif=t.abs_dif.at[cell].add(jnp.where(direct, 0.0, dep)),
        sfc_dir=t.sfc_dir.at[col].add(jnp.where(direct, w_sfc, 0.0)),
        sfc_dif=t.sfc_dif.at[col].add(jnp.where(direct, 0.0, w_sfc)),
        sfc_up=t.sfc_up.at[col].add(w_up),
        tod_up=t.tod_up.at[col].add(jnp.where(at_tod, w, 0.0)),
        residual=t.residual, n_capped=t.n_capped)

    bounce = at_sfc & (~floor_s)
    new_dir = jnp.where(scat[:, None], s_dir, dir_)
    new_dir = jnp.where(bounce[:, None], r_dir, new_dir)
    w_n = jnp.where(coll, w_c, jnp.where(bounce, w_s, w))
    alive_n = alive & (~floor_c) & (~at_tod) & (~(at_sfc & floor_s))
    direct_n = direct & (~scat) & (~bounce)
    pos_n = jnp.where(alive[:, None], pos_n, pos)
    ci_n = tuple(jnp.where(alive, a, b) for a, b in
                 zip((cix_n, ciy_n, ciz_n), ci))
    return (key, pos_n, new_dir, ci_n, w_n, alive_n, n + 1, direct_n, t)

  init_t = WeightedTally(zeros_c, zeros_c, zeros_s, zeros_s, zeros_s, zeros_s,
                         jnp.zeros((), dtype), jnp.zeros((), jnp.int32))
  init = (key, pos0.astype(dtype), dir0.astype(dtype), (cix0, ciy0, ciz0),
          jnp.ones((nb,), dtype), jnp.ones((nb,), bool),
          jnp.zeros((), jnp.int32), direct0.astype(bool), init_t)
  out = jax.lax.while_loop(cond, body, init)
  w, alive, t = out[4], out[5], out[8]
  return t._replace(residual=jnp.sum(jnp.where(alive, w, 0.0)),
                    n_capped=jnp.sum(alive).astype(jnp.int32))
