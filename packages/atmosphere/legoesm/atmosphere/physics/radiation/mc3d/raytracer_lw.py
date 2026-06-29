"""Longwave 3D Monte-Carlo radiative transfer by forward thermal-emission MC.

LW scattering is neglected (the dominant LW process is absorption/emission), so
the transport reduces to: emit photons from the volumetric Planck source
(``4pi k_abs B dV`` per cell) and the surface (``pi emissivity B_sfc dA``), walk
each to its FIRST real collision (absorption) / surface / TOD escape, and form
the net per-cell flux divergence as ``epsilon * (absorbed - emitted)``.

This REUSES ``photon_walk.trace_one`` with ``ssa=0`` (every real collision is an
absorption — no scattering) and ``albedo = 1 - emissivity`` (LW surface
reflection). Energy is conserved by construction: total emitted = total absorbed
(volume + surface) + escaped-to-space, so
``sum(net_atmos) + net_surface + OLR == 0`` to MC error.

See ``docs/specs/mc3d_raytracer.md``. Phase 3 uses gray (broadband) Planck +
optics; RRTMGP-spectral LW optics/Planck = Phase 3b.
"""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.radiation.mc3d import knull_grid
from legoesm.atmosphere.physics.radiation.mc3d import photon_walk
from legoesm.atmosphere.physics.radiation.mc3d import sampling
from legoesm.atmosphere.physics.radiation.mc3d.config import MC3DRadiationConfig
from legoesm.atmosphere.physics.radiation.mc3d.photon_walk import (
    PlaneRTGeometry,
    STATUS_SFC_ABS,
    STATUS_TOD_UP,
    STATUS_VOL_ABS,
)

Array: TypeAlias = jax.Array

# Solid-angle emission factors: isotropic volume emission integrates 4*pi over
# the sphere; a Lambertian surface integrates pi over the upward hemisphere.
_FOUR_PI = 4.0 * jnp.pi
_PI = jnp.pi


class LWResult(NamedTuple):
  """Longwave net fluxes [W/m^2] from the emission MC."""

  net_flux: Array   # (nx,ny,nz) net absorbed-minus-emitted per cell / ground area
  sfc_net: Array    # (nx,ny) net surface LW (absorbed - emitted) / ground area
  olr: Array        # scalar domain-mean outgoing LW (TOD escape) [W/m^2]
  maxiter_frac: Array  # diagnostic


def solve_lw_monochromatic(
    k_abs: Array,
    planck_cell: Array,
    planck_sfc: Array,
    geom: PlaneRTGeometry,
    *,
    emissivity: float,
    config: MC3DRadiationConfig,
    key: Array,
    planck_lower: Array | None = None,
    planck_upper: Array | None = None,
) -> LWResult:
  """Forward thermal-emission LW Monte-Carlo over one (gray/g-point) band.

  Args:
    k_abs: LW absorption coefficient ``(nx,ny,nz)`` [1/m].
    planck_cell: Planck radiance ``B(T)`` per cell ``(nx,ny,nz)`` [W/m^2/sr].
      Used as the cell emission weight when ``planck_lower/upper`` are not given
      (piecewise-constant in-cell emission).
    planck_sfc: surface Planck radiance ``(nx,ny)`` [W/m^2/sr].
    geom: plane geometry.
    emissivity: surface LW emissivity (reflection albedo = 1 - emissivity).
    config: MC numerics.
    key: PRNG key.
    planck_lower, planck_upper: optional per-cell Planck radiance at the LOWER
      (z_faces[iz]) and UPPER (z_faces[iz+1]) cell faces ``(nx,ny,nz)``. When
      BOTH are supplied, in-cell emission is LINEAR-in-z (Phase 3c): the cell
      weight uses the face mean and the emission depth is sampled proportional to
      the linear Planck profile, so photons escaping to space carry the cooler
      cell-top temperature (removes the cell-center over-emission bias).
  """
  dtype = k_abs.dtype
  nx, ny, nz = geom.nx, geom.ny, geom.nz
  ncells, ncols = nx * ny * nz, nx * ny
  p = int(config.photons_per_pixel)
  n_total = ncols * p
  area_cell = geom.dx * geom.dy

  dz = jnp.diff(geom.z_faces).astype(dtype)            # (nz,)
  dV = area_cell * dz[None, None, :]                   # (nx,ny,nz)

  # Emission power per emitter [W]. With face Planck values the per-cell weight
  # uses the face MEAN (= integral of a linear profile); else the cell value.
  linear_emission = planck_lower is not None and planck_upper is not None
  b_weight = (0.5 * (planck_lower + planck_upper) if linear_emission
              else planck_cell)
  e_vol = _FOUR_PI * k_abs * b_weight * dV             # (nx,ny,nz)
  e_sfc = _PI * emissivity * planck_sfc * area_cell    # (nx,ny)
  e_flat = jnp.concatenate([e_vol.reshape(-1), e_sfc.reshape(-1)])  # (ncells+ncols,)
  e_tot = jnp.sum(e_flat)
  n_emit = ncells + ncols
  # Guard an all-cold (zero-emission) domain: fall back to a uniform categorical
  # so jax.random.choice has a valid distribution; eps=0 then zeroes every
  # tally, so the chosen origins are irrelevant. The denominator is made finite
  # in BOTH jnp.where arms (both are evaluated) -> no e_flat/0 NaN, dtype-safe
  # (a 1e-300 floor would underflow to 0 in float32).
  e_tot_safe = jnp.where(e_tot > 0.0, e_tot, jnp.ones((), e_flat.dtype))
  prob = jnp.where(e_tot > 0.0, e_flat / e_tot_safe,
                   jnp.full((n_emit,), 1.0 / n_emit, dtype=e_flat.dtype))
  eps = e_tot / n_total                                # energy per photon [W]

  k_choose, k_pos, k_dir, k_walk, k_lin = jax.random.split(key, 5)
  origin = jax.random.choice(
      k_choose, ncells + ncols, shape=(n_total,), p=prob)
  is_vol = origin < ncells

  # Decode volume origin -> (ix,iy,iz); surface origin -> (ix,iy).
  v = jnp.clip(origin, 0, ncells - 1)
  vix = v // (ny * nz)
  vrem = v % (ny * nz)
  viy = vrem // nz
  viz = vrem % nz
  s = jnp.clip(origin - ncells, 0, ncols - 1)
  six = s // ny
  siy = s % ny

  ix = jnp.where(is_vol, vix, six)
  iy = jnp.where(is_vol, viy, siy)

  u = jax.random.uniform(k_pos, (n_total, 3), dtype=dtype)
  x = (ix.astype(dtype) + u[:, 0]) * geom.dx
  y = (iy.astype(dtype) + u[:, 1]) * geom.dy
  zlo = geom.z_faces[viz]
  zhi = geom.z_faces[jnp.minimum(viz + 1, nz)]
  if linear_emission:
    # Sample in-cell emission depth ~ linear Planck profile (lower->upper face)
    # via the inverse CDF of p(f) ∝ (1-f) B_lo + f B_hi on f in [0,1]:
    #   (B_hi-B_lo)/2 f^2 + B_lo f - uf*mean = 0.
    b_lo = planck_lower[vix, viy, viz]
    b_hi = planck_upper[vix, viy, viz]
    uf = jax.random.uniform(k_lin, (n_total,), dtype=dtype)
    mean_b = 0.5 * (b_lo + b_hi)
    diff = b_hi - b_lo
    disc = jnp.maximum(b_lo * b_lo + 2.0 * diff * uf * mean_b, 0.0)
    safe_diff = jnp.where(jnp.abs(diff) > 1e-30, diff, 1.0)
    f_lin = jnp.where(jnp.abs(diff) > 1e-30,
                      (jnp.sqrt(disc) - b_lo) / safe_diff, uf)
    f_lin = jnp.clip(f_lin, 0.0, 1.0)
  else:
    f_lin = u[:, 2]
  z_vol = zlo + f_lin * (zhi - zlo)
  z = jnp.where(is_vol, z_vol, geom.z_faces[0])         # surface emits at z_sfc
  pos0 = jnp.stack([x, y, z], axis=1)

  # Directions: isotropic for volume emission, cosine-up for surface emission.
  dkeys = jax.random.split(k_dir, n_total)
  def _dir(dk, vol):
    return jnp.where(vol, sampling.isotropic_direction(dk, dtype),
                     sampling.lambertian_reflect(dk, dtype))
  dir0 = jax.vmap(_dir)(dkeys, is_vol)

  ssa0 = jnp.zeros_like(k_abs)                          # no LW scattering
  g0 = jnp.zeros_like(k_abs)
  maj = knull_grid.build_majorant_grid(
      k_abs, geom.z_faces, config.knull_floor,
      coarsen_xy=int(config.knull_coarsen_xy),
      coarsen_z=int(config.knull_coarsen_z))
  albedo_lw = 1.0 - emissivity
  walk_keys = jax.random.split(k_walk, n_total)
  states = photon_walk.trace_batch(
      walk_keys, pos0, dir0, k_abs, ssa0, g0, maj, albedo_lw, geom,
      int(config.max_iterations))

  count_dt = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
  ones = jnp.ones((n_total,), count_dt)
  # Emitted-per-cell (volume) histogram from origins.
  emit_ids = jnp.where(is_vol, v, -1)
  emit_cnt = jax.ops.segment_sum(ones, emit_ids, num_segments=ncells)
  emit_sfc_ids = jnp.where(~is_vol, s, -1)
  emit_sfc_cnt = jax.ops.segment_sum(ones, emit_sfc_ids, num_segments=ncols)
  # Absorbed-per-cell histograms from the walk.
  st = states.status
  abs_ids = jnp.where(st == STATUS_VOL_ABS, states.vol_idx, -1)
  abs_cnt = jax.ops.segment_sum(ones, abs_ids, num_segments=ncells)
  abs_sfc_ids = jnp.where(st == STATUS_SFC_ABS, states.sfc_idx, -1)
  abs_sfc_cnt = jax.ops.segment_sum(ones, abs_sfc_ids, num_segments=ncols)
  escaped = jnp.sum(st == STATUS_TOD_UP).astype(count_dt)
  maxiter = jnp.sum(st == photon_walk.STATUS_MAXITER).astype(count_dt)

  net = (abs_cnt - emit_cnt).astype(dtype) * eps        # [W] per cell
  net_flux = net.reshape(nx, ny, nz) / area_cell        # [W/m^2] per ground area
  sfc_net = (abs_sfc_cnt - emit_sfc_cnt).astype(dtype) * eps
  sfc_net = sfc_net.reshape(nx, ny) / area_cell
  # OLR counts ONLY confirmed TOD escapes. Max-iter photons have an UNKNOWN fate
  # (not confirmed escaped), so they are NOT folded into OLR -- that would
  # over-count outgoing LW. Their energy is the "unresolved" budget term
  # (maxiter * eps): sum(net_atmos)*A + net_sfc*A + OLR*A + maxiter*eps == 0.
  # maxiter_frac surfaces it; it should be ~0 (raise the iteration cap if not).
  olr = escaped.astype(dtype) * eps / (geom.Lx * geom.Ly)
  return LWResult(
      net_flux=net_flux, sfc_net=sfc_net, olr=olr,
      maxiter_frac=maxiter.astype(dtype) / n_total)


def solve_lw_spectral(
    abs_optical_depth: Array,
    planck_src: Array,
    planck_src_sfc: Array,
    geom: PlaneRTGeometry,
    *,
    emissivity,
    config: MC3DRadiationConfig,
    key: Array,
    planck_src_lower: Array | None = None,
    planck_src_upper: Array | None = None,
) -> LWResult:
  """Spectrally-integrated longwave emission MC over a stack of g-points.

  Args:
    abs_optical_depth: per-g-point layer ABSORPTION optical depth
      ``(ngpt, nx, ny, nz)`` [-].
    planck_src: per-g-point cell-center Planck RADIANCE ``(ngpt, nx, ny, nz)``
      [W/m^2/sr]. RRTMGP's ``compute_planck_sources`` returns radiance B (the
      validated two-stream multiplies it by pi to form the surface flux), so it
      is passed straight to ``solve_lw_monochromatic`` which applies the 4pi
      (volume) / pi (surface) emission factors on radiance.
    planck_src_sfc: per-g-point surface Planck radiance ``(ngpt, nx, ny)``.
    geom: plane geometry. emissivity: surface LW emissivity.

  Net fluxes are summed over g-points (each band is an independent emission MC
  problem). Distinct RNG per g-point.
  """
  dtype = abs_optical_depth.dtype
  nx, ny, nz = geom.nx, geom.ny, geom.nz
  dz = jnp.diff(geom.z_faces).astype(dtype)
  ngpt = abs_optical_depth.shape[0]
  gkeys = jax.random.split(key, ngpt)
  # Phase-3c linear-in-tau emission when face Planck values are supplied.
  linear = planck_src_lower is not None and planck_src_upper is not None
  lower = planck_src_lower if linear else jnp.zeros_like(abs_optical_depth)
  upper = planck_src_upper if linear else jnp.zeros_like(abs_optical_depth)

  def gpt_step(carry, inp):
    net_acc, sfc_acc, olr_acc, max_acc = carry
    od_g, pl_g, plsfc_g, lo_g, up_g, gkey = inp
    k_abs = od_g / dz[None, None, :]
    # planck_src is radiance B; solve_lw_monochromatic applies the 4pi/pi
    # emission factors on radiance directly (no 1/pi conversion).
    res = solve_lw_monochromatic(
        k_abs, pl_g, plsfc_g, geom,
        emissivity=emissivity, config=config, key=gkey,
        planck_lower=lo_g if linear else None,
        planck_upper=up_g if linear else None)
    return (net_acc + res.net_flux, sfc_acc + res.sfc_net,
            olr_acc + res.olr, max_acc + res.maxiter_frac), None

  init = (jnp.zeros((nx, ny, nz), dtype), jnp.zeros((nx, ny), dtype),
          jnp.zeros((), dtype), jnp.zeros((), dtype))
  (net_acc, sfc_acc, olr_acc, max_acc), _ = jax.lax.scan(
      gpt_step, init,
      (abs_optical_depth, planck_src, planck_src_sfc, lower, upper, gkeys))
  return LWResult(net_flux=net_acc, sfc_net=sfc_acc, olr=olr_acc,
                  maxiter_frac=max_acc / ngpt)
