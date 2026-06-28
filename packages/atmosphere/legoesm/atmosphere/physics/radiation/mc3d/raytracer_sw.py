"""Shortwave 3D Monte-Carlo ray-tracing driver (single monochromatic field).

Launches a collimated solar beam at the top of the domain, one batch of photons
at a time (``lax.scan`` over ``n_batches`` to bound peak memory), walks them with
``photon_walk.trace_batch``, and accumulates analog tallies into an
``MC3DResult``. The g-point / spectral loop and the conversion of fractions to
physical W/m^2 + heating rates live in the integration layer (Phase 2); this
module is the spatial solver for one optical field.
"""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.radiation.mc3d import knull_grid
from legoesm.atmosphere.physics.radiation.mc3d import photon_walk
from legoesm.atmosphere.physics.radiation.mc3d import qrng
from legoesm.atmosphere.physics.radiation.mc3d import tally
from legoesm.atmosphere.physics.radiation.mc3d.config import MC3DRadiationConfig
from legoesm.atmosphere.physics.radiation.mc3d.photon_walk import (
    PlaneRTGeometry,
    STATUS_MAXITER,
    STATUS_SFC_ABS,
    STATUS_TOD_UP,
    STATUS_VOL_ABS,
)
from legoesm.atmosphere.physics.radiation.mc3d.tally import MC3DResult

Array: TypeAlias = jax.Array


class SWFluxResult(NamedTuple):
  """Spectrally-integrated shortwave MC fluxes [W/m^2]."""

  abs_flux: Array       # (nx,ny,nz) absorbed flux per cell (per ground area)
  sfc_abs_flux: Array   # (nx,ny) net SW absorbed at the surface
  tod_up_flux: Array    # scalar domain-mean TOD upwelling (reflected) flux


def _beam_direction(mu0: float, azimuth: float, dtype) -> Array:
  """Downward collimated solar beam unit vector (z up). ``mu0 = cos(zenith)``."""
  sin_z = (1.0 - mu0 * mu0) ** 0.5
  return jnp.asarray(
      [sin_z * jnp.cos(azimuth), sin_z * jnp.sin(azimuth), -mu0], dtype=dtype
  )


def solve_sw_monochromatic(
    k_ext: Array,
    ssa: Array,
    g: Array,
    geom: PlaneRTGeometry,
    *,
    mu0: float,
    azimuth: float,
    albedo: float,
    config: MC3DRadiationConfig,
    key: Array,
    rayleigh_frac: Array | None = None,
    mie_cdf: Array | None = None,
    mie_ang: Array | None = None,
    r_eff: Array | None = None,
) -> MC3DResult:
  """Trace the shortwave beam through one optical field; return analog tallies.

  ``rayleigh_frac`` (optional ``(nx,ny,nz)``): local fraction of total scattering
  that is gas Rayleigh; when given, scatter events split into an explicit
  Rayleigh (1+mu^2) phase vs HG with the cloud asymmetry (see
  ``sampling.scatter_direction_mixed``). Default ``None`` -> pure HG(g).

  Args:
    k_ext: extinction ``(nx,ny,nz)`` [1/m] (= optical_depth / layer thickness).
    ssa: single-scattering albedo ``(nx,ny,nz)`` in [0,1].
    g: asymmetry factor ``(nx,ny,nz)`` in (-1,1).
    geom: plane geometry.
    mu0: cosine of solar zenith angle (>0).
    azimuth: solar azimuth [rad].
    albedo: Lambertian surface albedo in [0,1].
    config: MC numerics.
    key: PRNG key.
  """
  dtype = k_ext.dtype
  nx, ny, nz = geom.nx, geom.ny, geom.nz
  ncells, ncols = nx * ny * nz, nx * ny
  p = int(config.photons_per_pixel)
  n_total = ncols * p
  nb = int(config.n_batches)
  if n_total % nb != 0:
    raise ValueError(
        f"n_batches={nb} must divide total photons N={n_total} "
        f"(nx*ny*photons_per_pixel)."
    )
  batch_size = n_total // nb

  if bool(config.use_qrng) and p > 2 ** qrng._N_DIGITS_BASE2:
    # Beyond the Halton digit capacity the launch sequence ALIASES (high-order
    # photon-index digits dropped) -> the low-discrepancy benefit silently
    # vanishes. Refuse loudly rather than degrade silently (no bias either way).
    raise ValueError(
        f"use_qrng with photons_per_pixel={p} exceeds the Halton base-2 "
        f"capacity 2^{qrng._N_DIGITS_BASE2}; reduce photons_per_pixel or "
        "raise qrng._N_DIGITS_BASE2."
    )

  # Counts are tallied as integers for exact conservation; refuse loudly if they
  # would overflow int32 without x64 (no silent coerce). See assert_count_capacity.
  tally.assert_count_capacity(n_total, x64_enabled=jax.config.jax_enable_x64)

  maj = knull_grid.build_majorant_grid(
      k_ext, geom.z_faces, config.knull_floor,
      coarsen_xy=int(config.knull_coarsen_xy),
      coarsen_z=int(config.knull_coarsen_z),
  )
  beam = _beam_direction(mu0, azimuth, dtype)
  z_tod = geom.z_faces[-1]

  # Per-photon column assignment (cheap int arrays; float state is per-batch).
  col = jnp.arange(ncols, dtype=jnp.int32)
  col_ix, col_iy = col // ny, col % ny
  photon_col = jnp.repeat(jnp.arange(ncols, dtype=jnp.int32), p)
  ix_all = col_ix[photon_col].reshape(nb, batch_size)
  iy_all = col_iy[photon_col].reshape(nb, batch_size)
  # Photon-within-column index + column index (for the quasi-random launch).
  pwc_all = (jnp.arange(n_total, dtype=jnp.int32) % p).reshape(nb, batch_size)
  col_all = photon_col.reshape(nb, batch_size)
  use_qrng = bool(config.use_qrng)
  # Per-column Cranley-Patterson rotation (decorrelates the columns' Halton
  # sets). Only consume a key when QRNG is on, so the default (pseudo-random)
  # path's RNG stream is unchanged.
  if use_qrng:
    key, k_rot = jax.random.split(key)
    col_rot = jax.random.uniform(k_rot, (ncols, 2), dtype=dtype)
  else:
    col_rot = jnp.zeros((ncols, 2), dtype=dtype)

  # Tally counts as integers so the analog-MC "sums to exactly 1" invariant is
  # not eroded by float rounding above 2^24 photons (float32) / 2^53 (float64).
  count_dt = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
  ones = jnp.ones((batch_size,), dtype=count_dt)

  def batch_step(carry, batch):
    vol_c, sfc_c, tod_c, max_c = carry
    ix_b, iy_b, pwc_b, col_b, bkey = batch
    k_pos, k_walk = jax.random.split(bkey)
    if use_qrng:
      # Low-discrepancy launch position (Halton + per-column rotation).
      hx, hy = qrng.halton_2d(pwc_b, col_rot[col_b])
      ux_, uy_ = hx, hy
    else:
      u = jax.random.uniform(k_pos, (batch_size, 2), dtype=dtype)
      ux_, uy_ = u[:, 0], u[:, 1]
    x = (ix_b.astype(dtype) + ux_) * geom.dx
    y = (iy_b.astype(dtype) + uy_) * geom.dy
    z = jnp.full((batch_size,), z_tod, dtype=dtype)
    pos0 = jnp.stack([x, y, z], axis=1)
    dir0 = jnp.broadcast_to(beam, (batch_size, 3))
    walk_keys = jax.random.split(k_walk, batch_size)

    states = photon_walk.trace_batch(
        walk_keys, pos0, dir0, k_ext, ssa, g, maj, albedo, geom,
        int(config.max_iterations), rayleigh_frac=rayleigh_frac,
        mie_cdf=mie_cdf, mie_ang=mie_ang, r_eff=r_eff,
    )
    st = states.status
    vol_ids = jnp.where(st == STATUS_VOL_ABS, states.vol_idx, -1)
    sfc_ids = jnp.where(st == STATUS_SFC_ABS, states.sfc_idx, -1)
    vol_c = vol_c + jax.ops.segment_sum(ones, vol_ids, num_segments=ncells)
    sfc_c = sfc_c + jax.ops.segment_sum(ones, sfc_ids, num_segments=ncols)
    tod_c = tod_c + jnp.sum(st == STATUS_TOD_UP).astype(count_dt)
    max_c = max_c + jnp.sum(st == STATUS_MAXITER).astype(count_dt)
    return (vol_c, sfc_c, tod_c, max_c), None

  bkeys = jax.random.split(key, nb)
  init = (
      jnp.zeros((ncells,), count_dt),
      jnp.zeros((ncols,), count_dt),
      jnp.zeros((), count_dt),
      jnp.zeros((), count_dt),
  )
  (vol_c, sfc_c, tod_c, max_c), _ = jax.lax.scan(
      batch_step, init, (ix_all, iy_all, pwc_all, col_all, bkeys)
  )

  inv_p = 1.0 / p
  inv_n = 1.0 / n_total
  # Max-iter photons are folded into the escaped (TOD-up) bin so the physical
  # budget closes exactly: vol_abs_total + sfc_abs_total + tod_up_frac == 1.
  # maxiter_frac is a diagnostic SUBSET of tod_up_frac (warn if non-negligible).
  escaped = tod_c + max_c
  return MC3DResult(
      abs_frac=vol_c.reshape(nx, ny, nz).astype(dtype) * inv_p,
      sfc_abs_frac=sfc_c.reshape(nx, ny).astype(dtype) * inv_p,
      tod_up_frac=escaped.astype(dtype) * inv_n,
      vol_abs_total=jnp.sum(vol_c).astype(dtype) * inv_n,
      sfc_abs_total=jnp.sum(sfc_c).astype(dtype) * inv_n,
      maxiter_frac=max_c.astype(dtype) * inv_n,
  )


def solve_sw_spectral(
    tau: Array,
    ssa: Array,
    g: Array,
    incident_flux: Array,
    geom: PlaneRTGeometry,
    *,
    mu0: float,
    azimuth: float,
    albedo: float,
    config: MC3DRadiationConfig,
    key: Array,
    rayleigh_frac: Array | None = None,
    mie_cdf_band: Array | None = None,
    mie_ang_band: Array | None = None,
    r_eff: Array | None = None,
) -> SWFluxResult:
  """Spectrally-integrated shortwave MC transport over a stack of g-points.

  ``rayleigh_frac`` (optional ``(ngpt, nx, ny, nz)``): per-g-point gas Rayleigh
  fraction of total scattering, enabling the explicit Rayleigh + cloud phase
  split. Default ``None`` -> pure HG(g). When ``mie_cdf_band``
  ``(ngpt, n_mie)`` / ``mie_ang_band`` ``(ngpt, n_r, n_mie)`` (the per-g-point
  Mie LUT band slices) and ``r_eff`` ``(nx,ny,nz)`` [um] are given, the cloud
  phase is the microhh Mie LUT instead of HG.

  Loops g-points serially (``lax.scan``) so only one g-point's optical field is
  resident at a time -- the key memory lever (see docs/specs). Each g-point's
  per-photon fractions are weighted by its incident vertical TOA flux and
  accumulated into physical W/m^2 fields.

  Args:
    tau: per-g-point layer optical depth ``(ngpt, nx, ny, nz)`` [-].
    ssa: single-scattering albedo ``(ngpt, nx, ny, nz)`` [-].
    g: asymmetry factor ``(ngpt, nx, ny, nz)`` [-].
    incident_flux: per-g-point downward (vertical) TOA flux ``(ngpt,)`` [W/m^2].
    geom: plane geometry (``z_faces`` gives layer thicknesses).
    mu0: cosine of solar zenith angle (sets beam slant; the incident_flux is
      already the vertical component).
    azimuth: solar azimuth [rad].
    albedo: Lambertian surface albedo.
    config: MC numerics.
    key: PRNG key (split per g-point).
  """
  dtype = tau.dtype
  nx, ny, nz = geom.nx, geom.ny, geom.nz
  dz = jnp.diff(geom.z_faces).astype(dtype)  # (nz,) [m]
  ngpt = tau.shape[0]
  gkeys = jax.random.split(key, ngpt)
  use_rayleigh = rayleigh_frac is not None
  rfrac = rayleigh_frac if use_rayleigh else jnp.zeros_like(tau)
  use_mie = (mie_cdf_band is not None and mie_ang_band is not None
             and r_eff is not None)
  cdf_in = mie_cdf_band if use_mie else jnp.zeros((ngpt, 1), dtype)
  ang_in = mie_ang_band if use_mie else jnp.zeros((ngpt, 1, 1), dtype)

  def gpt_step(carry, inp):
    abs_acc, sfc_acc, tod_acc = carry
    tau_g, ssa_g, g_g, f_in, rf_g, cdf_g, ang_g, gkey = inp
    # Extinction [1/m] from layer optical depth; broadcast dz over (nx,ny,nz).
    k_ext = tau_g / dz[None, None, :]
    res = solve_sw_monochromatic(
        k_ext, ssa_g, g_g, geom, mu0=mu0, azimuth=azimuth, albedo=albedo,
        config=config, key=gkey,
        rayleigh_frac=rf_g if use_rayleigh else None,
        mie_cdf=cdf_g if use_mie else None,
        mie_ang=ang_g if use_mie else None,
        r_eff=r_eff if use_mie else None)
    abs_acc = abs_acc + res.abs_frac * f_in
    sfc_acc = sfc_acc + res.sfc_abs_frac * f_in
    tod_acc = tod_acc + res.tod_up_frac * f_in
    return (abs_acc, sfc_acc, tod_acc), None

  init = (
      jnp.zeros((nx, ny, nz), dtype),
      jnp.zeros((nx, ny), dtype),
      jnp.zeros((), dtype),
  )
  (abs_acc, sfc_acc, tod_acc), _ = jax.lax.scan(
      gpt_step, init,
      (tau, ssa, g, incident_flux.astype(dtype), rfrac, cdf_in, ang_in, gkeys)
  )
  return SWFluxResult(
      abs_flux=abs_acc, sfc_abs_flux=sfc_acc, tod_up_flux=tod_acc
  )
