"""Shortwave 3D Monte-Carlo ray-tracing driver (single monochromatic field).

Launches photons at the top of the domain (collimated beam, plus a diffuse
cosine-weighted fraction as microhh's ``tod_inc_diffuse``), one batch at a time
(``lax.scan`` over ``n_batches`` to bound peak memory), walks them with
microhh's weighted estimator (``photon_walk.trace_batch_weighted``) and returns
the tallies as an ``MC3DResult``. The g-point / spectral loop and the conversion of fractions to
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
from legoesm.atmosphere.physics.radiation.mc3d import sampling
from legoesm.atmosphere.physics.radiation.mc3d.config import MC3DRadiationConfig
from legoesm.atmosphere.physics.radiation.mc3d.photon_walk import PlaneRTGeometry
from legoesm.atmosphere.physics.radiation.mc3d.tally import MC3DResult

Array: TypeAlias = jax.Array


class SWFluxResult(NamedTuple):
  """Spectrally-integrated shortwave MC fluxes [W/m^2]."""

  abs_flux: Array       # (nx,ny,nz) absorbed flux per cell (per ground area)
  sfc_abs_flux: Array   # (nx,ny) net SW absorbed at the surface
  tod_up_flux: Array    # scalar domain-mean TOD upwelling (reflected) flux
  residual_flux: Array  # scalar domain-mean flux still in flight at the step cap


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
    diffuse_frac: float,
    config: MC3DRadiationConfig,
    key: Array,
    rayleigh_frac: Array | None = None,
    mie_cdf: Array | None = None,
    mie_ang: Array | None = None,
    r_eff: Array | None = None,
) -> MC3DResult:
  """Trace shortwave photons through one optical field; return the tallies.

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
    diffuse_frac: diffuse share of the top-of-domain incident flux (microhh
      ``tod_inc_diffuse / (tod_inc_direct + tod_inc_diffuse)``); 0 when the
      domain top is the top of the atmosphere.
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

  maj = knull_grid.build_majorant_grid(
      k_ext, geom.z_faces, config.knull_floor,
      coarsen_xy=int(config.knull_coarsen_xy),
      coarsen_z=int(config.knull_coarsen_z),
  )
  beam = _beam_direction(mu0, azimuth, dtype)
  z_tod = geom.z_faces[-1]

  use_qrng = bool(config.use_qrng)
  # Per-column Cranley-Patterson rotation (decorrelates the columns' Halton
  # sets). Only consume a key when QRNG is on, so the default (pseudo-random)
  # path's RNG stream is unchanged.
  if use_qrng:
    key, k_rot = jax.random.split(key)
    col_rot = jax.random.uniform(k_rot, (ncols, 2), dtype=dtype)
  else:
    col_rot = jnp.zeros((ncols, 2), dtype=dtype)

  if not 0.0 <= float(diffuse_frac) <= 1.0:
    raise ValueError(f"diffuse_frac={diffuse_frac} must lie in [0, 1].")

  def batch_step(acc, batch):
    b, bkey = batch
    # Derive this batch's photon -> (column, within-column, ix, iy) from the
    # global photon index instead of carrying four n_total int arrays as scan xs
    # (memory): photon i is in column i//p, so col=pid//p, ix=col//ny, iy=col%ny.
    pid = b * batch_size + jnp.arange(batch_size, dtype=jnp.int32)
    col_b = pid // p
    pwc_b = (pid % p).astype(jnp.int32)
    ix_b = col_b // ny
    iy_b = col_b % ny
    k_pos, k_kind, k_dif, k_walk = jax.random.split(bkey, 4)
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
    # microhh reset_photon: diffuse with probability tod_inc_diffuse/total,
    # cosine-weighted downward; else the collimated beam.
    direct0 = jax.random.uniform(k_kind, (batch_size,), dtype=dtype) >= diffuse_frac
    dif_dir = jax.vmap(lambda k: sampling.lambertian_reflect(k, dtype))(
        jax.random.split(k_dif, batch_size)) * jnp.asarray([1.0, 1.0, -1.0], dtype)
    dir0 = jnp.where(direct0[:, None], beam[None, :], dif_dir)
    t = photon_walk.trace_batch_weighted(
        k_walk, pos0, dir0, direct0, k_ext, ssa, g, maj, albedo, geom,
        int(config.max_iterations), float(config.weight_floor),
        rayleigh_frac=rayleigh_frac, mie_cdf=mie_cdf, mie_ang=mie_ang,
        r_eff=r_eff,
        free_path_floor_sharpness=float(config.free_path_floor_sharpness),
    )
    return jax.tree.map(jnp.add, acc, t), None

  bkeys = jax.random.split(key, nb)
  zc = jnp.zeros((ncells,), dtype)
  zs = jnp.zeros((ncols,), dtype)
  init = photon_walk.WeightedTally(zc, zc, zs, zs, zs, zs,
                                   jnp.zeros((), dtype),
                                   jnp.zeros((), jnp.int32))
  t, _ = jax.lax.scan(
      batch_step, init, (jnp.arange(nb, dtype=jnp.int32), bkeys))

  inv_p = 1.0 / p
  inv_n = 1.0 / n_total
  cells = (nx, ny, nz)
  cols = (nx, ny)
  sfc_net = t.sfc_dir + t.sfc_dif - t.sfc_up
  return MC3DResult(
      abs_frac=(t.abs_dir + t.abs_dif).reshape(cells) * inv_p,
      sfc_abs_frac=sfc_net.reshape(cols) * inv_p,
      tod_up_frac=jnp.sum(t.tod_up) * inv_n,
      vol_abs_total=jnp.sum(t.abs_dir + t.abs_dif) * inv_n,
      sfc_abs_total=jnp.sum(sfc_net) * inv_n,
      maxiter_frac=t.n_capped.astype(dtype) * inv_n,
      abs_dir_frac=t.abs_dir.reshape(cells) * inv_p,
      abs_dif_frac=t.abs_dif.reshape(cells) * inv_p,
      sfc_dir_frac=t.sfc_dir.reshape(cols) * inv_p,
      sfc_dif_frac=t.sfc_dif.reshape(cols) * inv_p,
      sfc_up_frac=t.sfc_up.reshape(cols) * inv_p,
      tod_up_col_frac=t.tod_up.reshape(cols) * inv_p,
      residual_frac=t.residual * inv_n,
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
    diffuse_frac: float,
    config: MC3DRadiationConfig,
    key: Array,
    rayleigh_frac: Array | None = None,
    mie_lut_cdf: Array | None = None,
    mie_lut_ang: Array | None = None,
    band_of_gpt: Array | None = None,
    r_eff: Array | None = None,
) -> SWFluxResult:
  """Spectrally-integrated shortwave MC transport over a stack of g-points.

  ``rayleigh_frac`` (optional ``(ngpt, nx, ny, nz)``): per-g-point gas Rayleigh
  fraction of total scattering, enabling the explicit Rayleigh + cloud phase
  split. Default ``None`` -> pure HG(g). When the small per-band Mie LUT
  ``mie_lut_cdf`` ``(n_band, n_mie)`` / ``mie_lut_ang`` ``(n_band, n_r, n_mie)``,
  the ``band_of_gpt`` ``(ngpt,)`` g-point->band map, and ``r_eff`` ``(nx,ny,nz)``
  [um] are given, the cloud phase is the microhh Mie LUT instead of HG. The
  band slice is gathered per g-point INSIDE the scan, so only the (n_band,...)
  LUT is resident -- not a per-g-point ``(ngpt, n_r, n_mie)`` copy.

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
    diffuse_frac: diffuse share of each g-point's incident flux (see
      ``solve_sw_monochromatic``).
    config: MC numerics.
    key: PRNG key (split per g-point).
  """
  dtype = tau.dtype
  nx, ny, nz = geom.nx, geom.ny, geom.nz
  dz = jnp.diff(geom.z_faces).astype(dtype)  # (nz,) [m]
  ngpt = tau.shape[0]
  gkeys = jax.random.split(key, ngpt)
  use_rayleigh = rayleigh_frac is not None
  # Cheap broadcastable dummy (leading ngpt axis) when a field is absent, so the
  # scan never carries a full (ngpt,nx,ny,nz) array it won't read.
  rfrac = rayleigh_frac if use_rayleigh else jnp.zeros((ngpt, 1, 1, 1), dtype)
  use_mie = (mie_lut_cdf is not None and mie_lut_ang is not None
             and band_of_gpt is not None and r_eff is not None)
  band_in = (band_of_gpt.astype(jnp.int32) if use_mie
             else jnp.zeros((ngpt,), jnp.int32))

  def gpt_step(carry, inp):
    abs_acc, sfc_acc, tod_acc, res_acc = carry
    tau_g, ssa_g, g_g, f_in, rf_g, band_idx, gkey = inp
    # Extinction [1/m] from layer optical depth; broadcast dz over (nx,ny,nz).
    k_ext = tau_g / dz[None, None, :]
    # Gather only THIS g-point's Mie band slice (n_mie / n_r,n_mie) from the
    # small per-band LUT -- no per-g-point LUT copy is ever materialized.
    cdf_g = mie_lut_cdf[band_idx] if use_mie else None
    ang_g = mie_lut_ang[band_idx] if use_mie else None
    res = solve_sw_monochromatic(
        k_ext, ssa_g, g_g, geom, mu0=mu0, azimuth=azimuth, albedo=albedo,
        diffuse_frac=diffuse_frac, config=config, key=gkey,
        rayleigh_frac=rf_g if use_rayleigh else None,
        mie_cdf=cdf_g, mie_ang=ang_g,
        r_eff=r_eff if use_mie else None)
    abs_acc = abs_acc + res.abs_frac * f_in
    sfc_acc = sfc_acc + res.sfc_abs_frac * f_in
    tod_acc = tod_acc + res.tod_up_frac * f_in
    res_acc = res_acc + res.residual_frac * f_in
    return (abs_acc, sfc_acc, tod_acc, res_acc), None

  init = (
      jnp.zeros((nx, ny, nz), dtype),
      jnp.zeros((nx, ny), dtype),
      jnp.zeros((), dtype),
      jnp.zeros((), dtype),
  )
  (abs_acc, sfc_acc, tod_acc, res_acc), _ = jax.lax.scan(
      gpt_step, init,
      (tau, ssa, g, incident_flux.astype(dtype), rfrac, band_in, gkeys)
  )
  return SWFluxResult(
      abs_flux=abs_acc, sfc_abs_flux=sfc_acc, tod_up_flux=tod_acc,
      residual_flux=res_acc)
