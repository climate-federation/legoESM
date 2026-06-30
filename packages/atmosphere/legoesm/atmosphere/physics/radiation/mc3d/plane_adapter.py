"""Adapter wiring the 3D Monte-Carlo shortwave tracer to the plane LES/CRM dycore.

Handles the orientation/convention bridge between the atmospheric arrays and the
ray tracer (a fidelity-harness concern, kept OUT of the tracer per the
oracle-recipe doctrine):

* Atmospheric plane fields are ``(ny, nx, nlev)`` with vertical index 0 = top of
  domain (TOD) and index ``nlev-1`` = surface (``T[...,-1]`` is the surface).
  Physical height therefore DECREASES with index.
* The tracer wants ``(nx, ny, nz)`` with ``z`` INCREASING upward and
  ``z_faces[0]`` the surface. So this adapter transposes ``ny<->nx`` and flips
  the vertical axis on the way in, and undoes both on the way out.

Phase-2 optics provider is gray shortwave (closed-form, no RRTMGP plumbing);
ssa=g=0 ⇒ MC reduces to Beer-Lambert per column, so the live path reproduces the
gray ICA result for horizontally-uniform soundings (a correctness anchor). The
RRTMGP-spectral per-g-point optics provider is Phase 2b (docs/specs).
"""

from __future__ import annotations

from typing import TypeAlias

import jax
import jax.numpy as jnp
from legoesm import constants
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
from legoesm.atmosphere.physics.radiation.mc3d import raytracer_lw
from legoesm.atmosphere.physics.radiation.mc3d import raytracer_sw
from legoesm.atmosphere.physics.radiation.mc3d.config import MC3DRadiationConfig
from legoesm.atmosphere.physics.radiation.mc3d.photon_walk import PlaneRTGeometry

Array: TypeAlias = jax.Array


def geometry_from_plane(grid, z_faces_ascending: Array) -> PlaneRTGeometry:
  """Build a tracer geometry from a plane grid + ascending z interfaces.

  Args:
    grid: plane grid exposing ``nx, ny, dx, dy, Lx, Ly``.
    z_faces_ascending: ``(nz+1,)`` interface heights [m], strictly increasing,
      ``[0]`` = surface, ``[-1]`` = TOD.
  """
  nz = z_faces_ascending.shape[0] - 1
  return PlaneRTGeometry(
      dx=float(grid.dx), dy=float(grid.dy),
      Lx=float(grid.Lx), Ly=float(grid.Ly),
      z_faces=z_faces_ascending, nx=int(grid.nx), ny=int(grid.ny), nz=nz,
  )


def gray_sw_optical_field(
    p_half_col: Array, gray_cfg: GrayRadiationConfig, ny: int, nx: int
) -> tuple[Array, Array, Array]:
  """Gray-shortwave per-layer optical field for the MC tracer (top-down order).

  Mirrors ``gray._sw_beer_lambert``: cumulative ``tau_sw(sigma) =
  sw_tau_0 * sigma^sw_exponent`` from TOD, differenced into per-layer optical
  depth. Pure absorption ⇒ ssa = g = 0.

  Returns ``(tau, ssa, g)`` each shaped ``(1, ny, nx, nlev)`` (single band).
  """
  p_s = p_half_col[:, -1:]                              # surface pressure (ncol,1)
  sigma_half = p_half_col / p_s                         # (ncol, nlev+1)
  tau_cum = gray_cfg.sw_tau_0 * sigma_half ** gray_cfg.sw_exponent
  dtau = jnp.maximum(jnp.diff(tau_cum, axis=1), 0.0)    # (ncol, nlev) top-down
  nlev = dtau.shape[1]
  tau = dtau.reshape(1, ny, nx, nlev)
  return tau, jnp.zeros_like(tau), jnp.zeros_like(tau)


def _to_tracer(field_td: Array) -> Array:
  """(..., ny, nx, nz) top-down  ->  (..., nx, ny, nz) bottom-up."""
  swapped = jnp.swapaxes(field_td, -3, -2)
  return swapped[..., ::-1]


def _from_tracer(field_tr: Array) -> Array:
  """(nx, ny, nz) bottom-up  ->  (ny, nx, nz) top-down (inverse of _to_tracer)."""
  return jnp.swapaxes(field_tr[..., ::-1], 0, 1)


def compute_plane_sw_heating(
    tau_td: Array,
    ssa_td: Array,
    g_td: Array,
    incident_flux: Array,
    grid,
    z_half_td: Array,
    rho_td: Array,
    *,
    mu0,
    albedo,
    config: MC3DRadiationConfig,
    key: Array,
    azimuth: float = 0.0,
    cp: float = constants.c_pd,
    rayleigh_frac_td: Array | None = None,
    mie_lut_cdf: Array | None = None,
    mie_lut_ang: Array | None = None,
    band_of_gpt: Array | None = None,
    r_eff_td: Array | None = None,
) -> tuple[Array, Array, Array]:
  """3D-MC shortwave heating-rate tendency for the plane dycore.

  Args:
    tau_td, ssa_td, g_td: optical fields ``(ngpt, ny, nx, nlev)`` top-down.
    incident_flux: per-g-point vertical TOA flux ``(ngpt,)`` [W/m^2].
    grid: plane grid (nx, ny, dx, dy, Lx, Ly).
    z_half_td: interface heights ``(nlev+1,)`` top-down (index 0 = TOD).
    rho_td: total density ``(ny, nx, nlev)`` top-down [kg/m^3].
    mu0: cosine solar zenith angle. albedo: surface albedo. (may be traced scalars)
    rayleigh_frac_td: optional per-g-point gas Rayleigh fraction of total
      scattering ``(ngpt, ny, nx, nlev)`` -> explicit Rayleigh + HG-cloud phase.

  Returns:
    ``(dT_dt_sw (ny,nx,nlev) [K/s] top-down, sfc_abs_flux (ny,nx) [W/m^2],
    tod_up_flux scalar [W/m^2])``.
  """
  z_faces_asc = z_half_td[::-1]                         # ascending, surface first
  geom = geometry_from_plane(grid, z_faces_asc)
  tau_tr = _to_tracer(tau_td)
  ssa_tr = _to_tracer(ssa_td)
  g_tr = _to_tracer(g_td)
  rf_tr = _to_tracer(rayleigh_frac_td) if rayleigh_frac_td is not None else None
  reff_tr = _to_tracer(r_eff_td) if r_eff_td is not None else None

  res = raytracer_sw.solve_sw_spectral(
      tau_tr, ssa_tr, g_tr, incident_flux, geom,
      mu0=mu0, azimuth=azimuth, albedo=albedo, config=config, key=key,
      rayleigh_frac=rf_tr, mie_lut_cdf=mie_lut_cdf,
      mie_lut_ang=mie_lut_ang, band_of_gpt=band_of_gpt, r_eff=reff_tr)

  dz_asc = jnp.diff(z_faces_asc)                        # (nz,) [m]
  rho_tr = _to_tracer(rho_td)                           # (nx,ny,nz) bottom-up
  # dT/dt = F_abs / (rho * cp * dz)  [ (W/m^2)/(kg/m^3 * J/kg/K * m) = K/s ]
  denom = jnp.clip(rho_tr * cp * dz_asc[None, None, :], 1e-12, None)
  heating_tr = res.abs_flux / denom
  dT_dt_sw = _from_tracer(heating_tr)
  sfc_abs_flux = jnp.swapaxes(res.sfc_abs_flux, 0, 1)   # (nx,ny)->(ny,nx)
  return dT_dt_sw, sfc_abs_flux, res.tod_up_flux


def compute_plane_lw_heating(
    k_abs_td: Array,
    planck_td: Array,
    planck_sfc_td: Array,
    grid,
    z_half_td: Array,
    rho_td: Array,
    *,
    emissivity,
    config: MC3DRadiationConfig,
    key: Array,
    cp: float = constants.c_pd,
) -> tuple[Array, Array, Array]:
  """3D-MC longwave heating-rate tendency for the plane dycore.

  Args mirror ``compute_plane_sw_heating`` but for a single (gray/g-point) LW
  band: ``k_abs_td`` absorption coefficient ``(ny,nx,nlev)`` top-down [1/m],
  ``planck_td`` cell Planck radiance ``(ny,nx,nlev)``, ``planck_sfc_td`` surface
  Planck radiance ``(ny,nx)``.

  Returns ``(dT_dt_lw (ny,nx,nlev) [K/s] top-down, sfc_net (ny,nx) [W/m^2],
  olr scalar [W/m^2])``. dT/dt < 0 where a layer cools to space.
  """
  z_faces_asc = z_half_td[::-1]
  geom = geometry_from_plane(grid, z_faces_asc)
  k_abs_tr = _to_tracer(k_abs_td)
  planck_tr = _to_tracer(planck_td)
  planck_sfc_tr = jnp.swapaxes(planck_sfc_td, 0, 1)

  res = raytracer_lw.solve_lw_monochromatic(
      k_abs_tr, planck_tr, planck_sfc_tr, geom,
      emissivity=emissivity, config=config, key=key)

  dz_asc = jnp.diff(z_faces_asc)
  rho_tr = _to_tracer(rho_td)
  denom = jnp.clip(rho_tr * cp * dz_asc[None, None, :], 1e-12, None)
  heating_tr = res.net_flux / denom
  dT_dt_lw = _from_tracer(heating_tr)
  sfc_net = jnp.swapaxes(res.sfc_net, 0, 1)
  return dT_dt_lw, sfc_net, res.olr


def compute_plane_lw_heating_spectral(
    abs_od_td: Array,
    planck_td: Array,
    planck_sfc_td: Array,
    grid,
    z_half_td: Array,
    rho_td: Array,
    *,
    emissivity,
    config: MC3DRadiationConfig,
    key: Array,
    planck_bottom_td: Array | None = None,
    planck_top_td: Array | None = None,
    cp: float = constants.c_pd,
) -> tuple[Array, Array, Array]:
  """RRTMGP-spectral 3D-MC longwave heating for the plane dycore (Phase 3b/3c).

  Args (all top-down, per-g-point): ``abs_od_td`` absorption optical depth
  ``(ngpt, ny, nx, nlev)``, ``planck_td`` cell-center Planck radiance, optional
  ``planck_bottom_td`` / ``planck_top_td`` the lower-z / upper-z cell-FACE Planck
  radiances (Phase-3c linear-in-tau emission), ``planck_sfc_td`` surface Planck.
  Returns ``(dT_dt_lw (ny,nx,nlev) [K/s], sfc_net (ny,nx) [W/m^2], olr scalar)``.

  Orientation: ``_to_tracer`` flips top-down->bottom-up AND preserves each cell's
  lower-z/upper-z face labeling (the array reverse maps cell k's lower-z face to
  tracer cell i's lower-z face), so RRTMGP ``planck_src_bottom`` (lower-z face)
  threads straight to the tracer ``planck_lower``.
  """
  z_faces_asc = z_half_td[::-1]
  geom = geometry_from_plane(grid, z_faces_asc)
  abs_od_tr = _to_tracer(abs_od_td)
  planck_tr = _to_tracer(planck_td)
  planck_sfc_tr = jnp.swapaxes(planck_sfc_td, -2, -1)   # (ngpt,ny,nx)->(ngpt,nx,ny)
  lower_tr = _to_tracer(planck_bottom_td) if planck_bottom_td is not None else None
  upper_tr = _to_tracer(planck_top_td) if planck_top_td is not None else None

  res = raytracer_lw.solve_lw_spectral(
      abs_od_tr, planck_tr, planck_sfc_tr, geom,
      emissivity=emissivity, config=config, key=key,
      planck_src_lower=lower_tr, planck_src_upper=upper_tr)

  dz_asc = jnp.diff(z_faces_asc)
  rho_tr = _to_tracer(rho_td)
  denom = jnp.clip(rho_tr * cp * dz_asc[None, None, :], 1e-12, None)
  heating_tr = res.net_flux / denom
  dT_dt_lw = _from_tracer(heating_tr)
  sfc_net = jnp.swapaxes(res.sfc_net, 0, 1)
  return dT_dt_lw, sfc_net, res.olr
