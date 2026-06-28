"""Phase-3 tests: longwave 3D Monte-Carlo thermal-emission radiative transfer.

Truth tiers: energy conservation (emitted = absorbed + escaped), an analytic
optically-thin isothermal cooling rate, OLR sign, determinism. Run with
``JAX_ENABLE_X64=1``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.physics.radiation.mc3d import (
    solve_lw_monochromatic,
    solve_lw_spectral,
)
from legoesm.atmosphere.physics.radiation.mc3d.config import MC3DRadiationConfig
from legoesm.atmosphere.physics.radiation.mc3d.photon_walk import PlaneRTGeometry

_SIG = constants.sigma_sb


def _geom(nx=6, ny=6, nz=10, dx=50.0, H=1000.0):
  z_faces = jnp.linspace(0.0, H, nz + 1)
  return PlaneRTGeometry(dx=dx, dy=dx, Lx=nx * dx, Ly=ny * dx,
                         z_faces=z_faces, nx=nx, ny=ny, nz=nz)


def _planck(T):
  return _SIG * T ** 4 / jnp.pi


def test_lw_energy_conservation():
  """Emitted = absorbed(vol+sfc) + escaped: net_atmos + net_sfc + OLR == 0."""
  geom = _geom()
  nx, ny, nz = geom.nx, geom.ny, geom.nz
  area = geom.dx * geom.dy
  T = jnp.linspace(300.0, 230.0, nz)[None, None, :] * jnp.ones((nx, ny, nz))
  k_abs = jnp.full((nx, ny, nz), 3e-3)
  res = solve_lw_monochromatic(
      k_abs, _planck(T), _planck(jnp.full((nx, ny), 300.0)), geom,
      emissivity=1.0, config=MC3DRadiationConfig(photons_per_pixel=6000),
      key=jax.random.PRNGKey(0))
  p_atmos = float(jnp.sum(res.net_flux) * area)
  p_sfc = float(jnp.sum(res.sfc_net) * area)
  p_olr = float(res.olr * geom.Lx * geom.Ly)
  total = p_atmos + p_sfc + p_olr
  scale = abs(p_atmos) + abs(p_sfc) + abs(p_olr) + 1e-30
  assert abs(total) / scale < 0.02     # residual is MC noise only
  assert float(res.maxiter_frac) < 1e-3


def test_lw_optically_thin_isothermal_cooling():
  """Optically-thin isothermal atmosphere over a cold (non-emitting) surface:
  each cell emits ~4 k sigma T^4 dz and almost nothing is reabsorbed, so the
  net flux per cell -> -4 k sigma T^4 dz (cooling)."""
  geom = _geom(nz=8, H=800.0)
  nx, ny, nz = geom.nx, geom.ny, geom.nz
  H = float(geom.z_faces[-1])
  T0 = 250.0
  tau = 0.03
  k = tau / H
  k_abs = jnp.full((nx, ny, nz), k)
  res = solve_lw_monochromatic(
      k_abs, _planck(jnp.full((nx, ny, nz), T0)),
      jnp.zeros((nx, ny)),                       # cold surface: no emission
      geom, emissivity=1.0,
      config=MC3DRadiationConfig(photons_per_pixel=8000),
      key=jax.random.PRNGKey(1))
  dz = H / nz
  expected = -4.0 * k * _SIG * T0 ** 4 * dz       # W/m^2 per cell
  got = float(jnp.mean(res.net_flux))
  assert got < 0.0                                # net cooling
  assert abs(got - expected) < 0.15 * abs(expected)   # thin + MC tolerance


def test_lw_olr_positive_and_finite():
  geom = _geom()
  nx, ny, nz = geom.nx, geom.ny, geom.nz
  T = jnp.full((nx, ny, nz), 270.0)
  res = solve_lw_monochromatic(
      jnp.full((nx, ny, nz), 5e-3), _planck(T),
      _planck(jnp.full((nx, ny), 290.0)), geom, emissivity=1.0,
      config=MC3DRadiationConfig(photons_per_pixel=3000),
      key=jax.random.PRNGKey(2))
  assert float(res.olr) > 0.0
  assert bool(jnp.all(jnp.isfinite(res.net_flux)))


def test_lw_maxiter_unresolved_budget():
  """Force maxiter>0 (max_iterations=1): OLR excludes unresolved photons, and
  the budget closes only when the unresolved term (maxiter*eps) is added."""
  geom = _geom(nx=5, ny=5, nz=8, H=1000.0)
  nx, ny, nz = geom.nx, geom.ny, geom.nz
  area = geom.dx * geom.dy
  T = jnp.linspace(290.0, 240.0, nz)[None, None, :] * jnp.ones((nx, ny, nz))
  # k_abs below the majorant floor => accept ratio < 1 => null collisions =>
  # photons need >1 step, so max_iterations=1 forces maxiter>0.
  k_abs = jnp.full((nx, ny, nz), 5e-4)
  Tsfc = jnp.full((nx, ny), 295.0)
  res = solve_lw_monochromatic(
      k_abs, _planck(T), _planck(Tsfc), geom, emissivity=1.0,
      config=MC3DRadiationConfig(photons_per_pixel=4000, max_iterations=1),
      key=jax.random.PRNGKey(7))
  assert float(res.maxiter_frac) > 0.0     # the cap actually fired

  # Reconstruct total emitted power E_tot to size the unresolved term.
  dz = jnp.diff(geom.z_faces)
  dV = area * dz[None, None, :]
  e_tot = float(jnp.sum(4.0 * jnp.pi * k_abs * _planck(T) * dV)
                + jnp.sum(jnp.pi * 1.0 * _planck(Tsfc) * area))
  p_three = (float(jnp.sum(res.net_flux) * area)
             + float(jnp.sum(res.sfc_net) * area)
             + float(res.olr * geom.Lx * geom.Ly))
  unresolved = float(res.maxiter_frac) * e_tot
  assert abs(p_three + unresolved) / e_tot < 0.02   # closes WITH unresolved


def test_lw_all_cold_no_emission():
  """Zero-emission (all-cold) domain: valid categorical fallback, no NaN, and
  every net flux is exactly zero (eps=0)."""
  geom = _geom(nx=4, ny=4, nz=6)
  nx, ny, nz = geom.nx, geom.ny, geom.nz
  res = solve_lw_monochromatic(
      jnp.full((nx, ny, nz), 3e-3), jnp.zeros((nx, ny, nz)),
      jnp.zeros((nx, ny)), geom, emissivity=1.0,
      config=MC3DRadiationConfig(photons_per_pixel=64),
      key=jax.random.PRNGKey(0))
  assert bool(jnp.all(jnp.isfinite(res.net_flux)))
  assert float(jnp.max(jnp.abs(res.net_flux))) == 0.0
  assert float(res.olr) == 0.0


def test_lw_spectral_conservation_and_olr():
  """solve_lw_spectral over multiple g-points conserves energy (sum of per-band
  emission MC) and produces positive OLR. Planck inputs are IRRADIANCE (pi*B)."""
  geom = _geom(nx=6, ny=6, nz=10)
  ngpt, nx, ny, nz = 3, geom.nx, geom.ny, geom.nz
  area = geom.dx * geom.dy
  # Per-band absorption optical depth + Planck irradiance (arbitrary positive).
  abs_od = jnp.stack([jnp.full((nx, ny, nz), 0.3 / nz),
                      jnp.full((nx, ny, nz), 0.8 / nz),
                      jnp.full((nx, ny, nz), 1.5 / nz)])
  Tprof = jnp.linspace(295.0, 235.0, nz)[None, None, :] * jnp.ones((nx, ny, nz))
  planck = jnp.stack([constants.sigma_sb * Tprof ** 4 * w
                      for w in (0.2, 0.3, 0.5)])      # band-fraction irradiance
  planck_sfc = jnp.stack([constants.sigma_sb * 298.0 ** 4 * w
                          * jnp.ones((nx, ny)) for w in (0.2, 0.3, 0.5)])
  res = solve_lw_spectral(abs_od, planck, planck_sfc, geom, emissivity=1.0,
                          config=MC3DRadiationConfig(photons_per_pixel=4000),
                          key=jax.random.PRNGKey(0))
  p_atmos = float(jnp.sum(res.net_flux) * area)
  p_sfc = float(jnp.sum(res.sfc_net) * area)
  p_olr = float(res.olr * geom.Lx * geom.Ly)
  scale = abs(p_atmos) + abs(p_sfc) + abs(p_olr) + 1e-30
  assert abs(p_atmos + p_sfc + p_olr) / scale < 0.03   # residual = MC + unresolved
  assert float(res.olr) > 0.0
  assert float(res.maxiter_frac) < 1e-2
  assert res.net_flux.shape == (nx, ny, nz)


def test_lw_rrtmgp_optical_field_extraction():
  """Phase 3b: RRTMGP solver returns per-g-point LW absorption optical depth +
  Planck sources via lw_optical_field_only, with physical ranges."""
  from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
  from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
  try:
    solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
  except Exception as e:
    pytest.skip(f"RRTMGP optics data unavailable: {e}")
  ncol, nlev = 2, 12
  T = jnp.linspace(295.0, 230.0, nlev)[None].repeat(ncol, 0)
  p_full = jnp.linspace(2_000.0, 1.0e5, nlev)[None].repeat(ncol, 0)
  p_half = jnp.linspace(1_000.0, 1.01e5, nlev + 1)[None].repeat(ncol, 0)
  q_v = jnp.full((ncol, nlev), 5e-3)
  abs_od, planck, planck_bot, planck_top, planck_sfc = solver.solve_columns(
      T=T, p_full=p_full, p_half=p_half, sfc_temperature=jnp.full((ncol,), 300.0),
      q_v=q_v, cos_zenith=jnp.full((ncol,), 0.5), lw_optical_field_only=True)
  ngpt = solver.optics_lib.n_gpt_lw
  assert abs_od.shape == (ngpt, ncol, nlev)
  assert planck.shape == (ngpt, ncol, nlev)
  assert planck_bot.shape == (ngpt, ncol, nlev)
  assert planck_top.shape == (ngpt, ncol, nlev)
  assert planck_sfc.shape == (ngpt, ncol)
  assert bool(jnp.all(abs_od >= 0.0))
  assert bool(jnp.all(planck >= 0.0))
  assert float(jnp.sum(planck)) > 0.0


def test_lw_olr_scale_vs_rrtmgp_two_stream():
  """SCALE/normalization check: mc3d spectral-LW outgoing LW (OLR) is the right
  PHYSICAL magnitude vs the RRTMGP two-stream OLR. The Planck source is radiance
  B, so a pi mis-normalization would show as a ~3x (or ~10x for pi^2) mismatch;
  this asserts the ratio is O(1). The residual (~30% here) is the exact-MC vs
  two-stream angular spread PLUS the cell-center Planck approximation (vs the
  two-stream linear-in-tau layer source), which is exaggerated at this coarse
  16-level dtau and shrinks on fine LES grids (Phase-3c: linear-in-tau emission).
  OLR is dz-independent (k_abs*dV = abs_od*dx*dy), so any uniform z grid works.
  """
  from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
  from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
  try:
    solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
  except Exception as e:
    pytest.skip(f"RRTMGP optics data unavailable: {e}")

  ncol, nlev = 4, 16
  T = jnp.linspace(248.0, 288.0, nlev)[None].repeat(ncol, 0)       # TOA->sfc
  p_full = jnp.linspace(5_000.0, 1.0e5, nlev)[None].repeat(ncol, 0)
  p_half = jnp.linspace(2_000.0, 1.013e5, nlev + 1)[None].repeat(ncol, 0)
  q_v = jnp.full((ncol, nlev), 5e-3)
  T_sfc = jnp.full((ncol,), 290.0)
  cosz = jnp.full((ncol,), 0.5)

  # RRTMGP two-stream OLR (upward LW at TOA, top-down index 0).
  full = solver.solve_columns(T=T, p_full=p_full, p_half=p_half,
                              sfc_temperature=T_sfc, q_v=q_v, cos_zenith=cosz)
  olr_ref = float(jnp.mean(full.lw_flux_up[:, 0]))

  # mc3d spectral-LW OLR via the production adapter (correct top-down<->tracer
  # orientation) with Phase-3c linear-in-tau emission. Arbitrary uniform z grid
  # (OLR is dz-independent). Plane: ny=1, nx=ncol.
  from legoesm.atmosphere.physics.radiation.mc3d import plane_adapter
  abs_od, planck, planck_bot, planck_top, planck_sfc = solver.solve_columns(
      T=T, p_full=p_full, p_half=p_half, sfc_temperature=T_sfc, q_v=q_v,
      cos_zenith=cosz, lw_optical_field_only=True)
  ngpt = abs_od.shape[0]

  class _Grid:
    nx, ny, dx, dy, Lx, Ly = ncol, 1, 1.0, 1.0, float(ncol), 1.0
  z_half_td = jnp.linspace(16_000.0, 0.0, nlev + 1)     # top-down
  rho = jnp.ones((1, ncol, nlev))
  _dT, _sfc, olr = plane_adapter.compute_plane_lw_heating_spectral(
      abs_od.reshape(ngpt, 1, ncol, nlev),
      planck.reshape(ngpt, 1, ncol, nlev),
      planck_sfc.reshape(ngpt, 1, ncol),
      _Grid(), z_half_td, rho, emissivity=1.0,
      config=MC3DRadiationConfig(photons_per_pixel=8000),
      key=jax.random.PRNGKey(0),
      planck_bottom_td=planck_bot.reshape(ngpt, 1, ncol, nlev),
      planck_top_td=planck_top.reshape(ngpt, 1, ncol, nlev))
  olr_mc = float(olr)
  assert olr_ref > 100.0          # sanity: realistic OLR magnitude
  # Ratio O(1) certifies the Planck radiance normalization (a pi error -> ~3x,
  # pi^2 -> ~10x). The band is wide enough to admit the exact-vs-two-stream +
  # cell-center-Planck spread but tight enough to catch any pi mis-scaling.
  assert 0.6 < olr_mc / olr_ref < 1.5


def test_lw_determinism():
  geom = _geom()
  nx, ny, nz = geom.nx, geom.ny, geom.nz
  T = jnp.full((nx, ny, nz), 260.0)
  args = dict(emissivity=1.0,
              config=MC3DRadiationConfig(photons_per_pixel=512),
              key=jax.random.PRNGKey(3))
  k_abs = jnp.full((nx, ny, nz), 4e-3)
  a = solve_lw_monochromatic(k_abs, _planck(T),
                             _planck(jnp.full((nx, ny), 285.0)), geom, **args)
  b = solve_lw_monochromatic(k_abs, _planck(T),
                             _planck(jnp.full((nx, ny), 285.0)), geom, **args)
  np.testing.assert_array_equal(np.asarray(a.net_flux), np.asarray(b.net_flux))
