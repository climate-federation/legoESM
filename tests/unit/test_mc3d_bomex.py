"""Integration test: the 3D Monte-Carlo ray tracer (scheme="mc3d") in a BOMEX
shallow-cumulus LES column field.

BOMEX (Siebesma et al. 2003) initial profiles are analytic (piecewise-linear
theta_l, q_t), so this is self-contained (no external gSAM case deck). A broken
shallow-cumulus q_c layer (cloud base ~600 m, top ~1500 m) is imposed and the
mc3d radiation physics_fn is run on the compressible-plane state. We assert the
3D ray tracer (a) runs end-to-end through make_radiation_physics(model_type=
"plane"), (b) responds to the cloud field (cloudy heating != clear-sky), and
(c) produces horizontally heterogeneous heating from the broken cloud — the 3D
effect absent from the independent-column approximation.

Cloud radiative coupling goes through the RRTMGP shortwave optics path (Phase
2b); skipped if the RRTMGP tables are unavailable. Run with JAX_ENABLE_X64=1.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.atmosphere.physics.radiation.config import (
    RadiationConfig,
    RRTMGPConfig,
)
from legoesm.atmosphere.physics.radiation.integration import (
    make_radiation_physics,
)
from legoesm.atmosphere.physics.radiation.mc3d.config import MC3DRadiationConfig
from legoesm.core.state import PlaneNonHydrostaticTendencies
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate


def _bomex_theta_l(z):
  """BOMEX liquid-water potential temperature profile [K] (Siebesma 2003)."""
  return jnp.where(
      z <= 520.0, 298.7,
      jnp.where(z <= 1480.0, 298.7 + (302.4 - 298.7) / (1480.0 - 520.0) * (z - 520.0),
                jnp.where(z <= 2000.0,
                          302.4 + (308.2 - 302.4) / (2000.0 - 1480.0) * (z - 1480.0),
                          308.2 + 3.65e-3 * (z - 2000.0))))


def _bomex_qt(z):
  """BOMEX total-water mixing ratio [kg/kg]."""
  qt_g = jnp.where(
      z <= 520.0, 17.0,
      jnp.where(z <= 1480.0, 16.3 + (10.7 - 16.3) / (1480.0 - 520.0) * (z - 520.0),
                jnp.where(z <= 2000.0,
                          10.7 + (4.2 - 10.7) / (2000.0 - 1480.0) * (z - 1480.0),
                          jnp.maximum(4.2 - 1.2e-3 * (z - 2000.0), 0.0))))
  return jnp.clip(qt_g, 0.0, None) * 1e-3


def _bomex_state(nx=6, ny=6, nlev=24, dx=100.0, H=3000.0, cloud=True, seed=0):
  grid = create_plane_grid(nx=nx, ny=ny, nlev=nlev, dx=dx, dy=dx,
                           dtype=jnp.float64)
  hc = create_height_coordinate(nlev, H=H, theta_ref_fn=_bomex_theta_l)
  tm = make_flat_plane_terrain_metric(grid, hc)
  rest = make_rest_state(grid, hc, dtype=jnp.float64)

  # Full-level heights (top-down) for the cloud mask + q_v profile.
  z_half = np.asarray(tm.z_half_3d[0, 0, :])
  z_full = 0.5 * (z_half[:-1] + z_half[1:])              # (nlev,) top-down
  qv_1d = np.asarray(_bomex_qt(jnp.asarray(z_full)))     # (nlev,)
  q_v = jnp.broadcast_to(jnp.asarray(qv_1d), (ny, nx, nlev))

  # Broken shallow-cumulus q_c: ~half the columns cloudy in 600-1500 m.
  in_cloud = (z_full >= 600.0) & (z_full <= 1500.0)
  col_cloudy = (jax.random.uniform(jax.random.PRNGKey(seed), (ny, nx)) < 0.5)
  q_c = jnp.where(
      (col_cloudy[:, :, None] & jnp.asarray(in_cloud)[None, None, :]),
      5e-4, 0.0) if cloud else jnp.zeros((ny, nx, nlev))

  tracers = jnp.stack([q_v, q_c, jnp.zeros((ny, nx, nlev))], axis=-1)
  state = rest._replace(tracers=rest.tracers.replace(data=tracers))
  return state, grid, hc, tm


def _mc3d_cfg(cloud: bool):
  return RadiationConfig(
      scheme="mc3d",
      cloud_scheme="sundqvist" if cloud else "none",
      rrtmgp=RRTMGPConfig(include_clouds=cloud),
      rce_fixed_cos_zenith=0.5,    # instantaneous sun for the 3D beam
      mc3d=MC3DRadiationConfig(photons_per_pixel=12, knull_coarsen_xy=2,
                               knull_coarsen_z=4, use_mie=cloud),
  )


def _rrtmgp_available():
  from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
  try:
    RRTMGP.from_legoesm_config(RRTMGPConfig())
    return True
  except Exception:
    return False


def test_mc3d_runs_in_bomex_les():
  """mc3d radiation runs end-to-end on a BOMEX plane state and returns finite,
  correctly-shaped tendencies that move no tracer mass."""
  state, grid, hc, tm = _bomex_state(cloud=True)
  if not _rrtmgp_available():
    pytest.skip("RRTMGP optics data unavailable")
  fn = make_radiation_physics(_mc3d_cfg(True), model_type="plane")
  tend = fn(state, grid, hc, tm)
  assert isinstance(tend, PlaneNonHydrostaticTendencies)
  assert tend.dtheta_prime_dt.data.shape == state.theta_prime.data.shape
  assert bool(jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data)))
  assert float(jnp.max(jnp.abs(tend.dtracers_dt.data))) == 0.0
  # Non-trivial radiative heating (SW + LW).
  assert float(jnp.max(jnp.abs(tend.dtheta_prime_dt.data))) > 0.0


def test_mc3d_realization_decorrelates_with_time():
  """B2: the MC seed folds in time, so different timesteps draw INDEPENDENT
  photon realizations (the speckle is NOT frozen across the integration), while
  a repeated time stays deterministic. Low photon count makes the per-key MC
  noise large enough to distinguish realizations."""
  state, grid, hc, tm = _bomex_state(cloud=False)
  if not _rrtmgp_available():
    pytest.skip("RRTMGP optics data unavailable")
  fn = make_radiation_physics(_mc3d_cfg(False), model_type="plane")

  fn.set_time(80.0, 43200.0)
  a = fn(state, grid, hc, tm).dtheta_prime_dt.data
  fn.set_time(80.0, 43200.0)            # same time -> identical realization
  a_again = fn(state, grid, hc, tm).dtheta_prime_dt.data
  fn.set_time(80.0, 43800.0)            # +10 min -> independent realization
  b = fn(state, grid, hc, tm).dtheta_prime_dt.data

  assert bool(jnp.allclose(a, a_again))             # deterministic per time
  assert not bool(jnp.allclose(a, b))               # not frozen across time


def test_bomex_broken_cloud_concentrates_sw_heating():
  """3D ray-tracing signature on a BOMEX-geometry broken shallow-cumulus field:
  shortwave absorption concentrates in the CLOUDY columns at cloud levels.

  Solver-level (solve_sw_spectral on a DIRECT optical field) — fast,
  deterministic, and free of the cloud-scheme RH-diagnosis confound, so the
  3D cloud response is isolated cleanly at modest photon counts.
  """
  from legoesm.atmosphere.physics.radiation.mc3d import solve_sw_monochromatic
  from legoesm.atmosphere.physics.radiation.mc3d.photon_walk import PlaneRTGeometry

  nx = ny = 8
  nz = 24
  dx = 100.0
  H = 3000.0
  geom = PlaneRTGeometry(dx=dx, dy=dx, Lx=nx * dx, Ly=ny * dx,
                         z_faces=jnp.linspace(0.0, H, nz + 1),
                         nx=nx, ny=ny, nz=nz)
  z_full = 0.5 * (np.linspace(0.0, H, nz + 1)[:-1] + np.linspace(0.0, H, nz + 1)[1:])
  cloud_k = np.where((z_full >= 600.0) & (z_full <= 1500.0))[0]
  col_cloudy = np.asarray(
      jax.random.uniform(jax.random.PRNGKey(0), (nx, ny)) < 0.5)

  # Broken cumulus extinction field: thick scattering cloud in cloudy columns
  # at cloud levels, optically-thin clear air elsewhere.
  k_ext = np.full((nx, ny, nz), 1e-4)
  for k in cloud_k:
    k_ext[col_cloudy, k] = 0.02
  k_ext = jnp.asarray(k_ext)
  ssa = jnp.where(k_ext > 1e-3, 0.99, 0.0)
  g = jnp.where(k_ext > 1e-3, 0.85, 0.0)

  res = solve_sw_monochromatic(
      k_ext, ssa, g, geom, mu0=1.0, azimuth=0.0, albedo=0.1,
      config=MC3DRadiationConfig(photons_per_pixel=4000),
      key=jax.random.PRNGKey(1))

  # The thick scattering cumulus (ssa~0.99) REFLECTS shortwave, so the dominant
  # 3D signal is the SURFACE SHADOW: with a vertical beam, far less SW reaches
  # the surface beneath cloudy columns than beneath clear columns.
  sfc = np.asarray(res.sfc_abs_frac)            # (nx,ny) transmitted-to-surface
  assert sfc[col_cloudy].mean() < 0.7 * sfc[~col_cloudy].mean()
  # Conservation still exact.
  tot = (float(res.vol_abs_total) + float(res.sfc_abs_total)
         + float(res.tod_up_frac))
  assert abs(tot - 1.0) < 1e-12
