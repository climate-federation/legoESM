"""Phase-2 tests: 3D Monte-Carlo shortwave spectral driver + plane LES/CRM wiring.

Covers the g-point spectral accumulation, the plane orientation/heating adapter,
a two-stream cross-check, horizontal equivariance, and end-to-end dispatch
through ``make_radiation_physics(scheme="mc3d", model_type="plane")``.
Run with ``JAX_ENABLE_X64=1``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.radiation.mc3d import plane_adapter
from legoesm.atmosphere.physics.radiation.mc3d.config import MC3DRadiationConfig
from legoesm.atmosphere.physics.radiation.mc3d.photon_walk import PlaneRTGeometry
from legoesm.atmosphere.physics.radiation.mc3d.raytracer_sw import (
    solve_sw_spectral,
)


def _geom(nx=8, ny=8, nz=12, dx=50.0, H=1200.0):
  z_faces = jnp.linspace(0.0, H, nz + 1)
  return PlaneRTGeometry(dx=dx, dy=dx, Lx=nx * dx, Ly=ny * dx,
                         z_faces=z_faces, nx=nx, ny=ny, nz=nz)


# --------------------------------------------------------------------------- #
# Spectral g-point driver                                                     #
# --------------------------------------------------------------------------- #

def test_spectral_energy_budget():
  """Two absorbing bands, black surface, vertical beam: domain budget
  mean_ij(sum_z abs) + mean_ij(sfc_abs) + tod_up == sum(incident)."""
  geom = _geom()
  ngpt, nx, ny, nz = 2, geom.nx, geom.ny, geom.nz
  H = float(geom.z_faces[-1])
  # tau per layer so column tau ~ given totals.
  tau = jnp.stack([
      jnp.full((nx, ny, nz), 0.6 / nz),
      jnp.full((nx, ny, nz), 1.5 / nz),
  ])
  ssa = jnp.zeros_like(tau)
  g = jnp.zeros_like(tau)
  incident = jnp.array([300.0, 200.0])
  cfg = MC3DRadiationConfig(photons_per_pixel=3000)
  res = solve_sw_spectral(tau, ssa, g, incident, geom, mu0=1.0, azimuth=0.0,
                          albedo=0.0, config=cfg, key=jax.random.PRNGKey(0))
  col_abs = jnp.sum(res.abs_flux, axis=2)               # (nx,ny) per-column abs
  budget = (float(jnp.mean(col_abs)) + float(jnp.mean(res.sfc_abs_flux))
            + float(res.tod_up_flux))
  assert abs(budget - 500.0) < 500.0 * 0.02            # 2% MC tolerance


def test_spectral_beer_lambert_surface_flux():
  """Single absorbing band: surface-absorbed flux = incident*exp(-tau)."""
  geom = _geom()
  nx, ny, nz = geom.nx, geom.ny, geom.nz
  tau_tot = 0.7
  tau = jnp.full((1, nx, ny, nz), tau_tot / nz)
  ssa = jnp.zeros_like(tau)
  g = jnp.zeros_like(tau)
  incident = jnp.array([400.0])
  cfg = MC3DRadiationConfig(photons_per_pixel=4000)
  res = solve_sw_spectral(tau, ssa, g, incident, geom, mu0=1.0, azimuth=0.0,
                          albedo=0.0, config=cfg, key=jax.random.PRNGKey(1))
  expected = 400.0 * np.exp(-tau_tot)
  assert abs(float(jnp.mean(res.sfc_abs_flux)) - expected) < 0.02 * 400.0


def test_two_stream_cross_check_single_slab():
  """MC R/T/A for a homogeneous scattering slab vs the validated two-stream
  kernel (g=0, where two-stream is most accurate). Loose tol: two-stream is an
  approximation, MC is exact."""
  from legoesm.atmosphere.physics.radiation.rrtmgp.rte.monochromatic_two_stream import (
      sw_cell_properties,
  )
  geom = _geom(nx=6, ny=6, nz=1, H=1000.0)
  tau_v, ssa_v = 0.5, 0.7
  tau = jnp.full((1, geom.nx, geom.ny, 1), tau_v)
  ssa = jnp.full((1, geom.nx, geom.ny, 1), ssa_v)
  g = jnp.zeros_like(tau)
  cfg = MC3DRadiationConfig(photons_per_pixel=8000)
  res = solve_sw_spectral(tau, ssa, g, jnp.array([1.0]), geom, mu0=1.0,
                          azimuth=0.0, albedo=0.0, config=cfg,
                          key=jax.random.PRNGKey(2))
  R_mc = float(res.tod_up_flux)
  T_mc = float(jnp.mean(res.sfc_abs_flux))   # black sfc -> all transmitted absorbed
  A_mc = float(jnp.mean(jnp.sum(res.abs_flux, axis=2)))

  props = sw_cell_properties(0.0, jnp.array(tau_v), jnp.array(ssa_v),
                             jnp.array(0.0))
  t_noscat = float(np.exp(-tau_v))
  R_ts = float(props["r_dir"])
  T_ts = t_noscat + float(props["t_dir"])
  A_ts = 1.0 - R_ts - T_ts
  assert abs(R_mc - R_ts) < 0.05
  assert abs(T_mc - T_ts) < 0.05
  assert abs(A_mc - A_ts) < 0.05


def test_horizontal_equivariance_roll():
  """Rolling the optical field horizontally rolls the heating field: the
  horizontal-mean vertical absorption profile is invariant under the roll."""
  geom = _geom(nx=8, ny=8, nz=8)
  nx, ny, nz = geom.nx, geom.ny, geom.nz
  base = jnp.full((1, nx, ny, nz), 0.5 / nz)
  # cloud-like absorbing blob in one column region
  field = base.at[0, 2:4, 2:4, nz // 2:].set(3.0 / nz)
  ssa = jnp.zeros_like(field)
  g = jnp.zeros_like(field)
  cfg = MC3DRadiationConfig(photons_per_pixel=3000)
  kw = dict(mu0=1.0, azimuth=0.0, albedo=0.0, config=cfg)
  r0 = solve_sw_spectral(field, ssa, g, jnp.array([1.0]), geom,
                         key=jax.random.PRNGKey(3), **kw)
  rolled = jnp.roll(field, shift=3, axis=1)
  r1 = solve_sw_spectral(rolled, ssa, g, jnp.array([1.0]), geom,
                         key=jax.random.PRNGKey(4), **kw)
  prof0 = np.asarray(jnp.mean(r0.abs_flux, axis=(0, 1)))
  prof1 = np.asarray(jnp.mean(r1.abs_flux, axis=(0, 1)))
  assert np.max(np.abs(prof0 - prof1)) < 0.01    # horizontal-mean invariant


# --------------------------------------------------------------------------- #
# Plane adapter                                                               #
# --------------------------------------------------------------------------- #

def test_adapter_orientation_roundtrip():
  """_to_tracer / _from_tracer invert: (ny,nx,nz) top-down round-trips."""
  ny, nx, nz = 3, 4, 5
  f = jnp.arange(ny * nx * nz, dtype=jnp.float64).reshape(ny, nx, nz)
  rt = plane_adapter._from_tracer(plane_adapter._to_tracer(f))
  np.testing.assert_array_equal(np.asarray(f), np.asarray(rt))


def test_compute_plane_sw_heating_warms_and_shapes():
  """SW absorption warms (dT/dt > 0) and returns plane-shaped (ny,nx,nlev)."""
  class _Grid:
    nx, ny, dx, dy, Lx, Ly = 4, 4, 50.0, 50.0, 200.0, 200.0
  ny, nx, nz = 4, 4, 8
  # top-down z interfaces (index 0 = TOD, decreasing height)
  z_half_td = jnp.linspace(1000.0, 0.0, nz + 1)
  tau = jnp.full((1, ny, nx, nz), 0.5 / nz)
  ssa = jnp.zeros_like(tau)
  g = jnp.zeros_like(tau)
  rho = jnp.full((ny, nx, nz), 1.0)
  cfg = MC3DRadiationConfig(photons_per_pixel=2000)
  dT, sfc, tod = plane_adapter.compute_plane_sw_heating(
      tau, ssa, g, jnp.array([400.0]), _Grid(), z_half_td, rho,
      mu0=1.0, albedo=0.0, config=cfg, key=jax.random.PRNGKey(5))
  assert dT.shape == (ny, nx, nz)
  assert sfc.shape == (ny, nx)
  assert float(jnp.min(dT)) >= 0.0           # SW absorption only warms
  assert float(jnp.max(dT)) > 0.0


# --------------------------------------------------------------------------- #
# End-to-end dispatch                                                         #
# --------------------------------------------------------------------------- #

def _plane_setup():
  from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
      make_flat_plane_terrain_metric, make_rest_state,
  )
  from legoesm.grids.plane import create_plane_grid
  from legoesm.grids.vertical import create_height_coordinate
  nx = ny = 4
  nlev = 10
  grid = create_plane_grid(nx=nx, ny=ny, nlev=nlev, dx=4_000.0, dy=4_000.0,
                           dtype=jnp.float64)
  hc = create_height_coordinate(nlev, H=20_000.0)
  tm = make_flat_plane_terrain_metric(grid, hc)
  rest = make_rest_state(grid, hc, dtype=jnp.float64)
  tracers = jnp.zeros((ny, nx, nlev, 3), dtype=jnp.float64).at[..., -1, 0].set(0.01)
  state = rest._replace(tracers=rest.tracers.replace(data=tracers))
  return grid, hc, tm, state, (ny, nx, nlev)


def test_mc3d_plane_dispatch_shapes_and_heating():
  from legoesm.atmosphere.physics.radiation.config import (
      GrayRadiationConfig, RadiationConfig,
  )
  from legoesm.atmosphere.physics.radiation.integration import (
      make_radiation_physics,
  )
  from legoesm.core.state import PlaneNonHydrostaticTendencies
  grid, hc, tm, state, shape_3d = _plane_setup()
  cfg = RadiationConfig(
      scheme="mc3d", gray=GrayRadiationConfig(),
      mc3d=MC3DRadiationConfig(photons_per_pixel=16),
      rce_fixed_cos_zenith=0.62,   # 3D tracer needs an instantaneous sun angle
  )
  physics_fn = make_radiation_physics(cfg, model_type="plane")
  tend = physics_fn(state, grid, hc, tm)
  assert isinstance(tend, PlaneNonHydrostaticTendencies)
  assert tend.dtheta_prime_dt.data.shape == shape_3d
  assert tend.dtheta_prime_dt.dims == ("y", "x", "z")
  # Non-trivial heating tendency (SW + gray LW).
  assert float(jnp.max(jnp.abs(tend.dtheta_prime_dt.data))) > 0.0
  assert bool(jnp.all(jnp.isfinite(tend.dtheta_prime_dt.data)))
  # Radiation moves no tracer mass.
  assert float(jnp.max(jnp.abs(tend.dtracers_dt.data))) == 0.0


def test_rrtmgp_sw_optical_field_extraction():
  """Phase 2b: RRTMGP solver returns a per-g-point 3D SW optical field
  (tau/ssa/g) + solar weights via sw_optical_field_only, with physical ranges.
  Optics-only (no MC), so this is the fast Phase-2b correctness gate."""
  from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
  from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
  try:
    solver = RRTMGP.from_legoesm_config(RRTMGPConfig())
  except Exception as e:  # RRTMGP data tables unavailable in this env
    pytest.skip(f"RRTMGP optics data unavailable: {e}")
  ncol, nlev = 2, 12
  T = jnp.full((ncol, nlev), 250.0)
  p_full = jnp.linspace(2_000.0, 1.0e5, nlev)[None].repeat(ncol, 0)
  p_half = jnp.linspace(1_000.0, 1.01e5, nlev + 1)[None].repeat(ncol, 0)
  q_v = jnp.full((ncol, nlev), 1e-3)
  cos_z = jnp.full((ncol,), 0.6)
  tau, ssa, g, rayleigh_frac, r_eff_um, solar = solver.solve_columns(
      T=T, p_full=p_full, p_half=p_half, sfc_temperature=jnp.full((ncol,), 288.0),
      q_v=q_v, cos_zenith=cos_z, sw_optical_field_only=True)
  ngpt = solver.optics_lib.n_gpt_sw
  assert tau.shape == (ngpt, ncol, nlev)
  assert ssa.shape == (ngpt, ncol, nlev)
  assert g.shape == (ngpt, ncol, nlev)
  assert rayleigh_frac.shape == (ngpt, ncol, nlev)
  assert r_eff_um.shape == (ncol, nlev)
  assert solar.shape == (ngpt,)
  assert bool(jnp.all(tau >= 0.0))
  assert bool(jnp.all((ssa >= 0.0) & (ssa <= 1.0)))
  assert bool(jnp.all((g >= -1.0) & (g <= 1.0)))
  assert bool(jnp.all((rayleigh_frac >= 0.0) & (rayleigh_frac <= 1.0)))
  assert float(jnp.sum(solar)) > 0.0   # nonzero TOA solar source


def test_mc3d_requires_instantaneous_zenith():
  """Daily-mean/equinox forcing (cos_sza=None) has no beam direction -> the 3D
  tracer must refuse rather than silently trace an overhead beam."""
  from legoesm.atmosphere.physics.radiation.config import RadiationConfig
  from legoesm.atmosphere.physics.radiation.integration import (
      make_radiation_physics,
  )
  grid, hc, tm, state, _ = _plane_setup()
  cfg = RadiationConfig(scheme="mc3d", diurnal_cycle=False,
                        rce_fixed_cos_zenith=None,
                        mc3d=MC3DRadiationConfig(photons_per_pixel=200))
  physics_fn = make_radiation_physics(cfg, model_type="plane")
  with pytest.raises(ValueError, match="instantaneous solar zenith"):
    physics_fn(state, grid, hc, tm)


def test_mc3d_cloud_gate_rejects_silent_cloud_drop():
  """mc3d shares the RRTMGP optics path: an active cloud_scheme with
  include_clouds=False would silently drop cloud optics -> must raise."""
  from legoesm.atmosphere.physics.radiation.config import (
      RadiationConfig, RRTMGPConfig,
  )
  from legoesm.atmosphere.physics.radiation.integration import (
      make_radiation_physics,
  )
  cfg = RadiationConfig(scheme="mc3d", cloud_scheme="sundqvist",
                        rrtmgp=RRTMGPConfig(include_clouds=False))
  with pytest.raises(ValueError, match="Inconsistent cloud-radiation gate"):
    make_radiation_physics(cfg, model_type="plane")
  # Clear-sky mc3d (cloud_scheme="none") must NOT trip the gate even with
  # include_clouds=False (guards against future over-tightening).
  from legoesm.atmosphere.physics.radiation.integration import (
      _validate_cloud_gate,
  )
  _validate_cloud_gate(RadiationConfig(
      scheme="mc3d", cloud_scheme="none",
      rrtmgp=RRTMGPConfig(include_clouds=False)))


def test_mc3d_rejects_non_plane_model_type():
  from legoesm.atmosphere.physics.radiation.config import RadiationConfig
  from legoesm.atmosphere.physics.radiation.integration import (
      make_radiation_physics,
  )
  cfg = RadiationConfig(scheme="mc3d")
  with pytest.raises(ValueError, match="only wired for"):
    make_radiation_physics(cfg, model_type="hydrostatic")
