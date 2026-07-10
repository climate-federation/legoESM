"""Phase-1 tests for the 3D Monte-Carlo ray tracer (``radiation/mc3d``).

Truth tiers first: analytic slab (Beer-Lambert), exact photon-wise energy
conservation, MC 1/sqrt(N) convergence, horizontal-grid-bias check, determinism.
Run with ``JAX_ENABLE_X64=1``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.mc3d import (
    MC3DRadiationConfig,
    PlaneRTGeometry,
    solve_sw_monochromatic,
    tally_batch,
    trace_batch,
)


def _geom(nx=8, ny=8, nz=20, dx=50.0, H=1000.0):
  z_faces = jnp.linspace(0.0, H, nz + 1)
  return PlaneRTGeometry(dx=dx, dy=dx, Lx=nx * dx, Ly=ny * dx,
                         z_faces=z_faces, nx=nx, ny=ny, nz=nz)


def _uniform_fields(geom, k, ssa, g):
  shape = (geom.nx, geom.ny, geom.nz)
  return (jnp.full(shape, k), jnp.full(shape, ssa), jnp.full(shape, g))


def test_beer_lambert_slab_transmittance():
  """Pure-absorbing uniform slab, vertical beam, black surface:
  surface-absorbed fraction = exp(-tau) within MC error."""
  geom = _geom()
  H = float(geom.z_faces[-1])
  tau = 1.0
  k_ext, ssa, g = _uniform_fields(geom, tau / H, 0.0, 0.0)
  cfg = MC3DRadiationConfig(photons_per_pixel=4000)
  res = solve_sw_monochromatic(
      k_ext, ssa, g, geom, mu0=1.0, azimuth=0.0, albedo=0.0,
      config=cfg, key=jax.random.PRNGKey(0))

  exact = np.exp(-tau)
  n = geom.nx * geom.ny * cfg.photons_per_pixel
  stderr = np.sqrt(exact * (1 - exact) / n)
  assert abs(float(res.sfc_abs_total) - exact) < 5 * stderr
  assert float(res.tod_up_frac) == 0.0          # black surface, no upwelling
  assert float(res.maxiter_frac) == 0.0


def test_slant_beam_beer_lambert():
  """Slant beam path length tau/mu0: transmittance = exp(-tau/mu0)."""
  geom = _geom()
  H = float(geom.z_faces[-1])
  tau, mu0 = 0.8, 0.6
  k_ext, ssa, g = _uniform_fields(geom, tau / H, 0.0, 0.0)
  cfg = MC3DRadiationConfig(photons_per_pixel=4000)
  res = solve_sw_monochromatic(
      k_ext, ssa, g, geom, mu0=mu0, azimuth=0.7, albedo=0.0,
      config=cfg, key=jax.random.PRNGKey(1))
  exact = np.exp(-tau / mu0)
  n = geom.nx * geom.ny * cfg.photons_per_pixel
  stderr = np.sqrt(exact * (1 - exact) / n)
  assert abs(float(res.sfc_abs_total) - exact) < 5 * stderr


def test_energy_conservation_scattering():
  """Analog MC: every photon ends in exactly one bin -> the four fractions sum
  to 1 to floating-point, for a scattering + reflecting field."""
  geom = _geom()
  k_ext, ssa, g = _uniform_fields(geom, 3e-3, 0.9, 0.5)
  cfg = MC3DRadiationConfig(photons_per_pixel=2000)
  res = solve_sw_monochromatic(
      k_ext, ssa, g, geom, mu0=0.7, azimuth=0.3, albedo=0.2,
      config=cfg, key=jax.random.PRNGKey(2))
  total = (float(res.vol_abs_total) + float(res.sfc_abs_total)
           + float(res.tod_up_frac))
  assert abs(total - 1.0) < 1e-12         # integer counts -> exact budget
  assert float(res.maxiter_frac) < 1e-3   # tracking cap effectively never hit


def test_albedo_one_no_surface_absorption():
  """Perfect reflector: no surface absorption; photons end in volume or TOD."""
  geom = _geom()
  k_ext, ssa, g = _uniform_fields(geom, 2e-3, 0.8, 0.3)
  cfg = MC3DRadiationConfig(photons_per_pixel=2000)
  res = solve_sw_monochromatic(
      k_ext, ssa, g, geom, mu0=1.0, azimuth=0.0, albedo=1.0,
      config=cfg, key=jax.random.PRNGKey(3))
  assert float(res.sfc_abs_total) == 0.0
  total = float(res.vol_abs_total) + float(res.tod_up_frac)
  assert abs(total - 1.0) < 1e-12


def test_mc_convergence_rate():
  """Error in slab transmittance shrinks ~ 1/sqrt(N): 4x photons -> ~2x tighter.
  Checked statistically by requiring both estimates within their own stderr."""
  geom = _geom()
  H = float(geom.z_faces[-1])
  tau = 1.0
  k_ext, ssa, g = _uniform_fields(geom, tau / H, 0.0, 0.0)
  exact = np.exp(-tau)
  errs = []
  for p in (500, 2000):
    cfg = MC3DRadiationConfig(photons_per_pixel=p)
    res = solve_sw_monochromatic(
        k_ext, ssa, g, geom, mu0=1.0, azimuth=0.0, albedo=0.0,
        config=cfg, key=jax.random.PRNGKey(7))
    n = geom.nx * geom.ny * p
    errs.append(abs(float(res.sfc_abs_total) - exact))
    stderr = np.sqrt(exact * (1 - exact) / n)
    assert errs[-1] < 5 * stderr
  # Larger N should not be dramatically worse than the 4x-tighter expectation.
  assert errs[1] < errs[0] * 1.5


def test_no_horizontal_grid_bias():
  """Horizontally uniform field -> column surface-fraction is uniform to within
  MC noise (no spurious x/y bias from indexing/wrap)."""
  geom = _geom()
  k_ext, ssa, g = _uniform_fields(geom, 1e-3, 0.0, 0.0)
  cfg = MC3DRadiationConfig(photons_per_pixel=3000)
  res = solve_sw_monochromatic(
      k_ext, ssa, g, geom, mu0=1.0, azimuth=0.0, albedo=0.0,
      config=cfg, key=jax.random.PRNGKey(11))
  col = np.asarray(res.sfc_abs_frac)
  p = cfg.photons_per_pixel
  stderr = np.sqrt(col.mean() * (1 - col.mean()) / p)
  assert col.std() < 5 * stderr


def test_determinism_same_key():
  geom = _geom()
  k_ext, ssa, g = _uniform_fields(geom, 2e-3, 0.7, 0.4)
  cfg = MC3DRadiationConfig(photons_per_pixel=512)
  args = dict(mu0=0.8, azimuth=0.1, albedo=0.3, config=cfg,
              key=jax.random.PRNGKey(5))
  a = solve_sw_monochromatic(k_ext, ssa, g, geom, **args)
  b = solve_sw_monochromatic(k_ext, ssa, g, geom, **args)
  np.testing.assert_array_equal(np.asarray(a.abs_frac), np.asarray(b.abs_frac))


def test_jit_compiles():
  geom = _geom()
  k_ext, ssa, g = _uniform_fields(geom, 2e-3, 0.5, 0.2)
  cfg = MC3DRadiationConfig(photons_per_pixel=256)
  fn = jax.jit(
      lambda ke, w, gg, k: solve_sw_monochromatic(
          ke, w, gg, geom, mu0=0.9, azimuth=0.0, albedo=0.1,
          config=cfg, key=k),
  )
  res = fn(k_ext, ssa, g, jax.random.PRNGKey(0))
  assert np.isfinite(float(res.vol_abs_total))


def test_n_batches_invariance():
  """Splitting photons into batches must not change the answer for a fixed key
  vs a relabelled key path -- here we only require finiteness + conservation,
  since batching reshuffles the RNG stream."""
  geom = _geom()
  k_ext, ssa, g = _uniform_fields(geom, 2e-3, 0.6, 0.3)
  cfg = MC3DRadiationConfig(photons_per_pixel=1024, n_batches=4)
  res = solve_sw_monochromatic(
      k_ext, ssa, g, geom, mu0=0.8, azimuth=0.2, albedo=0.2,
      config=cfg, key=jax.random.PRNGKey(9))
  total = (float(res.vol_abs_total) + float(res.sfc_abs_total)
           + float(res.tod_up_frac))
  assert abs(total - 1.0) < 1e-12


def test_n_batches_must_divide():
  geom = _geom(nx=3, ny=3)
  k_ext, ssa, g = _uniform_fields(geom, 1e-3, 0.0, 0.0)
  cfg = MC3DRadiationConfig(photons_per_pixel=10, n_batches=7)
  with pytest.raises(ValueError, match="must divide"):
    solve_sw_monochromatic(k_ext, ssa, g, geom, mu0=1.0, azimuth=0.0,
                           albedo=0.0, config=cfg, key=jax.random.PRNGKey(0))


def test_overflow_guard_raises_without_x64():
  """Non-vacuity self-test: with x64 OFF and N > int32 capacity the integer
  tally would wrap, so the guard must refuse loudly (no silent coerce). x64 is
  a read-only JAX property, so the pure guard fn is exercised directly."""
  from legoesm.atmosphere.physics.radiation.mc3d.tally import (
      assert_count_capacity,
  )
  # Below capacity with x64 off: allowed. Above: raises.
  assert_count_capacity(2**31 - 1, x64_enabled=False)
  with pytest.raises(ValueError, match="int32"):
    assert_count_capacity(2**31, x64_enabled=False)
  # x64 on: any size allowed (true int64 accumulator).
  assert_count_capacity(2**40, x64_enabled=True)


def test_qrng_launch_unbiased_and_conserves():
  """Quasi-random (Halton) photon launch is UNBIASED: it reproduces the
  Beer-Lambert slab transmittance and conserves exactly, matching the
  pseudo-random launch in expectation (it only reduces variance for
  slant/heterogeneous launch — not asserted here)."""
  geom = _geom()
  H = float(geom.z_faces[-1])
  tau = 1.0
  k_ext, ssa, g = _uniform_fields(geom, tau / H, 0.0, 0.0)
  res = solve_sw_monochromatic(
      k_ext, ssa, g, geom, mu0=1.0, azimuth=0.0, albedo=0.0,
      config=MC3DRadiationConfig(photons_per_pixel=4000, use_qrng=True),
      key=jax.random.PRNGKey(0))
  exact = np.exp(-tau)
  n = geom.nx * geom.ny * 4000
  stderr = np.sqrt(exact * (1 - exact) / n)
  assert abs(float(res.sfc_abs_total) - exact) < 6 * stderr
  tot = (float(res.vol_abs_total) + float(res.sfc_abs_total)
         + float(res.tod_up_frac))
  assert abs(tot - 1.0) < 1e-12


def test_qrng_capacity_guard():
  """Non-vacuity: QRNG with photons_per_pixel beyond the Halton base-2 capacity
  must refuse loudly (silent aliasing would defeat the variance reduction)."""
  geom = _geom(nx=1, ny=1, nz=2)
  k_ext, ssa, g = _uniform_fields(geom, 1e-3, 0.0, 0.0)
  cfg = MC3DRadiationConfig(photons_per_pixel=2 ** 24 + 1, use_qrng=True)
  with pytest.raises(ValueError, match="Halton base-2 capacity"):
    solve_sw_monochromatic(k_ext, ssa, g, geom, mu0=1.0, azimuth=0.0,
                           albedo=0.0, config=cfg, key=jax.random.PRNGKey(0))


def test_rayleigh_phase_matches_isotropic_and_conserves():
  """Explicit Rayleigh (1+mu^2) gas scattering has g=0, so a conservative slab's
  reflectance matches isotropic HG(g=0) within MC error, differs from a forward
  HG(g=0.85) slab, and still conserves exactly."""
  geom = _geom(nz=12)
  nx, ny, nz = geom.nx, geom.ny, geom.nz
  k_ext = jnp.full((nx, ny, nz), 2.0 / float(geom.z_faces[-1]))  # tau~2
  ssa = jnp.ones((nx, ny, nz))                                   # conservative
  base = dict(mu0=1.0, azimuth=0.0, albedo=0.0,
              config=MC3DRadiationConfig(photons_per_pixel=4000))

  rayleigh = solve_sw_monochromatic(
      k_ext, ssa, jnp.zeros_like(k_ext), geom, key=jax.random.PRNGKey(0),
      rayleigh_frac=jnp.ones_like(k_ext), **base)            # pure Rayleigh
  iso = solve_sw_monochromatic(
      k_ext, ssa, jnp.zeros_like(k_ext), geom, key=jax.random.PRNGKey(1),
      **base)                                                # HG(g=0)=isotropic
  fwd = solve_sw_monochromatic(
      k_ext, ssa, jnp.full_like(k_ext, 0.85), geom, key=jax.random.PRNGKey(2),
      **base)                                                # forward HG

  n = nx * ny * 4000
  stderr = np.sqrt(0.25 / n)
  # Rayleigh (g=0) and isotropic (g=0) reflect the same to MC error.
  assert abs(float(rayleigh.tod_up_frac) - float(iso.tod_up_frac)) < 8 * stderr
  # Strong forward scattering reflects much less (transmits more).
  assert float(fwd.tod_up_frac) < float(rayleigh.tod_up_frac) - 20 * stderr
  # Conservation holds with the Rayleigh split.
  tot = (float(rayleigh.vol_abs_total) + float(rayleigh.sfc_abs_total)
         + float(rayleigh.tod_up_frac))
  assert abs(tot - 1.0) < 1e-12


def test_coarse_majorant_unbiased_vs_scalar():
  """Coarse majorant grid (Phase 1.1) is a pure speedup: a cloudy field gives
  the SAME absorption/transmission as the global-scalar majorant within MC
  error (delta tracking is unbiased for any valid per-region majorant)."""
  geom = _geom(nx=8, ny=8, nz=16)
  nx, ny, nz = geom.nx, geom.ny, geom.nz
  # clear background + thick absorbing/scattering cloud block aloft
  k_ext = jnp.full((nx, ny, nz), 1e-4).at[2:6, 2:6, nz // 2:].set(0.02)
  ssa = jnp.where(k_ext > 1e-3, 0.7, 0.0)
  g = jnp.where(k_ext > 1e-3, 0.5, 0.0)
  base = dict(mu0=1.0, azimuth=0.0, albedo=0.1)

  scalar = solve_sw_monochromatic(
      k_ext, ssa, g, geom, **base,
      config=MC3DRadiationConfig(photons_per_pixel=4000),
      key=jax.random.PRNGKey(0))
  coarse = solve_sw_monochromatic(
      k_ext, ssa, g, geom, **base,
      config=MC3DRadiationConfig(photons_per_pixel=4000,
                                 knull_coarsen_xy=2, knull_coarsen_z=4),
      key=jax.random.PRNGKey(1))
  # Domain budgets agree within MC error (different RNG streams).
  assert abs(float(scalar.vol_abs_total) - float(coarse.vol_abs_total)) < 0.02
  assert abs(float(scalar.sfc_abs_total) - float(coarse.sfc_abs_total)) < 0.02
  assert abs(float(scalar.tod_up_frac) - float(coarse.tod_up_frac)) < 0.02
  # Both still conserve exactly.
  for r in (scalar, coarse):
    tot = (float(r.vol_abs_total) + float(r.sfc_abs_total)
           + float(r.tod_up_frac))
    assert abs(tot - 1.0) < 1e-12


def test_coarse_majorant_unbiased_slant_scattering():
  """Slant beam + scattering => photons move horizontally and cross periodic
  coarse-cell seams (the DDA wrap path). Horizontally-coarsened result must
  still match the scalar majorant within MC error (seam consistency)."""
  geom = _geom(nx=8, ny=8, nz=12)
  nx, ny, nz = geom.nx, geom.ny, geom.nz
  k_ext = jnp.full((nx, ny, nz), 2e-3).at[1:4, 5:8, nz // 3:2 * nz // 3].set(0.02)
  ssa = jnp.where(k_ext > 1e-2, 0.85, 0.3)   # scattering everywhere
  g = jnp.full((nx, ny, nz), 0.4)
  base = dict(mu0=0.5, azimuth=0.7, albedo=0.3)   # slant + reflecting surface
  scalar = solve_sw_monochromatic(
      k_ext, ssa, g, geom, **base,
      config=MC3DRadiationConfig(photons_per_pixel=5000),
      key=jax.random.PRNGKey(0))
  coarse = solve_sw_monochromatic(
      k_ext, ssa, g, geom, **base,
      config=MC3DRadiationConfig(photons_per_pixel=5000,
                                 knull_coarsen_xy=2, knull_coarsen_z=3),
      key=jax.random.PRNGKey(1))
  assert abs(float(scalar.vol_abs_total) - float(coarse.vol_abs_total)) < 0.025
  assert abs(float(scalar.sfc_abs_total) - float(coarse.sfc_abs_total)) < 0.025
  assert abs(float(scalar.tod_up_frac) - float(coarse.tod_up_frac)) < 0.025
  for r in (scalar, coarse):
    tot = (float(r.vol_abs_total) + float(r.sfc_abs_total)
           + float(r.tod_up_frac))
    assert abs(tot - 1.0) < 1e-12
    assert float(r.maxiter_frac) < 5e-3


def test_coarse_majorant_grid_build():
  """build_majorant_grid: per-block max, floor, divisibility guard, z faces."""
  from legoesm.atmosphere.physics.radiation.mc3d import build_majorant_grid
  geom = _geom(nx=8, ny=8, nz=12)
  k = jnp.zeros((8, 8, 12)).at[0, 0, 0].set(5.0).at[7, 7, 11].set(3.0)
  m = build_majorant_grid(k, geom.z_faces, 1e-3, coarsen_xy=4, coarsen_z=6)
  assert m.grid.shape == (2, 2, 2)
  assert float(m.grid[0, 0, 0]) == 5.0      # block max
  assert float(m.grid[1, 1, 1]) == 3.0
  assert float(m.grid[0, 1, 0]) == 1e-3     # empty block -> floor
  assert m.zc_faces.shape == (3,)
  assert float(m.zc_faces[0]) == 0.0
  assert float(m.zc_faces[-1]) == float(geom.z_faces[-1])
  with pytest.raises(ValueError, match="must divide"):
    build_majorant_grid(k, geom.z_faces, 1e-3, coarsen_xy=3)


def test_horizontal_photon_on_top_face_no_nan():
  """A purely horizontal photon exactly on the top face must not produce a
  0/0 NaN in the z face-distance (u==0 -> inf guard)."""
  from legoesm.atmosphere.physics.radiation.mc3d import (
      build_majorant_grid, trace_one,
  )
  from legoesm.atmosphere.physics.radiation.mc3d.photon_walk import _cell_indices
  geom = _geom(nx=8, ny=8, nz=8)
  k_ext, ssa, g = _uniform_fields(geom, 2e-3, 0.0, 0.0)
  maj = build_majorant_grid(k_ext, geom.z_faces, 1e-3,
                            coarsen_xy=2, coarsen_z=2)
  ztop = float(geom.z_faces[-1])
  # Several zero-component directions on faces (x-horizontal, y-horizontal,
  # two-zero corner) — each must stay finite and terminate.
  cases = [
      (jnp.array([25.0, 25.0, ztop]), jnp.array([1.0, 0.0, 0.0])),
      (jnp.array([25.0, 25.0, ztop]), jnp.array([0.0, 1.0, 0.0])),
      (jnp.array([0.0, 25.0, 200.0]), jnp.array([-1.0, 0.0, 0.0])),  # on x-face
      (jnp.array([25.0, 25.0, 600.0]), jnp.array([0.0, 0.0, -1.0])),  # vertical
  ]
  for pos0, dir0 in cases:
    st = trace_one(jax.random.PRNGKey(0), pos0, dir0, k_ext, ssa, g, maj,
                   0.0, geom, 4096)
    assert bool(jnp.all(jnp.isfinite(st.pos)))
    assert int(st.status) in (1, 2, 3, 4)   # terminated in a real bin


def test_tally_batch_direct():
  """Exercise trace_batch + tally_batch directly (leaf-module coverage)."""
  geom = _geom(nx=4, ny=4, nz=8)
  k_ext, ssa, g = _uniform_fields(geom, 2e-3, 0.0, 0.0)
  from legoesm.atmosphere.physics.radiation.mc3d import build_majorant_grid
  maj = build_majorant_grid(k_ext, geom.z_faces, 1e-3)
  p = 200
  ncols = geom.nx * geom.ny
  n = ncols * p
  keys = jax.random.split(jax.random.PRNGKey(0), n)
  # one photon per column footprint, vertical beam at TOD
  col = jnp.repeat(jnp.arange(ncols), p)
  ix, iy = col // geom.ny, col % geom.ny
  pos = jnp.stack([(ix + 0.5) * geom.dx, (iy + 0.5) * geom.dx,
                   jnp.full((n,), geom.z_faces[-1])], axis=1)
  dirs = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0]), (n, 3))
  states = trace_batch(keys, pos, dirs, k_ext, ssa, g, maj, 0.0, geom, 4096)
  res = tally_batch(states, geom, p)
  total = (float(res.vol_abs_total) + float(res.sfc_abs_total)
           + float(res.tod_up_frac))
  assert abs(total - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# HG mixture sampling: g_cloud clip must stay STRICTLY inside (-1, 1)
# (FIX: an inclusive clip to +-1.0 feeds henyey_greenstein_mu its g=+1
# singularity s = (1-g^2)/(1-g+2gu) = 0/0 at the attainable draw u=0).
# ---------------------------------------------------------------------------


def test_mixed_scatter_clipped_g_finite_directions():
  """Cloud-dominated cell with g/(1-rayleigh_frac) > 1 (clip engaged at
  _HG_G_MAX): every sampled direction is a finite unit vector."""
  from legoesm.atmosphere.physics.radiation.mc3d import sampling
  direction = jnp.array([0.0, 0.0, -1.0])
  g = jnp.asarray(0.95)
  rayleigh_frac = jnp.asarray(0.1)   # g/one_m = 1.055... -> clipped
  keys = jax.random.split(jax.random.PRNGKey(7), 512)
  dirs = jax.vmap(
      lambda k: sampling.scatter_direction_mixed(k, direction, g,
                                                 rayleigh_frac))(keys)
  assert bool(jnp.all(jnp.isfinite(dirs)))
  np.testing.assert_allclose(
      np.asarray(jnp.linalg.norm(dirs, axis=-1)), 1.0, atol=1e-6)


def test_mixed_scatter_clipped_g_grad_finite():
  """The clipped extreme stays differentiable: d(direction)/dg finite (the
  clip VJP is zero past the bound, never NaN -- and jnp.where's both-branch
  evaluation must not leak a NaN from the cloud branch)."""
  from legoesm.atmosphere.physics.radiation.mc3d import sampling
  direction = jnp.array([0.0, 0.0, -1.0])
  rayleigh_frac = jnp.asarray(0.1)
  key = jax.random.PRNGKey(3)

  def loss(g):
    d = sampling.scatter_direction_mixed(key, direction, g, rayleigh_frac)
    return jnp.sum(d * jnp.array([1.0, 2.0, 3.0]))

  for g0 in (0.95, 0.89):   # clipped (0.95/0.9 > 1) and un-clipped
    grad = jax.grad(loss)(jnp.asarray(g0))
    assert bool(jnp.isfinite(grad)), f"non-finite grad at g={g0}"


def test_hg_mu_finite_at_clip_bound_worst_case_u0(monkeypatch):
  """Non-vacuous singularity guard: at the worst-case uniform draw u = 0.0
  (attainable from jax.random.uniform's [0,1) range) the HG inversion stays
  finite for ANY g in [-1, 1] because henyey_greenstein_mu clips g at the
  SOURCE. Both the pure-HG path (scatter_direction) and the direct call are
  protected, and the raw (unclipped) formula at g=1.0, u=0 is shown to NaN so
  the clip is provably load-bearing."""
  from legoesm.atmosphere.physics.radiation.mc3d import sampling

  def _u_zero(key, *args, dtype=jnp.float64, **kwargs):
    return jnp.zeros((), dtype=dtype)

  monkeypatch.setattr(jax.random, "uniform", _u_zero)
  # Source clip: even a caller passing exactly g = 1.0 gets a finite mu = -1
  # (analytic u=0 limit, s = 1 + g), not the 0/0 NaN.
  mu_boundary = sampling.henyey_greenstein_mu(
      jax.random.PRNGKey(0), jnp.asarray(1.0))
  assert bool(jnp.isfinite(mu_boundary))
  np.testing.assert_allclose(float(mu_boundary), -1.0, atol=1e-9)
  # The default (rayleigh_frac=None) pure-HG scatter path is finite at g=1.0
  # too — the HIGH-severity gap the source clip closes.
  new_dir = sampling.scatter_direction(
      jax.random.PRNGKey(0), jnp.asarray([0.0, 0.0, 1.0]), jnp.asarray(1.0))
  assert bool(jnp.all(jnp.isfinite(new_dir)))
  np.testing.assert_allclose(float(jnp.linalg.norm(new_dir)), 1.0, atol=1e-9)
  # Hazard is real: the RAW unclipped HG inversion at g=1.0, u=0 is 0/0 = NaN
  # (IEEE), proving the source clip is load-bearing (not a no-op).
  g_raw = jnp.asarray(1.0)
  u0 = jnp.asarray(0.0)
  s_raw = (1.0 - g_raw * g_raw) / (1.0 - g_raw + 2.0 * g_raw * u0)
  mu_raw = (1.0 + g_raw * g_raw - s_raw * s_raw) / (2.0 * g_raw)
  assert bool(jnp.isnan(mu_raw))
