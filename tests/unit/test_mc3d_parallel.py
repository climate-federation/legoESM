"""Phase-4 tests: photon-sharded parallelism for the 3D MC ray tracer.

Sharding photons (not the domain) + averaging the per-shard fractions must match
a single big run (same total photons) and be AD-safe (linear combine). Run with
``JAX_ENABLE_X64=1``.
"""

import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.radiation.mc3d import (
    pmean_result,
    solve_sw_monochromatic,
    solve_sw_sharded,
)
from legoesm.atmosphere.physics.radiation.mc3d.config import MC3DRadiationConfig
from legoesm.atmosphere.physics.radiation.mc3d.photon_walk import PlaneRTGeometry


def _geom(nx=8, ny=8, nz=16, dx=50.0, H=1000.0):
  return PlaneRTGeometry(dx=dx, dy=dx, Lx=nx * dx, Ly=ny * dx,
                         z_faces=jnp.linspace(0.0, H, nz + 1),
                         nx=nx, ny=ny, nz=nz)


def _fields(geom, k, ssa, g):
  shape = (geom.nx, geom.ny, geom.nz)
  return jnp.full(shape, k), jnp.full(shape, ssa), jnp.full(shape, g)


def test_sharded_matches_single_run():
  """4 shards x P photons == 1 run x 4P photons (Beer-Lambert + conservation)."""
  geom = _geom()
  H = float(geom.z_faces[-1])
  tau = 1.0
  k_ext, ssa, g = _fields(geom, tau / H, 0.0, 0.0)
  base = dict(mu0=1.0, azimuth=0.0, albedo=0.0)
  sharded = solve_sw_sharded(
      k_ext, ssa, g, geom, **base,
      config=MC3DRadiationConfig(photons_per_pixel=1500),
      key=jax.random.PRNGKey(0), n_shards=4)
  single = solve_sw_monochromatic(
      k_ext, ssa, g, geom, **base,
      config=MC3DRadiationConfig(photons_per_pixel=6000),
      key=jax.random.PRNGKey(1))
  exact = np.exp(-tau)
  n = geom.nx * geom.ny * 6000
  stderr = np.sqrt(exact * (1 - exact) / n)
  assert abs(float(sharded.sfc_abs_total) - exact) < 5 * stderr
  assert abs(float(sharded.sfc_abs_total) - float(single.sfc_abs_total)) < 6 * stderr
  # Combined fractions still conserve exactly (mean of per-shard exact-1 sums).
  tot = (float(sharded.vol_abs_total) + float(sharded.sfc_abs_total)
         + float(sharded.tod_up_frac))
  assert abs(tot - 1.0) < 1e-12


def test_sharded_variance_shrinks_with_shards():
  """More shards (more total photons) -> tighter estimate of slab transmittance."""
  geom = _geom()
  H = float(geom.z_faces[-1])
  k_ext, ssa, g = _fields(geom, 1.0 / H, 0.0, 0.0)
  exact = np.exp(-1.0)
  base = dict(mu0=1.0, azimuth=0.0, albedo=0.0,
              config=MC3DRadiationConfig(photons_per_pixel=800))
  e1 = abs(float(solve_sw_sharded(k_ext, ssa, g, geom, **base,
           key=jax.random.PRNGKey(3), n_shards=1).sfc_abs_total) - exact)
  e8 = abs(float(solve_sw_sharded(k_ext, ssa, g, geom, **base,
           key=jax.random.PRNGKey(3), n_shards=8).sfc_abs_total) - exact)
  assert e8 < e1 + 1e-6   # more photons should not be worse (usually tighter)




def test_solve_sw_sharded_guard():
  geom = _geom(nx=2, ny=2, nz=2)
  k_ext, ssa, g = _fields(geom, 1e-3, 0.0, 0.0)
  with pytest.raises(ValueError, match="n_shards"):
    solve_sw_sharded(k_ext, ssa, g, geom, mu0=1.0, azimuth=0.0, albedo=0.0,
                     config=MC3DRadiationConfig(photons_per_pixel=16),
                     key=jax.random.PRNGKey(0), n_shards=0)


def test_pmean_result_over_devices():
  """pmean_result composes under pmap (AD-safe collective). With one local
  device the mean is the identity; the path is exercised either way."""
  geom = _geom(nx=4, ny=4, nz=4)
  k_ext, ssa, g = _fields(geom, 2e-3, 0.0, 0.0)
  nd = jax.local_device_count()
  cfg = MC3DRadiationConfig(photons_per_pixel=64)
  keys = jax.random.split(jax.random.PRNGKey(0), nd)

  def per_device(key):
    res = solve_sw_monochromatic(k_ext, ssa, g, geom, mu0=1.0, azimuth=0.0,
                                 albedo=0.0, config=cfg, key=key)
    return pmean_result(res, "d")

  out = jax.pmap(per_device, axis_name="d")(keys)
  # All devices hold the combined (identical) result; finite + conserves.
  tot = (float(out.vol_abs_total[0]) + float(out.sfc_abs_total[0])
         + float(out.tod_up_frac[0]))
  assert abs(tot - 1.0) < 1e-9


def test_bench_smoke():
  """The bench hook runs and returns positive throughput metrics."""
  path = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "bench" / "bench_mc3d_raytracer.py")
  spec = importlib.util.spec_from_file_location("bench_mc3d_raytracer", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  m = mod.benchmark(nx=4, ny=4, nz=6, photons=32, n_reps=1)
  assert m["photons_per_second"] > 0.0
  assert m["n_total_photons"] == 4 * 4 * 32
