"""Throughput benchmark for the 3D Monte-Carlo shortwave ray tracer.

Measures photons/second and the coarse-majorant (Phase 1.1) speedup on a
cloud-in-clear field, and is the hook for GPU photon-shard scaling (run one
shard per device; combine with ``parallel.pmean_result``). Photon-sharding is
embarrassingly parallel — wall-clock ~ 1/n_devices for a fixed total photon
count, the optical field replicated per device (cheap, per-g-point).

Usage:
    JAX_ENABLE_X64=1 python scripts/bench/bench_mc3d_raytracer.py \
        --nx 64 --ny 64 --nz 64 --photons 128 --coarsen-xy 8 --coarsen-z 8
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.radiation.mc3d import solve_sw_monochromatic
from legoesm.atmosphere.physics.radiation.mc3d.config import MC3DRadiationConfig
from legoesm.atmosphere.physics.radiation.mc3d.photon_walk import PlaneRTGeometry


def _cloud_field(nx: int, ny: int, nz: int):
  """Clear background + a central scattering cloud block aloft."""
  k_ext = jnp.full((nx, ny, nz), 1e-4)
  k_ext = k_ext.at[nx // 4: 3 * nx // 4, ny // 4: 3 * ny // 4,
                   nz // 2:].set(0.02)
  ssa = jnp.where(k_ext > 1e-3, 0.99, 0.0)
  g = jnp.where(k_ext > 1e-3, 0.85, 0.0)
  return k_ext, ssa, g


def benchmark(nx=32, ny=32, nz=32, photons=64, coarsen_xy=0, coarsen_z=0,
              n_reps=1, dx=50.0, height=2000.0):
  """Time one shortwave solve; return a metrics dict (blocks on device)."""
  geom = PlaneRTGeometry(dx=dx, dy=dx, Lx=nx * dx, Ly=ny * dx,
                         z_faces=jnp.linspace(0.0, height, nz + 1),
                         nx=nx, ny=ny, nz=nz)
  k_ext, ssa, g = _cloud_field(nx, ny, nz)
  cfg = MC3DRadiationConfig(photons_per_pixel=photons,
                            knull_coarsen_xy=coarsen_xy,
                            knull_coarsen_z=coarsen_z)
  n_total = nx * ny * photons

  def run(key):
    r = solve_sw_monochromatic(k_ext, ssa, g, geom, mu0=0.7, azimuth=0.0,
                               albedo=0.1, config=cfg, key=key)
    # Touch EVERY output leaf so XLA cannot dead-code-eliminate part of the
    # solve out of the timed window (full-solve throughput, not just one total).
    return (r.vol_abs_total + r.sfc_abs_total + r.tod_up_frac
            + r.maxiter_frac + jnp.sum(r.abs_frac) + jnp.sum(r.sfc_abs_frac))

  # Warm up (compile) then time.
  run(jax.random.PRNGKey(0)).block_until_ready()
  t0 = time.perf_counter()
  for i in range(n_reps):
    run(jax.random.PRNGKey(i + 1)).block_until_ready()
  dt = (time.perf_counter() - t0) / n_reps
  return {
      "n_total_photons": n_total,
      "seconds_per_solve": dt,
      "photons_per_second": n_total / dt if dt > 0 else float("inf"),
      "coarsen_xy": coarsen_xy,
      "coarsen_z": coarsen_z,
  }


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--nx", type=int, default=32)
  ap.add_argument("--ny", type=int, default=32)
  ap.add_argument("--nz", type=int, default=32)
  ap.add_argument("--photons", type=int, default=64)
  ap.add_argument("--coarsen-xy", type=int, default=0)
  ap.add_argument("--coarsen-z", type=int, default=0)
  ap.add_argument("--reps", type=int, default=3)
  args = ap.parse_args(argv)
  m = benchmark(args.nx, args.ny, args.nz, args.photons,
                args.coarsen_xy, args.coarsen_z, args.reps)
  print(f"grid {args.nx}x{args.ny}x{args.nz}  photons/col {args.photons}  "
        f"coarsen xy={args.coarsen_xy} z={args.coarsen_z}")
  print(f"  {m['seconds_per_solve']:.3f} s/solve  "
        f"{m['photons_per_second']:.3e} photons/s "
        f"({m['n_total_photons']:.3e} total)")
  return m


if __name__ == "__main__":
  main()
