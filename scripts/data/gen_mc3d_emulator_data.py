"""Generate a training dataset for the mc3d 3D-CNN radiation emulator.

Each sample is a random broken-cumulus OPTICAL field (k_ext, ssa, g,
rayleigh_frac, r_eff) + solar geometry, paired with the Monte-Carlo absorbed-
flux fraction field (the expensive transport the emulator learns). Mie cloud
phase is used (microhh LUT, band 7) so the targets carry the oracle-faithful 3D
cloud scattering. Saves a compressed .npz of stacked inputs + targets.

Usage:
    JAX_ENABLE_X64=1 python scripts/data/gen_mc3d_emulator_data.py \
        --n 500 --nx 16 --nz 16 --photons 1500 --out scripts/tmp/_mc3d_emu.npz
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.radiation.mc3d import mie
from legoesm.atmosphere.physics.radiation.mc3d.config import MC3DRadiationConfig
from legoesm.atmosphere.physics.radiation.mc3d.photon_walk import PlaneRTGeometry
from legoesm.atmosphere.physics.radiation.mc3d.raytracer_sw import (
    solve_sw_monochromatic,
)


def _random_cloud_field(key, nx, ny, nz):
  """Random broken-cumulus optical field (numpy)."""
  rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31)))
  k_ext = np.full((nx, ny, nz), 1e-4, np.float64)
  ssa = np.full((nx, ny, nz), 0.4, np.float64)        # weak clear-air Rayleigh
  g = np.zeros((nx, ny, nz))
  rayl = np.ones((nx, ny, nz))                        # clear: all-Rayleigh scat
  reff = np.full((nx, ny, nz), 2.5)
  n_blocks = rng.integers(1, 6)
  for _ in range(n_blocks):
    bx, by = rng.integers(2, max(3, nx // 2)), rng.integers(2, max(3, ny // 2))
    x0 = rng.integers(0, nx - bx + 1); y0 = rng.integers(0, ny - by + 1)
    ztop = rng.integers(nz // 3, nz); zbot = rng.integers(0, ztop)
    kc = rng.uniform(0.005, 0.05)
    sl = (slice(x0, x0 + bx), slice(y0, y0 + by), slice(zbot, ztop))
    k_ext[sl] = kc
    ssa[sl] = rng.uniform(0.95, 0.999)
    g[sl] = rng.uniform(0.80, 0.88)
    rayl[sl] = 0.01
    reff[sl] = rng.uniform(5.0, 18.0)
  mu0 = float(rng.uniform(0.3, 1.0))
  azi = float(rng.uniform(0.0, 2 * np.pi))
  return (jnp.asarray(k_ext), jnp.asarray(ssa), jnp.asarray(g),
          jnp.asarray(rayl), jnp.asarray(reff), mu0, azi)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--n", type=int, default=500)
  ap.add_argument("--nx", type=int, default=16)
  ap.add_argument("--nz", type=int, default=16)
  ap.add_argument("--photons", type=int, default=1500)
  ap.add_argument("--dx", type=float, default=50.0)
  ap.add_argument("--height", type=float, default=2000.0)
  ap.add_argument("--band", type=int, default=7)
  ap.add_argument("--out", default="scripts/tmp/_mc3d_emu.npz")
  args = ap.parse_args(argv)

  nx = ny = args.nx
  nz = args.nz
  geom = PlaneRTGeometry(dx=args.dx, dy=args.dx, Lx=nx * args.dx, Ly=ny * args.dx,
                         z_faces=jnp.linspace(0.0, args.height, nz + 1),
                         nx=nx, ny=ny, nz=nz)
  lut = mie.load_mie_sampling_lut()
  cdf_b = lut.phase_cdf[args.band][None]              # (1,n_mie)
  ang_b = lut.phase_cdf_angle[args.band][None]        # (1,n_r,n_mie)
  cfg = MC3DRadiationConfig(photons_per_pixel=args.photons,
                            knull_coarsen_xy=max(nx // 4, 1),
                            knull_coarsen_z=max(nz // 4, 1))

  inputs, targets = [], []
  key = jax.random.PRNGKey(0)
  for i in range(args.n):
    key, kf, ks = jax.random.split(key, 3)
    k_ext, ssa, g, rayl, reff, mu0, azi = _random_cloud_field(kf, nx, ny, nz)
    res = solve_sw_monochromatic(
        k_ext, ssa, g, geom, mu0=mu0, azimuth=azi, albedo=0.1, config=cfg,
        key=ks, rayleigh_frac=rayl, mie_cdf=cdf_b[0], mie_ang=ang_b[0],
        r_eff=reff)
    inputs.append(dict(k_ext=np.asarray(k_ext), ssa=np.asarray(ssa),
                       g=np.asarray(g), rayleigh_frac=np.asarray(rayl),
                       r_eff=np.asarray(reff), mu0=mu0, azimuth=azi))
    targets.append(np.asarray(res.abs_frac))
    if (i + 1) % 25 == 0:
      print(f"  {i + 1}/{args.n}")

  np.savez_compressed(
      args.out,
      k_ext=np.stack([d["k_ext"] for d in inputs]),
      ssa=np.stack([d["ssa"] for d in inputs]),
      g=np.stack([d["g"] for d in inputs]),
      rayleigh_frac=np.stack([d["rayleigh_frac"] for d in inputs]),
      r_eff=np.stack([d["r_eff"] for d in inputs]),
      mu0=np.array([d["mu0"] for d in inputs]),
      azimuth=np.array([d["azimuth"] for d in inputs]),
      abs_frac=np.stack(targets),
  )
  print(f"saved {args.n} samples -> {args.out}")


if __name__ == "__main__":
  main()
