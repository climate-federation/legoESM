"""Compare the mc3d Mie cloud-phase sampler against the microhh oracle.

microhh's ray tracer is CUDA-only (`rte-rrtmgp-cpp/src_cuda_rt`), so a live
head-to-head photon run needs a GPU/NVHPC build. The *numerically comparable*
oracle is microhh's shipped Mie lookup table (`data/mie_lut_broadband.nc`) plus
its `raytracer_functions.h::mie_sample_angle` algorithm — both reproduced here
exactly. This script reports:

  1. BIT-EXACTNESS: mc3d `mie_sample_cos` vs a NumPy transcription of microhh's
     `mie_sample_angle` on microhh's own LUT (max |Δcos|).
  2. PHYSICAL FIDELITY: the sampled scattering-angle distribution vs microhh's
     tabulated phase function (mean angle, asymmetry g, binned L1) per band/r_eff.

Usage:
    JAX_ENABLE_X64=1 python scripts/validate/compare_mie_vs_microhh.py \
        [--nc scripts/tmp/_mie/mie_lut_broadband.nc]
"""

from __future__ import annotations

import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.radiation.mc3d.mie import (
    load_mie_sampling_lut,
    mie_sample_cos,
)


def _microhh_ref_cos(u, r_eff, cdf, ang):
  """NumPy transcription of microhh raytracer_functions.h::mie_sample_angle."""
  r_idx = min(max(int(r_eff - 2.5), 0), 18)
  r_rest = (r_eff - 2.5) % 1.0
  left, right = 0, len(cdf) - 1
  while left < right:
    mid = left + (right - left) // 2
    if u >= cdf[mid]:
      right = mid
    else:
      left = mid + 1
  i = min(max(left - 1, 0), len(cdf) - 2)
  dr = abs(cdf[i + 1] - cdf[i]) or 1.0
  a = lambda rr: (abs(u - cdf[i + 1]) * ang[rr, i]
                  + abs(cdf[i] - u) * ang[rr, i + 1]) / dr
  return np.cos(a(r_idx) * (1.0 - r_rest) + a(r_idx + 1) * r_rest)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--nc", default="scripts/tmp/_mie/mie_lut_broadband.nc",
                  help="full microhh Mie LUT (for the phase-table check)")
  ap.add_argument("--n-sample", type=int, default=300000)
  args = ap.parse_args(argv)

  lut = load_mie_sampling_lut()
  cdf = np.asarray(lut.phase_cdf)
  ang = np.asarray(lut.phase_cdf_angle)

  # 1. Bit-exactness vs microhh's mie_sample_angle.
  rng = np.random.default_rng(0)
  max_d = 0.0
  for _ in range(2000):
    b = int(rng.integers(0, lut.n_band))
    u = float(rng.uniform())
    r = float(rng.uniform(2.5, 21.5))
    ref = _microhh_ref_cos(u, r, cdf[b], ang[b])
    got = float(mie_sample_cos(jnp.float64(u), jnp.float64(r),
                               lut.phase_cdf[b], lut.phase_cdf_angle[b]))
    max_d = max(max_d, abs(got - ref))
  print(f"[1] bit-exactness vs microhh mie_sample_angle: "
        f"max|Δcos| = {max_d:.3e} over 2000 (band,u,r_eff)")

  # 2. Distribution vs microhh's tabulated phase function.
  ncp = pathlib.Path(args.nc)
  if not ncp.exists():
    print(f"[2] skipped: full LUT {ncp} not present (phase-table check)")
    return
  import netCDF4 as nc
  d = nc.Dataset(str(ncp))
  phase = np.asarray(d["phase"][:])       # (band, r, n_ang)
  pang = np.asarray(d["phase_angle"][:])  # (n_ang,)
  print("[2] sampled distribution vs microhh phase table:")
  print(f"     {'band':>4} {'r_eff':>6} {'g_mc':>7} {'g_ref':>7} "
        f"{'<ang>_mc':>9} {'<ang>_ref':>10} {'L1':>6}")
  for b in (1, 7, 13):
    for r_eff in (3.5, 10.5, 20.5):
      us = jax.random.uniform(jax.random.PRNGKey(b * 100 + int(r_eff)),
                              (args.n_sample,))
      cos = np.asarray(jax.vmap(lambda u: mie_sample_cos(
          u, jnp.float64(r_eff), lut.phase_cdf[b], lut.phase_cdf_angle[b]))(us))
      angs = np.arccos(np.clip(cos, -1.0, 1.0))
      ridx = int(r_eff - 2.5)
      pdf = phase[b, ridx] * np.sin(pang)
      pdf /= np.trapezoid(pdf, pang)
      g_ref = float(np.trapezoid(np.cos(pang) * pdf, pang))
      ang_ref = float(np.trapezoid(pang * pdf, pang))
      h, edges = np.histogram(angs, bins=120, range=(0, np.pi), density=True)
      ctr = 0.5 * (edges[:-1] + edges[1:])
      l1 = float(np.sum(np.abs(h - np.interp(ctr, pang, pdf))) /
                 np.sum(np.interp(ctr, pang, pdf)))
      print(f"     {b:>4} {r_eff:>6.1f} {float(cos.mean()):>7.3f} {g_ref:>7.3f} "
            f"{float(angs.mean()):>9.4f} {ang_ref:>10.4f} {l1:>6.3f}")


if __name__ == "__main__":
  main()
