"""Generate tests/atmosphere/oracle_fixtures/mc3d_walk.npz.

Oracle: microhh rte-rrtmgp-cpp v0.3 (fc291af01820674ff6dcd05889d7eb191b3daca4)
``Raytracer::trace_rays`` (src_cuda_rt/Raytracer.cu + ray_tracer_kernel in
src_kernels_cuda_rt/raytracer_kernels.cu), built in double precision by
build.sh.  The walk is stochastic and its RNG (curand Sobol launch + XORWOW) is
not ours, so the comparison is statistical: N_SEEDS independent oracle runs per
case, stored as per-quantity mean and standard deviation; the test runs the
same number of legoESM seeds and applies a Welch t-test per quantity with a
Bonferroni-corrected three-sigma familywise level.

Cases (16x16x16, dx = dy = dz = 100 m, sza 30 deg, albedo 0.2, 256 photons per
column, coarse null-collision grid 4x4x4):
  clear   : absorbing + Rayleigh-scattering gas only
  block   : clear + HG cloud block (tau 1 per layer, ssa 0.999, g 0.85)
  broken  : clear + lognormal broken cloud with the Mie phase (LUT band 9)
  thick   : clear + overcast HG deck, column tau 48 (stresses the weights)
  diffuse : block with 30 % diffuse top-of-domain incidence

usage:
  python make_fixture.py inputs <dir>                  # write walk_in_<case>_<seed>.bin
  (GPU)  walk_driver <dir>/walk_in_X.bin <dir>/walk_out_X.bin   for every input
  python make_fixture.py collect <dir> <rte-rrtmgp-cpp@v0.3> <out.npz>
"""
import json
import os
import subprocess
import sys

import numpy as np

NX = NY = NZ = 16
DX = DY = DZ = 100.0
KN = (4, 4, 4)
PPP = 256
SZA = np.deg2rad(30.0)
AZI_MICROHH = 0.6          # microhh convention: 0 = north, clockwise
ALBEDO = 0.2
N_SEEDS = 20
MIE_BAND = 9
CASES = ("clear", "block", "broken", "thick", "diffuse")
NCELL = NX * NY * NZ


def fields(case):
  """Optical fields as (nx, ny, nz) arrays: tau_tot, ssa_tot, tau_cld,
  ssa_cld, asy_cld, r_eff [um], and the diffuse share of the incidence."""
  shp = (NX, NY, NZ)
  tau_gas = np.full(shp, 0.02)
  ssa_gas = np.full(shp, 0.8)
  tau_cld = np.zeros(shp)
  ssa_cld = np.zeros(shp)
  asy_cld = np.zeros(shp)
  r_eff = np.full(shp, 10.0)
  dif = 0.0
  if case in ("block", "diffuse"):
    tau_cld[4:12, 4:12, 6:10] = 1.0
    ssa_cld[4:12, 4:12, 6:10] = 0.999
    asy_cld[4:12, 4:12, 6:10] = 0.85
    dif = 0.3 if case == "diffuse" else 0.0
  elif case == "broken":
    rng = np.random.default_rng(1103)
    lw = rng.lognormal(mean=-1.0, sigma=1.0, size=shp)
    cloudy = (lw > 0.6) & (np.arange(NZ)[None, None, :] >= 4) \
        & (np.arange(NZ)[None, None, :] < 12)
    tau_cld[cloudy] = lw[cloudy]
    ssa_cld[cloudy] = 0.9999
    asy_cld[cloudy] = 0.85
    r_eff[cloudy] = rng.uniform(4.0, 18.0, size=int(cloudy.sum()))
  elif case == "thick":
    tau_cld[:, :, 4:12] = 6.0
    ssa_cld[:, :, 4:12] = 0.9999
    asy_cld[:, :, 4:12] = 0.85
  tau_tot = tau_gas + tau_cld
  ssa_tot = (tau_gas * ssa_gas + tau_cld * ssa_cld) / tau_tot
  return tau_tot, ssa_tot, tau_cld, ssa_cld, asy_cld, r_eff, dif


def to_microhh(a):
  """(nx, ny, nz) -> flat idx = x + y*nx + z*nx*ny."""
  return np.ascontiguousarray(np.transpose(a, (2, 1, 0))).ravel()


def from_microhh(v, nz=None):
  if nz is None:
    return np.transpose(v.reshape(NY, NX), (1, 0))
  return np.transpose(v.reshape(nz, NY, NX), (2, 1, 0))


def mie_table():
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
  import jax
  jax.config.update("jax_enable_x64", True)
  import jax.numpy as jnp
  from legoesm.atmosphere.physics.radiation.mc3d import mie
  lut = mie.load_mie_sampling_lut(dtype=jnp.float64)
  return (np.asarray(lut.phase_cdf[MIE_BAND]),
          np.asarray(lut.phase_cdf_angle[MIE_BAND]))


def write_inputs(d):
  os.makedirs(d, exist_ok=True)
  cdf, ang = mie_table()
  for case in CASES:
    tau_tot, ssa_tot, tau_cld, ssa_cld, asy_cld, r_eff, dif = fields(case)
    use_mie = case == "broken"
    c = cdf if use_mie else np.zeros(0)
    a = ang if use_mie else np.zeros((0, 0))
    for s in range(N_SEEDS):
      with open(f"{d}/walk_in_{case}_{s}.bin", "wb") as f:
        np.array([NX, NY, NZ, *KN, c.size, a.shape[0]], np.int32).tofile(f)
        np.array([PPP], np.uint64).tofile(f)
        np.array([s], np.int32).tofile(f)
        np.array([DX, DY, DZ, SZA, AZI_MICROHH, 1.0 - dif, dif, ALBEDO],
                 np.float64).tofile(f)
        for x in (tau_tot, ssa_tot, tau_cld, ssa_cld, asy_cld, r_eff):
          to_microhh(x).astype(np.float64).tofile(f)
        c.astype(np.float64).tofile(f)
        a.astype(np.float64).ravel().tofile(f)
  print("INPUTS_OK", d)


def quantities(tod_up, sfc_dir, sfc_dif, sfc_up, abs_dir, abs_dif):
  """Compared quantities, all per launched photon per column (ours' *_frac
  convention); abs_* per ground area per cell."""
  ncol = NX * NY
  q = {
      "tod_up_mean": np.mean(tod_up), "sfc_dir_mean": np.mean(sfc_dir),
      "sfc_dif_mean": np.mean(sfc_dif), "sfc_up_mean": np.mean(sfc_up),
      "abs_dir_total": np.sum(abs_dir) / ncol,
      "abs_dif_total": np.sum(abs_dif) / ncol,
      "abs_dir_profile": np.mean(abs_dir, axis=(0, 1)),
      "abs_dif_profile": np.mean(abs_dif, axis=(0, 1)),
      "sfc_dir_map": sfc_dir, "sfc_dif_map": sfc_dif, "tod_up_map": tod_up,
  }
  return {k: np.asarray(v, np.float64) for k, v in q.items()}


def collect(d, oracle_dir, out):
  ncol = NX * NY
  stats = {}
  self_check = {}
  for case in CASES:
    per = []
    tod_dn_means = []
    for s in range(N_SEEDS):
      v = np.fromfile(f"{d}/walk_out_{case}_{s}.bin", np.float64)
      assert v.size == 5 * ncol + 2 * NCELL, v.size
      tod_dn, tod_up, sfc_dir, sfc_dif, sfc_up = (
          from_microhh(v[i * ncol:(i + 1) * ncol]) for i in range(5))
      abs_dir = from_microhh(v[5 * ncol:5 * ncol + NCELL], NZ) * DZ
      abs_dif = from_microhh(v[5 * ncol + NCELL:], NZ) * DZ
      per.append(quantities(tod_up, sfc_dir, sfc_dif, sfc_up, abs_dir, abs_dif))
      tod_dn_means.append(float(np.mean(tod_dn)))
      # Known-value self-check: the oracle's own budget closes in expectation.
      budget = (np.mean(sfc_dir + sfc_dif - sfc_up) + np.mean(tod_up)
                + np.sum(abs_dir + abs_dif) / ncol)
      self_check.setdefault(case, []).append(float(budget / np.mean(tod_dn)))
    for k in per[0]:
      arr = np.stack([p[k] for p in per])
      stats[f"{case}/{k}/mean"] = arr.mean(axis=0)
      stats[f"{case}/{k}/std"] = arr.std(axis=0, ddof=1)
    stats[f"{case}/tod_dn_mean"] = np.asarray(tod_dn_means)
  rev = subprocess.run(["git", "-C", oracle_dir, "rev-parse", "HEAD"],
                       capture_output=True, text=True).stdout.strip()
  meta = dict(oracle="microhh rte-rrtmgp-cpp v0.3", oracle_rev=rev,
              cases=list(CASES), n_seeds=N_SEEDS, nx=NX, ny=NY, nz=NZ, dx=DX,
              dz=DZ, kn=list(KN), ppp=PPP, sza=float(SZA),
              azimuth_microhh=AZI_MICROHH, albedo=ALBEDO, mie_band=MIE_BAND,
              budget_over_tod_dn=self_check)
  np.savez(out, meta=json.dumps(meta), **stats)
  for case in CASES:
    print(case, "budget/tod_dn min/max", min(self_check[case]),
          max(self_check[case]), "tod_dn mean", np.mean(stats[f"{case}/tod_dn_mean"]))
  print("FIXTURE_OK", out)


if __name__ == "__main__":
  if sys.argv[1] == "inputs":
    write_inputs(sys.argv[2])
  elif sys.argv[1] == "collect":
    collect(sys.argv[2], sys.argv[3], sys.argv[4])
  else:
    raise SystemExit("mode must be inputs or collect")
