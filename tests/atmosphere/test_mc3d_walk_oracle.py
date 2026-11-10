"""mc3d shortwave walk vs the running microhh ray tracer (statistical).

Fixture: tests/atmosphere/oracle_fixtures/mc3d_walk.npz, written by
scripts/validate/oracle/mc3d_walk/make_fixture.py from N_SEEDS independent
runs of microhh rte-rrtmgp-cpp v0.3 ``Raytracer::trace_rays`` (fp64, CUDA).
The test runs legoESM seeds on the same optical fields (8 by default, the
full 20 in the slow variant) and requires every compared quantity to pass a
Welch t-test at a Bonferroni-corrected three-sigma familywise level
(two-sided 0.0027 per case).
Run with JAX_ENABLE_X64=1.
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from legoesm.atmosphere.physics.radiation.mc3d import (
    MC3DRadiationConfig,
    PlaneRTGeometry,
    solve_sw_monochromatic,
)

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "oracle_fixtures", "mc3d_walk.npz")
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "validate",
                                "oracle", "mc3d_walk"))
import make_fixture as mf  # noqa: E402

FAMILY_ALPHA = 0.0027  # two-sided three sigma


def _ours(case, seed, d):
  tau_tot, ssa_tot, tau_cld, ssa_cld, asy_cld, r_eff, dif = mf.fields(case)
  k_sca_tot = tau_tot * ssa_tot
  k_sca_cld = tau_cld * ssa_cld
  rayleigh = np.where(k_sca_tot > 0, (k_sca_tot - k_sca_cld) / k_sca_tot, 1.0)
  g = np.where(k_sca_tot > 0, k_sca_cld * asy_cld / k_sca_tot, 0.0)
  geom = PlaneRTGeometry(dx=mf.DX, dy=mf.DY, Lx=mf.NX * mf.DX,
                         Ly=mf.NY * mf.DY,
                         z_faces=jnp.linspace(0.0, mf.NZ * mf.DZ, mf.NZ + 1),
                         nx=mf.NX, ny=mf.NY, nz=mf.NZ)
  cfg = MC3DRadiationConfig(photons_per_pixel=mf.PPP,
                            knull_coarsen_xy=mf.NX // mf.KN[0],
                            knull_coarsen_z=mf.NZ // mf.KN[2])
  mie = {}
  if case == "broken":
    cdf, ang = mf.mie_table()
    mie = dict(mie_cdf=jnp.asarray(cdf), mie_ang=jnp.asarray(ang),
               r_eff=jnp.asarray(r_eff))
  # microhh azimuth (0 = north, clockwise; beam heading -sun) -> ours.
  res = solve_sw_monochromatic(
      jnp.asarray(tau_tot / mf.DZ), jnp.asarray(ssa_tot), jnp.asarray(g), geom,
      mu0=float(np.cos(mf.SZA)), azimuth=float(-0.5 * np.pi - mf.AZI_MICROHH),
      albedo=mf.ALBEDO, diffuse_frac=dif, config=cfg,
      key=jax.random.PRNGKey(10_000 + seed),
      rayleigh_frac=jnp.asarray(rayleigh), **mie)
  assert float(res.residual_frac) == 0.0
  return mf.quantities(
      np.asarray(res.tod_up_col_frac), np.asarray(res.sfc_dir_frac),
      np.asarray(res.sfc_dif_frac), np.asarray(res.sfc_up_frac),
      np.asarray(res.abs_dir_frac), np.asarray(res.abs_dif_frac))


@pytest.fixture(scope="module")
def oracle():
  z = np.load(FIX)
  return json.loads(str(z["meta"])), z


def _pvalues(case, oracle, n1):
  meta, z = oracle
  n = meta["n_seeds"]
  per = [_ours(case, s, None) for s in range(n1)]
  out = {}
  for k in per[0]:
    a = np.stack([p[k] for p in per])
    m1, s1 = a.mean(axis=0), a.std(axis=0, ddof=1)
    m2, s2 = z[f"{case}/{k}/mean"], z[f"{case}/{k}/std"]
    se = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n)
    with np.errstate(invalid="ignore", divide="ignore"):
      t = (m1 - m2) / se
      dof = se ** 4 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n) ** 2 / (n - 1))
    p = np.where(se > 0, 2.0 * stats.t.sf(np.abs(t), dof), np.where(m1 == m2, 1.0, 0.0))
    out[k] = (np.ravel(p), np.ravel(m1), np.ravel(m2))
  return out


def _check(case, oracle, n1):
  out = _pvalues(case, oracle, n1)
  n_tests = sum(v[0].size for v in out.values())
  thresh = FAMILY_ALPHA / n_tests
  bad = {k: (float(v[0].min()), v[1][np.argmin(v[0])], v[2][np.argmin(v[0])])
         for k, v in out.items() if v[0].min() < thresh}
  assert not bad, f"{case}: {bad} (threshold {thresh:.2e}, {n_tests} tests)"


@pytest.mark.parametrize("case", mf.CASES)
def test_walk_matches_microhh_statistically(case, oracle):
  _check(case, oracle, 8)


@pytest.mark.slow
@pytest.mark.parametrize("case", mf.CASES)
def test_walk_matches_microhh_statistically_full(case, oracle):
  _check(case, oracle, mf.N_SEEDS)


def test_fixture_provenance(oracle):
  meta, z = oracle
  assert meta["oracle_rev"] == "fc291af01820674ff6dcd05889d7eb191b3daca4"
  assert tuple(meta["cases"]) == mf.CASES and meta["n_seeds"] == mf.N_SEEDS
  for case in mf.CASES:
    # oracle self-check: microhh launches ~one photon per column per PPP and
    # its budget closes in expectation (Russian roulette)
    assert abs(np.mean(z[f"{case}/tod_dn_mean"]) - 1.0) < 1e-2
    assert abs(np.mean(meta["budget_over_tod_dn"][case]) - 1.0) < 1e-2
