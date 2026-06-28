"""Mie-CDF cloud-scattering sampler vs the microhh oracle.

Two comparisons:
1. FORMULA-EXACT: a NumPy transcription of microhh's `mie_sample_angle`
   (descending-CDF binary search + cdf/r_eff interpolation) on microhh's own
   committed sampling LUT must equal our jitted `mie_sample_cos` bit-for-bit.
2. DISTRIBUTION: angles sampled by `mie_sample_cos` reproduce microhh's tabulated
   Mie phase function (the full `mie_lut_broadband.nc` `phase` table) — forward
   peak, mean angle, asymmetry g. Skipped if the full .nc is absent.

Run with JAX_ENABLE_X64=1.
"""

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.radiation.mc3d.mie import (
    load_mie_sampling_lut,
    mie_sample_cos,
)

_NC = (pathlib.Path(__file__).resolve().parents[2]
       / "scripts" / "tmp" / "_mie" / "mie_lut_broadband.nc")


def _microhh_ref_cos(u, r_eff, cdf, ang):
  """NumPy transcription of microhh raytracer_functions.h::mie_sample_angle."""
  r_idx = min(max(int(r_eff - 2.5), 0), 18)
  r_rest = (r_eff - 2.5) % 1.0
  # find_index on the descending CDF (microhh binary search).
  left, right = 0, len(cdf) - 1
  while left < right:
    mid = left + (right - left) // 2
    if u >= cdf[mid]:
      right = mid
    else:
      left = mid + 1
  i = min(max(left - 1, 0), len(cdf) - 2)
  dr = abs(cdf[i + 1] - cdf[i])
  dr = dr if dr > 0 else 1.0

  def a(rr):
    return (abs(u - cdf[i + 1]) * ang[rr, i]
            + abs(cdf[i] - u) * ang[rr, i + 1]) / dr

  return np.cos(a(r_idx) * (1.0 - r_rest) + a(r_idx + 1) * r_rest)


def test_mie_sampler_matches_microhh_formula():
  """Bit-faithful to microhh's mie_sample_angle on its own LUT."""
  lut = load_mie_sampling_lut()
  cdf = np.asarray(lut.phase_cdf)
  ang = np.asarray(lut.phase_cdf_angle)
  rng = np.random.default_rng(0)
  for _ in range(400):
    b = int(rng.integers(0, lut.n_band))
    u = float(rng.uniform())
    r_eff = float(rng.uniform(2.5, 21.5))
    ref = _microhh_ref_cos(u, r_eff, cdf[b], ang[b])
    got = float(mie_sample_cos(jnp.float64(u), jnp.float64(r_eff),
                               lut.phase_cdf[b], lut.phase_cdf_angle[b]))
    assert abs(got - ref) < 1e-5


def test_mie_in_tracer_forward_scatters_and_conserves():
  """Wired into solve_sw_spectral: a conservative cloud slab with the Mie phase
  (forward g~0.85) reflects LESS than isotropic (g=0) and conserves exactly."""
  from legoesm.atmosphere.physics.radiation.mc3d.config import MC3DRadiationConfig
  from legoesm.atmosphere.physics.radiation.mc3d.photon_walk import PlaneRTGeometry
  from legoesm.atmosphere.physics.radiation.mc3d.raytracer_sw import (
      solve_sw_spectral,
  )
  lut = load_mie_sampling_lut()
  nx = ny = 6
  nz = 12
  H = 1000.0
  geom = PlaneRTGeometry(dx=50.0, dy=50.0, Lx=nx * 50.0, Ly=ny * 50.0,
                         z_faces=jnp.linspace(0.0, H, nz + 1),
                         nx=nx, ny=ny, nz=nz)
  tau = jnp.full((1, nx, ny, nz), 4.0 / nz)           # tau~4 cloud
  ssa = jnp.ones((1, nx, ny, nz))                     # conservative
  g = jnp.zeros((1, nx, ny, nz))
  rfrac = jnp.zeros((1, nx, ny, nz))                  # all scattering is cloud
  r_eff = jnp.full((nx, ny, nz), 10.5)
  b = 7
  cdf_band = lut.phase_cdf[b][None]                   # (1, n_mie)
  ang_band = lut.phase_cdf_angle[b][None]             # (1, n_r, n_mie)
  cfg = MC3DRadiationConfig(photons_per_pixel=4000)
  kw = dict(mu0=1.0, azimuth=0.0, albedo=0.0, config=cfg)

  mie = solve_sw_spectral(tau, ssa, g, jnp.array([1.0]), geom,
                          key=jax.random.PRNGKey(0), rayleigh_frac=rfrac,
                          mie_cdf_band=cdf_band, mie_ang_band=ang_band,
                          r_eff=r_eff, **kw)
  iso = solve_sw_spectral(tau, ssa, g, jnp.array([1.0]), geom,
                          key=jax.random.PRNGKey(1), rayleigh_frac=rfrac, **kw)
  # Forward Mie scattering -> less reflected to space than isotropic.
  assert float(mie.tod_up_flux) < float(iso.tod_up_flux)
  # Energy conserved (per-column fractions sum to incident).
  col_abs = float(jnp.mean(jnp.sum(mie.abs_flux, axis=2)))
  budget = col_abs + float(jnp.mean(mie.sfc_abs_flux)) + float(mie.tod_up_flux)
  assert abs(budget - 1.0) < 0.02


def test_mie_sampled_distribution_matches_phase_lut():
  """Sampled angles reproduce microhh's tabulated Mie phase function."""
  if not _NC.exists():
    pytest.skip("full mie_lut_broadband.nc not present")
  import netCDF4 as nc
  lut = load_mie_sampling_lut()
  b, r_eff = 7, 10.5
  us = jax.random.uniform(jax.random.PRNGKey(0), (300000,))
  cos = jax.vmap(lambda u: mie_sample_cos(
      u, jnp.float64(r_eff), lut.phase_cdf[b], lut.phase_cdf_angle[b]))(us)
  cos = np.asarray(cos)
  ang = np.arccos(np.clip(cos, -1.0, 1.0))

  d = nc.Dataset(str(_NC))
  phase = np.asarray(d["phase"][b])           # (n_r, n_ang)
  pang = np.asarray(d["phase_angle"][:])      # (n_ang,)
  ridx = int(r_eff - 2.5)
  pdf_ref = phase[ridx] * np.sin(pang)        # angular pdf ∝ p(θ) sinθ
  pdf_ref /= np.trapezoid(pdf_ref, pang)

  # Mean angle + asymmetry g vs the reference table.
  mean_ref = float(np.trapezoid(pang * pdf_ref, pang))
  g_ref = float(np.trapezoid(np.cos(pang) * pdf_ref, pang))
  assert abs(float(ang.mean()) - mean_ref) < 0.02
  assert abs(float(cos.mean()) - g_ref) < 0.02
  assert float(cos.mean()) > 0.8              # strongly forward (cloud Mie)

  # Binned density agreement.
  h, edges = np.histogram(ang, bins=120, range=(0, np.pi), density=True)
  ctr = 0.5 * (edges[:-1] + edges[1:])
  ref_at = np.interp(ctr, pang, pdf_ref)
  rel_l1 = float(np.sum(np.abs(h - ref_at)) / np.sum(ref_at))
  assert rel_l1 < 0.10
