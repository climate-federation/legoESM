"""Mie cloud-scattering phase function sampled from microhh's broadband Mie LUT.

ORACLE: microhh `rte-rrtmgp-cpp` `data/mie_lut_broadband.nc` +
`raytracer_functions.h::mie_sample_angle`. This module loads microhh's exact
lookup table (compact sampling subset committed under ``data/``) and reproduces
microhh's inverse-CDF angle sampling bit-for-bit, so a cloud scatter event draws
the same Mie scattering angle as the oracle for the same (uniform deviate,
effective radius, band).

LUT layout (per shortwave band b, n_mie=876 CDF bins, 20 effective-radius bins
2.5..21.5 um):
  phase_cdf[b]            (n_mie,)        DESCENDING cumulative probability 1->0
  phase_cdf_angle[b]      (n_r, n_mie)    scattering angle [rad] at (r_eff, cdf)
microhh's ``find_index`` binary-searches the descending CDF; the vectorised
equivalent is ``(#{cdf > u}) - 1``.
"""

from __future__ import annotations

import pathlib
from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

Array: TypeAlias = jax.Array

# Effective-radius LUT geometry (microhh mie_sample_angle): r_eff in [2.5, 21.5]
# um on a 1 um grid; r_idx in [0, 18] selects the lower of two bracketing rows.
_R_EFF_MIN = 2.5
_R_IDX_MAX = 18


class MieSamplingLUT(NamedTuple):
  """microhh Mie sampling table (shortwave bands)."""

  phase_cdf: Array         # (n_band, n_mie) descending CDF axis
  phase_cdf_angle: Array   # (n_band, n_r, n_mie) angle [rad] at (r_eff, cdf)
  n_band: int
  n_mie: int


_LUT_CACHE: dict[str, MieSamplingLUT] = {}


def _default_lut_path() -> pathlib.Path:
  return pathlib.Path(__file__).with_name("data") / "mie_lut_sampling.npz"


def load_mie_sampling_lut(path: str | None = None, dtype=None) -> MieSamplingLUT:
  """Load microhh's Mie sampling LUT (cached). Heavy/static -> load outside JIT."""
  p = str(path) if path is not None else str(_default_lut_path())
  if p in _LUT_CACHE:
    return _LUT_CACHE[p]
  with np.load(p) as data:
    cdf = np.asarray(data["phase_cdf"])
    ang = np.asarray(data["phase_cdf_angle"])
  dt = dtype or (jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)
  lut = MieSamplingLUT(
      phase_cdf=jnp.asarray(cdf, dt),
      phase_cdf_angle=jnp.asarray(ang, dt),
      n_band=int(cdf.shape[0]),
      n_mie=int(cdf.shape[1]),
  )
  _LUT_CACHE[p] = lut
  return lut


def mie_sample_cos(
    u: Array, r_eff: Array, cdf_band: Array, ang_band: Array
) -> Array:
  """Sample ``cos(scattering angle)`` for a cloud Mie event (one photon).

  Faithful reproduction of microhh ``mie_sample_angle`` for a single band:

  Args:
    u: uniform deviate in [0,1) (scalar).
    r_eff: cloud effective radius [um] (scalar).
    cdf_band: ``(n_mie,)`` DESCENDING CDF axis for this band.
    ang_band: ``(n_r, n_mie)`` scattering angle [rad] at (r_eff bin, cdf bin).
  """
  n_mie = cdf_band.shape[0]
  dtype = cdf_band.dtype
  # r_eff interpolation: lower bin index + fractional remainder.
  r_idx = jnp.clip((r_eff - _R_EFF_MIN).astype(jnp.int32), 0, _R_IDX_MAX)
  r_rest = jnp.mod(r_eff - _R_EFF_MIN, 1.0).astype(dtype)
  # find_index on the DESCENDING CDF == (#{cdf > u}) - 1, clipped to [0,n_mie-2].
  i = jnp.clip(jnp.sum(cdf_band > u).astype(jnp.int32) - 1, 0, n_mie - 2)

  c_i = cdf_band[i]
  c_ip = cdf_band[i + 1]
  dr = jnp.abs(c_ip - c_i)
  dr = jnp.where(dr > 0.0, dr, 1.0)        # guard equal-CDF bins (-> a_i)
  w_ip = jnp.abs(u - c_ip)                 # weight toward angle[i]
  w_i = jnp.abs(c_i - u)                   # weight toward angle[i+1]

  def _ang_at(rr):
    a_i = ang_band[rr, i]
    a_ip = ang_band[rr, i + 1]
    return (w_ip * a_i + w_i * a_ip) / dr

  ang = _ang_at(r_idx) * (1.0 - r_rest) + _ang_at(r_idx + 1) * r_rest
  return jnp.cos(ang)
