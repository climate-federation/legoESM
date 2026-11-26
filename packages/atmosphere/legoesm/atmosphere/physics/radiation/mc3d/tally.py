"""Shortwave Monte-Carlo result type (microhh ``Raytracer::trace_rays`` outputs).

The weighted walk (``photon_walk.trace_batch_weighted``) scatter-adds its
deposits into grid tallies; ``raytracer_sw.solve_sw_monochromatic`` normalises
them into this result.
"""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import jax

Array: TypeAlias = jax.Array


class MC3DResult(NamedTuple):
  """Monochromatic MC tally, normalised by photons-per-column ``P``.

  All ``*_frac`` fields are dimensionless fractions of the launched photons:
  multiply by the incident flux (``F0 * mu0`` for SW) to get W/m^2. Per-cell and
  per-column fields are ``/P``; domain scalars are ``/N_total``.
  """

  abs_frac: Array         # (nx,ny,nz) volume-absorbed (= abs_dir + abs_dif)
  sfc_abs_frac: Array     # (nx,ny) net surface-absorbed (= dir + dif - up)
  tod_up_frac: Array      # scalar escaped through the top of the domain
  vol_abs_total: Array    # scalar domain volume-absorbed
  sfc_abs_total: Array    # scalar domain surface-absorbed (net)
  maxiter_frac: Array     # scalar fraction of photons stopped by the step cap
  abs_dir_frac: Array     # (nx,ny,nz) absorbed while still direct (microhh abs_dir)
  abs_dif_frac: Array     # (nx,ny,nz) absorbed after scattering (abs_dif)
  sfc_dir_frac: Array     # (nx,ny) direct flux reaching the surface (sfc_dir)
  sfc_dif_frac: Array     # (nx,ny) diffuse flux reaching the surface (sfc_dif)
  sfc_up_frac: Array      # (nx,ny) surface-reflected upward flux (sfc_up)
  tod_up_col_frac: Array  # (nx,ny) top-of-domain upward flux (tod_up)
  residual_frac: Array    # scalar weight still in flight at the step cap
  # Exact budget per realization:
  # vol_abs_total + sfc_abs_total + tod_up_frac + residual_frac == 1.
