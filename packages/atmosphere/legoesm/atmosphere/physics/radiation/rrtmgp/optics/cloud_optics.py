# Copyright 2024 The swirl_jatmos Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for computing the optical properties of clouds."""

import collections
from typing import Callable, TypeAlias

import jax
import jax.numpy as jnp

def _tiny(x=None):
  """Smallest normal value for the working dtype (float32 or float64)."""
  dtype = x.dtype if x is not None else jnp.float_
  return float(jnp.finfo(dtype).tiny)

from legoesm import constants
from legoesm.atmosphere.physics._shared import safe_divide
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_cloud_optics
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import optics_utils

Array: TypeAlias = jax.Array
Interpolant: TypeAlias = optics_utils.Interpolant
IndexAndWeight: TypeAlias = optics_utils.IndexAndWeight

_EPSILON = 1e-6
_KG_TO_G_FACTOR = 1e3
_M_TO_MICRONS_FACTOR = 1e6

# --- CAM6 ice-optics particle size (CAM cam_cesm2_1_rel_60, micro_mg2_0.F90:2795;
# micro_mg_utils.F90:126,128) ---
# CAM passes the ice optics ``deffi = effi*rhoi/rhows*2``: the PSD effective
# radius of bulk-density ice (rhoi = 500 kg/m^3, the density the M2005/MG2 ice
# PSD assumes) converted to the effective DIAMETER of solid ice
# (rhows = 917 kg/m^3 = ``constants.rho_ice``), which is what the RRTMGP ice
# table is indexed by (rte-rrtmgp v1.9 relabels that axis as diameter).
_RHO_ICE_BULK_CAM_MG2 = 500.0  # [kg/m^3] CAM micro_mg_utils.F90 ``rhoi``


def _particle_size_interpolant(
    f: Array, lower_bnd: float, upper_bnd: float, n_size: int
) -> dict[str, Callable[..., Interpolant]]:
  """Creates effective radius interpolant based on desired number of points."""
  # Reference grid in ``f``'s working dtype.  A bare ``jnp.linspace`` defaults
  # to float64 under ``jax_enable_x64`` and, via the interpolation weights in
  # ``create_linear_interpolant``, would re-promote the cloud optical
  # properties to float64 on the ``compute_fp32`` path.  ``f.dtype`` is float64
  # on the default path -> byte-identical there.
  interp = optics_utils.create_linear_interpolant(
      f, jnp.linspace(lower_bnd, upper_bnd, n_size, dtype=f.dtype)
  )
  return collections.OrderedDict({'r': lambda: interp})


def compute_optical_properties(
    lookup: lookup_cloud_optics.LookupCloudOptics,
    cloud_path_liq: Array,
    cloud_path_ice: Array,
    radius_eff_liq: Array,
    radius_eff_ice: Array,
    ibnd: Array,
    absorption_only: bool = False,
) -> dict[str, Array]:
  """Computes the optical properties of clouds from lookup tables.

  The optical depth, single-scattering albedo, and asymmetry factor are computed
  taking into account both the liquid and ice phase. The final optical depth
  is a simple sum of the optical depths of both phases. The final ssa is a
  a weighted sum (weighted by optical depth) of the ssa of both phases. The
  final asymmetry factor is also a weighted sum (weighted by ssa) of the
  asymmetry factors of both phases.

  Args:
    lookup: A `LookupCloudOptics` instance containing lookup tables for the
      extinction coefficient, the single-scattering albedo, and the asymmetry
      factor of condensates indexed by spectral band and effective radius of
      cloud particles.
    cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
      [kg/m²].
    cloud_path_ice: The cloud ice water path in each atmospheric grid cell
      [kg/m²].
    radius_eff_liq: The effective radius of cloud droplets in each atmospheric
      grid cell [m].
    radius_eff_ice: The effective radius of cloud ice particles in each
      atmospheric grid cell [m], bulk-density (500 kg/m^3) PSD convention; the
      table coordinate is CAM6's solid-ice diameter ``2 r 500/917``.
    ibnd: The spectral band index.
    absorption_only: If True, return the absorption-only (no-scattering)
      properties RTE-RRTMGP computes for a ``ty_optical_props_1scl`` target
      (mo_cloud_optics_rrtmgp.F90:527-541): optical depth
      ``(tau_liq - tau_ssa_liq) + (tau_ice - tau_ssa_ice)``, ssa = g = 0.  This
      is the longwave cloud treatment of the RRTMGP reference configuration
      (examples/all-sky/rrtmgp_allsky.F90) and of CAM6 RRTMG LW.

  Returns:
  A dictionary containing:
    'optical_depth': The combined optical depth contribution from condensates.
    'ssa': The combined single-scattering albedo contribution from condensates.
    'g': The combined asymmetry factor contribution from condensates.
  """
  # Convert effective radius from meter to microns to conform to the lookup
  # tables. Default to the lower bound to prevent an out-of-range error when
  # interpolating. These default values will later be eliminated by the cloud
  # mask constructed below.
  # Using the name `size` because the lookup tables for RRTMGP use radius for
  # cloud liquid and diameter for cloud ice.
  particle_size = [
      jnp.clip(
          _M_TO_MICRONS_FACTOR * radius_eff_liq,
          lookup.radius_liq_lower,
          lookup.radius_liq_upper,
      ),
      jnp.clip(
          _M_TO_MICRONS_FACTOR * radius_eff_ice
          * _RHO_ICE_BULK_CAM_MG2 / constants.rho_ice * 2.0,
          lookup.diameter_ice_lower,
          lookup.diameter_ice_upper,
      ),
  ]

  # Create interpolant for cloud droplet effective radius.
  size_lower = [lookup.radius_liq_lower, lookup.diameter_ice_lower]
  size_upper = [lookup.radius_liq_upper, lookup.diameter_ice_upper]
  n_size = [lookup.n_size_liq, lookup.n_size_ice]

  interpolants = jax.tree.map(
      _particle_size_interpolant, particle_size, size_lower, size_upper, n_size
  )

  roughness = lookup.ice_roughness.value
  # optimization_barrier: keep XLA from CONSTANT-FOLDING the per-band slices of
  # the (constant) cloud-optics lookup tables into the kernel — that fold is
  # what tips the rrtmgp executable over the XLA-CPU LLVM-JIT code-region limit
  # when clouds are active (clear-sky rrtmgp compiles; the cloud path overflows)
  # and bloats GPU/TPU compile.  Identity at runtime -> answer-preserving.
  _ob = jax.lax.optimization_barrier
  ext_tables = (_ob(lookup.ext_liq)[ibnd, :], _ob(lookup.ext_ice)[roughness, ibnd, :])
  ssa_tables = (_ob(lookup.ssa_liq)[ibnd, :], _ob(lookup.ssa_ice)[roughness, ibnd, :])
  asy_tables = (_ob(lookup.asy_liq)[ibnd, :], _ob(lookup.asy_ice)[roughness, ibnd, :])
  # Convert cloud path to g/m² to conform to the lookup tables.
  cloud_path = (
      cloud_path_liq * _KG_TO_G_FACTOR,
      cloud_path_ice * _KG_TO_G_FACTOR,
  )

  # Boolean threshold cast directly to the field dtype — avoids
  # materialising two ``ones_like`` / ``zeros_like`` buffers per band
  # per radiation step (`nbnd × 2 × ncol × nlev` extra writes).
  cld_mask_liq = (
      cloud_path_liq * _KG_TO_G_FACTOR >= _EPSILON
  ).astype(cloud_path_liq.dtype)
  cld_mask_ice = (
      cloud_path_ice * _KG_TO_G_FACTOR >= _EPSILON
  ).astype(cloud_path_ice.dtype)
  cld_mask = (cld_mask_liq, cld_mask_ice)

  optical_props = []
  for interp, ext, ssa, asy, cld_path, mask in zip(
      interpolants, ext_tables, ssa_tables, asy_tables, cloud_path, cld_mask
  ):
    props = {}
    props['tau'] = optics_utils.interpolate(ext, interp) * cld_path * mask
    # Weight the ssa by the optical depth.
    props['tau_ssa'] = optics_utils.interpolate(ssa, interp) * props['tau']
    # Weight the asymmetry factor by the single-scattering albedo.
    props['tau_ssa_g'] = (
        optics_utils.interpolate(asy, interp) * props['tau_ssa']
    )
    optical_props.append(props)

  if absorption_only:
    liq, ice = optical_props
    tau_abs = (liq['tau'] - liq['tau_ssa']) + (ice['tau'] - ice['tau_ssa'])
    zero = jnp.zeros_like(tau_abs)
    return {'optical_depth': tau_abs, 'ssa': zero, 'asymmetry_factor': zero}

  combined_props = jax.tree.map(jnp.add, *optical_props)
  # AD-safe denominators (restored from commit 59407953 after AIMIP-#312
  # merge reverted it).  The previous ``safe_tau = jnp.maximum(tau, tiny)``
  # pattern is forward-safe but NOT backward-safe under fp64: the local
  # VJP ``-num/denom**2`` evaluates ``denom**2 = tiny**2 = 4.8e-616`` which
  # underflows to 0, so ``-num/0 = NaN`` propagates via JAX's where-
  # backward (both branches are differentiated; the inactive branch's NaN
  # cotangent multiplies a zero mask to give 0*inf = NaN).  ``safe_divide``
  # uses ``where(mask, denom, 1.0)`` so the inactive denom is exactly 1.0
  # (no underflow on square), keeping backward finite.  Pinpointed via
  # ``jax.checkify(nan_checks)`` on AMIP+RRTMG reverse-mode AD.
  return {
      'optical_depth': combined_props['tau'],
      'ssa': safe_divide(
          combined_props['tau_ssa'], combined_props['tau'],
          eps=_EPSILON, fill=0.0,
      ),
      'asymmetry_factor': safe_divide(
          combined_props['tau_ssa_g'], combined_props['tau_ssa'],
          eps=_EPSILON, fill=0.0,
      ),
  }
