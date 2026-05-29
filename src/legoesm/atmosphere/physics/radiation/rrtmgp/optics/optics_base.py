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

"""Abstract base class defining the interface of an optics scheme."""

import abc
from collections.abc import Mapping
from typing import TypeAlias

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics._shared import safe_divide
from legoesm.atmosphere.physics.radiation.rrtmgp import interpolation
from legoesm.atmosphere.physics.radiation.rrtmgp import kernel_ops

Array: TypeAlias = jax.Array


def _shift_up(f: Array) -> Array:
  """output_i = f_{i-1}."""
  return kernel_ops.shift_from_minus(f, 2)


def _shift_down(f: Array) -> Array:
  """output_i = f_{i+1}."""
  return kernel_ops.shift_from_plus(f, 2)


def reconstruct_face_values(f: Array, f_lower_bc: Array) -> tuple[Array, Array]:
  """Reconstruct the face values using a high-order scheme.

  This function performs an interpolation from nodes to faces.

  Args:
    f: The cell-center values that will be interpolated.
    f_lower_bc: The boundary condition for f on the lower face (wall).

  Returns:
    A tuple with the reconstructed temperature at the bottom and top face,
    respectively.
  """
  # f is f_ccc.
  # Use WENO5.

  # Set halo value to the BC wall value.
  f = f.at[:, :, 0].set(f_lower_bc)

  # Enforce Neumann BC on the upper halo value.
  f = f.at[:, :, -1].set(f[:, :, -2])

  f_face_plus, f_face_minus = interpolation.weno5_node_to_face_for_rrtmgp(
      f, dim=2, f_lower_bc=f_lower_bc, neumann_upper_bc=True
  )
  # To get the final interpolation, just take the average of the plus and
  # minus reconstructions.
  f_bottom_ccf = 0.5 * (f_face_plus + f_face_minus)

  # Use centered interpolation: UNSTABLE.
  # f_bottom_ccf = interpolation.z_c_to_f(f)

  f_top = _shift_down(f_bottom_ccf)
  # Update the f_top face (wall) value.
  # Look into what is the appropriate BC here.
  f_top = f_top.at[:, :, -1].set(f_top[:, :, -2])
  return f_bottom_ccf, f_top


class OpticsScheme(abc.ABC):
  """Abstract base class for optics scheme."""

  _EPSILON = 1e-6

  def __init__(
      self,
  ):
    self._halo_width = 1  # halos in z
    # self._face_interp_scheme_order = params.face_interp_scheme_order

    self.cloud_optics_lw = None
    self.cloud_optics_sw = None
    self.gas_optics_lw = None
    self.gas_optics_sw = None

  @abc.abstractmethod
  def compute_lw_optical_properties(
      self,
      pressure: Array,
      temperature: Array,
      molecules: Array,
      igpt: Array,
      vmr_fields: dict[int, Array] | None = None,
      cloud_r_eff_liq: Array | None = None,
      cloud_path_liq: Array | None = None,
      cloud_r_eff_ice: Array | None = None,
      cloud_path_ice: Array | None = None,
  ) -> dict[str, Array]:
    """Computes the monochromatic longwave optical properties.

    Args:
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules/m²].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m²].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m²].

    Returns:
      A dictionary containing (for a single g-point):
        'optical_depth': The longwave optical depth.
        'ssa': The longwave single-scattering albedo.
        'asymmetry_factor': The longwave asymmetry factor.
    """

  @abc.abstractmethod
  def compute_sw_optical_properties(
      self,
      pressure: Array,
      temperature: Array,
      molecules: Array,
      igpt: Array,
      vmr_fields: dict[int, Array] | None = None,
      cloud_r_eff_liq: Array | None = None,
      cloud_path_liq: Array | None = None,
      cloud_r_eff_ice: Array | None = None,
      cloud_path_ice: Array | None = None,
  ) -> dict[str, Array]:
    """Computes the monochromatic shortwave optical properties.

    Args:
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules/m²].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m²].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m²].

    Returns:
      A dictionary containing (for a single g-point):
        'optical_depth': The shortwave optical depth.
        'ssa': The shortwave single-scattering albedo.
        'asymmetry_factor': The shortwave asymmetry factor.
    """

  @abc.abstractmethod
  def compute_planck_sources(
      self,
      pressure: Array,
      temperature: Array,
      igpt: Array,
      vmr_fields: dict[int, Array] | None = None,
      sfc_temperature: Array | None = None,
  ) -> dict[str, Array]:
    """Computes the Planck sources used in the longwave problem.

    Args:
      pressure: The pressure field [Pa].
      temperature: The temperature [K].
      igpt: The spectral interval index, or g-point.
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index, that will overwrite the global means.
      sfc_temperature: An optional 2D plane for the surface temperature [K].

    Returns:
      A dictionary containing the Planck source at the cell center
      (`planck_src`), the top cell boundary (`planck_src_top`), and the bottom
      cell boundary (`planck_src_bottom`).
    """

  @property
  @abc.abstractmethod
  def n_gpt_lw(self) -> int:
    """The number of g-points in the longwave bands."""

  @property
  @abc.abstractmethod
  def n_gpt_sw(self) -> int:
    """The number of g-points in the shortwave bands."""

  @property
  @abc.abstractmethod
  def solar_fraction_by_gpt(self) -> Array:
    """Mapping from g-point to the fraction of total solar radiation."""

  def combine_optical_properties(
      self,
      optical_props_1: Mapping[str, Array],
      optical_props_2: Mapping[str, Array],
  ) -> dict[str, Array]:
    """Combines the optical properties from two separate parameterizations."""
    tau1 = optical_props_1['optical_depth']
    tau2 = optical_props_2['optical_depth']
    ssa1 = optical_props_1['ssa']
    ssa2 = optical_props_2['ssa']
    g1 = optical_props_1['asymmetry_factor']
    g2 = optical_props_2['asymmetry_factor']

    # Combine optical depths
    tau = tau1 + tau2

    # Combine single-scattering albedos.
    ssa_unnormalized = tau1 * ssa1 + tau2 * ssa2

    # Combine asymmetry factors.  Use ``safe_divide`` instead of
    # ``num / jnp.maximum(denom, eps)``: the latter is forward-safe but
    # has a ``-num / denom**2`` reverse-mode VJP that overflows when
    # ``denom`` is at the floor (``eps**2 = 1e-12`` underflows to 0
    # under fp64).  Same pattern as the iter-14 restoration of commit
    # 59407953; codex iter-18 review flagged this as a survivor.  For
    # the gas-only LW path ``ssa_unnormalized`` is zero (gas scattering
    # ≈ 0 except for Rayleigh in SW), so the safe-divide ``fill=0.0``
    # branch is the routine case rather than an edge.
    g = safe_divide(
        tau1 * ssa1 * g1 + tau2 * ssa2 * g2,
        ssa_unnormalized,
        eps=self._EPSILON,
        fill=0.0,
    )

    return {
        'optical_depth': tau,
        'ssa': safe_divide(
            ssa_unnormalized, tau, eps=self._EPSILON, fill=0.0,
        ),
        'asymmetry_factor': g,
    }
