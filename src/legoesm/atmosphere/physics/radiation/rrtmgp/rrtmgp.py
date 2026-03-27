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

"""Implementation of a radiative transfer solver."""

from collections.abc import Sequence
from typing import TypeAlias

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.radiation.rrtmgp import constants
from legoesm.atmosphere.physics.radiation.rrtmgp import kernel_ops
from legoesm.atmosphere.physics.radiation.rrtmgp import stretched_grid_util
from legoesm.atmosphere.physics.radiation.rrtmgp import rrtmgp_common
from legoesm.atmosphere.physics.radiation.rrtmgp.config import radiative_transfer
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import atmospheric_state
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_volume_mixing_ratio
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import optics
from legoesm.atmosphere.physics.radiation.rrtmgp.rte import two_stream

Array: TypeAlias = jax.Array


def _humidity_to_volume_mixing_ratio(
    q_t: Array, q_c: Array
) -> Array:
  """For water vapor, convert humidity to a volume mixing ratio."""
  mol_ratio = constants.R_V / constants.R_D
  q_v = q_t - q_c
  mix_ratio = q_v / (1 - q_t)
  return mol_ratio * mix_ratio


def _air_molecules_per_area(p_xxc: Array, vmr_h2o_xxc: Array) -> Array:
  """Compute the number of molecules in a grid cell per area."""
  dp_xxc = kernel_ops.centered_difference(p_xxc, dim=2)
  mol_m_air_xxc = (
      constants.DRY_AIR_MOL_MASS + constants.WATER_MOL_MASS * vmr_h2o_xxc
  )
  return -(dp_xxc / constants.G) * constants.AVOGADRO / mol_m_air_xxc


def _compute_cloud_path(
    rho: Array, q_c: Array, dz: float, sg_map: dict[str, Array]
) -> Array:
  """Compute the cloud water/ice path in each atmospheric grid cell."""
  use_stretched_grid_z = stretched_grid_util.get_use_stretched_grid(sg_map)[2]
  if use_stretched_grid_z:
    h = sg_map[stretched_grid_util.hc_key(2)]
  else:
    h = dz
  return rho * q_c * h


def _horiz_mean(f: Array) -> Array:
  """Compute the horizontal mean of `f`.

  Accumulate means in float64 to avoid potential numerical issues when
  accumulating, but then convert back to float32.

  Args:
    f: The array to compute the horizontal mean of.

  Returns:
    The horizontal mean of `f`.
  """
  return jnp.mean(f, axis=(0, 1)).astype(jnp.float32)


class RRTMGP:
  """Rapid Radiative Transfer Model for General Circulation Models (RRTMGP)."""

  def __init__(
      self,
      radiative_transfer_cfg: radiative_transfer.RadiativeTransfer,
      dz: float,
      diagnostic_fields: Sequence[str] = tuple(),
  ):
    self._dz = dz  # Store dz (only used if not using stretched grid in z).
    self._diagnostic_fields = diagnostic_fields
    self._save_lw_sw_heating_rates = (
        radiative_transfer_cfg.save_lw_sw_heating_rates
    )
    self._do_clear_sky = radiative_transfer_cfg.do_clear_sky

    # Load and store the atmospheric gas concentrations.
    self.atmospheric_state = atmospheric_state.from_config(
        radiative_transfer_cfg.atmospheric_state_cfg
    )
    # Create the optics library.
    self.optics_lib = optics.optics_factory(
        radiative_transfer_cfg.optics, self.atmospheric_state.vmr
    )

  def compute_heating_rate(
      self,
      rho_xxc: Array,
      q_t: Array,
      q_liq: Array,
      q_ice: Array,
      q_c: Array,
      cloud_r_eff_liq: Array,
      cloud_r_eff_ice: Array,
      temperature: Array,
      sfc_temperature: Array,
      p_ref_xxc: Array,
      sg_map: dict[str, Array],
      use_scan: bool = False,
  ) -> dict[str, Array]:
    """Compute the local heating rate due to radiative transfer.

    The optical properties of the layered atmosphere are computed using RRTMGP
    and the two-stream radiative transfer equation is solved for the net fluxes
    at the grid cell faces.  Based on the overall net radiative flux of the grid
    cell, a local heating rate is determined.

    Returns:
      A dictionary containing the following keys:
        rrtmgp_common.KEY_STORED_RADIATION: The heating rate [K/s].
      Optional keys depending on config:
        'rad_heat_sw_3d': The net shortwave radiative heating rate [K/s].
        'rad_heat_lw_3d': The net longwave radiative heating rate [K/s].
        'sw_flux_up_full': Full shortwave upward flux profile [W/m²]
        'sw_flux_down_full': Full shortwave downward flux profile [W/m²]
        'lw_flux_up_full': Full longwave upward flux profile [W/m²]
        'lw_flux_down_full': Full longwave downward flux profile [W/m²]
    """
    # Temperature may have NaNs in the halos (this is intentional).  These NaNs
    # cause problems later on, so fill in the halo values with linear
    # extrapolations.  Note that NaNs in other fields do not cause issues
    # because halos are discarded later on, but there are places where the halos
    # of the temperature are used to determine interior values.
    def fill_halo(f: Array) -> Array:
      bottom_halo_val = 2 * f[:, :, 1] - f[:, :, 2]
      top_halo_val = 2 * f[:, :, -2] - f[:, :, -3]
      f = f.at[:, :, 0].set(bottom_halo_val)
      f = f.at[:, :, -1].set(top_halo_val)
      return f

    temperature = fill_halo(temperature)

    # Sometimes, the simulation can result in q_t or q_c with negative values.
    # Clip these to zero, otherwise there can be extremely deleterious effects
    # in the radiation solver.  E.g., negative relative abundance of gas species
    # and negative values in optical depth and Planck fraction that are
    # inherently nonnegative.
    q_t = jnp.clip(q_t, 0.0, None)
    q_liq = jnp.clip(q_liq, 0.0, None)
    q_ice = jnp.clip(q_ice, 0.0, None)
    q_c = jnp.clip(q_c, 0.0, None)

    # Reconstruct the volume mixing ratio (vmr) of relevant gas species.
    vmr_lib = self.atmospheric_state.vmr
    vmr_fields = (
        lookup_volume_mixing_ratio.reconstruct_vmr_fields_from_pressure(
            vmr_lib, p_ref_xxc
        )
    )
    # Derive the water vapor vmr from the simulation state itself.
    vmr_fields['h2o'] = _humidity_to_volume_mixing_ratio(q_t, q_c)

    # Compute molecules
    molecules_per_area = _air_molecules_per_area(p_ref_xxc, vmr_fields['h2o'])

    # Compute water paths for liquid and ice cloud condensate.
    liq_water_path = _compute_cloud_path(rho_xxc, q_liq, self._dz, sg_map)
    ice_water_path = _compute_cloud_path(rho_xxc, q_ice, self._dz, sg_map)

    lw_fluxes = two_stream.solve_lw(
        p_ref_xxc,
        temperature,
        molecules_per_area,
        self.optics_lib,
        self.atmospheric_state,
        vmr_fields,
        sfc_temperature,
        cloud_r_eff_liq=cloud_r_eff_liq,
        cloud_path_liq=liq_water_path,
        cloud_r_eff_ice=cloud_r_eff_ice,
        cloud_path_ice=ice_water_path,
        use_scan=use_scan,
    )
    sw_fluxes = two_stream.solve_sw(
        p_ref_xxc,
        temperature,
        molecules_per_area,
        self.optics_lib,
        self.atmospheric_state,
        vmr_fields,
        cloud_r_eff_liq=cloud_r_eff_liq,
        cloud_path_liq=liq_water_path,
        cloud_r_eff_ice=cloud_r_eff_ice,
        cloud_path_ice=ice_water_path,
        use_scan=use_scan,
    )

    # Compute the heating rate in K/s.
    lw_heating_rate = two_stream.compute_heating_rate(
        lw_fluxes['flux_net'], p_ref_xxc
    )
    sw_heating_rate = two_stream.compute_heating_rate(
        sw_fluxes['flux_net'], p_ref_xxc
    )
    # Compute the total heating rate (temperature tendency due to radiation).
    heating_rate = lw_heating_rate + sw_heating_rate

    output = {rrtmgp_common.KEY_STORED_RADIATION: heating_rate}

    # add LW, SW heating rate or fluxes if desired.
    if self._save_lw_sw_heating_rates:
      output['rad_heat_sw_3d'] = sw_heating_rate
      output['rad_heat_lw_3d'] = lw_heating_rate

    # Compute diagnostics, if desired.
    lw_flux_down = lw_fluxes['flux_down']
    lw_flux_up = lw_fluxes['flux_up']
    sw_flux_down = sw_fluxes['flux_down']
    sw_flux_up = sw_fluxes['flux_up']
    hw = 1  # halo width.

    # Add full flux profiles (remove surface halos)
    output['sw_flux_up_full'] = sw_flux_up[:, :, hw:]  # Full profile (..., nlev+1)
    output['sw_flux_down_full'] = sw_flux_down[:, :, hw:]  # Full profile (..., nlev+1)
    output['lw_flux_up_full'] = lw_flux_up[:, :, hw:]  # Full profile (..., nlev+1)
    output['lw_flux_down_full'] = lw_flux_down[:, :, hw:]  # Full profile (..., nlev+1)

    # 2D diagnostics
    if (v := 'surf_lw_flux_down_2d_xy') in self._diagnostic_fields:
      output[v] = lw_flux_down[:, :, hw]
    if (v := 'surf_lw_flux_up_2d_xy') in self._diagnostic_fields:
      output[v] = lw_flux_up[:, :, hw]
    if (v := 'surf_sw_flux_down_2d_xy') in self._diagnostic_fields:
      output[v] = sw_flux_down[:, :, hw]
    if (v:= 'surf_sw_flux_up_2d_xy') in self._diagnostic_fields:
      output[v] = sw_flux_up[:, :, hw]
    # Add clear sky surf

    if (v := 'toa_sw_flux_incoming_2d_xy') in self._diagnostic_fields:
      output[v] = sw_flux_down[:, :, -hw]
    if (v := 'toa_sw_flux_outgoing_2d_xy') in self._diagnostic_fields:
      output[v] = sw_flux_up[:, :, -hw]
    if (v := 'toa_lw_flux_outgoing_2d_xy') in self._diagnostic_fields:
      output[v] = lw_flux_up[:, :, -hw]
    # Add clear sky toa

    # 1D diagnostics
    if (v := 'rad_heat_lw_1d_z') in self._diagnostic_fields:
      output[v] = _horiz_mean(lw_heating_rate)
    if (v := 'rad_heat_sw_1d_z') in self._diagnostic_fields:
      output[v] = _horiz_mean(sw_heating_rate)

    # Compute clear-sky radiative transfer and diagnostics, if desired.
    if self._do_clear_sky:
      lw_fluxes_clearsky = two_stream.solve_lw(
          p_ref_xxc,
          temperature,
          molecules_per_area,
          self.optics_lib,
          self.atmospheric_state,
          vmr_fields,
          sfc_temperature,
          cloud_r_eff_liq=None,
          cloud_path_liq=None,
          cloud_r_eff_ice=None,
          cloud_path_ice=None,
          use_scan=use_scan,
      )
      sw_fluxes_clearsky = two_stream.solve_sw(
          p_ref_xxc,
          temperature,
          molecules_per_area,
          self.optics_lib,
          self.atmospheric_state,
          vmr_fields,
          cloud_r_eff_liq=None,
          cloud_path_liq=None,
          cloud_r_eff_ice=None,
          cloud_path_ice=None,
          use_scan=use_scan,
      )
      # Compute the heating rate in K/s.
      lw_heating_rate_clearsky = two_stream.compute_heating_rate(
          lw_fluxes_clearsky['flux_net'], p_ref_xxc
      )
      sw_heating_rate_clearsky = two_stream.compute_heating_rate(
          sw_fluxes_clearsky['flux_net'], p_ref_xxc
      )

      if self._save_lw_sw_heating_rates:
        output['rad_heat_sw_clearsky_3d'] = sw_heating_rate_clearsky
        output['rad_heat_lw_clearsky_3d'] = lw_heating_rate_clearsky

      lw_flux_down_clearsky = lw_fluxes_clearsky['flux_down']
      lw_flux_up_clearsky = lw_fluxes_clearsky['flux_up']
      sw_flux_down_clearsky = sw_fluxes_clearsky['flux_down']
      sw_flux_up_clearsky = sw_fluxes_clearsky['flux_up']

      # Add clear-sky full flux profiles (remove surface halos)
      output['sw_flux_up_clearsky_full'] = sw_flux_up_clearsky[:, :, hw:]  # Full profile (..., nlev+1)
      output['sw_flux_down_clearsky_full'] = sw_flux_down_clearsky[:, :, hw:]  # Full profile (..., nlev+1)
      output['lw_flux_up_clearsky_full'] = lw_flux_up_clearsky[:, :, hw:]  # Full profile (..., nlev+1)
      output['lw_flux_down_clearsky_full'] = lw_flux_down_clearsky[:, :, hw:]  # Full profile (..., nlev+1)

      # 2D diagnostics
      if (v := 'surf_lw_flux_down_clearsky_2d_xy') in self._diagnostic_fields:
        output[v] = lw_flux_down_clearsky[:, :, hw]
      if (v := 'surf_lw_flux_up_clearsky_2d_xy') in self._diagnostic_fields:
        output[v] = lw_flux_up_clearsky[:, :, hw]
      if (v := 'surf_sw_flux_down_clearsky_2d_xy') in self._diagnostic_fields:
        output[v] = sw_flux_down_clearsky[:, :, hw]
      if (v := 'surf_sw_flux_up_clearsky_2d_xy') in self._diagnostic_fields:
        output[v] = sw_flux_up_clearsky[:, :, hw]

      df = self._diagnostic_fields
      if (v := 'toa_sw_flux_outgoing_clearsky_2d_xy') in df:
        output[v] = sw_flux_up_clearsky[:, :, -hw]
      if (v := 'toa_lw_flux_outgoing_clearsky_2d_xy') in df:
        output[v] = lw_flux_up_clearsky[:, :, -hw]

      # 1D diagnostics
      if (v := 'rad_heat_lw_clearsky_1d_z') in self._diagnostic_fields:
        output[v] = _horiz_mean(lw_heating_rate_clearsky)
      if (v := 'rad_heat_sw_clearsky_1d_z') in self._diagnostic_fields:
        output[v] = _horiz_mean(sw_heating_rate_clearsky)

    return output
