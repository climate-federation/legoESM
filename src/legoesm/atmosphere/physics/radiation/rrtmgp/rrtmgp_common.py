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

"""Common variables for rrtmgp."""

from legoesm.atmosphere.physics.radiation.rrtmgp.config import radiative_transfer

# Key for the applied radiative heating rate, in K/s.  When supercycling this
# field is only nonzero on the steps the radiative term is applied.
KEY_APPLIED_RADIATION = 'rad_heat_src_applied'
# Key for the stored radiative heating rate, in K/s.  This is the stored
# radiative heating rate and is nonzero for every step, even if supercycling.
KEY_STORED_RADIATION = 'rad_heat_src'

# Key for the longwave radiative fluxes, in W/m^2.
KEY_RADIATIVE_FLUX_LW = 'rad_flux_lw'
# Key for the shortwave radiative fluxes, in W/m^2.
KEY_RADIATIVE_FLUX_SW = 'rad_flux_sw'
# Key for the longwave radiative fluxes with cloud effects removed, in W/m^2.
KEY_RADIATIVE_FLUX_LW_CLEAR = 'rad_flux_lw_clear'
# Key for the shortave radiative fluxes with cloud effects removed, in W/m^2.
KEY_RADIATIVE_FLUX_SW_CLEAR = 'rad_flux_sw_clear'


# List of keys for allowed radiative diagnostics.
DIAGNOSTICS_KEYS = [
    # 2D diagnostics.
    'surf_lw_flux_down_2d_xy',
    'surf_lw_flux_up_2d_xy',
    'surf_sw_flux_down_2d_xy',
    'surf_sw_flux_up_2d_xy',
    'toa_sw_flux_incoming_2d_xy',
    'toa_sw_flux_outgoing_2d_xy',
    'toa_lw_flux_outgoing_2d_xy',
    # 1D diagnostics.
    'rad_heat_lw_1d_z',
    'rad_heat_sw_1d_z',
    # Clear sky, 2D diagnostics.
    'surf_lw_flux_down_clearsky_2d_xy',
    'surf_lw_flux_up_clearsky_2d_xy',
    'surf_sw_flux_down_clearsky_2d_xy',
    'surf_sw_flux_up_clearsky_2d_xy',
    'toa_sw_flux_outgoing_clearsky_2d_xy',
    'toa_lw_flux_outgoing_clearsky_2d_xy',
    # Clear sky, 1D diagnostics.
    'rad_heat_lw_clearsky_1d_z',
    'rad_heat_sw_clearsky_1d_z',
]


def required_keys(
    radiative_transfer_cfg: radiative_transfer.RadiativeTransfer | None,
) -> list[str]:
  """Return the required keys for the rrtmgp radiative transfer library."""
  if radiative_transfer_cfg is None:
    return []
  else:
    keys = [KEY_APPLIED_RADIATION, KEY_STORED_RADIATION]
    return keys
