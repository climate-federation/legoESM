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

"""A data class for atmospheric optical properties."""

import dataclasses

import jax

from legoesm.atmosphere.physics.radiation.rrtmgp.config import radiative_transfer
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_volume_mixing_ratio


@dataclasses.dataclass(frozen=True, kw_only=True)
class AtmosphericState:
  """Atmospheric gas concentrations and miscellaneous optical properties.

  Surface emissivity and surface albedo may be scalars (single-column mode)
  or per-column JAX arrays of shape ``(ncol, 1)`` (multi-column AMIP mode).
  The zenith angle may be a scalar or a per-column array of shape
  ``(ncol, 1, 1)`` so that it broadcasts correctly against the 3-D
  ``(ncol, 1, nlev+2)`` optical-depth fields in the two-stream solver.
  """
  # Surface emissivity; scalar or per-column array (ncol, 1).
  sfc_emis: float | jax.Array
  # Surface albedo; scalar or per-column array (ncol, 1).
  sfc_alb: float | jax.Array
  # Solar zenith angle; scalar or per-column array (ncol, 1).
  zenith: float | jax.Array
  # The total solar irradiance (in W/m²).
  irrad: float
  # Volume mixing ratio lookup for each gas species. Only water vapor and ozone
  # are assumed to be variable. Global means are used for all the other species.
  vmr: lookup_volume_mixing_ratio.LookupVolumeMixingRatio
  # The longwave incident flux at the top of the atmosphere (in W/m²).
  toa_flux_lw: float = 0.0


def from_config(
    atmospheric_state_cfg: radiative_transfer.AtmosphericStateCfg,
) -> AtmosphericState:
  """Instantiates an `AtmosphericState` object from config.

  Args:
    atmospheric_state_cfg: A `radiative_transfer.AtmosphericStateCfg` object
      containing atmospheric conditions and the path to a file containing
      volume mixing ratio sounding data.

  Returns:
    An `AtmosphericState` instance.
  """
  return AtmosphericState(
      sfc_emis=atmospheric_state_cfg.sfc_emis,
      sfc_alb=atmospheric_state_cfg.sfc_alb,
      zenith=atmospheric_state_cfg.zenith,
      irrad=atmospheric_state_cfg.irrad,
      toa_flux_lw=atmospheric_state_cfg.toa_flux_lw,
      vmr=lookup_volume_mixing_ratio.from_config(atmospheric_state_cfg),
  )
