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

"""Configuration for the RRTMGP radiation library."""

import dataclasses
import dataclasses_json  # Used for JSON serialization.


@dataclasses.dataclass(frozen=True, kw_only=True)
class RRTMOptics:
  """Parameters required by the radiation optics library."""

  # Path of NetCDF file containing the longwave lookup tables.
  longwave_nc_filepath: str
  # Path of NetCDF file containing the shortwave lookup tables.
  shortwave_nc_filepath: str
  # Path of NetCDF file containing the cloud longwave lookup table.
  cloud_longwave_nc_filepath: str
  # Path of NetCDF file containing the cloud shortwave lookup table.
  cloud_shortwave_nc_filepath: str


@dataclasses.dataclass(frozen=True, kw_only=True)
class GrayAtmosphereOptics:
  """Parameters for gray atmosphere optics."""

  # Reference surface pressure.
  p0: float = 1e5
  # The ratio of the pressure scale height to the partial-pressure scale height
  # of the infrared absorber.
  alpha: float = 3.5
  # Longwave optical depth of the entire gray atmosphere.
  d0_lw: float = 0.0
  # Shortwave optical depth of the entire gray atmosphere.
  d0_sw: float = 0.0


@dataclasses.dataclass(frozen=True, kw_only=True)
class OpticsParameters(dataclasses_json.DataClassJsonMixin):
  optics: RRTMOptics | GrayAtmosphereOptics

# iter-54 dead-code trim: removed ``AtmosphericStateCfg`` and
# ``RadiativeTransfer`` swirl_jatmos config classes.  Both were only
# wired through the iter-23-removed ``RRTMGP.__init__`` /
# ``atmospheric_state.from_config`` / ``lookup_volume_mixing_ratio.from_config``
# chain; legoESM constructs ``AtmosphericState`` and
# ``LookupVolumeMixingRatio`` directly via ``solve_columns`` and
# ``_build_optics_and_vmr``.
