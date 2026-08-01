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
class OpticsParameters:
  # Plain dataclass: the upstream swirl_jatmos source inherited
  # ``dataclasses_json.DataClassJsonMixin``, but nothing in legoESM ever called
  # its ``to_json``/``from_json``/``schema``, so the mixin cost a hard runtime
  # dependency and bought nothing.  ``dataclasses.asdict`` + ``json`` covers it
  # if serialisation is ever wanted.
  optics: RRTMOptics

# iter-54 / iter-61 dead-code trim: removed ``AtmosphericStateCfg``,
# ``RadiativeTransfer``, and ``GrayAtmosphereOptics`` swirl_jatmos
# config classes.  The first two were only wired through the
# iter-23-removed ``RRTMGP.__init__`` / ``atmospheric_state.from_config`` /
# ``lookup_volume_mixing_ratio.from_config`` chain.  The third was
# the type tag for an RRTMGP-internal gray-radiation impl that
# legoESM never used (legoESM has its own gray scheme at
# ``legoesm.atmosphere.physics.radiation.gray``).
