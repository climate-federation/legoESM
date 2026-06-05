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

"""A Dataclass for shortwave optical properties of gases."""

from collections.abc import Mapping
import dataclasses
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import data_loader_base
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_gas_optics_base

Array: TypeAlias = jax.Array


@dataclasses.dataclass(frozen=True)
class LookupGasOpticsShortwave(lookup_gas_optics_base.AbstractLookupGasOptics):
  """Lookup table of gases' optical properties in the shortwave bands."""

  # Total solar irradiation
  solar_src_tot: float
  # Relative solar source contribution from each `g-point` `(n_gpt)`.
  solar_src_scaled: Array
  # Rayleigh absorption coefficient for lower atmosphere `(n_t_ref, n_η,
  # n_gpt)`.
  rayl_lower: Array
  # Rayleigh absorption coefficient for upper atmosphere `(n_t_ref, n_η,
  # n_gpt)`.
  rayl_upper: Array


def _load_data(
    tables: Mapping[str, Array],
    dims: Mapping[str, int],
    strings: Mapping[str, list[str]],
) -> dict[str, Any]:
  """Preprocesses the RRTMGP shortwave gas optics data.

  Args:
    tables: The extracted data as a dictionary of ``Array``s.
    dims: A dictionary containing dimension information for the tables.
    strings: A dictionary of decoded string variables.

  Returns:
    A dictionary containing dimension information and the preprocessed RRTMGP
    data as ``Array``s.
  """
  data = lookup_gas_optics_base.load_data(tables, dims, strings)
  solar_src = tables['solar_source_quiet']
  data['solar_src_tot'] = jnp.sum(solar_src)
  data['solar_src_scaled'] = solar_src / data['solar_src_tot']
  data['rayl_lower'] = tables['rayl_lower']
  data['rayl_upper'] = tables['rayl_upper']
  return data


def from_data_file(path: str) -> LookupGasOpticsShortwave:
  """Instantiate a ``LookupGasOpticsShortwave`` from a Zarr store or NetCDF file.

  Args:
    path: The full path to a Zarr store or NetCDF file containing the shortwave
      absorption coefficient lookup table.

  Returns:
    A ``LookupGasOpticsShortwave`` object.
  """
  tables, dims, strings = data_loader_base.parse_data_file(path)
  kwargs = _load_data(tables, dims, strings)
  return LookupGasOpticsShortwave(**kwargs)
