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

"""A Dataclass for longwave optical properties of gases."""

from collections.abc import Mapping
import dataclasses
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import data_loader_base
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_gas_optics_base

Array: TypeAlias = jax.Array


@dataclasses.dataclass(frozen=True)
class LookupGasOpticsLongwave(lookup_gas_optics_base.AbstractLookupGasOptics):
  """Lookup tables of gases' optical properties in the longwave bands."""

  # Planck fraction `(n_t_ref, n_p_ref, n_η, n_gpt)`.
  planck_fraction: Array
  # Number of reference temperatures, for Planck source calculations.
  n_t_plnk: int
  # reference temperatures for Planck source calculations `(n_t_plnk)`.
  t_planck: Array
  # total Planck source for each band `(n_bnd, n_t_plnk)`.
  totplnk: Array
  # Polynomial-fit coefficients ``(n_bnd, 2)`` for the optimal longwave
  # diffusivity-angle secant used by upstream ``compute_optimal_angles``
  # (mo_gas_optics_rrtmgp.F90):
  # ``secant(col, gpt) = optimal_angle_fit[band, 0] * exp(-tau_total)
  #                      + optimal_angle_fit[band, 1]``.
  # Fortran stores ``optimal_angle_fit(coefficient, band)`` (col-major);
  # netCDF on disk is therefore ``(n_bnd, 2)`` in C-order so the Python
  # access ``[band, coeff]`` matches Fortran ``(coeff, band)`` byte-for-
  # byte.  Replaces the fixed Fu-Liou ``1.66`` factor with a per-band,
  # per-column value that improves the diffusive integration when
  # column optical depths vary widely (e.g. stratosphere vs deep
  # cloud).  ``None`` when the netCDF file lacks ``optimal_angle_fit``
  # (older data files).
  optimal_angle_fit: Array | None = None


def _load_data(
    tables: Mapping[str, Array],
    dims: Mapping[str, int],
    strings: Mapping[str, list[str]],
) -> dict[str, Any]:
  """Preprocess the RRTMGP longwave gas optics data.

  Args:
    tables: The extracted data as a dictionary of ``Array``s.
    dims: A dictionary containing dimension information for the tables.
    strings: A dictionary of decoded string variables.

  Returns:
    A dictionary containing dimension information and the preprocessed RRTMGP
    data as ``Array``s.
  """
  data = lookup_gas_optics_base.load_data(tables, dims, strings)
  data['n_t_plnk'] = dims['temperature_Planck']
  data['planck_fraction'] = tables['plank_fraction']
  # Similarly to the original RRTM Fortran code, here we assume that
  # temperature minimum and maximum are the same for the absorption
  # coefficient grid and the Planck grid and the Planck grid is equally
  # spaced.
  data['t_planck'] = jnp.linspace(
      data['temperature_ref_min'],
      data['temperature_ref_max'],
      data['n_t_plnk'],
      dtype=jnp.float_,
  )
  data['totplnk'] = tables['totplnk']
  # ``optimal_angle_fit`` was added in recent rte-rrtmgp releases.  Tolerate
  # older data files that lack it by emitting ``None`` — the solver falls
  # back to the fixed ``1.66`` diffusivity factor in that case.  Loaded as
  # ``(2, n_bnd)`` matching the upstream Fortran storage convention.
  data['optimal_angle_fit'] = tables.get('optimal_angle_fit')
  return data


def from_data_file(path: str) -> LookupGasOpticsLongwave:
  """Instantiate a ``LookupGasOpticsLongwave`` from a Zarr store or NetCDF file.

  Args:
    path: The full path to a Zarr store or NetCDF file containing the longwave
      absorption coefficient lookup table.

  Returns:
    A ``LookupGasOpticsLongwave`` object.
  """
  tables, dims, strings = data_loader_base.parse_data_file(path)
  kwargs = _load_data(tables, dims, strings)
  return LookupGasOpticsLongwave(**kwargs)
