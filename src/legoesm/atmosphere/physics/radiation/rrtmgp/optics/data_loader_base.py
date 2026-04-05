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

"""A base loader for the RRTMGP lookup tables (Zarr-first, NetCDF fallback)."""

from collections.abc import Sequence
import os
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

Array: TypeAlias = jax.Array


def create_index(name_arr: Sequence[str]) -> dict[str, int]:
  """Utility function for generating an index from a sequence of names."""
  return {name: idx for idx, name in enumerate(name_arr)}


def _decode_string_var(arr: np.ndarray) -> list[str]:
  """Decode a byte-string or object array to a list of stripped Python strings."""
  result = []
  for item in arr.flat:
    if isinstance(item, bytes):
      result.append(item.decode("utf-8").strip())
    elif isinstance(item, str):
      result.append(item.strip())
    else:
      result.append(str(item).strip())
  return result


def _open_dataset(path: str) -> xr.Dataset:
  """Open a Zarr store or NetCDF file, auto-detecting format."""
  if os.path.isdir(path):
    return xr.open_zarr(path)
  if path.endswith(".zarr"):
    return xr.open_zarr(path)
  # NetCDF fallback
  return xr.open_dataset(path)


def parse_data_file(
    path: str,
) -> tuple[dict[str, Array], dict[str, int], dict[str, list[str]]]:
  """Load RRTMGP lookup tables from a Zarr store or NetCDF file.

  Args:
    path: Full path to a Zarr store directory or NetCDF file.

  Returns:
    A 3-tuple of:
      1) a dictionary of numeric variables as JAX Arrays,
      2) a dictionary of dimension name -> size,
      3) a dictionary of string variables as lists of decoded strings.
  """
  ds = _open_dataset(path)

  array_dict: dict[str, Array] = {}
  string_dict: dict[str, list[str]] = {}
  dim_map = dict(ds.sizes)

  for key in ds.data_vars:
    val = ds[key].values
    if val.dtype.kind in ("S", "U", "O"):
      # String / byte-string variable
      string_dict[key] = _decode_string_var(val)
      continue
    if np.issubdtype(val.dtype, np.floating):
      dtype = jnp.float_
    elif np.issubdtype(val.dtype, np.integer):
      dtype = jnp.int_
    else:
      raise ValueError(f"Unexpected dtype: {val.dtype}")
    array_dict[key] = jnp.array(val, dtype=dtype)

  ds.close()
  return array_dict, dim_map, string_dict
