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

"""Stretched grid utilities."""

from collections.abc import Mapping
from typing import Literal, TypeAlias

import jax

Array: TypeAlias = jax.Array
STRETCHED_GRID_KEY_PREFIX = 'stretched_grid'


def hc_key(dim: Literal[0, 1, 2]) -> str:
  """Gets the key for the stretched grid scale factor on nodes, given `dim`."""
  return STRETCHED_GRID_KEY_PREFIX + f'_hc{dim}'


def hf_key(dim: Literal[0, 1, 2]) -> str:
  """Gets the key for the stretched grid scale factor on faces, given `dim`."""
  return STRETCHED_GRID_KEY_PREFIX + f'_hf{dim}'


def sg_map_from_states(states: Mapping[str, Array]) -> dict[str, Array]:
  """Extracts stretched grid scale factors from the states dict."""
  sg_map = {}
  for dim in (0, 1, 2):
    if hc_key(dim) in states and hf_key(dim) in states:
      sg_map[hc_key(dim)] = states[hc_key(dim)]
      sg_map[hf_key(dim)] = states[hf_key(dim)]
  return sg_map


def get_use_stretched_grid(
    sg_map: Mapping[str, Array],
) -> tuple[bool, bool, bool]:
  """Gets the use_stretched_grid tuple from the stretched grid map."""
  return (
      hc_key(0) in sg_map and hf_key(0) in sg_map,
      hc_key(1) in sg_map and hf_key(1) in sg_map,
      hc_key(2) in sg_map and hf_key(2) in sg_map,
  )


def get_dxdydz(
    sg_map: dict[str, Array],
    grid_spacings: tuple[float, float, float],
    faces: bool = True,
) -> tuple[Array | float, Array | float, Array | float]:
  """Get dx, dy, dz as broadcastable 1D arrays (or floats for uniform grid)."""
  if faces:
    hx_key, hy_key, hz_key = hf_key(dim=0), hf_key(dim=1), hf_key(dim=2)
  else:
    hx_key, hy_key, hz_key = hc_key(dim=0), hc_key(dim=1), hc_key(dim=2)

  dx = sg_map.get(hx_key, grid_spacings[0])
  dy = sg_map.get(hy_key, grid_spacings[1])
  dz = sg_map.get(hz_key, grid_spacings[2])
  return dx, dy, dz
