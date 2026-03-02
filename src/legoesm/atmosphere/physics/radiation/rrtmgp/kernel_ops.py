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

"""Kernel operations, written for single-core mode (unoptimized)."""

from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array


def forward_sum(f: Array, dim: Literal[0, 1, 2]) -> Array:
  """Compute the forward sum of f in the given dimension."""
  return f + jnp.roll(f, -1, axis=dim)


def backward_sum(f: Array, dim: Literal[0, 1, 2]) -> Array:
  """Compute the backward sum of f in the given dimension."""
  return jnp.roll(f, 1, axis=dim) + f


def centered_sum(f: Array, dim: Literal[0, 1, 2]) -> Array:
  """Compute the centered sum of f in the given dimension."""
  return jnp.roll(f, 1, axis=dim) + jnp.roll(f, -1, axis=dim)


def weighted_121_sum(f: Array, dim: Literal[0, 1, 2]) -> Array:
  """Compute the weighted 121 sum of f in the given dimension."""
  f_back = jnp.roll(f, 1, axis=dim)
  f_fwd = jnp.roll(f, -1, axis=dim)
  return f_back + 2 * f + f_fwd


def forward_difference(f: Array, dim: Literal[0, 1, 2]) -> Array:
  """Compute the forward difference of f in the given dimension."""
  return jnp.roll(f, -1, axis=dim) - f


def backward_difference(f: Array, dim: Literal[0, 1, 2]) -> Array:
  """Compute the backward difference of f in the given dimension."""
  return f - jnp.roll(f, 1, axis=dim)


def centered_difference(f: Array, dim: Literal[0, 1, 2]) -> Array:
  """Compute the centered difference of f in the given dimension."""
  return jnp.roll(f, -1, axis=dim) - jnp.roll(f, 1, axis=dim)


def shift_from_plus(f: Array, dim: Literal[0, 1, 2]) -> Array:
  """output_i = f_{i+1}."""
  return jnp.roll(f, -1, axis=dim)


def shift_from_minus(f: Array, dim: Literal[0, 1, 2]) -> Array:
  """output_i = f_{i-1}."""
  return jnp.roll(f, 1, axis=dim)
