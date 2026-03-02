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

"""Utility library for solving the radiative transfer equation (RTE)."""

import inspect
from typing import Any, Callable, TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array


def recurrent_op_1d(
    f: Callable[[Any], tuple[Array, Array]],
    init: Array,
    inputs: dict[str, Array],
    forward: bool = True,
) -> tuple[Array, Array]:
  """Compute sequence of recurrent operations for 1D array inputs.

  This version work on a 1D array and is written to have a similar style as
  _local_recurrent_op().  It uses a for loop, not scan.  However, it returns
  both the carry state and the accumulated output, even though these are equal
  here, in order to match the style of jax.lax.scan().  If this is slow, we can
  change it to only return the output.

  Currently not allowing length to be an optional input; could add that.

  Args:
    f: The recurrent operation to apply.
    init: The initial state of the recurrent operation.
    inputs: A dictionary of inputs to the recurrent operation.
    forward: Whether to run the recurrent operation in the forward direction.

  Returns:
    A tuple of the final carry state and the accumulated output.
  """
  for v in inputs:
    break
  n = len(inputs[v])  # pylint: disable=undefined-loop-variable, disable=unused-variable

  output = jnp.zeros(n)
  carry = init
  for i in range(n):
    slice_idx = i if forward else -i - 1
    plane_args = {k: v[slice_idx] for k, v in inputs.items()}

    arg_list = [
        plane_args[k]
        for j, k in enumerate(inspect.getfullargspec(f).args)
        if j != 0  # Assuming the first argument is the carry state.
    ]
    carry, next_layer = f(carry, *arg_list)

    if forward:
      i_set = i
    else:
      i_set = -i - 1
    output = output.at[i_set].set(next_layer)

  return carry, output


def recurrent_op_1d_scan(
    f: Callable[[Any], tuple[Array, Array]],
    init: Array,
    inputs: dict[str, Array],
    forward: bool = True,
) -> tuple[Array, Array]:
  """Compute sequence of recurrent operations for 1D array inputs using scan.

  Note: jax.lax.scan() may be inefficient on GPUs, because each iteration must
  launch a new kernel. May want to use the version with for loops on GPU.

  Args:
    f: The recurrent operation to apply.
    init: The initial state of the recurrent operation.
    inputs: A dictionary of inputs to the recurrent operation.
    forward: Whether to run the recurrent operation in the forward direction.

  Returns:
    A tuple of the final carry state and the accumulated output.
  """

  # `scan()` requires a single inputs argument, so use a wrapped version of `f`
  # that assumes the 2nd argument is a dictionary, and unpacks it and sends it
  # on to `f`.
  def wrapped_f_for_scan(
      carry: Any, inputs: dict[str, Array]
  ) -> tuple[Array, Array]:
    return f(carry, **inputs)

  return jax.lax.scan(wrapped_f_for_scan, init, inputs, reverse=not forward)


def recurrent_op(
    f: Callable[..., tuple[Array, Array]],
    init: Array,
    inputs: dict[str, Array],
    forward: bool = True,
) -> tuple[Array, Array]:
  """Compute sequence of recurrent operations.

  This version work on 3D arrays over the last dimension, and is written to have
  a similar style as _local_recurrent_op().  It uses a for loop, not scan.

  Currently not allowing length to be an optional input; could add that.

  For this and the scan() version below, as an alternate implementation, it may
  also be possible to simply vmap over the 1D versions of the recurrent_op
  functions.

  Args:
    f: The recurrent operation to apply.
    init: The initial state of the recurrent operation.
    inputs: A dictionary of inputs to the recurrent operation.
    forward: Whether to run the recurrent operation in the forward direction.

  Returns:
    A tuple of the final carry state and the accumulated output.
  """
  for v in inputs:
    break
  shape = inputs[v].shape  # pylint: disable=undefined-loop-variable, pylint: disable=undefined-variable
  assert len(shape) == 3
  nz = shape[2]
  output = jnp.zeros_like(inputs[v])  # pylint: disable=undefined-loop-variable, pylint: disable=undefined-variable
  carry = init
  for i in range(nz):
    # prev_idx = i - 1
    slice_idx = i if forward else -i - 1
    plane_args = {k: v[:, :, slice_idx] for k, v in inputs.items()}
    # prev_slice_idx = prev_idx if forward else -prev_idx - 1

    arg_list = [
        plane_args[k]
        for j, k in enumerate(inspect.getfullargspec(f).args)
        if j != 0
    ]
    carry, next_layer = f(carry, *arg_list)

    if forward:
      i_set = i
    else:
      i_set = -i - 1
    output = output.at[:, :, i_set].set(next_layer)

  return carry, output


def recurrent_op_scan(
    f: Callable[..., tuple[Array, Array]],
    init: Array,
    inputs: dict[str, Array],
    forward: bool = True,
) -> tuple[Array, Array]:
  """Compute sequence of recurrent operations for 3D array inputs using scan.

  Scan over the last dimension (internally transform to first dimension).

  Note: jax.lax.scan() may be inefficient on GPUs, because each iteration must
  launch a new kernel. May want to use the version with for loops on GPU.

  Args:
    f: The recurrent operation to apply.
    init: The initial state of the recurrent operation.
    inputs: A dictionary of inputs to the recurrent operation.
    forward: Whether to run the recurrent operation in the forward direction.

  Returns:
    A tuple of the final carry state and the accumulated output.
  """
  # Transpose axes because scan() always scans over the first dimension, whereas
  # we want to scan over the last dimension.

  def wrapped_f_for_scan(carry, inputs):
    return f(carry, **inputs)

  # Move last axis (z) to first before going into the scan. so that inputs are
  # (nz, nx, ny) instead of (nx, ny, nz).
  inputs = {k: jnp.moveaxis(v, -1, 0) for k, v in inputs.items()}

  carry, output = jax.lax.scan(
      wrapped_f_for_scan, init, inputs, reverse=not forward
  )

  # carry is (nx, ny); output is (nz, nx, ny).
  # Move first axis (z) to last after coming out of the scan so that the
  # output field has the desired shape (nx, ny, nz).
  output = jnp.moveaxis(output, 0, -1)

  return carry, output


def recurrent_op_with_halos(
    f: Callable[..., tuple[Array, Array]],
    init: Array,
    inputs: dict[str, Array],
    forward: bool = True,
    use_scan: bool = False,
) -> tuple[Array, Array]:
  """Compute sequence of recurrent operations, accounting for halos in z."""
  halo_width = 1  # Assumed and hardcoded.

  # Step 1. Remove halos in z dimension.
  inputs = {k: v[:, :, halo_width:-halo_width] for k, v in inputs.items()}

  # Step 2. Run recurrent operation.
  if use_scan:
    carry, output = recurrent_op_scan(f, init, inputs, forward)
  else:
    carry, output = recurrent_op(f, init, inputs, forward)

  # Step 3. Pad result with halo layers that were removed so the output shape
  # matches the input shape. The values in the halos are set to be zero.
  output = jnp.pad(
      output,
      ((0, 0), (0, 0), (halo_width, halo_width)),
      mode='constant',
      constant_values=0.0,
  )

  # Step 4. For the face that initiates the recurrence, set the halo layer to be
  # the initial value.
  if forward:
    output = output.at[:, :, 0].set(init)
  else:
    output = output.at[:, :, -1].set(init)

  return carry, output
