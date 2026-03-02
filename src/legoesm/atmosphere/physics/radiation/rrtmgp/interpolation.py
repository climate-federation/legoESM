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

"""Interpolation functionality."""

import functools
from typing import Literal, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.radiation.rrtmgp import kernel_ops

Array: TypeAlias = jax.Array
T = TypeVar('T')


def centered_node_to_face(v_node: Array, dim: Literal[0, 1, 2]) -> Array:
  """Perform centered 2nd-order interpolation from nodes to faces.

  * An array evaluated on nodes has index i <==> coordinate location x_i
  * An array evaluated on faces has index i <==> coordinate location x_{i-1/2}

  E.g., interpolating in dim 0:
    v_face[i, j, k] = 0.5 * (v_node[i, j, k] + v_node[i-1, j, k])

  Args:
    v_node: A 3D array, evaluated on nodes.
    dim: The dimension along with the interpolation is performed.

  Returns:
    A 3D array interpolated from `v_node`, which is evaluted on faces in
    dimension `dim`, and evaluated on nodes in other dimensions.
  """
  v_face = kernel_ops.backward_sum(v_node, dim) / 2
  return v_face


def centered_face_to_node(v_face: Array, dim: Literal[0, 1, 2]) -> Array:
  """Perform centered 2nd-order interpolation from faces to nodes.

  E.g., interpolating in dim 0:
    v_node[i, j, k] = 0.5 * (v_face[i, j, k] + v_face[i+1, j, k])

  Args:
    v_face: A 3D array, evaluated on faces.
    dim: The dimension along with the interpolation is performed.

  Returns:
    A 3D array interpolated from `v_face`, which is evaluted on faces in
    dimension `dim`, and evaluated on nodes in other dimensions.
  """
  v_node = kernel_ops.forward_sum(v_face, dim) / 2
  return v_node


# Define aliases for centered interpolation in each dimension, for clearer
# naming.
def x_f_to_c(v_face: Array) -> Array:
  return centered_face_to_node(v_face, 0)


def y_f_to_c(v_face: Array) -> Array:
  return centered_face_to_node(v_face, 1)


def z_f_to_c(v_face: Array) -> Array:
  return centered_face_to_node(v_face, 2)


def x_c_to_f(v_node: Array) -> Array:
  return centered_node_to_face(v_node, 0)


def y_c_to_f(v_node: Array) -> Array:
  return centered_node_to_face(v_node, 1)


def z_c_to_f(v_node: Array) -> Array:
  return centered_node_to_face(v_node, 2)


##### Centered interpolations ####
def interp2(f_l1: T, f_r1: T) -> T:
  """Perform centered 2nd-order interpolation from pointwise values."""
  return (f_l1 + f_r1) / 2


def interp2_cell_avg(f_l1: T, f_r1: T) -> T:
  """Perform centered 2nd-order interpolation from cell average values."""
  return (f_l1 + f_r1) / 2


def interp4(f_l2: T, f_l1: T, f_r1: T, f_r2: T) -> T:
  """Perform centered 4th-order interpolation from pointwise values.

  Geometrical representation without using indices:

      .     .  |  .     .
      ^     ^     ^     ^
      f_l2  f_l1  f_r1  f_r2

  Args:
    f_l2: The value two points left of the center.
    f_l1: The value one point left of the center.
    f_r1: The value one point right of the center.
    f_r2: The value two points right of the center.

  Returns:
    The interpolated value at the center.
  """
  return 9 / 16 * (f_l1 + f_r1) - 1 / 16 * (f_l2 + f_r2)


def interp4_cell_avg(f_l2: T, f_l1: T, f_r1: T, f_r2: T) -> T:
  """Perform centered 4th-order interpolation from cell average values.

  Same geometrical interpretation as `interp4`, but the input values `f` are
  cell average values rather than pointwise values.  The interpolated value is a
  pointwise value.

  Args:
    f_l2: The value two points left of the center.
    f_l1: The value one point left of the center.
    f_r1: The value one point right of the center.
    f_r2: The value two points right of the center.

  Returns:
    The interpolated value at the center.
  """
  return 7 / 12 * (f_l1 + f_r1) - 1 / 12 * (f_l2 + f_r2)


def interp4_node_to_face(f_node: Array, dim: Literal[0, 1, 2]) -> Array:
  """Perform centered 4th-order interpolation from nodes to faces."""
  f_l2 = jnp.roll(f_node, 2, axis=dim)
  f_l1 = jnp.roll(f_node, 1, axis=dim)
  f_r1 = f_node
  f_r2 = jnp.roll(f_node, -1, axis=dim)
  return interp4(f_l2, f_l1, f_r1, f_r2)


def interp4_face_to_node(
    f_face: Array, dim: Literal[0, 1, 2]
) -> Array:
  """Perform centered 4th-order interpolation from faces to nodes."""
  f_l2 = jnp.roll(f_face, 1, axis=dim)
  f_l1 = f_face
  f_r1 = jnp.roll(f_face, -1, axis=dim)
  f_r2 = jnp.roll(f_face, -2, axis=dim)
  return interp4(f_l2, f_l1, f_r1, f_r2)


def interp4_cell_avg_node_to_face(
    f_node: Array, dim: Literal[0, 1, 2]
) -> Array:
  """Perform centered 4th-order interpolation from nodes to faces."""
  f_l2 = jnp.roll(f_node, 2, axis=dim)
  f_l1 = jnp.roll(f_node, 1, axis=dim)
  f_r1 = f_node
  f_r2 = jnp.roll(f_node, -1, axis=dim)
  return interp4_cell_avg(f_l2, f_l1, f_r1, f_r2)


def interp4_cell_avg_face_to_node(
    f_face: Array, dim: Literal[0, 1, 2]
) -> Array:
  """Perform centered 4th-order interpolation from faces to nodes."""
  f_l2 = jnp.roll(f_face, 1, axis=dim)
  f_l1 = f_face
  f_r1 = jnp.roll(f_face, -1, axis=dim)
  f_r2 = jnp.roll(f_face, -2, axis=dim)
  return interp4_cell_avg(f_l2, f_l1, f_r1, f_r2)


##### WENO3 interpolation using the original WENO3-JS. #####
def weno3(f_l2: T, f_l1: T, f_r1: T, f_r2: T) -> tuple[T, T]:
  """Perform WENO3-JS interpolation.

  Given cell average values f as depicted on a uniform computational grid, find
  an interpolation to the center value indicated (halfway between grid points).

  Geometrical representation without using indices:

      .     .  |  .     .
      ^     ^     ^     ^
     f_l2  f_l1  f_r1  f_r2

  Return left-biased and right-biased interpolations evaluated at the point
  indicated by the '|'.  The left-biased interpolation is labeled '+' (e.g., for
  positive velocities) and the right-biased interpolation is labeled '-' (e.g.,
  for negative velocities).

  Args:
    f_l2: The value two points left of the center.
    f_l1: The value one point left of the center.
    f_r1: The value one point right of the center.
    f_r2: The value two points right of the center.

  Returns:
    A tuple of two arrays, the left-biased and right-biased interpolations.
  """
  # Compute smoothness indicators beta.
  beta1_plus = (f_l1 - f_l2) ** 2
  beta2_plus = (f_r1 - f_l1) ** 2
  beta1_minus = (f_r2 - f_r1) ** 2
  beta2_minus = beta2_plus

  c1 = 1 / 3  # Optimal linear weights for WENO3-JS.
  c2 = 2 / 3
  epsilon = 1e-6
  alpha1_plus = c1 / (beta1_plus + epsilon) ** 2
  alpha2_plus = c2 / (beta2_plus + epsilon) ** 2
  alpha1_minus = c1 / (beta1_minus + epsilon) ** 2
  alpha2_minus = c2 / (beta2_minus + epsilon) ** 2
  # Compute the nonlinear weights.
  w1_plus = alpha1_plus / (alpha1_plus + alpha2_plus)
  w2_plus = alpha2_plus / (alpha1_plus + alpha2_plus)
  w1_minus = alpha1_minus / (alpha1_minus + alpha2_minus)
  w2_minus = alpha2_minus / (alpha1_minus + alpha2_minus)

  # Compute the local reconstructions from different stencils.
  f1_plus = -1 / 2 * f_l2 + 3 / 2 * f_l1
  f2_plus = 1 / 2 * f_l1 + 1 / 2 * f_r1
  f1_minus = 3 / 2 * f_r1 - 1 / 2 * f_r2
  f2_minus = f2_plus

  # Obtain the WENO reconstruction by combining the nonlinear weights with the
  # local reconstructions on the faces.
  f_face_plus = w1_plus * f1_plus + w2_plus * f2_plus
  f_face_minus = w1_minus * f1_minus + w2_minus * f2_minus
  return f_face_plus, f_face_minus


def weno3_node_to_face(
    f_node: Array, dim: Literal[0, 1, 2]
) -> tuple[Array, Array]:
  """Perform WENO3-JS interpolation from nodes to faces.

  Uses the fact that the face to the *left* of a node has the same array index
  as the node.

  Boundary conditions are not handled here.  This function only works for
  periodic boundaries.  Other boundary conditions must be handled manually
  downstream.

  Args:
    f_node: A 3D array, evaluated on nodes in dimension `dim`.
    dim: The dimension along with the interpolation is performed.

  Returns:
    A tuple of two 3D arrays interpolated from `f_node`.  The outputs are
    evaluated on faces in dimension `dim`, and in the other dimensions are
    located at whichever location `f_node` is.  The first array is the "plus"
    (left-biased) interpolation, and the second array is the "minus"
    (right-biased) interpolation.
  """
  # Handle boundary conditions downstream.
  f_l2 = jnp.roll(f_node, 2, axis=dim)
  f_l1 = jnp.roll(f_node, 1, axis=dim)
  f_r1 = f_node
  f_r2 = jnp.roll(f_node, -1, axis=dim)
  return weno3(f_l2, f_l1, f_r1, f_r2)  # Return f+, f-. on the face i-1/2.


def weno3_face_to_node(
    f_face: Array, dim: Literal[0, 1, 2]
) -> tuple[Array, Array]:
  """Perform WENO3-JS interpolation from faces to nodes.

  Uses the fact that the face to the *left* of a node has the same array index
  as the node.

  Boundary conditions are not handled here.  This function only works for
  periodic boundaries.  Other boundary conditions must be handled manually
  downstream.

  Args:
    f_face: A 3D array, evaluated on faces in dimension `dim`.
    dim: The dimension along with the interpolation is performed.

  Returns:
    A tuple of two 3D arrays interpolated from `f_face`.  The outputs are
    evaluated on nodes in dimension `dim`, and in the other dimensions are
    located at whichever location `f_face` is.  The first array is the "plus"
    (left-biased) interpolation, and the second array is the "minus"
    (right-biased) interpolation.
  """
  # Handle boundary conditions downstream.
  f_l2 = jnp.roll(f_face, 1, axis=dim)
  f_l1 = f_face
  f_r1 = jnp.roll(f_face, -1, axis=dim)
  f_r2 = jnp.roll(f_face, -2, axis=dim)
  return weno3(f_l2, f_l1, f_r1, f_r2)  # Return f+, f- on the node i.


##### WENO5 interpolation using the original WENO5-JS. #####
def weno5_js(
    f_l3: T, f_l2: T, f_l1: T, f_r1: T, f_r2: T, f_r3: T
) -> tuple[T, T]:
  """Perform WENO5-JS interpolation.

  Given cell average values f as depicted on a uniform computational grid, find
  an interpolation to the center value indicated (halfway between grid points).

  Geometrical representation without using indices:

      .     .     .  |  .     .     .
      ^     ^     ^     ^     ^     ^
     f_l3  f_l2  f_l1  f_r1  f_r2  f_r3

  Return left-biased and right-biased interpolations evaluated at the point
  indicated by the '|'.  The left-biased interpolation is labeled '+' (e.g., for
  positive velocities) and the right-biased interpolation is labeled '-' (e.g.,
  for negative velocities).

  Args:
    f_l3: The value three points left of the center.
    f_l2: The value two points left of the center.
    f_l1: The value one point left of the center.
    f_r1: The value one point right of the center.
    f_r2: The value two points right of the center.
    f_r3: The value three points right of the center.

  Returns:
    A tuple of two arrays, the left-biased and right-biased interpolations.
  """
  # Compute smoothness indicators beta.
  beta1_plus = (
      13 / 12 * (f_l3 - 2 * f_l2 + f_l1) ** 2
      + 1 / 4 * (f_l3 - 4 * f_l2 + 3 * f_l1) ** 2
  )
  beta2_plus = (
      13 / 12 * (f_l2 - 2 * f_l1 + f_r1) ** 2 + 1 / 4 * (f_l2 - f_r1) ** 2
  )
  beta3_plus = (
      13 / 12 * (f_l1 - 2 * f_r1 + f_r2) ** 2
      + 1 / 4 * (3 * f_l1 - 4 * f_r1 + f_r2) ** 2
  )

  beta1_minus = (
      13 / 12 * (f_r1 - 2 * f_r2 + f_r3) ** 2
      + 1 / 4 * (3 * f_r1 - 4 * f_r2 + f_r3) ** 2
  )
  beta2_minus = (
      13 / 12 * (f_l1 - 2 * f_r1 + f_r2) ** 2 + 1 / 4 * (f_l1 - f_r2) ** 2
  )
  beta3_minus = (
      13 / 12 * (f_l2 - 2 * f_l1 + f_r1) ** 2
      + 1 / 4 * (f_l2 - 4 * f_l1 + 3 * f_r1) ** 2
  )

  c1, c2, c3 = 0.1, 0.6, 0.3  # Optimal linear weights for WENO5-JS.
  epsilon = 1e-15
  alpha1_plus = c1 / (beta1_plus + epsilon) ** 2
  alpha2_plus = c2 / (beta2_plus + epsilon) ** 2
  alpha3_plus = c3 / (beta3_plus + epsilon) ** 2
  alpha1_minus = c1 / (beta1_minus + epsilon) ** 2
  alpha2_minus = c2 / (beta2_minus + epsilon) ** 2
  alpha3_minus = c3 / (beta3_minus + epsilon) ** 2

  # Compute the nonlinear weights.
  alpha_plus_sum = alpha1_plus + alpha2_plus + alpha3_plus
  alpha_minus_sum = alpha1_minus + alpha2_minus + alpha3_minus
  w1_plus = alpha1_plus / alpha_plus_sum
  w2_plus = alpha2_plus / alpha_plus_sum
  w3_plus = alpha3_plus / alpha_plus_sum
  w1_minus = alpha1_minus / alpha_minus_sum
  w2_minus = alpha2_minus / alpha_minus_sum
  w3_minus = alpha3_minus / alpha_minus_sum

  # Compute the local reconstructions from different stencils.
  f1_plus = 1 / 3 * f_l3 - 7 / 6 * f_l2 + 11 / 6 * f_l1
  f2_plus = -1 / 6 * f_l2 + 5 / 6 * f_l1 + 1 / 3 * f_r1
  f3_plus = 1 / 3 * f_l1 + 5 / 6 * f_r1 - 1 / 6 * f_r2

  f1_minus = 11 / 6 * f_r1 - 7 / 6 * f_r2 + 1 / 3 * f_r3
  f2_minus = 1 / 3 * f_l1 + 5 / 6 * f_r1 - 1 / 6 * f_r2
  f3_minus = -1 / 6 * f_l2 + 5 / 6 * f_l1 + 1 / 3 * f_r1

  # Obtain the WENO reconstruction by combining the nonlinear weights with the
  # local reconstructions on the faces.
  f_face_plus = w1_plus * f1_plus + w2_plus * f2_plus + w3_plus * f3_plus
  f_face_minus = w1_minus * f1_minus + w2_minus * f2_minus + w3_minus * f3_minus
  return f_face_plus, f_face_minus


def weno5_z(
    f_l3: Array, f_l2: Array, f_l1: Array, f_r1: Array, f_r2: Array, f_r3: Array
) -> tuple[Array, Array]:
  """Perform WENO5-Z interpolation.

  For general information on the WENO reconstruction, see the docstring for
  `weno5` (WENO5-JS)above.  For details on the WENO5-Z method, see R. Borges
  (2008) et al.

  Args:
    f_l3: The value three points left of the center.
    f_l2: The value two points left of the center.
    f_l1: The value one point left of the center.
    f_r1: The value one point right of the center.
    f_r2: The value two points right of the center.
    f_r3: The value three points right of the center.

  Returns:
    A tuple of two arrays, the left-biased and right-biased interpolations.
  """
  # Compute smoothness indicators beta.
  beta1_plus = (
      13 / 12 * (f_l3 - 2 * f_l2 + f_l1) ** 2
      + 1 / 4 * (f_l3 - 4 * f_l2 + 3 * f_l1) ** 2
  )
  beta2_plus = (
      13 / 12 * (f_l2 - 2 * f_l1 + f_r1) ** 2 + 1 / 4 * (f_l2 - f_r1) ** 2
  )
  beta3_plus = (
      13 / 12 * (f_l1 - 2 * f_r1 + f_r2) ** 2
      + 1 / 4 * (3 * f_l1 - 4 * f_r1 + f_r2) ** 2
  )

  beta1_minus = (
      13 / 12 * (f_r1 - 2 * f_r2 + f_r3) ** 2
      + 1 / 4 * (3 * f_r1 - 4 * f_r2 + f_r3) ** 2
  )
  beta2_minus = (
      13 / 12 * (f_l1 - 2 * f_r1 + f_r2) ** 2 + 1 / 4 * (f_l1 - f_r2) ** 2
  )
  beta3_minus = (
      13 / 12 * (f_l2 - 2 * f_l1 + f_r1) ** 2
      + 1 / 4 * (f_l2 - 4 * f_l1 + 3 * f_r1) ** 2
  )

  tau5_plus = jnp.abs(beta1_plus - beta3_plus)
  tau5_minus = jnp.abs(beta1_minus - beta3_minus)

  c1, c2, c3 = 0.1, 0.6, 0.3  # Optimal linear weights
  epsilon = 1e-20
  alpha1_plus = c1 * (1 + tau5_plus / (beta1_plus + epsilon))
  alpha2_plus = c2 * (1 + tau5_plus / (beta2_plus + epsilon))
  alpha3_plus = c3 * (1 + tau5_plus / (beta3_plus + epsilon))
  alpha1_minus = c1 * (1 + tau5_minus / (beta1_minus + epsilon))
  alpha2_minus = c2 * (1 + tau5_minus / (beta2_minus + epsilon))
  alpha3_minus = c3 * (1 + tau5_minus / (beta3_minus + epsilon))

  # Compute the nonlinear weights.
  alpha_plus_sum = alpha1_plus + alpha2_plus + alpha3_plus
  alpha_minus_sum = alpha1_minus + alpha2_minus + alpha3_minus
  w1_plus = alpha1_plus / alpha_plus_sum
  w2_plus = alpha2_plus / alpha_plus_sum
  w3_plus = alpha3_plus / alpha_plus_sum
  w1_minus = alpha1_minus / alpha_minus_sum
  w2_minus = alpha2_minus / alpha_minus_sum
  w3_minus = alpha3_minus / alpha_minus_sum

  # Compute the local reconstructions from different stencils.
  f1_plus = 1 / 3 * f_l3 - 7 / 6 * f_l2 + 11 / 6 * f_l1
  f2_plus = -1 / 6 * f_l2 + 5 / 6 * f_l1 + 1 / 3 * f_r1
  f3_plus = 1 / 3 * f_l1 + 5 / 6 * f_r1 - 1 / 6 * f_r2

  f1_minus = 11 / 6 * f_r1 - 7 / 6 * f_r2 + 1 / 3 * f_r3
  f2_minus = 1 / 3 * f_l1 + 5 / 6 * f_r1 - 1 / 6 * f_r2
  f3_minus = -1 / 6 * f_l2 + 5 / 6 * f_l1 + 1 / 3 * f_r1

  # Obtain the WENO reconstruction by combining the nonlinear weights with the
  # local reconstructions on the faces.
  f_face_plus = w1_plus * f1_plus + w2_plus * f2_plus + w3_plus * f3_plus
  f_face_minus = w1_minus * f1_minus + w2_minus * f2_minus + w3_minus * f3_minus
  return f_face_plus, f_face_minus


def weno5_node_to_face(
    f_node: Array, dim: Literal[0, 1, 2], method: Literal['JS', 'Z']
) -> tuple[Array, Array]:
  """Perform WENO5 interpolation from nodes to faces.

  Uses the fact that the face to the *left* of a node has the same array index
  as the node.

  Boundary conditions are not handled here.  This function only works for
  periodic boundaries.  Other boundary conditions must be handled manually
  downstream.

  Args:
    f_node: A 3D array, evaluated on nodes in dimension `dim`.
    dim: The dimension along with the interpolation is performed.
    method: The WENO5 method to use.  Must be 'JS' or 'Z' for WENO5-{JS,Z}.

  Returns:
    A tuple of two 3D arrays interpolated from `f_node`.  The outputs are
    evaluated on faces in dimension `dim`, and in the other dimensions are
    located at whichever location `f_node` is.  The first array is the "plus"
    (left-biased) interpolation, and the second array is the "minus"
    (right-biased) interpolation.
  """
  # Handle boundary conditions downstream.
  f_l3 = jnp.roll(f_node, 3, axis=dim)
  f_l2 = jnp.roll(f_node, 2, axis=dim)
  f_l1 = jnp.roll(f_node, 1, axis=dim)
  f_r1 = f_node
  f_r2 = jnp.roll(f_node, -1, axis=dim)
  f_r3 = jnp.roll(f_node, -2, axis=dim)
  # Return f+, f-. on the face i-1/2.
  if method == 'JS':
    return weno5_js(f_l3, f_l2, f_l1, f_r1, f_r2, f_r3)
  elif method == 'Z':
    return weno5_z(f_l3, f_l2, f_l1, f_r1, f_r2, f_r3)
  else:
    raise ValueError(f'Unknown method: {method}')


def weno5_face_to_node(
    f_face: Array, dim: Literal[0, 1, 2], method: Literal['JS', 'Z']
) -> tuple[Array, Array]:
  """Perform WENO5-JS interpolation from faces to nodes.

  Uses the fact that the face to the *left* of a node has the same array index
  as the node.

  Boundary conditions are not handled here.  This function only works for
  periodic boundaries.  Other boundary conditions must be handled manually
  downstream.

  Args:
    f_face: A 3D array, evaluated on faces in dimension `dim`.
    dim: The dimension along with the interpolation is performed.
    method: The WENO5 method to use.  Must be 'JS' or 'Z' for WENO5-{JS,Z}.

  Returns:
    A tuple of two 3D arrays interpolated from `f_face`.  The outputs are
    evaluated on nodes in dimension `dim`, and in the other dimensions are
    located at whichever location `f_face` is.  The first array is the "plus"
    (left-biased) interpolation, and the second array is the "minus"
    (right-biased) interpolation.
  """
  # Handle boundary conditions downstream.
  f_l3 = jnp.roll(f_face, 2, axis=dim)
  f_l2 = jnp.roll(f_face, 1, axis=dim)
  f_l1 = f_face
  f_r1 = jnp.roll(f_face, -1, axis=dim)
  f_r2 = jnp.roll(f_face, -2, axis=dim)
  f_r3 = jnp.roll(f_face, -3, axis=dim)
  # Return f+, f- on the node i.
  if method == 'JS':
    return weno5_js(f_l3, f_l2, f_l1, f_r1, f_r2, f_r3)
  elif method == 'Z':
    return weno5_z(f_l3, f_l2, f_l1, f_r1, f_r2, f_r3)
  else:
    raise ValueError(f'Unknown method: {method}')


# Convenience definitions.
weno5_js_node_to_face = functools.partial(weno5_node_to_face, method='JS')
weno5_z_node_to_face = functools.partial(weno5_node_to_face, method='Z')
weno5_js_face_to_node = functools.partial(weno5_face_to_node, method='JS')
weno5_z_face_to_node = functools.partial(weno5_face_to_node, method='Z')


##### Below this, WENO5 interpolation special for RRTMGP #####
def _weno5_nonlinear_weights(
    f_node: Array, dim: Literal[0, 1, 2], wall_bc: bool
) -> tuple[Array, Array, Array, Array, Array, Array]:
  """Compute the nonlinear weights for WENO5."""
  # The following quantities are all on nodes.
  f = f_node
  f_iminus3 = jnp.roll(f, 3, axis=dim)
  f_iminus2 = jnp.roll(f, 2, axis=dim)
  f_iminus1 = jnp.roll(f, 1, axis=dim)
  f_iplus1 = jnp.roll(f, -1, axis=dim)
  f_iplus2 = jnp.roll(f, -2, axis=dim)

  # Compute beta, which are the smoothness indicators.
  beta1_plus = (
      13 / 12 * (f_iminus3 - 2 * f_iminus2 + f_iminus1) ** 2
      + 1 / 4 * (f_iminus3 - 4 * f_iminus2 + 3 * f_iminus1) ** 2
  )
  beta2_plus = (
      13 / 12 * (f_iminus2 - 2 * f_iminus1 + f) ** 2
      + 1 / 4 * (f_iminus2 - f) ** 2
  )
  beta3_plus = (
      13 / 12 * (f_iminus1 - 2 * f + f_iplus1) ** 2
      + 1 / 4 * (3 * f_iminus1 - 4 * f + f_iplus1) ** 2
  )

  beta1_minus = (
      13 / 12 * (f - 2 * f_iplus1 + f_iplus2) ** 2
      + 1 / 4 * (3 * f - 4 * f_iplus1 + f_iplus2) ** 2
  )
  beta2_minus = (
      13 / 12 * (f_iminus1 - 2 * f + f_iplus1) ** 2
      + 1 / 4 * (f_iminus1 - f_iplus1) ** 2
  )
  beta3_minus = (
      13 / 12 * (f_iminus2 - 2 * f_iminus1 + f) ** 2
      + 1 / 4 * (f_iminus2 - 4 * f_iminus1 + 3 * f) ** 2
  )

  c1, c2, c3 = 0.1, 0.6, 0.3  # Optimal linear weights for WENO5-JS.
  epsilon = 1e-5

  alpha1_plus = c1 / (beta1_plus + epsilon)**2
  alpha2_plus = c2 / (beta2_plus + epsilon)**2
  alpha3_plus = c3 / (beta3_plus + epsilon)**2

  alpha1_minus = c1 / (beta1_minus + epsilon)**2
  alpha2_minus = c2 / (beta2_minus + epsilon)**2
  alpha3_minus = c3 / (beta3_minus + epsilon)**2

  # Deal with boundaries, if we have boundaries instead of periodic BCs.
  # Here we, ASSUME the values in the halos are usable with legitimate values.
  # Strategy: For the first interior face, we set the unnormalized weight alpha1
  # to 0 because there are not enough points for its stencil.  E.g., note that
  # beta1_plus (and hence alpha1_plus) uses f_iminus3, which is not defined for
  # the first interior face.  For the last interior face we do the same thing --
  # set alpha1_minus to 0.
  # The result is that WENO5 will adapt to using the other stencils.
  # For the second interior face, alpha1_plus uses f_iminus3 which is the halo
  # node.  So if the halo node value is ok, then this should be fine.
  hw = 1  # Assumed halo width of 1.
  if wall_bc:
    alpha1_plus = alpha1_plus.at[:, :, hw + 1].set(0)
    alpha1_minus = alpha1_minus.at[:, :, -hw - 1].set(0)

  # Compute the nonlinear weights.
  w1_plus = alpha1_plus / (alpha1_plus + alpha2_plus + alpha3_plus)
  w2_plus = alpha2_plus / (alpha1_plus + alpha2_plus + alpha3_plus)
  w3_plus = alpha3_plus / (alpha1_plus + alpha2_plus + alpha3_plus)

  w1_minus = alpha1_minus / (alpha1_minus + alpha2_minus + alpha3_minus)
  w2_minus = alpha2_minus / (alpha1_minus + alpha2_minus + alpha3_minus)
  w3_minus = alpha3_minus / (alpha1_minus + alpha2_minus + alpha3_minus)
  return w1_plus, w2_plus, w3_plus, w1_minus, w2_minus, w3_minus


def _weno5_local_reconstructions(
    f_node: Array, dim: Literal[0, 1, 2]
) -> tuple[Array, Array, Array, Array, Array, Array]:
  """Compute the local reconstructions from different stencils for WENO5.

  Args:
    f_node: A 3D array, evaluated on nodes.
    dim: The dimension along with the interpolation is performed.

  Returns:
    A tuple of six 3D arrays, the local reconstructions of the input nodal array
    on the face i - 1/2, for different stencils.
  """
  # The following quantities are on nodes.
  f = f_node
  f_iminus3 = jnp.roll(f, 3, axis=dim)
  f_iminus2 = jnp.roll(f, 2, axis=dim)
  f_iminus1 = jnp.roll(f, 1, axis=dim)
  f_iplus1 = jnp.roll(f, -1, axis=dim)
  f_iplus2 = jnp.roll(f, -2, axis=dim)

  # Compute the local reconstructions from the various stencils.
  # These are approximations on the face i - 1/2.  We could name the variable
  # with an additiona subscript _face_iminushalf, but that would be verbose.
  f1_plus = 1/3 * f_iminus3 - 7/6 * f_iminus2 + 11/6 * f_iminus1
  f2_plus = -1/6 * f_iminus2 + 5/6 * f_iminus1 + 1/3 * f
  f3_plus = 1/3 * f_iminus1 + 5/6 * f - 1/6 * f_iplus1

  f1_minus = 11/6 * f - 7/6 * f_iplus1 + 1/3 * f_iplus2
  f2_minus = 1/3 * f_iminus1 + 5/6 * f - 1/6 * f_iplus1
  f3_minus = -1/6 * f_iminus2 + 5/6 * f_iminus1 + 1/3 * f
  return f1_plus, f2_plus, f3_plus, f1_minus, f2_minus, f3_minus


def weno5_node_to_face_for_rrtmgp(
    f_node: Array,
    dim: Literal[0, 1, 2],
    f_lower_bc: Array | None = None,
    neumann_upper_bc: bool = False,
) -> tuple[Array, Array]:
  """Perform WENO5-JS interpolation from nodes to faces.

  * An array evaluated on nodes has index i <==> coordinate location x_i
  * An array evaluated on faces has index i <==> coordinate location x_{i-1/2}

  See also QUICK interpolation in convection.py.

  When dealing with the boundaries, for now we are USING the values in the halos
  (halo width = 1) as part of the process, and that the values in the halos are
  set appropriately.  This is not ideal, and is inconsistent with the rest of
  the code (which does not use halos values).  We should get rid of the use of
  halo values later.

  Refs: Jiang and Shu, "Efficient Implementation of Weighted ENO Schemes",
    JCP 126, 202-228 (1996).

  Args:
    f_node: A 3D array, evaluated on nodes.
    dim: The dimension along with the interpolation is performed.
    f_lower_bc: If not None, this value is used as the boundary condition for f
      on the lower face (the wall).  This should be a 2D array.
    neumann_upper_bc: If True, then a Neumann BC is used for f on the upper
      face.

  Returns:
    A tuple of two 3D arrays interpolated from `f_node`, which is evaluted on
    faces in dimension `dim`. The first array is the "plus" (left-biased)
    interpolation, and the second array is the "minus" (right-biased)
    interpolation.
  """
  if f_lower_bc is not None and neumann_upper_bc:
    # We have walls on both faces of the domain.
    wall_bc = True
  else:
    # We don't have walls on both faces; revert to periodic treatment.
    wall_bc = False

  w1_plus, w2_plus, w3_plus, w1_minus, w2_minus, w3_minus = (
      _weno5_nonlinear_weights(f_node, dim, wall_bc)
  )
  # Get various local reconstructions of f on the face i - 1/2.
  f1_plus, f2_plus, f3_plus, f1_minus, f2_minus, f3_minus = (
      _weno5_local_reconstructions(f_node, dim)
  )
  # Obtain the WENO reconstruction by combining the nonlinear weights with the
  # local reconstructions on the faces.
  f_face_plus = w1_plus * f1_plus + w2_plus * f2_plus + w3_plus * f3_plus
  f_face_minus = w1_minus * f1_minus + w2_minus * f2_minus + w3_minus * f3_minus

  # Deal with boundaries.  Here, assume a possible Dirichlet lower BC and a
  # Neumann upper BC.
  # These two `if`s only deal with the face value on the halos, not interior
  # faces.  Note: we really should be assigning the lower BC to the wall-face,
  # not a halo node, but let's fix that up later.
  hw = 1  # Assumed halo width.
  if f_lower_bc is not None:
    # Assign value in the halo ...
    f_face_plus = f_face_plus.at[:, :, 0].set(f_lower_bc)
    f_face_minus = f_face_minus.at[:, :, 0].set(f_lower_bc)

  if neumann_upper_bc:
    f_face_minus = f_face_minus.at[:, :, -hw].set(f_node[:, :, -hw - 1])

  return f_face_plus, f_face_minus
