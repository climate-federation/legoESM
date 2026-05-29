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
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array
#
# iter-45 dead-code trim: kept only the WENO5-for-RRTMGP path used by
# optics_base.reconstruct_face_values.  The original swirl_jatmos
# file shipped ~520 lines of general-purpose interpolation utilities
# (centered, WENO3-JS, WENO5-JS/Z) that had zero callers in legoESM.
#

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