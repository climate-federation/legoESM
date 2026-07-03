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

"""Utility library for common operations on the RRTMGP tables."""

import collections
from collections.abc import Sequence
import dataclasses
import inspect
import itertools
import string
from typing import Callable, TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array


@dataclasses.dataclass
class IndexAndWeight:
  """Wrapper for a pair of index tensor and associated interpolation weight."""

  # Index tensor.
  idx: Array
  # Interpolant weight tensor associated with index.
  weight: Array


@dataclasses.dataclass
class Interpolant:
  """Wrapper for a single dimension interpolant."""

  # Index and interpolation weight of floor reference value.
  interp_low: IndexAndWeight
  # Index and interpolation weight of upper endpoint of matching reference
  # interval.
  interp_high: IndexAndWeight


def _einsum_expression_from_lookup_table(table: Array):
  """Return an einsum expression for performing matmul on a lookup table."""
  rank = table.ndim
  eq_idx = ''
  eq_tb = '...'
  for i in range(rank):
    dim_var = string.ascii_lowercase[i]
    eq_idx += f'...{dim_var},'
    eq_tb += f'{dim_var}'
  return eq_idx + eq_tb + '->...'


def lookup_values(vals: Array, idx_list: Sequence[Array]) -> Array:
  """Gather values from `vals` as specified by a list of index arrays.

  On TPU the one-hot einsum path leverages the MXU; on CPU/GPU direct
  indexing is faster and uses less memory.

  Args:
    vals: An array of coefficients to be gathered.
    idx_list: A list of length equal to the rank of `vals` containing arrays of
      indices, one for each axis of `vals` and in the same order.

  Returns:
    An array having the same shape as an element of `idx_list` where the indices
    have been replaced by the corresponding value from `vals`.
  """
  backend = jax.default_backend().lower()
  if backend == "tpu":
      # One-hot einsum: avoids slow `gather` on TPU by using the MXU.
      eq = _einsum_expression_from_lookup_table(vals)
      inputs = [
          jax.nn.one_hot(idx, vals.shape[i], dtype=vals.dtype)
          for i, idx in enumerate(idx_list)
      ]
      inputs.append(vals)
      return jnp.einsum(eq, *inputs)
  # CPU/GPU: direct indexing is faster and more memory-efficient.
  return lookup_values_direct_indexing(vals, idx_list)


def lookup_values_direct_indexing(vals, idx_list: Sequence[Array]) -> Array:
  """Gather values from `vals` as specified by a list of index arrays.

  This function implements the same lookup as `lookup_values` but does so via
  direct indexing into the array, rather than using a one-hot vector and
  matrix multiplication.  It serves as a reference implementation.

  Args:
    vals: An array of coefficients to be gathered.
    idx_list: A list of length equal to the rank of `vals` containing arrays of
      indices, one for each axis of `vals` and in the same order.

  Returns:
    An array having the same shape as an element of `idx_list` where the indices
    have been replaced by the corresponding value from `vals`.
  """
  # Convert idx_list to a tuple for advanced indexing (list is deprecated).
  return vals[tuple(idx_list)]


def _one_hot_weighted(
    weighted_idx: IndexAndWeight, vals: Array, axis: int
) -> Array:
  """Convert an instance of IndexAndWeight to a vector representation."""
  return weighted_idx.weight[..., jnp.newaxis] * jax.nn.one_hot(
      weighted_idx.idx, vals.shape[axis], dtype=vals.dtype
  )


def _axis_endpoints(
    entry: Interpolant | IndexAndWeight,
) -> list[IndexAndWeight]:
  """Return the 1 (IndexAndWeight) or 2 (Interpolant) endpoints for one axis.

  Mirrors the per-axis branching in :func:`combine_linearly`: an
  ``IndexAndWeight`` contributes a single weighted endpoint; an ``Interpolant``
  contributes its low and high endpoints (the one-hot path encodes both inside
  one vector via ``low_endpoint + high_endpoint``).
  """
  if isinstance(entry, IndexAndWeight):
    return [entry]
  return [entry.interp_low, entry.interp_high]


def combine_linearly_gather(
    vals: Array,
    weight_idx_list: Sequence[Interpolant | IndexAndWeight],
) -> Array:
  """Gather-based equivalent of :func:`combine_linearly` for CPU/GPU.

  Algebraically identical to the one-hot+einsum path: the one-hot einsum
  evaluates ``sum over the Cartesian product of per-axis endpoints of
  (product of endpoint weights) * vals[endpoint indices]`` — exactly the
  multilinear-interpolation corner sum.  This re-expresses that sum directly:
  one advanced-index gather per corner (reusing
  :func:`lookup_values_direct_indexing`, so the gather numerics are NOT
  duplicated) scaled by the product of that corner's endpoint weights.

  Why it helps reverse-mode AD: the integer indices carry no gradient
  (``stop_gradient`` + ``floor`` is piecewise-constant — identical to the
  one-hot path, where ``one_hot(idx)`` is also constant in the continuous
  inputs), so only the float weights backprop.  The VJP is therefore a few
  ``scatter-add`` corners instead of the DENSE one-hot einsum VJP over the
  full reference axes (~``n_temperature`` x ``n_pressure`` x
  ``n_mixing_fraction``), which is what blows up the radiation backward graph.

  Numerics: the corner sum is the SAME products of the SAME weights and the
  SAME table entries as the einsum; the two differ ONLY in floating-point
  summation association over the (<= 2**n_interpolant-axis, typically 2-8)
  corners, i.e. a few ULP — no change to signs, the weights-sum-to-one
  interpolation invariant, or conservation.

  Out-of-bounds: ``jax.nn.one_hot(idx, n)`` returns an all-zero vector for an
  index outside ``[0, n)`` — that corner contributes ZERO — whereas an
  advanced-index gather CLAMPS the index to the edge.  Indices here are almost
  always in-bounds (``floor_idx`` clips to ``[0, size-1]`` and
  ``idx_high = min(idx_low+1, size-1)``), but a ``create_linear_interpolant``
  ``offset`` (the gas_optics troposphere split) can shift an index past the
  table edge.  Each corner is therefore masked to zero when ANY axis index is
  out of bounds, so the gather path is bit-identical to the one-hot path even
  in that case — no reliance on an unverified table-sizing invariant.
  """
  per_axis_endpoints = [_axis_endpoints(e) for e in weight_idx_list]
  result = None
  for combo in itertools.product(*per_axis_endpoints):
    # `combo` is one interpolation corner: an IndexAndWeight per axis, in
    # axis order.  Integer indices are stop_gradient (no continuous grad).
    idx_list = [jax.lax.stop_gradient(c.idx) for c in combo]
    gathered = lookup_values_direct_indexing(vals, idx_list)
    weight = combo[0].weight
    for c in combo[1:]:
      weight = weight * c.weight
    # Zero-on-OOB to match one_hot (clamp-on-OOB gather would diverge).
    in_bounds = None
    for axis, idx in enumerate(idx_list):
      ib = (idx >= 0) & (idx < vals.shape[axis])
      in_bounds = ib if in_bounds is None else (in_bounds & ib)
    term = weight * gathered
    if in_bounds is not None:
      term = jnp.where(in_bounds, term, jnp.zeros_like(term))
    result = term if result is None else result + term
  return result


def combine_linearly(
    vals: Array,
    weight_idx_list: Sequence[Interpolant | IndexAndWeight],
) -> Array:
  """Perform a linear combination on `vals` according to `weight_idx_list`.

  Used for the optimized lookup.  On TPU the one-hot + einsum path keeps the
  contraction on the MXU; on CPU/GPU it dispatches to
  :func:`combine_linearly_gather`, which is far cheaper to compile and yields a
  SPARSE reverse-mode VJP (mirrors the existing backend switch in
  :func:`lookup_values`).

  Args:
    vals: The array of coefficients that will be combined.
    weight_idx_list: A list of `IndexAndWeight`s containing a pair of index
      array and weight array for each axis of the `coeffs` array.

  Returns:
    An array of the same shape as an element of `weight_idx_list` containing
    the gathered coefficients scaled by the pointwise product of associated
    weights.
  """
  if jax.default_backend().lower() != "tpu":
    # CPU/GPU: gather corner sum — smaller XLA graph + sparse VJP.
    return combine_linearly_gather(vals, weight_idx_list)
  # TPU: one-hot + einsum keeps the contraction on the MXU.
  eq = _einsum_expression_from_lookup_table(vals)
  inputs = []
  for i, idx in enumerate(weight_idx_list):
    if isinstance(idx, IndexAndWeight):
      inputs.append(_one_hot_weighted(idx, vals, i))
    else:
      low_endpoint = _one_hot_weighted(idx.interp_low, vals, i)
      high_endpoint = _one_hot_weighted(idx.interp_high, vals, i)
      inputs.append(low_endpoint + high_endpoint)
  inputs.append(vals)
  return jnp.einsum(eq, *inputs)


def floor_idx(f: Array, reference_values: Array) -> Array:
  """Return the indices of the floor reference values.

  Note that the `reference_values` should consist of evenly spaced points. Each
  index returned corresponds to the highest index k such that
  reference_values[k] <= f.

  Args:
    f: The array whose values will be mapped to a floor reference value.
    reference_values: A 1D array of reference values.

  Returns:
    An `Array` of the same shape as `f` containing the indices of the floor
    reference values for the values of `f`. Each index corresponds to the
    highest index k such that reference_values[k] <= f.
  """
  delta = reference_values[1] - reference_values[0]
  size = reference_values.shape[0]
  truncated_div = jnp.floor_divide(f - reference_values[0], delta)
  truncated_div = truncated_div.astype(jnp.int_)
  return jnp.clip(truncated_div, 0, size - 1)


def create_linear_interpolant(
    f: Array, f_ref: Array, offset: Array | None = None
) -> Interpolant:
  """Create a linear interpolant based on the evenly spaced reference values.

  The linear interpolant is created by matching the values of `f` to an interval
  of the reference values and storing information about the location of the
  endpoints. Linear interpolation weights are computed based on the distance of
  the value from each endpoint.

  Args:
    f: A tensor of arbitrary shape whose values must be in the range of
      reference values in `f_ref`.
    f_ref: The 1-D tensor of evenly spaced reference values for the variable.
    offset: An optional tensor of the same shape as `f` that should be added to
      the interpolant indices.

  Returns:
    An `Interpolant` object containing the pointwise floor and ceiling indices
    and interpolation weights of `f`.
  """
  size = f_ref.shape[0]
  delta = f_ref[1] - f_ref[0]
  idx_low = floor_idx(f, f_ref)
  idx_high = jnp.minimum(idx_low + 1, size - 1)
  # Compute the interpolant weights for the two endpoints.
  lower_reference_vals = f_ref[0] + delta * idx_low.astype(f_ref.dtype)
  weight2 = jnp.abs((f - lower_reference_vals) / delta)
  weight1 = 1.0 - weight2
  if offset is not None:
    idx_low += offset
    idx_high += offset
  idx_weight_low = IndexAndWeight(idx_low, weight1)
  idx_weight_high = IndexAndWeight(idx_high, weight2)
  return Interpolant(idx_weight_low, idx_weight_high)


def interpolate_optimized(
    coeffs: Array,
    interpolant_fns: collections.OrderedDict[str, Callable[..., Interpolant]],
) -> Array:
  """Interpolate coefficients linearly according to the `interpolant_fns`.

  Multi-dimensional linear interpolation built from a chain of weighted
  one-hot matmul operations encoded as a single `einsum` per partial
  graph.  Topologically-ordered `interpolant_fns` declare both
  independent and dependency-bearing variables; the latter cause the
  graph to branch into separate upper/lower-endpoint subtrees that are
  evaluated in parallel and reduced with `jax.tree.reduce(jnp.add, ...)`
  at the end.

  Args:
    coeffs: The array of coefficients of arbitrary shape whose values will be
      interpolated.
    interpolant_fns: An ordered dictionary of interpolant functions keyed by the
      name of the variable they correspond to. There should be one for each axis
      of `coeffs` and their order should match the order of the axes. Note that
      they should be sorted in topological order (dependent indices appearing
      after the indices they depend on). The axes of `coeffs` are assumed to
      already conform to this ordering.

  Returns:
    An `Array` of the same shape as any of the index arrays, but with the
    indices replaced by the interpolated coefficients.
  """
  # Set of interpolation variables that are depended on by other variables.
  dependency_vars = set()
  # Mapping of interpolation variable name to a list of its dependencies' names.
  dependency_args = {}
  # Initial pass-through over all interpolation variables to determine
  # dependencies between them.
  for k, interpolant_fn in interpolant_fns.items():
    interp_args = inspect.getfullargspec(interpolant_fn).args
    dependency_vars.update(interp_args)
    dependency_args[k] = interp_args

  # `interpolation_subtrees` refers to the partial interpolation graphs that can
  # be computed with a chain of matmul operations in a single `einsum` op.
  interpolation_subtrees = [collections.OrderedDict()]

  for varname, interpolant_fn in interpolant_fns.items():
    for interpolant_dict in list(interpolation_subtrees):
      # Get all the predecessor indices that the current variable `varname`
      # depends on.
      interpolant_fn_kwargs = {
          k: v
          for k, v in interpolant_dict.items()
          if k in dependency_args[varname]
      }
      interpolant = interpolant_fn(**interpolant_fn_kwargs)
      # If `varname` is a variable that other interpolation variables depend on,
      # branch out into separate graphs for the upper and lower endpoints.
      if varname in dependency_vars:
        interpolant_dict_high = interpolant_dict.copy()
        interpolant_dict_high[varname] = interpolant.interp_high
        interpolation_subtrees.append(interpolant_dict_high)
        # Reuse existing dictionary for the lower index subgraph.
        interpolant_dict[varname] = interpolant.interp_low
      else:
        # If no other interpolant depends on this variable, both endpoints of
        # interpolation interval can be encoded in the same matmul operation.
        interpolant_dict[varname] = interpolant

  partial_interpolated_values = [
      combine_linearly(coeffs, list(x.values())) for x in interpolation_subtrees
  ]
  weighted_sum = jax.tree.reduce(jnp.add, partial_interpolated_values)
  return weighted_sum


def interpolate(
    coeffs: Array,
    interpolant_fns: collections.OrderedDict[str, Callable[..., Interpolant]],
) -> Array:
  """Interpolate coefficients linearly according to the `interpolant_fns`.

  The `interpolant_fns` are provided as functions taking in a dictionary of
  `IndexAndWeight` objects that the interpolant might depend on and returning
  an `Interpolant` object. This is particularly useful when the variables used
  to index into `coeffs` have dependencies between them. Take as an example the
  RRTMGP `kmajor` coefficient table, which is indexed by temperature, pressure,
  and the relative abundance fraction. Temperature and pressure are independent
  variables, but the relative abundance calculation depends on both of them.
  Providing the interpolant as a function allows a straightforward expression of
  this dependency.

  For example, if `coeffs` has rank 2, where the first axis is indexed by a
  variable `t`, the second axis is indexed by a variable `s`, and `s` depends on
  `t`, the interpolation can be done as follows:

  independent_t_interpolant = create_linear_interpolant(t, t_ref)

  def dependent_s_interpolant_func(dep: Dict[Text, IndexAndWeight]):
    t_idx = dep['t'].idx
    s = compute_s(t_idx, ...)
    return create_linear_interpolant(s, s_ref)

  interpolant_fns = {
      't': lambda: independent_x_interp,
      's': dependent_s_interpolant_func,
  }

  interpolated_vals = interpolate(coeffs, interpolant_fns)

  Args:
    coeffs: The array of coefficients of arbitrary shape whose values will be
      interpolated.
    interpolant_fns: An ordered dictionary of interpolant functions keyed by the
      name of the variable they correspond to. There should be one for each axis
      of `coeffs` and their order should match the order of the axes. Note that
      they should be sorted in topological order (dependent indices appearing
      after the indices they depend on). The axes of `coeffs` are assumed to
      already conform to this ordering.

  Returns:
    An `Array` of the same shape as any of the index arrays, but with the
    indices replaced by the interpolated coefficients.
  """
  return interpolate_optimized(coeffs, interpolant_fns)


