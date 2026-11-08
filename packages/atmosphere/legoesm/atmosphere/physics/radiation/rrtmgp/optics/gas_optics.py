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

"""Utility functions for computing optical properties of atmospheric gases."""

import collections
import math
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_gas_optics_base
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_gas_optics_longwave
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_gas_optics_shortwave
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_volume_mixing_ratio
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import optics_utils
from legoesm.core.smooth import positive_part_c1, smoothstep_cubic


Array: TypeAlias = jax.Array
Interpolant: TypeAlias = optics_utils.Interpolant
# pylint: disable=line-too-long
AbstractLookupGasOptics: TypeAlias = (
    lookup_gas_optics_base.AbstractLookupGasOptics
)
LookupGasOpticsLongwave: TypeAlias = (
    lookup_gas_optics_longwave.LookupGasOpticsLongwave
)
LookupGasOpticsShortwave: TypeAlias = (
    lookup_gas_optics_shortwave.LookupGasOpticsShortwave
)
LookupVolumeMixingRatio: TypeAlias = (
    lookup_volume_mixing_ratio.LookupVolumeMixingRatio
)
# pylint: enable=line-too-long

_PASCAL_TO_HPASCAL_FACTOR = 0.01
_M2_TO_CM2_FACTOR = 1e4
# Safe-division floor for the binary-species fraction eta = vmr_1 / (vmr_1 +
# r * vmr_2).  The upstream Fortran uses a Python-equivalent ``if mix > 2*tiny
# else 0.5`` branch; in JAX we use ``jnp.where`` plus a safe denominator so
# reverse-mode AD does not propagate NaN gradients through the dead branch.
_VMR_SAFE_DIV_EPS = 1.0e-30


def _clip_to_table_range(temperature: Array, t_ref: Array,
                         sharpness: float = math.inf) -> Array:
  """Clip temperature into the lookup-table reference range.

  RRTMGP tables (``kmajor``, ``kminor_*``, ``krayl_*``,
  ``planck_fraction``, ``totplnk``) are defined on
  ``[t_ref[0], t_ref[-1]]`` (typically 160 K–355 K).  ``floor_idx`` clips
  the integer index but ``create_linear_interpolant`` lets the weight
  exceed [0, 1], producing linear *extrapolation* outside the table.
  For the Planck source ``totplnk(T) ∝ T**4`` this is catastrophic at
  cold mesospheric or top-halo temperatures.

  Upstream Fortran (mo_gas_optics_rrtmgp.F90 lines 300/303/519) raises
  an error when any tlay/tlev/tsfc lies outside the range.  legoESM
  must remain differentiable, so we clamp the temperature into the
  table range before interpolation.  Halo cells produced by linear
  extrapolation (``2*T[-1]-T[-2]``) are the most common offender; they
  are stripped from the returned fluxes anyway.

  ``sharpness`` (static; ``math.inf`` = the hard clip, bitwise): a finite
  value smooths each end with a C1 band of width |bound|/sharpness on the
  INSIDE of the table (the interpolant mirrors below the first point and is
  flat above the last), exact for T in [t0 + t0/s, tN - tN/s].
  """
  if sharpness == math.inf:
    return jnp.clip(temperature, t_ref[0], t_ref[-1])
  lo, hi = t_ref[0], t_ref[-1]
  u = (temperature - lo) / lo                 # exactly T where u >= 1/s
  t = jnp.where(u * sharpness >= 1.0, temperature,
                lo + lo * positive_part_c1(u, sharpness))
  v = (hi - t) / hi
  return jnp.where(v * sharpness >= 1.0, t, hi - hi * positive_part_c1(v, sharpness))


def _pressure_interpolant(
    p: Array, p_ref: Array, troposphere_offset: Array | None = None
) -> Interpolant:
  """Create a pressure interpolant based on reference pressure values."""
  log_p = jnp.log(p)
  log_p_ref = jnp.log(p_ref)
  return optics_utils.create_linear_interpolant(
      log_p, log_p_ref, offset=troposphere_offset
  )


def _branch_pressure_interpolant(p: Array, p_ref: Array, p_tropo, tropo: int) -> Interpolant:
  """Pressure interpolant of ONE side of the tropopause split, extended past it.

  As ``_pressure_interpolant`` with offset ``tropo`` (identical bits for cells on
  that side), but the interval is held to the side's own half of the table:
  the lower branch keeps its last interval ending at ``p_tropo`` for p < p_tropo,
  the upper branch its first interval starting there for p > p_tropo, and the
  weights of that extension are signed (linear extrapolation;
  ``create_linear_interpolant``'s ``abs`` would mirror).  Used only by the
  smoothed split (vote J1).
  """
  log_p, log_p_ref = jnp.log(p), jnp.log(p_ref)
  size = log_p_ref.shape[0]
  delta = log_p_ref[1] - log_p_ref[0]
  k_tropo = optics_utils.floor_idx(jnp.log(p_tropo) + 0.5 * delta, log_p_ref)
  own = optics_utils.floor_idx(log_p, log_p_ref)
  if tropo == 0:
    idx_low = jnp.minimum(own, k_tropo - 1)
  else:
    idx_low = jnp.maximum(own, k_tropo)
  idx_high = jnp.minimum(idx_low + 1, size - 1)
  lower_reference_vals = log_p_ref[0] + delta * idx_low.astype(log_p_ref.dtype)
  weight2 = (log_p - lower_reference_vals) / delta
  # signed only where the branch is extended past the tropopause; elsewhere the
  # original abs (incl. its behaviour beyond the table ends), so own-side cells
  # are bit-identical to ``_pressure_interpolant``.
  weight2 = jnp.where(idx_low == own, jnp.abs(weight2), weight2)
  return Interpolant(optics_utils.IndexAndWeight(idx_low + tropo, 1.0 - weight2),
                     optics_utils.IndexAndWeight(idx_high + tropo, weight2))


def _mixing_fraction_interpolant(
    f: Array, n_mixing_fraction: int
) -> Interpolant:
  """Create a mixing fraction interpolant based on desired number of points."""
  # Build the reference grid in the working dtype taken from ``f`` (the
  # relative-abundance field).  Using ``jnp.float_`` here forced float64 even
  # when ``f`` is float32 (``compute_fp32`` path): ``create_linear_interpolant``
  # mixes ``f_ref`` into the interpolation weights, so a float64 grid
  # re-promoted the whole optical-depth lookup to float64 and broke the fp32
  # scan-carry invariant downstream.  ``f.dtype`` is float64 on the default
  # path -> byte-identical there.
  return optics_utils.create_linear_interpolant(
      f, jnp.linspace(0.0, 1.0, n_mixing_fraction, dtype=f.dtype)
  )


def get_vmr(
    lookup_gas_optics: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    species_idx: Array,
    vmr_fields: dict[int, Array] | None = None,
) -> Array:
  """Get the volume mixing ratio, given major gas species index and pressure.

  Args:
    lookup_gas_optics: An `AbstractLookupGasOptics` object containing an index
      for the gas species.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    species_idx: An `Array` containing indices of gas species whose VMR will be
      computed.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    An `Array` of the same shape as `species_idx` or `pressure_idx` containing
    the pointwise volume mixing ratios of the corresponding gas species at that
    pressure level.
  """
  idx_gases = lookup_gas_optics.idx_gases
  # Indices of background gases for which a global mean VMR is available.
  vmr_gm = [0.0] * len(idx_gases)
  # Map the gas names in `vmr_lib.global_means` dict to indices consistent with
  # the RRTMGP `key_species` table.
  for k, v in vmr_lib.global_means.items():
    vmr_gm[idx_gases[k]] = v

  # Stack the global-mean VMRs in the optics tables' working dtype rather than
  # ``jnp.float_``.  ``vmr`` multiplies into the optical depth, so forcing
  # float64 here re-promoted the fp32 optics path back to float64 (and broke
  # the float32 scan carry in the RTE solve).  ``vmr_ref`` carries the table
  # dtype (float32 after ``_cast_optics_f64_to_f32``, float64 otherwise), so
  # the default path stays byte-identical.
  vmr = optics_utils.lookup_values(
      jnp.stack(vmr_gm, dtype=lookup_gas_optics.vmr_ref.dtype), (species_idx,)
  )

  # Overwrite with available precomputed vmr.
  if vmr_fields is not None:
    for gas_idx, vmr_field in vmr_fields.items():
      vmr = jnp.where(species_idx == gas_idx, vmr_field, vmr)

  # Note: Skipping these checks for now. Cannot have these boolean checks inside
  # of a traced function.
  # if jnp.any(vmr < 0.0):
  #   raise ValueError('At least one volume mixing ratio (VMR) is negative.')
  # if jnp.any(vmr > 1.0):
  #   raise ValueError('At least one volume mixing ratio (VMR) is above 1.')

  return vmr


def _compute_relative_abundance_interpolant(
    lookup_gas_optics: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    troposphere_idx: Array,
    temperature_idx: Array,
    ibnd: Array,
    scale_by_mixture: bool,
    vmr_fields: dict[int, Array] | None = None,
) -> Interpolant:
  """Create an `Interpolant` object for relative abundance of a major species.

  Args:
    lookup_gas_optics: An `AbstractLookupGasOptics` object containing a RRTMGP
      index for all relevant gas species.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    troposphere_idx: An `Array` that is 1 where the corresponding pressure level
      is below the troposphere limit and 0 otherwise. This informs whether an
      offset should be added to the reference pressure indices when indexing
      into the `kmajor` table.
    temperature_idx: An `Array` containing indices of reference temperature
      values.
    ibnd: The frequency band for which the relative abundance is computed.
    scale_by_mixture: Whether to scale the weights by the gas mixture.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    An `Interpolant` object for the relative abundance of the major gas species
      in a particular electromagnetic frequency band.
  """
  major_species_idx = []
  vmr_for_interp = []
  vmr_ref = []
  for i in range(2):
    major_species_idx.append(
        optics_utils.lookup_values(
            lookup_gas_optics.key_species[ibnd, :, i], (troposphere_idx,)
        )
    )
    vmr_for_interp.append(
        get_vmr(lookup_gas_optics, vmr_lib, major_species_idx[i], vmr_fields)
    )
    vmr_ref.append(
        optics_utils.lookup_values(
            lookup_gas_optics.vmr_ref,
            (temperature_idx, major_species_idx[i], troposphere_idx),
        )
    )
  # Defensive safe-divide for ``vmr_ref[1]`` (the second major species's
  # reference VMR).  All shipped rte-rrtmgp gas-optics tables have positive
  # ``vmr_ref`` entries (verified: dry air = 1.0, all real gases ≥ 1e-10),
  # but a future table or a manually-constructed lookup might have a zero
  # entry that would otherwise produce ``inf/NaN`` in this ratio and
  # propagate through the reverse-mode AD path.  See iter-5 commit for the
  # analogous fix on ``combined_vmr``.
  # coeff-ok: floor never hit — shipped tables have vmr_ref >= 1e-10 >> eps.
  vmr_ref_ratio = vmr_ref[0] / jnp.maximum(vmr_ref[1], _VMR_SAFE_DIV_EPS)
  combined_vmr = vmr_for_interp[0] + vmr_ref_ratio * vmr_for_interp[1]
  # Consistent with how the RRTM absorption coefficient tables are designed, the
  # relative abundance defaults to 0.5 when the volume mixing ratio of both
  # dominant species is exactly 0.  Use a safe denominator (``jnp.maximum``
  # against ``_VMR_SAFE_DIV_EPS``) so reverse-mode AD does not propagate NaN
  # gradients through the dead ``where`` branch when ``combined_vmr == 0``.
  #
  # **Use the same eps for the where condition** (``combined_vmr > eps``,
  # not ``> 0``): if combined_vmr is in (0, eps] the division
  # ``vmr_for_interp[0] / max(combined_vmr, eps) == vmr / eps`` and its
  # gradient ``1/eps ~ 1e30`` blow up.  Selecting the 0.5 fallback in
  # that regime keeps both forward and backward bounded.  This matches
  # the upstream Fortran branch ``col_mix > 2 * tiny(col_mix)``.
  safe_combined = jnp.maximum(combined_vmr, _VMR_SAFE_DIV_EPS)
  relative_abundance = jnp.where(
      combined_vmr > _VMR_SAFE_DIV_EPS,
      vmr_for_interp[0] / safe_combined,
      0.5,
  )
  interpolant = _mixing_fraction_interpolant(
      relative_abundance, lookup_gas_optics.n_mixing_fraction
  )
  if scale_by_mixture:
    interpolant.interp_low.weight *= combined_vmr
    interpolant.interp_high.weight *= combined_vmr
  return interpolant


def compute_planck_sources(
    lookup: LookupGasOpticsLongwave,
    planck_fraction: Array,
    temperature: Array,
    igpt: Array,
    sharpness: float = math.inf,
) -> Array:
  """Computes the Planck source for the longwave problem.

  Args:
    lookup: An `LookupGasOpticsLongwave` object containing a RRTMGP index for
      all relevant gases and a lookup table for the Planck source.
    planck_fraction: The Planck fraction that scales the Planck source.
    temperature: The temperature [K] for which the Planck source will be
      computed.
    igpt: The absorption rank (g-point) index for which the optical depth will
      be computed.

  Returns:
    The planck source emanating from the points with given `temperature` [W/m²].
  """
  ibnd = lookup.g_point_to_bnd[igpt]

  # 1-D interpolation of the Planck source.  Clip temperature to the
  # ``t_planck`` table range — extrapolation here is **catastrophic**
  # because ``totplnk(T) ∝ T**4`` and the linear interpolant produces
  # ``weight1*table[0] + weight2*table[1]`` with ``weight1 = 1 -
  # weight2``; for cold halo/mesospheric temperatures ``weight2 ≫ 1``
  # and ``weight1 < 0`` gives nonsense.  Upstream errors out on
  # out-of-range T; we clamp instead to stay differentiable.
  t_for_planck = _clip_to_table_range(temperature, lookup.t_planck, sharpness)
  interpolant = optics_utils.create_linear_interpolant(
      t_for_planck, lookup.t_planck
  )
  return planck_fraction * optics_utils.interpolate(
      jax.lax.optimization_barrier(lookup.totplnk)[ibnd, :],
      collections.OrderedDict({'t': lambda: interpolant}),
  )


# Every g-point is evaluated in one pass: the g-point-independent interpolation
# runs once per cell and every table read is a row gather over the g-point axis,
# as in the RRTMGP GPU kernels and jax-rrtmgp 537d97e.  ``optimization_barrier``
# on a table keeps XLA from constant-folding its reads into the kernels, which
# overflows the CPU LLVM-JIT code region and makes compiles pathologically slow.
_SIDES = ('interp_high', 'interp_low')


class _Cells:
  """Per-cell interpolation state shared by every g-point."""

  def __init__(self, lookup, vmr_lib, temperature, p, vmr_fields,
               sharpness=math.inf, tropo=None):
    """``tropo`` (0 lower, 1 upper) forces one branch of the tropopause split
    everywhere; ``None`` takes p <= p_ref_tropo per cell (the oracle)."""
    self.lookup, self.vmr_lib, self.vmr_fields = lookup, vmr_lib, vmr_fields
    self.temperature, self.p = temperature, p
    if tropo is None:
      self.tropo_idx = jnp.where(p <= lookup.p_ref_tropo, 1, 0)
    else:
      assert tropo in (0, 1), tropo
      self.tropo_idx = jnp.full(p.shape, tropo, dtype=jnp.int32)
    self.t_interp = optics_utils.create_linear_interpolant(
        _clip_to_table_range(temperature, lookup.t_ref, sharpness), lookup.t_ref)
    if tropo is None:
      self.p_interp = _pressure_interpolant(p, lookup.p_ref, self.tropo_idx)
    else:
      self.p_interp = _branch_pressure_interpolant(p, lookup.p_ref, lookup.p_ref_tropo, tropo)
    self._mix = {}

  def mix(self, ibnd: int, scale_by_mixture: bool) -> dict[str, Interpolant]:
    """Relative-abundance interpolant at each temperature corner of band ``ibnd``."""
    key = (tuple(np.asarray(self.lookup.key_species)[ibnd].ravel().tolist()),
           scale_by_mixture)
    if key not in self._mix:
      self._mix[key] = {
          side: _compute_relative_abundance_interpolant(
              self.lookup, self.vmr_lib, self.tropo_idx,
              getattr(self.t_interp, side).idx, ibnd, scale_by_mixture,
              self.vmr_fields)
          for side in _SIDES
      }
    return self._mix[key]


def _rows_tpm(table: Array, cells: _Cells, mix: dict[str, Interpolant]) -> Array:
  """Eight-corner interpolation of a ``(t, p, eta, n)`` table, ``(*cells, n)``."""
  total = 0.0
  for t_side in _SIDES:
    t = getattr(cells.t_interp, t_side)
    for p_side in _SIDES:
      pp = getattr(cells.p_interp, p_side)
      for m_side in _SIDES:
        m = getattr(mix[t_side], m_side)
        total = total + table[t.idx, pp.idx, m.idx] * (
            t.weight * pp.weight * m.weight)[..., None]
  return total


def _rows_te(table: Array, cells: _Cells, mix: dict[str, Interpolant]) -> Array:
  """Four-corner interpolation of a ``(t, eta, n)`` table, ``(*cells, n)``."""
  total = 0.0
  for t_side in _SIDES:
    t = getattr(cells.t_interp, t_side)
    for m_side in _SIDES:
      m = getattr(mix[t_side], m_side)
      total = total + table[t.idx, m.idx] * (t.weight * m.weight)[..., None]
  return total


def _band_groups(lookup) -> list[tuple[list[int], np.ndarray]]:
  """Bands sharing a key-species pair, hence interpolants, with their g-points."""
  key_species = np.asarray(lookup.key_species)
  lims = np.asarray(lookup.bnd_lims_gpt)
  by_pair: dict[tuple[int, ...], list[int]] = {}
  for ibnd in range(key_species.shape[0]):
    by_pair.setdefault(tuple(key_species[ibnd].ravel().tolist()), []).append(ibnd)
  return [
      (bands, np.concatenate(
          [np.arange(lims[b, 0], lims[b, 1] + 1) for b in bands]))
      for bands in by_pair.values()
  ]


def _in_gpt_order(pieces: list[Array], groups) -> Array:
  """Per-group ``(*cells, n)`` pieces as one ``(n_gpt, *cells)`` array."""
  order = np.concatenate([gpts for _, gpts in groups])
  return jnp.moveaxis(jnp.concatenate(pieces, axis=-1), -1, 0)[np.argsort(order)]


def _minor_band(lookup, cells: _Cells, molecules, ibnd: int) -> Array | None:
  """Minor optical depth of band ``ibnd``'s g-points; ``None`` if it has none."""
  lims = np.asarray(lookup.bnd_lims_gpt)
  width = int(lims[ibnd, 1] - lims[ibnd, 0]) + 1
  wdtype = cells.temperature.dtype
  vmr_fields = cells.vmr_fields
  if vmr_fields is not None and lookup.idx_h2o in vmr_fields:
    dry_factor = 1.0 / (1.0 + vmr_fields[lookup.idx_h2o].astype(wdtype))
  else:
    dry_factor = 1.0
  mix = cells.mix(ibnd, False)
  total = None
  for lower, table, n_minor, start, end, gases, density, scaling_gases, complement, shift in (
      (True, lookup.kminor_lower, lookup.n_minor_absrb_lower,
       lookup.minor_lower_bnd_start, lookup.minor_lower_bnd_end,
       lookup.idx_minor_gases_lower, lookup.minor_lower_scales_with_density,
       lookup.idx_scaling_gases_lower, lookup.lower_scale_by_complement,
       lookup.minor_lower_gpt_shift),
      (False, lookup.kminor_upper, lookup.n_minor_absrb_upper,
       lookup.minor_upper_bnd_start, lookup.minor_upper_bnd_end,
       lookup.idx_minor_gases_upper, lookup.minor_upper_scales_with_density,
       lookup.idx_scaling_gases_upper, lookup.upper_scale_by_complement,
       lookup.minor_upper_gpt_shift),
  ):
    first, last = int(np.asarray(start)[ibnd]), int(np.asarray(end)[ibnd])
    if first == n_minor:  # the table's empty-band sentinel
      continue
    if first < 0 or first > n_minor or last >= n_minor:
      raise ValueError(
          'minor absorber band limits must lie in [0, n) with n as the '
          'empty-band sentinel')
    table = jax.lax.optimization_barrier(table)
    side = None
    for i in range(first, last + 1):
      gas = int(np.asarray(gases)[i]) * jnp.ones_like(cells.tropo_idx)
      scaling = get_vmr(lookup, cells.vmr_lib, gas, vmr_fields).astype(wdtype)
      scaling = scaling * molecules / _M2_TO_CM2_FACTOR
      if int(np.asarray(density)[i]) == 1:
        # The physical temperature, not the clipped one: density scaling is a
        # real-physics relation outside the table, so minor optical depth does
        # not fully saturate at the table boundary.
        scaling = scaling * (_PASCAL_TO_HPASCAL_FACTOR * cells.p / cells.temperature)
        sgas = int(np.asarray(scaling_gases)[i])
        if sgas > 0:
          vmr = get_vmr(lookup, cells.vmr_lib, sgas * jnp.ones_like(cells.tropo_idx),
                        vmr_fields).astype(wdtype)
          scaling = scaling * (
              1.0 - vmr * dry_factor
              if int(np.asarray(complement)[i]) == 1 else vmr * dry_factor)
      k0 = int(np.asarray(shift)[i])
      term = _rows_te(table[..., k0:k0 + width], cells, mix).astype(wdtype) * scaling[..., None]
      side = term if side is None else side + term
    if side is not None:
      in_side = cells.tropo_idx == (0 if lower else 1)    # p > p_ref_tropo for lower
      side = jnp.where(in_side[..., None], side, 0.0)
      total = side if total is None else total + side
  return total


def major_optical_depth_all_gpts(lookup, vmr_lib, molecules, temperature, p, vmr_fields=None,
                                 cells=None):
  """Major-species optical depth of every g-point, ``(n_gpt, *cells)``."""
  cells = cells or _Cells(lookup, vmr_lib, temperature, p, vmr_fields)
  kmajor = jax.lax.optimization_barrier(lookup.kmajor)
  groups = _band_groups(lookup)
  return _in_gpt_order(
      [(molecules / _M2_TO_CM2_FACTOR)[..., None]
       * _rows_tpm(kmajor[..., gpts], cells, cells.mix(bands[0], True))
       for bands, gpts in groups], groups)


def minor_optical_depth_all_gpts(lookup, vmr_lib, molecules, temperature, p, vmr_fields=None,
                                 cells=None):
  """Minor-species optical depth of every g-point, ``(n_gpt, *cells)``."""
  cells = cells or _Cells(lookup, vmr_lib, temperature, p, vmr_fields)
  lims = np.asarray(lookup.bnd_lims_gpt)
  groups = _band_groups(lookup)
  pieces = []
  for bands, _ in groups:
    minors = []
    for b in bands:
      minor = _minor_band(lookup, cells, molecules, b)
      width = int(lims[b, 1] - lims[b, 0]) + 1
      minors.append(jnp.zeros(molecules.shape + (width,), temperature.dtype)
                    if minor is None else minor)
    pieces.append(jnp.concatenate(minors, axis=-1))
  return _in_gpt_order(pieces, groups)


def _tropo_blend(lookup, p, sharpness, fn):
  """``fn(tropo)`` across the tropopause split p <= p_ref_tropo.

  ``sharpness = math.inf``: ``fn(None)``, the oracle's per-cell switch.  Finite:
  both branches evaluated everywhere and blended by a centred C1 cubic in ln p
  of width S/sharpness, S = one p_ref interval in ln p; exactly one branch
  outside the band (the split is a value jump, measured up to 100 % per
  g-point).
  """
  if sharpness == math.inf:
    return fn(None)
  log_p_ref = jnp.log(lookup.p_ref)
  scale = jnp.abs(log_p_ref[1] - log_p_ref[0])
  w = smoothstep_cubic((jnp.log(lookup.p_ref_tropo) - jnp.log(p)) * (sharpness / scale) + 0.5)
  up, lo = fn(1), fn(0)
  return jnp.where(w >= 1.0, up, jnp.where(w <= 0.0, lo, w * up + (1.0 - w) * lo))


def optical_depth_all_gpts(lookup, vmr_lib, molecules, temperature, p, vmr_fields=None,
                           sharpness=math.inf):
  """Major plus minor optical depth of every g-point, ``(n_gpt, *cells)``."""
  def fn(tropo):
    cells = _Cells(lookup, vmr_lib, temperature, p, vmr_fields, sharpness, tropo)
    args = (lookup, vmr_lib, molecules, temperature, p, vmr_fields, cells)
    return minor_optical_depth_all_gpts(*args) + major_optical_depth_all_gpts(*args)
  return _tropo_blend(lookup, p, sharpness, fn)


def planck_fraction_all_gpts(lookup, vmr_lib, p, temperature, vmr_fields=None,
                             sharpness=math.inf):
  """Planck fraction of every g-point, ``(n_gpt, *cells)``."""
  def fn(tropo):
    cells = _Cells(lookup, vmr_lib, temperature, p, vmr_fields, sharpness, tropo)
    table = jax.lax.optimization_barrier(lookup.planck_fraction)
    groups = _band_groups(lookup)
    return _in_gpt_order(
        [_rows_tpm(table[..., gpts], cells, cells.mix(bands[0], False))
         for bands, gpts in groups], groups)
  return _tropo_blend(lookup, p, sharpness, fn)


def rayleigh_optical_depth_all_gpts(lkp, vmr_lib, molecules, temperature, p, vmr_fields=None,
                                    sharpness=math.inf):
  """Rayleigh optical depth of every g-point, ``(n_gpt, *cells)``."""
  lower = jax.lax.optimization_barrier(lkp.rayl_lower)
  upper = jax.lax.optimization_barrier(lkp.rayl_upper)
  factor = 1.0
  if vmr_fields is not None and lkp.idx_h2o in vmr_fields:
    factor = 1.0 + vmr_fields[lkp.idx_h2o]
  groups = _band_groups(lkp)

  def fn(tropo):
    cells = _Cells(lkp, vmr_lib, temperature, p, vmr_fields, sharpness, tropo)
    pieces = []
    for bands, gpts in groups:
      mix = cells.mix(bands[0], False)
      tau = jnp.where((cells.tropo_idx == 1)[..., None],
                      _rows_te(upper[..., gpts], cells, mix),
                      _rows_te(lower[..., gpts], cells, mix))
      pieces.append((factor * molecules / _M2_TO_CM2_FACTOR)[..., None] * tau)
    return _in_gpt_order(pieces, groups)
  return _tropo_blend(lkp, p, sharpness, fn)
