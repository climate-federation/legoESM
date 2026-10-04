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
from typing import TypeAlias

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_gas_optics_base
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_gas_optics_longwave
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_gas_optics_shortwave
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_volume_mixing_ratio
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import optics_utils


Array: TypeAlias = jax.Array
Interpolant: TypeAlias = optics_utils.Interpolant
IndexAndWeight: TypeAlias = optics_utils.IndexAndWeight
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


def _clip_to_table_range(temperature: Array, t_ref: Array) -> Array:
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
  """
  return jnp.clip(temperature, t_ref[0], t_ref[-1])


def _pressure_interpolant(
    p: Array, p_ref: Array, troposphere_offset: Array | None = None
) -> Interpolant:
  """Create a pressure interpolant based on reference pressure values."""
  log_p = jnp.log(p)
  log_p_ref = jnp.log(p_ref)
  return optics_utils.create_linear_interpolant(
      log_p, log_p_ref, offset=troposphere_offset
  )


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


def compute_major_optical_depth(
    lookup_gas_optics: AbstractLookupGasOptics,
    vmr: LookupVolumeMixingRatio,
    molecules: Array,
    temperature: Array,
    p: Array,
    igpt: Array,
    vmr_fields: dict[int, Array] | None = None,
) -> Array:
  """Compute the optical depth contributions from major gases.

  Args:
    lookup_gas_optics: An `AbstractLookupGasOptics` object containing a RRTMGP
      index for all major gas species.
    vmr: A `LookupVolumeMixingRatio` object containing the volume mixing ratio
      of all relevant atmospheric gases.
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m^2]
    temperature: An `Array` containing temperature values (in K).
    p: An `Array` containing pressure values (in Pa).
    igpt: The absorption variable index (g-point) for which the optical depth is
      computed.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    An `Array` with the pointwise optical depth contributions from the major
    species.
  """
  # Take the troposphere limit into account when indexing into the major species
  # and absorption coefficients.
  # The troposphere index is 1 for levels above the troposphere limit and 0
  # otherwise.
  troposphere_idx = jnp.where(p <= lookup_gas_optics.p_ref_tropo, 1, 0)
  # Clip temperature to the table range before interpolation; halo cells and
  # cold mesospheric layers can sit below ``t_ref[0]=160K`` where the linear
  # interpolant otherwise extrapolates with ``weight > 1`` (see
  # ``_clip_to_table_range`` docstring).
  t_for_table = _clip_to_table_range(temperature, lookup_gas_optics.t_ref)
  t_interp = optics_utils.create_linear_interpolant(
      t_for_table, lookup_gas_optics.t_ref
  )
  p_interp = _pressure_interpolant(
      p=p, p_ref=lookup_gas_optics.p_ref, troposphere_offset=troposphere_idx
  )
  # The frequency band for which the optical depth is computed.
  ibnd = lookup_gas_optics.g_point_to_bnd[igpt]

  def mix_interpolant_fn(t: IndexAndWeight) -> Interpolant:
    """Relative abundance interpolant function that depends on temperature.

    The arg name 't' is used throughout the interpolation logic to refer to the
    temperature index variable. The relative abundance variable is the only
    indexing variable that depends on another variable used to index the lookup
    tables.

    Args:
      t: An instance of `IndexAndWeight` encapsulating a point on the uniform
      temperature grid and its interpolation weight.

    Returns:
      An instance of `Interpolant` encapsulating the linear interpolation
      interval and weights on the relative abundance uniform grid.
    """
    return _compute_relative_abundance_interpolant(
        lookup_gas_optics,
        vmr,
        troposphere_idx,
        t.idx,
        ibnd,
        scale_by_mixture=True,
        vmr_fields=vmr_fields,
    )

  # Interpolant functions ordered according to the axes in `kmajor`.
  interpolant_fn_dict = collections.OrderedDict((
      ('t', lambda: t_interp),
      ('p', lambda: p_interp),
      ('m', mix_interpolant_fn),
  ))
  return (
      molecules
      / _M2_TO_CM2_FACTOR
      * optics_utils.interpolate(
          # ``optimization_barrier`` keeps XLA from CONSTANT-FOLDING the
          # per-g-point slice/gather of the (large, compile-time-constant)
          # k-distribution table into the kernel: without it XLA bakes folded
          # table kernels that overflow the CPU LLVM-JIT executable code region
          # (``Failed to materialize symbols`` / ``Cannot allocate memory``)
          # and make GPU/TPU compile pathologically slow.  The barrier is the
          # IDENTITY at runtime (answer-preserving) and turns the table read
          # into a cheap RUNTIME gather on all backends.
          jax.lax.optimization_barrier(lookup_gas_optics.kmajor)[..., igpt],
          interpolant_fns=interpolant_fn_dict,
      )
  )


def _compute_minor_optical_depth(
    lookup: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    molecules: Array,
    temperature: Array,
    p: Array,
    igpt: Array,
    is_lower_atmosphere: bool,
    vmr_fields: dict[int, Array] | None = None,
) -> Array:
  """Compute the optical depth from minor gases given atmosphere region.

  Args:
    lookup: An `AbstractLookupGasOptics` object containing a RRTMGP index for
      all relevant gases and a lookup table for minor absorption coefficients.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m^2]
    temperature: The temperature of the flow field [K].
    p: The pressure field (in Pa).
    igpt: The absorption rank (g-point) index for which the optical depth will
      be computed.
    is_lower_atmosphere: A boolean indicating whether in the lower atmosphere.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    An `Array` with the pointwise optical depth contributions from the minor
    species.
  """
  # The troposphere index is 1 for levels above the troposphere limit and 0
  # otherwise.
  tropo_idx = jnp.where(p <= lookup.p_ref_tropo, 1, 0)
  if is_lower_atmosphere:
    minor_absorber_intervals = lookup.n_minor_absrb_lower
    minor_bnd_start = lookup.minor_lower_bnd_start
    minor_bnd_end = lookup.minor_lower_bnd_end
    idx_gases_minor = lookup.idx_minor_gases_lower
    minor_scales_with_density = lookup.minor_lower_scales_with_density
    idx_scaling_gas = lookup.idx_scaling_gases_lower
    scale_by_complement = lookup.lower_scale_by_complement
    minor_gpt_shift = lookup.minor_lower_gpt_shift
    max_per_bnd = lookup.max_minor_per_bnd_lower
    kminor = jax.lax.optimization_barrier(lookup.kminor_lower)
  else:
    minor_absorber_intervals = lookup.n_minor_absrb_upper
    minor_bnd_start = lookup.minor_upper_bnd_start
    minor_bnd_end = lookup.minor_upper_bnd_end
    idx_gases_minor = lookup.idx_minor_gases_upper
    minor_scales_with_density = lookup.minor_upper_scales_with_density
    idx_scaling_gas = lookup.idx_scaling_gases_upper
    scale_by_complement = lookup.upper_scale_by_complement
    minor_gpt_shift = lookup.minor_upper_gpt_shift
    max_per_bnd = lookup.max_minor_per_bnd_upper
    kminor = jax.lax.optimization_barrier(lookup.kminor_upper)

  ibnd = lookup.g_point_to_bnd[igpt]
  loc_in_bnd = igpt - lookup.bnd_lims_gpt[ibnd, 0]
  # Clip temperature to the table range before interpolation.  See
  # ``_clip_to_table_range`` docstring for rationale.  Note: only the
  # **kminor table lookup** uses the clipped temperature; the
  # density-scaling factor ``p / T`` in ``scale_with_density_fn``
  # below uses the **physical** temperature (not clipped) because the
  # Lorentz line-shape density scaling is a real-physics relation
  # meaningful outside the table range.  Consequence: minor OD does
  # not fully saturate at the table boundary — see test
  # ``test_out_of_range_T_saturates_major_OD``'s docstring for
  # rationale.
  t_for_table = _clip_to_table_range(temperature, lookup.t_ref)
  temperature_interpolant = optics_utils.create_linear_interpolant(
      t_for_table, lookup.t_ref
  )

  # Working dtype — all table lookups are cast to this to ensure consistent
  # dtypes through jax.lax.cond and jax.lax.fori_loop (which require matching
  # carry dtypes).  Without this, x64-mode tables (float64) mixed with float32
  # inputs crash in minor-gas loops.
  _wdtype = temperature.dtype

  if vmr_fields is not None and lookup.idx_h2o in vmr_fields:
    dry_factor = 1.0 / (1.0 + vmr_fields[lookup.idx_h2o].astype(_wdtype))
  else:
    dry_factor = 1.0

  def mix_interpolant_fn(t: IndexAndWeight) -> Interpolant:
    """Relative abundance interpolant that depends on `t`."""
    return _compute_relative_abundance_interpolant(
        lookup, vmr_lib, tropo_idx, t.idx, ibnd, False, vmr_fields
    )

  def scale_with_gas_fn(i):
    sgas = jnp.maximum(idx_scaling_gas[i], 0)
    sgas_idx = sgas * jnp.ones_like(tropo_idx)
    scaling_vmr = get_vmr(lookup, vmr_lib, sgas_idx, vmr_fields).astype(_wdtype)
    scaling = jax.lax.cond(
        scale_by_complement[i] == 1,
        lambda: (1.0 - scaling_vmr * dry_factor),
        lambda: scaling_vmr * dry_factor,
    )
    return lambda: scaling

  def scale_with_density_fn(i):
    scaling = _PASCAL_TO_HPASCAL_FACTOR * p / temperature
    scaling *= jax.lax.cond(
        idx_scaling_gas[i] > 0,
        scale_with_gas_fn(i),
        lambda: jnp.ones_like(scaling),
    )
    return lambda: scaling

  # Optical depth will be aggregated over all the minor absorbers contributing
  # to the frequency band.  We use fori_loop over a static trip count and
  # mask inactive iterations so that reverse-mode AD works (while_loop with
  # dynamic stopping is not reverse-mode differentiable).
  minor_start_idx = minor_bnd_start[ibnd]
  _sentinel = jnp.array(minor_absorber_intervals, dtype=minor_start_idx.dtype)
  i0 = jax.lax.cond(
      minor_start_idx >= 0,
      true_fun=lambda: minor_start_idx,
      false_fun=lambda: _sentinel,
  )
  bnd_end = minor_bnd_end[ibnd]

  # Minor absorbers are grouped by band, so this g-point's band owns the
  # contiguous intervals [i0, bnd_end]; loop over at most ``max_per_bnd`` of
  # them instead of every interval in the table (the skipped ones only ever
  # added exact zeros).  Bands with no minor absorber carry the sentinel
  # ``minor_absorber_intervals`` in i0/bnd_end and so stay inactive.
  def body_fn(j, tau_minor):
    i_raw = i0 + j
    active = jnp.logical_and(i_raw <= bnd_end,
                             i_raw < minor_absorber_intervals)
    i = jnp.minimum(i_raw, minor_absorber_intervals - 1)
    # Map the minor contributor to the RRTMGP gas index.
    gas_idx = idx_gases_minor[i] * jnp.ones_like(tropo_idx)
    vmr_minor = get_vmr(lookup, vmr_lib, gas_idx, vmr_fields).astype(_wdtype)
    scaling = vmr_minor * molecules / _M2_TO_CM2_FACTOR
    scaling *= jax.lax.cond(
        minor_scales_with_density[i] == 1,
        scale_with_density_fn(i),
        lambda: jnp.ones_like(temperature),
    )
    # Obtain the global contributor index needed to index into the `kminor`
    # table.
    k_loc = minor_gpt_shift[i] + loc_in_bnd
    delta_tau = (
        optics_utils.interpolate(
            kminor[..., k_loc],
            collections.OrderedDict((
                ('t', lambda: temperature_interpolant),
                ('m', mix_interpolant_fn),
            )),
        ).astype(_wdtype)
        * scaling
    )
    return tau_minor + jnp.where(active, delta_tau, 0.0)

  tau_minor_0 = jnp.zeros_like(temperature)

  return jax.lax.fori_loop(0, max_per_bnd, body_fn, tau_minor_0)


def compute_minor_optical_depth(
    lookup: AbstractLookupGasOptics,
    vmr_lib: LookupVolumeMixingRatio,
    molecules: Array,
    temperature: Array,
    p: Array,
    igpt: Array,
    vmr_fields: dict[int, Array] | None = None,
) -> Array:
  """Compute the optical depth contributions from minor gases.

  Args:
    lookup: An instance of `AbstractLookupGasOptics` containing a RRTMGP index
      for all relevant gases and a lookup table for minor absorption
      coefficients.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m^2]
    temperature: The temperature of the flow field [K].
    p: The pressure field (in Pa).
    igpt: The absorption rank (g-point) index for which the optical depth will
      be computed.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    An `Array` with the pointwise optical depth contributions from the minor
    species.
  """

  # The troposphere index is 1 for levels above the troposphere limit and 0
  # otherwise.
  def minor_tau(is_lower_atmos: bool) -> Array:
    """Compute the minor optical depth assuming an atmosphere level."""
    return _compute_minor_optical_depth(
        lookup,
        vmr_lib,
        molecules,
        temperature,
        p,
        igpt,
        is_lower_atmos,
        vmr_fields,
    )

  return jnp.where(
      p > lookup.p_ref_tropo,
      minor_tau(is_lower_atmos=True),
      minor_tau(is_lower_atmos=False),
  )


def compute_rayleigh_optical_depth(
    lkp: LookupGasOpticsShortwave,
    vmr_lib: LookupVolumeMixingRatio,
    molecules: Array,
    temperature: Array,
    p: Array,
    igpt: Array,
    vmr_fields: dict[int, Array] | None = None,
) -> Array:
  """Compute the optical depth contribution from Rayleigh scattering.

  Args:
    lkp: An instance of `AbstractLookupGasOptics` containing a RRTMGP index
      for all relevant gases and a lookup table for Rayleigh absorption
      coefficients.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m^2].
    temperature: Temperature variable (in K).
    p: The pressure field (in Pa).
    igpt: The absorption variable index (g-point) for which the optical depth
      will be computed.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    An `Array` with the pointwise optical depth contributions from Rayleigh
      scattering.
  """
  # The troposphere index is 1 for levels above the troposphere limit and 0
  # otherwise.
  tropo_idx = jnp.where(p <= lkp.p_ref_tropo, 1, 0)
  # Clip temperature to the table range before interpolation.  See
  # ``_clip_to_table_range`` docstring for rationale.
  t_for_table = _clip_to_table_range(temperature, lkp.t_ref)
  temperature_interpolant = optics_utils.create_linear_interpolant(
      t_for_table, lkp.t_ref
  )
  ibnd = lkp.g_point_to_bnd[igpt]

  def mix_interpolant_fn(t: IndexAndWeight) -> Interpolant:
    """Relative abundance interpolant function that depends on `t` and `p`."""
    return _compute_relative_abundance_interpolant(
        lkp, vmr_lib, tropo_idx, t.idx, ibnd, False, vmr_fields
    )

  interpolant_fns = collections.OrderedDict(
      (('t', lambda: temperature_interpolant), ('m', mix_interpolant_fn))
  )
  # optimization_barrier: prevent XLA constant-folding the per-g-point Rayleigh
  # table slices (see compute_major_optical_depth) — identity at runtime.
  rayl_tau_lower = optics_utils.interpolate(
      jax.lax.optimization_barrier(lkp.rayl_lower)[..., igpt], interpolant_fns
  )
  rayl_tau_upper = optics_utils.interpolate(
      jax.lax.optimization_barrier(lkp.rayl_upper)[..., igpt], interpolant_fns
  )
  if vmr_fields is not None and lkp.idx_h2o in vmr_fields:
    factor = 1.0 + vmr_fields[lkp.idx_h2o]
  else:
    factor = 1.0

  return (
      factor
      * molecules
      / _M2_TO_CM2_FACTOR
      * jnp.where(tropo_idx == 1, rayl_tau_upper, rayl_tau_lower)
  )


def compute_planck_fraction(
    lookup: LookupGasOpticsLongwave,
    vmr_lib: LookupVolumeMixingRatio,
    p: Array,
    temperature: Array,
    igpt: Array,
    vmr_fields: dict[int, Array] | None = None,
) -> Array:
  """Computes the Planck fraction that will be used to weight the Planck source.

  Args:
    lookup: An `LookupGasOpticsLongwave` object containing a RRTMGP index for
      all relevant gases and a lookup table for the Planck source.
    vmr_lib: A `LookupVolumeMixingRatio` object containing the volume mixing
      ratio of all relevant atmospheric gases.
    p: The pressure of the flow field [Pa].
    temperature: The temperature at the grid cell center [K].
    igpt: The absorption rank (g-point) index for which the optical depth will
      be computed.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index, that will overwrite the global means for
      those gases that have a vmr field already available.

  Returns:
    The pointwise Planck fraction associated with the temperature field.
  """
  # The troposphere index is 1 for levels above the troposphere limit and 0
  # otherwise.
  tropo_idx = jnp.where(p <= lookup.p_ref_tropo, 1, 0)
  # Clip temperature to the table range before interpolation.  See
  # ``_clip_to_table_range`` docstring for rationale.
  t_for_table = _clip_to_table_range(temperature, lookup.t_ref)
  temperature_interpolant = optics_utils.create_linear_interpolant(
      t_for_table, lookup.t_ref
  )
  pressure_interpolant = _pressure_interpolant(
      p, lookup.p_ref, tropo_idx
  )
  ibnd = lookup.g_point_to_bnd[igpt]

  def mix_interpolant_fn(t: IndexAndWeight) -> Interpolant:
    """Relative abundance interpolant function that depends on `temperature`."""
    return _compute_relative_abundance_interpolant(
        lookup, vmr_lib, tropo_idx, t.idx, ibnd, False, vmr_fields
    )

  interpolants_fns = collections.OrderedDict((
      ('t', lambda: temperature_interpolant),
      ('p', lambda: pressure_interpolant),
      ('m', mix_interpolant_fn),
  ))

  # 3-D interpolation of the Planck fraction.
  return optics_utils.interpolate(
      jax.lax.optimization_barrier(lookup.planck_fraction)[..., igpt],
      interpolants_fns
  )


def compute_planck_sources(
    lookup: LookupGasOpticsLongwave,
    planck_fraction: Array,
    temperature: Array,
    igpt: Array,
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
  t_for_planck = _clip_to_table_range(temperature, lookup.t_planck)
  interpolant = optics_utils.create_linear_interpolant(
      t_for_planck, lookup.t_planck
  )
  return planck_fraction * optics_utils.interpolate(
      jax.lax.optimization_barrier(lookup.totplnk)[ibnd, :],
      collections.OrderedDict({'t': lambda: interpolant}),
  )
