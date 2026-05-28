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

"""A library for solving the two-stream radiative transfer equation."""

from typing import TypeAlias, cast

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics._shared import safe_divide
from legoesm.atmosphere.physics.radiation.rrtmgp import constants
from legoesm.atmosphere.physics.radiation.rrtmgp import kernel_ops
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import atmospheric_state
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_gas_optics_base
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import optics
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import optics_base
from legoesm.atmosphere.physics.radiation.rrtmgp.rte import monochromatic_two_stream

Array: TypeAlias = jax.Array
AbstractLookupGasOptics: TypeAlias = (
    lookup_gas_optics_base.AbstractLookupGasOptics
)
AtmosphericState: TypeAlias = atmospheric_state.AtmosphericState


def _compute_local_properties_lw(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    igpt: Array,
    optics_lib: optics_base.OpticsScheme,
    vmr_fields: dict[int, Array] | None = None,
    sfc_temperature: Array | float | None = None,
    cloud_r_eff_liq: Array | None = None,
    cloud_path_liq: Array | None = None,
    cloud_r_eff_ice: Array | None = None,
    cloud_path_ice: Array | None = None,
    cloud_fraction: Array | None = None,
    lw_diffusive_factor: float | Array = monochromatic_two_stream._LW_DIFFUSIVE_FACTOR,
    precomputed_lw_optical_props: dict[str, Array] | None = None,
) -> dict[str, Array]:
  """Compute local optical properties for longwave radiative transfer.

  ``precomputed_lw_optical_props`` lets the caller share an already-
  computed optics dict (e.g. the optimal-angle path in ``solve_lw``
  needs the optical depth to derive the per-column secant, and would
  otherwise repeat the table interpolation here).  When ``None`` the
  function calls ``optics_lib.compute_lw_optical_properties`` itself.
  XLA's CSE pass already deduplicates identical-input calls under
  JIT, but threading the dict through keeps the graph compact and
  makes the dependency explicit.
  """
  if isinstance(sfc_temperature, float):
    # Create a plane for the surface temperature representation.
    nx, ny, _ = temperature.shape
    sfc_temperature = sfc_temperature * jnp.ones(
        (nx, ny), dtype=temperature.dtype
    )

  if precomputed_lw_optical_props is not None:
    lw_optical_props = precomputed_lw_optical_props
  else:
    # Compute optical properties: `optical_depth`, `ssa`, & `asymmetry_factor`.
    lw_optical_props = optics_lib.compute_lw_optical_properties(
        pressure,
        temperature,
        molecules,
        igpt,
        vmr_fields,
        cloud_r_eff_liq,
        cloud_path_liq,
        cloud_r_eff_ice,
        cloud_path_ice,
        cloud_fraction=cloud_fraction,
    )

  # Compute Planck sources: `planck_src`, `planck_src_bottom`, `planck_src_top`,
  # and `planck_src_sfc`.
  planck_srcs = optics_lib.compute_planck_sources(
      pressure, temperature, igpt, vmr_fields, sfc_temperature=sfc_temperature
  )

  halo_width = 1
  sfc_src = planck_srcs.get(
      'planck_src_sfc', planck_srcs['planck_src_bottom'][:, :, halo_width]
  )

  # Compute combined Planck sources.  Output keys are `planck_src_bottom` and
  # `planck_src_top`.
  combined_srcs = monochromatic_two_stream.lw_combine_sources(planck_srcs)

  # Compute `t_diff`, `r_diff`, `src_up`, and `src_down`.
  src_and_properties = monochromatic_two_stream.lw_cell_source_and_properties(
      lw_optical_props['optical_depth'],
      lw_optical_props['ssa'],
      combined_srcs['planck_src_bottom'],
      combined_srcs['planck_src_top'],
      lw_optical_props['asymmetry_factor'],
      lw_diffusive_factor=lw_diffusive_factor,
  )
  src_and_properties['sfc_src'] = sfc_src

  return src_and_properties


def _reindex_vmr_fields(
    vmr_fields: dict[str, Array], gas_optics_lib: AbstractLookupGasOptics
) -> dict[int, Array]:
  """Converts the chemical formulas of the gas species to RRTM indices."""
  return {gas_optics_lib.idx_gases[k]: v for k, v in vmr_fields.items()}


def _replace_top_flux(f: Array) -> Array:
  """Modify problematic value for the fluxes at the top boundary (top halo).

  Use quadratic polynomials to evaluate the flux at the top boundary making use
  of the points just below the top boundary.

  Args:
    f: The array to fix the top boundary of.

  Returns:
    The array with the top boundary value fixed.
  """
  top_bdy_f = 3 * f[:, :, -2] - 3 * f[:, :, -3] + f[:, :, -4]
  f = f.at[:, :, -1].set(top_bdy_f)
  return f


def _compute_optimal_lw_secant(
    optical_depth: Array,
    band_idx: Array,
    optimal_angle_fit: Array,
    halo_width: int = 1,
) -> Array:
  """Compute upstream RRTMGP's optimal longwave diffusivity secant.

  Replicates ``rte-rrtmgp``'s ``compute_optimal_angles``: a per-band linear
  fit on the column transmissivity ``trans = exp(-sum_z tau)``::

      secant(col, gpt) = optimal_angle_fit[band, 0] * trans
                       + optimal_angle_fit[band, 1]

  Operates on the per-g-point optical depth slice ``(ncol, 1, nlev+2)``,
  excluding halo cells from the sum since halos carry linearly-extrapolated
  values that are stripped before the recurrent integration anyway.

  Args:
    optical_depth: ``(ncol, 1, nlev+2)`` per-g-point optical depth slice.
    band_idx: 0-D scalar with the spectral band corresponding to the
      current g-point (``g_point_to_bnd[igpt]``).
    optimal_angle_fit: ``(n_bnd, 2)`` polynomial-fit coefficients loaded
      from the longwave gas-optics file.
    halo_width: Vertical halo width to exclude from the column sum.

  Returns:
    ``(ncol, 1, 1)`` secant ready to broadcast against the
    ``(ncol, 1, nlev+2)`` optical-depth array.
  """
  # Interior column (halos excluded) total optical depth.
  hw = halo_width
  if hw > 0:
    tau_interior = optical_depth[:, :, hw:-hw]
  else:
    tau_interior = optical_depth
  # ``jnp.maximum(tau, 0.0)`` is a deliberate departure from the literal
  # upstream formula ``tau_total = sum(tau)`` (codex iter-2 review, LOW).
  # Upstream is invoked on freshly computed positive optical depths so the
  # difference is zero in practice; in legoESM the same array is reused
  # downstream after the recurrence strips halos, but during scan tracing
  # a halo cell whose interpolated tau briefly dipped below zero would
  # otherwise inject a negative term into ``trans_total`` and amplify
  # ``-secant`` errors.  Clamping to zero matches the physical meaning of
  # "no optical depth" and keeps ``exp(-tau_total) ∈ [0, 1]``.  Interior
  # cells (the ones that actually contribute) almost always have
  # ``tau >= 0`` from the kmajor/kminor lookup, so this clamp acts only
  # as a guard.
  tau_total = jnp.sum(jnp.maximum(tau_interior, 0.0), axis=-1, keepdims=True)
  # ``tau_total`` shape ``(ncol, 1, 1)``.  Compute column transmissivity.
  trans_total = jnp.exp(-tau_total)
  c0 = optimal_angle_fit[band_idx, 0]
  c1 = optimal_angle_fit[band_idx, 1]
  return c0 * trans_total + c1


def solve_lw(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    optics_lib: optics_base.OpticsScheme,
    atmos_state: AtmosphericState,
    vmr_fields: dict[str, Array] | None = None,
    sfc_temperature: Array | float | None = None,
    cloud_r_eff_liq: Array | None = None,
    cloud_path_liq: Array | None = None,
    cloud_r_eff_ice: Array | None = None,
    cloud_path_ice: Array | None = None,
    cloud_fraction: Array | None = None,
    use_scan: bool | None = None,
    use_optimal_angle: bool = False,
) -> dict[str, Array]:
  """Solves two-stream radiative transfer equation over the longwave spectrum.

  Local optical properties like optical depth, single-scattering albedo, and
  asymmetry factor are computed using an optics library and transformed to
  two-stream approximations of reflectance and transmittance. The sources of
  longwave radiation are the Planck sources, which are a function only of
  temperature. To obtain the cell-centered directional Planck sources, the
  sources are first computed at the cell boundaries and the net source
  emanating from the grid cell is determined. Each spectral interval,
  represented by a g-point, is a separate radiative transfer problem, and can
  be computed in parallel. Finally, the independently solved fluxes are summed
  over the full spectrum to yield the final upwelling and downwelling fluxes.

  Args:
    pressure: The pressure field [Pa].
    temperature: The temperature field [K].
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m²].
    optics_lib: An instance of an optics library.
    atmos_state: An instance containing the atmospheric state.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by the chemical formula.
    sfc_temperature: The optional surface temperature represented as either a 2D
      field or as a scalar [K].
    cloud_r_eff_liq: The effective radius of cloud droplets [m].
    cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
      [kg/m²].
    cloud_r_eff_ice: The effective radius of cloud ice particles [m].
    cloud_path_ice: The cloud ice water path in each atmospheric grid cell
      [kg/m²].
    use_scan: Whether to use scan or for loops for the recurrent operation.

  Returns:
    A dictionary with the following entries (in units of W/m²):
      `flux_up`: The upwelling longwave radiative flux at cell face i - 1/2.
      `flux_down`: The downwelling longwave radiative flux at face i - 1/2.
      `flux_net`: The net longwave radiative flux at face i - 1/2.
  """
  optics_lib = cast(optics.RRTMOptics | optics.GrayAtmosphereOptics, optics_lib)
  if vmr_fields is not None:
    # Convert the chemical formulas of the gas species to RRTM-consistent
    # numerical identifiers.
    vmr_fields = _reindex_vmr_fields(vmr_fields, optics_lib.gas_optics_lw)

  # Resolve the optimal-angle table once, outside the scan body, so the
  # branch is selected at trace time and does not introduce a Python ``if``
  # on a traced value inside the scan.  Raise explicitly when the caller
  # requested ``use_optimal_angle=True`` but the gas-optics file does not
  # ship ``optimal_angle_fit`` — silently degrading to the fixed Fu-Liou
  # 1.66 contradicts the config contract documented on
  # ``RRTMGPConfig.use_optimal_angle`` (codex iter-2 review, MEDIUM).
  optimal_angle_fit = None
  if use_optimal_angle:
    candidate = None
    if hasattr(optics_lib, 'gas_optics_lw'):
      candidate = getattr(optics_lib.gas_optics_lw, 'optimal_angle_fit', None)
    if candidate is None:
      raise ValueError(
          "solve_lw(use_optimal_angle=True) requires the longwave "
          "gas-optics file to ship 'optimal_angle_fit' (added to "
          "rrtmgp-gas-lw-* in rte-rrtmgp >= 1.7).  Either upgrade the "
          "data file or set use_optimal_angle=False to keep the fixed "
          "Fu-Liou 1.66 diffusivity secant."
      )
    optimal_angle_fit = candidate

  def step_fn(igpt, cumulative_flux):
    # Compute the LW optics once per g-point; reuse for both the
    # optimal-angle secant and the source-and-properties solve.
    # Without this, the optimal-angle path would call
    # ``compute_lw_optical_properties`` twice per igpt and rely on
    # XLA's CSE to deduplicate — explicit reuse keeps the graph
    # smaller and the dependency obvious.
    precomputed_props = optics_lib.compute_lw_optical_properties(
        pressure, temperature, molecules, igpt, vmr_fields,
        cloud_r_eff_liq, cloud_path_liq,
        cloud_r_eff_ice, cloud_path_ice,
        cloud_fraction=cloud_fraction,
    )
    if optimal_angle_fit is not None:
      band_idx = optics_lib.gas_optics_lw.g_point_to_bnd[igpt]
      lw_diffusive_factor = _compute_optimal_lw_secant(
          precomputed_props['optical_depth'], band_idx, optimal_angle_fit
      )
    else:
      lw_diffusive_factor = monochromatic_two_stream._LW_DIFFUSIVE_FACTOR

    optical_props_2stream = _compute_local_properties_lw(
        pressure,
        temperature,
        molecules,
        igpt,
        optics_lib,
        vmr_fields,
        sfc_temperature,
        cloud_r_eff_liq,
        cloud_path_liq,
        cloud_r_eff_ice,
        cloud_path_ice,
        cloud_fraction=cloud_fraction,
        lw_diffusive_factor=lw_diffusive_factor,
        precomputed_lw_optical_props=precomputed_props,
    )

    # Boundary conditions.
    sfc_src = optical_props_2stream['sfc_src']
    toa_flux_down_lw = atmos_state.toa_flux_lw * jnp.ones_like(sfc_src)
    sfc_emissivity_lw = atmos_state.sfc_emis * jnp.ones_like(sfc_src)

    fluxes = monochromatic_two_stream.lw_transport(
        optical_props_2stream['t_diff'],
        optical_props_2stream['r_diff'],
        optical_props_2stream['src_up'],
        optical_props_2stream['src_down'],
        toa_flux_down_lw,
        sfc_src,
        sfc_emissivity_lw,
        use_scan,
    )
    # cumulative_flux keys: 'flux_up', 'flux_down', 'flux_net'
    return jax.tree.map(jnp.add, fluxes, cumulative_flux)

  flux_keys = ['flux_up', 'flux_down', 'flux_net']
  init_val = {key: jnp.zeros_like(temperature) for key in flux_keys}

  # Replace ``jax.lax.fori_loop`` with ``jax.lax.scan`` wrapping
  # ``step_fn`` in ``jax.checkpoint(policy=nothing_saveable)`` so the
  # backward pass recomputes per-g-point intermediates one at a time
  # instead of materialising all 256 g-points' activations.  Memory
  # for the per-call backward drops from ~21 MiB/col (T11 -> ~14 GiB
  # at v7 settings) to roughly ~1 MiB/col (~0.7 GiB at T11), making
  # T127 (~1°) feasible on a 24 GiB GPU under
  # ``eqx.filter_value_and_grad``.  Forward semantics are identical:
  # ``scan`` and ``fori_loop`` both iterate the same step function
  # and accumulate the cumulative flux.
  def _scan_step(carry, igpt):
    return step_fn(igpt, carry), None

  _scan_step_ckpt = jax.checkpoint(
      _scan_step,
      prevent_cse=True,
      policy=jax.checkpoint_policies.nothing_saveable,
  )
  fluxes, _ = jax.lax.scan(
      _scan_step_ckpt, init_val, jnp.arange(optics_lib.n_gpt_lw),
  )
  # There are problematic values for the fluxes at the top boundary (the top
  # halo), so fix using a quadratic polynomial to evaluate the flux at the top
  # boundary.
  for key in flux_keys:
    fluxes[key] = _replace_top_flux(fluxes[key])

  return fluxes


def solve_sw(
    pressure: Array,
    temperature: Array,
    molecules: Array,
    optics_lib: optics_base.OpticsScheme,
    atmos_state: AtmosphericState,
    vmr_fields: dict[str, Array] | None = None,
    cloud_r_eff_liq: Array | None = None,
    cloud_path_liq: Array | None = None,
    cloud_r_eff_ice: Array | None = None,
    cloud_path_ice: Array | None = None,
    cloud_fraction: Array | None = None,
    aerosol_optical_depth: Array | None = None,
    aerosol_single_scattering_albedo: float = 0.93,
    aerosol_asymmetry_factor: float = 0.70,
    solar_fraction_by_gpt: Array | None = None,
    use_scan: bool | None = None,
) -> dict[str, Array]:
  """Solves the two-stream radiative transfer equation for shortwave.

  Local optical properties like optical depth, single-scattering albedo, and
  asymmetry factor are computed using an optics library and transformed to
  two-stream approximations of reflectance and transmittance. The sources of
  shortwave radiation are determined by the diffuse propagation of direct
  solar radiation through the layered atmosphere. Each spectral interval,
  represented by a g-point, is a separate radiative transfer problem, and can
  be computed in parallel. Finally, the independently solved fluxes are summed
  over the full spectrum to yield the final upwelling and downwelling fluxes.

  Args:
    pressure: The pressure field [Pa].
    temperature: The temperature field [K].
    molecules: The number of molecules in an atmospheric grid cell per area
      [molecules/m²].
    optics_lib: An instance of an optics library.
    atmos_state: An instance containing the atmospheric state.
    vmr_fields: An optional dictionary containing precomputed volume mixing
      ratio fields, keyed by gas index.
    cloud_r_eff_liq: The effective radius of cloud droplets [m].
    cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
      [kg/m²].
    cloud_r_eff_ice: The effective radius of cloud ice particles [m].
    cloud_path_ice: The cloud ice water path in each atmospheric grid cell
      [kg/m²].
    aerosol_optical_depth: Optional aerosol optical depth per layer (same
      shape as `temperature`), added to SW extinction.
    aerosol_single_scattering_albedo: Bulk aerosol single-scattering albedo.
    aerosol_asymmetry_factor: Bulk aerosol asymmetry factor.
    solar_fraction_by_gpt: Optional external spectral solar weights by g-point.
    use_scan: Whether to use scan or for loops for the recurrent operation.

  Returns:
    A dictionary with the following entries (in units of W/m²):
      `flux_up`: The upwelling shortwave radiative flux at cell face i - 1/2.
      `flux_down`: The downwelling shortwave radiative flux at face i - 1/2.
      `flux_net`: The net shortwave radiative flux at face i - 1/2.
  """
  zenith = atmos_state.zenith
  optics_lib = cast(optics.RRTMOptics | optics.GrayAtmosphereOptics, optics_lib)
  if vmr_fields is not None:
    # Convert the chemical formulas of the gas species to RRTM-consistent
    # numerical identifiers.
    vmr_fields = _reindex_vmr_fields(vmr_fields, optics_lib.gas_optics_sw)

  # --- Per-column day/night handling ---
  # ``zenith`` may be a scalar (single-column) or an array with a column
  # dimension, e.g. shape ``(ncol, 1)``.  To avoid division-by-zero in
  # ``exp(-tau / cos(zenith))`` for nighttime columns (cos(zenith) ~ 0),
  # we clamp the zenith used in the solve to at most ~89.4 degrees
  # (cos > 0.01).  After the solve, nighttime columns are zeroed out.
  _ZENITH_MAX = jnp.arccos(jnp.asarray(0.01, dtype=temperature.dtype))
  safe_zenith = jnp.minimum(zenith, _ZENITH_MAX)

  # Build a per-column boolean mask that is True for daytime columns.
  # Works for both scalar zenith and array zenith.
  is_day_col = zenith < 0.5 * jnp.pi  # shape () or (ncol, 1)

  # Check whether *any* column is illuminated to short-circuit a global
  # nighttime domain (preserves the original optimisation).
  any_day = jnp.any(is_day_col)

  def step_fn(igpt, partial_fluxes):
    sw_optical_props = optics_lib.compute_sw_optical_properties(
        pressure,
        temperature,
        molecules,
        igpt,
        vmr_fields,
        cloud_r_eff_liq,
        cloud_path_liq,
        cloud_r_eff_ice,
        cloud_path_ice,
        cloud_fraction=cloud_fraction,
    )
    if aerosol_optical_depth is not None:
      tau_bg = jnp.maximum(sw_optical_props['optical_depth'], 1.0e-12)
      tau_aer = jnp.maximum(aerosol_optical_depth, 0.0)
      tau_tot = tau_bg + tau_aer
      w_bg = sw_optical_props['ssa']
      g_bg = sw_optical_props['asymmetry_factor']
      w_num = tau_bg * w_bg + tau_aer * aerosol_single_scattering_albedo
      # AD-safe SW optical-property mixing (restored from commit 59407953
      # after AIMIP-#312 merge reverted it).  ``a / jnp.maximum(b, eps)``
      # has a ``-a/b**2`` VJP that overflows when ``b`` is at the floor —
      # for cloud-free, low-water-vapor stratospheric layers ``tau_tot``
      # can reach the 1e-12 floor and the backward propagates NaN to every
      # upstream traced parameter whose state path touches gas absorption.
      # ``safe_divide`` masks the bad branch before the divide.  The outer
      # ``jnp.clip`` preserves the original output range; with ``fill=0.0``
      # the bad branch lands inside that range.
      w_tot = jnp.clip(
          safe_divide(w_num, tau_tot, eps=1.0e-12, fill=0.0),
          0.0,
          1.0,
      )
      g_num = (
          tau_bg * w_bg * g_bg
          + tau_aer * aerosol_single_scattering_albedo * aerosol_asymmetry_factor
      )
      g_denom = tau_tot * jnp.maximum(w_tot, 1.0e-12)
      g_tot = jnp.clip(
          safe_divide(g_num, g_denom, eps=1.0e-12, fill=0.0),
          -1.0,
          1.0,
      )
      sw_optical_props = {
          'optical_depth': tau_tot,
          'ssa': w_tot,
          'asymmetry_factor': g_tot,
      }
    optical_props_2stream = monochromatic_two_stream.sw_cell_properties(
        safe_zenith,
        sw_optical_props['optical_depth'],
        sw_optical_props['ssa'],
        sw_optical_props['asymmetry_factor'],
    )

    # Surface albedo: broadcast per-column array or scalar to 2D plane
    # with the same horizontal sharding as the temperature.
    sfc_albedo = atmos_state.sfc_alb * jnp.ones_like(temperature[:, :, 0])

    # Monochromatic top of atmosphere flux.
    if solar_fraction_by_gpt is None:
      spectral_weight = optics_lib.solar_fraction_by_gpt[igpt]
    else:
      spectral_weight = solar_fraction_by_gpt[igpt]
    solar_flux = atmos_state.irrad * spectral_weight
    toa_flux = solar_flux * jnp.ones_like(temperature[:, :, 0])

    sources_2stream = monochromatic_two_stream.sw_cell_source(
        t_dir=optical_props_2stream['t_dir'],
        r_dir=optical_props_2stream['r_dir'],
        optical_depth=sw_optical_props['optical_depth'],
        toa_flux=toa_flux,
        sfc_albedo_direct=sfc_albedo,
        zenith=safe_zenith,
        use_scan=use_scan,
    )

    sw_fluxes = monochromatic_two_stream.sw_transport(
        t_diff=optical_props_2stream['t_diff'],
        r_diff=optical_props_2stream['r_diff'],
        src_up=sources_2stream['src_up'],
        src_down=sources_2stream['src_down'],
        sfc_src=sources_2stream['sfc_src'],
        sfc_albedo=sfc_albedo,
        flux_down_dir=sources_2stream['flux_down_dir'],
        use_scan=use_scan,
    )
    total_sw_fluxes = jax.tree.map(jnp.add, sw_fluxes, partial_fluxes)
    return total_sw_fluxes

  flux_keys = ['flux_up', 'flux_down', 'flux_net']
  fluxes_0 = {key: jnp.zeros_like(temperature) for key in flux_keys}

  def _compute_fluxes(_):
    # ``fori_loop`` -> ``scan`` + per-g-point ``jax.checkpoint`` so the
    # backward pass recomputes one g-point at a time instead of
    # storing all 224 SW g-points' activations.  See solve_lw above
    # for the longwave companion change.
    def _scan_step(carry, igpt):
      return step_fn(igpt, carry), None

    _scan_step_ckpt = jax.checkpoint(
        _scan_step,
        prevent_cse=True,
        policy=jax.checkpoint_policies.nothing_saveable,
    )
    fluxes, _ = jax.lax.scan(
        _scan_step_ckpt, fluxes_0, jnp.arange(optics_lib.n_gpt_sw),
    )
    # There are problematic values for the fluxes at the top boundary (the top
    # halo), so fix using a quadratic polynomial to evaluate the flux at the top
    # boundary.
    for key in flux_keys:
      fluxes[key] = _replace_top_flux(fluxes[key])

    # Zero out nighttime columns.  ``is_day_col`` broadcasts from
    # shape ``()`` or ``(ncol, 1)`` against ``(ncol, 1, nlev+2)``.
    day_mask_3d = jnp.asarray(is_day_col, dtype=temperature.dtype)
    for key in flux_keys:
      fluxes[key] = fluxes[key] * day_mask_3d
    return fluxes

  # Short-circuit: if the entire domain is nighttime, skip the solve.
  return jax.lax.cond(
      any_day,
      _compute_fluxes,
      lambda _: fluxes_0,
      operand=None,
  )


def compute_heating_rate(
    flux_net: Array,
    pressure: Array,
    dp: Array | None = None,
) -> Array:
  """Computes cell-center heating rate from pressure and net radiative flux.

  The net radiative flux corresponds to the bottom cell face. The difference
  of the net flux at the top face and that at the bottom face gives the total
  net flux out of the grid cell. Using the pressure difference across the grid
  cell, the net flux can be converted to a heating rate, in K/s.

  Args:
    flux_net: The net flux at the bottom face [W/m²].
    pressure: The pressure field [Pa].
    dp: Exact layer pressure thickness [Pa].  When provided, used directly
        instead of the centered-difference approximation from ``pressure``.

  Returns:
    The heating rate of the grid cell [K/s].
  """
  if dp is None:
      # Fallback: centered-difference approximation.
      dp = 0.5 * kernel_ops.centered_difference(pressure, dim=2)

  # Compute the forward pressure difference of fluxes on faces (like a
  # derivative of face_to_node).  This is the net upward flux out of the
  # cell; a positive value means the cell radiates away energy and cools.
  dflux = kernel_ops.forward_difference(flux_net, dim=2)

  # Heating rate at the grid cell center [K/s].  **Minus sign** (restored
  # from commit 0be22f0f after the AIMIP-#312 merge reverted it): net
  # flux *out* cools the cell.  ``abs(dp)`` so the sign is set by the
  # flux divergence alone, not by the vertical-axis orientation
  # (``solve_columns`` passes positive layer thickness; the legacy
  # ``RRTMGP.compute_heating_rate`` callpath passes a centered-difference
  # negative ``dp`` that abs() canonicalises).
  #
  # Pre-fix bug symptom: free-tropospheric LW heating was +2..+5 K/day
  # (radiative warming) instead of −1..−2 K/day (radiative cooling),
  # driving thermal runaway in long AMIP integrations (T̄ 261 → 293 K
  # over 120 days, NaN blowup at day 125).
  return -constants.G * dflux / jnp.abs(dp) / constants.CP_D
