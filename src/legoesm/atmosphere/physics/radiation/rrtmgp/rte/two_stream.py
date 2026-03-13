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
import numpy as np
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
) -> dict[str, Array]:
  """Compute local optical properties for longwave radiative transfer."""
  if isinstance(sfc_temperature, float):
    # Create a plane for the surface temperature representation.
    nx, ny, _ = temperature.shape
    sfc_temperature = sfc_temperature * jnp.ones(
        (nx, ny), dtype=temperature.dtype
    )

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
    use_scan: bool = False,
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

  def step_fn(igpt, cumulative_flux):
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

  fluxes = jax.lax.fori_loop(0, optics_lib.n_gpt_lw, step_fn, init_val)
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
    aerosol_optical_depth: Array | None = None,
    aerosol_single_scattering_albedo: float = 0.93,
    aerosol_asymmetry_factor: float = 0.70,
    solar_fraction_by_gpt: Array | None = None,
    use_scan: bool = False,
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
    )
    if aerosol_optical_depth is not None:
      tau_bg = jnp.maximum(sw_optical_props['optical_depth'], 1.0e-12)
      tau_aer = jnp.maximum(aerosol_optical_depth, 0.0)
      tau_tot = tau_bg + tau_aer
      w_bg = sw_optical_props['ssa']
      g_bg = sw_optical_props['asymmetry_factor']
      w_num = tau_bg * w_bg + tau_aer * aerosol_single_scattering_albedo
      w_tot = jnp.clip(w_num / jnp.maximum(tau_tot, 1.0e-12), 0.0, 1.0)
      g_num = (
          tau_bg * w_bg * g_bg
          + tau_aer * aerosol_single_scattering_albedo * aerosol_asymmetry_factor
      )
      g_tot = jnp.clip(
          g_num / jnp.maximum(tau_tot * jnp.maximum(w_tot, 1.0e-12), 1.0e-12),
          -1.0,
          1.0,
      )
      sw_optical_props = {
          'optical_depth': tau_tot,
          'ssa': w_tot,
          'asymmetry_factor': g_tot,
      }
    optical_props_2stream = monochromatic_two_stream.sw_cell_properties(
        zenith,
        sw_optical_props['optical_depth'],
        sw_optical_props['ssa'],
        sw_optical_props['asymmetry_factor'],
    )

    # Create an xy plane for the surface albedo and top-of-atmospehre flux, but
    # keep the same horizontal sharding as the temperature.
    sfc_albedo = atmos_state.sfc_alb * jnp.ones_like(temperature)[:, :, 0]

    # Monochromatic top of atmosphere flux.
    if solar_fraction_by_gpt is None:
      spectral_weight = optics_lib.solar_fraction_by_gpt[igpt]
    else:
      spectral_weight = solar_fraction_by_gpt[igpt]
    solar_flux = atmos_state.irrad * spectral_weight
    toa_flux = solar_flux * jnp.ones_like(temperature)[:, :, 0]

    sources_2stream = monochromatic_two_stream.sw_cell_source(
        t_dir=optical_props_2stream['t_dir'],
        r_dir=optical_props_2stream['r_dir'],
        optical_depth=sw_optical_props['optical_depth'],
        toa_flux=toa_flux,
        sfc_albedo_direct=sfc_albedo,
        zenith=zenith,
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
    fluxes = jax.lax.fori_loop(0, optics_lib.n_gpt_sw, step_fn, fluxes_0)
    # There are problematic values for the fluxes at the top boundary (the top
    # halo), so fix using a quadratic polynomial to evaluate the flux at the top
    # boundary.
    for key in flux_keys:
      fluxes[key] = _replace_top_flux(fluxes[key])
    return fluxes

  # Use JAX control flow to avoid Python boolean conversion on tracers
  return jax.lax.cond(
      zenith >= 0.5 * jnp.pi,
      lambda _: fluxes_0,
      _compute_fluxes,
      operand=None,
  )


def compute_heating_rate(
    flux_net: Array,
    pressure: Array,
) -> Array:
  """Computes cell-center heating rate from pressure and net radiative flux.

  The net radiative flux corresponds to the bottom cell face. The difference
  of the net flux at the top face and that at the bottom face gives the total
  net flux out of the grid cell. Using the pressure difference across the grid
  cell, the net flux can be converted to a heating rate, in K/s.

  Args:
    flux_net: The net flux at the bottom face [W/m²].
    pressure: The pressure field [Pa].

  Returns:
    The heating rate of the grid cell [K/s].
  """
  # Compute the centered pressure difference in z:
  #   dp_{i,j,k} = (p_{i,j,k+1} - p_{i,j,k-1}) / 2.
  # This is an approximation to the pressure difference across the cell, which
  # would use the pressure at the upper and lower faces, but it should be ok.
  dp = 0.5 * kernel_ops.centered_difference(pressure, dim=2)

  # Compute the forward pressure difference of fluxes on faces (like a
  # derivative of face_to_node).
  dflux = kernel_ops.forward_difference(flux_net, dim=2)

  # Compute the heating rate at the grid cell center in K/s.
  return constants.G * dflux / dp / constants.CP_D
