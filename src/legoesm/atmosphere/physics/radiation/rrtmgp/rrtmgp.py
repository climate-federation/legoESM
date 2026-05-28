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

"""Implementation of a radiative transfer solver."""

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.radiation.rrtmgp import constants
from legoesm.atmosphere.physics.radiation.rrtmgp import kernel_ops
from legoesm.atmosphere.physics.radiation.rrtmgp import stretched_grid_util
from legoesm.atmosphere.physics.radiation.rrtmgp import rrtmgp_common
from legoesm.atmosphere.physics.radiation.rrtmgp.config import radiative_transfer
from legoesm.atmosphere.physics.radiation.rrtmgp.config.radiative_transfer import (
    OpticsParameters,
    RRTMOptics as RRTMOpticsConfig,
)
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import atmospheric_state
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import constants as optics_constants
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import lookup_volume_mixing_ratio
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import optics
from legoesm.atmosphere.physics.radiation.rrtmgp.optics.lookup_volume_mixing_ratio import (
    LookupVolumeMixingRatio,
)
from legoesm.atmosphere.physics.radiation.rrtmgp.optics.optics import optics_factory
from legoesm.atmosphere.physics.radiation.rrtmgp.rte import two_stream

Array: TypeAlias = jax.Array

# ---------------------------------------------------------------------------
# Default RRTMGP data file paths (Zarr preferred, NetCDF fallback)
# ---------------------------------------------------------------------------
_ZARR_DIR = Path(__file__).parent.parent.parent.parent / "data" / "zarr"
_NC_DIR = Path(__file__).parent / "optics" / "rrtmgp_data"


def _default_data_path(basename_nc: str) -> str:
    """Return path to Zarr store if available, else fall back to NetCDF."""
    zarr_path = _ZARR_DIR / basename_nc.replace(".nc", ".zarr")
    if zarr_path.exists():
        return str(zarr_path)
    return str(_NC_DIR / basename_nc)


_DEFAULT_LW_GAS = _default_data_path("rrtmgp-gas-lw-g128.nc")
_DEFAULT_SW_GAS = _default_data_path("rrtmgp-gas-sw-g112.nc")
_DEFAULT_LW_CLOUD = _default_data_path("cloudysky_lw.nc")
_DEFAULT_SW_CLOUD = _default_data_path("cloudysky_sw.nc")

# Module-level optics cache (shared across RRTMGP instances)
_legoesm_optics_cache: dict = {}


# ---------------------------------------------------------------------------
# Helper functions for the legoESM column solver
# ---------------------------------------------------------------------------

def _add_halos(f_3d):
    """Add 1-cell vertical halos via linear extrapolation.

    Input shape (ncol, 1, nlev), output shape (ncol, 1, nlev+2).
    """
    bottom_halo = 2 * f_3d[:, :, 0:1] - f_3d[:, :, 1:2]
    top_halo = 2 * f_3d[:, :, -1:] - f_3d[:, :, -2:-1]
    return jnp.concatenate([bottom_halo, f_3d, top_halo], axis=2)


def _standard_o3_profile(p_full):
    """Simple climatological ozone profile (VMR). US Std Atm 1976 fit."""
    p_hPa = p_full / 100.0
    o3 = 8.0e-6 * jnp.exp(-0.5 * ((jnp.log(p_hPa) - jnp.log(10.0)) / 1.5) ** 2)
    return jnp.clip(o3, 1.0e-10, None)


def standard_o3_profile(p_full):
    """Public alias of the climatological ozone-VMR profile.

    Use this when an external ozone source is not configured but RRTMGP
    must still see a realistic ozone column — driver code historically
    initialised ``o3_vmr`` to zeros, which RRTMGP then clipped to
    ``1e-10`` and which silently disabled stratospheric heating.  Pass
    the result of ``standard_o3_profile(p_full)`` instead of zeros, or
    pass ``None`` to let RRTMGP build the same profile internally.
    """
    return _standard_o3_profile(p_full)


def _humidity_to_volume_mixing_ratio(
    q_t: Array, q_c: Array
) -> Array:
  """For water vapor, convert humidity to a volume mixing ratio."""
  mol_ratio = constants.R_V / constants.R_D
  q_v = q_t - q_c
  mix_ratio = q_v / (1 - q_t)
  return mol_ratio * mix_ratio


def _air_molecules_per_area(p_xxc: Array, vmr_h2o_xxc: Array) -> Array:
  """Compute the number of molecules in a grid cell per area."""
  dp_xxc = kernel_ops.centered_difference(p_xxc, dim=2)
  mol_m_air_xxc = (
      constants.DRY_AIR_MOL_MASS + constants.WATER_MOL_MASS * vmr_h2o_xxc
  )
  return -(dp_xxc / constants.G) * constants.AVOGADRO / mol_m_air_xxc


def _compute_cloud_path(
    rho: Array, q_c: Array, dz: float, sg_map: dict[str, Array]
) -> Array:
  """Compute the cloud water/ice path in each atmospheric grid cell."""
  use_stretched_grid_z = stretched_grid_util.get_use_stretched_grid(sg_map)[2]
  if use_stretched_grid_z:
    h = sg_map[stretched_grid_util.hc_key(2)]
  else:
    h = dz
  return rho * q_c * h


def _horiz_mean(f: Array) -> Array:
  """Compute the horizontal mean of `f`.

  Args:
    f: The array to compute the horizontal mean of.

  Returns:
    The horizontal mean of `f`, preserving the input dtype.
  """
  return jnp.mean(f, axis=(0, 1))


class RRTMGP:
  """Rapid Radiative Transfer Model for General Circulation Models (RRTMGP)."""

  def __init__(
      self,
      radiative_transfer_cfg: radiative_transfer.RadiativeTransfer,
      dz: float,
      diagnostic_fields: Sequence[str] = tuple(),
  ):
    self._dz = dz  # Store dz (only used if not using stretched grid in z).
    self._diagnostic_fields = diagnostic_fields
    self._save_lw_sw_heating_rates = (
        radiative_transfer_cfg.save_lw_sw_heating_rates
    )
    self._do_clear_sky = radiative_transfer_cfg.do_clear_sky

    # Load and store the atmospheric gas concentrations.
    self.atmospheric_state = atmospheric_state.from_config(
        radiative_transfer_cfg.atmospheric_state_cfg
    )
    # Create the optics library.
    self.optics_lib = optics.optics_factory(
        radiative_transfer_cfg.optics, self.atmospheric_state.vmr
    )

  def compute_heating_rate(
      self,
      rho_xxc: Array,
      q_t: Array,
      q_liq: Array,
      q_ice: Array,
      q_c: Array,
      cloud_r_eff_liq: Array,
      cloud_r_eff_ice: Array,
      temperature: Array,
      sfc_temperature: Array,
      p_ref_xxc: Array,
      sg_map: dict[str, Array],
      use_scan: bool | None = None,
  ) -> dict[str, Array]:
    """Compute the local heating rate due to radiative transfer.

    The optical properties of the layered atmosphere are computed using RRTMGP
    and the two-stream radiative transfer equation is solved for the net fluxes
    at the grid cell faces.  Based on the overall net radiative flux of the grid
    cell, a local heating rate is determined.

    Returns:
      A dictionary containing the following keys:
        rrtmgp_common.KEY_STORED_RADIATION: The heating rate [K/s].
      Optional keys depending on config:
        'rad_heat_sw_3d': The net shortwave radiative heating rate [K/s].
        'rad_heat_lw_3d': The net longwave radiative heating rate [K/s].
        'sw_flux_up_full': Full shortwave upward flux profile [W/m²]
        'sw_flux_down_full': Full shortwave downward flux profile [W/m²]
        'lw_flux_up_full': Full longwave upward flux profile [W/m²]
        'lw_flux_down_full': Full longwave downward flux profile [W/m²]
    """
    # Temperature may have NaNs in the halos (this is intentional).  These NaNs
    # cause problems later on, so fill in the halo values with linear
    # extrapolations.  Note that NaNs in other fields do not cause issues
    # because halos are discarded later on, but there are places where the halos
    # of the temperature are used to determine interior values.
    def fill_halo(f: Array) -> Array:
      # Linear extrapolation into the bottom/top halo cells.  Single
      # concatenate-of-three replaces two scatter ops (and preserves
      # the interior ``f[:, :, 1:-1]`` values verbatim).
      bottom_halo_val = 2 * f[:, :, 1] - f[:, :, 2]
      top_halo_val = 2 * f[:, :, -2] - f[:, :, -3]
      return jnp.concatenate(
          [bottom_halo_val[..., None], f[:, :, 1:-1],
           top_halo_val[..., None]],
          axis=-1,
      )

    temperature = fill_halo(temperature)

    # Sometimes, the simulation can result in q_t or q_c with negative values.
    # Clip these to zero, otherwise there can be extremely deleterious effects
    # in the radiation solver.  E.g., negative relative abundance of gas species
    # and negative values in optical depth and Planck fraction that are
    # inherently nonnegative.
    q_t = jnp.clip(q_t, 0.0, None)
    q_liq = jnp.clip(q_liq, 0.0, None)
    q_ice = jnp.clip(q_ice, 0.0, None)
    q_c = jnp.clip(q_c, 0.0, None)

    # Reconstruct the volume mixing ratio (vmr) of relevant gas species.
    vmr_lib = self.atmospheric_state.vmr
    vmr_fields = (
        lookup_volume_mixing_ratio.reconstruct_vmr_fields_from_pressure(
            vmr_lib, p_ref_xxc
        )
    )
    # Derive the water vapor vmr from the simulation state itself.
    vmr_fields['h2o'] = _humidity_to_volume_mixing_ratio(q_t, q_c)

    # Compute molecules
    molecules_per_area = _air_molecules_per_area(p_ref_xxc, vmr_fields['h2o'])

    # Compute water paths for liquid and ice cloud condensate.
    liq_water_path = _compute_cloud_path(rho_xxc, q_liq, self._dz, sg_map)
    ice_water_path = _compute_cloud_path(rho_xxc, q_ice, self._dz, sg_map)

    lw_fluxes = two_stream.solve_lw(
        p_ref_xxc,
        temperature,
        molecules_per_area,
        self.optics_lib,
        self.atmospheric_state,
        vmr_fields,
        sfc_temperature,
        cloud_r_eff_liq=cloud_r_eff_liq,
        cloud_path_liq=liq_water_path,
        cloud_r_eff_ice=cloud_r_eff_ice,
        cloud_path_ice=ice_water_path,
        use_scan=use_scan,
    )
    sw_fluxes = two_stream.solve_sw(
        p_ref_xxc,
        temperature,
        molecules_per_area,
        self.optics_lib,
        self.atmospheric_state,
        vmr_fields,
        cloud_r_eff_liq=cloud_r_eff_liq,
        cloud_path_liq=liq_water_path,
        cloud_r_eff_ice=cloud_r_eff_ice,
        cloud_path_ice=ice_water_path,
        use_scan=use_scan,
    )

    # Compute the heating rate in K/s.
    lw_heating_rate = two_stream.compute_heating_rate(
        lw_fluxes['flux_net'], p_ref_xxc
    )
    sw_heating_rate = two_stream.compute_heating_rate(
        sw_fluxes['flux_net'], p_ref_xxc
    )
    # Compute the total heating rate (temperature tendency due to radiation).
    heating_rate = lw_heating_rate + sw_heating_rate

    output = {rrtmgp_common.KEY_STORED_RADIATION: heating_rate}

    # add LW, SW heating rate or fluxes if desired.
    if self._save_lw_sw_heating_rates:
      output['rad_heat_sw_3d'] = sw_heating_rate
      output['rad_heat_lw_3d'] = lw_heating_rate

    # Compute diagnostics, if desired.
    lw_flux_down = lw_fluxes['flux_down']
    lw_flux_up = lw_fluxes['flux_up']
    sw_flux_down = sw_fluxes['flux_down']
    sw_flux_up = sw_fluxes['flux_up']
    hw = 1  # halo width.

    # Add full flux profiles (remove surface halos)
    output['sw_flux_up_full'] = sw_flux_up[:, :, hw:]  # Full profile (..., nlev+1)
    output['sw_flux_down_full'] = sw_flux_down[:, :, hw:]  # Full profile (..., nlev+1)
    output['lw_flux_up_full'] = lw_flux_up[:, :, hw:]  # Full profile (..., nlev+1)
    output['lw_flux_down_full'] = lw_flux_down[:, :, hw:]  # Full profile (..., nlev+1)

    # 2D diagnostics
    if (v := 'surf_lw_flux_down_2d_xy') in self._diagnostic_fields:
      output[v] = lw_flux_down[:, :, hw]
    if (v := 'surf_lw_flux_up_2d_xy') in self._diagnostic_fields:
      output[v] = lw_flux_up[:, :, hw]
    if (v := 'surf_sw_flux_down_2d_xy') in self._diagnostic_fields:
      output[v] = sw_flux_down[:, :, hw]
    if (v:= 'surf_sw_flux_up_2d_xy') in self._diagnostic_fields:
      output[v] = sw_flux_up[:, :, hw]
    # Add clear sky surf

    if (v := 'toa_sw_flux_incoming_2d_xy') in self._diagnostic_fields:
      output[v] = sw_flux_down[:, :, -hw]
    if (v := 'toa_sw_flux_outgoing_2d_xy') in self._diagnostic_fields:
      output[v] = sw_flux_up[:, :, -hw]
    if (v := 'toa_lw_flux_outgoing_2d_xy') in self._diagnostic_fields:
      output[v] = lw_flux_up[:, :, -hw]
    # Add clear sky toa

    # 1D diagnostics
    if (v := 'rad_heat_lw_1d_z') in self._diagnostic_fields:
      output[v] = _horiz_mean(lw_heating_rate)
    if (v := 'rad_heat_sw_1d_z') in self._diagnostic_fields:
      output[v] = _horiz_mean(sw_heating_rate)

    # Compute clear-sky radiative transfer and diagnostics, if desired.
    if self._do_clear_sky:
      lw_fluxes_clearsky = two_stream.solve_lw(
          p_ref_xxc,
          temperature,
          molecules_per_area,
          self.optics_lib,
          self.atmospheric_state,
          vmr_fields,
          sfc_temperature,
          cloud_r_eff_liq=None,
          cloud_path_liq=None,
          cloud_r_eff_ice=None,
          cloud_path_ice=None,
          use_scan=use_scan,
      )
      sw_fluxes_clearsky = two_stream.solve_sw(
          p_ref_xxc,
          temperature,
          molecules_per_area,
          self.optics_lib,
          self.atmospheric_state,
          vmr_fields,
          cloud_r_eff_liq=None,
          cloud_path_liq=None,
          cloud_r_eff_ice=None,
          cloud_path_ice=None,
          use_scan=use_scan,
      )
      # Compute the heating rate in K/s.
      lw_heating_rate_clearsky = two_stream.compute_heating_rate(
          lw_fluxes_clearsky['flux_net'], p_ref_xxc
      )
      sw_heating_rate_clearsky = two_stream.compute_heating_rate(
          sw_fluxes_clearsky['flux_net'], p_ref_xxc
      )

      if self._save_lw_sw_heating_rates:
        output['rad_heat_sw_clearsky_3d'] = sw_heating_rate_clearsky
        output['rad_heat_lw_clearsky_3d'] = lw_heating_rate_clearsky

      lw_flux_down_clearsky = lw_fluxes_clearsky['flux_down']
      lw_flux_up_clearsky = lw_fluxes_clearsky['flux_up']
      sw_flux_down_clearsky = sw_fluxes_clearsky['flux_down']
      sw_flux_up_clearsky = sw_fluxes_clearsky['flux_up']

      # Add clear-sky full flux profiles (remove surface halos)
      output['sw_flux_up_clearsky_full'] = sw_flux_up_clearsky[:, :, hw:]  # Full profile (..., nlev+1)
      output['sw_flux_down_clearsky_full'] = sw_flux_down_clearsky[:, :, hw:]  # Full profile (..., nlev+1)
      output['lw_flux_up_clearsky_full'] = lw_flux_up_clearsky[:, :, hw:]  # Full profile (..., nlev+1)
      output['lw_flux_down_clearsky_full'] = lw_flux_down_clearsky[:, :, hw:]  # Full profile (..., nlev+1)

      # 2D diagnostics
      if (v := 'surf_lw_flux_down_clearsky_2d_xy') in self._diagnostic_fields:
        output[v] = lw_flux_down_clearsky[:, :, hw]
      if (v := 'surf_lw_flux_up_clearsky_2d_xy') in self._diagnostic_fields:
        output[v] = lw_flux_up_clearsky[:, :, hw]
      if (v := 'surf_sw_flux_down_clearsky_2d_xy') in self._diagnostic_fields:
        output[v] = sw_flux_down_clearsky[:, :, hw]
      if (v := 'surf_sw_flux_up_clearsky_2d_xy') in self._diagnostic_fields:
        output[v] = sw_flux_up_clearsky[:, :, hw]

      df = self._diagnostic_fields
      if (v := 'toa_sw_flux_outgoing_clearsky_2d_xy') in df:
        output[v] = sw_flux_up_clearsky[:, :, -hw]
      if (v := 'toa_lw_flux_outgoing_clearsky_2d_xy') in df:
        output[v] = lw_flux_up_clearsky[:, :, -hw]

      # 1D diagnostics
      if (v := 'rad_heat_lw_clearsky_1d_z') in self._diagnostic_fields:
        output[v] = _horiz_mean(lw_heating_rate_clearsky)
      if (v := 'rad_heat_sw_clearsky_1d_z') in self._diagnostic_fields:
        output[v] = _horiz_mean(sw_heating_rate_clearsky)

    return output

  # =========================================================================
  # legoESM integration API
  # =========================================================================

  @staticmethod
  def _cache_key(config):
      """Compute a hashable cache key from an RRTMGPConfig.

      .. deprecated:: issue #273 follow-up
         This key drives **both** the heavy optics-table cache
         (``_legoesm_optics_cache`` — keyed only on table-shape
         inputs) and the lighter solver-instance cache
         (``_instance_cache`` in ``rrtmgp_radiation.py`` — needs to
         additionally invalidate on solver-behavior fields like
         ``use_scan``).  Sharing one key for both caches meant a
         first call with ``use_scan=False`` would cache a solver
         instance whose ``_config.use_scan`` is ``False``; a later
         call with ``use_scan=True`` (or the new ``None`` auto-pick)
         would reuse that stale instance and silently keep running
         the for-loop path — defeating the GPU scan auto-pick this
         module ships.

         Production code should call the more specific keys:

         * ``_optics_cache_key(config)`` for the optics tables
           (omits behavior fields that don't change tables).
         * ``_instance_cache_key(config)`` for solver instances
           (includes behavior fields like ``use_scan``).

         ``_cache_key`` remains as a backward-compatible alias for
         ``_optics_cache_key``.
      """
      return RRTMGP._optics_cache_key(config)

  @staticmethod
  def _optics_cache_key(config):
      """Hashable key for the optics-table cache.

      Includes only fields that change the loaded NetCDF tables
      (gas files, cloud files, table-shape inputs) plus the active
      x64 precision (table dtype depends on it).  Solver-behavior
      fields like ``use_scan`` deliberately omitted — the tables
      themselves are independent of how the solver traverses them.
      """
      import jax
      x64 = bool(jax.config.jax_enable_x64)
      return (config.lw_gas_file, config.sw_gas_file,
              config.lw_cloud_file, config.sw_cloud_file,
              config.include_clouds,
              config.co2_ppmv, config.ch4_ppbv, config.n2o_ppbv,
              x64)

  @staticmethod
  def _instance_cache_key(config):
      """Hashable key for the solver-instance cache.

      Extends the optics key with the behavior fields that change
      the *result* of ``solve_columns`` (or its compile-time graph)
      without changing the optics tables.  Critically includes
      ``use_scan`` so the issue-#273 GPU auto-pick (``None`` ⇒ scan
      on GPU/TPU, for-loop on CPU) is honored even when an earlier
      call cached an explicit ``False``.  Also includes
      ``use_optimal_angle`` (iter-2): the optimal-angle path traces
      a different two-stream graph (per-band/per-column secant) than
      the fixed-1.66 path, so a config flip must rebuild the solver
      instance.
      """
      return (
          RRTMGP._optics_cache_key(config),
          config.use_scan,
          getattr(config, "use_optimal_angle", False),
      )

  @staticmethod
  def _build_optics_and_vmr(config):
      """Build (or retrieve from cache) optics scheme and VMR library.

      Parameters
      ----------
      config : RRTMGPConfig
          legoESM radiation configuration.

      Returns
      -------
      (optics_lib, vmr_lib)
      """
      key = RRTMGP._optics_cache_key(config)
      if key not in _legoesm_optics_cache:
          lw_file = config.lw_gas_file or _DEFAULT_LW_GAS
          sw_file = config.sw_gas_file or _DEFAULT_SW_GAS
          lw_cloud = config.lw_cloud_file or _DEFAULT_LW_CLOUD
          sw_cloud = config.sw_cloud_file or _DEFAULT_SW_CLOUD

          rrtm_optics = RRTMOpticsConfig(
              longwave_nc_filepath=lw_file,
              shortwave_nc_filepath=sw_file,
              cloud_longwave_nc_filepath=lw_cloud,
              cloud_shortwave_nc_filepath=sw_cloud,
          )
          optics_params = OpticsParameters(optics=rrtm_optics)

          # Build VMR library with global means from legoESM config.
          global_means = {
              optics_constants.DRY_AIR_KEY: optics_constants.DRY_AIR_VMR,
              "co2": config.co2_ppmv * 1.0e-6,
              "ch4": config.ch4_ppbv * 1.0e-9,
              "n2o": config.n2o_ppbv * 1.0e-9,
              "o2": 0.20948,
              "n2": 0.78084,
              "co": 1.5e-7,
              "ccl4": 7.5e-11,
              "cfc11": 2.2e-10,
              "cfc12": 5.0e-10,
              "cfc22": 2.4e-10,
              "cf4": 8.5e-11,
              "no2": 3.0e-10,
          }
          vmr_lib = LookupVolumeMixingRatio(
              global_means=global_means, profiles=None,
          )

          optics_lib = optics_factory(optics_params, vmr_lib)
          _legoesm_optics_cache[key] = (optics_lib, vmr_lib)
      return _legoesm_optics_cache[key]

  @classmethod
  def from_legoesm_config(cls, config) -> 'RRTMGP':
      """Construct an RRTMGP solver from legoESM's RRTMGPConfig.

      Parameters
      ----------
      config : RRTMGPConfig
          legoESM radiation configuration.

      Returns
      -------
      RRTMGP
          Ready-to-use solver instance.
      """
      instance = object.__new__(cls)
      optics_lib, vmr_lib = cls._build_optics_and_vmr(config)
      # Minimal atmospheric state with VMR library; zenith/albedo/emissivity
      # are overridden per-call in solve_columns().
      instance.atmospheric_state = atmospheric_state.AtmosphericState(
          sfc_emis=config.sfc_emissivity,
          sfc_alb=config.sfc_albedo,
          zenith=0.0,
          irrad=config.S_0,
          vmr=vmr_lib,
          toa_flux_lw=0.0,
      )
      instance.optics_lib = optics_lib
      instance._config = config
      # Unused by solve_columns() but set for compatibility with
      # compute_heating_rate().
      instance._dz = 0.0
      instance._diagnostic_fields = ()
      instance._save_lw_sw_heating_rates = False
      instance._do_clear_sky = False
      return instance

  @classmethod
  def preload(cls, config) -> None:
      """Preload optics tables outside JIT."""
      cls._build_optics_and_vmr(config)

  @classmethod
  def preload_mpi(cls, config) -> None:
      """MPI-aware preload: rank 0 reads, broadcasts to others."""
      try:
          from mpi4py import MPI
      except ImportError:
          cls.preload(config)
          return
      comm = MPI.COMM_WORLD
      rank = comm.Get_rank()
      if rank == 0:
          cls.preload(config)
      key = cls._optics_cache_key(config)
      data = _legoesm_optics_cache.get(key) if rank == 0 else None
      data = comm.bcast(data, root=0)
      if rank != 0:
          _legoesm_optics_cache[key] = data

  def solve_columns(
      self,
      T: jnp.ndarray,
      p_full: jnp.ndarray,
      p_half: jnp.ndarray,
      sfc_temperature: jnp.ndarray,
      q_v: jnp.ndarray,
      cos_zenith: jnp.ndarray,
      sfc_albedo: jnp.ndarray | float | None = None,
      sfc_emissivity: jnp.ndarray | float | None = None,
      o3_vmr: jnp.ndarray | None = None,
      cloud_path_liq: jnp.ndarray | None = None,
      cloud_path_ice: jnp.ndarray | None = None,
      cloud_r_eff_liq: jnp.ndarray | None = None,
      cloud_r_eff_ice: jnp.ndarray | None = None,
      cloud_fraction: jnp.ndarray | None = None,
      aerosol_optical_depth: jnp.ndarray | None = None,
      solar_spectral_fraction: jnp.ndarray | None = None,
      ghg_vmr_override: dict | None = None,
  ):
      """Compute radiation for legoESM column arrays.

      This is the canonical entry point for the RRTMGP solver when used
      from legoESM's physics integration layer.  It handles:

      1. Reshaping (ncol, nlev) arrays to jax-rrtmgp's (ncol, 1, nlev+2)
      2. Adding 1-cell vertical halos
      3. Building VMR dict from config concentrations
      4. Computing atmospheric state and solving two-stream
      5. Stripping halos and reshaping back

      Parameters
      ----------
      T : jnp.ndarray
          Temperature at full levels (ncol, nlev) [K].
      p_full : jnp.ndarray
          Pressure at full levels (ncol, nlev) [Pa].
      p_half : jnp.ndarray
          Pressure at interface levels (ncol, nlev+1) [Pa].
      sfc_temperature : jnp.ndarray
          Surface temperature (ncol,) [K].
      q_v : jnp.ndarray
          Water vapor specific humidity (ncol, nlev) [kg/kg].
      cos_zenith : jnp.ndarray
          Cosine of solar zenith angle (ncol,).
      sfc_albedo : jnp.ndarray | float | None
          Per-column surface albedo override.
      sfc_emissivity : jnp.ndarray | float | None
          Per-column surface emissivity override.
      o3_vmr : jnp.ndarray | None
          External ozone VMR (ncol, nlev), index 0 = TOA.
      cloud_path_liq : jnp.ndarray | None
          Liquid water path per layer (ncol, nlev) [kg/m^2].
      cloud_path_ice : jnp.ndarray | None
          Ice water path per layer (ncol, nlev) [kg/m^2].
      cloud_r_eff_liq : jnp.ndarray | None
          Liquid cloud effective radius (ncol, nlev) [m].
      cloud_r_eff_ice : jnp.ndarray | None
          Ice cloud effective radius (ncol, nlev) [m].
      aerosol_optical_depth : jnp.ndarray | None
          Prescribed aerosol optical depth per layer (ncol, nlev).
      solar_spectral_fraction : jnp.ndarray | None
          Per-g-point solar source weights (ngpt_sw,).
      ghg_vmr_override : dict | None
          Runtime GHG VMR overrides (e.g. ``{"co2": 4.15e-4}``).

      Returns
      -------
      RadiationOutput
          Fluxes and heating rates.
      """
      from legoesm.atmosphere.physics.radiation.output import RadiationOutput

      config = self._config
      ncol, nlev = T.shape

      # --- 0. Determine working dtype ---
      # The optics tables are loaded at whatever precision JAX was configured
      # with at load time (float32 if x64 off, float64 if x64 on).  Promote
      # all inputs to match so that table lookups, lax.cond branches, and
      # lax.scan carries have consistent dtypes throughout the solver.
      _table_dtype = self.optics_lib.gas_optics_lw.kmajor.dtype
      T = T.astype(_table_dtype)
      p_full = p_full.astype(_table_dtype)
      p_half = p_half.astype(_table_dtype)
      q_v = q_v.astype(_table_dtype)
      sfc_temperature = jnp.asarray(sfc_temperature).astype(_table_dtype)
      cos_zenith = jnp.asarray(cos_zenith).astype(_table_dtype)

      # --- 1. Reshape (ncol, nlev) -> (ncol, 1, nlev+2) with halos ---
      T_3d = _add_halos(T[:, None, ::-1])
      p_3d = _add_halos(p_full[:, None, ::-1])
      p_3d = jnp.clip(p_3d, 1.0, None)
      # Upper-clip q_v strictly below 1 so the (1 - q_v) denominator in
      # the VMR conversion is bounded away from zero.  0.99 is far above
      # any physically plausible specific humidity (peak tropical surface
      # values are ~0.025); the bound only ever fires on numerical
      # pathology during spin-up and prevents singular/negative VMRs.
      #
      # **Clip AFTER ``_add_halos``**: linear extrapolation of a steep
      # boundary profile can produce halo values OUTSIDE [0, 0.99]
      # (e.g. q_v = [0.99, 0.0, ...] extrapolates to halo = 1.98), which
      # would yield singular / negative h2o_vmr via the
      # ``1 − q_v`` denominator.  Clipping the interior alone is not
      # enough — the halo cells are passed straight to the RRTMGP
      # solve.  Codex iter-79 stop-time review.
      q_v_3d = _add_halos(q_v[:, None, ::-1])
      q_v_3d = jnp.clip(q_v_3d, 0.0, 0.99)

      # --- 2. Build VMR fields ---
      mol_ratio = constants.R_V / constants.R_D
      h2o_vmr = mol_ratio * q_v_3d / (1.0 - q_v_3d)

      if o3_vmr is not None:
          o3_3d = _add_halos(jnp.clip(o3_vmr, 1.0e-10, None)[:, None, ::-1])
      else:
          o3_3d = _standard_o3_profile(p_3d)

      vmr_fields = {
          "h2o": h2o_vmr,
          "o3": o3_3d,
      }

      if ghg_vmr_override is not None:
          for gas_name, vmr_value in ghg_vmr_override.items():
              vmr_fields[gas_name] = jnp.full_like(p_3d, vmr_value)

      # Exact layer thickness from interface pressures.
      # p_half is (ncol, nlev+1) with index 0 = TOA (sigma=0 → p≈0).
      # dp = p_half[k+1] - p_half[k] is positive (p increases toward surface).
      # Reverse to surface-first order to match T_3d, p_3d, q_v_3d.
      dp_exact = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev) TOA-first
      dp_3d = dp_exact[:, None, ::-1]  # (ncol, 1, nlev) surface-first
      # Pad with halo values (replicate boundary layers)
      dp_3d = jnp.concatenate([
          dp_3d[:, :, :1], dp_3d, dp_3d[:, :, -1:],
      ], axis=2)
      dp_3d = jnp.clip(dp_3d, 1.0, None)

      mol_m_air = (constants.DRY_AIR_MOL_MASS
                   + constants.WATER_MOL_MASS * h2o_vmr)
      molecules = (dp_3d / constants.G) * constants.AVOGADRO / mol_m_air

      # --- 3. Build atmospheric state ---
      optics_lib = self.optics_lib
      vmr_lib = self.atmospheric_state.vmr

      cos_z_col = jnp.clip(cos_zenith, 0.0, 1.0)
      zenith_col = jnp.arccos(cos_z_col)[:, None, None]

      def _resolve_surface_field(override, fallback):
          """Promote a per-column surface field to ``(ncol, 1)``.

          Accepts a Python scalar, a 0-D JAX scalar, or a ``(ncol,)``
          / ``(ncol, 1)`` array.  AIMIP's spatial surface
          parameterization populates ``config.sfc_albedo`` /
          ``config.sfc_emissivity`` with ``(ncol,)`` arrays produced
          by a low-rank lat-lon expansion; this branch preserves the
          legacy scalar path while supporting that AIMIP use case
          without forcing every call site to thread an explicit
          override kwarg through ``_call_radiation_backend``.
          """
          source = override if override is not None else fallback
          val = jnp.asarray(source, dtype=p_3d.dtype)
          if val.ndim == 0:
              return jnp.full((ncol, 1), val, dtype=p_3d.dtype)
          if val.ndim == 1:
              return val.reshape(ncol, 1)
          return val  # already (ncol, 1)

      eff_albedo = _resolve_surface_field(sfc_albedo, config.sfc_albedo)
      eff_emis = _resolve_surface_field(sfc_emissivity, config.sfc_emissivity)

      atmos_state = atmospheric_state.AtmosphericState(
          sfc_emis=eff_emis,
          sfc_alb=eff_albedo,
          zenith=zenith_col,
          irrad=config.S_0,
          vmr=vmr_lib,
          toa_flux_lw=0.0,
      )

      sfc_T_2d = sfc_temperature[:, None]

      # --- Cloud properties ---
      has_clouds = config.include_clouds and (
          cloud_path_liq is not None or cloud_path_ice is not None
      )
      if has_clouds:
          _zero = jnp.zeros((ncol, nlev), dtype=T.dtype)
          _r_min = jnp.full((ncol, nlev), 1.0e-6, dtype=T.dtype)
          _cpl = cloud_path_liq if cloud_path_liq is not None else _zero
          _cpi = cloud_path_ice if cloud_path_ice is not None else _zero
          _crl = cloud_r_eff_liq if cloud_r_eff_liq is not None else _r_min
          _cri = cloud_r_eff_ice if cloud_r_eff_ice is not None else _r_min
          cpl_3d = _add_halos(jnp.clip(_cpl, 0.0, None)[:, None, ::-1])
          cpi_3d = _add_halos(jnp.clip(_cpi, 0.0, None)[:, None, ::-1])
          crl_3d = _add_halos(jnp.clip(_crl, 1.0e-6, None)[:, None, ::-1])
          cri_3d = _add_halos(jnp.clip(_cri, 1.0e-6, None)[:, None, ::-1])
          if cloud_fraction is not None:
              cf_3d = _add_halos(jnp.clip(cloud_fraction, 0.0, 1.0)[:, None, ::-1])
          else:
              cf_3d = None
      else:
          cpl_3d = cpi_3d = crl_3d = cri_3d = cf_3d = None

      # Optional aerosol optical depth
      if aerosol_optical_depth is not None:
          aerosol_od_3d = _add_halos(
              jnp.clip(aerosol_optical_depth, 0.0, None)[:, None, ::-1],
          )
      else:
          aerosol_od_3d = None

      # Optional spectral solar forcing
      if solar_spectral_fraction is not None:
          solar_weights = jnp.clip(jnp.asarray(solar_spectral_fraction), 0.0, None)
          denom = jnp.maximum(jnp.sum(solar_weights), 1.0e-30)
          solar_weights = solar_weights / denom
          if solar_weights.shape[0] != optics_lib.n_gpt_sw:
              raise ValueError(
                  "solar_spectral_fraction has wrong length: "
                  f"{solar_weights.shape[0]} (expected {optics_lib.n_gpt_sw})",
              )
      else:
          solar_weights = None

      # --- 4. Solve LW ---
      lw_fluxes = two_stream.solve_lw(
          p_3d,
          T_3d,
          molecules,
          optics_lib,
          atmos_state,
          vmr_fields,
          sfc_T_2d,
          cloud_r_eff_liq=crl_3d,
          cloud_path_liq=cpl_3d,
          cloud_r_eff_ice=cri_3d,
          cloud_path_ice=cpi_3d,
          cloud_fraction=cf_3d,
          use_scan=config.use_scan,
          use_optimal_angle=getattr(config, "use_optimal_angle", False),
      )

      # --- 5. Solve SW ---
      sw_fluxes = two_stream.solve_sw(
          p_3d,
          T_3d,
          molecules,
          optics_lib,
          atmos_state,
          vmr_fields,
          cloud_r_eff_liq=crl_3d,
          cloud_path_liq=cpl_3d,
          cloud_r_eff_ice=cri_3d,
          cloud_path_ice=cpi_3d,
          cloud_fraction=cf_3d,
          aerosol_optical_depth=aerosol_od_3d,
          aerosol_single_scattering_albedo=config.aerosol_ssa,
          aerosol_asymmetry_factor=config.aerosol_g,
          solar_fraction_by_gpt=solar_weights,
          use_scan=config.use_scan,
      )

      # --- 6. Compute heating rates using exact layer thickness ---
      lw_hr_3d = two_stream.compute_heating_rate(
          lw_fluxes['flux_net'], p_3d, dp=dp_3d,
      )
      sw_hr_3d = two_stream.compute_heating_rate(
          sw_fluxes['flux_net'], p_3d, dp=dp_3d,
      )

      # --- 7. Strip halos, flip back to legoESM convention, reshape ---
      hw = 1
      lw_up = lw_fluxes['flux_up'][:, 0, hw:][:, ::-1]
      lw_down = lw_fluxes['flux_down'][:, 0, hw:][:, ::-1]
      sw_up = sw_fluxes['flux_up'][:, 0, hw:][:, ::-1]
      sw_down = sw_fluxes['flux_down'][:, 0, hw:][:, ::-1]

      lw_hr = lw_hr_3d[:, 0, hw:-hw][:, ::-1]
      sw_hr = sw_hr_3d[:, 0, hw:-hw][:, ::-1]

      return RadiationOutput(
          lw_flux_up=lw_up,
          lw_flux_down=lw_down,
          sw_flux_up=sw_up,
          sw_flux_down=sw_down,
          heating_rate=lw_hr + sw_hr,
          lw_heating_rate=lw_hr,
          sw_heating_rate=sw_hr,
      )
