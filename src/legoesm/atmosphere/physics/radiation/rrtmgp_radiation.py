"""RRTMGP correlated-k radiation backend.

This module uses a bundled copy of jax-rrtmgp
(https://github.com/climate-analytics-lab/jax-rrtmgp, Apache 2.0 license)
by Jeff Parker (Google), Duncan Watson-Parris (UCSD), and
Juan Nathaniel (Columbia University).

This backend handles:
1. Reshaping legoESM's (ncol, nlev) arrays to jax-rrtmgp's (ncol, 1, nlev+2)
2. Adding 1-cell vertical halos (jax-rrtmgp convention)
3. Building VMR dict from config concentrations
4. Computing atmospheric state and solving two-stream
5. Stripping halos and reshaping back

References
----------
- Pincus, R. et al. (2019). Balancing Accuracy, Efficiency, and
  Flexibility in Radiation Calculations for Dynamical Models.
  JAMES, 11, 3074-3089.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.output import RadiationOutput
from legoesm.atmosphere.physics.radiation.rrtmgp.config.radiative_transfer import (
    OpticsParameters,
    RRTMOptics,
)
from legoesm.atmosphere.physics.radiation.rrtmgp.optics.atmospheric_state import (
    AtmosphericState,
)
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import constants as rrtmgp_optics_constants
from legoesm.atmosphere.physics.radiation.rrtmgp.optics.lookup_volume_mixing_ratio import (
    LookupVolumeMixingRatio,
)
from legoesm.atmosphere.physics.radiation.rrtmgp.optics.optics import optics_factory
from legoesm.atmosphere.physics.radiation.rrtmgp.rte import two_stream
from legoesm.atmosphere.physics.radiation.rrtmgp import constants as rrtmgp_constants
from legoesm.atmosphere.physics.radiation.rrtmgp import kernel_ops

# Path to bundled RRTMGP NetCDF data files
_DATA_DIR = Path(__file__).parent / "rrtmgp" / "optics" / "rrtmgp_data"

# Default data file paths (128/112 g-point versions for speed)
_DEFAULT_LW_GAS = str(_DATA_DIR / "rrtmgp-gas-lw-g128.nc")
_DEFAULT_SW_GAS = str(_DATA_DIR / "rrtmgp-gas-sw-g112.nc")
_DEFAULT_LW_CLOUD = str(_DATA_DIR / "cloudysky_lw.nc")
_DEFAULT_SW_CLOUD = str(_DATA_DIR / "cloudysky_sw.nc")

# Module-level optics cache
_optics_cache: dict = {}


def _get_optics(config: RRTMGPConfig):
    """Get or create cached optics scheme and VMR library.

    This function should be called outside model-step JIT traces so the
    cached objects are created from concrete arrays rather than tracers.
    """
    key = (config.lw_gas_file, config.sw_gas_file,
           config.lw_cloud_file, config.sw_cloud_file,
           config.include_clouds,
           config.co2_ppmv, config.ch4_ppbv, config.n2o_ppbv)
    if key not in _optics_cache:
        lw_file = config.lw_gas_file or _DEFAULT_LW_GAS
        sw_file = config.sw_gas_file or _DEFAULT_SW_GAS
        lw_cloud = config.lw_cloud_file or _DEFAULT_LW_CLOUD
        sw_cloud = config.sw_cloud_file or _DEFAULT_SW_CLOUD

        rrtm_optics = RRTMOptics(
            longwave_nc_filepath=lw_file,
            shortwave_nc_filepath=sw_file,
            cloud_longwave_nc_filepath=lw_cloud,
            cloud_shortwave_nc_filepath=sw_cloud,
        )
        optics_params = OpticsParameters(optics=rrtm_optics)

        # Build VMR library with global means from legoESM config.
        # Include all gases that appear in the RRTMGP absorption tables;
        # omitting them causes their VMR to default to zero, dropping
        # real absorbers/scatterers (e.g. O2 A-band, CFC window absorption).
        global_means = {
            rrtmgp_optics_constants.DRY_AIR_KEY: rrtmgp_optics_constants.DRY_AIR_VMR,
            "co2": config.co2_ppmv * 1.0e-6,
            "ch4": config.ch4_ppbv * 1.0e-9,
            "n2o": config.n2o_ppbv * 1.0e-9,
            # Major atmospheric constituents
            "o2": 0.20948,
            "n2": 0.78084,
            # Minor trace gases (present-day approximate global means)
            "co": 1.5e-7,       # ~150 ppbv
            "ccl4": 7.5e-11,    # ~75 pptv (declining)
            "cfc11": 2.2e-10,   # ~220 pptv
            "cfc12": 5.0e-10,   # ~500 pptv
            "cfc22": 2.4e-10,   # ~240 pptv
            "cf4": 8.5e-11,     # ~85 pptv
            "no2": 3.0e-10,     # ~0.3 ppbv (stratospheric column mean)
        }
        vmr_lib = LookupVolumeMixingRatio(global_means=global_means, profiles=None)

        optics_lib = optics_factory(optics_params, vmr_lib)
        _optics_cache[key] = (optics_lib, vmr_lib)
    return _optics_cache[key]


def preload_rrtmgp_optics(config: RRTMGPConfig) -> None:
    """Preload optics tables/cache outside JIT for stable runtime reuse."""
    _get_optics(config)


def preload_rrtmgp_optics_mpi(config: RRTMGPConfig) -> None:
    """MPI-aware preload: rank 0 reads NetCDF files, broadcasts to others.

    Avoids N parallel filesystem reads of ~90 JAX arrays (tens of MB)
    by having only rank 0 read the NetCDF data, then using MPI broadcast
    (pickle serialization) to distribute the optics objects.

    Falls back to per-rank loading if mpi4py is not available.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        preload_rrtmgp_optics(config)
        return

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Rank 0 loads from NetCDF files.
    if rank == 0:
        preload_rrtmgp_optics(config)

    # Build the cache key (must match _get_optics).
    key = (config.lw_gas_file, config.sw_gas_file,
           config.lw_cloud_file, config.sw_cloud_file,
           config.include_clouds,
           config.co2_ppmv, config.ch4_ppbv, config.n2o_ppbv)

    # Broadcast the optics objects (pickle-serialized, includes JAX arrays).
    data = _optics_cache.get(key) if rank == 0 else None
    data = comm.bcast(data, root=0)

    if rank != 0:
        _optics_cache[key] = data


def _standard_o3_profile(p_full: jnp.ndarray) -> jnp.ndarray:
    """Simple climatological ozone profile (VMR).

    Based on a fit to the US Standard Atmosphere 1976 ozone profile.
    Returns volume mixing ratio (dimensionless).

    Parameters
    ----------
    p_full : jnp.ndarray
        Pressure at full levels [Pa], any shape.

    Returns
    -------
    jnp.ndarray
        O3 volume mixing ratio, same shape as p_full.
    """
    p_hPa = p_full / 100.0
    # Simple Gaussian-like profile peaking around 10 hPa
    o3 = 8.0e-6 * jnp.exp(-0.5 * ((jnp.log(p_hPa) - jnp.log(10.0)) / 1.5) ** 2)
    return jnp.clip(o3, 1.0e-10, None)


def _add_halos(f_3d: jnp.ndarray) -> jnp.ndarray:
    """Add 1-cell vertical halos via linear extrapolation.

    Input shape (ncol, 1, nlev), output shape (ncol, 1, nlev+2).
    """
    bottom_halo = 2 * f_3d[:, :, 0:1] - f_3d[:, :, 1:2]
    top_halo = 2 * f_3d[:, :, -1:] - f_3d[:, :, -2:-1]
    return jnp.concatenate([bottom_halo, f_3d, top_halo], axis=2)


def rrtmgp_radiation(
    T: jnp.ndarray,
    p_full: jnp.ndarray,
    p_half: jnp.ndarray,
    sfc_temperature: jnp.ndarray,
    q_v: jnp.ndarray,
    cos_zenith: jnp.ndarray,
    config: RRTMGPConfig,
    sfc_albedo_override: jnp.ndarray | float | None = None,
    sfc_emissivity_override: jnp.ndarray | float | None = None,
    o3_vmr: jnp.ndarray | None = None,
    cloud_path_liq: jnp.ndarray | None = None,
    cloud_path_ice: jnp.ndarray | None = None,
    cloud_r_eff_liq: jnp.ndarray | None = None,
    cloud_r_eff_ice: jnp.ndarray | None = None,
    aerosol_optical_depth: jnp.ndarray | None = None,
    solar_spectral_fraction: jnp.ndarray | None = None,
    ghg_vmr_override: dict | None = None,
) -> RadiationOutput:
    """Compute radiation using jax-rrtmgp.

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
        Water vapor mixing ratio (ncol, nlev) [kg/kg].
    cos_zenith : jnp.ndarray
        Cosine of solar zenith angle (ncol,).
    config : RRTMGPConfig
    sfc_albedo_override : jnp.ndarray | float | None
        If provided, overrides config.sfc_albedo.  For AMIP, this is the
        ice/ocean blended albedo per column (ncol,).  Per-column values
        are passed through to the two-stream solver so that surface
        heterogeneity (e.g. ice vs ocean) is retained in the radiation.
        A scalar is broadcast to all columns.
    sfc_emissivity_override : jnp.ndarray | float | None
        Same as sfc_albedo_override but for surface emissivity.
    o3_vmr : jnp.ndarray | None
        External ozone volume mixing ratio (ncol, nlev) in legoESM
        convention (index 0 = TOA).  If None, the built-in US Standard
        Atmosphere 1976 profile is used.
    cloud_path_liq : jnp.ndarray | None
        Grid-mean liquid water path per layer (ncol, nlev) [kg/m^2].
        In legoESM convention (index 0 = TOA).
    cloud_path_ice : jnp.ndarray | None
        Grid-mean ice water path per layer (ncol, nlev) [kg/m^2].
    cloud_r_eff_liq : jnp.ndarray | None
        Liquid cloud effective radius per layer (ncol, nlev) [m].
    cloud_r_eff_ice : jnp.ndarray | None
        Ice cloud effective radius per layer (ncol, nlev) [m].
    aerosol_optical_depth : jnp.ndarray | None
        Prescribed aerosol optical depth per layer (ncol, nlev), legoESM
        ordering (index 0 = TOA). If provided, SW extinction/scattering
        is augmented in the radiative transfer solve.
    solar_spectral_fraction : jnp.ndarray | None
        Optional per-g-point solar source weights (ngpt_sw,). If provided,
        this overrides the default RRTMGP solar partitioning.
    ghg_vmr_override : dict or None
        Runtime GHG overrides.  Keys are gas names (``"co2"``, ``"ch4"``,
        ``"n2o"``); values are volume mixing ratios (dimensionless).
        These are injected as 3D ``vmr_fields`` entries, overriding
        the cached global means in the VMR library.

    Returns
    -------
    RadiationOutput
        Fluxes and heating rates.
    """
    ncol, nlev = T.shape

    # --- 1. Reshape (ncol, nlev) -> (ncol, 1, nlev+2) with halos ---
    # jax-rrtmgp convention: index 0 = surface (high p), increasing = upward.
    # legoESM convention: index 0 = top (low p), increasing = downward.
    # Flip vertical axis so surface is at low index.
    T_3d = _add_halos(T[:, None, ::-1])
    p_3d = _add_halos(p_full[:, None, ::-1])
    p_3d = jnp.clip(p_3d, 1.0, None)  # Avoid negative pressure in halos
    q_v_3d = _add_halos(jnp.clip(q_v, 0.0, None)[:, None, ::-1])

    # --- 2. Build VMR fields ---
    # Water vapor: convert mass mixing ratio to volume mixing ratio
    mol_ratio = rrtmgp_constants.R_V / rrtmgp_constants.R_D
    h2o_vmr = mol_ratio * q_v_3d / (1.0 - q_v_3d)

    # Ozone: use external profile if provided, otherwise built-in.
    if o3_vmr is not None:
        # External ozone in legoESM convention (TOA-first) → flip + add halos.
        o3_3d = _add_halos(jnp.clip(o3_vmr, 1.0e-10, None)[:, None, ::-1])
    else:
        o3_3d = _standard_o3_profile(p_3d)

    vmr_fields = {
        "h2o": h2o_vmr,
        "o3": o3_3d,
    }

    # Inject runtime GHG overrides as 3D vmr_fields (bypasses cached optics).
    if ghg_vmr_override is not None:
        for gas_name, vmr_value in ghg_vmr_override.items():
            vmr_fields[gas_name] = jnp.full_like(p_3d, vmr_value)

    # Compute molecules per area (centered difference preserves shape via roll).
    # VMR is n_h2o/n_dry, so molar mass per mole of dry air is M_dry + M_h2o * vmr
    # (matching the bundled rrtmgp.py convention, NOT the mole-fraction formula).
    dp = kernel_ops.centered_difference(p_3d, dim=2)
    mol_m_air = (rrtmgp_constants.DRY_AIR_MOL_MASS
                 + rrtmgp_constants.WATER_MOL_MASS * h2o_vmr)
    molecules = -(dp / rrtmgp_constants.G) * rrtmgp_constants.AVOGADRO / mol_m_air

    # --- 3. Get optics and build atmospheric state ---
    optics_lib, vmr_lib = _get_optics(config)

    # Per-column zenith angle: shape (ncol, 1, 1) so it broadcasts against
    # the (ncol, 1, nlev+2) fields in the two-stream solver.  Clipping
    # cos_zenith to [0, 1] handles nighttime columns (cos <= 0) by
    # clamping to horizon; the solver zeros out nighttime contributions
    # via per-column masking.
    cos_z_col = jnp.clip(cos_zenith, 0.0, 1.0)        # (ncol,)
    zenith_col = jnp.arccos(cos_z_col)[:, None, None]  # (ncol, 1, 1)

    # Surface properties: use overrides if provided (e.g. ice/ocean blend),
    # otherwise fall back to config defaults.  Keep per-column structure
    # so that surface heterogeneity is retained in the radiation solve.
    if sfc_albedo_override is not None:
        eff_albedo = jnp.asarray(sfc_albedo_override, dtype=p_3d.dtype)
        # Ensure shape is (ncol, 1) for broadcasting.
        if eff_albedo.ndim == 0:
            eff_albedo = jnp.broadcast_to(eff_albedo, (ncol,))
        eff_albedo = eff_albedo.reshape(ncol, 1)
    else:
        eff_albedo = jnp.full((ncol, 1), config.sfc_albedo, dtype=p_3d.dtype)

    if sfc_emissivity_override is not None:
        eff_emis = jnp.asarray(sfc_emissivity_override, dtype=p_3d.dtype)
        if eff_emis.ndim == 0:
            eff_emis = jnp.broadcast_to(eff_emis, (ncol,))
        eff_emis = eff_emis.reshape(ncol, 1)
    else:
        eff_emis = jnp.full((ncol, 1), config.sfc_emissivity, dtype=p_3d.dtype)

    atmos_state = AtmosphericState(
        sfc_emis=eff_emis,
        sfc_alb=eff_albedo,
        zenith=zenith_col,
        irrad=config.S_0,
        vmr=vmr_lib,
        toa_flux_lw=0.0,
    )

    sfc_T_2d = sfc_temperature[:, None]  # (ncol, 1)

    # --- Cloud properties: reshape to jax-rrtmgp convention ---
    # Respect config.include_clouds: even if cloud arrays are passed,
    # disable cloud optics when the config says so.
    has_clouds = config.include_clouds and cloud_path_liq is not None
    if has_clouds:
        cpl_3d = _add_halos(jnp.clip(cloud_path_liq, 0.0, None)[:, None, ::-1])
        cpi_3d = _add_halos(jnp.clip(cloud_path_ice, 0.0, None)[:, None, ::-1])
        crl_3d = _add_halos(jnp.clip(cloud_r_eff_liq, 1.0e-6, None)[:, None, ::-1])
        cri_3d = _add_halos(jnp.clip(cloud_r_eff_ice, 1.0e-6, None)[:, None, ::-1])
    else:
        cpl_3d = cpi_3d = crl_3d = cri_3d = None

    # Optional externally prescribed aerosol optical depth.
    if aerosol_optical_depth is not None:
        aerosol_od_3d = _add_halos(jnp.clip(aerosol_optical_depth, 0.0, None)[:, None, ::-1])
    else:
        aerosol_od_3d = None

    # Optional full spectral solar forcing (weights by SW g-point).
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

    # --- 4. Solve LW (vmr_fields keyed by chemical formula; solve_lw reindexes) ---
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
        use_scan=config.use_scan,
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
        aerosol_optical_depth=aerosol_od_3d,
        aerosol_single_scattering_albedo=config.aerosol_ssa,
        aerosol_asymmetry_factor=config.aerosol_g,
        solar_fraction_by_gpt=solar_weights,
        use_scan=config.use_scan,
    )

    # --- 6. Compute heating rates using bundled method ---
    lw_hr_3d = two_stream.compute_heating_rate(lw_fluxes['flux_net'], p_3d)
    sw_hr_3d = two_stream.compute_heating_rate(sw_fluxes['flux_net'], p_3d)

    # --- 7. Strip halos, flip back to legoESM convention, reshape ---
    hw = 1  # halo width
    # Fluxes: strip surface halo, flip back (TOA at index 0)
    lw_up = lw_fluxes['flux_up'][:, 0, hw:][:, ::-1]       # (ncol, nlev+1)
    lw_down = lw_fluxes['flux_down'][:, 0, hw:][:, ::-1]
    sw_up = sw_fluxes['flux_up'][:, 0, hw:][:, ::-1]
    sw_down = sw_fluxes['flux_down'][:, 0, hw:][:, ::-1]

    # Heating rates: strip both halos, flip back
    lw_hr = lw_hr_3d[:, 0, hw:-hw][:, ::-1]                # (ncol, nlev)
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
