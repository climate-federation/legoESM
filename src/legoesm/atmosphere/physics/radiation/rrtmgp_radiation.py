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
    """Get or create cached optics scheme and VMR library."""
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

        # Build VMR library with global means from legoESM config
        global_means = {
            rrtmgp_optics_constants.DRY_AIR_KEY: rrtmgp_optics_constants.DRY_AIR_VMR,
            "co2": config.co2_ppmv * 1.0e-6,
            "ch4": config.ch4_ppbv * 1.0e-9,
            "n2o": config.n2o_ppbv * 1.0e-9,
        }
        vmr_lib = LookupVolumeMixingRatio(global_means=global_means, profiles=None)

        optics_lib = optics_factory(optics_params, vmr_lib)
        _optics_cache[key] = (optics_lib, vmr_lib)
    return _optics_cache[key]


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
    vmr_fields = {
        "h2o": h2o_vmr,
        "o3": _standard_o3_profile(p_3d),
    }

    # Compute molecules per area (centered difference preserves shape via roll)
    dp = kernel_ops.centered_difference(p_3d, dim=2)
    mol_m_air = (rrtmgp_constants.DRY_AIR_MOL_MASS
                 + rrtmgp_constants.WATER_MOL_MASS * h2o_vmr)
    molecules = -(dp / rrtmgp_constants.G) * rrtmgp_constants.AVOGADRO / mol_m_air

    # --- 3. Get optics and build atmospheric state ---
    optics_lib, vmr_lib = _get_optics(config)

    # Compute a mean zenith angle (scalar) — bundled API uses a scalar zenith
    zenith = jnp.arccos(jnp.clip(jnp.mean(cos_zenith), 0.0, 1.0))

    atmos_state = AtmosphericState(
        sfc_emis=config.sfc_emissivity,
        sfc_alb=config.sfc_albedo,
        zenith=zenith,
        irrad=config.S_0,
        vmr=vmr_lib,
        toa_flux_lw=0.0,
    )

    sfc_T_2d = sfc_temperature[:, None]  # (ncol, 1)

    # --- 4. Solve LW (vmr_fields keyed by chemical formula; solve_lw reindexes) ---
    lw_fluxes = two_stream.solve_lw(
        p_3d,
        T_3d,
        molecules,
        optics_lib,
        atmos_state,
        vmr_fields,
        sfc_T_2d,
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
