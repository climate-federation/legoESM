"""Wrapper around jax-rrtmgp for correlated-k radiation.

This module provides an optional dependency on jax-rrtmgp
(https://github.com/climate-analytics-lab/jax-rrtmgp). If the package
is not installed, importing this module will work but calling
`rrtmgp_radiation` will raise an ImportError with a clear message.

The wrapper handles:
1. Reshaping legoESM's (ncol, nlev) arrays to jax-rrtmgp's (ncol, 1, nlev)
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

import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.output import RadiationOutput

# Guarded import — jax-rrtmgp is optional
_RRTMGP_AVAILABLE = False
try:
    from jax_rrtmgp import (
        AtmosphericState,
        optics_factory,
        two_stream,
    )
    _RRTMGP_AVAILABLE = True
except ImportError:
    pass

# Module-level optics cache
_optics_cache: dict = {}


def _check_rrtmgp():
    """Raise a clear error if jax-rrtmgp is not installed."""
    if not _RRTMGP_AVAILABLE:
        raise ImportError(
            "jax-rrtmgp is required for RRTMGP radiation but is not installed. "
            "Install it with: pip install jax-rrtmgp  "
            "(or see https://github.com/climate-analytics-lab/jax-rrtmgp)"
        )


def _get_optics(config: RRTMGPConfig):
    """Get or create cached optics scheme."""
    key = (config.lw_gas_file, config.sw_gas_file,
           config.lw_cloud_file, config.sw_cloud_file,
           config.include_clouds)
    if key not in _optics_cache:
        kwargs = {}
        if config.lw_gas_file:
            kwargs["lw_gas_file"] = config.lw_gas_file
        if config.sw_gas_file:
            kwargs["sw_gas_file"] = config.sw_gas_file
        if config.include_clouds:
            if config.lw_cloud_file:
                kwargs["lw_cloud_file"] = config.lw_cloud_file
            if config.sw_cloud_file:
                kwargs["sw_cloud_file"] = config.sw_cloud_file
        _optics_cache[key] = optics_factory(**kwargs)
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
    _check_rrtmgp()

    ncol, nlev = T.shape

    # --- 1. Reshape (ncol, nlev) -> (ncol, 1, nlev) for jax-rrtmgp ---
    T_3d = T[:, None, :]
    p_full_3d = p_full[:, None, :]
    p_half_3d = p_half[:, None, :]
    q_v_3d = q_v[:, None, :]

    # --- 2. Build VMR dict ---
    # Water vapor: convert mass mixing ratio to volume mixing ratio
    # VMR_h2o = q_v / epsilon / (1 + q_v / epsilon)  ≈ q_v / epsilon for small q
    h2o_vmr = q_v_3d / constants.epsilon
    vmr = {
        "h2o": h2o_vmr,
        "co2": jnp.full_like(T_3d, config.co2_ppmv * 1.0e-6),
        "ch4": jnp.full_like(T_3d, config.ch4_ppbv * 1.0e-9),
        "n2o": jnp.full_like(T_3d, config.n2o_ppbv * 1.0e-9),
        "o3": _standard_o3_profile(p_full_3d),
    }

    # --- 3. Get optics and solve ---
    optics = _get_optics(config)

    # Surface properties
    sfc_emis = jnp.full((ncol, 1), config.sfc_emissivity)
    sfc_alb_direct = jnp.full((ncol, 1), config.sfc_albedo)
    sfc_alb_diffuse = sfc_alb_direct
    cos_zen_2d = cos_zenith[:, None]

    # Build atmospheric state
    atm = AtmosphericState(
        temperature=T_3d,
        pressure=p_full_3d,
        pressure_hl=p_half_3d,
        vmr=vmr,
        surface_temperature=sfc_temperature[:, None],
    )

    # Solve LW
    lw_result = two_stream.solve_lw(
        atm, optics,
        surface_emissivity=sfc_emis,
        use_scan=config.use_scan,
    )

    # Solve SW
    sw_result = two_stream.solve_sw(
        atm, optics,
        cos_zenith=cos_zen_2d,
        toa_flux=jnp.full((ncol, 1), config.S_0),
        surface_albedo_direct=sfc_alb_direct,
        surface_albedo_diffuse=sfc_alb_diffuse,
        use_scan=config.use_scan,
    )

    # --- 4. Extract fluxes and reshape back to (ncol, nlev+1) ---
    lw_up = lw_result.flux_up[:, 0, :]       # (ncol, nlev+1)
    lw_down = lw_result.flux_down[:, 0, :]
    sw_up = sw_result.flux_up[:, 0, :]
    sw_down = sw_result.flux_down[:, 0, :]

    # --- 5. Compute heating rates ---
    def _heating_rate(f_up, f_down, p_hl):
        F_net = f_down - f_up
        dF = F_net[:, 1:] - F_net[:, :-1]
        dp = p_hl[:, 1:] - p_hl[:, :-1]
        dp = jnp.clip(dp, 1.0, None)
        return (constants.g / constants.c_pd) * dF / dp

    lw_hr = _heating_rate(lw_up, lw_down, p_half)
    sw_hr = _heating_rate(sw_up, sw_down, p_half)

    return RadiationOutput(
        lw_flux_up=lw_up,
        lw_flux_down=lw_down,
        sw_flux_up=sw_up,
        sw_flux_down=sw_down,
        heating_rate=lw_hr + sw_hr,
        lw_heating_rate=lw_hr,
        sw_heating_rate=sw_hr,
    )
