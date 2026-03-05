"""Two-stream gray radiation (Frierson et al. 2006).

Implements a simplified but physically consistent radiation scheme:

**Longwave (LW):**
  Hemispheric-mean two-stream with latitude- and moisture-dependent
  optical depth following Frierson et al. (2006). The LW sweeps use
  `jax.lax.scan` for JIT-friendliness and differentiability.

**Shortwave (SW):**
  Beer-Lambert absorption with surface reflection (no scattering).

**Heating rate:**
  dT/dt = (g / c_p) * dF_net / dp  at each layer.

References
----------
- Frierson, D. M. W., Held, I. M., & Zurita-Gotor, P. (2006).
  A Gray-Radiation Aquaplanet Moist GCM. Part I: Static Stability
  and Eddy Scale. J. Atmos. Sci., 63, 2548-2566.
- O'Gorman, P. A. & Schneider, T. (2008). The Hydrological Cycle
  over a Wide Range of Climates Simulated with an Idealized GCM.
  J. Climate, 21, 3815-3832.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
from legoesm.atmosphere.physics.radiation.output import RadiationOutput


def _compute_lw_optical_depth(
    p_half: jnp.ndarray,
    p_s: jnp.ndarray,
    lat: jnp.ndarray,
    q_v: jnp.ndarray | None,
    config: GrayRadiationConfig,
) -> jnp.ndarray:
    """Compute LW optical depth per layer.

    tau_dry(sigma, lat) = [f_l*sigma + (1-f_l)*sigma^4] * [tau_e + (tau_p - tau_e)*sin^2(lat)]
    tau_total = tau_dry + tau_moist * q_v
    delta_tau_k = tau(sigma_{k+1/2}) - tau(sigma_{k-1/2})

    Parameters
    ----------
    p_half : jnp.ndarray
        Pressure at interfaces (ncol, nlev+1) [Pa].
    p_s : jnp.ndarray
        Surface pressure (ncol,) [Pa].
    lat : jnp.ndarray
        Latitude (ncol,) [radians].
    q_v : jnp.ndarray or None
        Water vapor mixing ratio (ncol, nlev) [kg/kg].
    config : GrayRadiationConfig

    Returns
    -------
    jnp.ndarray
        Optical depth per layer (ncol, nlev).
    """
    f_l = config.linear_frac
    tau_e = config.tau_equator
    tau_p = config.tau_pole

    # sigma at half levels: (ncol, nlev+1)
    sigma_half = p_half / p_s[:, None]

    # Latitude-dependent reference optical depth
    sin2_lat = jnp.sin(lat) ** 2
    tau_ref = tau_e + (tau_p - tau_e) * sin2_lat  # (ncol,)

    # Cumulative optical depth at each interface
    tau_at_half = (
        f_l * sigma_half + (1.0 - f_l) * sigma_half ** 4
    ) * tau_ref[:, None]  # (ncol, nlev+1)

    # Optical depth per layer (dry)
    dtau_dry = tau_at_half[:, 1:] - tau_at_half[:, :-1]  # (ncol, nlev)
    dtau_dry = jnp.maximum(dtau_dry, 0.0)

    # Moisture feedback
    if q_v is not None:
        # Moisture optical depth: proportional to column water per layer
        # dp per layer in Pa
        dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev)
        # Column water [kg/m^2] per layer ≈ q_v * dp / g
        col_water = q_v * dp / constants.g
        dtau_moist = config.tau_moist_coeff * col_water / p_s[:, None]
        dtau = dtau_dry + jnp.maximum(dtau_moist, 0.0)
    else:
        dtau = dtau_dry

    return dtau


def _lw_two_stream(
    T: jnp.ndarray,
    sfc_temperature: jnp.ndarray,
    dtau: jnp.ndarray,
    config: GrayRadiationConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute LW fluxes using hemispheric-mean two-stream.

    Uses jax.lax.scan for the upward and downward sweeps.

    Parameters
    ----------
    T : jnp.ndarray
        Temperature at full levels (ncol, nlev) [K].
    sfc_temperature : jnp.ndarray
        Surface temperature (ncol,) [K].
    dtau : jnp.ndarray
        LW optical depth per layer (ncol, nlev).
    config : GrayRadiationConfig

    Returns
    -------
    lw_up : jnp.ndarray
        Upward LW flux at interfaces (ncol, nlev+1) [W/m^2].
    lw_down : jnp.ndarray
        Downward LW flux at interfaces (ncol, nlev+1) [W/m^2].
    """
    D = config.lw_diff_factor
    eps_sfc = config.sfc_emissivity
    ncol, nlev = T.shape

    # Layer transmittance and emissivity
    transmittance = jnp.exp(-D * dtau)  # (ncol, nlev)
    emissivity = 1.0 - transmittance     # (ncol, nlev)

    # Planck emission at each layer center
    B = constants.sigma_sb * T ** 4  # (ncol, nlev)

    # --- Downward sweep (top to bottom) using scan ---
    # BC: F_down(TOA) = 0
    F_down_toa = jnp.zeros(ncol)
    t_T = jnp.moveaxis(transmittance, 1, 0)    # (nlev, ncol)
    eB_T = jnp.moveaxis(emissivity * B, 1, 0)  # (nlev, ncol)

    def downward_step(F_above, layer_data):
        t_k, eB_k = layer_data  # each (ncol,)
        F_below = t_k * F_above + eB_k
        return F_below, F_below

    _, F_down_interfaces = jax.lax.scan(
        downward_step,
        F_down_toa,
        (t_T, eB_T),  # (nlev, ncol) — top to bottom
    )
    # F_down_interfaces: (nlev, ncol) — interfaces 1..nlev (below each layer)
    F_down_interfaces = jnp.moveaxis(F_down_interfaces, 0, 1)  # (ncol, nlev)
    lw_down = jnp.concatenate([
        F_down_toa[:, None],
        F_down_interfaces,
    ], axis=1)  # (ncol, nlev+1)

    # --- Upward sweep (bottom to top) using scan ---
    # Surface LW boundary: emitted + reflected downward LW for non-black surface.
    F_down_sfc = lw_down[:, -1]
    F_up_sfc = (
        eps_sfc * constants.sigma_sb * sfc_temperature ** 4
        + (1.0 - eps_sfc) * F_down_sfc
    )  # (ncol,)

    # Scan from bottom (level nlev-1) to top (level 0)
    # At each level k: F_up(k) = t_k * F_up(k+1) + eps_k * B_k
    # lax.scan iterates over the FIRST axis, so transpose to (nlev, ncol)
    t_rev = t_T[::-1]    # (nlev, ncol)
    eB_rev = eB_T[::-1]  # (nlev, ncol)

    def upward_step(F_below, layer_data):
        t_k, eB_k = layer_data  # each (ncol,)
        F_above = t_k * F_below + eB_k
        return F_above, F_above

    _, F_up_interfaces_rev = jax.lax.scan(
        upward_step,
        F_up_sfc,
        (t_rev, eB_rev),
    )
    # F_up_interfaces_rev: (nlev, ncol) — from bottom+1 upward to TOA
    # Reverse back to top-to-bottom order and transpose to (ncol, nlev)
    F_up_inner = jnp.moveaxis(F_up_interfaces_rev[::-1], 0, 1)  # (ncol, nlev)
    lw_up = jnp.concatenate([
        F_up_inner,
        F_up_sfc[:, None],
    ], axis=1)  # (ncol, nlev+1): interfaces 0..nlev

    return lw_up, lw_down


def _sw_beer_lambert(
    p_half: jnp.ndarray,
    p_s: jnp.ndarray,
    insolation: jnp.ndarray,
    config: GrayRadiationConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute SW fluxes using Beer-Lambert absorption.

    No scattering — downward flux attenuated exponentially,
    surface-reflected upward flux also attenuated.

    Parameters
    ----------
    p_half : jnp.ndarray
        Pressure at interfaces (ncol, nlev+1) [Pa].
    p_s : jnp.ndarray
        Surface pressure (ncol,) [Pa].
    insolation : jnp.ndarray
        TOA insolation (ncol,) [W/m^2].
    config : GrayRadiationConfig

    Returns
    -------
    sw_up : jnp.ndarray
        Upward SW flux at interfaces (ncol, nlev+1) [W/m^2].
    sw_down : jnp.ndarray
        Downward SW flux at interfaces (ncol, nlev+1) [W/m^2].
    """
    D = config.lw_diff_factor
    tau_sw_0 = config.sw_tau_0
    alpha = config.sfc_albedo

    # sigma at interfaces
    sigma_half = p_half / p_s[:, None]  # (ncol, nlev+1)

    # Cumulative SW optical depth from TOA: tau_sw(sigma) = tau_sw_0 * sigma
    tau_sw = tau_sw_0 * sigma_half  # (ncol, nlev+1)

    # Downward SW flux: F_down(k) = insolation * exp(-D * tau_sw(k))
    sw_down = insolation[:, None] * jnp.exp(-D * tau_sw)  # (ncol, nlev+1)

    # Surface reflection: F_up(sfc) = alpha * F_down(sfc)
    F_up_sfc = alpha * sw_down[:, -1]  # (ncol,)

    # Upward SW flux attenuated from surface: F_up(k) = F_up(sfc) * exp(-D * (tau_sw(sfc) - tau_sw(k)))
    tau_sw_sfc = tau_sw[:, -1:]  # (ncol, 1)
    sw_up = F_up_sfc[:, None] * jnp.exp(-D * (tau_sw_sfc - tau_sw))  # (ncol, nlev+1)

    return sw_up, sw_down


def _compute_heating_rate(
    flux_up: jnp.ndarray,
    flux_down: jnp.ndarray,
    p_half: jnp.ndarray,
) -> jnp.ndarray:
    """Compute heating rate from net flux divergence.

    The standard radiative heating rate in pressure coordinates is:

        dT/dt = (g / c_p) * d(F_up - F_down) / dp

    Using the net upward flux F_net↑ = F_up - F_down:

        dT/dt_k = (g / c_p) * [F_net↑(k+1) - F_net↑(k)] / [p(k+1) - p(k)]

    where interface k is at the top (low p) and k+1 at the bottom (high p)
    of layer k. When more net upward flux exits the bottom than the top,
    the layer has gained energy (positive heating).

    Parameters
    ----------
    flux_up : jnp.ndarray
        Upward flux at interfaces (ncol, nlev+1) [W/m^2].
    flux_down : jnp.ndarray
        Downward flux at interfaces (ncol, nlev+1) [W/m^2].
    p_half : jnp.ndarray
        Pressure at interfaces (ncol, nlev+1) [Pa].

    Returns
    -------
    jnp.ndarray
        Heating rate (ncol, nlev) [K/s].
    """
    # Net upward flux at each interface
    F_net_up = flux_up - flux_down  # (ncol, nlev+1)

    # Flux divergence across each layer: F_net_up(bottom) - F_net_up(top)
    # Interface k is the top of layer k, interface k+1 is the bottom.
    dF = F_net_up[:, 1:] - F_net_up[:, :-1]  # (ncol, nlev)
    dp = p_half[:, 1:] - p_half[:, :-1]       # (ncol, nlev)
    dp = jnp.clip(dp, 1.0, None)

    return (constants.g / constants.c_pd) * dF / dp


def gray_radiation(
    T: jnp.ndarray,
    p_full: jnp.ndarray,
    p_half: jnp.ndarray,
    sfc_temperature: jnp.ndarray,
    lat: jnp.ndarray,
    q_v: jnp.ndarray | None,
    insolation: jnp.ndarray,
    config: GrayRadiationConfig,
) -> RadiationOutput:
    """Compute two-stream gray radiation (LW + SW).

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
    lat : jnp.ndarray
        Latitude (ncol,) [radians].
    q_v : jnp.ndarray or None
        Water vapor mixing ratio (ncol, nlev) [kg/kg]. None for dry.
    insolation : jnp.ndarray
        TOA insolation (ncol,) [W/m^2].
    config : GrayRadiationConfig

    Returns
    -------
    RadiationOutput
        Fluxes and heating rates.
    """
    p_s = p_half[:, -1]  # surface pressure

    # LW optical depth per layer
    dtau_lw = _compute_lw_optical_depth(p_half, p_s, lat, q_v, config)

    # LW fluxes
    lw_up, lw_down = _lw_two_stream(T, sfc_temperature, dtau_lw, config)

    # SW fluxes
    sw_up, sw_down = _sw_beer_lambert(p_half, p_s, insolation, config)

    # Heating rates
    lw_hr = _compute_heating_rate(lw_up, lw_down, p_half)
    sw_hr = _compute_heating_rate(sw_up, sw_down, p_half)
    total_hr = lw_hr + sw_hr

    return RadiationOutput(
        lw_flux_up=lw_up,
        lw_flux_down=lw_down,
        sw_flux_up=sw_up,
        sw_flux_down=sw_down,
        heating_rate=total_hr,
        lw_heating_rate=lw_hr,
        sw_heating_rate=sw_hr,
    )
