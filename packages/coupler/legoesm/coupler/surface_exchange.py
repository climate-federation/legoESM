"""Extract coupling fields from atmospheric state.

These functions are the ONLY place where full atmospheric state is accessed.
The coupler itself only ever sees AtmToSurface.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.coupler.config import CouplerConfig
from legoesm.core.coupling_fields import AtmToSurface, lowest_level_height
from legoesm.core.state import HydrostaticState, NonHydrostaticState
from legoesm.grids.vertical import SigmaCoordinate
from legoesm.atmosphere.physics.thermodynamics import (
    pressure_from_eos,
    temperature_from_theta,
)


def extract_atm_to_surface(
    state: HydrostaticState,
    sigma_coord: SigmaCoordinate,
    config: CouplerConfig,
    sw_down: jnp.ndarray | None = None,
    lw_down: jnp.ndarray | None = None,
    precip_total: jnp.ndarray | None = None,
    precip_snow: jnp.ndarray | None = None,
    cos_zenith: jnp.ndarray | None = None,
) -> AtmToSurface:
    """Extract coupling fields from hydrostatic atmospheric state.

    Reads only the lowest model level (index -1) plus surface pressure.
    Radiation/precipitation fields are passed in from the physics
    parameterizations that produce them.

    Parameters
    ----------
    state : HydrostaticState
        Current atmospheric state.
    sigma_coord : SigmaCoordinate
        Vertical coordinate for pressure reconstruction.
    config : CouplerConfig
        Coupler configuration.
    sw_down, lw_down : optional arrays
        Downward radiative fluxes at surface [W/m2]. Shape (6, n, n).
    precip_total, precip_snow : optional arrays
        Precipitation rates [kg/m2/s]. Shape (6, n, n).
    cos_zenith : optional array
        Cosine of solar zenith angle. Shape (6, n, n).
    """
    shape = state.p_s.data.shape  # (6, n, n)
    p_s = state.p_s.data

    # Lowest-level pressure from sigma coordinate
    p_lowest = sigma_coord.pressure_at_full(p_s)[..., -1]

    # Lowest-level fields
    T_lowest = state.T.data[..., -1]
    u_lowest = state.u.data[..., -1]
    v_lowest = state.v.data[..., -1]
    # Pin defaulted allocations to the state precision so x64 zeros do
    # not silently widen the AtmToSurface struct precision.  This is on
    # the hot path: surface_exchange runs every coupling step.
    _state_dtype = T_lowest.dtype

    # Humidity: extract from tracers if available; else assume dry.
    if state.tracers is not None and "q_v" in state.tracers:
        _qv_raw = state.tracers["q_v"]
        _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
        q_lowest = _qv_data[..., -1]
    else:
        q_lowest = jnp.zeros(shape, dtype=_state_dtype)

    # Air density from ideal gas law for moist air: ``p = ρ · R_d · T_v``
    # where ``T_v = T · (1 + (1/ε − 1) · q_v)``.  The previous dry form
    # ``ρ = p / (R_d · T)`` underestimated density by ~0.6 % in the
    # tropics (q_v ~ 17 g/kg, T_v − T ~ 1.7 K), biasing bulk-flux
    # surface stress and turbulent fluxes via every downstream caller
    # that uses ``forcing.rho_lowest``.
    T_v_lowest = T_lowest * (1.0 + (1.0 / constants.epsilon - 1.0) * q_lowest)
    rho_lowest = p_lowest / (constants.R_d * T_v_lowest)

    # Default unavailable fields to zero with flags
    zero = jnp.zeros(shape, dtype=_state_dtype)
    has_rad = jnp.array(1.0) if sw_down is not None else jnp.array(0.0)
    has_precip = jnp.array(1.0) if precip_total is not None else jnp.array(0.0)

    return AtmToSurface(
        z_lowest=lowest_level_height(
            T_lowest, sigma_coord.pressure_at_half(p_s),
            sigma_coord.pressure_at_full(p_s)),
        sw_down=sw_down if sw_down is not None else zero,
        lw_down=lw_down if lw_down is not None else zero,
        precip_total=precip_total if precip_total is not None else zero,
        precip_snow=precip_snow if precip_snow is not None else zero,
        T_lowest=T_lowest,
        q_lowest=q_lowest,
        u_lowest=u_lowest,
        v_lowest=v_lowest,
        p_lowest=p_lowest,
        p_surface=p_s,
        rho_lowest=rho_lowest,
        cos_zenith=cos_zenith if cos_zenith is not None else zero,
        co2_ppmv=jnp.array(config.co2_ppmv_default),
        has_radiation=has_rad,
        has_precipitation=has_precip,
    )


def extract_atm_to_surface_nh(
    state: NonHydrostaticState,
    height_coord,
    terrain_metric,
    config: CouplerConfig,
    sw_down: jnp.ndarray | None = None,
    lw_down: jnp.ndarray | None = None,
    precip_total: jnp.ndarray | None = None,
    precip_snow: jnp.ndarray | None = None,
    cos_zenith: jnp.ndarray | None = None,
) -> AtmToSurface:
    """Extract coupling fields from non-hydrostatic atmospheric state.

    Reads only the lowest model level (index -1) plus surface fields.
    """
    shape = state.phis.data.shape  # (6, n, n)

    # Lowest-level fields
    theta_prime_low = state.theta_prime.data[..., -1]
    rho_prime_low = state.rho_prime.data[..., -1]

    theta_0_low = height_coord.theta_ref[-1]
    rho_0_low = height_coord.rho_ref[-1]

    theta_low = theta_prime_low + theta_0_low
    rho_low = rho_prime_low + rho_0_low

    p_lowest = pressure_from_eos(rho_low, theta_low)
    T_lowest = temperature_from_theta(theta_low, p_lowest)

    u_lowest = state.u.data[..., -1]
    v_lowest = state.v.data[..., -1]

    # Humidity from tracers if available.  Use a Python ``if`` rather
    # than ``jnp.where`` because the latter still traces both branches,
    # and ``state.tracers.data[..., -1, 0]`` crashes at trace time when
    # the n_tracers axis is empty (dry NH simulations have shape
    # ``(..., nlev, 0)`` and indexing axis-0 position 0 is out of
    # bounds).  Mirrors the hydrostatic branch's static-shape check
    # (lines 69-74).
    if state.tracers.data.shape[-1] > 0:
        q_lowest = state.tracers.data[..., -1, 0]
    else:
        q_lowest = jnp.zeros_like(T_lowest)

    rho_lowest = rho_low

    # Surface pressure: approximate from lowest-level via hydrostatic correction
    dz_sfc = height_coord.z_half[-1] - height_coord.z_half[-2]
    p_surface = p_lowest + constants.g * rho_low * jnp.abs(dz_sfc) * 0.5

    # Pin to the state precision (T_lowest is derived from theta_prime + theta_0).
    zero = jnp.zeros(shape, dtype=T_lowest.dtype)
    has_rad = jnp.array(1.0) if sw_down is not None else jnp.array(0.0)
    has_precip = jnp.array(1.0) if precip_total is not None else jnp.array(0.0)

    return AtmToSurface(
        z_lowest=(terrain_metric.z_full_3d[..., -1]
                  - terrain_metric.z_half_3d[..., -1]),
        sw_down=sw_down if sw_down is not None else zero,
        lw_down=lw_down if lw_down is not None else zero,
        precip_total=precip_total if precip_total is not None else zero,
        precip_snow=precip_snow if precip_snow is not None else zero,
        T_lowest=T_lowest,
        q_lowest=q_lowest,
        u_lowest=u_lowest,
        v_lowest=v_lowest,
        p_lowest=p_lowest,
        p_surface=p_surface,
        rho_lowest=rho_lowest,
        cos_zenith=cos_zenith if cos_zenith is not None else zero,
        co2_ppmv=jnp.array(config.co2_ppmv_default),
        has_radiation=has_rad,
        has_precipitation=has_precip,
    )
