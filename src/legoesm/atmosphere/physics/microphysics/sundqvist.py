"""Sundqvist large-scale diagnostic condensation scheme.

A diagnostic scheme that activates condensation when relative humidity
exceeds a critical threshold. Produces large-scale (non-convective)
precipitation through autoconversion and sub-cloud evaporation.

All operations use smooth (differentiable) approximations.

References
----------
- Sundqvist et al. (1989): Condensation and cloud parameterization
  studies with a mesoscale numerical weather prediction model.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_specific_humidity
from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
)


class SundqvistProcessRates(NamedTuple):
    """Intermediate Sundqvist process rates used to assemble tendencies."""

    condensation: jax.Array
    autoconversion: jax.Array
    evaporation: jax.Array
    precipitation: jax.Array


def diagnose_sundqvist_process_rates(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: SundqvistConfig = SundqvistConfig(),
) -> SundqvistProcessRates:
    """Diagnose the Sundqvist condensation, rain conversion, and evaporation terms."""
    del p_half  # Included for signature parity with ``sundqvist_microphysics``.
    q_c = hydrometeors.q_c
    sharpness = config.sigmoid_sharpness

    # Saturation
    q_sat = saturation_specific_humidity(T, p_full)
    RH = q_v / jnp.clip(q_sat, 1e-10)

    # 1. Smooth condensation activation.
    #
    # RH_crit controls the onset of large-scale condensation via the smooth
    # activation factor ``f``.  The actual condensed amount must still be the
    # supersaturated excess above q_sat, not the excess above RH_crit * q_sat;
    # otherwise the scheme spuriously condenses in subsaturated columns whenever
    # RH lies between RH_crit and 1.  That produces unphysical heating/rain
    # spikes in initialized forecasts.
    f = jax.nn.sigmoid(sharpness * (RH - config.RH_crit))
    condensation = (
        f * jnp.maximum(q_v - q_sat, 0.0) / dt
    )  # [kg/kg/s]

    # 2. Autoconversion
    # condensation is a tendency [kg/kg/s]; multiply by dt to get increment [kg/kg]
    P_auto = config.auto_rate * jnp.maximum(q_c + condensation * dt, 0.0)

    # 3. Sub-cloud evaporation
    evap_mask = jax.nn.sigmoid(sharpness * (config.RH_crit - RH))
    P_flux_layer = P_auto * rho * dz

    def scan_fn(carry, x):
        P_above = carry
        P_local, evap_m, rho_k, dz_k = x
        P_total = P_above + P_local
        evap = config.evap_coeff * evap_m * P_total / jnp.clip(rho_k * dz_k, 1.0)
        evap = jnp.minimum(evap, P_total / jnp.clip(rho_k * dz_k, 1.0))
        P_out = jnp.clip(P_total - evap * rho_k * dz_k, 0.0)
        return P_out, evap

    # Transpose for scan: (nlev, ncol)
    inputs = (
        jnp.moveaxis(P_flux_layer, 1, 0),
        jnp.moveaxis(evap_mask, 1, 0),
        jnp.moveaxis(rho, 1, 0),
        jnp.moveaxis(dz, 1, 0),
    )
    P_init = jnp.zeros(T.shape[0], dtype=T.dtype)
    P_final, evap_col = jax.lax.scan(scan_fn, P_init, inputs)
    evaporation = jnp.moveaxis(evap_col, 0, 1)
    return SundqvistProcessRates(
        condensation=condensation,
        autoconversion=P_auto,
        evaporation=evaporation,
        precipitation=P_final,
    )


def sundqvist_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: SundqvistConfig = SundqvistConfig(),
) -> MicrophysicsOutput:
    """Compute Sundqvist diagnostic condensation tendencies.

    Parameters
    ----------
    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
        Same interface as all microphysics backends.

    Returns
    -------
    MicrophysicsOutput
    """
    ncol, nlev = T.shape
    rates = diagnose_sundqvist_process_rates(
        T=T,
        q_v=q_v,
        hydrometeors=hydrometeors,
        p_full=p_full,
        p_half=p_half,
        rho=rho,
        dz=dz,
        dt=dt,
        config=config,
    )

    # 4. Latent heating
    net_cond = rates.condensation - rates.evaporation
    dT_dt = constants.L_v * net_cond / constants.c_pd

    # Tendencies
    dq_v_dt = -rates.condensation + rates.evaporation
    dq_c_dt = rates.condensation - rates.autoconversion
    dq_r_dt = rates.autoconversion - rates.evaporation

    z = jnp.zeros((ncol, nlev))
    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        dq_i_dt=z,
        dq_s_dt=z,
        dq_g_dt=z,
        dN_c_dt=z,
        dN_r_dt=z,
        dN_i_dt=z,
        precipitation=rates.precipitation,
    )
