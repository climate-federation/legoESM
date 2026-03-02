"""Flux accumulator for asynchronous coupling.

Accumulates dt-weighted surface->atm fluxes across sub-steps,
then returns the time-mean when the coupling interval elapses.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.coupler.coupling_fields import SurfaceToAtm


class FluxAccumulator(NamedTuple):
    """Accumulated surface->atm fluxes, weighted by sub-step dt."""
    sum_T_surface: jax.Array
    sum_albedo: jax.Array
    sum_emissivity: jax.Array
    sum_z0: jax.Array
    sum_q_surface: jax.Array
    sum_shflx: jax.Array
    sum_lhflx: jax.Array
    sum_tau_x: jax.Array
    sum_tau_y: jax.Array
    sum_lw_up: jax.Array
    sum_u_ocean_sfc: jax.Array
    sum_v_ocean_sfc: jax.Array
    sum_co2_flux: jax.Array
    total_dt: jax.Array          # Scalar: total accumulated dt


def reset_accumulator(shape: tuple[int, ...]) -> FluxAccumulator:
    """Zero-initialize accumulator for given spatial shape."""
    z = jnp.zeros(shape)
    return FluxAccumulator(
        sum_T_surface=z, sum_albedo=z, sum_emissivity=z,
        sum_z0=z, sum_q_surface=z,
        sum_shflx=z, sum_lhflx=z,
        sum_tau_x=z, sum_tau_y=z, sum_lw_up=z,
        sum_u_ocean_sfc=z, sum_v_ocean_sfc=z,
        sum_co2_flux=z,
        total_dt=jnp.array(0.0),
    )


def accumulate(
    acc: FluxAccumulator,
    sfc: SurfaceToAtm,
    dt: float,
) -> FluxAccumulator:
    """Add one sub-step to the accumulator, weighted by dt."""
    return FluxAccumulator(
        sum_T_surface=acc.sum_T_surface + dt * sfc.T_surface,
        sum_albedo=acc.sum_albedo + dt * sfc.albedo,
        sum_emissivity=acc.sum_emissivity + dt * sfc.emissivity,
        sum_z0=acc.sum_z0 + dt * sfc.z0,
        sum_q_surface=acc.sum_q_surface + dt * sfc.q_surface,
        sum_shflx=acc.sum_shflx + dt * sfc.shflx,
        sum_lhflx=acc.sum_lhflx + dt * sfc.lhflx,
        sum_tau_x=acc.sum_tau_x + dt * sfc.tau_x,
        sum_tau_y=acc.sum_tau_y + dt * sfc.tau_y,
        sum_lw_up=acc.sum_lw_up + dt * sfc.lw_up,
        sum_u_ocean_sfc=acc.sum_u_ocean_sfc + dt * sfc.u_ocean_sfc,
        sum_v_ocean_sfc=acc.sum_v_ocean_sfc + dt * sfc.v_ocean_sfc,
        sum_co2_flux=acc.sum_co2_flux + dt * sfc.co2_flux,
        total_dt=acc.total_dt + dt,
    )


def mean_accumulator(acc: FluxAccumulator) -> SurfaceToAtm:
    """Compute dt-weighted mean from the accumulator."""
    inv_dt = 1.0 / jnp.clip(acc.total_dt, 1e-30, None)
    return SurfaceToAtm(
        T_surface=acc.sum_T_surface * inv_dt,
        albedo=acc.sum_albedo * inv_dt,
        emissivity=acc.sum_emissivity * inv_dt,
        z0=acc.sum_z0 * inv_dt,
        q_surface=acc.sum_q_surface * inv_dt,
        shflx=acc.sum_shflx * inv_dt,
        lhflx=acc.sum_lhflx * inv_dt,
        tau_x=acc.sum_tau_x * inv_dt,
        tau_y=acc.sum_tau_y * inv_dt,
        lw_up=acc.sum_lw_up * inv_dt,
        u_ocean_sfc=acc.sum_u_ocean_sfc * inv_dt,
        v_ocean_sfc=acc.sum_v_ocean_sfc * inv_dt,
        co2_flux=acc.sum_co2_flux * inv_dt,
    )
