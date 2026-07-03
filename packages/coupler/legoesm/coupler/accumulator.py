"""Flux accumulator for asynchronous coupling.

Accumulates dt-weighted surface->atm fluxes across sub-steps,
then returns the time-mean when the coupling interval elapses.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.precision import resolve_dtype
from legoesm.core.coupling_fields import SurfaceToAtm


def _tiny(dtype=None):
    """Smallest normal float for the given dtype (or the active accumulate dtype)."""
    if dtype is None:
        dtype = resolve_dtype(None, "accumulate")
    return float(jnp.finfo(dtype).tiny)


class FluxAccumulator(NamedTuple):
    """Accumulated surface->atm fluxes, weighted by sub-step dt."""
    sum_T_sfc: jax.Array
    # dt-weighted surface EMISSION FLUX  eps * sigma * T_rad^4  [W/m^2 * s].
    # Accumulated in FLUX space (NOT as mean T_rad) so the window-mean LW
    # boundary is exact: averaging T_rad and emissivity independently would give
    # mean(eps)*sigma*mean(T_rad)^4 != mean(eps*sigma*T_rad^4) for varying
    # substeps.  mean_accumulator reconstructs T_rad from this and sum_emissivity.
    sum_emit: jax.Array
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
    sum_freshwater_flux: jax.Array
    sum_ocean_heat_extraction: jax.Array
    sum_ocean_stress_x: jax.Array
    sum_ocean_stress_y: jax.Array
    sum_surface_mass_flux: jax.Array
    sum_salt_flux: jax.Array
    sum_river_runoff_flux: jax.Array
    sum_ice_lake_freshwater_flux: jax.Array
    total_dt: jax.Array          # Scalar: total accumulated dt


def reset_accumulator(
    shape: tuple[int, ...],
    *,
    dtype: jnp.dtype | None = None,
) -> FluxAccumulator:
    """Zero-initialize accumulator for given spatial shape."""
    z = jnp.zeros(shape, dtype=dtype)
    return FluxAccumulator(
        sum_T_sfc=z, sum_emit=z, sum_albedo=z, sum_emissivity=z,
        sum_z0=z, sum_q_surface=z,
        sum_shflx=z, sum_lhflx=z,
        sum_tau_x=z, sum_tau_y=z, sum_lw_up=z,
        sum_u_ocean_sfc=z, sum_v_ocean_sfc=z,
        sum_co2_flux=z,
        sum_freshwater_flux=z,
        sum_ocean_heat_extraction=z,
        sum_ocean_stress_x=z,
        sum_ocean_stress_y=z,
        sum_surface_mass_flux=z,
        sum_salt_flux=z,
        sum_river_runoff_flux=z,
        sum_ice_lake_freshwater_flux=z,
        total_dt=jnp.array(0.0, dtype=z.dtype),
    )


def accumulate(
    acc: FluxAccumulator,
    sfc: SurfaceToAtm,
    dt: float,
) -> FluxAccumulator:
    """Add one sub-step to the accumulator, weighted by dt."""
    dt_arr = jnp.asarray(dt, dtype=acc.total_dt.dtype)
    return FluxAccumulator(
        sum_T_sfc=acc.sum_T_sfc + dt_arr * sfc.T_sfc,
        sum_emit=acc.sum_emit + dt_arr * (
            sfc.emissivity * constants.sigma_sb * sfc.T_rad ** 4),
        sum_albedo=acc.sum_albedo + dt_arr * sfc.albedo,
        sum_emissivity=acc.sum_emissivity + dt_arr * sfc.emissivity,
        sum_z0=acc.sum_z0 + dt_arr * sfc.z0,
        sum_q_surface=acc.sum_q_surface + dt_arr * sfc.q_surface,
        sum_shflx=acc.sum_shflx + dt_arr * sfc.shflx,
        sum_lhflx=acc.sum_lhflx + dt_arr * sfc.lhflx,
        sum_tau_x=acc.sum_tau_x + dt_arr * sfc.tau_x,
        sum_tau_y=acc.sum_tau_y + dt_arr * sfc.tau_y,
        sum_lw_up=acc.sum_lw_up + dt_arr * sfc.lw_up,
        sum_u_ocean_sfc=acc.sum_u_ocean_sfc + dt_arr * sfc.u_ocean_sfc,
        sum_v_ocean_sfc=acc.sum_v_ocean_sfc + dt_arr * sfc.v_ocean_sfc,
        sum_co2_flux=acc.sum_co2_flux + dt_arr * sfc.co2_flux,
        sum_freshwater_flux=acc.sum_freshwater_flux + dt_arr * sfc.freshwater_flux,
        sum_ocean_heat_extraction=(
            acc.sum_ocean_heat_extraction + dt_arr * sfc.ocean_heat_extraction
        ),
        sum_ocean_stress_x=acc.sum_ocean_stress_x + dt_arr * sfc.ocean_stress_x,
        sum_ocean_stress_y=acc.sum_ocean_stress_y + dt_arr * sfc.ocean_stress_y,
        sum_surface_mass_flux=(
            acc.sum_surface_mass_flux + dt_arr * sfc.surface_mass_flux
        ),
        sum_salt_flux=acc.sum_salt_flux + dt_arr * sfc.salt_flux,
        sum_river_runoff_flux=(
            acc.sum_river_runoff_flux + dt_arr * sfc.river_runoff_flux
        ),
        sum_ice_lake_freshwater_flux=(
            acc.sum_ice_lake_freshwater_flux + dt_arr * sfc.ice_lake_freshwater_flux
        ),
        total_dt=acc.total_dt + dt_arr,
    )


def mean_accumulator(acc: FluxAccumulator) -> SurfaceToAtm:
    """Compute dt-weighted mean from the accumulator."""
    inv_dt = 1.0 / jnp.clip(acc.total_dt, _tiny(acc.total_dt.dtype), None)
    return SurfaceToAtm(
        T_sfc=acc.sum_T_sfc * inv_dt,
        # Reconstruct the window-mean radiative-equivalent T from the dt-weighted
        # EMISSION FLUX and emissivity:  mean_eps * sigma * T_rad^4 == mean(emit).
        # (Averaging T_rad directly would break the paired LW-flux invariant for
        # varying substeps.)  inv_dt cancels in the ratio.
        T_rad=(acc.sum_emit / jnp.maximum(
            acc.sum_emissivity * constants.sigma_sb,
            _tiny(acc.total_dt.dtype))) ** 0.25,
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
        freshwater_flux=acc.sum_freshwater_flux * inv_dt,
        ocean_heat_extraction=acc.sum_ocean_heat_extraction * inv_dt,
        ocean_stress_x=acc.sum_ocean_stress_x * inv_dt,
        ocean_stress_y=acc.sum_ocean_stress_y * inv_dt,
        surface_mass_flux=acc.sum_surface_mass_flux * inv_dt,
        salt_flux=acc.sum_salt_flux * inv_dt,
        river_runoff_flux=acc.sum_river_runoff_flux * inv_dt,
        ice_lake_freshwater_flux=acc.sum_ice_lake_freshwater_flux * inv_dt,
    )


def accumulator_from_flux(
    sfc: SurfaceToAtm,
    dt: float,
    *,
    dtype: jnp.dtype | None = None,
) -> FluxAccumulator:
    """Create an accumulator seeded with one constant-flux segment."""
    dt_arr = jnp.asarray(dt, dtype=dtype if dtype is not None else sfc.T_sfc.dtype)
    return FluxAccumulator(
        sum_T_sfc=dt_arr * sfc.T_sfc,
        sum_emit=dt_arr * (
            sfc.emissivity * constants.sigma_sb * sfc.T_rad ** 4),
        sum_albedo=dt_arr * sfc.albedo,
        sum_emissivity=dt_arr * sfc.emissivity,
        sum_z0=dt_arr * sfc.z0,
        sum_q_surface=dt_arr * sfc.q_surface,
        sum_shflx=dt_arr * sfc.shflx,
        sum_lhflx=dt_arr * sfc.lhflx,
        sum_tau_x=dt_arr * sfc.tau_x,
        sum_tau_y=dt_arr * sfc.tau_y,
        sum_lw_up=dt_arr * sfc.lw_up,
        sum_u_ocean_sfc=dt_arr * sfc.u_ocean_sfc,
        sum_v_ocean_sfc=dt_arr * sfc.v_ocean_sfc,
        sum_co2_flux=dt_arr * sfc.co2_flux,
        sum_freshwater_flux=dt_arr * sfc.freshwater_flux,
        sum_ocean_heat_extraction=dt_arr * sfc.ocean_heat_extraction,
        sum_ocean_stress_x=dt_arr * sfc.ocean_stress_x,
        sum_ocean_stress_y=dt_arr * sfc.ocean_stress_y,
        sum_surface_mass_flux=dt_arr * sfc.surface_mass_flux,
        sum_salt_flux=dt_arr * sfc.salt_flux,
        sum_river_runoff_flux=dt_arr * sfc.river_runoff_flux,
        sum_ice_lake_freshwater_flux=dt_arr * sfc.ice_lake_freshwater_flux,
        total_dt=dt_arr,
    )
