"""Compatibility facade for the shared NEMO NCAR/OMIP bulk implementation.

The source-literal NCAR identity lives once in :mod:`legoesm.core.bulk_flux`.
This historical ocean import path remains for callers, but contains no second
NCAR solver. ``ly09_2coeff`` is the separately named legacy approximation.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.bulk_flux import (
    large_yeager_neutral_cd,
    nemo_ncar_ocean_bulk,
    nemo_ncar_potential_air_temperature,
    nemo_ncar_pressure_at_height,
    nemo_ncar_rho_air,
    nemo_ncar_seawater_q_sat,
    nemo_ncar_theta_exner,
    nemo_ncar_transfer_coefficients as _shared_ncar_transfer_coefficients,
)

_VALID_OMIP_BULK_ALGOS = ("ncar", "ly09_2coeff")


def exner_potential_temperature(T_K, p_Pa):
    return nemo_ncar_theta_exner(
        jnp.asarray(T_K, dtype=jnp.float64),
        jnp.asarray(p_Pa, dtype=jnp.float64),
    )


def pressure_at_height(q_air, slp_Pa, z_m, T_abs_K):
    pressure, _ = nemo_ncar_pressure_at_height(
        q_air, slp_Pa, z_m, absolute_temperature=T_abs_K)
    return pressure


def potential_air_temperature_10m(T_air_K, q_air, slp_Pa=None):
    pressure = constants.p_atm_std if slp_Pa is None else slp_Pa
    return nemo_ncar_potential_air_temperature(T_air_K, q_air, pressure)


def seawater_q_sat(T_sfc_K, slp_Pa=None):
    pressure = constants.p_atm_std if slp_Pa is None else slp_Pa
    return nemo_ncar_seawater_q_sat(T_sfc_K, pressure)


def rho_air_moist(T_air_K, q_air, slp_Pa=None):
    pressure = constants.p_atm_std if slp_Pa is None else slp_Pa
    temperature = jnp.maximum(
        jnp.asarray(T_air_K, dtype=jnp.float64), constants.T_goff_floor_nemo)
    humidity = jnp.maximum(
        jnp.asarray(q_air, dtype=jnp.float64), constants.ncar_humidity_floor_nemo)
    return nemo_ncar_rho_air(temperature, humidity, pressure)


def latent_heat_vaporization_sst(T_sfc_K):
    from legoesm.thermo import latent_heat_vaporization_sst as shared
    return shared(jnp.asarray(T_sfc_K, dtype=jnp.float64))


def moist_air_cp(q_air):
    from legoesm.thermo import moist_air_cp as shared
    return shared(jnp.asarray(q_air, dtype=jnp.float64))


def ncar_transfer_coefficients(
    theta_air_K, q_air, theta_sst_K, q_sfc, wind_speed, *, nb_iter=None,
):
    cd, ch, ce, _theta_zu, _q_zu, bulk_wind = (
        _shared_ncar_transfer_coefficients(
            theta_sst_K, theta_air_K, q_sfc, q_air, wind_speed,
            iterations=nb_iter,
        )
    )
    return cd, ch, ce, bulk_wind


def large_yeager_cd(u10_speed):
    return large_yeager_neutral_cd(jnp.asarray(u10_speed, dtype=jnp.float64))


def large_yeager_ch(T_air_K, T_sfc_K):
    return jnp.where(
        T_sfc_K > T_air_K,
        constants.large_yeager_legacy_ch_unstable,
        constants.large_yeager_legacy_ch_stable,
    )


def air_sea_fluxes(
    u10, v10, T_air_K, q_air, T_sfc_K, q_sfc=None, rho_air=None, *,
    slp_Pa=None, algo: str = "ncar", nb_iter=None, L_latent=None,
    u_oce=None, v_oce=None, vfac: float = 0.0,
):
    """Return atmospheric-sign stress and downward ocean heat fluxes."""
    if algo not in _VALID_OMIP_BULK_ALGOS:
        raise ValueError(
            f"Unknown OMIP bulk algo {algo!r}; expected one of "
            f"{_VALID_OMIP_BULK_ALGOS}.")
    u = jnp.asarray(u10, dtype=jnp.float64)
    v = jnp.asarray(v10, dtype=jnp.float64)
    tair = jnp.asarray(T_air_K, dtype=jnp.float64)
    qair = jnp.asarray(q_air, dtype=jnp.float64)
    surface = jnp.asarray(T_sfc_K, dtype=jnp.float64)
    if vfac != 0.0 and u_oce is not None and v_oce is not None:
        u = u - vfac * jnp.asarray(u_oce, dtype=jnp.float64)
        v = v - vfac * jnp.asarray(v_oce, dtype=jnp.float64)

    if algo == "ly09_2coeff":
        if q_sfc is None:
            raise ValueError(
                "air_sea_fluxes(algo='ly09_2coeff') requires q_sfc")
        wind = jnp.sqrt(
            u * u + v * v + constants.large_yeager_legacy_wind2_floor)
        density = constants.rho_air if rho_air is None else rho_air
        cd = large_yeager_cd(wind)
        ch = large_yeager_ch(tair, surface)
        latent_heat = constants.L_v if L_latent is None else L_latent
        tau_x = -density * cd * wind * u
        tau_y = -density * cd * wind * v
        sensible = density * constants.c_pd * ch * wind * (tair - surface)
        latent = density * latent_heat * ch * wind * (qair - q_sfc)
        return tau_x, tau_y, sensible, latent, -latent / latent_heat

    pressure = constants.p_atm_std if slp_Pa is None else slp_Pa
    zero = jnp.zeros_like(surface)
    values = nemo_ncar_ocean_bulk(
        u, v, tair, qair, surface - constants.T_freeze, pressure,
        zero, zero, zero, zero,
        surface_humidity=q_sfc, air_density=rho_air, iterations=nb_iter,
    )
    return (
        -values["utau"], -values["vtau"], values["sensible"],
        values["latent"], values["evap"],
    )


__all__ = [
    "air_sea_fluxes",
    "exner_potential_temperature",
    "large_yeager_cd",
    "large_yeager_ch",
    "latent_heat_vaporization_sst",
    "moist_air_cp",
    "ncar_transfer_coefficients",
    "potential_air_temperature_10m",
    "pressure_at_height",
    "rho_air_moist",
    "seawater_q_sat",
]
