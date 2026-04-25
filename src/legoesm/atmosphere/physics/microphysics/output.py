"""Microphysics output containers.

HydrometeorState holds the prognostic hydrometeor fields passed to backends.
MicrophysicsOutput is the common interface returned by all backends.

All backends accept and return the same containers so that integration
code can be backend-agnostic.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class HydrometeorState(NamedTuple):
    """Hydrometeor state for backends. All fields shape (ncol, nlev)."""
    q_c: jax.Array    # cloud water [kg/kg]
    q_r: jax.Array    # rain water [kg/kg]
    q_i: jax.Array    # cloud ice [kg/kg]
    q_s: jax.Array    # snow [kg/kg]
    q_g: jax.Array    # graupel [kg/kg]
    N_c: jax.Array    # cloud droplet number [1/kg]
    N_r: jax.Array    # rain drop number [1/kg]
    N_i: jax.Array    # ice crystal number [1/kg]


class MicrophysicsOutput(NamedTuple):
    """Backend-agnostic output. All (ncol, nlev) except precipitation (ncol,)."""
    dT_dt: jax.Array          # latent heating [K/s]
    dq_v_dt: jax.Array        # vapor tendency [kg/kg/s]
    dq_c_dt: jax.Array        # cloud water tendency
    dq_r_dt: jax.Array        # rain tendency
    dq_i_dt: jax.Array        # ice tendency
    dq_s_dt: jax.Array        # snow tendency
    dq_g_dt: jax.Array        # graupel tendency
    dN_c_dt: jax.Array        # cloud number tendency [1/kg/s]
    dN_r_dt: jax.Array        # rain number tendency
    dN_i_dt: jax.Array        # ice number tendency
    precipitation: jax.Array  # surface precip [kg/m^2/s]


def make_zero_hydrometeors(ncol: int, nlev: int) -> HydrometeorState:
    """Create a zero-initialized HydrometeorState."""
    z = jnp.zeros((ncol, nlev))
    return HydrometeorState(
        q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
        N_c=z, N_r=z, N_i=z,
    )


def make_zero_output(ncol: int, nlev: int) -> MicrophysicsOutput:
    """Create a zero-initialized MicrophysicsOutput."""
    z2 = jnp.zeros((ncol, nlev))
    z1 = jnp.zeros((ncol,))
    return MicrophysicsOutput(
        dT_dt=z2, dq_v_dt=z2, dq_c_dt=z2, dq_r_dt=z2,
        dq_i_dt=z2, dq_s_dt=z2, dq_g_dt=z2,
        dN_c_dt=z2, dN_r_dt=z2, dN_i_dt=z2,
        precipitation=z1,
    )


def sedimentation_tendency(
    q: jax.Array,
    rho: jax.Array,
    V_t: jax.Array,
    dz: jax.Array,
) -> jax.Array:
    """Compute sedimentation tendency from vertical flux divergence.

    Parameters
    ----------
    q : jax.Array
        Hydrometeor mixing ratio [kg/kg], shape (ncol, nlev).
    rho : jax.Array
        Air density [kg/m^3], shape (ncol, nlev).
    V_t : jax.Array
        Terminal velocity [m/s], shape (ncol, nlev).
    dz : jax.Array
        Layer thickness [m], shape (ncol, nlev).

    Returns
    -------
    jax.Array
        Sedimentation tendency [kg/kg/s], shape (ncol, nlev).
    """
    q_pos = jnp.clip(q, 0.0, None)
    flux = V_t * q_pos * rho  # (ncol, nlev)

    # Flux from above: zero at top, flux[k-1] enters level k.  Use
    # ``jnp.pad`` (single Pad HLO) instead of allocating a fresh
    # zero buffer + concatenate.
    flux_in = jnp.pad(flux[:, :-1], ((0, 0), (1, 0)))
    dz_safe = jnp.clip(dz, 1.0, None)
    return (flux_in - flux) / (rho * dz_safe)
