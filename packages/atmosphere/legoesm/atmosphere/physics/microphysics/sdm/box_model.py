"""Adiabatic-parcel driver for the Super-Droplet Method.

A 0-D *persistent-particle* test bed — the natural home for SDM, which is
Lagrangian and needs particle state carried across time (the stateless
column-physics interface cannot). A closed air parcel rises at a constant
updraft ``w``: it cools dry-adiabatically, the relative humidity rises above
saturation, the super-droplets grow by condensation (``condensation.py``),
their latent heat warms the air and depletes the vapor (``coupling.py``), and
the supersaturation peaks then relaxes — the classic warm-cloud activation
problem (Köhler).

This is a driver (not a single-tendency scheme); the physics lives in
``condensation`` / ``coupling``. ``multiplicity`` keeps the same meaning as
everywhere else in the package — the number of real droplets represented (a
count) — here the count *in the parcel*; the liquid mixing ratio is
``q_l = (Σ ξ_i m_i) / M_air`` with ``M_air`` the parcel's conserved dry-air
mass [kg]. Condensation only (no coalescence) — the canonical activation
parcel. ``lax.scan`` over steps; pure and ``jit``-able.

State variables: parcel temperature ``T`` [K], pressure ``p`` [Pa], vapor
mixing ratio ``q_v`` [kg/kg], height ``z`` [m], and the droplet ensemble.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from legoesm import constants
from legoesm.thermo import relative_humidity
from legoesm.atmosphere.physics.microphysics.sdm.config import SDMConfig
from legoesm.atmosphere.physics.microphysics.sdm.condensation import integrate_radius
from legoesm.atmosphere.physics.microphysics.sdm.coupling import condensation_exchange
from legoesm.atmosphere.physics.microphysics.sdm.particles import (
    SuperDropletState,
    represented_water_mass,
)


class ParcelState(NamedTuple):
    """Thermodynamic + droplet state of a rising adiabatic parcel.

    ``droplets.multiplicity`` is the number of real droplets represented in the
    parcel (a count — the same convention as the rest of the package).
    """

    droplets: SuperDropletState  # multiplicities are real-droplet counts [-]
    T: jax.Array                 # temperature [K]
    p: jax.Array                 # pressure [Pa]
    q_v: jax.Array               # vapor mixing ratio [kg/kg]
    z: jax.Array                 # height [m]


def liquid_mixing_ratio(
    droplets: SuperDropletState, M_air: float | jax.Array,
) -> jax.Array:
    """Liquid water mixing ratio ``q_l = (Σ_i ξ_i m_i) / M_air`` [kg/kg].

    ``M_air`` is the parcel's conserved dry-air mass [kg]; the numerator is the
    total represented liquid water in the parcel (``represented_water_mass``
    returns the per-droplet contribution ``ξ_i m_i``, summed here).
    """
    return jnp.sum(represented_water_mass(droplets)) / M_air


def saturation_ratio(parcel: ParcelState) -> jax.Array:
    """Ambient saturation ratio ``S = e/e_sat`` [-] (vapor-pressure based).

    Uses :func:`legoesm.thermo.relative_humidity` — the saturation ratio the
    droplet growth law requires, NOT the mixing-ratio ratio ``q_v/q_sat``.
    """
    return relative_humidity(parcel.T, parcel.p, parcel.q_v)


def parcel_step(
    parcel: ParcelState,
    w: float | jax.Array,
    dt: float | jax.Array,
    cfg: SDMConfig,
    M_air: float | jax.Array,
) -> ParcelState:
    """Advance the parcel by ``dt`` (operator split: lift, then condense).

    1. lift: ``z += w·dt``; dry-adiabatic cooling ``T -= (g/c_p)·w·dt``;
       hydrostatic ``p -= ρ·g·w·dt`` with ``ρ = p/(R_d T)``;
    2. condense: grow droplets at the current saturation ratio ``S = e/e_sat``,
       then move the condensed mass from vapor to liquid and release its latent
       heat (total water conserved).
    """
    g = constants.g
    c_p = constants.c_pd
    R_d = constants.R_d

    w = jnp.asarray(w, dtype=parcel.T.dtype)
    dt = jnp.asarray(dt, dtype=parcel.T.dtype)

    # 1. adiabatic lift
    rho_air = parcel.p / (R_d * parcel.T)
    dz = w * dt
    T_lift = parcel.T - (g / c_p) * dz
    p_lift = parcel.p - rho_air * g * dz
    z_new = parcel.z + dz

    # 2. condensation at the lifted state
    S = relative_humidity(T_lift, p_lift, parcel.q_v)
    q_l_before = liquid_mixing_ratio(parcel.droplets, M_air)
    droplets_new = integrate_radius(parcel.droplets, S, T_lift, dt, cfg)
    q_l_after = liquid_mixing_ratio(droplets_new, M_air)
    q_v_new, T_new = condensation_exchange(
        q_l_before, q_l_after, parcel.q_v, T_lift, c_p=c_p, L_v=constants.L_v)

    return ParcelState(droplets=droplets_new, T=T_new, p=p_lift, q_v=q_v_new, z=z_new)


def run_parcel(
    parcel0: ParcelState,
    w: float | jax.Array,
    dt: float | jax.Array,
    n_steps: int,
    cfg: SDMConfig,
    M_air: float | jax.Array = 1.0,
) -> tuple[ParcelState, dict]:
    """Integrate the parcel for ``n_steps`` steps.

    ``M_air`` is the parcel's dry-air mass [kg] (default 1 kg — a unit parcel,
    so ``multiplicity`` is the real-droplet count per kg of air). Returns
    ``(final_parcel, history)`` where ``history`` stacks the saturation ratio
    ``S``, liquid mixing ratio ``q_l``, temperature ``T``, total water
    ``q_t = q_v + q_l``, and mean droplet radius at every step (length
    ``n_steps`` arrays) for diagnostics/validation.

    When wrapping in ``jax.jit`` use
    ``static_argnames=("n_steps", "cfg")`` — ``n_steps`` is the ``lax.scan``
    length and ``cfg`` carries Python str/bool/int fields.
    """
    def body(parcel, _):
        parcel = parcel_step(parcel, w, dt, cfg, M_air)
        d = parcel.droplets
        q_l = liquid_mixing_ratio(d, M_air)
        active = d.active
        mean_r = jnp.sum(active * d.radius) / jnp.maximum(jnp.sum(active), 1.0)
        diag = {
            "S": saturation_ratio(parcel),
            "q_l": q_l,
            "T": parcel.T,
            "q_t": parcel.q_v + q_l,
            "mean_radius": mean_r,
        }
        return parcel, diag

    final, history = lax.scan(body, parcel0, xs=None, length=n_steps)
    return final, history
