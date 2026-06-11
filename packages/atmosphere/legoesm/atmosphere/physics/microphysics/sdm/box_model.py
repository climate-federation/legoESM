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
``condensation`` / ``coupling``. Multiplicities are *per unit air mass*
[1/kg], so the liquid mixing ratio is simply ``q_l = Σ ξ_i m_i`` and no cell
volume is needed. Condensation only (no coalescence) — the canonical
activation parcel. ``lax.scan`` over steps; pure and ``jit``-able.

State variables: parcel temperature ``T`` [K], pressure ``p`` [Pa], vapor
mixing ratio ``q_v`` [kg/kg], height ``z`` [m], and the droplet ensemble.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.microphysics.sdm.config import SDMConfig
from legoesm.atmosphere.physics.microphysics.sdm.condensation import integrate_radius
from legoesm.atmosphere.physics.microphysics.sdm.coupling import condensation_exchange
from legoesm.atmosphere.physics.microphysics.sdm.particles import (
    SuperDropletState,
    water_mass_per_droplet,
)


class ParcelState(NamedTuple):
    """Thermodynamic + droplet state of a rising adiabatic parcel."""

    droplets: SuperDropletState  # per-unit-air-mass multiplicities [1/kg]
    T: jax.Array                 # temperature [K]
    p: jax.Array                 # pressure [Pa]
    q_v: jax.Array               # vapor mixing ratio [kg/kg]
    z: jax.Array                 # height [m]


def liquid_mixing_ratio(droplets: SuperDropletState) -> jax.Array:
    """Liquid water mixing ratio ``q_l = Σ_i ξ_i m_i`` [kg/kg] (per-mass ξ)."""
    return jnp.sum(droplets.active * droplets.multiplicity
                   * water_mass_per_droplet(droplets))


def saturation_ratio(parcel: ParcelState) -> jax.Array:
    """Ambient saturation ratio ``S = q_v / q_sat(T, p)`` [-]."""
    return parcel.q_v / saturation_mixing_ratio(parcel.T, parcel.p)


def parcel_step(
    parcel: ParcelState,
    w: float | jax.Array,
    dt: float | jax.Array,
    cfg: SDMConfig,
) -> ParcelState:
    """Advance the parcel by ``dt`` (operator split: lift, then condense).

    1. lift: ``z += w·dt``; dry-adiabatic cooling ``T -= (g/c_p)·w·dt``;
       hydrostatic ``p -= ρ·g·w·dt`` with ``ρ = p/(R_d T)``;
    2. condense: grow droplets at the current saturation ratio, then move the
       condensed mass from vapor to liquid and release its latent heat
       (total water conserved).
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
    q_sat = saturation_mixing_ratio(T_lift, p_lift)
    S = parcel.q_v / q_sat
    q_l_before = liquid_mixing_ratio(parcel.droplets)
    droplets_new = integrate_radius(parcel.droplets, S, T_lift, dt, cfg)
    q_l_after = liquid_mixing_ratio(droplets_new)
    q_v_new, T_new = condensation_exchange(
        q_l_before, q_l_after, parcel.q_v, T_lift, c_p=c_p, L_v=constants.L_v)

    return ParcelState(droplets=droplets_new, T=T_new, p=p_lift, q_v=q_v_new, z=z_new)


def run_parcel(
    parcel0: ParcelState,
    w: float | jax.Array,
    dt: float | jax.Array,
    n_steps: int,
    cfg: SDMConfig,
) -> tuple[ParcelState, dict]:
    """Integrate the parcel for ``n_steps`` steps.

    Returns ``(final_parcel, history)`` where ``history`` stacks the
    saturation ratio ``S``, liquid mixing ratio ``q_l``, temperature ``T``,
    total water ``q_t = q_v + q_l``, and mean droplet radius at every step
    (length ``n_steps`` arrays) for diagnostics/validation.
    """
    def body(parcel, _):
        parcel = parcel_step(parcel, w, dt, cfg)
        d = parcel.droplets
        q_l = liquid_mixing_ratio(d)
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
