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
from jax import random

from legoesm.thermo import relative_humidity
from legoesm.atmosphere.physics.microphysics.sdm.config import SDMConfig
from legoesm.atmosphere.physics.microphysics.sdm.coalescence import coalescence_step
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


# ==========================================================================
# Composed persistent box: condensation + collision-coalescence
# ==========================================================================
class BoxState(NamedTuple):
    """Persistent state of a well-mixed SDM box (no lift).

    ``droplets.multiplicity`` is the number of real droplets represented in the
    box (a count). ``key`` is the threaded ``jax.random`` state consumed by the
    stochastic coalescence.
    """

    droplets: SuperDropletState
    T: jax.Array            # temperature [K]
    p: jax.Array            # pressure [Pa]
    q_v: jax.Array          # vapor mixing ratio [kg/kg]
    key: jax.Array          # PRNG key


def box_step(
    box: BoxState,
    V_cell: float | jax.Array,
    M_air: float | jax.Array,
    dt: float | jax.Array,
    cfg: SDMConfig,
    do_condensation: bool = True,
    do_coalescence: bool = True,
) -> BoxState:
    """One full SDM box step: condensation then collision-coalescence.

    Mirrors the ERF process order (phaseChange -> coalescence) on a *persistent*
    super-droplet population in a well-mixed box (no advection or sedimentation).
    ``do_condensation`` / ``do_coalescence`` are static feature flags (e.g.
    collision-only for the Golovin test).

    ``M_air`` is the box's **conserved** dry-air mass [kg] (an isobaric box's
    volume expands with latent heating, but its air mass is fixed); the liquid
    mixing ratio is ``q_l = Σξm / M_air`` and the box air density is
    ``ρ = M_air/V_cell``. Coalescence consumes a fresh split of ``box.key``.
    """
    droplets, T, p, q_v, key = box
    dt = jnp.asarray(dt, dtype=T.dtype)
    rho = M_air / V_cell

    # 1. condensation / evaporation (latent heat + vapor exchange)
    if do_condensation:
        S = relative_humidity(T, p, q_v)
        q_l_before = liquid_mixing_ratio(droplets, M_air)
        droplets = integrate_radius(droplets, S, T, dt, cfg)
        q_l_after = liquid_mixing_ratio(droplets, M_air)
        q_v, T = condensation_exchange(
            q_l_before, q_l_after, q_v, T, c_p=constants.c_pd, L_v=constants.L_v)

    # 2. collision-coalescence (Shima Monte-Carlo)
    if do_coalescence:
        key, sub = random.split(key)
        droplets = coalescence_step(droplets, V_cell, rho, p, T, dt, sub, cfg)

    return BoxState(droplets=droplets, T=T, p=p, q_v=q_v, key=key)


def run_box(
    box0: BoxState,
    V_cell: float | jax.Array,
    dt: float | jax.Array,
    n_steps: int,
    cfg: SDMConfig,
    do_condensation: bool = True,
    do_coalescence: bool = True,
) -> tuple[BoxState, dict]:
    """Integrate the composed SDM box for ``n_steps`` steps (``lax.scan``).

    The box's dry-air mass ``M_air = ρ_0·V_cell`` (from the initial state) is
    held constant (isobaric box). Returns ``(final_box, history)`` with per-step
    stacked diagnostics: total liquid mixing ratio ``q_l``, rain liquid
    ``q_rain`` (droplets with ``R >= cfg.r_rain``), vapor ``q_v``, temperature
    ``T``, total water ``q_t = q_v + q_l``, represented number ``N`` (Σξ), and
    mass-weighted mean radius. ``static_argnames=("n_steps", "cfg",
    "do_condensation", "do_coalescence")`` when wrapping in ``jax.jit``.
    """
    # Conserved DRY-air mass: q_v is a dry-air mixing ratio, so the dry-air
    # partial pressure is p - e (e = vapor partial pressure), not the total p.
    e0 = box0.p * box0.q_v / (constants.epsilon + box0.q_v)
    M_air = (box0.p - e0) / (constants.R_d * box0.T) * V_cell

    def body(box, _):
        box = box_step(box, V_cell, M_air, dt, cfg, do_condensation, do_coalescence)
        d = box.droplets
        m = represented_water_mass(d)                 # per-droplet ξ_i m_i
        q_l = jnp.sum(m) / M_air
        is_rain = d.radius >= cfg.r_rain
        q_rain = jnp.sum(jnp.where(is_rain, m, 0.0)) / M_air
        N = jnp.sum(d.active * d.multiplicity)
        # mass-weighted mean radius Σ(ξ m R)/Σ(ξ m)
        mw = m
        mean_r = jnp.sum(mw * d.radius) / jnp.maximum(jnp.sum(mw), 1e-300)
        # per-step finiteness flag (1.0 = all finite) for full-trajectory stability
        finite = (jnp.all(jnp.isfinite(d.radius))
                  & jnp.all(jnp.isfinite(d.multiplicity))
                  & jnp.isfinite(box.T) & jnp.isfinite(box.q_v)).astype(box.T.dtype)
        diag = {
            "q_l": q_l, "q_rain": q_rain, "q_v": box.q_v, "T": box.T,
            "q_t": box.q_v + q_l, "N": N, "mean_radius": mean_r, "finite": finite,
        }
        return box, diag

    final, history = lax.scan(body, box0, xs=None, length=n_steps)
    return final, history


def box_water(
    box: BoxState,
    V_cell: float | jax.Array,
    r_rain: float = float("inf"),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Liquid diagnostics ``(q_l, q_rain, N)`` for a box state.

    Uses the SAME dry-air-mass normalizer as :func:`run_box`
    (``M_air = (p - e)/(R_d T)·V_cell``), so an explicitly computed *initial*
    baseline matches the scanned history exactly (the history's first entry is
    the post-first-step state, not the initial). ``q_rain`` is the liquid in
    droplets with ``R >= r_rain`` (default inf -> 0). ``N = Σξ``.
    """
    e = box.p * box.q_v / (constants.epsilon + box.q_v)
    M_air = (box.p - e) / (constants.R_d * box.T) * V_cell
    m = represented_water_mass(box.droplets)
    q_l = jnp.sum(m) / M_air
    q_rain = jnp.sum(jnp.where(box.droplets.radius >= r_rain, m, 0.0)) / M_air
    N = jnp.sum(box.droplets.active * box.droplets.multiplicity)
    return q_l, q_rain, N
