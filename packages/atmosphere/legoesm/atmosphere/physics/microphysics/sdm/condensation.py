"""Diffusional growth (condensation/evaporation) of super-droplets.

Faithful JAX port of the ERF super-droplet growth ODE
(``ERF_SuperDropletPCMassChange.H`` ``dRsqdt`` + ``MassChange.cpp`` coefficient
setup). Each droplet's wet radius evolves by vapor diffusion (Maxwell-Mason),
modified by the Kelvin curvature and Raoult solute terms (Köhler theory) and a
Fukuta-Walter transitional (Knudsen) correction to the diffusivity.

The ODE is integrated in ``u = R²`` (radius squared), which is the natural
variable: for a pure droplet far from activation it grows linearly,
``d(R²)/dt = 2(S-1)/(F_k+F_d)``.

.. math::

    \\frac{d R^2}{dt}
       = \\frac{2(S-1)}{F_k+F_d}
       - \\frac{2\\,(a/T)}{F_k+F_d}\\,\\frac{1}{R}
       + \\frac{2\\,b\\,N_s}{F_k+F_d}\\,\\frac{1}{R^3}

with

* ``F_k = (L/(R_v T) - 1)·(L ρ_l)/(K T)``     (latent-heat conduction term),
* ``F_d = (ρ_l R_v T)/(d_cf·D·e_s)``          (vapor-diffusion term),
* ``d_cf = (1+Kn)/(1+2Kn(1+Kn))``, ``Kn = λ_v/R``, ``λ_v = 2D/√(8 T R_v/π)``,
* ``a = 2σ/(R_v ρ_l)``                         (Kelvin curvature coeff),
* ``b = (3/4π)(M_w/ρ_l)``                      (Raoult solute coeff),
* ``N_s = m_s · i / M_s``                       (effective solute moles).

All physical constants come from :mod:`legoesm.constants`; the saturation
vapor pressure ``e_s(T)`` comes from :mod:`legoesm.thermo` (audit-mandated:
no re-implemented saturation curve). This module is JIT- and grad-compatible.

References
----------
* Shima et al. (2009) QJRMS 135:1307-1320.
* Rogers & Yau (1989) *A Short Course in Cloud Physics*.
* Pruppacher & Klett (1997) *Microphysics of Clouds and Precipitation*.
* ERF ``Source/Particles/ERF_SuperDropletPCMassChange.H`` (oracle).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from legoesm import constants
from legoesm.thermo import saturation_vapor_pressure
from legoesm.atmosphere.physics.microphysics.sdm.config import SDMConfig
from legoesm.atmosphere.physics.microphysics.sdm.particles import SuperDropletState

__physics_contract__ = {
    "summary": (
        "Super-droplet diffusional growth: vapor-diffusion (Maxwell-Mason) "
        "condensation/evaporation with Kelvin curvature and Raoult solute "
        "(Köhler) terms and a Knudsen diffusivity correction. The "
        "differentiable=True claim covers the default fixed-substep "
        "rk4/euler integrators; the opt-in rk4_adaptive mode (lax.while_loop) "
        "is jit-compatible but NOT reverse-mode differentiable."
    ),
    "inputs": {
        "radius": "m",
        "S": "1",
        "T": "K",
        "solute_mass": "kg",
        "dt": "s",
    },
    "outputs": {"radius": "m"},
    "sign_convention": (
        "S>1 (supersaturated) grows the droplet (dR>0); S<1 (subsaturated) "
        "evaporates it (dR<0). d(R^2)/dt has the sign of the net of the "
        "(S-1) supersaturation, Kelvin (negative), and solute (positive) terms."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Shima et al. (2009) QJRMS 135:1307; Rogers & Yau (1989); ERF SuperDropletPCMassChange (dRsqdt)",
    "idealized_test": (
        "Fixed S>1, no curvature/solute, R~15um: d(R^2)/dt = 2(S-1)/(F_k+F_d) "
        "(constant) reproduces the analytic Maxwell growth to machine precision; "
        "S<1 shrinks the droplet; a soluble droplet at S<=1 has a finite "
        "Köhler equilibrium radius (dR/dt = 0 fixed point)."
    ),
}

# Minimum radius [m] used to floor R² so the 1/R and 1/R³ terms and √ stay
# finite for (near-)evaporated droplets — a safety floor, not a tunable
# (well below any aerosol dry radius ~1e-9 m).
_R_FLOOR = 1.0e-9
_R_SQ_FLOOR = _R_FLOOR * _R_FLOOR

# Water molar mass in kg/mol (constants.M_H2O is g/mol; g->kg is a unit factor).
_M_H2O_KG = constants.M_H2O * 1.0e-3


def drsq_dt(
    r_sq: jax.Array,
    S: jax.Array,
    T: jax.Array,
    e_s: jax.Array,
    N_s: jax.Array,
    include_curvature: bool = True,
    include_solute: bool = True,
) -> jax.Array:
    """Right-hand side ``d(R²)/dt`` [m²/s] of the diffusional-growth ODE.

    Elementwise over arrays; ``S, T, e_s`` may be scalars (broadcast over the
    droplet ensemble) or per-droplet arrays. ``include_curvature`` and
    ``include_solute`` are static booleans (feature gating — Python ``if`` on a
    config flag, not a traced ``where``, so disabled terms are not traced).

    Parameters
    ----------
    r_sq : jax.Array
        Radius squared R² [m²].
    S : jax.Array
        Saturation ratio ``S = e/e_sat`` [-] (vapor-pressure based WMO RH —
        see :func:`legoesm.thermo.relative_humidity`; NOT ``q_v/q_sat``).
    T : jax.Array
        Temperature [K].
    e_s : jax.Array
        Saturation vapor pressure [Pa].
    N_s : jax.Array
        Effective solute moles per droplet [mol] (van't Hoff ``m_s·i/M_s``).
    include_curvature, include_solute : bool
        Toggle the Kelvin and Raoult terms.
    """
    D = constants.D_vapor
    K = constants.k_air
    Rv = constants.R_v
    rho_l = constants.rho_water
    L = constants.L_v

    r_sq = jnp.maximum(r_sq, _R_SQ_FLOOR)
    R = jnp.sqrt(r_sq)

    # Fukuta-Walter transitional correction to the diffusivity (Knudsen).
    lambda_v = 2.0 * D / jnp.sqrt(8.0 * T * Rv / jnp.pi)
    Kn = lambda_v / R
    d_cf = (1.0 + Kn) / (1.0 + 2.0 * Kn * (1.0 + Kn))

    F_k = (L / (Rv * T) - 1.0) * (L * rho_l) / (K * T)
    F_d = (rho_l * Rv * T) / (d_cf * D * e_s)
    denom = F_k + F_d

    out = 2.0 * (S - 1.0) / denom

    if include_curvature:
        a = 2.0 * constants.sigma_water / (Rv * rho_l)   # Kelvin coeff [m·K]
        out = out - 2.0 * (a / T) / denom / R

    if include_solute:
        b = (3.0 / (4.0 * jnp.pi)) * (_M_H2O_KG / rho_l)  # Raoult coeff [m³/mol]
        out = out + 2.0 * b * N_s / denom / (r_sq * R)    # / R³

    return out


def _rk4_step(r_sq, h, S, T, e_s, N_s, include_curvature, include_solute):
    k1 = drsq_dt(r_sq, S, T, e_s, N_s, include_curvature, include_solute)
    k2 = drsq_dt(r_sq + 0.5 * h * k1, S, T, e_s, N_s, include_curvature, include_solute)
    k3 = drsq_dt(r_sq + 0.5 * h * k2, S, T, e_s, N_s, include_curvature, include_solute)
    k4 = drsq_dt(r_sq + h * k3, S, T, e_s, N_s, include_curvature, include_solute)
    return r_sq + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _euler_step(r_sq, h, S, T, e_s, N_s, include_curvature, include_solute):
    return r_sq + h * drsq_dt(
        r_sq, S, T, e_s, N_s, include_curvature, include_solute
    )


def drsq_dt_jac(
    r_sq: jax.Array,
    T: jax.Array,
    e_s: jax.Array,
    N_s: jax.Array,
    include_curvature: bool = True,
    include_solute: bool = True,
) -> jax.Array:
    """Jacobian ``∂(dR²/dt)/∂(R²)`` [1/s] of the growth ODE (ERF ``rhs_jac``).

    Only the Kelvin (``β/R``) and Raoult (``γ/R³``) terms depend on ``R²``; the
    Maxwell ``α`` term is constant, so its derivative is zero — a pure droplet
    with curvature/solute off has ``jac = 0`` (no stiffness)::

        d(β u^{-1/2})/du   = -½ β u^{-3/2}
        d(γ u^{-3/2})/du   = -(3/2) γ u^{-5/2}

    with ``β = -2(a/T)/(F_k+F_d)`` and ``γ = +2 b N_s/(F_k+F_d)`` exactly as in
    :func:`drsq_dt`. Used by the adaptive integrator as the stiffness estimate
    ``τ`` (``dt = cfl/|τ|``).

    This is the oracle's APPROXIMATE Jacobian: like ERF, it evaluates
    ``F_k+F_d`` (including the Knudsen correction ``d_cf``) at the current
    ``R²`` but neglects ``∂(F_d)/∂R²`` through ``d_cf`` — so it differs from
    ``jax.grad(drsq_dt)`` (and is exactly 0 for a pure Maxwell droplet).
    Adequate for step-size control; do not use it as an exact derivative.
    """
    D = constants.D_vapor
    K = constants.k_air
    Rv = constants.R_v
    rho_l = constants.rho_water
    L = constants.L_v

    r_sq = jnp.maximum(r_sq, _R_SQ_FLOOR)
    R = jnp.sqrt(r_sq)

    lambda_v = 2.0 * D / jnp.sqrt(8.0 * T * Rv / jnp.pi)
    Kn = lambda_v / R
    d_cf = (1.0 + Kn) / (1.0 + 2.0 * Kn * (1.0 + Kn))
    F_k = (L / (Rv * T) - 1.0) * (L * rho_l) / (K * T)
    F_d = (rho_l * Rv * T) / (d_cf * D * e_s)
    denom = F_k + F_d

    R_inv = 1.0 / R
    R_inv3 = R_inv * R_inv * R_inv
    R_inv5 = R_inv3 * R_inv * R_inv

    out = jnp.zeros_like(r_sq)
    if include_curvature:
        a = 2.0 * constants.sigma_water / (Rv * rho_l)
        beta = -2.0 * (a / T) / denom
        out = out - 0.5 * beta * R_inv3
    if include_solute:
        b = (3.0 / (4.0 * jnp.pi)) * (_M_H2O_KG / rho_l)
        gamma = 2.0 * b * N_s / denom
        out = out - 1.5 * gamma * R_inv5
    return out


def _integrate_adaptive_rk4(
    r_sq0, t_final, S, T, e_s, N_s,
    include_curvature, include_solute, cfl, stol, max_steps,
):
    """ERF adaptive stiffness-based RK4 in ``u = R²`` for ONE droplet.

    Faithful port of the ERF ``TI::rk4`` loop semantics:

    * each accepted step recomputes ``dt = cfl/|τ|`` from the stiffness
      ``τ = rhs_jac(u)`` (``τ = 0`` — pure Maxwell — gives the whole remaining
      interval, exactly ERF's ``cfl/0 → ∞`` then limit-to-``t_final``);
    * any RK stage ``≤ 0`` or a non-finite/non-positive result halves ``dt``
      and retries;
    * the ERF too-small exit (``dt < 1e-12·cfl/|τ|`` AND ``dt < 1e-12·t_final``)
      marks the droplet *unconverged*: its radius is left UNCHANGED (ERF skips
      the particle update for unconverged droplets);
    * steady-state exit when ``snorm = |u_new-u|/u < stol``;
    * the step cap counts ACCEPTED steps only (ERF ``n_step``); halvings are
      bounded by the too-small exit (geometric halving terminates in ~40
      rejections). Hitting the cap with ``t < t_final`` returns the partially
      integrated radius — exactly what ERF does (``a_success`` stays true at
      the cap); with the default cap of 100 accepted steps this is rare.

    Implemented as a single ``lax.while_loop`` whose iterations are either an
    accepted step or one halving. NOT reverse-mode differentiable
    (``while_loop``) — use the fixed-substep integrators for gradient work.
    """
    dtype = r_sq0.dtype
    eps_exit = jnp.asarray(1.0e-12, dtype)

    def _dt_from_tau(u, t):
        tau = jnp.abs(drsq_dt_jac(u, T, e_s, N_s, include_curvature, include_solute))
        dt_stiff = jnp.where(tau > 0.0, cfl / tau, t_final - t)
        return jnp.minimum(dt_stiff, t_final - t), tau

    def cond(st):
        u, t, dt, n, failed, steady = st
        # n counts ACCEPTED steps only (ERF's n_step) — halvings are bounded
        # by the too-small exit, exactly as the oracle's unbounded inner loop.
        return (~failed) & (~steady) & (t < t_final) & (n < max_steps)

    def body(st):
        u, t, dt, n, failed, steady = st
        k1 = drsq_dt(u, S, T, e_s, N_s, include_curvature, include_solute)
        u2 = u + 0.5 * dt * k1
        k2 = drsq_dt(u2, S, T, e_s, N_s, include_curvature, include_solute)
        u3 = u + 0.5 * dt * k2
        k3 = drsq_dt(u3, S, T, e_s, N_s, include_curvature, include_solute)
        u4 = u + dt * k3
        k4 = drsq_dt(u4, S, T, e_s, N_s, include_curvature, include_solute)
        u_new = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        ok = ((u2 > 0.0) & (u3 > 0.0) & (u4 > 0.0)
              & jnp.isfinite(u_new) & (u_new > 0.0))

        # Accepted: advance, check steady, recompute dt from fresh stiffness.
        snorm = jnp.abs(u_new - u) / jnp.maximum(u, _R_SQ_FLOOR)
        steady_acc = snorm < stol
        t_acc = t + dt
        dt_acc, _ = _dt_from_tau(u_new, t_acc)

        # Rejected: halve; ERF too-small exit -> unconverged failure.
        dt_half = 0.5 * dt
        _, tau_here = _dt_from_tau(u, t)
        dt_ref = jnp.where(tau_here > 0.0, cfl / tau_here, t_final)
        too_small = (dt_half < eps_exit * dt_ref) & (dt_half < eps_exit * t_final)

        u = jnp.where(ok, u_new, u)
        t = jnp.where(ok, t_acc, t)
        dt = jnp.where(ok, jnp.maximum(dt_acc, 0.0), dt_half)
        steady = jnp.where(ok, steady_acc, steady)
        failed = jnp.where(ok, failed, too_small)
        n = jnp.where(ok, n + 1, n)   # accepted steps only, like ERF n_step
        return (u, t, dt, n, failed, steady)

    dt0, _ = _dt_from_tau(r_sq0, jnp.zeros((), dtype))
    init = (r_sq0, jnp.zeros((), dtype), dt0,
            jnp.zeros((), jnp.int32),
            jnp.zeros((), jnp.bool_), jnp.zeros((), jnp.bool_))
    u, t, dt, n, failed, steady = lax.while_loop(cond, body, init)
    # Unconverged droplets keep their initial radius (ERF skips their update).
    return jnp.where(failed, r_sq0, u)


def integrate_radius(
    state: SuperDropletState,
    S: jax.Array,
    T: jax.Array,
    dt: float | jax.Array,
    cfg: SDMConfig,
) -> SuperDropletState:
    """Advance every droplet's radius by diffusional growth over ``dt``.

    The growth ODE is integrated in R² with ``cfg.n_substeps_condensation``
    equal sub-steps using the configured integrator. Only ``active`` droplets
    change; inactive slots keep their radius. Returns a new
    :class:`SuperDropletState` with the updated ``radius``.

    Parameters
    ----------
    state : SuperDropletState
        Droplet ensemble.
    S, T : jax.Array
        Ambient saturation ratio [-] and temperature [K] (scalar or per-droplet).
    dt : float
        Physics step [s].
    cfg : SDMConfig
        Scheme configuration. ``cfg`` is a **static** argument (it carries
        Python ``str``/``bool``/``int`` fields): when wrapping a call in
        ``jax.jit`` pass it via closure or ``static_argnames=("cfg",)`` —
        it cannot be a traced argument. This matches how every legoESM
        scheme threads its config (resolved at trace time, never traced).

    Notes
    -----
    ``"rk4"``/``"euler"`` use ``cfg.n_substeps_condensation`` fixed equal
    sub-steps (reverse-mode differentiable; the ODE stiffens as ``R -> 0``, so
    increase the sub-step count for strong evaporation). ``"rk4_adaptive"`` is
    the ERF stiffness-based integrator (``dt = cfl/|τ|`` from
    :func:`drsq_dt_jac`, stage-positivity step-halving, too-small unconverged
    exit leaving the radius unchanged, steady-state early exit) — per-droplet
    ``lax.while_loop``, NOT reverse-mode differentiable.
    """
    if cfg.condensation_integrator == "rk4":
        step_fn = _rk4_step
    elif cfg.condensation_integrator == "euler":
        step_fn = _euler_step
    elif cfg.condensation_integrator == "rk4_adaptive":
        step_fn = None
    else:
        raise ValueError(
            f"Unknown SDM condensation_integrator: {cfg.condensation_integrator!r} "
            "(expected 'rk4', 'euler', or 'rk4_adaptive')"
        )

    n_sub = int(cfg.n_substeps_condensation)
    if n_sub < 1:
        raise ValueError(
            f"n_substeps_condensation must be >= 1, got {cfg.n_substeps_condensation!r}"
        )

    # Pin S, T, dt to the droplet dtype so e_s and the R² update do not silently
    # promote/demote across the integration (float32 radius vs float64 scalars).
    dtype = state.radius.dtype
    S = jnp.asarray(S, dtype=dtype)
    T = jnp.asarray(T, dtype=dtype)
    e_s = saturation_vapor_pressure(T)
    if cfg.include_solute:
        N_s = state.solute_mass * cfg.solute_ionization / cfg.solute_molar_mass
    else:
        N_s = jnp.zeros_like(state.radius)

    include_curvature = cfg.include_curvature
    include_solute = cfg.include_solute

    if cfg.condensation_integrator == "rk4_adaptive":
        # Per-droplet adaptive integration: broadcast the ambient fields to the
        # droplet axis and vmap the single-droplet while_loop.
        n_sd = state.radius.shape[0]
        t_final = jnp.asarray(dt, dtype=dtype)
        S_b = jnp.broadcast_to(S, (n_sd,)).astype(dtype)
        T_b = jnp.broadcast_to(T, (n_sd,)).astype(dtype)
        e_s_b = jnp.broadcast_to(e_s, (n_sd,)).astype(dtype)
        cfl = jnp.asarray(cfg.adaptive_cfl, dtype=dtype)
        stol = jnp.asarray(cfg.adaptive_stol, dtype=dtype)
        max_steps = jnp.asarray(int(cfg.adaptive_max_steps), jnp.int32)
        r_sq = jax.vmap(
            lambda u0, s, t, es, ns: _integrate_adaptive_rk4(
                u0, t_final, s, t, es, ns,
                include_curvature, include_solute, cfl, stol, max_steps,
            )
        )(state.radius**2, S_b, T_b, e_s_b, N_s)
    else:
        h = jnp.asarray(dt, dtype=dtype) / n_sub

        def body(_, r_sq):
            r_sq = step_fn(r_sq, h, S, T, e_s, N_s, include_curvature, include_solute)
            return jnp.maximum(r_sq, _R_SQ_FLOOR)

        r_sq = lax.fori_loop(0, n_sub, body, state.radius**2)

    radius_new = jnp.sqrt(jnp.maximum(r_sq, _R_SQ_FLOOR))
    # Inactive droplets do not grow.
    radius_new = jnp.where(state.active > 0, radius_new, state.radius)
    return state._replace(radius=radius_new)
