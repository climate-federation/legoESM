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


def _integrate_adaptive(
    r_sq0, t_final, T, e_s, N_s,
    include_curvature, include_solute, cfl, stol, max_steps,
    attempt,
):
    """ERF adaptive stiffness-based outer loop in ``u = R²`` for ONE droplet.

    Shared by every ERF ``TI`` integrator (rk4/be/cn/dirk2) — the per-method
    stage math is supplied as ``attempt(u, dt) -> (u_new, ok)``:

    * each accepted step recomputes ``dt = cfl/|τ|`` from the stiffness
      ``τ = rhs_jac(u)`` (``τ = 0`` — pure Maxwell — gives the whole remaining
      interval, exactly ERF's ``cfl/0 → ∞`` then limit-to-``t_final``);
    * a rejected attempt (``ok`` False: bad stage, non-convergence,
      non-finite/non-positive result) halves ``dt`` and retries;
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
        u_new, ok = attempt(u, dt)
        ok = ok & jnp.isfinite(u_new) & (u_new > 0.0)

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


def _newton_solve(u_init, rhs, mu, S, T, e_s, N_s,
                  include_curvature, include_solute,
                  rtol, atol, stol, maxits):
    """Scalar Newton solve of ``mu·u − F(u) − rhs = 0`` (ERF ``NewtonSolver``).

    ``F`` is the growth RHS :func:`drsq_dt` and the Newton slope uses the
    oracle's approximate Jacobian (``mu − rhs_jac``). Exit conditions exactly
    as ERF: absolute residual ``≤ atol``; relative residual (to the first
    iterate's) ``≤ rtol``; step size ``|du|/|u| ≤ stol``; non-finite residual
    or ``u ≤ 0`` fails. Returns ``(u, converged)``.
    """
    dtype = u_init.dtype

    def cond(st):
        u, res0, k, converged, failed = st
        return (~converged) & (~failed) & (k < maxits)

    def body(st):
        u, res0, k, converged, failed = st
        residual = mu * u - (rhs + drsq_dt(u, S, T, e_s, N_s,
                                           include_curvature, include_solute))
        res_a = jnp.abs(residual)
        res0 = jnp.where(k == 0, jnp.where(res_a > 0.0, res_a, 1.0), res0)
        res_r = res_a / res0
        conv_now = (res_a <= atol) | (res_r <= rtol)
        bad = ~jnp.isfinite(res_a)

        slope = mu - drsq_dt_jac(u, T, e_s, N_s,
                                 include_curvature, include_solute)
        du = -residual / slope
        small_step = jnp.abs(du) / jnp.maximum(jnp.abs(u), _R_SQ_FLOOR) <= stol
        u_next = u + du
        nonpos = u_next <= 0.0

        # ERF order: convergence checked BEFORE the update; the small-step
        # exit accepts the pre-update u (treated as converged).
        u = jnp.where(conv_now | bad | small_step, u, u_next)
        converged = converged | conv_now | (small_step & ~bad)
        failed = failed | bad | (~conv_now & ~small_step & nonpos)
        return (u, res0, k + 1, converged, failed)

    init = (u_init, jnp.ones((), dtype), jnp.zeros((), jnp.int32),
            jnp.zeros((), jnp.bool_), jnp.zeros((), jnp.bool_))
    u, _, _, converged, failed = lax.while_loop(cond, body, init)
    return u, converged & ~failed & (u > 0.0)


def _make_attempt(
    method, S, T, e_s, N_s, include_curvature, include_solute,
    newton_rtol, newton_atol, newton_stol, newton_maxits, dtype,
):
    """Build the per-method ``attempt(u, dt) -> (u_new, ok)`` stage kernel.

    Faithful per-step math of the ERF ``TI`` integrators:

    * ``rk4`` — 4 explicit stages; any stage ``≤ 0`` rejects.
    * ``be`` — Newton solve of ``μu' − F(u') = μu``, ``μ = 1/dt``.
    * ``cn`` — Crank-Nicolson: Newton solve of ``μu₂ − F(u₂) = μ(u + ½dt·f₁)``
      with ``μ = 1/(½dt)``; reject if ``u₂ ≤ 0``; the returned update is
      ``u + ½dt(f₁ + F(u₂))`` (ERF recomputes ``f₂`` after the solve).
    * ``dirk2`` — ERF ``dirk212``: stage 1 is the BE solve; stage 2 solves
      ``μu₂ − F(u₂) = μ(u − dt·f₁)``; update ``u + ½dt(f₁ + f₂)``; both
      stages must converge and stay positive.
    """
    tiny = jnp.asarray(1.0e-300, dtype)

    def rhs(u):
        return drsq_dt(u, S, T, e_s, N_s, include_curvature, include_solute)

    def newton(u_init, rhs_const, mu):
        return _newton_solve(u_init, rhs_const, mu, S, T, e_s, N_s,
                             include_curvature, include_solute,
                             newton_rtol, newton_atol, newton_stol,
                             newton_maxits)

    if method == "rk4_adaptive":
        def attempt(u, dt):
            k1 = rhs(u)
            u2 = u + 0.5 * dt * k1
            k2 = rhs(u2)
            u3 = u + 0.5 * dt * k2
            k3 = rhs(u3)
            u4 = u + dt * k3
            k4 = rhs(u4)
            u_new = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return u_new, (u2 > 0.0) & (u3 > 0.0) & (u4 > 0.0)
    elif method == "be":
        def attempt(u, dt):
            mu = 1.0 / jnp.maximum(dt, tiny)
            u_new, conv = newton(u, mu * u, mu)
            return u_new, conv
    elif method == "cn":
        def attempt(u, dt):
            mu = 1.0 / jnp.maximum(0.5 * dt, tiny)
            f1 = rhs(u)
            u2, conv = newton(u, mu * (u + 0.5 * dt * f1), mu)
            f2 = rhs(u2)
            u_new = u + 0.5 * dt * (f1 + f2)
            return u_new, conv & (u2 > 0.0)
    elif method == "dirk2":
        def attempt(u, dt):
            mu = 1.0 / jnp.maximum(dt, tiny)
            u1, conv1 = newton(u, mu * u, mu)
            f1 = rhs(u1)
            u2, conv2 = newton(u1, mu * (u - dt * f1), mu)
            f2 = rhs(u2)
            u_new = u + 0.5 * dt * (f1 + f2)
            return u_new, conv1 & conv2 & (u1 > 0.0) & (u2 > 0.0)
    else:  # pragma: no cover — guarded by integrate_radius dispatch
        raise ValueError(f"unknown adaptive method {method!r}")
    return attempt


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
    increase the sub-step count for strong evaporation). The adaptive family —
    ``"rk4_adaptive"`` (explicit), ``"be"`` (backward Euler + Newton), ``"cn"``
    (Crank-Nicolson + Newton), ``"dirk2"`` (2-stage DIRK + Newton) — shares the
    ERF stiffness-based outer loop (``dt = cfl/|τ|`` from :func:`drsq_dt_jac`,
    step-halving on rejection, too-small unconverged exit leaving the radius
    unchanged, steady-state early exit) — per-droplet ``lax.while_loop``, NOT
    reverse-mode differentiable.
    """
    _ADAPTIVE = ("rk4_adaptive", "be", "cn", "dirk2")
    if cfg.condensation_integrator == "rk4":
        step_fn = _rk4_step
    elif cfg.condensation_integrator == "euler":
        step_fn = _euler_step
    elif cfg.condensation_integrator in _ADAPTIVE:
        step_fn = None
    else:
        raise ValueError(
            f"Unknown SDM condensation_integrator: {cfg.condensation_integrator!r} "
            "(expected 'rk4', 'euler', 'rk4_adaptive', 'be', 'cn', or 'dirk2')"
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

    if cfg.condensation_integrator in _ADAPTIVE:
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
        n_rtol = jnp.asarray(cfg.newton_rtol, dtype=dtype)
        n_atol = jnp.asarray(cfg.newton_atol, dtype=dtype)
        n_stol = jnp.asarray(cfg.newton_stol, dtype=dtype)
        n_maxits = jnp.asarray(int(cfg.newton_maxits), jnp.int32)
        method = cfg.condensation_integrator

        def _one(u0, s, t, es, ns):
            attempt = _make_attempt(
                method, s, t, es, ns, include_curvature, include_solute,
                n_rtol, n_atol, n_stol, n_maxits, dtype)
            return _integrate_adaptive(
                u0, t_final, t, es, ns,
                include_curvature, include_solute, cfl, stol, max_steps,
                attempt)

        r_sq = jax.vmap(_one)(state.radius**2, S_b, T_b, e_s_b, N_s)
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
