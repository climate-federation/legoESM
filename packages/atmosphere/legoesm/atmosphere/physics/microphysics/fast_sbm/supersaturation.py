"""Analytic supersaturation integration (oracle ``JERSUPSAT_KS``, warm path).

Within a condensation substep the supersaturation obeys the linear ODE

    dS/dt = −R·S + F

with relaxation rate ``R`` (oracle ``RW``) built from the spectrum's
relaxation integral and the psychrometric factor, and dynamical forcing
``F`` (oracle ``DYN1``). The oracle integrates this exactly:

    S(dt)  = S₀ e^{−R dt} + (F/R)(1 − e^{−R dt})
    ∫S dt  = S₀ (1−e^{−R dt})/R + F (dt − (1−e^{−R dt})/R)/R

(``DEL1N``/``DEL1INT``; the ∫S dt drives the bin mass growth
``Δm_k = B_k ∫S dt`` in ``JERDFUN``). The oracle switches to a 5-term
Taylor ``EXPM1`` statement function for |R dt| ≤ 1e-6 purely because
Fortran's ``exp`` loses precision there; ``jnp.expm1`` IS that limit to
machine precision, so the port uses one smooth ``expm1`` formulation for
all magnitudes — analytically identical to both oracle branches and
differentiable without a discontinuity.

Relaxation rate (oracle ONECOND1: ``RW = (OPER2(QPS) + B5L·AL1)·DOPL·SFNL``):

    R = [ ∂ln e/∂q_v + (L_v/(R_v T²))·(L_v/c_pd) ] · (1+S) · SFN
        with  ∂ln e/∂q_v = ε / ((ε + (1−ε) q_v) q_v)   (OPER2)

The oracle hardcodes ``BB1_MY = 5.42e3 K ≈ L_v/R_v`` and
``AL1 = 2500 K ≈ L_v/c_pd`` (CGS-era roundings, ≤0.5% off); the port
derives both from ``legoesm.constants`` — faithful to the physics rather
than to the roundings (documented deviation).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants

__physics_contract__ = {
    "summary": (
        "Exact one-substep integration of the linear supersaturation ODE "
        "dS/dt = -R S + F (oracle JERSUPSAT_KS warm branch) plus the "
        "psychrometric relaxation rate R from the spectrum integral "
        "(oracle ONECOND1 RW)."
    ),
    "inputs": {
        "S": "1 (supersaturation e/e_s - 1)",
        "T": "K",
        "q_v": "kg/kg",
        "sfn": "s^-1 (relaxation integral from JERTIMESC)",
        "forcing": "s^-1 (dynamical supersaturation source, oracle DYN1)",
        "dt": "s",
    },
    "outputs": {
        "S_new": "1",
        "S_int": "s (time-integrated supersaturation over the substep)",
    },
    "sign_convention": (
        "R >= 0 for any physical spectrum; with F=0, |S| decays "
        "monotonically toward 0 and S_int has the sign of S0; with R=0 the "
        "ballistic limit S_new = S0 + F dt, S_int = S0 dt + F dt^2/2 holds."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Khain & Sednev (1996); Khain et al. (2004) JAS 61:2963; WRF "
        "module_mp_fast_sbm.F JERSUPSAT_KS / ONECOND1"
    ),
    "idealized_test": (
        "Matches the oracle's closed-form DEL1N/DEL1INT expressions "
        "verbatim for |R dt| > 1e-6 and its Taylor branch for small R dt; "
        "equilibrium S(t->inf) = F/R; R reproduces the oracle's "
        "(OPER2 + B5L*AL1)*(1+S)*SFN with derived L_v/R_v and L_v/c_pd "
        "within the documented 0.5% rounding difference."
    ),
}


class SupersatStep(NamedTuple):
    """Result of one analytic supersaturation substep."""

    S_new: jax.Array   # supersaturation at the end of the substep [-]
    S_int: jax.Array   # ∫ S dt over the substep [s]


def supersat_relaxation_rate(
    T: jax.Array,
    q_v: jax.Array,
    S: jax.Array,
    sfn: jax.Array,
) -> jax.Array:
    """Relaxation rate ``R`` [s⁻¹] of the warm supersaturation ODE
    (oracle ``RW = (OPER2(QPS) + B5L·AL1)·DOPL·SFNL``).

    ``q_v`` is floored at a tiny positive value in the OPER2 term: at a
    bone-dry level (``q_v=0``) the ``ε/((…)q)`` vapor-pressure derivative is
    a 1/0 singularity, and although ``sfn`` is also 0 there (no droplets),
    the product ``∞·0`` is NaN in both the forward pass and reverse-mode AD.
    The floor keeps the psychrometric factor finite so the empty-spectrum
    cell correctly yields ``R = finite·0 = 0`` (surfaced by the end-to-end
    hydrostatic grad test over a dry upper atmosphere)."""
    eps = constants.epsilon
    q_safe = jnp.maximum(q_v, 1.0e-12)
    dlne_dq = eps / ((eps + (1.0 - eps) * q_safe) * q_safe)
    dlnes_dT = constants.L_v / (constants.R_v * T * T)
    psychro = dlne_dq + dlnes_dT * (constants.L_v / constants.c_pd)
    return psychro * (1.0 + S) * sfn


def integrate_supersaturation(
    S: jax.Array,
    relax_rate: jax.Array,
    forcing: jax.Array,
    dt: jax.Array | float,
) -> SupersatStep:
    """Exact solution of ``dS/dt = −R S + F`` over ``dt`` (oracle
    ``JERSUPSAT_KS`` water-only branch, both magnitude regimes).

    Uses ``em = −expm1(−R dt) = 1 − e^{−R dt}`` so the R→0 limit is exact:
    ``S_new → S + F dt``, ``S_int → S dt + F dt²/2`` (the oracle's
    IRW==0 ballistic branch), with no branching and smooth gradients.
    """
    dt = jnp.asarray(dt, dtype=jnp.result_type(S))
    R = relax_rate
    x = R * dt
    em = -jnp.expm1(-x)                      # 1 - exp(-R dt), exact small-x
    # Guard the R→0 division with SERIES (not constants) so gradients in R
    # stay correct through the switch (codex review: a constant-limit
    # branch zeroes d/dR near R=0):
    #   em/R           = dt·φ(x),  φ = (1−e^{−x})/x = 1 − x/2 + x²/6 − …
    #   (dt − em/R)/R  = dt²·ψ(x), ψ = (1−φ)/x     = 1/2 − x/6 + x²/24 − …
    # Truncation O(x³) < 1e-12 inside the |x| < 1e-4 window.
    small = jnp.abs(x) < 1.0e-4
    R_safe = jnp.where(small, 1.0, R)
    em_over_R = jnp.where(
        small, dt * (1.0 - x / 2.0 + x * x / 6.0), em / R_safe)
    tail = jnp.where(
        small, dt * dt * (0.5 - x / 6.0 + x * x / 24.0),
        (dt - em / R_safe) / R_safe)
    S_new = S * (1.0 - em) + forcing * em_over_R
    S_int = S * em_over_R + forcing * tail
    return SupersatStep(S_new=S_new, S_int=S_int)
