"""Liquid-only saturation adjustment over a prescribed uniform total-water PDF.

PARKED in ``_future/`` (ponytail item pdf_condensation, user-approved
2026-10-02): not wired — no production driver, factory or registry imports
this module, and its tests are skipped.  Wire it into production (moving it back) or delete it.

A uniform total-water PDF on [q_t - D, q_t + D] with D = (1 - rh_crit) * q_sat is
algebraically identical to the Sundqvist (1978) cloud scheme: cloud fraction and
condensate follow from closed-form integrals of the PDF above saturation (ECMWF
cloud-cover parametrization notes, eqs. 7-10, uniform PDF == Sundqvist).  The
grid mean can then sit BELOW saturation with cloud present, which the grid-mean
adjustment it replaces cannot.  Elementwise, liquid-only, JAX-differentiable; no
ice, no sedimentation, no transport (the callers own those).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio

__all__ = ["pdf_saturation_adjustment", "uniform_pdf_cloud"]

__physics_contract__ = {
    "summary": (
        "Saturation adjustment over a prescribed uniform total-water PDF of "
        "half-width D = (1 - rh_crit) q_sat: cloud fraction and grid-mean "
        "condensate are the closed-form PDF integrals (== Sundqvist 1978 cover "
        "relation with a condensate budget), so the grid mean can sit below "
        "saturation with cloud present. The signed vapour<->cloud transfer over "
        "one step solves q_c + x = Q(q_t, T + L_v x/c_pd) implicitly (bisection, "
        "implicit-function gradient). Liquid only; callers own ice, sedimentation "
        "and transport."
    ),
    "inputs": {"T": "K", "q_v": "kg/kg", "q_c": "kg/kg", "p": "Pa", "dt": "s",
               "rh_crit": "1"},
    "outputs": {"x": "kg/kg (signed transfer per step; +condensation)",
                "c": "1 (cloud fraction)", "T_new": "K"},
    "sign_convention": (
        "x > 0 condenses vapour into cloud water and warms by L_v x/c_pd; "
        "x < 0 evaporates cloud water and cools. q_v' = q_v - x, q_c' = q_c + x."
    ),
    "conserves": ["moisture", "energy"],   # q_v + q_c and c_pd T + L_v q_v, exactly
    "differentiable": True,
    "reference": (
        "Sundqvist (1978) QJRMS 104, 677-690; ECMWF cloud-cover parametrization "
        "notes (uniform-PDF equivalence, eqs. 7-10)"
    ),
    "idealized_test": "tests/unit/test_pdf_condensation.py (quadrature, persistence, response, Sundqvist identity, conservation, AD)",
}


def uniform_pdf_cloud(q_t, T, p, rh_crit, sat_fn=saturation_mixing_ratio):
    """Sundqvist-equivalent uniform-PDF cloud closure (elementwise).

    With s = sat_fn(T, p), half-width D = min((1 - rh_crit) * s, q_t) (floored)
    and z = q_t - s, returns (c, Q, s): cloud fraction c = clip((z + D)/(2D), 0, 1);
    grid-box condensate Q = 0 for z <= -D, (z + D)**2/(4D) = D c**2 for |z| < D,
    z for z >= D; and the saturation mixing ratio s.
    """
    s = sat_fn(T, p)
    D = jnp.maximum(jnp.minimum((1.0 - rh_crit) * s, q_t), 1e-12)
    z = q_t - s
    c = jnp.clip((z + D) / (2.0 * D), 0.0, 1.0)
    Q = jnp.where(z <= -D, 0.0, jnp.where(z >= D, z, (z + D) ** 2 / (4.0 * D)))
    return c, Q, s


_BISECT_ITERS = 24   # coeff-ok: bisection depth (2^-24 of the bracket), numerics not physics


def pdf_saturation_adjustment(T, q_v, q_c, p, dt, rh_crit, n_iter=_BISECT_ITERS,
                              sat_fn=saturation_mixing_ratio, l_over_cp=None):
    """Signed liquid saturation adjustment over one step (Sundqvist-compatible).

    Holds q_t = q_v + q_c fixed and solves, on [-q_c, q_v] (F is monotone
    increasing), F(x) = q_c + x - Q(q_t, T + l_over_cp*x, p) = 0 for the signed
    transfer x [kg/kg] (x > 0 condensation, x < 0 evaporation) by n_iter
    bisection steps; the root is detached (stop_gradient) and the exact gradient
    re-attached with one implicit-function Newton correction with F_x held
    fixed.  l_over_cp defaults to L_v/c_pd.

    Conservation identities: q_v' = q_v - x, q_c' = q_c + x, T' = T + l_over_cp*x,
    so d(q_v + q_c) = 0 and d(c_pd T + L_v q_v) = 0 exactly.
    Equilibrium identities (0 < c < 1): q_c' = D c**2, q_v' = s - D (1 - c)**2,
    RH = 1 - (1 - rh_crit)(1 - c)**2, i.e. c = 1 - sqrt((1 - RH)/(1 - rh_crit)),
    the Sundqvist cover relation.  Callers use max(x, 0)/dt as the condensation
    rate and max(-x, 0)/dt as the evaporation rate; dt enters only there.
    """
    if l_over_cp is None:
        l_over_cp = constants.L_v / constants.c_pd
    q_t = q_v + q_c

    def residual(xv):
        _, Q, _ = uniform_pdf_cloud(q_t, T + l_over_cp * xv, p, rh_crit, sat_fn)
        return q_c + xv - Q

    lo = -q_c
    hi = q_v
    x = 0.5 * (lo + hi)
    for _ in range(n_iter):
        F = residual(x)
        lo = jnp.where(F <= 0.0, x, lo)
        hi = jnp.where(F <= 0.0, hi, x)
        x = 0.5 * (lo + hi)
    x = jax.lax.stop_gradient(x)
    F_root = residual(x)
    _, dQ_dT = jax.jvp(
        lambda t: uniform_pdf_cloud(q_t, t, p, rh_crit, sat_fn)[1],
        (T + l_over_cp * x,), (jnp.ones_like(T + l_over_cp * x),))
    F_x = jax.lax.stop_gradient(1.0 - dQ_dT * l_over_cp)
    x = x - F_root / F_x
    x = jnp.clip(x, -q_c, q_v)
    c = uniform_pdf_cloud(q_t, T + l_over_cp * x, p, rh_crit, sat_fn)[0]
    return x, c, T + l_over_cp * x
