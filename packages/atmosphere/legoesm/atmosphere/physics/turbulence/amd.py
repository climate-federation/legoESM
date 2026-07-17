"""Anisotropic Minimum-Dissipation (AMD) sub-grid eddy viscosity — shared core.

Rozema, W., Bae, H. J., Moin, P. & Verstappen, R. (2015), *Minimum-dissipation
models for large-eddy simulation*, Phys. Fluids 27, 085107; anisotropic
(scaled-gradient) form of Abkar, Bae & Moin (2016), Phys. Rev. Fluids 1, 041701.

Given the resolved velocity-gradient tensor ``a_cd = ∂u_c/∂x_d`` (c = velocity
component u,v,w; d = direction x,y,z) and the per-direction filter widths
Δx, Δy, Δz, the AMD eddy viscosity is

    S_ij = ½(a_ij + a_ji)                                (resolved strain rate)
    N    = − Σ_k Δ_k² (∂_k u_i)(∂_k u_j) S_ij            (= − Σ_k Δ_k² g_kᵀ S g_k)
    ν_t  = C · max(N, 0) / (a_cd a_cd)                   (a_cd a_cd = Σ_cd a_cd²)

where ``g_k = (a_1k, a_2k, a_3k)`` is the k-th column of the gradient tensor
(the derivative of the velocity VECTOR in direction k).  Key properties:

* **Minimum dissipation** (all inequalities below assume ``c_amd >= 0``; a
  negative ``c_amd`` — not validated here — flips every sign). ``max(·,0)`` makes
  ν_t drop to ``ν_floor`` wherever the resolved flow needs no sub-grid dissipation
  (``N <= 0``) — in particular a pure off-diagonal 1-D shear (e.g. a well-resolved
  ``∂u/∂z``) gives ν_t = ν_floor exactly (→ 0 when nu_floor=0) — and the
  eddy-viscosity term is never negative (no spurious backscatter), so ν_t ≥ 0
  provided ``nu_floor >= 0`` too.  (A pure NORMAL strain draws ν_t > ν_floor or
  = ν_floor depending on its sign — a COMPRESSIVE ∂u/∂x < 0 gives N > 0 ⇒
  ν_t > ν_floor — the intended minimum-dissipation behaviour, NOT a laminar
  cutoff of every single-gradient field, and NOT Vreman's rank-1 property.)
* **Anisotropy.** the per-direction Δ_k² weights make it well-behaved on
  Δx ≠ Δz grids, like Vreman.
* **Purely local** (no plane average) ⇒ MPI-safe algebra.

Like :mod:`legoesm.atmosphere.physics.turbulence.vreman`, this module holds ONLY
the algebra (no grid/staggering): the caller supplies the nine gradients at the
target points (cell centres), so the spectral and finite-volume cores share one
definition.  AD-safe by construction — ``max(·,0)`` and the floored denominator
avoid the ``sqrt(0)`` gradient trap the Vreman/Smagorinsky paths must guard; the
grad at the ``N = 0`` kink is a finite SUBGRADIENT (JAX's ``maximum`` balanced
tie: the AVERAGE of the two one-sided derivatives, i.e. ½·the N>0-side slope),
not the classical two-sided derivative.

Faithfulness: the closed form (S_ij, the −Σ_k Δ_k² g_kᵀ S g_k numerator with
COLUMN g_k, the unscaled a_cd a_cd denominator, max(·,0), + ν_floor) is pinned to
round-off against an independent Rozema-2015 reimplementation in
``tests/unit/test_amd_faithful.py`` (structure canaries: the − sign, Δ_k²
numerator-only weighting, column-not-row index, and the ``max(N, 0)`` clip).
"""
from __future__ import annotations

import jax.numpy as jnp

__physics_contract__ = {
    "summary": (
        "Anisotropic Minimum-Dissipation (Rozema 2015 / Abkar-Bae-Moin 2016) "
        "algebraic sub-grid eddy viscosity from the nine resolved velocity "
        "gradients: nu_t = C max(N,0)/(a_ij a_ij), N = -sum_k dk^2 (d_k u_i)"
        "(d_k u_j) S_ij, with per-direction filter widths."
    ),
    "inputs": {
        "a11": "1/s", "a12": "1/s", "a13": "1/s",
        "a21": "1/s", "a22": "1/s", "a23": "1/s",
        "a31": "1/s", "a32": "1/s", "a33": "1/s",
        "dx": "m", "dy": "m", "dz": "m",
        "c_amd": "1", "nu_floor": "m^2/s",
    },
    "outputs": {"nu_t": "m^2/s"},
    "sign_convention": (
        "Diagnostic eddy viscosity only (no tendency). For c_amd >= 0: nu_t >= "
        "nu_floor by construction (max(N,0), the minimum-dissipation property; "
        "the eddy-viscosity term is never negative, so nu_t >= 0 when nu_floor "
        ">= 0 too), and = nu_floor wherever N <= 0 (an off-diagonal 1-D shear, or "
        "an EXPANDING normal strain); a COMPRESSIVE normal strain gives N>0 => "
        "nu_t > nu_floor."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Rozema, Bae, Moin & Verstappen (2015), Phys. Fluids 27, 085107; "
        "Abkar, Bae & Moin (2016), Phys. Rev. Fluids 1, 041701"
    ),
    "idealized_test": (
        "tests/unit/test_amd_sgs.py + test_amd_faithful.py (all inequalities for "
        "c_amd >= 0): a pure off-diagonal 1-D shear gives nu_t = nu_floor; nu_t "
        ">= nu_floor (>= 0 when nu_floor >= 0, never negative); a 3-D field gives "
        "nu_t > nu_floor; differentiable at rest (arg=0)."
    ),
}

_EPS = 1.0e-30


def amd_nu_t(a11, a12, a13, a21, a22, a23, a31, a32, a33,
             dx, dy, dz, c_amd, nu_floor=0.0):
    """Anisotropic Minimum-Dissipation eddy viscosity from the nine gradients.

    Parameters
    ----------
    a11..a33 : arrays
        ``a_cd = ∂u_c/∂x_d`` at the target points (all same shape).
    dx, dy, dz : float or array
        Per-direction filter widths. ``dz`` may be scalar or broadcastable to
        the gradient shape (e.g. a per-level ``(nz,)`` on a stretched grid).
    c_amd : float
        Model constant (modified Poincaré constant; ≈ 0.3 for the anisotropic
        scaled-gradient form). Assumed ``c_amd >= 0`` (not validated here).
    nu_floor : float, default 0.0
        Optional additive background viscosity floor [m²/s].

    Returns
    -------
    nu_t : array
        Eddy viscosity, same shape as the gradients. ν_t ≥ ``nu_floor`` for
        ``c_amd`` ≥ 0 (and ≥ 0 when ``nu_floor`` ≥ 0 too); neither is validated.
    """
    d1, d2, d3 = dx ** 2, dy ** 2, dz ** 2

    # Resolved strain rate S_ij = ½(a_ij + a_ji).
    s11, s22, s33 = a11, a22, a33
    s12 = 0.5 * (a12 + a21)
    s13 = 0.5 * (a13 + a31)
    s23 = 0.5 * (a23 + a32)

    def _quad(p, q, r):
        # g^T S g for g = (p, q, r) (a gradient-tensor column = ∂_k of (u,v,w)).
        return (s11 * p * p + s22 * q * q + s33 * r * r
                + 2.0 * s12 * p * q + 2.0 * s13 * p * r + 2.0 * s23 * q * r)

    # Column k = derivative of (u, v, w) in direction k: k=1 -> (a11,a21,a31).
    quad1 = _quad(a11, a21, a31)
    quad2 = _quad(a12, a22, a32)
    quad3 = _quad(a13, a23, a33)

    numerator = -(d1 * quad1 + d2 * quad2 + d3 * quad3)
    aa = (a11 ** 2 + a12 ** 2 + a13 ** 2 + a21 ** 2 + a22 ** 2 + a23 ** 2
          + a31 ** 2 + a32 ** 2 + a33 ** 2)
    # max(N, 0): the minimum-dissipation property — no SGS dissipation where the
    # resolved flow does not need it; the eddy-viscosity term is >= 0 for
    # c_amd >= 0. AD-safe (no sqrt); the floored denominator keeps the gradient
    # finite at rest (aa=0).
    nu_t = c_amd * jnp.maximum(numerator, 0.0) / (aa + _EPS)
    return nu_t + nu_floor
