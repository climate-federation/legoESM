"""Vreman (2004) sub-grid eddy viscosity — shared, grid-agnostic core.

Vreman, A.W. (2004), *An eddy-viscosity subgrid-scale model for turbulent shear
flow: Algebraic theory and applications*, Phys. Fluids 16, 3670.

Given the resolved velocity-gradient tensor ``a_cd = ∂u_c/∂x_d`` (c = velocity
component u,v,w; d = direction x,y,z) at a set of points, returns the eddy
viscosity

    α_ij = ∂u_j/∂x_i = a_ji
    β_ij = Σ_m Δ_m² α_mi α_mj = Σ_m Δ_m² a_im a_jm          (Δ_m = Δx,Δy,Δz)
    Bβ   = β11β22 − β12² + β11β33 − β13² + β22β33 − β23²
    ν_t  = c · √( max(Bβ, 0) / (a_ij a_ij) )  +  ν_floor

(the c·√(…) term is the raw Vreman closed form; ``ν_floor`` is an optional
legoESM additive background viscosity, default 0.)

with ``c ≈ 2.5 C_s²`` (≈0.07 for C_s=0.17). ν_t ≥ ν_floor ≥ 0 by construction
(for c ≥ 0). Key properties FOR c>0 AND filter widths Δ_m>0: ν_t = ν_floor
(→ 0 only when ν_floor=0) exactly when the resolved velocity-gradient tensor has
rank ≤ 1 (Bβ=0 ⟺ rank a ≤ 1) — i.e. rest (rank 0) or all gradients aligned with
a single spatial direction (rank 1), e.g. a laminar 1-D shear ∂u/∂z; a genuine
3-D strain OR a merely rank-2 "two-dimensional" flow such as diag(∂u/∂x,∂v/∂y,0)
gives ν_t > ν_floor, so this is a rank ≤ 1 property, NOT a generic 2-D one. The
PER-DIRECTION filter widths Δx,Δy,Δz make it well-behaved on ANISOTROPIC grids
(Δx≠Δz).
This makes it a robust drop-in alternative to |S|-Smagorinsky for both
the incompressible spectral LES core and the compressible plane CRM (where it is
an OPTIONAL, non-default closure — the SAM-faithful default stays Smagorinsky).

This module holds ONLY the algebra (no grid/staggering): each caller supplies the
nine gradients evaluated at the location where ν_t is needed (cell centres), so
the spectral and finite-volume cores share one definition.
"""
from __future__ import annotations

import jax.numpy as jnp

__physics_contract__ = {
    "summary": (
        "Vreman (2004) algebraic sub-grid eddy viscosity from the nine "
        "resolved velocity gradients: nu_t = c sqrt(max(Bbeta,0)/(a_ij a_ij)) "
        "+ nu_floor, with per-direction filter widths for anisotropic grids."
    ),
    "inputs": {
        "a11": "1/s", "a12": "1/s", "a13": "1/s",
        "a21": "1/s", "a22": "1/s", "a23": "1/s",
        "a31": "1/s", "a32": "1/s", "a33": "1/s",
        "dx": "m", "dy": "m", "dz": "m",
        "c_vreman": "1", "nu_floor": "m^2/s",
    },
    "outputs": {"nu_t": "m^2/s"},
    "sign_convention": (
        "Diagnostic eddy viscosity only (no tendency). nu_t >= nu_floor >= 0 "
        "by construction (for c_vreman >= 0). For c_vreman > 0 and positive "
        "filter widths: nu_t = nu_floor (0 only if nu_floor=0) exactly when the "
        "resolved velocity-gradient tensor has rank <= 1 (Bbeta=0: rest, or all "
        "gradients aligned with one spatial direction, e.g. a laminar 1-D "
        "shear), and a rank-2 (e.g. diag(du/dx,dv/dy,0)) or 3-D strain gives "
        "nu_t > nu_floor."
    ),
    "conserves": ["none"],
    # AD-safe (finite grad, no NaN) everywhere; at the measure-zero rank-1 kink
    # (Bbeta=0) the double-where returns a valid SUBGRADIENT (0), not the
    # classical two-sided derivative (which does not exist there).
    "differentiable": True,
    "reference": "Vreman (2004), Phys. Fluids 16, 3670, doi:10.1063/1.1785131",
    "idealized_test": (
        "tests/unit/test_vreman_sgs_plane.py: a pure 1-D shear (single "
        "non-zero gradient direction) gives nu_t = nu_floor; nu_t >= nu_floor "
        "always; the gradient stays finite at rest (arg=0)."
    ),
}

_EPS = 1.0e-30


def vreman_nu_t(a11, a12, a13, a21, a22, a23, a31, a32, a33,
                dx, dy, dz, c_vreman, nu_floor=0.0):
    """Vreman eddy viscosity from the nine resolved velocity gradients.

    Parameters
    ----------
    a11..a33 : arrays
        ``a_cd = ∂u_c/∂x_d`` at the target points (all same shape).
    dx, dy, dz : float or array
        Per-direction filter widths. ``dz`` may be a scalar or broadcastable to
        the gradient shape (e.g. a per-level ``(nz,)`` on a stretched grid).
    c_vreman : float
        Model constant (≈ 2.5 C_s²; Vreman 2004 ≈ 0.07).
    nu_floor : float, default 0.0
        Optional additive background viscosity floor [m²/s].

    Returns
    -------
    nu_t : array
        Eddy viscosity ν_t ≥ ``nu_floor`` (same shape as the gradients).
    """
    d1, d2, d3 = dx ** 2, dy ** 2, dz ** 2
    # β_ij = Σ_m Δ_m² a_im a_jm   (m = direction 1,2,3)
    b11 = d1 * a11 * a11 + d2 * a12 * a12 + d3 * a13 * a13
    b22 = d1 * a21 * a21 + d2 * a22 * a22 + d3 * a23 * a23
    b33 = d1 * a31 * a31 + d2 * a32 * a32 + d3 * a33 * a33
    b12 = d1 * a11 * a21 + d2 * a12 * a22 + d3 * a13 * a23
    b13 = d1 * a11 * a31 + d2 * a12 * a32 + d3 * a13 * a33
    b23 = d1 * a21 * a31 + d2 * a22 * a32 + d3 * a23 * a33
    Bbeta = (b11 * b22 - b12 ** 2 + b11 * b33 - b13 ** 2 + b22 * b33 - b23 ** 2)
    aa = (a11 ** 2 + a12 ** 2 + a13 ** 2 + a21 ** 2 + a22 ** 2 + a23 ** 2
          + a31 ** 2 + a32 ** 2 + a33 ** 2)
    arg = jnp.maximum(Bbeta, 0.0) / (aa + _EPS)
    # AD-safe sqrt: d/dx √x = ∞ at x=0, so √(arg) NaNs the gradient wherever
    # rank a ≤ 1 (arg=0: rest or a 1-D shear). The double-where keeps value AND gradient finite
    # there; the returned grad at the arg=0 KINK is a valid SUBGRADIENT (0), not
    # the classical two-sided derivative (√(Bβ/aa) has ±c-type one-sided slopes
    # at Bβ=0). Same guard as `safe_sqrt_strain` in the Smagorinsky path.
    safe = jnp.where(arg > 0.0, arg, 1.0)
    root = jnp.where(arg > 0.0, jnp.sqrt(safe), 0.0)
    return c_vreman * root + nu_floor
