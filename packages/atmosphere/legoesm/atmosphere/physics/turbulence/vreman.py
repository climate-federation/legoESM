"""Vreman (2004) sub-grid eddy viscosity — shared, grid-agnostic core.

Vreman, A.W. (2004), *An eddy-viscosity subgrid-scale model for turbulent shear
flow: Algebraic theory and applications*, Phys. Fluids 16, 3670.

Given the resolved velocity-gradient tensor ``a_cd = ∂u_c/∂x_d`` (c = velocity
component u,v,w; d = direction x,y,z) at a set of points, returns the eddy
viscosity

    α_ij = ∂u_j/∂x_i = a_ji
    β_ij = Σ_m Δ_m² α_mi α_mj = Σ_m Δ_m² a_im a_jm          (Δ_m = Δx,Δy,Δz)
    Bβ   = β11β22 − β12² + β11β33 − β13² + β22β33 − β23²
    ν_t  = 0                        if Bβ < 1e-12   (Vreman's own zero-guard)
         = c · √( Bβ / (a_ij a_ij) ) otherwise
    ν_t += ν_floor

(the c·√(…) term and the ``Bβ < 1e-12`` zero-guard are the author's reference
Fortran, http://www.vremanresearch.nl/Vreman_Subgridmodel_Fortran.txt —
``if (bbeta.lt.1d-12) eddyv=0``; the threshold is DIMENSIONAL (Bβ in m⁴/s⁴) and
is ported unchanged.  The author's comment
``in case of single precision (real*4): "if (bbeta.lt.1d-6) then"`` is NOT
ported: the guard is 1e-12 at every precision (open question).  ``ν_floor``
is an optional legoESM additive background viscosity, default 0, with no
counterpart in the oracle.)

RULE 5 smooth default (user decision 2026-10-28): the zero-guard is ported as
a C1 smoothstep ``s(t) = t²(3 − 2t)``, ``t = clip((Bβ − (1e-12 − w))/(2w), 0, 1)``
multiplying ``c·√(Bβ/aa)``, with half-width ``w = bbeta_guard_width`` (default
1e-13 m⁴/s⁴).  ``bbeta_guard_width = 0`` recovers the author's exact switch,
which matches the running Fortran to <= 1e-10 (tests/atmosphere/test_vreman_oracle.py).
KNOWN DIFFERENCE at the default: points with 0.9e-12 < Bβ < 1.1e-12 move; on
the oracle fixture that is 36 of 6030 points, by at most 4.0e-3 m²/s (signed
per-point gaps stored in ``tests/atmosphere/oracle_fixtures/vreman_default_gap.npz``).

with ``c ≈ 2.5 C_s²`` (≈0.07 for C_s=0.17). ν_t ≥ ν_floor ≥ 0 by construction
(for c ≥ 0). Key properties FOR c>0 AND filter widths Δ_m>0: ν_t = ν_floor
(→ 0 only when ν_floor=0) when the resolved velocity-gradient tensor has
rank ≤ 1 (Bβ=0 ⟺ rank a ≤ 1) — i.e. rest (rank 0) or all gradients aligned with
a single spatial direction (rank 1), e.g. a laminar 1-D shear ∂u/∂z — and also
at quiescent points with 0 < Bβ < 1e-12 (the author's guard); a genuine
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
        "resolved velocity gradients: nu_t = 0 if Bbeta < 1e-12 else "
        "c sqrt(Bbeta/(a_ij a_ij)), plus nu_floor, with per-direction filter "
        "widths for anisotropic grids (Vreman's reference Fortran); by default "
        "the Bbeta < 1e-12 guard is a C1 smoothstep of half-width "
        "bbeta_guard_width = 1e-13 m^4/s^4 (0 = the author's hard switch)."
    ),
    "inputs": {
        "a11": "1/s", "a12": "1/s", "a13": "1/s",
        "a21": "1/s", "a22": "1/s", "a23": "1/s",
        "a31": "1/s", "a32": "1/s", "a33": "1/s",
        "dx": "m", "dy": "m", "dz": "m",
        "c_vreman": "1", "nu_floor": "m^2/s", "bbeta_guard_width": "m^4/s^4",
    },
    "outputs": {"nu_t": "m^2/s"},
    "sign_convention": (
        "Diagnostic eddy viscosity only (no tendency). nu_t >= nu_floor >= 0 "
        "by construction (for c_vreman >= 0). For c_vreman > 0 and positive "
        "filter widths: nu_t = nu_floor (0 only if nu_floor=0) when the "
        "resolved velocity-gradient tensor has rank <= 1 (Bbeta=0: rest, or all "
        "gradients aligned with one spatial direction, e.g. a laminar 1-D "
        "shear) or when Bbeta < 1e-12 - w m^4/s^4 (the author's zero-guard, "
        "smoothstep over 1e-12 +/- w at the default w = 1e-13); a "
        "rank-2 (e.g. diag(du/dx,dv/dy,0)) or 3-D strain with Bbeta >= 1e-12 "
        "gives nu_t > nu_floor."
    ),
    "conserves": ["none"],
    # AD-safe at rest and at rank <= 1 / quiescent points (double-where below
    # 1e-12 - w).  At the default the guard is a C1 smoothstep, so the value
    # and gradient are continuous across it; at bbeta_guard_width = 0 it is the
    # author's hard switch (value jumps from 0 to c*sqrt(1e-12/aa)).  NOT finite
    # for unphysical scale combinations where
    # aa**2 underflows while Bbeta >= 1e-12 (e.g. fp32 |a| ~ 1e-11 1/s with
    # widths ~ 1e8 m): the reverse divide then gives inf/NaN.
    "differentiable": True,
    "reference": "Vreman (2004), Phys. Fluids 16, 3670, doi:10.1063/1.1785131",
    "idealized_test": (
        "tests/atmosphere/test_vreman_oracle.py (the author's Fortran, run); "
        "tests/unit/test_vreman_sgs_plane.py: a pure 1-D shear (single "
        "non-zero gradient direction) gives nu_t = nu_floor; nu_t >= nu_floor "
        "always; the gradient stays finite at rest (arg=0)."
    ),
}

# Vreman's reference Fortran zero-guard: ``if (bbeta.lt.1d-12) eddyv=0``
# (http://www.vremanresearch.nl/Vreman_Subgridmodel_Fortran.txt). Dimensional
# (Bbeta in m^4/s^4); ported unchanged.
_BBETA_ZERO_GUARD = 1.0e-12
# RULE 5 smooth default (user decision 2026-10-28): half-width of the C1
# smoothstep across the guard [m^4/s^4]; 0.0 = the author's hard switch.
VREMAN_BBETA_GUARD_WIDTH = 1.0e-13


def vreman_nu_t(a11, a12, a13, a21, a22, a23, a31, a32, a33,
                dx, dy, dz, c_vreman, nu_floor=0.0,
                bbeta_guard_width=VREMAN_BBETA_GUARD_WIDTH):
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
    bbeta_guard_width : float, default ``VREMAN_BBETA_GUARD_WIDTH`` (1e-13)
        Half-width w [m⁴/s⁴] of the C1 smoothstep across the author's
        ``Bβ < 1e-12`` zero-guard (static Python float).  ``0.0`` is the
        author's exact hard switch (the oracle).  Widths ``>= 1e-12`` (the
        band would reach rest, giving NaN) or negative raise ``ValueError``.

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
    if bbeta_guard_width == 0.0:
        # Hard limit = author's reference form: eddyv = 0 if bbeta < 1e-12,
        # else coef*sqrt(bbeta/abeta).  Double-where: the inner where feeds a
        # safe 1.0 on the guarded side so neither the divide (aa may be 0
        # there) nor sqrt' (infinite at 0) is ever differentiated; the guarded
        # side returns 0 with gradient 0.  Bbeta >= 1e-12 implies aa > 0.
        small = Bbeta < _BBETA_ZERO_GUARD
        arg = jnp.where(small, 1.0, Bbeta / jnp.where(small, 1.0, aa))
        root = jnp.where(small, 0.0, jnp.sqrt(arg))
        return c_vreman * root + nu_floor
    if not 0.0 < bbeta_guard_width < _BBETA_ZERO_GUARD:
        # At w >= 1e-12 the band reaches Bbeta = 0 (rest), where aa = 0 and
        # the divide/sqrt give NaN value and gradient (user decision N7).
        raise ValueError(
            f"bbeta_guard_width must be 0 or in (0, {_BBETA_ZERO_GUARD!r}), "
            f"got {bbeta_guard_width!r}")
    # Smooth default (RULE 5): multiply by a C1 smoothstep s(t) = t^2 (3 - 2t),
    # t = clip((Bbeta - (g - w)) / (2 w), 0, 1), across the guard g = 1e-12
    # +/- w.  Below g - w s = 0 exactly and the same double-where keeps the
    # divide and sqrt off the tape; inside the band Bbeta >= g - w > 0, so
    # aa > 0 and sqrt is differentiable.
    lo = _BBETA_ZERO_GUARD - bbeta_guard_width
    off = Bbeta < lo
    t = jnp.clip((Bbeta - lo) / (2.0 * bbeta_guard_width), 0.0, 1.0)
    ramp = t * t * (3.0 - 2.0 * t)
    arg = jnp.where(off, 1.0, Bbeta / jnp.where(off, 1.0, aa))
    root = jnp.where(off, 0.0, jnp.sqrt(arg))
    return c_vreman * root * ramp + nu_floor
