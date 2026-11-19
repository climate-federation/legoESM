"""Anisotropic Minimum-Dissipation (AMD) sub-grid eddy viscosity — shared core.

Rozema, W., Bae, H. J., Moin, P. & Verstappen, R. (2015), *Minimum-dissipation
models for large-eddy simulation*, Phys. Fluids 27, 085107; anisotropic
(scaled-gradient) form of Abkar, Bae & Moin (2016), Phys. Rev. Fluids 1, 041701.

Given the resolved velocity-gradient tensor ``a_cd = ∂u_c/∂x_d`` (c = velocity
component u,v,w; d = direction x,y,z) and the per-direction filter widths
Δx, Δy, Δz, the AMD eddy viscosity is

    S_ij = ½(a_ij + a_ji)                                (resolved strain rate)
    N    = − Σ_k Δ_k² (∂_k u_i)(∂_k u_j) S_ij            (= − Σ_k Δ_k² g_kᵀ S g_k)
    ν_t  = max(10·eps, C · N / max(10·eps, a_cd a_cd))   (a_cd a_cd = Σ_cd a_cd²)

with ``eps`` the fp64 machine epsilon (AMR-Wind's floors, so the "zero" of the
minimum-dissipation clip is 2.2e-15 m²/s, not 0).

RULE 5 smooth default (user decision 2026-10-28): the OUTER floor is
``10·eps + w·softplus((x − 10·eps)/w)`` with ``x = C·N/max(10·eps, a_cd a_cd)``
and ``w = floor_width`` (default 1e-8 m²/s); ``floor_width = 0`` is AMR-Wind's
hard ``max`` and matches the running oracle (tests/atmosphere/test_amd_oracle.py).
KNOWN DIFFERENCES at the default output width (this width alone): wherever
``N`` is near 0 the viscosity sits up to ``w·ln2 = 6.9e-9`` m²/s above the
oracle's floor (exactly that at rest or in a pure off-diagonal 1-D shear);
points with x well below the floor and active points are unchanged to
round-off.  On the oracle fixture 3 of 6030 points move by more than
1e-12 m²/s, at most 6.9e-9 (re-measured 2026-10-31).  The signed per-point gaps
at BOTH default widths are in
``tests/atmosphere/oracle_fixtures/amd_default_gap.npz``.  The DENOMINATOR
floor ``max(10·eps, a_cd a_cd)`` is smoothed the same way (user decision
2026-10-31): ``10·eps + w_d·softplus((a_cd a_cd − 10·eps)/w_d)`` with
``w_d = denom_floor_width`` (default 10·eps = 2.22e-15 s⁻²); ``0`` = the hard
``max``.  It moves only points whose ``a_cd a_cd`` is within a few ``w_d`` of
10·eps (at rest the denominator is 10·eps·(1 + ln(1 + e⁻¹)) = 1.31·10·eps).
KNOWN DIFFERENCE (measured 2026-10-31 on the oracle fixture, fp64, this width
alone): 10 of 6030 points move by more than 1e-12 m²/s, at most 3.1e-7 m²/s
(1.5e-3 relative).  With
both defaults 13 points move, at most 3.1e-7 m²/s (the stored file above).

where ``g_k = (a_1k, a_2k, a_3k)`` is the k-th column of the gradient tensor
(the derivative of the velocity VECTOR in direction k).  Key properties:

* **Minimum dissipation** (all inequalities below assume ``c_amd >= 0``; a
  negative ``c_amd`` — not validated here — flips every sign; equalities to the
  floor below hold at ``floor_width = 0``, see the RULE 5 note above). The outer floor makes
  ν_t drop to ``ν_floor + 10·eps`` wherever the resolved flow needs no sub-grid dissipation
  (``N <= 0``) — in particular a pure off-diagonal 1-D shear (e.g. a well-resolved
  ``∂u/∂z``) gives ν_t = ν_floor + 10·eps — and the
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
definition.  AD-safe by construction — the two floors (softplus by default,
``jnp.maximum`` at width 0) and the absence of any sqrt avoid the ``sqrt(0)``
gradient trap the Vreman/Smagorinsky paths must guard; at width 0 the gradient
on the floored side is 0.

Faithfulness: a line-by-line port of AMR-Wind v4.2.0
(0ee09c22b9afeca06231e4910276c0ba05fb52dd) ``src/turbulence/LES/AMD.H::amd_muvel``
with ``beta = 0`` — same loop and accumulation order, same ``10·eps`` floors (at both widths 0) on the
denominator and on ν_t — matched to the RUNNING oracle in
``tests/atmosphere/test_amd_oracle.py``.  Not ported: AMR-Wind's Abkar–Moin
buoyancy numerator (legoESM's cited references are shear-only); ``nu_floor`` is a
legoESM additive extension (0 = oracle).  The closed form is also pinned against
an independent Rozema-2015 reimplementation in ``tests/unit/test_amd_faithful.py``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

__physics_contract__ = {
    "summary": (
        "Anisotropic Minimum-Dissipation (Rozema 2015 / Abkar-Bae-Moin 2016) "
        "algebraic sub-grid eddy viscosity from the nine resolved velocity "
        "gradients: nu_t = max(10 eps, C N/max(10 eps, a_ij a_ij)), N = -sum_k "
        "dk^2 (d_k u_i)(d_k u_j) S_ij, with per-direction filter widths "
        "(AMR-Wind v4.2.0 amd_muvel, pure shear)."
    ),
    "inputs": {
        "a11": "1/s", "a12": "1/s", "a13": "1/s",
        "a21": "1/s", "a22": "1/s", "a23": "1/s",
        "a31": "1/s", "a32": "1/s", "a33": "1/s",
        "dx": "m", "dy": "m", "dz": "m",
        "c_amd": "1", "nu_floor": "m^2/s", "floor_width": "m^2/s",
        "denom_floor_width": "1/s^2",
    },
    "outputs": {"nu_t": "m^2/s"},
    "sign_convention": (
        "Diagnostic eddy viscosity only (no tendency). For c_amd >= 0: nu_t >= "
        "nu_floor + 10 eps by construction (the minimum-dissipation floor; the "
        "eddy-viscosity term is never negative). At floor_width = 0 nu_t = "
        "nu_floor + 10 eps wherever N <= 0 (an off-diagonal 1-D shear, or an "
        "EXPANDING normal strain); at the default softplus floor (w = 1e-8 m^2/s) "
        "it is at most nu_floor + 10 eps + w ln2 there; a "
        "COMPRESSIVE normal strain gives N>0 => nu_t > nu_floor."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Rozema, Bae, Moin & Verstappen (2015), Phys. Fluids 27, 085107; "
        "Abkar, Bae & Moin (2016), Phys. Rev. Fluids 1, 041701; "
        "AMR-Wind v4.2.0 src/turbulence/LES/AMD.H amd_muvel (oracle)"
    ),
    "idealized_test": (
        "tests/atmosphere/test_amd_oracle.py (AMR-Wind amd_muvel, run); "
        "tests/unit/test_amd_sgs.py + test_amd_faithful.py (all inequalities for "
        "c_amd >= 0): a pure off-diagonal 1-D shear gives nu_t = nu_floor + "
        "10 eps at floor_width = 0 (+ w ln2 at the default); nu_t >= nu_floor + 10 eps (never negative when nu_floor >= 0); "
        "a 3-D field gives nu_t > nu_floor; differentiable at rest (aa=0)."
    ),
}

# AMR-Wind v4.2.0 src/turbulence/LES/AMD.H (amd_muvel): both the denominator and the
# returned viscosity are floored at ``std::numeric_limits<double>::epsilon() * 1.0e1``
# (amrex::Real = double).  Ported as the same fp64 literal at every precision.
_AMR_WIND_FLOOR = 10.0 * 2.220446049250313e-16
# RULE 5 smooth default (user decision 2026-10-28): width [m^2/s] of the
# softplus floor on nu_t; 0.0 = AMR-Wind's hard floor (the oracle).
AMD_FLOOR_WIDTH = 1.0e-8
# RULE 5 smooth default (user decision 2026-10-31): width [s^-2] of the softplus
# floor on the denominator a_cd a_cd, equal to the floor itself (10 eps); 0.0 =
# AMR-Wind's hard floor (the oracle).
AMD_DENOM_FLOOR_WIDTH = _AMR_WIND_FLOOR


def amd_nu_t(a11, a12, a13, a21, a22, a23, a31, a32, a33,
             dx, dy, dz, c_amd, nu_floor=0.0, floor_width=AMD_FLOOR_WIDTH,
             denom_floor_width=AMD_DENOM_FLOOR_WIDTH):
    """Anisotropic Minimum-Dissipation eddy viscosity from the nine gradients.

    Parameters
    ----------
    a11..a33 : arrays
        ``a_cd = ∂u_c/∂x_d`` at the target points (all same shape).
    dx, dy, dz : float or array
        Per-direction filter widths. ``dz`` may be scalar or broadcastable to
        the gradient shape (e.g. a per-level ``(nz,)`` on a stretched grid).
    c_amd : float
        Model constant (modified Poincaré constant; legoESM default 0.3,
        AMR-Wind's default ``C_poincare`` is 1/3). Assumed ``c_amd >= 0``.
    nu_floor : float, default 0.0
        Optional additive background viscosity [m²/s] (legoESM extension; the
        oracle has none, so ``nu_floor=0`` is the oracle).
    floor_width : float, default ``AMD_FLOOR_WIDTH`` (1e-8)
        Width w [m²/s] of the softplus output floor
        ``floor + w·softplus((x − floor)/w)`` (static Python float).  ``0.0``
        is AMR-Wind's hard ``max(10·eps, x)`` (the oracle).
    denom_floor_width : float, default ``AMD_DENOM_FLOOR_WIDTH`` (10·eps)
        Width [s⁻²] of the softplus floor on the denominator ``a_cd a_cd``
        (static Python float); ``0.0`` is AMR-Wind's hard ``max(10·eps, ·)``.

    Returns
    -------
    nu_t : array
        Eddy viscosity, same shape as the gradients:
        ``max(10·eps, −C·Σ/max(10·eps, a_cd a_cd)) + nu_floor``.
    """
    # Faithful port of AMR-Wind v4.2.0 amd_muvel (pure-shear part, beta=0): the
    # SAME loop order, accumulation order and floors, so the result matches the
    # running oracle to round-off.  g[i][j] = du_i/dx_j = AMR-Wind
    # gradVel(i*3 + j) (component-major, src/fvm/gradient.H).
    g = ((a11, a12, a13), (a21, a22, a23), (a31, a32, a33))
    h = (dx, dy, dz)
    num_shear = 0.0
    denom = 0.0
    for ii in range(3):
        for jj in range(3):
            diuj = g[ii][jj]
            djui = g[jj][ii]
            denom = denom + diuj * diuj
            sij = 0.5 * (diuj + djui)
            for kk in range(3):
                num_shear = num_shear + g[ii][kk] * g[jj][kk] * h[kk] * h[kk] * sij
    # Floors: hard (jnp.maximum, the oracle) at width 0, else the RULE 5
    # softplus floor.  No sqrt, so AD-safe at rest (aa=0).
    if denom_floor_width == 0.0:
        denom = jnp.maximum(_AMR_WIND_FLOOR, denom)
    elif denom_floor_width > 0.0:
        denom = _AMR_WIND_FLOOR + denom_floor_width * jax.nn.softplus(
            (denom - _AMR_WIND_FLOOR) / denom_floor_width)
    else:
        raise ValueError(f"denom_floor_width must be >= 0, got {denom_floor_width!r}")
    num_shear = num_shear * (-c_amd)
    x = num_shear / denom
    if floor_width == 0.0:
        nu_t = jnp.maximum(_AMR_WIND_FLOOR, x)   # hard limit = the oracle
    elif floor_width > 0.0:
        # RULE 5 smooth default: softplus floor (-> max(floor, x) as w -> 0).
        nu_t = _AMR_WIND_FLOOR + floor_width * jax.nn.softplus(
            (x - _AMR_WIND_FLOOR) / floor_width)
    else:
        raise ValueError(f"floor_width must be >= 0, got {floor_width!r}")
    return nu_t + nu_floor
