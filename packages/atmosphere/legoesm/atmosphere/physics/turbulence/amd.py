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

where ``g_k = (a_1k, a_2k, a_3k)`` is the k-th column of the gradient tensor
(the derivative of the velocity VECTOR in direction k).  Key properties:

* **Minimum dissipation** (all inequalities below assume ``c_amd >= 0``; a
  negative ``c_amd`` — not validated here — flips every sign). The outer floor makes
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
definition.  AD-safe by construction — the two hard floors (``jnp.maximum``) and
the absence of any sqrt avoid the ``sqrt(0)`` gradient trap the Vreman/Smagorinsky
paths must guard; on the floored side the gradient is 0.

Faithfulness: a line-by-line port of AMR-Wind v4.2.0
(0ee09c22b9afeca06231e4910276c0ba05fb52dd) ``src/turbulence/LES/AMD.H::amd_muvel``
with ``beta = 0`` — same loop and accumulation order, same ``10·eps`` floors on the
denominator and on ν_t — matched to the RUNNING oracle in
``tests/atmosphere/test_amd_oracle.py``.  Not ported: AMR-Wind's Abkar–Moin
buoyancy numerator (legoESM's cited references are shear-only); ``nu_floor`` is a
legoESM additive extension (0 = oracle).  The closed form is also pinned against
an independent Rozema-2015 reimplementation in ``tests/unit/test_amd_faithful.py``.
"""
from __future__ import annotations

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
        "c_amd": "1", "nu_floor": "m^2/s",
    },
    "outputs": {"nu_t": "m^2/s"},
    "sign_convention": (
        "Diagnostic eddy viscosity only (no tendency). For c_amd >= 0: nu_t >= "
        "nu_floor + 10 eps by construction (the minimum-dissipation floor; the "
        "eddy-viscosity term is never negative), and = nu_floor + 10 eps wherever "
        "N <= 0 (an off-diagonal 1-D shear, or an EXPANDING normal strain); a "
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
        "10 eps; nu_t >= nu_floor + 10 eps (never negative when nu_floor >= 0); "
        "a 3-D field gives nu_t > nu_floor; differentiable at rest (aa=0)."
    ),
}

# AMR-Wind v4.2.0 src/turbulence/LES/AMD.H (amd_muvel): both the denominator and the
# returned viscosity are floored at ``std::numeric_limits<double>::epsilon() * 1.0e1``
# (amrex::Real = double).  Ported as the same fp64 literal at every precision.
_AMR_WIND_FLOOR = 10.0 * 2.220446049250313e-16


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
        Model constant (modified Poincaré constant; legoESM default 0.3,
        AMR-Wind's default ``C_poincare`` is 1/3). Assumed ``c_amd >= 0``.
    nu_floor : float, default 0.0
        Optional additive background viscosity [m²/s] (legoESM extension; the
        oracle has none, so ``nu_floor=0`` is the oracle).

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
    # Hard floors exactly as the oracle (jnp.maximum: piecewise differentiable,
    # gradient 0 on the floored side).  No sqrt, so AD-safe at rest (aa=0).
    denom = jnp.maximum(_AMR_WIND_FLOOR, denom)
    num_shear = num_shear * (-c_amd)
    nu_t = jnp.maximum(_AMR_WIND_FLOOR, num_shear / denom)
    return nu_t + nu_floor
