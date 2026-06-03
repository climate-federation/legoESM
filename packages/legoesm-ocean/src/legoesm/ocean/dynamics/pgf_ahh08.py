"""Grid-neutral building blocks for the Adcroft, Hallberg & Hill 2008
analytic finite-volume pressure-gradient force.

Reference: Adcroft, A., Hallberg, R., & Hill, M. (2008), "A finite
volume discretization of the pressure gradient force using analytic
integration", Ocean Modelling, 22(3-4), 106-113.

Algorithm
---------
On a cell of homogeneous (T, S) and reference Wright EOS

    ρ(T, S, p) = (p + p₀) / (λ + α₀·(p + p₀))

the hydrostatic equation ``dp/dz = ρg`` (z positive downward) admits an
analytic solution.  Letting ``u = p + p₀`` (so ``du = dp`` within the
cell), the ODE separates to

    λ·d(ln u) + α₀·du = g·dz

and integrates to the implicit relation

    λ·ln(u_b/u_t) + α₀·(u_b − u_t) = g·(z_b − z_t)        [1]

Given ``u_t = u(z_top)``, equation [1] is solved by Newton iteration
for ``u_b = u(z_bot)``.  Two iterations suffice to ~1e-12 relative
error.

The cell-volume integral of pressure is then closed-form (change of
variables ``dz = (λ + α₀·u)/(g·u)·du``):

    ∫_{z_t}^{z_b} p dz = (1/g)·[(λ − p₀·α₀)·(u_b − u_t)
                                 + (α₀/2)·(u_b² − u_t²)
                                 − p₀·λ·ln(u_b/u_t)]                [2]

Derivation check: ``d/du_b [...]`` equals ``(u_b − p₀)·(λ + α₀·u_b)/u_b
= p(z_b)·dz/du_b``, which is the integrand by Leibniz.

Why analytic FV beats centered/Adcroft on partial cells
-------------------------------------------------------
For a face shared by two adjacent columns L, R with horizontally-
uniform T(z), S(z) (rest state), the cell-averaged pressure
``P̄_cell = ∫p dz / h`` is identical in every column at every level —
the analytic integral [2] depends only on (T_cell, S_cell, u_top,
h_cell), and on a rest state every column has identical (T, S, u_top,
h) at every full-cell level.  At a partial-cell step level, h_cell
differs between sides — but the FV momentum balance on the deeper
cell (face contribution + sidewall contribution) reduces to a
difference of two integrals over the wet z-range, both of which use
the SAME ρ(z) profile on a rest state and cancel exactly.

The centered scheme misses this cancellation because it differences
point pressures at cell centroids (which sit at different z on the
two sides of a step).  The Adcroft & Campin 2004 face correction
extrapolates both pressures to a common reference depth using local ρ
— exact only to O(h_step) since it ignores curvature of ρ(z).  AHH08
captures the full ρ(z) curvature analytically and gives machine-zero
residual on a rest state regardless of step structure.

Empirical context (from project_mpas_etopo_instability.md §8d, §8h):
the rest-state PGF residual under "centered" is ~1.3e-4 m/s/step on
ETOPO+ico4; under "adcroft" ~1.4e-6; under "smc03" ~8.6e-8; AHH08 is
expected to give ~1e-12 (machine epsilon for float32 promoted to
float64 in the EOS iteration).  Whether that residual reduction
translates into longer NaN time on the production stack is the
empirical question Phase 4 of the AHH08 implementation answers.
SMC03's 17× residual reduction over adcroft *did not* extend NaN
time (it shortened it ~10%), which means a smaller residual is not
sufficient for stability — but Phase 4 closes out the seed-side
option list either way.

The grid-neutral primitives in this module operate on per-column
arrays of shape ``(..., nlev)``; only the trailing axis is touched.
Voronoi/lat-lon edge wrappers live in ``mpas_partial_cell_helpers.py``
and ``latlon_cgrid_operators.py`` respectively.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.ocean.eos import (
    _a0, _a1, _a2,                 # specific volume α₀(T, S)
    _b0, _b1, _b2, _b3, _b4, _b5,  # pressure offset p₀(T, S)
    _c0, _c1, _c2, _c3, _c4, _c5,  # lambda(T, S)
)


def wright_eos_coefficients(
    T: jnp.ndarray, S: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the three Wright (1997) EOS coefficients (α₀, p₀, λ).

    These are the closed-form polynomials of (T, S) used by both the
    forward EOS ``ρ = (p+p₀)/(λ + α₀(p+p₀))`` and the AHH08 analytic
    pressure integral.  Re-exposing them as a triple lets the analytic
    integrator avoid a redundant evaluation when the caller already has
    (T, S) per cell.
    """
    al0 = _a0 + _a1 * T + _a2 * S
    p0  = (_b0 + _b4 * S) + T * (_b1 + T * (_b2 + _b3 * T) + _b5 * S)
    lam = (_c0 + _c4 * S) + T * (_c1 + T * (_c2 + _c3 * T) + _c5 * S)
    return al0, p0, lam


def _newton_step_u_bottom(
    u_b: jnp.ndarray,
    u_t: jnp.ndarray,
    al0: jnp.ndarray,
    lam: jnp.ndarray,
    g_dz: jnp.ndarray,
) -> jnp.ndarray:
    """One Newton iteration on ``F(u_b) = λ·ln(u_b/u_t) + α₀·(u_b−u_t) − g·Δz``.

    F'(u_b) = λ/u_b + α₀ > 0 always, so Newton is monotonic.  Two
    iterations from a linear-pressure starting guess get us to ~1e-12
    relative error in ``u_b`` (verified in the unit test).
    """
    F  = lam * jnp.log(u_b / u_t) + al0 * (u_b - u_t) - g_dz
    Fp = lam / u_b + al0
    return u_b - F / Fp


def solve_u_bottom(
    T_cell: jnp.ndarray,
    S_cell: jnp.ndarray,
    u_top: jnp.ndarray,
    h_cell: jnp.ndarray,
    g: float,
    n_iter: int = 3,
) -> jnp.ndarray:
    """Solve the implicit hydrostatic relation for ``u`` at the cell bottom.

    Parameters
    ----------
    T_cell, S_cell : jax.Array, shape (..., nlev)
        Per-cell potential temperature [°C] and salinity [PSU].
    u_top : jax.Array, shape (..., nlev)
        ``u = p + p₀`` at the top interface of each cell [Pa].  Built
        column-wise by the caller.
    h_cell : jax.Array, shape (..., nlev)
        Per-cell layer thickness [m].  Use ``h_partial`` on partial
        cells; partial cells with ``h=0`` (below seafloor) are handled
        cleanly by the Newton update because ``g·dz = 0`` and ``u_b``
        converges to ``u_top`` (no-op).
    g : float
        Gravitational acceleration [m/s²].
    n_iter : int, default 3
        Number of Newton iterations.  Each iteration converges
        quadratically; 3 is overkill in practice (2 is sufficient to
        ~1e-12 relative) but cheap.

    Returns
    -------
    u_bot : jax.Array, shape (..., nlev)
        ``u`` at the bottom interface of each cell.

    Notes
    -----
    Compute is done in float64 internally to keep the log/exp pair
    well-conditioned at u ~ 5.8e8 + p, where catastrophic cancellation
    eats float32.  Output is cast back to T_cell.dtype.
    """
    orig_dtype = T_cell.dtype

    T64 = T_cell.astype(jnp.float64)
    S64 = S_cell.astype(jnp.float64)
    al0, p0, lam = wright_eos_coefficients(T64, S64)
    u_t = u_top.astype(jnp.float64)
    g_dz = jnp.float64(g) * h_cell.astype(jnp.float64)

    # Linear-pressure starting guess: u_b ≈ u_t + ρ_t·g·dz, with
    # ρ_t = u_t / (λ + α₀·u_t).
    rho_t = u_t / (lam + al0 * u_t)
    u_b = u_t + rho_t * g_dz

    for _ in range(n_iter):
        u_b = _newton_step_u_bottom(u_b, u_t, al0, lam, g_dz)

    return u_b.astype(orig_dtype)


def integral_p_dz_cell(
    T_cell: jnp.ndarray,
    S_cell: jnp.ndarray,
    u_top: jnp.ndarray,
    u_bot: jnp.ndarray,
    g: float,
) -> jnp.ndarray:
    """Closed-form ``∫_{z_top}^{z_bot} p(z) dz`` per cell.

    Equation [2] of the module docstring:

        ∫p dz = (1/g)·[(λ − p₀·α₀)·(u_b − u_t) + (α₀/2)·(u_b² − u_t²)
                       − p₀·λ·ln(u_b/u_t)]

    Parameters
    ----------
    T_cell, S_cell, u_top, u_bot : jax.Array, shape (..., nlev)
        Per-cell T, S, and ``u = p + p₀`` at the top and bottom
        interfaces.  ``u_bot`` is typically the output of
        :func:`solve_u_bottom`.
    g : float
        Gravitational acceleration [m/s²].

    Returns
    -------
    F : jax.Array, shape (..., nlev)
        ``∫p dz`` per cell [Pa·m].  Cells with ``u_bot == u_top``
        (zero-thickness partials below seafloor) return 0 cleanly
        because every term in the formula is zero.

    Notes
    -----
    For numerical safety, the ``ln(u_b/u_t)`` term is computed as
    ``log(u_b) − log(u_t)`` to avoid the overflow / underflow that a
    direct ratio would risk at u ~ 5.8e8.  At u_b ≈ u_t (very thin
    cells) the difference of logs is well-conditioned because both
    arguments are O(1).
    """
    orig_dtype = T_cell.dtype
    T64 = T_cell.astype(jnp.float64)
    S64 = S_cell.astype(jnp.float64)
    al0, p0, lam = wright_eos_coefficients(T64, S64)
    u_t = u_top.astype(jnp.float64)
    u_b = u_bot.astype(jnp.float64)

    du = u_b - u_t
    # ``ln(u_b/u_t) = log(u_b) - log(u_t)`` is the identity used here.
    # The two logs are O(20) and their difference is well-conditioned
    # because u_b and u_t are both O(5.8e8).
    dlog = jnp.log(u_b) - jnp.log(u_t)
    half_du2 = 0.5 * (u_b * u_b - u_t * u_t)

    F = (1.0 / jnp.float64(g)) * (
        (lam - p0 * al0) * du
        + al0 * half_du2
        - p0 * lam * dlog
    )
    return F.astype(orig_dtype)


def column_pressure_integrals_ahh08(
    T_3d: jnp.ndarray,
    S_3d: jnp.ndarray,
    h_3d: jnp.ndarray,
    g: float,
    p_atmos: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-column AHH08 pressure stack.

    Walks every column from the surface down, solving the analytic
    hydrostatic relation cell-by-cell to produce three arrays:

    * ``u_top_cell[..., k]`` — ``u = p + p₀(T_k, S_k)`` at the TOP
      interface of cell k.  Note that ``p_atmos + p₀(T_0, S_0)`` at
      the surface differs from ``p_atmos + p₀(T_1, S_1)`` at the top
      of cell 1 because p₀ depends on (T, S), so this array is built
      per-cell rather than per-interface.
    * ``u_bot_cell[..., k]`` — ``u`` at the BOTTOM interface of cell k.
      ``u_top_cell[..., k+1]`` is reconstructed from ``u_bot_cell[...,
      k]`` by adding ``p₀(T_{k+1}, S_{k+1}) − p₀(T_k, S_k)`` to
      account for the EOS-coefficient discontinuity at the interface.
    * ``F_cell[..., k]`` — the per-cell ``∫p dz`` integral [Pa·m].

    Parameters
    ----------
    T_3d, S_3d : jax.Array, shape (..., nlev)
        Per-cell T, S at the cell center (homogeneous within a cell).
    h_3d : jax.Array, shape (..., nlev)
        Per-cell layer thickness [m].  On partial cells use
        ``h_partial`` (cells below the seafloor have ``h = 0`` and
        contribute zero to F and zero to the running pressure).
    g : float
    p_atmos : float, default 0
        Atmospheric pressure at the surface [Pa].  The model uses 0
        as the reference (free surface ``η`` carries the barotropic
        pressure separately), so this defaults to 0.

    Returns
    -------
    u_top_cell : jax.Array, shape (..., nlev)
    u_bot_cell : jax.Array, shape (..., nlev)
    F_cell : jax.Array, shape (..., nlev)
        ``∫p dz`` per cell [Pa·m].

    Notes
    -----
    Implemented as a Python ``for k in range(nlev)`` loop, which JAX
    unrolls during tracing.  ``nlev`` is small (~20-50) and statically
    known; ``lax.scan`` would not buy speed here and would require
    pytree gymnastics for the (u_top, u_bot, F) carry.
    """
    nlev = T_3d.shape[-1]

    # p₀(T, S) per cell — needed both as part of u and to compute the
    # interface jump in u between cells of different (T, S).
    _, p0_cell, _ = wright_eos_coefficients(
        T_3d.astype(jnp.float64), S_3d.astype(jnp.float64),
    )

    u_top_list = []
    u_bot_list = []
    F_list = []

    # Surface: u_top_cell[k=0] = p_atmos + p₀(T_0, S_0).
    u_t_k = jnp.float64(p_atmos) + p0_cell[..., 0]
    for k in range(nlev):
        T_k = T_3d[..., k]
        S_k = S_3d[..., k]
        h_k = h_3d[..., k]
        u_b_k = solve_u_bottom(T_k, S_k, u_t_k, h_k, g, n_iter=3)
        F_k = integral_p_dz_cell(T_k, S_k, u_t_k, u_b_k, g)

        u_top_list.append(u_t_k)
        u_bot_list.append(u_b_k)
        F_list.append(F_k)

        if k < nlev - 1:
            # Interface jump: pressure p is continuous, but u = p + p₀
            # is not because p₀ jumps when (T, S) changes between cells.
            # u_top of cell k+1 = p(z_{k+1, top}) + p₀(T_{k+1}, S_{k+1})
            #                   = (u_b_k − p₀(T_k, S_k)) + p₀(T_{k+1}, S_{k+1}).
            u_t_k = (u_b_k - p0_cell[..., k]) + p0_cell[..., k + 1]

    u_top_cell = jnp.stack(u_top_list, axis=-1).astype(T_3d.dtype)
    u_bot_cell = jnp.stack(u_bot_list, axis=-1).astype(T_3d.dtype)
    F_cell = jnp.stack(F_list, axis=-1).astype(T_3d.dtype)
    return u_top_cell, u_bot_cell, F_cell
