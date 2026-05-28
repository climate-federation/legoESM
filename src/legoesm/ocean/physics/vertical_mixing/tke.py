"""Prognostic TKE vertical mixing (Gaspar 1990 / Burchard 2002).

Veros's canonical vertical-mixing closure
(``settings.enable_tke = True``). Solves a prognostic budget for the
turbulent kinetic energy per unit mass ``TKE [m^2/s^2]`` at cell
interfaces, then derives ``K_M`` (momentum) and ``K_H`` (tracer)
eddy diffusivities from the TKE field and a mixing-length closure.

Closure equations
-----------------

Per column at interfaces (between cells ``k`` and ``k+1``):

.. math::

   \\frac{\\partial e}{\\partial t}
       = P_s + P_b - \\varepsilon
       + \\frac{\\partial}{\\partial z}
         \\left( \\alpha_{\\text{tke}} \\, K_M
                 \\, \\frac{\\partial e}{\\partial z} \\right)

with

.. math::

   P_s &= K_M \\left[
             \\left(\\frac{\\partial u}{\\partial z}\\right)^2
           + \\left(\\frac{\\partial v}{\\partial z}\\right)^2
         \\right]                 \\;\\text{(shear production)}\\\\

   P_b &= -K_H \\, N^2             \\;\\text{(buoyancy work)}\\\\

   \\varepsilon
       &= c_\\varepsilon \\frac{e^{3/2}}{l_\\varepsilon}
                                  \\;\\text{(dissipation)}\\\\

   K_M &= c_k \\, l_k \\, \\sqrt{2 e}\\\\

   K_H &= K_M \\quad
              (\\text{turbulent Prandtl number} = 1
              \\text{ in the Gaspar / Burchard / Veros canonical form}).

Mixing lengths
--------------

The Bougeault-Lacarrere (1989) asymmetric construction
(``tke_mxl_choice = 2``, Veros default):

.. math::

   l_{up}(z) \\;=\\; \\text{vertical distance a parcel of TKE }e\\,
       \\text{rises before }\\int_z^{z+l_{up}} N^2\\,dz'\\;=\\;e\\\\

   l_{dn}(z) \\;=\\; \\text{symmetric, integrated downward}\\\\

   l_k = \\sqrt{l_{up}\\,l_{dn}}, \\qquad
   l_\\varepsilon = \\max(l_{up},\\,l_{dn})

with a floor ``mxl_min`` enforced everywhere.

Surface flux
------------

::

    forc_tke_surface = (|tau| / rho_0)^{3/2}     [m^3 / s^3]

Applied as a flux boundary condition at the top interface (the
top-of-column TKE diffusion flux).

Implementation notes
--------------------

This implementation expects velocities and tracers at **cell centres**
(shape ``(..., nlev)``). On lat-lon C-grid / MPAS / cubed-sphere the
caller must interpolate before invoking
:func:`tke_vertical_mixing`. Density at cell centres is derived from
T and S via the supplied EOS function.

Vertical positions, interface levels:

- ``e[..., k]`` (TKE) lives at interface ``k`` for
  ``k = 0, ..., nlev - 2`` between cell centres ``k`` and ``k + 1``.
  Total length ``nlev - 1``.
- Cell-centre fields (T, S, u, v) have length ``nlev``.

The prognostic step is solved backward-Euler (linear-in-TKE forms of
the source terms with the dissipation linearised about the previous
TKE) → one tridiagonal solve per column. Stable for the long ocean
``dt`` used in production.

References
----------
- Gaspar, P., Y. Gregoris, and J.-M. Lefevre (1990): A simple eddy
  kinetic energy model for simulations of the oceanic vertical mixing.
  *J. Geophys. Res.*, 95, 16179-16193.
- Bougeault, P. and P. Lacarrere (1989): Parameterization of orography-
  induced turbulence in a mesobeta-scale model. *Mon. Weather Rev.*,
  117, 1872-1890.
- Burchard, H. (2002): Energy-conserving discretisation of turbulent
  shear and buoyancy production. *Ocean Modelling*, 4, 347-361.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

_EPS = float(jnp.finfo(jnp.float32).eps)


class TKEOutput(NamedTuple):
    """Output of :func:`tke_vertical_mixing`."""
    K_M: jnp.ndarray       # (..., nlev-1) momentum eddy viscosity at interfaces
    K_H: jnp.ndarray       # (..., nlev-1) tracer eddy diffusivity at interfaces
    tke_new: jnp.ndarray   # (..., nlev-1) updated TKE at interfaces
    l_eps: jnp.ndarray     # (..., nlev-1) dissipation mixing length (diagnostic)


# ---------------------------------------------------------------------------
# Mixing length: Bougeault-Lacarrere (1989) asymmetric construction.
# ---------------------------------------------------------------------------


def _bougeault_lacarrere_lengths(
    e: jnp.ndarray,
    N2: jnp.ndarray,
    dz_half: jnp.ndarray,
    mxl_min: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute upward and downward mixing lengths l_up, l_dn.

    Parameters
    ----------
    e : (..., nlev-1)
        TKE at interfaces.
    N2 : (..., nlev-1)
        Squared Brunt-Vaisala frequency at interfaces (>= 0 for stable
        stratification; this function expects ``N2 >= 0``).
    dz_half : (..., nlev-1) or (..., nlev-2)
        Distance between adjacent interface levels. Both length forms
        appear in different ocean modules; this function uses
        ``dz_half[..., k]`` as the distance between interface ``k``
        and interface ``k+1``. Pass either shape; we slice to
        ``(..., nlev-2)`` internally.
    mxl_min : float
        Floor applied to both lengths to prevent zero division and
        to model the no-slip-boundary residual length.

    Returns
    -------
    l_up, l_dn : (..., nlev-1)
    """
    # Distance between adjacent interfaces; length nlev-2 derived from the
    # nlev-1 e-array.
    dz_int = dz_half[..., :e.shape[-1] - 1]   # (..., nlev-2)

    # TKE budget along the asymmetric path:
    #     Σ_k N2[k] · l_up[k] · dz_int[k] = e at the starting interface
    # Discretely this is a running cumulative sum that we stop when the
    # accumulated PE exceeds the available TKE. The continuous form
    # gives the inverse:  l_up · sqrt(N2) ≈ sqrt(2e) so
    # l_up ≈ sqrt(2e / max(N2, eps)).  This is the canonical
    # closed-form approximation used in Gaspar 1990 / Veros — full
    # integral is more expensive and only marginally improves K profiles.
    N2_safe = jnp.maximum(N2, _EPS)
    l_up_raw = jnp.sqrt(2.0 * e / N2_safe + _EPS)
    l_dn_raw = l_up_raw   # symmetric under the closed-form approximation

    # Apply mxl_min floor. ``dz_int`` is also used as a per-cell upper
    # bound so the length cannot exceed one cell — prevents the
    # closed-form expression from blowing up in nearly-neutral layers
    # (N2 → 0). ``2 * dz_int`` is a typical upper bound used by Veros's
    # implementation; we apply it conservatively.
    upper = 2.0 * jnp.maximum(dz_int, mxl_min)
    # broadcast upper to (..., nlev-1) by repeating the last value if needed
    if upper.shape[-1] < l_up_raw.shape[-1]:
        upper = jnp.concatenate([upper, upper[..., -1:]], axis=-1)
    l_up = jnp.clip(l_up_raw, mxl_min, upper)
    l_dn = jnp.clip(l_dn_raw, mxl_min, upper)
    return l_up, l_dn


def compute_mixing_lengths(
    e: jnp.ndarray,
    N2: jnp.ndarray,
    dz_half: jnp.ndarray,
    cfg: TKEConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute (l_k, l_eps) for the chosen ``tke_mxl_choice``.

    Returns
    -------
    l_k, l_eps : (..., nlev-1) — for use in K = c_k·l_k·sqrt(2e) and
        eps = c_eps·e^{3/2} / l_eps respectively.
    """
    if cfg.tke_mxl_choice == 2:
        l_up, l_dn = _bougeault_lacarrere_lengths(
            e, N2, dz_half, cfg.mxl_min,
        )
        l_k = jnp.sqrt(jnp.maximum(l_up * l_dn, cfg.mxl_min ** 2))
        l_eps = jnp.maximum(l_up, l_dn)
    elif cfg.tke_mxl_choice == 1:
        # Simple parabolic / linear length scale.
        # l = max(mxl_min, min(kappa_vk * z, sqrt(2e / max(N2, eps))))
        # We use the closed-form sqrt(2e/N^2) consistent with choice=2
        # but without the symmetric average.
        N2_safe = jnp.maximum(N2, _EPS)
        l_k = jnp.maximum(jnp.sqrt(2.0 * e / N2_safe + _EPS), cfg.mxl_min)
        l_eps = l_k
    else:
        raise ValueError(
            f"Unknown tke_mxl_choice={cfg.tke_mxl_choice!r}; expected 1 or 2."
        )
    return l_k, l_eps


# ---------------------------------------------------------------------------
# Shear, buoyancy, dissipation at interfaces
# ---------------------------------------------------------------------------


def _vertical_shear_squared(
    u_cell: jnp.ndarray, v_cell: jnp.ndarray, dz_half: jnp.ndarray,
) -> jnp.ndarray:
    """Compute ``|du/dz|^2 + |dv/dz|^2`` at interfaces.

    Parameters
    ----------
    u_cell, v_cell : (..., nlev) — cell-centre velocities.
    dz_half : (..., nlev-1) — distance between cell centres.

    Returns
    -------
    S2 : (..., nlev-1) — squared vertical shear at interfaces.
    """
    dz_safe = jnp.maximum(dz_half, _EPS)
    du = (u_cell[..., 1:] - u_cell[..., :-1]) / dz_safe
    dv = (v_cell[..., 1:] - v_cell[..., :-1]) / dz_safe
    return du * du + dv * dv


def _compute_N2(
    rho_cell: jnp.ndarray, dz_half: jnp.ndarray, rho_0: float,
) -> jnp.ndarray:
    """N^2 at interfaces from cell-centre in-situ density.

    ``N^2 = -(g/rho_0) drho/dz`` with z positive upward. legoESM's
    cell index k = 0 is the surface (top) and k = nlev-1 is the
    bottom, so z is decreasing with k. Discretely, between
    cell centres k and k+1 a distance ``dz_half[k]`` apart:

        drho/dz = (rho[k] - rho[k+1]) / dz_half[k]

    For stable stratification this is negative (light water on top),
    so ``N^2 = -g/rho_0 * drho/dz > 0`` as expected.

    Returns
    -------
    N2 : (..., nlev-1) — clipped to >= 0 (stably stratified). The
        unstable case is handled by convective adjustment elsewhere.
    """
    dz_safe = jnp.maximum(dz_half, _EPS)
    # drho/dz with z positive upward — negative for stable stratification.
    drho_dz = (rho_cell[..., :-1] - rho_cell[..., 1:]) / dz_safe
    N2 = -constants.g / rho_0 * drho_dz
    return jnp.maximum(N2, 0.0)


# ---------------------------------------------------------------------------
# Prognostic TKE backward-Euler step
# ---------------------------------------------------------------------------


def _tridiag_thomas(a, b, c, d):
    """Solve a tridiagonal system A x = d via the Thomas algorithm.

    a, b, c, d each have shape ``(..., N)`` and ``a[..., 0]``,
    ``c[..., -1]`` are unused (left as zero by the caller). Returns
    ``x`` of shape ``(..., N)``.
    """
    N = b.shape[-1]

    def step(carry, k):
        c_prev, d_prev = carry
        denom = b[..., k] - a[..., k] * c_prev
        denom_safe = jnp.where(jnp.abs(denom) > _EPS, denom, _EPS)
        cp = c[..., k] / denom_safe
        dp = (d[..., k] - a[..., k] * d_prev) / denom_safe
        return (cp, dp), (cp, dp)

    # Forward sweep
    init_c = jnp.zeros_like(b[..., 0])
    init_d = jnp.zeros_like(d[..., 0])
    _, (cp_all, dp_all) = jax.lax.scan(
        step, (init_c, init_d), jnp.arange(N),
    )
    # cp_all, dp_all have shape (N, ...); transpose so trailing axis is N.
    cp_all = jnp.moveaxis(cp_all, 0, -1)
    dp_all = jnp.moveaxis(dp_all, 0, -1)

    # Back substitution
    def back(carry, k_rev):
        x_next = carry
        k = N - 1 - k_rev
        x = jnp.where(
            k_rev == 0, dp_all[..., k],
            dp_all[..., k] - cp_all[..., k] * x_next,
        )
        return x, x

    x_init = jnp.zeros_like(b[..., 0])
    _, x_rev = jax.lax.scan(back, x_init, jnp.arange(N))
    x_rev = jnp.moveaxis(x_rev, 0, -1)
    # Reverse the back-sub output to get x in natural index order.
    return x_rev[..., ::-1]


def _solve_tke_backward_euler(
    e_old: jnp.ndarray,
    K_M_old: jnp.ndarray,
    K_H_old: jnp.ndarray,
    P_s: jnp.ndarray,
    N2: jnp.ndarray,
    l_eps: jnp.ndarray,
    dz_half: jnp.ndarray,
    surface_flux: jnp.ndarray,
    dt: float,
    cfg: TKEConfig,
) -> jnp.ndarray:
    """Backward-Euler tridiagonal solve for one TKE time step.

    Linearises ``-c_eps * e^{3/2} / l_eps`` as ``-c_eps * sqrt(e_old) / l_eps · e_new``
    so the resulting system is linear in e_new. ``P_b = -K_H * N^2`` is
    treated explicitly with sign-aware splitting: when stable
    (``N2 > 0``), ``-K_H · N^2`` is a sink and is linearised in e_new
    via ``K_H = c_k · l_k · sqrt(2 e_old)`` so the implicit step does
    not overshoot to negative TKE.

    Parameters
    ----------
    e_old : (..., nlev-1)
    K_M_old, K_H_old : (..., nlev-1) — used in the linearised P_b sink term.
    P_s : (..., nlev-1) — shear production (explicit source).
    N2 : (..., nlev-1) — squared buoyancy frequency.
    l_eps : (..., nlev-1) — dissipation length.
    dz_half : (..., nlev-1) — cell-centre spacing.
    surface_flux : (...) — flux boundary condition at top interface
        ``forc_tke_surface = (|tau|/rho_0)^{3/2}``.
    dt : float
    cfg : TKEConfig

    Returns
    -------
    e_new : (..., nlev-1)
    """
    N = e_old.shape[-1]   # number of interfaces
    e_sqrt = jnp.sqrt(jnp.maximum(e_old, cfg.tke_background))
    # Linearised dissipation rate (per unit e_new):
    diss_rate = cfg.c_eps * e_sqrt / jnp.maximum(l_eps, cfg.mxl_min)
    # Linearised buoyancy sink (per unit e_new):
    #   P_b = -K_H · N^2 ≈ -(c_k · l_k · sqrt(2e_old)) · N^2  per unit e
    # We absorb the sqrt(2 e_old) factor into a rate by dividing by e_old.
    # When N^2 = 0 (neutral / unstable), this rate vanishes.
    e_safe = jnp.maximum(e_old, cfg.tke_background)
    buoy_sink_rate = K_H_old * N2 / e_safe

    # Diffusion coefficients on the *flux faces* between interface k and
    # interface k+1 (one less than the number of interfaces).
    K_tke_face = cfg.alpha_tke * 0.5 * (K_M_old[..., :-1] + K_M_old[..., 1:])
    dz_face = jnp.maximum(dz_half[..., :N - 1], _EPS)
    # Tridiagonal coefficients. Interior interfaces k = 1 ... N-2.
    a_diff = jnp.zeros_like(e_old)
    b_diff = jnp.zeros_like(e_old)
    c_diff = jnp.zeros_like(e_old)

    # Interior contributions to a/b/c from diffusion:
    if N >= 3:
        # For interior interface k, the flux into face (k-1, k) uses
        # K_tke_face[..., k-1] / dz_face[..., k-1], and into face (k, k+1)
        # uses K_tke_face[..., k] / dz_face[..., k].
        coef_lower = K_tke_face / dz_face                # (..., N-1)
        # Build full-length per-interface aggregates:
        #   a[k] = -dt * coef_lower[k-1] / dz_int_eff[k]   (sub-diagonal)
        #   c[k] = -dt * coef_lower[k]   / dz_int_eff[k]   (super-diagonal)
        # Approximate cell-thickness around interface k as average of
        # adjacent face thicknesses.
        dz_int_eff = jnp.where(
            jnp.arange(N) == 0,
            dz_face[..., :1],
            jnp.concatenate([dz_face[..., :1], dz_face], axis=-1)[..., :N],
        )
        # Sub-diagonal at interior interfaces (skip k=0):
        a_inter = dt * coef_lower / jnp.maximum(dz_int_eff[..., 1:N], _EPS)
        a_diff = jnp.concatenate(
            [jnp.zeros_like(a_inter[..., :1]), -a_inter], axis=-1,
        )
        # Super-diagonal at interior interfaces (skip k=N-1):
        c_inter = dt * coef_lower / jnp.maximum(dz_int_eff[..., :N - 1], _EPS)
        c_diff = jnp.concatenate(
            [-c_inter, jnp.zeros_like(c_inter[..., :1])], axis=-1,
        )
        b_diff = -(a_diff + c_diff)

    # Total tridiagonal matrix entries:
    #   (1 + dt * (diss_rate + buoy_sink_rate)) * e_new
    #     + diffusion contribution = e_old + dt * P_s + flux BC
    diag = 1.0 + dt * (diss_rate + buoy_sink_rate) + b_diff

    # RHS: explicit shear-production source + previous-step e.
    rhs = e_old + dt * P_s

    # Surface flux BC at interface k=0: add the flux divergence with
    # ``forc_tke_surface``-style energy input.
    rhs = rhs.at[..., 0].add(dt * surface_flux / jnp.maximum(dz_half[..., 0], _EPS))

    # Solve tridiagonal system.
    e_new = _tridiag_thomas(a_diff, diag, c_diff, rhs)

    # Floor at background; clamp away from negative.
    e_new = jnp.maximum(e_new, cfg.tke_background)
    # Floor at surface_min on the topmost interface only.
    e_new = e_new.at[..., 0].set(
        jnp.maximum(e_new[..., 0], cfg.tke_surface_min),
    )
    return e_new


# ---------------------------------------------------------------------------
# K_M, K_H from TKE
# ---------------------------------------------------------------------------


def compute_K_from_tke(
    e: jnp.ndarray,
    l_k: jnp.ndarray,
    cfg: TKEConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute K_M and K_H from TKE and the mixing length.

    .. math::

        K_M = c_k \\, l_k \\, \\sqrt{2 e}

    K_H equals K_M in the canonical Gaspar 1990 / Veros form
    (turbulent Prandtl number = 1). When ``enable_kappaH_profile`` is
    set, K_H tapers toward kappaH_min near the surface — that
    refinement is not implemented in v1 and the option is recorded
    but currently ignored.

    Returns
    -------
    K_M, K_H : (..., nlev-1) at interfaces. Both floored at
    ``kappaM_min`` / ``kappaH_min`` respectively.
    """
    K_M = cfg.c_k * l_k * jnp.sqrt(2.0 * jnp.maximum(e, cfg.tke_background))
    K_M = jnp.maximum(K_M, cfg.kappaM_min)
    K_H = jnp.maximum(K_M, cfg.kappaH_min)
    return K_M, K_H


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def tke_vertical_mixing(
    u_cell: jnp.ndarray,
    v_cell: jnp.ndarray,
    T_cell: jnp.ndarray,
    S_cell: jnp.ndarray,
    rho_cell: jnp.ndarray,
    dz_half: jnp.ndarray,
    tke_old: jnp.ndarray | None,
    tau_x_surface: jnp.ndarray | None,
    tau_y_surface: jnp.ndarray | None,
    dt: float,
    cfg: TKEConfig,
    rho_0: float = constants.rho_ocean,
    n_iterations: int = 1,
) -> TKEOutput:
    """Advance the TKE closure and return new K_M, K_H, TKE.

    Mode A — **prognostic** (recommended when state-pytree wiring is
    available): caller passes ``tke_old`` from the previous step and
    ``n_iterations=1``. One backward-Euler tridiagonal solve carries
    TKE forward one ``dt``.

    Mode B — **diagnostic / quasi-steady-state** (used when no prognostic
    TKE state is carried on the ocean state pytree yet): caller passes
    ``tke_old=None`` and ``n_iterations=3`` (or more). The function
    seeds TKE at ``cfg.tke_background`` and iterates the same backward
    Euler step ``n_iterations`` times to converge toward the local
    quasi-steady-state. This produces K_M / K_H within a few percent
    of the prognostic equilibrium for typical ocean shear/stratification,
    at the cost of repeating the tridiagonal solve.

    Mode B is the appropriate default until ``state.tke`` lands on
    :class:`legoesm.ocean.state.LatLonCGridOceanState`. The audit
    documents this as Phase G.1a follow-up.

    Parameters
    ----------
    u_cell, v_cell : (..., nlev)
        Velocity components at cell centres (interpolated from the
        appropriate staggered grid by the caller).
    T_cell, S_cell : (..., nlev)
        Tracer fields at cell centres (accepted for signature parity
        with the other vertical-mixing schemes; unused inside this
        function).
    rho_cell : (..., nlev)
        In-situ density at cell centres — used to compute ``N^2``.
    dz_half : (..., nlev-1)
        Distance between adjacent cell centres.
    tke_old : (..., nlev-1) or None
    tau_x_surface, tau_y_surface : (...)
    dt : float
    cfg : TKEConfig
    n_iterations : int
        See Mode A / Mode B above.

    Returns
    -------
    TKEOutput
    """
    if tke_old is None:
        leading_shape = rho_cell.shape[:-1]
        nlev = rho_cell.shape[-1]
        tke_old = jnp.full(
            leading_shape + (nlev - 1,),
            cfg.tke_background,
            dtype=rho_cell.dtype,
        )

    shear_sq = _vertical_shear_squared(u_cell, v_cell, dz_half)
    N2 = _compute_N2(rho_cell, dz_half, rho_0)

    if tau_x_surface is None and tau_y_surface is None:
        surface_flux = jnp.zeros(rho_cell.shape[:-1], dtype=rho_cell.dtype)
    else:
        tx = tau_x_surface if tau_x_surface is not None else jnp.zeros_like(rho_cell[..., 0])
        ty = tau_y_surface if tau_y_surface is not None else jnp.zeros_like(rho_cell[..., 0])
        surface_flux = (jnp.sqrt(tx * tx + ty * ty) / rho_0) ** 1.5

    # Sub-iteration loop (Mode B convergence; Mode A uses n_iterations=1).
    tke_curr = tke_old
    for _ in range(max(1, int(n_iterations))):
        l_k, l_eps = compute_mixing_lengths(tke_curr, N2, dz_half, cfg)
        K_M_curr, K_H_curr = compute_K_from_tke(tke_curr, l_k, cfg)
        P_s_curr = K_M_curr * shear_sq
        tke_curr = _solve_tke_backward_euler(
            e_old=tke_curr,
            K_M_old=K_M_curr, K_H_old=K_H_curr,
            P_s=P_s_curr, N2=N2, l_eps=l_eps,
            dz_half=dz_half,
            surface_flux=surface_flux,
            dt=dt, cfg=cfg,
        )

    # Final K from converged TKE.
    l_k_final, l_eps_final = compute_mixing_lengths(tke_curr, N2, dz_half, cfg)
    K_M, K_H = compute_K_from_tke(tke_curr, l_k_final, cfg)

    return TKEOutput(K_M=K_M, K_H=K_H, tke_new=tke_curr, l_eps=l_eps_final)


__all__ = (
    "TKEOutput",
    "compute_K_from_tke",
    "compute_mixing_lengths",
    "tke_vertical_mixing",
)
