"""Shared vertical-mixing kernels.

Factored to remove the identical stack -> vmap -> unstack block that the
constant- and Richardson-coefficient schemes duplicated (they differ only in
the per-field diffusion function and the viscosity/diffusivity coefficients).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.eos import (
    thermal_expansion_coeff,
    haline_contraction_coeff,
)

# Shared float32-eps floor for the vertical-mixing kernels.
_EPS = float(jnp.finfo(jnp.float32).eps)


def vertical_shear_squared(
    u_cell: jnp.ndarray, v_cell: jnp.ndarray, dz_half: jnp.ndarray,
) -> jnp.ndarray:
    """Compute ``|du/dz|^2 + |dv/dz|^2`` at interfaces.

    Shared by the TKE (Gaspar/Burchard) and CATKE prognostic closures.

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


def richardson_number(
    N2: jnp.ndarray,
    u_cell: jnp.ndarray,
    v_cell: jnp.ndarray,
    dz_actual: jnp.ndarray,
    *,
    eps: float,
    clip_negative: bool = False,
) -> jnp.ndarray:
    """Gradient Richardson number ``Ri = N^2 / S^2`` at interfaces.

    Single source for the Richardson-scheme (``richardson.py``) and KPP
    interior-shear (``kpp.py``) blocks, which computed this from byte-identical
    inline code (#518 item 5).  ``N2`` is supplied by the caller (both already
    call ``eos.compute_buoyancy_frequency``; KPP also reuses ``N2`` for its
    static-instability term).

    The squared shear uses the floor-on-the-SQUARED-denominator form
    ``S^2 = (du^2 + dv^2) / max(dz_half^2, eps)`` with ``du = u[k] - u[k+1]``
    — preserved bit-for-bit from both call sites.  NOTE: this floor placement
    differs from :func:`vertical_shear_squared` (which floors the LINEAR
    ``dz_half`` before dividing); the two are equal away from vanishing
    ``dz_half`` but diverge in sub-eps-thin layers, so they are intentionally
    NOT merged here — reconciling the floor convention is a numerics-affecting
    change for a separate PR.

    Parameters
    ----------
    N2 : (..., nlev-1) — buoyancy frequency squared at interfaces.
    u_cell, v_cell : (..., nlev) — cell-centre velocities.
    dz_actual : (..., nlev) — actual layer thickness (z* Jacobian applied).
    eps : float — denominator floor (caller's scheme eps; applied to both
        ``dz_half**2`` and ``S2``).
    clip_negative : bool — when True, clip ``Ri`` to ``>= 0`` (the Richardson
        scheme's "unstable -> max mixing" convention); KPP passes False and
        clamps downstream via ``Ri / Ri_0``.

    Returns
    -------
    Ri : (..., nlev-1)
    """
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    du = u_cell[..., :-1] - u_cell[..., 1:]
    dv = v_cell[..., :-1] - v_cell[..., 1:]
    S2 = (du**2 + dv**2) / jnp.maximum(dz_half**2, eps)
    Ri = N2 / jnp.maximum(S2, eps)
    if clip_negative:
        Ri = jnp.maximum(Ri, 0.0)
    return Ri


def compute_N2(
    rho_cell: jnp.ndarray, dz_half: jnp.ndarray, rho_0: float,
    g: float = constants.g,
    *,
    T_cell: jnp.ndarray | None = None,
    S_cell: jnp.ndarray | None = None,
    p_cell: jnp.ndarray | None = None,
    dz_ref: jnp.ndarray | None = None,
    jacobian: jnp.ndarray | None = None,
    eos_fn=None,
    n2_mode: str = "insitu",
    adiabatic_over_dz_half: bool = False,
) -> jnp.ndarray:
    """N^2 at interfaces (shared by the TKE and CATKE closures).

    ``n2_mode="insitu"`` (default): from cell-centre *in-situ* density,
    ``N^2 = -(g/rho_0) drho/dz`` with z positive upward, clipped >= 0 (the
    in-situ contrast carries compressibility and is biased too stable, so its
    sign is not a reliable convection trigger).  ``n2_mode="adiabatic"``: the
    true static stability via adiabatic parcel displacement to the upper cell's
    pressure (delegating to ``eos.compute_buoyancy_frequency_adiabatic``),
    SIGNED (N^2 < 0 marks a statically unstable interface — the convection
    trigger CATKE / signed-N2 TKE need); requires ``T_cell``/``S_cell``/
    ``p_cell``/``dz_ref``/``jacobian``.  ``adiabatic_over_dz_half=True`` divides
    the adiabatic contrast by the caller's ``dz_half`` (the Veros ``dzw`` slot).

    Returns
    -------
    N2 : (..., nlev-1). Clipped >= 0 for ``"insitu"``; signed for ``"adiabatic"``.
    """
    if n2_mode in ("insitu", "insitu_signed"):
        dz_safe = jnp.maximum(dz_half, _EPS)
        # drho/dz with z positive upward — negative for stable stratification.
        drho_dz = (rho_cell[..., :-1] - rho_cell[..., 1:]) / dz_safe
        N2 = -g / rho_0 * drho_dz
        # "insitu": clipped >= 0 (in-situ contrast is biased too stable; sign is
        # not a reliable convection trigger — TKE's default).  "insitu_signed":
        # SIGNED (N2<0 marks static instability), the cheap convection trigger
        # for convection-aware closures (CATKE) without the adiabatic EOS call.
        return N2 if n2_mode == "insitu_signed" else jnp.maximum(N2, 0.0)
    if n2_mode == "adiabatic":
        if (T_cell is None or S_cell is None or p_cell is None
                or dz_ref is None or jacobian is None):
            raise ValueError(
                "n2_mode='adiabatic' requires T_cell, S_cell, p_cell "
                "(cell-centre pressure [Pa]), dz_ref and jacobian to "
                "displace parcels through the EOS."
            )
        from legoesm.ocean.eos import compute_buoyancy_frequency_adiabatic
        return compute_buoyancy_frequency_adiabatic(
            T_cell, S_cell, p_cell, dz_ref, jacobian,
            eos_fn=eos_fn, rho_ref=rho_0, g=g,
            dz_half=dz_half if adiabatic_over_dz_half else None,
        )
    raise ValueError(
        f"Unknown n2_mode={n2_mode!r}; expected 'insitu', 'insitu_signed' "
        f"or 'adiabatic'."
    )


def surface_buoyancy_flux(
    q_net,
    fw,
    salt,
    T_sfc: jnp.ndarray,
    S_sfc: jnp.ndarray,
    *,
    g: float,
    rho_0: float,
    c_sw: float,
    real_salt_in_qs: bool,
):
    """Surface buoyancy flux ``B_f`` [m^2/s^3] (>0 destabilising) + the kinematic
    surface heat / salt fluxes for the KPP / CATKE boundary-layer closures.

    Single grid-agnostic source for the surface-forcing buoyancy block
    (#518 item 1).  Previously three byte-identical-in-logic copies existed:
    the lat-lon ``integration.py`` inline, ``k_profiles._surface_buoyancy_flux``,
    and ``mpas_integration._mpas_surface_buoyancy_flux``.

    Sign convention (KPP / CATKE: ``B_f > 0`` = unstable = convection):
    ``B_f = -g*alpha*Q_T + g*beta*Q_S`` where ``Q_T = q_net/(rho_0*c_sw)`` and
    ``Q_S`` is the kinematic salt flux.  ``alpha``/``beta`` are the EOS
    thermal-expansion / haline-contraction coefficients at the surface
    (``p = 0``).

    ``real_salt_in_qs`` (the ONE deliberate grid difference):

    * ``True`` (lat-lon): the real brine salt-mass flux feeds BOTH the surface
      buoyancy AND the returned non-local salt flux ``Q_sfc_S`` — they are
      accumulated into a single ``Q_sfc_S`` and ``B_salt = g*beta*Q_sfc_S``.
    * ``False`` (MPAS): the real salt feeds the surface BUOYANCY ONLY; the
      returned ``Q_sfc_S`` carries the freshwater (virtual-salt) term ONLY,
      starting from an explicit ``0`` baseline (NOT ``None``) so the KPP caller
      does not gradient-diagnose the non-local salt flux.  The real-salt
      injection is instead the mass-exact floored-h_k source in
      ``mpas_ocean_baroclinic_tendencies`` (adding it to the non-local term too
      would double-count and break partial-cell mass conservation).

    The float-operation ORDER per branch is preserved bit-for-bit from each
    original site.

    Parameters
    ----------
    q_net : surface net heat flux [W/m^2] (>0 into ocean) or ``None``.
    fw : surface freshwater flux [kg/m^2/s] (>0 P-E into ocean) or ``None``.
    salt : real surface salt-mass flux [kg/m^2/s] (>0 brine into ocean) or
        ``None``.
    T_sfc, S_sfc : (...,) — top-layer temperature [degC] and salinity [PSU].
    g, rho_0, c_sw : gravity [m/s^2], Boussinesq reference density [kg/m^3],
        seawater specific heat [J/(kg K)] — passed by the caller from its own
        constant source (``constants_config`` for lat-lon, module constants for
        MPAS; both equal the canonical values by default).
    real_salt_in_qs : see above.

    Returns
    -------
    (B_f, Q_sfc_T, Q_sfc_S) : each ``None`` when its forcing channel is absent
    (``B_f`` is ``None`` only when BOTH heat and freshwater/salt are absent).
    """
    p_sfc = jnp.zeros_like(T_sfc)

    Q_sfc_T = None
    B_f = None
    if q_net is not None:
        Q_sfc_T = q_net / (rho_0 * c_sw)
        alpha = thermal_expansion_coeff(T_sfc, S_sfc, p_sfc)
        B_f = -g * alpha * Q_sfc_T

    Q_sfc_S = None
    if fw is not None or salt is not None:
        beta = haline_contraction_coeff(T_sfc, S_sfc, p_sfc)
        Q_sfc_S = jnp.zeros_like(S_sfc)
        if real_salt_in_qs:
            # Lat-lon: freshwater dilution PLUS real salt both accumulate into
            # one Q_sfc_S; the buoyancy uses that single combined flux.
            if fw is not None:
                Q_sfc_S = Q_sfc_S - S_sfc * fw / rho_0
            if salt is not None:
                Q_sfc_S = Q_sfc_S + salt * 1.0e3 / rho_0
            B_salt = g * beta * Q_sfc_S
        else:
            # MPAS: Q_sfc_S carries the freshwater term only; the real salt
            # adds to the surface buoyancy alone (separate B_salt term).
            B_salt = jnp.zeros_like(S_sfc)
            if fw is not None:
                Q_sfc_S = -S_sfc * jnp.asarray(fw, S_sfc.dtype) / rho_0
                B_salt = B_salt + g * beta * Q_sfc_S
            if salt is not None:
                B_salt = B_salt + g * beta * (
                    jnp.asarray(salt, S_sfc.dtype) * 1.0e3 / rho_0)
        B_f = B_salt if B_f is None else (B_f + B_salt)

    return B_f, Q_sfc_T, Q_sfc_S


def tridiag_thomas(a, b, c, d):
    """Solve a tridiagonal system A x = d via the Thomas algorithm.

    a, b, c, d each have shape ``(..., N)`` and ``a[..., 0]``, ``c[..., -1]``
    are unused (left as zero by the caller). Returns ``x`` of shape ``(..., N)``.
    Shared by the TKE and CATKE backward-Euler vertical solves.
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


def vmap_vertical_diffusion(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    vel_coeff,
    tracer_coeff,
    diffuse_fn: Callable,
    apply_diffusion: bool,
):
    """Apply a per-field vertical-diffusion function to the [u, v] and [T, S]
    pairs via a single vmap each, returning stacked tendencies.

    The velocity pair uses ``vel_coeff`` (viscosity), the tracer pair uses
    ``tracer_coeff`` (diffusivity); ``diffuse_fn(q, coeff)`` returns the
    per-field tendency (it captures z_coord / jacobian / dt). Shared by the
    constant and Richardson schemes, which differ only in ``diffuse_fn`` and the
    coefficients.

    When ``apply_diffusion`` is False, returns zero-tendency stacks (the
    diffusion is deferred to the implicit backward-Euler solve in the dynamics
    step).

    Returns
    -------
    vel_tend, tr_tend : array ``(2, ...)`` each — ``[du_dt, dv_dt]`` and
        ``[dT_dt, dS_dt]``.
    """
    if apply_diffusion:
        vel_tend = jax.vmap(
            lambda q: diffuse_fn(q, vel_coeff), in_axes=0, out_axes=0,
        )(jnp.stack([u, v], axis=0))
        tr_tend = jax.vmap(
            lambda q: diffuse_fn(q, tracer_coeff), in_axes=0, out_axes=0,
        )(jnp.stack([T, S], axis=0))
    else:
        zero_uv = jnp.zeros_like(u)
        vel_tend = jnp.stack([zero_uv, zero_uv], axis=0)
        zero_T = jnp.zeros_like(T)
        tr_tend = jnp.stack([zero_T, zero_T], axis=0)
    return vel_tend, tr_tend
