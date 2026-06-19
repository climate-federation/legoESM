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
    if n2_mode == "insitu":
        dz_safe = jnp.maximum(dz_half, _EPS)
        # drho/dz with z positive upward — negative for stable stratification.
        drho_dz = (rho_cell[..., :-1] - rho_cell[..., 1:]) / dz_safe
        N2 = -g / rho_0 * drho_dz
        return jnp.maximum(N2, 0.0)
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
        f"Unknown n2_mode={n2_mode!r}; expected 'insitu' or 'adiabatic'."
    )


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
