"""QUICKEST and upwind fluxes for spectral wave transport.

Direct port of WADEN 46aadc7 ``src/waden/flux.py``.  The TVD limiter
bounds and the land-boundary fallback are reproduced verbatim.

The stencil padding mode selects neighbours only; ghost cells are
refreshed separately before each step.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

__physics_contract__ = {
    "summary": (
        "Third-order QUICKEST advection with a TVD limiter plus the WW3 "
        "land-boundary fallback, and first-order upwind for the learned "
        "refraction velocities."
    ),
    "inputs": {
        "N": "spectral density times the metric factor, (B,Nlon,Nlat,Ndir,Nfreq)",
        "Cg": "transport velocity times dt [m], or the direction/wavenumber velocity",
        "Dgrid": "grid spacing [m] along the transport axis",
        "land_mask": "True where sea",
    },
    "outputs": {
        "FLN_minus": "flux at the low interface",
        "FLN_plus": "flux at the high interface",
    },
    "sign_convention": (
        "FLN_minus - FLN_plus divided by Dgrid is the tendency; positive Cg "
        "transports toward increasing index."),
    "conserves": ["tracer"],
    "differentiable": True,
    "reference": "WADEN 46aadc7 src/waden/flux.py; WW3 manual v6.07 §3.3.",
    "idealized_test": (
        "tests/wave/unit/test_waden_contracts.py::"
        "test_thin_island_uses_incident_sea_speed"),
}


# QUICKEST/TVD limiter bounds (Leonard 1979; WADEN flux.py).
_TVD_MIN = -0.1
_TVD_MAX = 1.1


def _normalize_mode(mode: str) -> str:
    if mode == "circular":
        return "wrap"
    if mode == "replicate":
        return "edge"
    raise ValueError(f"Unsupported padding mode: {mode}")


def _expand_trailing_dims(array: jax.Array, ndim: int) -> jax.Array:
    while array.ndim < ndim:
        array = jnp.expand_dims(array, axis=-1)
    return array


def _broadcast_like(array: jax.Array, reference: jax.Array) -> jax.Array:
    array = _expand_trailing_dims(array, reference.ndim)
    return jnp.broadcast_to(array, reference.shape)


def get_stencils(
    tensor: jax.Array,
    target_dim: int,
    pad_left: int = 2,
    pad_right: int = 1,
    mode: str = "circular",
) -> Tuple[jax.Array, ...]:
    if target_dim < 0:
        target_dim = tensor.ndim + target_dim
    pad_width = [(0, 0)] * tensor.ndim
    pad_width[target_dim] = (pad_left, pad_right)
    padded = jnp.pad(tensor, pad_width, mode=_normalize_mode(mode))
    original_size = tensor.shape[target_dim]
    return tuple(
        jax.lax.slice_in_dim(
            padded,
            start_index=offset,
            limit_index=offset + original_size,
            axis=target_dim,
        )
        for offset in range(pad_left + pad_right + 1)
    )


def first_order_flux(
    N: jax.Array,
    Cg: jax.Array,
    land_mask: jax.Array,
    target_dim: int = 1,
    mode: Optional[str] = None,
) -> Tuple[jax.Array, jax.Array]:
    """First-order upwind flux with the WW3 land-to-sea fallback."""
    if mode is None:
        mode = "circular" if target_dim in [1] else "replicate"
    N_im1, N_i = get_stencils(N, target_dim, pad_left=1, pad_right=0, mode=mode)
    Cg_i, Cg_ip1 = get_stencils(Cg, target_dim, pad_left=0, pad_right=1, mode=mode)

    land_mask = land_mask.astype(jnp.float32)
    lm_i, lm_ip1 = get_stencils(land_mask, target_dim, pad_left=0, pad_right=1, mode=mode)
    land2sea_mask = _broadcast_like((lm_i == 0) & (lm_ip1 == 1), Cg_i)

    Cg_eff = jnp.where(land2sea_mask, Cg_ip1, Cg_i)
    Cg_eff_im1 = get_stencils(Cg_eff, target_dim, pad_left=1, pad_right=0, mode=mode)[0]

    Cg_b = 0.5 * (Cg_eff_im1 + Cg_eff)
    N_upwind = jnp.where(Cg_b >= 0, N_im1, N_i)
    FLN_minus = Cg_b * N_upwind
    FLN_plus = get_stencils(FLN_minus, target_dim, pad_left=0, pad_right=1, mode=mode)[1]
    return FLN_minus, FLN_plus


def _quickest_tvd_limiter(
    N: jax.Array,
    Cg_b: jax.Array,
    Nb: jax.Array,
    CFL: jax.Array,
    target_dim: int,
    mode: str,
    eps: float,
    N_stencils: Optional[Tuple[jax.Array, ...]] = None,
) -> jax.Array:
    if N_stencils is None:
        N_stencils = get_stencils(N, target_dim, pad_left=2, pad_right=1, mode=mode)
    N_im2, N_im1, N_i, N_ip1 = N_stencils

    is_positive = Cg_b >= 0
    N_c = jnp.where(is_positive, N_im1, N_i)
    N_u = jnp.where(is_positive, N_im2, N_ip1)
    N_d = jnp.where(is_positive, N_i, N_im1)

    DQ = N_d - N_u
    DQNZ = jnp.sign(DQ) * jnp.clip(jnp.abs(DQ), min=eps)
    DQNZ = jnp.where(DQ == 0, -eps, DQNZ)

    Nc_tilde = jnp.clip((N_c - N_u) / DQNZ, _TVD_MIN, _TVD_MAX)
    Nb_tilde = jnp.maximum((Nb - N_u) / DQNZ, Nc_tilde)
    CFL = jnp.clip(jnp.abs(CFL), min=eps)

    Nclimit = Nc_tilde / CFL
    Nb_tilde = jnp.minimum(Nb_tilde, jnp.clip(Nclimit, max=1.0))

    Nclimit2 = ((N_c / CFL) - N_u) / DQNZ
    Nc_tilde = jnp.where(
        DQ < 0,
        jnp.maximum(Nc_tilde, Nclimit2),
        jnp.minimum(Nc_tilde, Nclimit2),
    )
    Nc_rev = N_u + Nc_tilde * DQ
    Nb_rev = N_u + Nb_tilde * DQ

    CFAC = jnp.clip(jnp.floor(2.0 * jnp.abs(Nc_tilde - 0.5)), 0.0, 1.0)
    N_c = jnp.where(CFL > 1.0, Nc_rev, N_c)
    return (1.0 - CFAC) * Nb_rev + CFAC * N_c


def _theta_tvd_limiter(
    N: jax.Array,
    Cg_b: jax.Array,
    Nb: jax.Array,
    CFL: jax.Array,
    target_dim: int,
    mode: str,
    eps: float,
    N_stencils: Optional[Tuple[jax.Array, ...]] = None,
) -> jax.Array:
    if N_stencils is None:
        N_stencils = get_stencils(N, target_dim, pad_left=2, pad_right=1, mode=mode)
    N_im2, N_im1, N_i, N_ip1 = N_stencils

    is_positive = Cg_b >= 0
    N_c = jnp.where(is_positive, N_im1, N_i)
    N_u = jnp.where(is_positive, N_im2, N_ip1)
    N_d = jnp.where(is_positive, N_i, N_im1)

    DQ = N_d - N_u
    DQNZ = jnp.sign(DQ) * jnp.clip(jnp.abs(DQ), min=eps)
    DQNZ = jnp.where(DQ == 0, -eps, DQNZ)

    Nc_tilde = (N_c - N_u) / DQNZ
    Nb_tilde = jnp.maximum(
        (Nb - N_u) / DQNZ, jnp.clip(Nc_tilde, min=_TVD_MIN, max=_TVD_MAX))
    CFL = jnp.clip(jnp.abs(CFL), min=eps)

    Nclimit = ((N_c / CFL) - N_u) / DQNZ
    Nb_tilde = jnp.minimum(Nb_tilde, jnp.clip(Nclimit, max=1.0))
    Nb_tilde = jnp.where(
        DQ < 0,
        jnp.maximum(Nb_tilde, Nclimit),
        jnp.minimum(Nb_tilde, Nclimit),
    )
    Nc_tilde = jnp.where(
        DQ < 0,
        jnp.maximum(Nc_tilde, Nclimit),
        jnp.minimum(Nc_tilde, Nclimit),
    )

    Nc_rev = N_u + Nc_tilde * DQ
    Nb_rev = N_u + Nb_tilde * DQ
    CFAC = jnp.clip(jnp.floor(2.0 * jnp.abs(Nc_tilde - 0.5)), 0.0, 1.0)
    return (1.0 - CFAC) * Nb_rev + CFAC * Nc_rev


def _handle_land_boundaries(
    N: jax.Array,
    Cg: jax.Array,
    Cg_b: jax.Array,
    Nb: jax.Array,
    land_mask: jax.Array,
    target_dim: int,
    mode: str,
    N_stencils: Optional[Tuple[jax.Array, jax.Array, jax.Array, jax.Array]] = None,
    Cg_stencils: Optional[Tuple[jax.Array, jax.Array]] = None,
    boundary_mask: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    if N_stencils is None:
        N_im1, N_i = get_stencils(N, target_dim, pad_left=1, pad_right=0, mode=mode)
    else:
        _, N_im1, N_i, _ = N_stencils

    wet_left, wet_right = get_stencils(
        _expand_trailing_dims(land_mask.astype(bool), N.ndim),
        target_dim, 1, 0, mode)
    if Cg_stencils is None:
        speed_left, speed_right = get_stencils(Cg, target_dim, 1, 0, mode)
    else:
        speed_left, speed_right = Cg_stencils
    # WW3 shoreline: sea-cell speed, zero land inflow, absorbed outgoing waves.
    speed = jnp.where(~wet_left & wet_right, speed_right,
                      jnp.where(wet_left & ~wet_right, speed_left, Cg_b))
    donor = jnp.where(speed >= 0, jnp.where(wet_left, N_im1, 0),
                      jnp.where(wet_right, N_i, 0))
    boundary_face = ~wet_left | ~wet_right
    if boundary_mask is not None:
        prescribed = _broadcast_like(boundary_mask, N)
        fixed_left, fixed_right = get_stencils(prescribed, target_dim, 1, 0, mode)
        boundary_face = boundary_face | fixed_left | fixed_right
    FLN_minus = jnp.where(boundary_face, speed * donor, Cg_b * Nb)
    FLN_plus = get_stencils(FLN_minus, target_dim, pad_left=0, pad_right=1, mode=mode)[1]
    return FLN_minus, FLN_plus



def quickest_flux(
    N: jax.Array,
    Cg: jax.Array,
    Dgrid: jax.Array,
    target_dim: int = 1,
    land_mask: Optional[jax.Array] = None,
    dt: float = 1.0,
    eps: float = 1e-9,
    mode: Optional[str] = None,
    boundary_mask: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Third-order QUICKEST flux with the TVD limiter."""
    if mode is None:
        mode = "circular" if target_dim in [1, 3] else "replicate"

    N_im2, N_im1, N_i, N_ip1 = get_stencils(N, target_dim, pad_left=2, pad_right=1, mode=mode)
    Cg_im1, Cg_i, _ = get_stencils(Cg, target_dim, pad_left=1, pad_right=1, mode=mode)
    N_stencils = (N_im2, N_im1, N_i, N_ip1)

    Dgrid_expanded = _broadcast_like(Dgrid, N_i)
    safe_Dgrid = jnp.clip(Dgrid_expanded, min=eps)
    Cg_b = 0.5 * (Cg_im1 + Cg_i)
    CFL = (Cg_b * dt) / safe_Dgrid

    is_positive = Cg_b >= 0
    CU_positive = (N_im2 - 2.0 * N_im1 + N_i) / (safe_Dgrid**2)
    CU_negative = (N_im1 - 2.0 * N_i + N_ip1) / (safe_Dgrid**2)
    CU = jnp.where(is_positive, CU_positive, CU_negative)

    Nb = 0.5 * ((1 + CFL) * N_im1 + (1 - CFL) * N_i)
    Nb = Nb - (1 - CFL**2) / 6.0 * CU * (Dgrid_expanded**2)
    Nb = _quickest_tvd_limiter(N, Cg_b, Nb, CFL, target_dim, mode, eps, N_stencils=N_stencils)

    if land_mask is not None:
        return _handle_land_boundaries(
            N, Cg, Cg_b, Nb, land_mask, target_dim, mode,
            N_stencils=N_stencils, Cg_stencils=(Cg_im1, Cg_i),
            boundary_mask=boundary_mask,
        )

    FLN_minus = Cg_b * Nb
    FLN_plus = get_stencils(FLN_minus, target_dim, pad_left=0, pad_right=1, mode=mode)[1]
    return FLN_minus, FLN_plus


def theta_flux(
    N: jax.Array,
    Cg: jax.Array,
    Dgrid: jax.Array,
    target_dim: int = 3,
    land_mask: Optional[jax.Array] = None,
    dt: float = 1.0,
    eps: float = 1e-9,
) -> Tuple[jax.Array, jax.Array]:
    """Directional flux with the theta-specific limiter."""
    mode = "circular" if target_dim in [1, 3] else "replicate"

    N_im2, N_im1, N_i, N_ip1 = get_stencils(N, target_dim, pad_left=2, pad_right=1, mode=mode)
    Cg_im1, Cg_i, _ = get_stencils(Cg, target_dim, pad_left=1, pad_right=1, mode=mode)
    N_stencils = (N_im2, N_im1, N_i, N_ip1)

    Dgrid_expanded = _broadcast_like(Dgrid, N_i)
    safe_Dgrid = jnp.clip(Dgrid_expanded, min=eps)
    Cg_b = 0.5 * (Cg_im1 + Cg_i)
    CFL = (jnp.abs(Cg_b) * dt) / safe_Dgrid

    is_positive = Cg_b >= 0
    CU_positive = (N_im2 - 2.0 * N_im1 + N_i) / (safe_Dgrid**2)
    CU_negative = (N_im1 - 2.0 * N_i + N_ip1) / (safe_Dgrid**2)
    CU = jnp.where(is_positive, CU_positive, CU_negative)

    Nb = 0.5 * ((1 + CFL) * N_im1 + (1 - CFL) * N_i)
    Nb = Nb - (1 - CFL**2) / 6.0 * CU * (Dgrid_expanded**2)
    Nb = _theta_tvd_limiter(N, Cg_b, Nb, CFL, target_dim, mode, eps, N_stencils=N_stencils)

    if land_mask is not None:
        return _handle_land_boundaries(
            N, Cg, Cg_b, Nb, land_mask, target_dim, mode,
            N_stencils=N_stencils, Cg_stencils=(Cg_im1, Cg_i),
        )

    FLN_minus = Cg_b * Nb
    FLN_plus = get_stencils(FLN_minus, target_dim, pad_left=0, pad_right=1, mode=mode)[1]
    return FLN_minus, FLN_plus


def calculate_flux(
    N: jax.Array,
    Cg: jax.Array,
    scheme: str = "first_order",
    target_dim: int = 1,
    **kwargs,
) -> Tuple[jax.Array, jax.Array]:
    """Flux factory; unknown schemes raise."""
    scheme = scheme.lower()
    if scheme == "first_order":
        return first_order_flux(N=N, Cg=Cg, target_dim=target_dim, **kwargs)
    if scheme == "quickest":
        return quickest_flux(N=N, Cg=Cg, target_dim=target_dim, **kwargs)
    if scheme == "theta":
        return theta_flux(N=N, Cg=Cg, target_dim=target_dim, **kwargs)
    raise ValueError(f"Unknown flux scheme: {scheme}")
