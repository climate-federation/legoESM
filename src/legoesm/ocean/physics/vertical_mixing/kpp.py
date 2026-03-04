"""K-Profile Parameterization (KPP).

Large, McWilliams & Doney (1994): Oceanic vertical mixing —
A review and a model with a nonlocal boundary layer parameterization.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.ocean.eos import (
    wright_eos,
    compute_buoyancy_frequency,
    compute_hydrostatic_pressure,
    rho_0 as rho_0_ref,
)
from legoesm.ocean.physics.mixing import vertical_diffusion_variable_K
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.physics.vertical_mixing.output import VerticalMixingOutput
from legoesm.ocean.vertical import OceanZStarCoordinate


def _boundary_layer_depth(
    rho: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: KPPConfig,
    g: float = 9.80616,
) -> jnp.ndarray:
    """Estimate boundary layer depth h via bulk Richardson number.

    Ri_b(k) = g * delta_rho * |z(k)| / (rho_0 * (delta_V^2 + V_t^2))
    BL depth is where Ri_b first exceeds Ri_crit.

    Returns shape (...) boundary layer depth [m, positive].
    """
    eps = 1e-12

    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    # Depth of cell centers below surface (positive downward)
    z_depth = jnp.cumsum(dz_actual, axis=-1) - 0.5 * dz_actual

    # Density and velocity differences from surface
    delta_rho = rho - rho[..., :1]
    delta_u = u - u[..., :1]
    delta_v = v - v[..., :1]
    delta_V2 = delta_u**2 + delta_v**2

    # Unresolved shear: V_t^2 = Cv * sqrt(|N2|) * h (approx with z_depth)
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)
    # Extend N2 to full levels by padding surface with first interface value
    N2_full = jnp.concatenate([N2[..., :1], N2], axis=-1)
    V_t2 = cfg.Cv * jnp.sqrt(jnp.maximum(jnp.abs(N2_full), 0.0)) * z_depth

    # Bulk Richardson number
    Ri_b = (g * delta_rho * z_depth) / (
        rho_0_ref * jnp.maximum(delta_V2 + V_t2, eps)
    )

    # Find first level where Ri_b > Ri_crit
    exceeds = Ri_b > cfg.Ri_crit
    # Use argmax to find first True; if none found, use bottom
    nlev = rho.shape[-1]
    idx = jnp.argmax(exceeds, axis=-1)
    # If no level exceeds, set to bottom
    any_exceeds = jnp.any(exceeds, axis=-1)
    idx = jnp.where(any_exceeds, idx, nlev - 1)

    # Boundary layer depth: z_depth at that index
    # Gather using advanced indexing
    shape = rho.shape[:-1]
    flat_idx = idx.ravel()
    z_depth_flat = z_depth.reshape(-1, nlev)
    h = z_depth_flat[jnp.arange(z_depth_flat.shape[0]), flat_idx]
    h = h.reshape(shape)
    h = jnp.maximum(h, dz_actual[..., 0])  # At least one layer

    return h


def kpp_vertical_mixing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    rho: jnp.ndarray,
    eta: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    cfg: KPPConfig,
    g: float = 9.80616,
) -> VerticalMixingOutput:
    """Apply KPP vertical mixing.

    Parameters
    ----------
    u, v : array (6, n, n, nlev)
    T, S : array (6, n, n, nlev)
    rho : array (6, n, n, nlev)
    eta : array (6, n, n)
    z_coord : OceanZStarCoordinate
    jacobian : array (6, n, n)
    cfg : KPPConfig
    g : float

    Returns
    -------
    VerticalMixingOutput
    """
    eps = 1e-12
    nlev = u.shape[-1]

    # --- Boundary layer depth ---
    h_bl = _boundary_layer_depth(rho, u, v, z_coord, jacobian, cfg, g)

    # --- Depth coordinate ---
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    z_depth = jnp.cumsum(dz_actual, axis=-1) - 0.5 * dz_actual
    # sigma = z_depth / h_bl: normalized depth within BL
    sigma = z_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)

    # --- Shape function G(sigma) = sigma * (1 - sigma)^2 ---
    sigma_clip = jnp.clip(sigma, 0.0, 1.0)
    G = sigma_clip * (1.0 - sigma_clip) ** 2

    # --- Friction velocity (simple estimate from surface wind stress) ---
    # Use surface shear as proxy: u* = sqrt(A_bg * |du/dz|_surface)
    # Simplified: u_star ~ max(|u_surface|, 0.01) * 0.01
    speed_sfc = jnp.sqrt(u[..., 0]**2 + v[..., 0]**2 + eps)
    u_star = jnp.maximum(speed_sfc * 0.01, 1e-4)

    # --- BL diffusivity at full levels ---
    K_bl_full = h_bl[..., jnp.newaxis] * cfg.kappa_vk * u_star[..., jnp.newaxis] * G
    K_bl_full = jnp.minimum(K_bl_full, cfg.K_max)

    # --- Interior: Richardson-number mixing below BL ---
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    du = u[..., :-1] - u[..., 1:]
    dv = v[..., :-1] - v[..., 1:]
    S2 = (du**2 + dv**2) / jnp.maximum(dz_half**2, eps)
    Ri = jnp.maximum(N2 / jnp.maximum(S2, eps), 0.0)
    K_interior = cfg.K_bg / (1.0 + 5.0 * Ri) ** 2 + cfg.K_bg

    # --- K at interfaces (average of full level K_bl) ---
    K_bl_half = 0.5 * (K_bl_full[..., :-1] + K_bl_full[..., 1:])

    # sigma at interfaces
    z_half_depth = 0.5 * (z_depth[..., :-1] + z_depth[..., 1:])
    sigma_half = z_half_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)
    in_bl = sigma_half < 1.0

    # Combine BL and interior
    K_v = jnp.where(in_bl, K_bl_half, K_interior) + cfg.K_bg
    A_v = jnp.where(in_bl, K_bl_half * 1.0, K_interior) + cfg.A_bg
    K_v = jnp.minimum(K_v, cfg.K_max)
    A_v = jnp.minimum(A_v, cfg.K_max)

    # --- Apply diffusion ---
    vel = jnp.stack([u, v], axis=0)
    vel_tend = jax.vmap(
        lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, A_v),
        in_axes=0, out_axes=0,
    )(vel)

    tracers = jnp.stack([T, S], axis=0)
    tr_tend = jax.vmap(
        lambda q: vertical_diffusion_variable_K(q, z_coord, jacobian, K_v),
        in_axes=0, out_axes=0,
    )(tracers)

    # --- Non-local flux for T, S within boundary layer ---
    # gamma * Q_s / (h * c_sw * rho_0), approximated by surface buoyancy flux
    # Use surface T gradient as proxy for heat flux
    dT_dz_sfc = (T[..., 0] - T[..., 1]) / jnp.maximum(dz_half[..., 0], eps)
    Q_proxy = cfg.K_bg * dT_dz_sfc  # [K·m/s]
    nonlocal_T = cfg.gamma_T * Q_proxy[..., jnp.newaxis] / jnp.maximum(
        h_bl[..., jnp.newaxis], eps
    )
    # Apply only within BL
    in_bl_full = sigma < 1.0
    dT_nonlocal = jnp.where(in_bl_full, nonlocal_T, 0.0)

    dS_dz_sfc = (S[..., 0] - S[..., 1]) / jnp.maximum(dz_half[..., 0], eps)
    S_proxy = cfg.K_bg * dS_dz_sfc
    nonlocal_S = cfg.gamma_S * S_proxy[..., jnp.newaxis] / jnp.maximum(
        h_bl[..., jnp.newaxis], eps
    )
    dS_nonlocal = jnp.where(in_bl_full, nonlocal_S, 0.0)

    return VerticalMixingOutput(
        du_dt=vel_tend[0],
        dv_dt=vel_tend[1],
        dT_dt=tr_tend[0] + dT_nonlocal,
        dS_dt=tr_tend[1] + dS_nonlocal,
        K_v=K_v,
        A_v=A_v,
    )
