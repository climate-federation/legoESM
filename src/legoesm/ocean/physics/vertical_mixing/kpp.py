"""LMD94-style K-Profile Parameterization (KPP).

Boundary-layer parameterization following Large, McWilliams & Doney (1994)
with:

- Bulk Richardson number BL-depth diagnosis with linear interpolation
  of the crossing depth between model levels.
- Turbulent velocity scales w_s(sigma) from surface forcing (u_star, B_f).
- Cubic shape function G(sigma) = sigma * (1 - sigma)^2.
- Non-local tracer transport for unstable (convective) conditions only.
- Interior mixing: Richardson-number dependent + convective instability
  enhancement for statically unstable layers below the BL.

The caller should provide surface wind stress and buoyancy flux when
available.  If tau_x/tau_y are None, a simplified u_star proxy from
surface speed is used.

References
----------
- Large, W. G., McWilliams, J. C., & Doney, S. C. (1994). Oceanic
  vertical mixing: A review and a model with a nonlocal boundary layer
  parameterization. Rev. Geophys., 32, 363-403.
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
    u_star: jnp.ndarray,
    B_f: jnp.ndarray,
    cfg: KPPConfig,
    g: float = 9.80616,
) -> jnp.ndarray:
    """Estimate boundary layer depth h via bulk Richardson number.

    Uses linear interpolation to find the depth where Ri_b crosses
    Ri_crit, rather than snapping to the nearest model level.

    Returns shape (...) boundary layer depth [m, positive downward].
    """
    eps = 1e-12
    nlev = rho.shape[-1]

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
    N2_full = jnp.concatenate([N2[..., :1], N2], axis=-1)
    V_t2 = cfg.Cv * jnp.sqrt(jnp.maximum(jnp.abs(N2_full), 0.0)) * z_depth

    # Bulk Richardson number
    Ri_b = (g * delta_rho * z_depth) / (
        rho_0_ref * jnp.maximum(delta_V2 + V_t2, eps)
    )

    # --- Differentiable soft interpolation of crossing depth ---
    # Instead of argmax (non-differentiable), use a sigmoid-weighted
    # average over all levels.  Each level contributes a weight
    # proportional to how much Ri_b crosses Ri_crit there.
    #
    # Weight at level k = sigmoid(sharpness * (Ri_b[k] - Ri_crit))
    #                    - sigmoid(sharpness * (Ri_b[k-1] - Ri_crit))
    # This is ~1 at the crossing level and ~0 elsewhere.
    sharpness = 20.0
    sig = jax.nn.sigmoid(sharpness * (Ri_b - cfg.Ri_crit))  # (..., nlev)

    # Crossing weight: difference of adjacent sigmoid values
    sig_prev = jnp.concatenate([jnp.zeros_like(sig[..., :1]), sig[..., :-1]], axis=-1)
    w_cross = sig - sig_prev  # (..., nlev), peaks at crossing level
    w_cross = jnp.maximum(w_cross, 0.0)
    w_sum = jnp.sum(w_cross, axis=-1, keepdims=True)
    w_norm = w_cross / jnp.maximum(w_sum, eps)

    # Weighted average depth gives the BL depth estimate
    h = jnp.sum(w_norm * z_depth, axis=-1)  # (...)

    # At least one layer thick
    h = jnp.maximum(h, dz_actual[..., 0])

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
    tau_x: jnp.ndarray | None = None,
    tau_y: jnp.ndarray | None = None,
    B_f: jnp.ndarray | None = None,
) -> VerticalMixingOutput:
    """Apply LMD94-style KPP vertical mixing.

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
    tau_x, tau_y : array (6, n, n) or None
        Surface wind stress [Pa]. If None, a proxy from surface speed is used.
    B_f : array (6, n, n) or None
        Surface buoyancy flux [m^2/s^3], positive = destabilizing (convective).
        If None, estimated from surface density gradient.

    Returns
    -------
    VerticalMixingOutput
    """
    eps = 1e-12
    nlev = u.shape[-1]

    # --- Friction velocity ---
    if tau_x is not None and tau_y is not None:
        # Proper u_star from wind stress: u_star = sqrt(|tau| / rho_0)
        tau_mag = jnp.sqrt(tau_x**2 + tau_y**2 + eps)
        u_star = jnp.sqrt(tau_mag / rho_0_ref)
    else:
        # Simplified proxy: u_star ~ 0.01 * |U_surface|
        speed_sfc = jnp.sqrt(u[..., 0]**2 + v[..., 0]**2 + eps)
        u_star = jnp.maximum(speed_sfc * 0.01, 1e-4)

    # --- Surface buoyancy flux ---
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    dz_half0 = 0.5 * (dz_actual[..., 0] + dz_actual[..., 1])
    if B_f is None:
        # Estimate from near-surface density gradient
        drho_dz_sfc = (rho[..., 0] - rho[..., 1]) / jnp.maximum(dz_half0, eps)
        B_f = -g / rho_0_ref * cfg.K_bg * drho_dz_sfc  # simplified proxy

    # --- Boundary layer depth ---
    h_bl = _boundary_layer_depth(rho, u, v, z_coord, jacobian, u_star, B_f, cfg, g)

    # --- Depth coordinate ---
    z_depth = jnp.cumsum(dz_actual, axis=-1) - 0.5 * dz_actual
    sigma = z_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)

    # --- Shape function G(sigma) = sigma * (1 - sigma)^2 ---
    sigma_clip = jnp.clip(sigma, 0.0, 1.0)
    G = sigma_clip * (1.0 - sigma_clip) ** 2

    # --- Turbulent velocity scale w_s(sigma) (LMD94 Appendix B) ---
    # w_s depends on stability (B_f) and depth d = sigma * h_bl
    d = sigma_clip * h_bl[..., jnp.newaxis]
    # Monin-Obukhov length: L_MO = u_star^3 / (kappa * B_f)
    L_MO = u_star[..., jnp.newaxis]**3 / (
        cfg.kappa_vk * jnp.where(jnp.abs(B_f[..., jnp.newaxis]) > eps,
                                  B_f[..., jnp.newaxis], eps)
    )
    zeta_kpp = d / L_MO

    # Unstable (B_f > 0 or zeta < 0): w_s = kappa * u_star * (1 - c_s * zeta)^p
    # Stable (B_f <= 0): w_s = kappa * u_star / (1 + 5*zeta)
    is_unstable = B_f[..., jnp.newaxis] > 0.0
    w_s_unstable = (cfg.kappa_vk * u_star[..., jnp.newaxis]
                    * jnp.power(jnp.maximum(1.0 + 16.0 * jnp.abs(zeta_kpp), 1.0), 0.25))
    w_s_stable = (cfg.kappa_vk * u_star[..., jnp.newaxis]
                  / jnp.maximum(1.0 + 5.0 * jnp.maximum(zeta_kpp, 0.0), 1.0))
    w_s = jnp.where(is_unstable, w_s_unstable, w_s_stable)
    w_s = jnp.maximum(w_s, 1e-10)

    # --- BL diffusivity at full levels ---
    K_bl_full = h_bl[..., jnp.newaxis] * w_s * G
    K_bl_full = jnp.minimum(K_bl_full, cfg.K_max)

    # --- Interior mixing: Richardson-number dependent ---
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    du = u[..., :-1] - u[..., 1:]
    dv = v[..., :-1] - v[..., 1:]
    S2 = (du**2 + dv**2) / jnp.maximum(dz_half**2, eps)
    Ri_int = N2 / jnp.maximum(S2, eps)
    K_interior = cfg.K_bg / (1.0 + 5.0 * jnp.maximum(Ri_int, 0.0)) ** 2 + cfg.K_bg

    # Interior static instability: enhanced mixing where N2 < 0
    K_conv = jnp.where(N2 < cfg.Ri_conv, cfg.K_conv, 0.0)
    K_interior = K_interior + K_conv

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

    # --- Non-local flux for T, S (LMD94 Eq. 19) ---
    #
    # LMD94 defines a counter-gradient term:
    #   gamma_T(sigma) = C_s * Q_0 / (w_s(sigma) * h)   [K/m]
    # where Q_0 is the surface kinematic heat flux [K*m/s].
    #
    # The non-local tendency is  -d/dz(K_bl * gamma_T).
    # Substituting K_bl = h * w_s * G(sigma):
    #   K_bl * gamma_T = h * w_s * G * C_s * Q_0 / (w_s * h) = C_s * Q_0 * G(sigma)
    #
    # So the non-local tendency reduces to:
    #   dT/dt_nonlocal = -d/dz[ C_s * Q_0 * G(sigma) ]            [K/s]
    #
    # We discretize this as the vertical divergence of the non-local
    # flux F_nl = C_s * Q_0 * G(sigma) evaluated at interfaces.

    # Surface kinematic heat flux proxy: Q_0 = K_sfc * dT/dz  [K*m/s]
    # Use the BL diffusivity at the surface as the flux velocity scale.
    dT_dz_sfc = (T[..., 0] - T[..., 1]) / jnp.maximum(dz_half[..., 0], eps)
    K_sfc = K_bl_full[..., 0]  # BL diffusivity at surface level [m^2/s]
    Q_T = K_sfc * dT_dz_sfc    # [K*m/s]

    dS_dz_sfc = (S[..., 0] - S[..., 1]) / jnp.maximum(dz_half[..., 0], eps)
    Q_S = K_sfc * dS_dz_sfc    # [psu*m/s]

    # Only apply non-local transport for unstable (convective) columns.
    is_unstable_col = B_f > 0.0

    # G(sigma) at interior interfaces (half levels between full levels)
    sigma_half_full = z_half_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)
    sigma_half_clip = jnp.clip(sigma_half_full, 0.0, 1.0)
    G_half = sigma_half_clip * (1.0 - sigma_half_clip) ** 2  # (..., nlev-1)

    in_bl_full = sigma < 1.0

    # --- Temperature non-local tendency ---
    # Non-local flux at interfaces: F_nl = C_s * Q_T * G_half  [K*m/s]
    F_T = cfg.gamma_T * Q_T[..., jnp.newaxis] * G_half  # (..., nlev-1)
    # Tendency = -dF/dz at full levels (zero-flux BCs at surface and bottom)
    dT_nonlocal_top = -F_T[..., :1] / dz_actual[..., :1]
    dT_nonlocal_int = (F_T[..., :-1] - F_T[..., 1:]) / dz_actual[..., 1:-1]
    dT_nonlocal_bot = F_T[..., -1:] / dz_actual[..., -1:]
    dT_nonlocal = jnp.concatenate(
        [dT_nonlocal_top, dT_nonlocal_int, dT_nonlocal_bot], axis=-1
    )  # (..., nlev)  [K/s]
    dT_nonlocal = jnp.where(
        in_bl_full & is_unstable_col[..., jnp.newaxis], dT_nonlocal, 0.0
    )

    # --- Salinity non-local tendency ---
    F_S = cfg.gamma_S * Q_S[..., jnp.newaxis] * G_half  # (..., nlev-1)
    dS_nonlocal_top = -F_S[..., :1] / dz_actual[..., :1]
    dS_nonlocal_int = (F_S[..., :-1] - F_S[..., 1:]) / dz_actual[..., 1:-1]
    dS_nonlocal_bot = F_S[..., -1:] / dz_actual[..., -1:]
    dS_nonlocal = jnp.concatenate(
        [dS_nonlocal_top, dS_nonlocal_int, dS_nonlocal_bot], axis=-1
    )  # (..., nlev)  [psu/s]
    dS_nonlocal = jnp.where(
        in_bl_full & is_unstable_col[..., jnp.newaxis], dS_nonlocal, 0.0
    )

    return VerticalMixingOutput(
        du_dt=vel_tend[0],
        dv_dt=vel_tend[1],
        dT_dt=tr_tend[0] + dT_nonlocal,
        dS_dt=tr_tend[1] + dS_nonlocal,
        K_v=K_v,
        A_v=A_v,
    )
