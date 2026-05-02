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

from legoesm import constants
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

_EPS = float(jnp.finfo(jnp.float32).eps)  # Float32 machine epsilon (~1.19e-7)


def _boundary_layer_depth(
    rho: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    u_star: jnp.ndarray,
    B_f: jnp.ndarray,
    cfg: KPPConfig,
    g: float = constants.g,
    h_bl_prev: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Estimate boundary layer depth h via bulk Richardson number.

    Uses linear interpolation to find the depth where Ri_b crosses
    Ri_crit, rather than snapping to the nearest model level.

    Returns shape (...) boundary layer depth [m, positive downward].
    """
    eps = _EPS
    nlev = rho.shape[-1]

    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    # Depth of cell centers below surface (positive downward)
    z_depth = jnp.cumsum(dz_actual, axis=-1) - 0.5 * dz_actual

    # Density and velocity differences from surface
    delta_rho = rho - rho[..., :1]
    delta_u = u - u[..., :1]
    delta_v = v - v[..., :1]
    delta_V2 = delta_u**2 + delta_v**2

    # LMD94 Eq. 23: V_t^2 = Cv * sqrt(|N2|) / sqrt(c_s * epsilon) *
    #   max(Ri_crit * h - d, 0) * d / h
    # Uses h_bl from the previous time step to break the coupling.
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, jacobian)
    N2_full = jnp.concatenate([N2[..., :1], N2], axis=-1)
    max_depth = z_depth[..., -1]
    h_est = max_depth if h_bl_prev is None else h_bl_prev
    h_safe = jnp.maximum(h_est[..., jnp.newaxis], eps)
    V_t2 = (cfg.Cv * jnp.sqrt(jnp.maximum(jnp.abs(N2_full), 0.0))
            / jnp.sqrt(jnp.maximum(cfg.c_s * cfg.epsilon_lmd, eps))
            * jnp.maximum(cfg.Ri_crit * h_safe - z_depth, 0.0)
            * z_depth / h_safe)

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
    sharpness = cfg.crossing_sharpness
    sig = jax.nn.sigmoid(sharpness * (Ri_b - cfg.Ri_crit))  # (..., nlev)

    # Crossing weight: difference of adjacent sigmoid values.  ``jnp.pad``
    # along the trailing axis is one HLO op; the previous
    # ``concatenate([zeros_like(sig[..., :1]), sig[..., :-1]])`` allocated
    # a fresh zero buffer and concatenated.
    pad_axes = ((0, 0),) * (sig.ndim - 1)
    sig_prev = jnp.pad(sig[..., :-1], (*pad_axes, (1, 0)))
    w_cross = sig - sig_prev  # (..., nlev), peaks at crossing level
    w_cross = jnp.maximum(w_cross, 0.0)
    w_sum = jnp.sum(w_cross, axis=-1, keepdims=True)
    w_norm = w_cross / jnp.maximum(w_sum, eps)

    # Crossing-based depth estimate
    h_crossing = jnp.sum(w_norm * z_depth, axis=-1)  # (...)

    # Fallback for columns where Ri_b never crosses Ri_crit:
    # - If column is mostly unstable (sig ≈ 0): BL extends to full depth
    # - If column is mostly stable (sig ≈ 1): BL is one layer
    column_stability = jnp.mean(sig, axis=-1)  # 0 = all unstable, 1 = all stable
    max_depth = z_depth[..., -1]
    min_depth = dz_actual[..., 0]
    h_fallback = (1.0 - column_stability) * max_depth + column_stability * min_depth

    # Blend: use crossing depth when crossing signal is strong, fallback otherwise
    crossing_strength = w_sum[..., 0]
    blend = jax.nn.sigmoid(cfg.crossing_sharpness * (crossing_strength - cfg.crossing_threshold))
    h = blend * h_crossing + (1.0 - blend) * h_fallback

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
    g: float = constants.g,
    tau_x: jnp.ndarray | None = None,
    tau_y: jnp.ndarray | None = None,
    B_f: jnp.ndarray | None = None,
    Q_sfc_T: jnp.ndarray | None = None,
    Q_sfc_S: jnp.ndarray | None = None,
    h_bl_prev: jnp.ndarray | None = None,
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
    Q_sfc_T : array (6, n, n) or None
        Surface kinematic heat flux [K*m/s] for non-local transport (LMD94
        Eq. 19).  If None, falls back to diagnosed K_sfc * dT/dz proxy.
    Q_sfc_S : array (6, n, n) or None
        Surface kinematic salt flux [PSU*m/s]. Same convention as Q_sfc_T.
    h_bl_prev : array (6, n, n) or None
        BL depth from the previous time step [m, positive downward].
        Used to break the implicit V_t-h_bl coupling in the Ri_b diagnosis
        (LMD94 Eq. 23).  If None, uses the full column depth as estimate.

    Returns
    -------
    VerticalMixingOutput
    """
    eps = _EPS
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
    h_bl = _boundary_layer_depth(
        rho, u, v, z_coord, jacobian, u_star, B_f, cfg, g,
        h_bl_prev=h_bl_prev,
    )

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
    # Use copysign(eps, B_f) to preserve the sign of B_f near zero,
    # preventing a stability classification flip (issue #168 bug 1).
    B_f_safe = jnp.where(
        jnp.abs(B_f[..., jnp.newaxis]) > eps,
        B_f[..., jnp.newaxis],
        jnp.copysign(eps, B_f[..., jnp.newaxis]),
    )
    L_MO = u_star[..., jnp.newaxis]**3 / (cfg.kappa_vk * B_f_safe)
    zeta_kpp = d / L_MO

    # LMD94 Appendix B turbulent velocity scales:
    # Stable (B_f <= 0): w_s = kappa * u_star / (1 + 5*zeta)
    # Unstable, weakly (epsilon*d < |L|): w_s = kappa * u_star * phi_m^{-1}
    #   where phi_m^{-1} = (1 - 16*zeta)^{1/4}
    # Unstable, strongly convective (epsilon*d > |L|):
    #   w_s = (kappa * (u_star^3 + c_b * kappa * (-B_f) * d))^{1/3}
    is_unstable = B_f[..., jnp.newaxis] > 0.0
    epsilon_lmd = cfg.epsilon_lmd

    # Weakly unstable: phi_m^{-1} formulation
    w_s_weak = (cfg.kappa_vk * u_star[..., jnp.newaxis]
                * jnp.power(jnp.maximum(1.0 + 16.0 * jnp.abs(zeta_kpp), 1.0), 0.25))

    # Strongly convective: includes convective velocity scale
    Bf_pos = jnp.maximum(B_f[..., jnp.newaxis], 0.0)
    w_s_conv = jnp.power(
        cfg.kappa_vk * (u_star[..., jnp.newaxis]**3
                        + cfg.c_b * cfg.kappa_vk * Bf_pos * d),
        1.0 / 3.0,
    )

    # Transition: use convective scale when epsilon*d > |L_MO|
    is_strongly_convective = epsilon_lmd * d > jnp.abs(L_MO)
    w_s_unstable = jnp.where(is_strongly_convective, w_s_conv, w_s_weak)

    # Stable: standard suppression
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
    # LMD94 interior shear instability: K = K_0 * (1 - (Ri/Ri_0)^2)^3
    # for Ri < Ri_0, zero above.
    Ri_ratio = jnp.clip(Ri_int / cfg.Ri_0, 0.0, 1.0)
    K_interior = cfg.K_0_shear * (1.0 - Ri_ratio**2) ** 3 + cfg.K_bg

    # Interior static instability: enhanced mixing where N2 < 0
    K_conv = jnp.where(N2 < cfg.Ri_conv, cfg.K_conv, 0.0)
    K_interior = K_interior + K_conv

    # --- K at interfaces (average of full level K_bl) ---
    K_bl_half = 0.5 * (K_bl_full[..., :-1] + K_bl_full[..., 1:])

    # sigma at interfaces
    z_half_depth = 0.5 * (z_depth[..., :-1] + z_depth[..., 1:])
    sigma_half = z_half_depth / jnp.maximum(h_bl[..., jnp.newaxis], eps)
    in_bl = sigma_half < 1.0

    # Combine BL and interior.
    # K_interior already includes ``+ cfg.K_bg`` at line ~282; the BL
    # branch needs the floor added explicitly so the merged field is
    # consistent.  Adding ``+ cfg.K_bg`` *after* the where would
    # double-count the floor on the interior branch.
    K_v = jnp.where(in_bl, K_bl_half + cfg.K_bg, K_interior)
    A_v = jnp.where(in_bl, K_bl_half + cfg.A_bg, K_interior)
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

    # Surface kinematic heat/salt flux for non-local transport (LMD94 Eq. 19).
    # Use the IMPOSED surface flux when available (from bulk formulas or
    # prescribed forcing).  Fall back to diagnosed K_sfc * dT/dz proxy
    # only when no external flux is provided (issue #168 bug 2).
    if Q_sfc_T is not None:
        Q_T = Q_sfc_T  # [K*m/s]
    else:
        dT_dz_sfc = (T[..., 0] - T[..., 1]) / jnp.maximum(dz_half[..., 0], eps)
        K_sfc = K_bl_full[..., 0]
        Q_T = K_sfc * dT_dz_sfc

    if Q_sfc_S is not None:
        Q_S = Q_sfc_S  # [PSU*m/s]
    else:
        dS_dz_sfc = (S[..., 0] - S[..., 1]) / jnp.maximum(dz_half[..., 0], eps)
        K_sfc = K_bl_full[..., 0]
        Q_S = K_sfc * dS_dz_sfc

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
