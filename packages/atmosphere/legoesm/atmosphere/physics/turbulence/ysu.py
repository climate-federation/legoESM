"""YSU (Yonsei University) PBL turbulence scheme.

Nonlocal K-profile with entrainment flux at PBL top. The K-profile
follows the same structure as Holtslag-Boville but adds an explicit
entrainment term modeled as a Gaussian envelope centered at the PBL top.

References
----------
- Hong, S.-Y., Noh, Y., & Dudhia, J. (2006). A new vertical diffusion
  package with an explicit treatment of entrainment processes. Mon.
  Wea. Rev., 134, 2318-2341.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import (
    virtual_temperature,
    mixing_length,
    buoyancy_coefficient,
)
from legoesm.atmosphere.physics.turbulence.config import YSUConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
)
from legoesm.atmosphere.physics.turbulence.holtslag_boville import (
    diffuse_theta_with_countergradient,
)


def _crossing_pbl_height(Ri_b, z_full, ricr, sharpness):
    """Lowest height where the bulk Richardson number ``Ri_b`` first reaches
    ``ricr``, by smooth first-crossing interpolation (differentiable).

    Levels are top-first (index ``nlev-1`` = surface).  We scan surface-up and
    select the FIRST pair of adjacent levels whose UPPER member reaches
    ``ricr`` (smooth indicator), then linearly interpolate the crossing height
    inside that pair.  When ``Ri_b`` never reaches ``ricr`` (a fully unstable /
    unbounded column) the height defaults to the model-top level.  Robust to an
    arbitrarily sharp inversion because the crossing is detected per-pair, not
    by sampling a single level at ``Ri_b == ricr``.  Mirrors the
    Holtslag-Boville ``_crossing_height`` structure.
    """
    ncol, nlev = Ri_b.shape
    # Surface-first ordering.
    Ri_s = Ri_b[:, ::-1]
    z_s = z_full[:, ::-1]
    Ri_lo = Ri_s[:, :-1]   # lower (closer to surface) member of each pair
    Ri_hi = Ri_s[:, 1:]
    z_lo = z_s[:, :-1]
    z_hi = z_s[:, 1:]

    # crossed[k] = P(Ri >= ricr at the UPPER level of pair k).
    crossed = jax.nn.sigmoid(sharpness * (Ri_hi - ricr))
    # Exclusive "not yet crossed below pair k".
    not_crossed = jnp.cumprod(
        jnp.concatenate(
            [jnp.ones((ncol, 1), Ri_b.dtype), 1.0 - crossed[:, :-1]], axis=1,
        ),
        axis=1,
    )
    w = not_crossed * crossed  # first-crossing weight per pair

    # Linear interpolation of the crossing height inside the pair.
    dRi = Ri_hi - Ri_lo
    frac = jnp.clip(
        (ricr - Ri_lo) / jnp.where(jnp.abs(dRi) > 1.0e-12, dRi, 1.0e-12),
        0.0, 1.0,
    )
    z_cross = z_lo + frac * (z_hi - z_lo)

    # Fallback (no in-column crossing) -> model-top height.
    z_top = z_full[:, 0]
    fallback_w = jnp.prod(1.0 - crossed, axis=1)
    num = jnp.sum(w * z_cross, axis=1) + fallback_w * z_top
    den = jnp.sum(w, axis=1) + fallback_w + 1.0e-30
    return jnp.clip(num / den, 100.0, None)


def ysu_turbulence(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: YSUConfig,
) -> TurbulenceOutput:
    """Compute turbulence tendencies using YSU PBL scheme.

    Parameters
    ----------
    u, v : jax.Array
        Wind components at full levels [m/s], shape (ncol, nlev).
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    z_full : jax.Array
        Height at full levels [m], shape (ncol, nlev).
    z_half : jax.Array
        Height at half levels [m], shape (ncol, nlev+1).
    T_sfc : jax.Array
        Surface temperature [K], shape (ncol,).
    q_sfc : jax.Array
        Surface saturation mixing ratio [kg/kg], shape (ncol,).
    rho : jax.Array
        Air density at full levels [kg/m^3], shape (ncol, nlev).
    dt : float
        Time step [s].
    config : YSUConfig

    Returns
    -------
    TurbulenceOutput
    """
    ncol, nlev = T.shape

    # ----- Half-level gradients (same as Louis/HB) -----
    dz_half = jnp.abs(z_full[:, :-1] - z_full[:, 1:])  # (ncol, nlev-1)
    dz_half = jnp.clip(dz_half, 1.0, None)

    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2 = du_dz ** 2 + dv_dz ** 2 + 1e-10
    S = jnp.sqrt(S2)

    # Virtual potential temperature
    exner = (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
    theta_v = virtual_temperature(T, q_v) * exner

    theta_v_bar = 0.5 * (theta_v[:, :-1] + theta_v[:, 1:])
    dtheta_v_dz = (theta_v[:, :-1] - theta_v[:, 1:]) / dz_half
    # N² = (g/θ_v)·∂θ_v/∂z (Brunt-Väisälä) via the shared buoyancy coefficient.
    N2 = buoyancy_coefficient(jnp.clip(theta_v_bar, 1.0, None)) * dtheta_v_dz
    Ri = N2 / S2  # (ncol, nlev-1)

    # ----- Surface fluxes -----
    tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface,
    )
    ustar = jnp.clip(ustar, 1e-4, None)  # coeff-ok: u* floor [m/s]

    # ----- PBL height via smooth bulk-Ri (Troen-Mahrt 1986 / Hong et al. 2006)
    # The bulk-Richardson PBL height carries an UNSTABLE surface-excess parcel
    # temperature θ_T = b·(w'θ')_0 / w_s so a convective surface heat flux
    # deepens the diagnosed PBL.  Without θ_T the numerator is purely the
    # resolved θ_v gradient, so a well-mixed (neutral) column pins h_pbl at the
    # floor REGARDLESS of surface heating — the K-profile then normalizes z by
    # the stale h_pbl (z_norm = z/h_pbl, (1-z_norm)² suppression) and stays
    # shallow even as w* grows (codex atm-turb-gwd finding A; Hong et al. 2006
    # eq. for h via Ri_b with the thermal excess).  The excess needs w_s, which
    # needs h_pbl, so we use the standard two-pass: first guess h_pbl with no
    # excess, then refine with θ_T(h_pbl_guess).
    theta_v_sfc = theta_v[:, -1]
    z_sfc = z_full[:, -1:]
    u_sfc = u[:, -1:]
    v_sfc = v[:, -1:]
    dz_from_sfc = jnp.abs(z_full - z_sfc) + 1.0
    dtheta_v_bulk = theta_v - theta_v_sfc[:, None]
    # Wind shear from surface (not absolute wind)
    dV2 = (u - u_sfc) ** 2 + (v - v_sfc) ** 2 + 1e-4  # coeff-ok: wind-shear floor [m^2/s^2]

    # Kinematic surface heat flux and the well-mixed θ_v scale (shared by both
    # passes and the convective velocity scale below).
    wtheta_sfc = shflx / (rho[:, -1] * constants.c_pd)  # kinematic (ncol,)
    theta_bar = jnp.mean(theta_v, axis=1)  # (ncol,)
    g_over_thbar = buoyancy_coefficient(jnp.clip(theta_bar, 1.0, None))

    def _bulk_pbl_height(excess_theta):
        # excess_theta : (ncol,) surface-parcel virtual-θ excess [K], >=0.
        # h_pbl = LOWEST height where the bulk Richardson number first reaches
        # Ri_crit, found by smooth first-crossing interpolation (surface-up).
        # The earlier ``sigma*(1-sigma)`` centre-of-mass weighting is fragile:
        # at a sharp capping inversion the crossing falls BETWEEN two grid
        # levels, so EVERY level saturates the sigmoid (w~0) and the weighted
        # mean collapses to the simple domain-mean height (codex round-2: the
        # excess-parcel fix exposed this, yielding a 7 km PBL above a 2 km
        # inversion).  A first-crossing interpolation is robust to an
        # arbitrarily sharp inversion and is the standard bulk-Ri PBL diagnosis
        # (matches the HB ``_crossing_height`` structure).
        Ri_b = g_over_thbar[:, None] * (
            (dtheta_v_bulk - excess_theta[:, None]) * dz_from_sfc / dV2
        )
        return _crossing_pbl_height(
            Ri_b, z_full, config.Ri_crit, config.pbl_smooth_sharpness,
        )

    # Pass 1: no excess.
    h_pbl_guess = _bulk_pbl_height(jnp.zeros_like(theta_v_sfc))

    # Mixed-layer velocity scale w_s evaluated at the surface layer (z=0.1·h),
    # the standard Troen-Mahrt level for the excess parcel.  w* from the
    # first-guess depth; w_s = (u*³ + c·κ·w*³·0.1)^{1/3}.
    # (g/θ_bar)·(w'θ')_0·h via the shared buoyancy coefficient.
    buoy_guess = (
        buoyancy_coefficient(jnp.clip(theta_bar, 1.0, None))
        * jnp.maximum(wtheta_sfc, 0.0) * h_pbl_guess
    )
    w_star_guess = jnp.cbrt(jnp.maximum(buoy_guess, 1e-20))
    w_s_sfc = jnp.cbrt(
        ustar ** 3 + config.ws_conv_coeff * constants.kappa_vk
        * w_star_guess ** 3 * config.sfc_excess_zfrac
    )
    # Thermal excess θ_T = b·(w'θ')_0 / w_s, only for unstable (kbfs>0).
    excess_theta = config.countergrad_coeff * jnp.maximum(wtheta_sfc, 0.0) / jnp.clip(
        w_s_sfc, 1e-6, None
    )

    # Pass 2: refined PBL height with the unstable surface excess.
    h_pbl = _bulk_pbl_height(excess_theta)

    # ----- Convective velocity scale w* -----
    # w* = (g · h · (w'θ')_sfc / θ_bar)^{1/3}, defined only for unstable
    # (positive surface buoyancy flux).  Used by the mixed-layer velocity
    # scale w_s (below), the PBL-top entrainment flux, and the nonlocal
    # countergradient.  Recomputed with the refined (deeper) h_pbl.
    # w* = ((g/θ_bar)·h·(w'θ')_sfc)^{1/3} via the shared buoyancy coefficient.
    buoyancy_flux = (
        buoyancy_coefficient(jnp.clip(theta_bar, 1.0, None))
        * jnp.maximum(wtheta_sfc, 0.0) * h_pbl
    )
    w_star = jnp.cbrt(jnp.maximum(buoyancy_flux, 1e-20))  # (ncol,)

    # ----- K-profile -----
    z_half_inner = 0.5 * (z_full[:, :-1] + z_full[:, 1:])
    z_norm = z_half_inner / h_pbl[:, None]
    z_norm_clip = jnp.clip(z_norm, 0.0, 1.0)
    # Hong et al. (2006) mixed-layer velocity scale: blends mechanical (u*)
    # and convective (w*) scaling, w_s = (u*³ + c·κ·w*³·z/h)^{1/3}.  The
    # previous code used bare u*, so the convective mixed layer carried NO
    # convective enhancement and was systematically under-mixed.
    w_s = jnp.cbrt(
        ustar[:, None] ** 3
        + config.ws_conv_coeff * constants.kappa_vk
        * w_star[:, None] ** 3 * z_norm_clip
    )
    Km_profile = (
        constants.kappa_vk * w_s * z_half_inner * (1.0 - z_norm_clip) ** 2
    )

    # Local Ri-based Km above PBL: shared Blackadar mixing-length helper
    # (same expression as Louis/TKE/CLUBB-lite/EDMF/Smagorinsky).
    l_mix = mixing_length(z_half_inner, config.l_mix_max)
    # Louis (1982) stability constants come from config; the previous
    # hardcoded ``b_louis = 5.0`` and ``5.0`` literals violated the
    # constant-discipline rule (CLAUDE.md).  Note that Louis (1982)
    # distinguishes three coefficients (b, c, d): ``b`` enters both
    # branches, ``d`` is the stable-branch sqrt coefficient, and ``c``
    # is the unstable-branch denominator coefficient — the repository's
    # louis.py already follows this split, and YSU now does too.
    b_louis = config.louis_b
    c_louis = config.louis_c
    d_louis = config.louis_d
    Ri_pos = jnp.maximum(Ri, 0.0)
    f_stable = 1.0 / (1.0 + 2.0 * b_louis * Ri_pos / jnp.sqrt(1.0 + d_louis * Ri_pos))
    Ri_neg = jnp.minimum(Ri, 0.0)
    f_unstable = 1.0 - 2.0 * b_louis * Ri_neg / (
        1.0 + 3.0 * b_louis * c_louis * l_mix ** 2
        * jnp.sqrt(jnp.abs(Ri_neg) + 1e-10) / (dz_half ** 2 + 1e-10)
    )
    blend_ri = jax.nn.sigmoid(config.blend_ri_sharpness * Ri)
    f_m = (1.0 - blend_ri) * f_unstable + blend_ri * f_stable
    Km_local = l_mix ** 2 * S * f_m

    # Smooth blend from K-profile to local
    blend_pbl = jax.nn.sigmoid(config.blend_pbl_sharpness * (z_norm - 1.0))

    # ----- Entrainment flux at PBL top -----
    # (w*, wtheta_sfc, theta_bar computed above with the K-profile.)
    # Gaussian envelope for entrainment: K_ent = c_ent * w* * h * exp(-((z-h)/(f*h))^2)
    # Width fraction f (default 0.3) is used for robustness at GCM-typical
    # vertical resolution; exposed as config.entrainment_width_frac.
    width = config.entrainment_width_frac * h_pbl[:, None]  # (ncol, 1)
    K_ent = (
        config.entrainment_coeff * w_star[:, None] * h_pbl[:, None]
        * jnp.exp(-((z_half_inner - h_pbl[:, None]) / jnp.clip(width, 1.0, None)) ** 2)
    )

    # Combined: profile inside PBL, local above, entrainment added everywhere
    # (entrainment Gaussian is self-localizing around PBL top)
    Km_half = (1.0 - blend_pbl) * Km_profile + blend_pbl * Km_local + K_ent
    Kh_half = Km_half / config.Pr_t

    # Interpolate to full levels for diagnostics — single concat per
    # field instead of the previous ``zeros + 3 .at[].set`` triple
    # scatter (lowers to one HLO op).
    Km_interior = 0.5 * (Km_half[:, :-1] + Km_half[:, 1:])
    Km_full = jnp.concatenate(
        [Km_half[:, :1], Km_interior, Km_half[:, -1:]], axis=1,
    )
    Kh_interior = 0.5 * (Kh_half[:, :-1] + Kh_half[:, 1:])
    Kh_full = jnp.concatenate(
        [Kh_half[:, :1], Kh_interior, Kh_half[:, -1:]], axis=1,
    )

    # Layer thicknesses
    dz_layer = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    dz_layer = jnp.clip(dz_layer, 1.0, None)

    sflx_u = tau_x
    sflx_v = tau_y
    sflx_T = shflx / constants.c_pd
    sflx_q = lhflx / constants.L_v

    # ----- Nonlocal countergradient (Troen-Mahrt 1986 / Hong et al. 2006) -----
    # γ_c = b·(w'θ')_0 / (w_*·h)  [K/m] (Troen & Mahrt 1986; Hong et al. 2006),
    # YSU's defining nonlocal upward heat transport in the convective BL.  Gated
    # to unstable surface forcing via max(w'θ', 0) (zero for neutral/stable),
    # with the convective velocity scale w* in the denominator (⇒ the usual
    # γ_c ∝ (w'θ')^{2/3} convective scaling).  The previous YSU lumped the
    # entire countergradient into the lowest-level surface BC (an enhanced
    # surface flux with a single column-mean Kh), depositing all the nonlocal
    # heating in the surface-adjacent layer instead of distributing it through
    # the mixed layer.  Per Troen-Mahrt/Hong the nonlocal term enters the heat
    # equation as the modified flux F = −Kh·(∂θ/∂z − γ_c) whose vertical
    # divergence redistributes heat across the BL — apply it that way via the
    # shared HB countergradient diffusion, with γ_c as a half-level field gated
    # to inside the PBL (1 − blend_pbl ≈ 1 below h_pbl, ≈ 0 above).
    counter_grad = config.countergrad_coeff * jnp.maximum(wtheta_sfc, 0.0) / (
        jnp.clip(w_star, 1e-6, None) * h_pbl
    )  # (ncol,) [K/m]
    gamma_theta_half = counter_grad[:, None] * (1.0 - blend_pbl)  # (ncol, nlev-1) [K/m]

    # Implicit vertical diffusion.  Heat in θ-space (dry-adiabat neutral),
    # with the nonlocal countergradient applied as a distributed flux
    # divergence (shared with Holtslag-Boville).
    u_new = implicit_vertical_diffusion(u, Km_half, rho, dz_layer, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, Km_half, rho, dz_layer, dz_half, dt, sflx_v)
    T_new = diffuse_theta_with_countergradient(
        T, Kh_half, rho, dz_layer, dz_half, p_full, dt, sflx_T, gamma_theta_half,
    )
    q_new = implicit_vertical_diffusion(q_v, Kh_half, rho, dz_layer, dz_half, dt, sflx_q)

    return TurbulenceOutput(
        du_dt=(u_new - u) / dt,
        dv_dt=(v_new - v) / dt,
        dT_dt=(T_new - T) / dt,
        dq_v_dt=(q_new - q_v) / dt,
        Km=Km_full,
        Kh=Kh_full,
        shflx=shflx,
        lhflx=lhflx,
        ustar=ustar,
        h_pbl=h_pbl,
    )
