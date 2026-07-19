"""YSU (Yonsei University) PBL turbulence scheme — differentiable variant.

Nonlocal K-profile with an explicit entrainment flux at the PBL top.
The K-profile follows the same structure as Holtslag-Boville but adds a
free-convective-LIMIT surrogate for the Hong et al. (2006) PBL-top
entrainment: a prescribed heat flux ``(w'θ')_h = −e_ratio·(w'θ')_0`` (a fixed
fraction of the SURFACE flux, dropping Hong06's shear-dependent ``w_m³`` term),
applied as a flux-matched diffusivity localized at the inversion (see the
entrainment block in :func:`ysu_turbulence` and the Faithfulness section below).

.. note::

   **Implementation note vs published YSU.** Published YSU (Hong et al.
   2006) prescribes the PBL-top entrainment as an explicit,
   gradient-independent inversion-flux source term (not a boundary
   condition), scaled by the entrainment velocity ``w_m³ = w*³ + 5·u*³`` (so it
   carries a mechanical ``u*``/shear contribution).  This module makes
   TWO simplifications: (i) the entrainment flux is taken as a fixed
   fraction of the SURFACE flux, ``w'θ'_h ≈ −0.15·w'θ'_0`` — the
   FREE-CONVECTIVE limit, dropping the shear term; and (ii) because the
   module builds K profiles and integrates them with implicit vertical
   diffusion, that prescribed flux enters as a FLUX-MATCHED eddy
   diffusivity
   ``K_h,ent = e_ratio·(w'θ')_0 / max(∂θ_v/∂z, floor)`` under a
   Gaussian envelope centred on ``h_pbl`` (width
   ``config.entrainment_width_frac``): the resulting down-gradient flux
   ``−K_ent·∂θ_v/∂z`` at the inversion reproduces the prescribed
   ``(w'θ')_h`` exactly where the inversion gradient exceeds the floor,
   and saturates (stability-capped) in weakly stratified interfaces.
   Chosen as a smooth, fully differentiable closure (no prescribed-
   inversion-flux branch) at GCM-typical vertical resolution.

Faithfulness to Hong et al. (2006) / Troen-Mahrt (1986)
-------------------------------------------------------
FAITHFUL (published forms + constants):
  * **Nonlocal K-profile** ``K_m = κ·w_s·z·(1 − z/h)²`` (Troen-Mahrt 1986 /
    Hong06), whose shape ``(z/h)(1 − z/h)²`` peaks at exactly ``4/27`` at
    ``z = h/3`` (``_KPROFILE_PEAK_FRAC``, used as the entrainment-K stability cap).
  * **Mixed-layer velocity scale** ``w_s = (u*³ + c·κ·w*³·z/h)^{1/3}`` blending
    mechanical and convective scaling (Hong06), with the convective velocity
    ``w* = ((g/θ)·h·(w'θ')_0)^{1/3}``.
  * **Bulk-Richardson PBL height** with the Troen-Mahrt thermal-excess parcel
    ``θ_T = b·(w'θ')_0/w_s`` — an unstable surface heat flux deepens ``h_pbl``.
  * **Local-Ri free-atmosphere mixing** above the PBL (a genuine YSU component) and
    the entrainment COEFFICIENT ``e_ratio = 0.15`` (the Hong06 free-convective
    entrainment value).  Sign: unstable ``(w'θ')_0 > 0`` ⇒ downward entrainment flux
    (``K_ent ≥ 0``).
DEPARTURES / SURROGATES:
  * **Fixed surface-flux-ratio entrainment**: the implemented
    ``(w'θ')_h = −e_ratio·(w'θ')_0`` is only the FREE-CONVECTIVE LIMIT of Hong06.
    The published scheme scales the entrainment velocity by ``w_m³ = w*³ + 5·u*³``,
    so its true PBL-top flux carries a mechanical (``u*``/shear) contribution that
    this shear-independent fixed ratio DROPS.  Only the 0.15 coefficient is the
    published value; the fixed-fraction-of-surface-flux LAW is a surrogate.
  * **Flux-matched K realization** of that prescribed flux (see the note above):
    ``K_ent = e_ratio·(w'θ')_0/max(∂θ_v/∂z, floor)`` under a Gaussian envelope at
    ``h_pbl`` — a smooth, fully differentiable stand-in for the prescribed
    inversion flux, NOT that inversion-flux source term itself.
  * **Entrainment stability cap** ``K_ent ≤ (4/27)·κ·w_s(h)·h`` and a gradient floor
    ``∂θ_v/∂z ≥ 1e-4`` — guards (not tunables) that break the ``e_ratio`` linearity
    once the flux-matched K would exceed the peak mixed-layer K.
  * **Local-K coefficients + smooth blend**: the above-PBL local-Ri mixing is
    faithful YSU, but its specific Louis (1982) coefficients (shared
    ``_shared.louis_stability_functions``) and the sigmoid PBL/local blend are a
    re-tuned, smoothed closure.
  * **Smooth switches**: sigmoid PBL/local blend and a smooth first-crossing PBL-top
    detection replace hard index searches, for differentiability.
Non-behavioral pins: ``tests/atmosphere/hydrostatic/unit/test_ysu_faithful.py``.

References
----------
- Hong, S.-Y., Noh, Y., & Dudhia, J. (2006). A new vertical diffusion
  package with an explicit treatment of entrainment processes. Mon.
  Wea. Rev., 134, 2318-2341.
- Troen, I., & Mahrt, L. (1986). A simple model of the atmospheric
  boundary layer; sensitivity to surface evaporation. Boundary-Layer
  Meteorol., 37, 129-148.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import (
    virtual_temperature,
    mixing_length,
    buoyancy_coefficient,
    exner_function,
    louis_stability_functions,
)
from legoesm.atmosphere.physics.turbulence.config import YSUConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import (
    first_crossing_height,
)
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
)
from legoesm.atmosphere.physics.turbulence.holtslag_boville import (
    diffuse_theta_with_countergradient,
)


__physics_contract__ = {
    "summary": (
        "YSU (Yonsei University) nonlocal K-profile PBL scheme: a convective "
        "mixed-layer K-profile with a countergradient nonlocal heat flux and "
        "a free-convective-limit PBL-top entrainment flux (w'th')_h = "
        "-e_ratio*(w'th')_0 (a surrogate for Hong06's shear-dependent w_m "
        "entrainment) applied as a flux-matched K at the inversion, "
        "all applied by implicit vertical diffusion with surface-flux BCs."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K", "q_v": "kg/kg",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "T_sfc": "K", "q_sfc": "kg/kg", "rho": "kg/m^3", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "Km": "m^2/s", "Kh": "m^2/s", "shflx": "W/m^2", "lhflx": "W/m^2",
        "ustar": "m/s", "h_pbl": "m",
    },
    "sign_convention": (
        "Nonlocal down-gradient mixing: heat flux F = -Kh (dtheta/dz - "
        "gamma_c) with countergradient gamma_c >= 0 (upward heat transport in "
        "the convective BL, gated to unstable surface forcing). Km, Kh >= 0; "
        "an unstable surface heat flux (shflx > 0 upward) deepens h_pbl via a "
        "thermal-excess parcel. The prescribed (free-convective-limit) PBL-top "
        "entrainment flux (w'th')_h = -e_ratio*(w'th')_0 is NEGATIVE (downward) "
        "for unstable "
        "surface forcing — warm air entrained down across the inversion, "
        "warming the PBL top region — and is realized down-gradient via a "
        "flux-matched K_ent >= 0 at the inversion. shflx, lhflx positive "
        "UPWARD and injected as the lower boundary condition (a source), so "
        "the column is NOT closed. z increases upward; level index 0 is the "
        "top, -1 the surface."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Hong, Noh & Dudhia (2006), Mon. Wea. Rev. 134, 2318-2341; "
        "Troen & Mahrt (1986), Boundary-Layer Meteorol. 37, 129-148"
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_atm_turb_gwd_validation.py "
        "and test_turbulence.py: an unstable surface heat flux deepens h_pbl "
        "(two-pass thermal excess); Km, Kh >= 0 with entrainment K localized "
        "at the PBL top; the PBL top is found by smooth first-crossing "
        "interpolation (differentiable, no hard index search)."
    ),
}


# --- PBL-top entrainment guards (Hong et al. 2006 flux-matched K) ----------
# Floor [K/m] on the inversion theta_v gradient in
# K_h,ent = e_ratio*(w'th')_0 / max(dtheta_v/dz, floor): below it (a well-mixed
# or unstably-stratified interface at GCM vertical resolution) the flux-matched
# K saturates and the cap below takes over.  Numerics guard, not a tunable.
_ENTRAIN_DTHDZ_FLOOR = 1.0e-4
# Cap fraction for the entrainment diffusivity: the interior K-profile shape
# (z/h)*(1 - z/h)^2 peaks at exactly 4/27 (at z = h/3, exact math), so
# kappa*w_s(h)*h*(4/27) is the peak mixed-layer diffusivity magnitude —
# the entrainment K may not exceed it (stability guard, not a tunable).
_KPROFILE_PEAK_FRAC = 4.0 / 27.0


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

    # Virtual potential temperature via the canonical inverse-Exner helper
    # (exner_pref = 1/Π = (p_ref/p)^κ; the helper applies the same 1 Pa floor).
    exner_pref = 1.0 / exner_function(p_full)
    theta_v = virtual_temperature(T, q_v) * exner_pref

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
        # (the shared ``first_crossing_height`` kernel, also used by
        # Holtslag-Boville).  The 100 m floor is this scheme's h_pbl floor.
        Ri_b = g_over_thbar[:, None] * (
            (dtheta_v_bulk - excess_theta[:, None]) * dz_from_sfc / dV2
        )
        return jnp.clip(
            first_crossing_height(
                Ri_b, z_full, config.Ri_crit, config.pbl_smooth_sharpness,
            ),
            100.0, None,
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
    # Louis (1982) stability functions via the SHARED helper
    # (physics._shared.louis_stability_functions — identical b/c/d coefficient
    # split and sigmoid stable/unstable blend as louis.py; YSU consumes only
    # the momentum function f_m).  Coefficients config.louis_b /
    # config.louis_c / config.louis_d and config.blend_ri_sharpness.
    f_m, _ = louis_stability_functions(
        Ri, l_mix, dz_half,
        config.louis_b, config.louis_c, config.louis_d,
        config.blend_ri_sharpness,
    )
    Km_local = l_mix ** 2 * S * f_m

    # Smooth blend from K-profile to local
    blend_pbl = jax.nn.sigmoid(config.blend_pbl_sharpness * (z_norm - 1.0))

    # ----- Entrainment flux at PBL top (Hong et al. 2006, explicit closure) --
    # YSU's defining innovation over Holtslag-Boville: a PRESCRIBED entrainment
    # heat flux at the inversion, tied to the surface flux
    #     (w'θ')_h = −e_ratio · (w'θ')_0,   e_ratio = config.entrainment_ratio.
    # Sign convention (flux positive-UP, z up): unstable surface forcing
    # (w'θ')_0 > 0 gives (w'θ')_h < 0 — the entrainment flux at h is DOWNWARD
    # (warm free-tropospheric air entrained down across the inversion, warming
    # the PBL-top region).  This module builds K profiles and then runs
    # implicit vertical diffusion, so the prescribed flux enters as a
    # FLUX-MATCHED diffusivity at the PBL-top interfaces:
    #     K_h,ent = −(w'θ')_h / max(∂θ_v/∂z, floor)
    #             =  e_ratio·(w'θ')_0 / max(∂θ_v/∂z, floor)  ≥ 0,
    # so the down-gradient flux −K_h,ent·∂θ_v/∂z at the inversion (where
    # ∂θ_v/∂z ≫ floor) equals the prescribed (w'θ')_h EXACTLY — pinned to the
    # surface flux, independent of the resolved gradient (the previous Gaussian
    # K_ent = c·w*·h blob produced an ordinary down-gradient flux whose
    # magnitude scaled with the local gradient, NOT Hong06's ratio closure).
    # A Gaussian envelope centered at h (width config.entrainment_width_frac·h)
    # localizes the entrainment K to the inversion; moisture entrains
    # down-gradient with the same K (Hong06 applies w_e to all scalars).
    width = config.entrainment_width_frac * h_pbl[:, None]  # (ncol, 1)
    envelope = jnp.exp(
        -((z_half_inner - h_pbl[:, None]) / jnp.clip(width, 1.0, None)) ** 2
    )
    # Inversion θ_v gradient, floored (see _ENTRAIN_DTHDZ_FLOOR): below the
    # floor the flux-matched K saturates and the stability cap takes over.
    dthdz_inv = jnp.maximum(dtheta_v_dz, _ENTRAIN_DTHDZ_FLOOR)
    K_ent_h = (
        config.entrainment_ratio
        * jnp.maximum(wtheta_sfc, 0.0)[:, None] / dthdz_inv
    )
    # Stability cap: entrainment K may not exceed the peak mixed-layer
    # K-profile magnitude κ·w_s(h)·h·(4/27) (see _KPROFILE_PEAK_FRAC), with
    # w_s(h) the mixed-layer velocity scale evaluated at z = h.
    w_s_h = jnp.cbrt(
        ustar ** 3 + config.ws_conv_coeff * constants.kappa_vk * w_star ** 3
    )
    K_ent_cap = (
        _KPROFILE_PEAK_FRAC * constants.kappa_vk * w_s_h * h_pbl
    )[:, None]
    K_ent_h = jnp.minimum(K_ent_h, K_ent_cap) * envelope

    # Combined: profile inside PBL, local above, entrainment added everywhere
    # (the flux-matched entrainment K is self-localizing around the PBL top).
    # Momentum analog per Hong06 (Prandtl number at h, here the scheme Pr_t):
    # Km carries Pr_t·K_ent_h so heat receives exactly the flux-matched
    # K_ent_h after the Kh = Km/Pr_t division.
    Km_half = (
        (1.0 - blend_pbl) * Km_profile + blend_pbl * Km_local
        + config.Pr_t * K_ent_h
    )
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
    # γ_c = b·(w'θ')_0 / (w_s0·h)  [K/m] (Troen & Mahrt 1986; Hong et al.
    # 2006), YSU's defining nonlocal upward heat transport in the convective
    # BL.  w_s0 is the MIXED-LAYER velocity scale evaluated at the top of the
    # surface layer, w_s0 = w_s(0.1h) = (u*³ + c·κ·w*³·0.1)^{1/3} — the SAME
    # ``w_s_sfc`` already used for the surface thermal-excess parcel
    # θ_T = b·(w'θ')_0/w_s0 above, so γ_c = θ_T/h BY CONSTRUCTION (the papers
    # tie both to w_s0; ``excess_theta`` carries the same clip floor).  The
    # previous code divided by the PURE convective w*, which overestimates γ_c
    # in windy, weakly-convective columns (u* ≫ w*, where w_s0 → u* but
    # w* → 0) and breaks the γ_c = θ_T/h identity.  Gated to unstable surface
    # forcing via max(w'θ', 0) inside θ_T (zero for neutral/stable).  The
    # previous YSU lumped the entire countergradient into the lowest-level
    # surface BC (an enhanced surface flux with a single column-mean Kh),
    # depositing all the nonlocal heating in the surface-adjacent layer
    # instead of distributing it through the mixed layer.  Per Troen-Mahrt/
    # Hong the nonlocal term enters the heat equation as the modified flux
    # F = −Kh·(∂θ/∂z − γ_c) whose vertical divergence redistributes heat
    # across the BL — apply it that way via the shared HB countergradient
    # diffusion, with γ_c as a half-level field gated to inside the PBL
    # (1 − blend_pbl ≈ 1 below h_pbl, ≈ 0 above).
    counter_grad = excess_theta / h_pbl  # (ncol,) [K/m] = b·(w'θ')_0/(w_s0·h)
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
