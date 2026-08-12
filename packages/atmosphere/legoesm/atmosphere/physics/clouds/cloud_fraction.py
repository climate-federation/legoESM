"""Diagnostic cloud fraction and cloud optical property computation.

Implements two cloud fraction schemes:

1. **Sundqvist, Berge & Kristjansson (1989)**: RH-based, simple and robust.
   The sqrt-form ``(1 − b)² = (1 − RH)/(1 − RH_crit)``, i.e.
   ``cf = 1 − sqrt((1 − RH)/(1 − RH_crit))`` for RH ≥ RH_crit, else 0
   (clamped to [0, 1]).  See ``sundqvist_cloud_fraction`` for the
   AD-safe double-``where`` implementation.

2. **Xu-Randall (1996)**: RH + condensate-based, more physical.
   ``cf = RH^p * [1 - exp(-alpha * q_c / ((1 - RH) * q_s)^gamma)]``

Both schemes compute cloud fraction per column per layer and derive
cloud liquid/ice water paths for RRTMGP cloud optics.

Faithfulness to Xu-Randall (1996) / Sundqvist-Berge-Kristjansson (1989)
----------------------------------------------------------------------
FAITHFUL (forms + published constants):
  * **Xu-Randall (1996)** cloud fraction is the paper's semiempirical Eq. (6)
    ``cf = RH^p · [1 − exp(−α·q_c / ((1−RH)·q_sat)^γ)]`` with the PAPER-RECOMMENDED
    constants ``α = 100``, ``p = 0.25``, ``γ = 0.49`` (``alpha_xr``/``p_xr``/
    ``gamma_xr`` config defaults). ``q_c`` is the total cloud condensate.
  * **Sundqvist-Berge-Kristjansson (1989)** cloud fraction is the √-form
    ``cf = 1 − √((1−RH)/(1−RH_crit))`` for RH ≥ RH_crit (else 0), as used by ECHAM
    — NOT a linear RH ramp (an earlier mislabeled linear form was replaced).
DEPARTURES / SURROGATES:
  * **AD guards** (Xu-Randall): the denominator base ``(1−RH)·q_sat`` is floored at
    1e-10 BEFORE the fractional power γ<1, and the ``RH`` base of ``RH^p`` (p<1) is
    clipped to [1e-6, 1], so the otherwise-infinite fractional-power DERIVATIVES at
    0 cannot leak an inf/NaN reverse-mode cotangent. For physical inputs (q_c ≥ 0,
    q_sat > 0) the forward value equals the bare Eq. (6) wherever NEITHER guard
    alters its argument (RH ≥ 1e-6 AND (1−RH)·q_sat ≥ 1e-10); in a guarded cell it
    CAN differ from bare Eq. (6), and the diagnosed cf → 0 as q_c → 0 (or, at fixed
    RH < 1, as q_sat → ∞).
  * **Boundary enforcement**: both cloud fractions are clipped to [0, 1]. For
    Sundqvist this clip IS the ``RH < RH_crit → 0`` branch; for Xu-Randall the clip
    is redundant for physical inputs (q_c ≥ 0, q_sat > 0 ⇒ both factors in [0, 1] ⇒
    cf ∈ [0, 1]; the upper end is reached when exp underflows to 0 at saturation).
  * **Model choices** (not Xu-Randall/Sundqvist forms): ``rh_crit = 0.77`` is a
    TUNED critical RH (the cloud-fraction diagnostic value, distinct from the
    microphysics ``SundqvistConfig.rh_crit``), and the temperature ice-fraction
    split is a linear ramp (T_freeze → T_ice_only).
Non-behavioral pins: ``tests/atmosphere/hydrostatic/unit/test_xu_randall_faithful.py``.

References
----------
- Sundqvist, H. (1988). Parameterization of condensation and
  associated clouds in models for weather prediction and general
  circulation simulation. *Physically-Based Modelling and Simulation
  of Climate and Climatic Change*, NATO ASI Series, 243, 433-461.
- Xu, K.-M. & Randall, D. A. (1996). A semiempirical cloudiness
  parameterization for use in climate models. *J. Atmos. Sci.*,
  53, 3084-3102.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax, nn

from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice
from legoesm import constants

# Machine-checked scheme contract (see tests/test_physics_contracts.py). This is
# a DIAGNOSTIC (state -> cloud fraction + optical properties), not a tendency
# scheme, so it conserves nothing.
__physics_contract__ = {
    "summary": (
        "Diagnostic cloud fraction and cloud optical properties (grid-mean "
        "liquid/ice water paths + effective radii) from the column state: "
        "RH-based Sundqvist, RH+condensate Xu-Randall, or resolved-condensate "
        "schemes, with an optional Slingo convective (cumulus) cover."
    ),
    "inputs": {
        "T": "K", "p_full": "Pa", "q_v": "kg/kg", "dp": "Pa",
        "q_cloud": "kg/kg", "q_ice": "kg/kg",
        "n_cloud": "1/m^3 (droplet number, per volume)",
        "n_ice": "1/kg (ice number, per mass)",
        "conv_precip": "kg/m^2/s",
    },
    "outputs": {
        "cloud_fraction": "1 (0-1 area fraction)",
        "lwp": "kg/m^2 (grid-mean liquid water path per layer)",
        "iwp": "kg/m^2 (grid-mean ice water path per layer)",
        "r_eff_liq": "m", "r_eff_ice": "m",
    },
    "sign_convention": (
        "Diagnostic only (no state tendency): cloud_fraction in [0,1], "
        "monotonically non-decreasing in RH (0 below rh_crit, 1 at RH>=1); "
        "lwp, iwp >= 0; effective radii > 0; ice fraction ramps 0->1 as T "
        "drops from T_freeze to T_ice_only."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Sundqvist (1988), NATO ASI Ser. 243, 433-461; Xu & Randall (1996), "
        "JAS 53, 3084-3102; Slingo (1987) convective cloud cover"
    ),
    "idealized_test": (
        "tests/unit/test_resolved_cloud_fraction.py; sub-saturated column "
        "(RH<rh_crit) -> cf=0, RH>=1 -> cf=1, cf monotone in RH; lwp,iwp>=0; "
        "unknown scheme raises ValueError"
    ),
}

# Cloud-optics defaults (fixed): effective-radius bounds.
_CLOUD_R_EFF_MAX_M = 60.0e-6     # max liquid effective radius for lamc clip [m]
_R_EFF_ICE_PSD_COEFF = 1.5       # ice effective-radius PSD coefficient
_R_EFF_ICE_DEFAULT_M = 25.0e-6   # fallback ice effective radius [m]
# --- two_region sub-grid cloud-optics inhomogeneity ---
_INHOM_CF_FLOOR = 1.0e-3         # min cloud fraction for the in-cloud water path
_INHOM_R_EFF_FLOOR_M = 1.0e-6    # min effective radius in the tau estimate [m]
_TAU_GEOMETRIC_COEFF = 1.5       # 3 Q_ext/4 with Q_ext≈2 (geometric-optics extinction)



class CloudProperties(NamedTuple):
    """Cloud properties for radiation coupling.

    All arrays have shape (ncol, nlev).

    Fields
    ------
    cloud_fraction : jnp.ndarray
        Cloud fraction per layer [0, 1].
    lwp : jnp.ndarray
        Grid-mean liquid water path per layer [kg/m^2].
    iwp : jnp.ndarray
        Grid-mean ice water path per layer [kg/m^2].
    r_eff_liq : jnp.ndarray
        Effective radius for liquid droplets [m].
    r_eff_ice : jnp.ndarray
        Effective radius for ice crystals [m].
    """
    cloud_fraction: jnp.ndarray
    lwp: jnp.ndarray
    iwp: jnp.ndarray
    r_eff_liq: jnp.ndarray
    r_eff_ice: jnp.ndarray
    # LONGWAVE radiative water paths [kg/m^2] (``None`` == same as lwp/iwp).
    # Populated ONLY by ``cloud_partial_coverage_optics="two_column"``: the
    # coverage inversion is nonlinear and DIFFERENT in reflectance (SW) and
    # emissivity (LW) space, so a single scaled path cannot serve both
    # streams — the SW-space factor applied to the LW path over-trapped OLR
    # by ~10 W/m2 globally against the subcolumn ICA treatment (#1521).
    lwp_lw: jnp.ndarray | None = None
    iwp_lw: jnp.ndarray | None = None

    def to_rrtmg_kwargs(self) -> dict:
        """Return cloud kwargs dict for ``rrtmgp_radiation`` / ``solve_columns``.

        **Deliberately omits cloud_fraction.** ``compute_cloud_properties``
        returns GRID-MEAN water paths (``lwp = q_c * dp / g``, q_c the
        grid-mean prognostic cloud water), which already carry the
        partial-coverage discount ``LWP_grid = cf · LWP_in-cloud``.
        RRTMG's optics multiplies cloud optical depth by
        ``cloud_fraction`` again ("scale cloud optical depth by cloud
        fraction for partial coverage", optics.py) — that scaling
        expects IN-CLOUD paths.  Passing grid-mean LWP *and*
        ``cloud_fraction`` double-counts the discount:
        ``τ_used = cf² · τ_in-cloud`` instead of ``cf · τ_in-cloud``,
        making clouds ~cf× too optically thin in both SW and LW
        (→ OSR too low, OLR too high).  Calibration probe found the
        bug cost 59 W/m² OSR at cf=0.6 and 113 W/m² at cf=0.3 (commit
        4c9591bb).

        Since τ is linear in LWP, "grid-mean LWP, no cf scaling" gives
        the same OPTICAL DEPTH as "in-cloud LWP × cf scaling", so this
        helper centralises the right behaviour and the ``cf²`` bug
        cannot resurface at a third call site (iter-15 restored it in
        ``physics_pipeline.py``, iter-16 fixed an independent copy in
        ``integration.py``).

        **That equality is in τ ONLY, not in flux.**  Both forms feed a
        SINGLE homogeneous column of depth ``cf·τ_in-cloud``, whereas
        the independent-column answer for a partly cloudy layer is
        ``cf·R(τ_in-cloud) + (1−cf)·R(0)``.  ``R`` is concave, so this
        path is always the BRIGHTER of the two — the partial-coverage
        plane-parallel bias.  There is no McICA/subcolumn/overlap
        machinery anywhere under ``radiation/rrtmgp`` to recover it.
        ``cloud_partial_coverage_optics="two_column"`` corrects it by
        thinning the path; see :func:`_partial_coverage_factor`.
        """
        kwargs = {
            "cloud_path_liq": self.lwp,
            "cloud_path_ice": self.iwp,
            "cloud_r_eff_liq": self.r_eff_liq,
            "cloud_r_eff_ice": self.r_eff_ice,
        }
        # Separate LONGWAVE paths (two_column coverage optics only): the
        # emissivity-space inversion produces a different effective path for
        # the LW stream.  Emitted only when present so every other scheme's
        # kwargs (and their consumers) stay byte-identical.
        if self.lwp_lw is not None:
            kwargs["cloud_path_liq_lw"] = self.lwp_lw
        if self.iwp_lw is not None:
            kwargs["cloud_path_ice_lw"] = self.iwp_lw
        return kwargs


def _ice_fraction(T: jnp.ndarray, config: CloudConfig) -> jnp.ndarray:
    """Fraction of condensate that is ice, based on temperature.

    Linear ramp from 0 (all liquid) at T_freeze to 1 (all ice) at T_ice_only.
    """
    frac = (config.T_freeze - T) / jnp.maximum(
        config.T_freeze - config.T_ice_only, 1.0
    )
    return jnp.clip(frac, 0.0, 1.0)


def sundqvist_cloud_fraction(
    RH: jnp.ndarray,
    config: CloudConfig,
) -> jnp.ndarray:
    """Sundqvist (1989) cloud fraction from relative humidity.

    Sundqvist, Berge & Kristjánsson (1989, MWR 117) relate the cloud
    cover ``b`` to the grid-mean RH by ``(1−b)² = (1−RH)/(1−RH_crit)``,
    i.e.

        ``b = 1 − √((1−RH)/(1−RH_crit))``   for RH ≥ RH_crit, else 0.

    This √-form (used by ECHAM and most Sundqvist implementations) is the
    faithful scheme; the earlier code here used a *linear* ramp
    ``(RH−RH_crit)/(1−RH_crit)`` mislabeled as Sundqvist.  The √-form stays
    BELOW the linear ramp on the whole interior (e.g. 1−√½ ≈ 0.29 vs 0.5 at
    the midpoint): its slope at RH_crit is ``0.5/(1−RH_crit)`` — HALF the
    linear ramp's ``1/(1−RH_crit)`` — and diverges only as RH → 1⁻.

    Parameters
    ----------
    RH : jnp.ndarray
        Relative humidity [0, 1+], shape (ncol, nlev).
    config : CloudConfig

    Returns
    -------
    jnp.ndarray
        Cloud fraction [0, 1], same shape as RH.
    """
    arg = (1.0 - RH) / jnp.maximum(1.0 - config.rh_crit, 1.0e-6)
    # Double-``where`` for the √-form: at RH ≥ 1 (arg ≤ 0) the cloud is
    # full so b = 1 *exactly*, while keeping ``√`` off zero so its
    # otherwise-infinite derivative cannot leak a NaN cotangent through
    # the dead branch (same AD-safe pattern as the Smagorinsky–Lilly
    # cutoff).  For RH < RH_crit, arg > 1 ⇒ b < 0 ⇒ clipped to 0.
    arg_safe = jnp.where(arg > 0.0, arg, 1.0)
    cf = jnp.where(arg > 0.0, 1.0 - jnp.sqrt(arg_safe), 1.0)
    return jnp.clip(cf, 0.0, 1.0)


def xu_randall_cloud_fraction(
    RH: jnp.ndarray,
    q_condensate: jnp.ndarray,
    q_sat: jnp.ndarray,
    config: CloudConfig,
) -> jnp.ndarray:
    """Xu-Randall (1996) cloud fraction from RH and condensate.

    Parameters
    ----------
    RH : jnp.ndarray
        Relative humidity [0, 1+], shape (ncol, nlev).
    q_condensate : jnp.ndarray
        Total cloud condensate (q_cloud + q_ice) [kg/kg], shape (ncol, nlev).
    q_sat : jnp.ndarray
        Saturation mixing ratio [kg/kg], shape (ncol, nlev).
    config : CloudConfig

    Returns
    -------
    jnp.ndarray
        Cloud fraction [0, 1].
    """
    # Xu-Randall (1996) denominator ((1−RH)·q_sat)^γ.  Floor the base
    # at 1e-10 *before* the power so the fractional γ<1 exponent never
    # sees 0 (1e-10^γ stays finite + positive ⇒ AD-safe).
    denominator = jnp.maximum((1.0 - RH) * q_sat, 1.0e-10) ** config.gamma_xr
    exponent = -config.alpha_xr * q_condensate / denominator
    # Floor the RH base of the fractional power at 1e-6 (not 0): p_xr < 1, so
    # ``RH**p_xr`` has an infinite derivative at RH=0 (0**-0.75), giving an inf
    # reverse-mode gradient d(cf)/d(q_v) for any dry layer (RH=0 ⇒ q_v=0, e.g.
    # upper stratosphere / dry init).  The forward value CAN change below the floor
    # (RH**p_xr evaluated at 1e-6, not RH; identical only where q_c = 0, as both
    # forms are then 0); the clip's role is to zero the reverse-mode gradient chain
    # there.  (A dry layer physically carries q_c ~ 0, so cf ~ 0 regardless.)
    cf = jnp.power(jnp.clip(RH, 1.0e-6, 1.0), config.p_xr) * (
        1.0 - jnp.exp(exponent)
    )
    return jnp.clip(cf, 0.0, 1.0)


def convective_cloud_fraction(
    conv_precip: jnp.ndarray,
    p_full: jnp.ndarray,
    config: CloudConfig,
) -> jnp.ndarray:
    """Slingo (1987)-style convective (cumulus) cloud fraction.

    Adjustment convection schemes (sbm Betts-Miller) hold the grid-mean column
    near ``RH_ref`` (~0.7) and detrain no ``q_c``, so the RH/condensate
    stratiform schemes diagnose ~0 cloud in the convecting tropics — the
    surface then radiates LW straight to space (the measured ~4.5 K coupled
    cold bias: tropical ``LW_net_sfc`` ~−137 W/m², precip ~1 mm/day).
    Following Slingo (1987), tie a *bounded* cumulus cloud cover to the
    convective precipitation rate:

        ``cf_conv = clip(conv_cloud_coeff · ln(1 + P_conv/P0), 0, conv_cloud_max)``

    distributed over the free-tropospheric convective deck
    ``[conv_cloud_sigma_top, conv_cloud_sigma_base]``.  The log-of-precip form
    saturates (so heavy ITCZ precip gives a capped, not overcast, cover — the
    failure mode of feeding a moistened column to the steep Sundqvist √-curve,
    which goes to cf≈1).  Combine with the stratiform fraction by MAXIMUM
    overlap in :func:`compute_cloud_properties`.

    Parameters
    ----------
    conv_precip : jnp.ndarray
        Column precipitation rate [kg/m²/s] used as the cumulus-activity proxy,
        shape (ncol,).  The coupled driver passes the (lagged) TOTAL column
        precip — in the convecting tropics this is dominated by convection (the
        target regime); precipitating extratropical columns also gain a modest
        capped cover, which is physically reasonable (rain ⇒ cloud).  Swap in a
        convective-only rate here if extratropical over-clouding appears.
    p_full : jnp.ndarray
        Full-level pressure [Pa], shape (ncol, nlev).
    config : CloudConfig

    Returns
    -------
    jnp.ndarray
        Convective cloud fraction [0, 1], shape (ncol, nlev).
    """
    # Normalize to strict (ncol,): accept (ncol,) or (ncol, 1) without the
    # ``[:, None]`` below producing a rank-3 broadcast (codex review).
    P = jnp.maximum(
        jnp.reshape(jnp.asarray(conv_precip, p_full.dtype), (p_full.shape[0],)),
        0.0,
    )
    cf_col = jnp.clip(
        config.conv_cloud_coeff * jnp.log1p(P / config.conv_precip_scale),
        0.0, config.conv_cloud_max,
    )  # (ncol,)
    # Sigma from the column's own surface (bottom full level ≈ surface).
    sigma = p_full / jnp.maximum(p_full[:, -1:], 1.0)
    deck = (
        (sigma >= config.conv_cloud_sigma_top)
        & (sigma <= config.conv_cloud_sigma_base)
    ).astype(p_full.dtype)
    return cf_col[:, None] * deck


# Smooth cloudy-DECK membership gate for the geometric-depth integral in
# ``_adiabatic_incloud_condensate``.  A sigmoid in cf (centred at
# ``_ADIAB_DECK_CF0``, width ~1/``_ADIAB_DECK_SHARPNESS``) turns cf into a smooth
# 0->1 STEP that is NOT proportional to cf, so the accumulated depth ``D`` is the
# GEOMETRIC cloudy depth and cloud COVERAGE enters exactly ONCE (the outer
# ``cf_strat``) rather than as cf^2.  Deck-MEMBERSHIP numerics, not a tunable.
_ADIAB_DECK_CF0 = 0.05          # cf at which a layer is a HALF deck member
_ADIAB_DECK_SHARPNESS = 200.0   # inverse membership-transition width [1/cf]


def _adiabatic_incloud_condensate(cf, dp, T, p_full, config):
    """Capped adiabatic IN-CLOUD LIQUID water content [kg/kg] for the stratiform
    radiative floor.

    Real in-cloud LWC grows ~linearly with height above cloud base (adiabatic
    ascent), so a THIN low cloud holds far less water than a deep one.  The flat
    ``q_c_diagnostic`` floor ignores this and over-brightens shallow marine
    stratocumulus (measured: the floor is the radiative q_c there, ~11x the
    prognostic).  Returns ``q_ad = min(adiabatic_lwc_rate * D, q_c_diagnostic)``,
    ``D`` the LAYER-MEAN cloudy geometric depth above cloud base.

    This is the LIQUID in-cloud value ONLY.  PHASE and COVERAGE are applied by the
    caller (``compute_cloud_properties``): it weights this by cloud fraction and by
    the LIQUID fraction ``(1 - f_ice)`` and floors the ICE part SEPARATELY with
    ``q_c_diagnostic`` — so the liquid adiabatic gradient never leaks into IWP,
    ``adiabatic_lwc_rate`` stays the physical gradient (coverage is not squared),
    and phase is applied EXACTLY once.

    Depth integral (surface-last: index 0 = model top, -1 = surface; height
    increases toward index 0 — reverse to surface->up, cumulative-sum, reverse
    back = a within-column suffix sum that grows base->top):

    * a SMOOTH deck-membership gate ``sigmoid(_ADIAB_DECK_SHARPNESS*(cf -
      _ADIAB_DECK_CF0))`` (a step in cf, not proportional to cf) weights each
      layer's geometric thickness ``dz = dp/(rho g)``;
    * the value is taken at the layer MIDPOINT (``- 0.5*dz``): a linear base->top
      profile averages to the midpoint, so the layer top over-counts a 1-layer
      cloud by 2x;
    * ``q_ad`` is capped at ``q_c_diagnostic`` so it can only DIM, never exceed the
      validated floor — deep clouds are ~unchanged above the cap depth
      (``q_c_diagnostic/rate`` ~ 667 m); only near-base layers dim.

    STACKED decks are ISOLATED by a hard reset at clear gaps (``gate < 0.5``): an
    upper deck gets its OWN base->top depth, not the lower deck's.  The caller also
    floors ICE separately at ``q_c_diagnostic``, so no liquid gradient reaches
    IWP/OLR even for a stacked cold deck.

    Mostly subdifferentiable (sigmoid, cumsum, cummax, minimum, maximum); the only
    kink is the gap boolean at cf~0.05 (a subgradient there, forward-exact, and
    negligible for genuinely cloudy cells).  No Python control flow on traced
    values.
    """
    # Layer geometric thickness dz = dp / (rho g), with rho = p / (R_d T) [m].
    rho = p_full / (constants.R_d * T)
    dz = dp / jnp.maximum(rho * constants.g, 1.0e-12)
    # Smooth deck-membership gate (sigmoid step in cf): coverage stays single.
    gate = nn.sigmoid(_ADIAB_DECK_SHARPNESS * (cf - _ADIAB_DECK_CF0))
    cloudy_dz = gate * dz
    # Cloudy geometric depth from cloud base to the layer MIDPOINT, RESET at clear
    # gaps so a stacked upper deck does NOT inherit a lower deck's depth.  Work on
    # the reversed (surface->up) axis: ``cr`` is the running cloudy depth (non-
    # decreasing), and ``cr`` minus the running-MAX of ``cr`` sampled at gap
    # layers (``gate < 0.5``) is the depth SINCE the last gap — a valid segmented
    # sum.  Then subtract half the current layer (midpoint).  The gap boolean is a
    # subgradient KINK at the deck threshold cf~0.05 (negligible for cloudy cells,
    # forward-exact); everything else is smooth.
    xr = cloudy_dz[..., ::-1]
    _vax = xr.ndim - 1
    cr = jnp.cumsum(xr, axis=_vax)
    is_gap = gate[..., ::-1] < 0.5
    last_gap = lax.cummax(jnp.where(is_gap, cr, 0.0), axis=_vax)
    depth_top = jnp.maximum(cr - last_gap, 0.0)
    depth_mid = jnp.maximum(depth_top - 0.5 * xr, 0.0)[..., ::-1]  # surface-last
    return jnp.minimum(config.adiabatic_lwc_rate * depth_mid,
                       config.q_c_diagnostic)


def _two_region_inhomogeneity_factor(
    tau: jnp.ndarray, fsd, g,
) -> jnp.ndarray:
    r"""Tau-dependent sub-grid cloud-optics inhomogeneity factor (chi_eff).

    Two-region (Shonk & Hogan 2008 "Tripleclouds") split of the in-cloud
    optical depth ``tau`` into equal-area optically-THIN ``tau(1-fsd)`` and
    optically-THICK ``tau(1+fsd)`` sub-columns.  Inverting the domain-mean
    conservative two-stream reflectance
    ``R_bar = 1/2[R(tau(1-fsd)) + R(tau(1+fsd))]``, ``R(t) = t/(t+gamma0)``,
    ``gamma0 = 2/(1-g)``, to an effective optical depth ``tau_eff`` and taking
    ``chi_eff = tau_eff/tau`` has the exact CLOSED FORM

        chi_eff = 1 - fsd^2 * tau / (gamma0 + tau).

    Written closed-form (not via ``gamma0 R_bar/(1-R_bar)``) so there is NO
    ``1/(1-R_bar)`` division (fp32-unsafe as ``R_bar -> 1`` for a thick cloud),
    NO overflow, and NO clip/zero-guard: the value is analytically bounded in
    ``[1 - fsd^2, 1]``, monotone-decreasing in ``tau``, and smooth =>
    ``jax.grad``-safe everywhere.

    Behaviour: ``chi_eff = 1`` for a thin cloud (``tau -> 0``) or a homogeneous
    one (``fsd -> 0``), and DECREASES with ``tau`` toward the asymptote
    ``1 - fsd^2`` -- so a THICK cloud is reduced MORE than a thin one (unlike a
    constant scalar).  Note the reduction is BOUNDED by ``1 - fsd^2``: for
    ``fsd < 1`` the effective ``tau`` still grows without limit, so this
    corrects the plane-parallel albedo bias by the physically-correct
    inhomogeneity amount but does NOT drive a very thick cloud optically thin.
    A genuine "clear sub-column" that caps a thick cloud's albedo (breaking the
    saturation outright) is the ``fsd -> 1`` limit, where ``tau_eff -> gamma0``.
    """
    gamma0 = 2.0 / jnp.maximum(1.0 - g, 1.0e-6)   # g is a fixed numerics const
    fsd2 = fsd * fsd
    # Formed as (1 - fsd^2) + fsd^2 gamma0/(gamma0 + tau), NOT the equal
    # 1 - fsd^2 tau/(gamma0 + tau): this builds the small residual DIRECTLY,
    # avoiding the 1-minus-almost-1 fp32 cancellation for a thick cloud.
    # 0 < gamma0/(gamma0 + tau) <= 1 (gamma0 > 0, tau >= 0) => chi_eff in
    # [1 - fsd^2, 1], smooth and finite (no clip/where; caller validates
    # fsd in [0, 1] via ExperimentConfig.validate_strict).
    return (1.0 - fsd2) + fsd2 * gamma0 / (gamma0 + tau)


def _partial_coverage_factor(
    tau_grid: jnp.ndarray, cf: jnp.ndarray, g,
) -> jnp.ndarray:
    r"""Partial-cloud-COVER optics factor (chi_cover) for the radiative path.

    ``tau_grid`` is the GRID-MEAN optical depth of the layer (``cf tau_ic``),
    summed over phases, and ``cf`` is the TRUE cloud fraction -- not the floored
    one.  Both choices are deliberate; see NUMERICS and MIXED PHASE below.

    The RRTMGP path solves ONE homogeneous column carrying the GRID-MEAN water
    path, i.e. reflectance ``R(cf tau_ic)``.  The independent-column (ICA)
    answer weights two SEPARATE solves,

        R_ICA = cf R(tau_ic) + (1 - cf) R(0),   R(t) = t/(t + gamma0).

    ``R`` is CONCAVE, so ``R(cf tau_ic) >= R_ICA`` for every ``cf < 1``: the
    single-column form is ALWAYS too bright.  This is the partial-coverage
    plane-parallel albedo bias, and it is INDEPENDENT of the in-cloud (fsd)
    variability that :func:`_two_region_inhomogeneity_factor` corrects -- that
    one subdivides the CLOUDY region, this one accounts for the CLEAR fraction.

    Solving ``R(tau_eff) = R_ICA`` for the effective optical depth gives the
    exact closed form ``tau_eff = gamma0 cf tau_ic/(gamma0 + (1-cf) tau_ic)``.
    Substituting ``tau_ic = tau_grid/cf`` and dividing by ``tau_grid`` (the
    optical depth is linear in the water path, so this IS the path scaling):

        chi_cover = gamma0 cf / (gamma0 cf + (1 - cf) tau_grid).

    NUMERICS -- this grid-mean form, not the algebraically equal
    ``gamma0/(gamma0 + (1-cf) tau_ic)``, is what ships.  The latter needs
    ``tau_ic = tau_grid/cf``, which forces a cf floor; the radiative path then
    PLATEAUS at the artificial floor value instead of vanishing as the true
    ``cf -> 0``, leaving a hidden optically-thick cloud in a layer that reports
    ``cloud_fraction = 0`` (codex review, 2026-07-31).  The form above never
    divides by ``cf``, so it takes the TRUE ``cf`` and reaches the right limit.
    There is likewise no ``1/(1-R)`` division and no cancellation; the value is
    analytically bounded in ``[0, 1]``, smooth, and jax.grad-safe.

    ``tau_grid`` MUST be the optical depth the layer actually has AFTER any
    in-cloud (fsd) correction -- see COMPOSITION below.

    MIXED PHASE -- ``tau_grid`` is the SUM over liquid and ice, and the single
    resulting factor is applied to BOTH paths.  Cloud cover is a property of the
    LAYER, and RRTMGP adds the phase optical depths into one layer total
    (rrtmgp/optics/cloud_optics.py), so inverting each phase separately and
    summing is not the same function and leaves the layer too bright whenever
    both phases are present.  This is the one place the coverage factor
    deliberately differs from the fsd factor, which IS per-phase because
    in-cloud VARIABILITY genuinely differs between a patchy liquid deck and a
    uniform ice layer.

    Limits (each exercised in tests/unit/test_partial_coverage_optics.py):
      * ``cf -> 1``       => chi_cover -> 1 (overcast: nothing to correct).
      * ``tau_grid -> 0`` => chi_cover -> 1 (thin: R already linear in tau).
      * ``cf -> 0`` at fixed grid-mean water => chi_cover -> 0, i.e. no
        radiative effect, the correct ICA limit.
      * ``tau_ic -> inf`` => tau_eff -> gamma0 cf/(1-cf), BOUNDED -- a sky that
        is only fraction ``cf`` cloudy cannot reflect more than ``cf``, which
        the single-column form violates outright.

    SIGN: ``chi_cover <= 1`` always, so this can only DIM the cloud, never
    brighten it -- the same one-way guarantee the fsd factor carries.

    COMPOSITION WITH ``two_region``.  The three-region subcolumn is: fraction
    ``(1-cf)`` clear, and the cloudy ``cf`` split into equal-area ``tau(1±fsd)``
    halves.  Its ICA reflectance is ``cf * Rbar_fsd = cf * R(tau chi_fsd)``, so
    the coverage inversion must be evaluated at ``C = tau * chi_fsd``, NOT at
    the raw ``tau``.  Evaluating at ``tau`` OVER-thins (codex review: cf=0.4,
    fsd=0.75, tau=100 gives tau_eff 3.66 against the correct 6.17, a reflectance
    0.101 too DIM).  The caller therefore applies the factors SEQUENTIALLY,
    recomputing tau between them.

    ACCURACY / SCOPE -- this factor is EXACT only for a SINGLE layer under a
    grey conservative two-stream.  Known, deliberate approximations:
      * SHORTWAVE, verified against the model's OWN RRTMGP by the committed
        solver-level regression ``test_rrtmgp_single_layer_moves_toward_ica``:
        the corrected TOA albedo lies strictly between the uncorrected value
        and the two-column ICA reference.  Residual = grey-vs-per-band g /
        Rayleigh / gas mismatch (RRTMGP spans g 0.71-0.98 and single-scatter
        albedo 0.53-1.0 across bands and phases, against the single grey
        gamma0 = 2/(1-0.85) here), so exact agreement is NOT expected or
        claimed.
      * LONGWAVE: the path also feeds LW, where the nonlinearity is
        ``eps = 1-exp(-tau)``, not ``R``.  The corrected LW emissivity is never
        FURTHER from the ICA value ``cf(1-exp(-tau_ic))`` than the uncorrected
        one (cf=0.5, tau_ic=10: 0.993 uncorrected -> 0.974 here, against 0.500
        ICA); equality holds at ``tau=0`` and at ``cf in {0,1}``.  This is an
        improvement in sign but NOT an LW fix -- a proper treatment needs a
        separate emissivity-space inversion.  NOTE the bound
        ``cf(1-e^-t) <= 1-e^-x <= 1-e^-ct`` relies on ``gamma0 >= 2``, i.e.
        ``g >= 0``, NOT merely on ``chi_cover <= 1``; a negative
        ``cloud_optics_asymmetry_g`` would break it.  The pre-existing fsd
        factor shares the SW-surrogate-applied-to-LW flaw -- precedent, not
        justification.
      * VERTICAL OVERLAP: applied per layer independently, because the solver
        carries no overlap state.  For a multi-layer cloud that is between the
        maximum- and random-overlap ICA answers (codex review, two identical
        cf=0.4/tau=25 layers: 0.600 uncorrected, 0.414 here, 0.316 max-overlap,
        0.439 random-overlap) -- i.e. still too bright under maximum overlap and
        slightly too dim under random.  "Always too bright" holds per layer, NOT
        for a deep cloud.
    """
    gamma0 = 2.0 / jnp.maximum(1.0 - g, 1.0e-6)   # g is a fixed numerics const
    num = gamma0 * cf
    # Floor only guards the 0/0 at cf=0 AND tau_grid=0 (a cloud-free layer,
    # where the path it multiplies is itself zero); it never perturbs a real
    # cloud, since the denominator is then >= min(gamma0 cf, tau_grid) >> eps.
    return num / jnp.maximum(num + (1.0 - cf) * tau_grid, 1.0e-30)


# --- LW grey absorption surrogate (geometric-optics limit) ---
# The LW coverage inversion needs the in-cloud ABSORPTION optical depth; the
# pipeline carries the geometric EXTINCTION depth tau_ext = 1.5*WP/(rho*r_eff)
# (Q_ext ~ 2).  For cloud particles large against LW wavelengths the
# absorption efficiency is Q_abs ~ 1 (van de Hulst 1957 large-sphere limit;
# Stephens 1984 review), i.e. tau_abs ~ 0.5 * tau_ext.  Using tau_ext raw
# over-thins the effective LW path (k_abs < k_ext) and OVERSHOOTS the ICA
# OLR by ~12% of the correction in the single-layer solver test; 0.5 lands
# the per-band RRTMGP answer on the ICA reference.  Fixed published limit,
# not a tunable (both the thin and thick limits of the inversion are
# k-independent — this only places the transition region).
_LW_ABS_TO_EXT_RATIO = 0.5

# Underflow guard for the LW emissivity inversion: floors the TRANSMITTED
# fraction ``(1-cf) + cf e^{-tau} = 1 + cf expm1(-tau)`` so ``-log1p`` stays
# finite when an overcast (cf -> 1) layer's in-cloud tau is large enough that
# ``expm1(-tau)`` rounds to exactly -1.  The floor must survive the
# ``x - 1.0`` subtraction in BOTH dtypes (1e-30 - 1.0 == -1.0 exactly, which
# defeats the guard): 1e-6 - 1.0 is representable in fp32 and fp64.  Caps
# ``tau_eff`` at ``-ln(1e-6) ~ 13.8`` — layer emissivity 0.999999, a
# < 1e-3 W/m2 flux difference from a fully black layer, invisible against
# any real cloud signal.  Numerics floor, not a tunable.
_LW_EMISS_ARG_FLOOR = 1.0e-6


def _partial_coverage_factor_lw(tau_ic: jnp.ndarray, cf: jnp.ndarray) -> jnp.ndarray:
    r"""Partial-cloud-COVER optics factor for the LONGWAVE path (chi_lw).

    The SW factor (:func:`_partial_coverage_factor`) inverts the ICA answer in
    REFLECTANCE space, ``R(t) = t/(t + gamma0)``.  The LW nonlinearity is the
    layer EMISSIVITY ``eps(t) = 1 - e^{-t}``, a different concave function, so
    the SW-derived effective tau applied to the LW path systematically
    OVER-TRAPS OLR: for a thick cloud the SW inversion caps ``tau_eff`` at
    ``gamma0 cf/(1-cf)`` (finite), whose emissivity ``1-e^{-tau_eff}`` far
    exceeds the ICA value ``cf``.  Measured on the AMIP day-365 state the
    surrogate cost ~10 W/m2 of global OLR against the subcolumn-ICA (McICA
    max_random) treatment, concentrated in the small-cf / large-tau tropical
    anvil (#1521, #929).  The SW factor's own docstring flags this: "a proper
    treatment needs a separate emissivity-space inversion" — this is it.

    Inversion (grey, per layer, in ABSORPTION space): with the in-cloud
    absorption depth ``tau_abs_ic = _LW_ABS_TO_EXT_RATIO * tau_ic`` (Q_abs ~
    Q_ext/2 in the large-particle limit; see the constant), the ICA
    emissivity of a partly cloudy layer

        eps_ICA = cf (1 - e^{-tau_abs_ic})

    equals the homogeneous column's ``1 - e^{-tau_eff}`` at

        tau_eff = -ln((1 - cf) + cf e^{-tau_abs_ic}),

    so the path scaling is ``chi_lw = tau_eff / (cf tau_abs_ic)`` (tau is
    linear in the water path, and RRTMGP applies its own per-band k_abs to
    the scaled path, so the k in numerator and denominator cancels).

    GREY SURROGATE: ``tau_ic`` is the same fsd-corrected GEOMETRIC
    (extinction) in-cloud optical depth the SW factor uses; the fixed 0.5
    absorption fraction stands in for the spectral LW k_abs.  Both limits
    are k-INDEPENDENT — thin (``tau -> 0`` => chi -> 1, no correction
    needed) and thick (``tau -> inf`` => eps_eff -> cf EXACTLY, for any
    absorption coefficient) — so the surrogate only places the transition
    region, unlike the reflectance-space surrogate it replaces, which is
    wrong in the thick limit itself.  Verified against the model's own
    per-band RRTMGP in tests/unit/test_partial_coverage_optics.py
    (single-layer OLR lands on the two-column ICA reference).

    Limits (each exercised in tests/unit/test_partial_coverage_optics.py):
      * ``cf -> 1``      => tau_eff = tau_ic exactly => chi_lw = 1 (overcast).
      * ``tau_ic -> 0``  => chi_lw -> 1 (thin: eps already linear in tau).
      * ``tau_ic -> inf``=> eps_eff -> cf: a sky only fraction ``cf`` cloudy
        cannot emit/absorb more than ``cf`` of a black layer.
      * ``cf -> 0``      => chi_lw -> (1-e^{-tau_abs_ic})/tau_abs_ic
        (bounded, <= 1); the grid-mean path it multiplies -> 0, and
        eps_eff -> eps_ICA -> 0.

    SIGN: concavity of ``1-e^{-t}`` gives ``tau_eff <= cf tau_ic`` always, so
    ``chi_lw <= 1`` — the factor can only DIM the LW path, never brighten it
    (same one-way guarantee as the SW factor; pinned by test).

    NUMERICS: no division by ``cf`` anywhere (the SW factor's NUMERICS note);
    the log argument is analytically in ``(0, 1]`` and floored only against
    e^{-tau} underflow at overcast+thick (see ``_LW_EMISS_ARG_FLOOR``); the
    ``tau -> 0`` limit is taken by a ``where`` on the denominator with the
    exact limit value 1 (AD-safe: both branches finite).
    """
    # Absorption depth from the pipeline's extinction depth (see
    # ``_LW_ABS_TO_EXT_RATIO``): the inversion must be solved — and the
    # resulting effective tau re-expressed as a PATH scaling — in ABSORPTION
    # space, since RRTMGP's LW optics applies its own per-band k_abs to the
    # path this factor scales.
    tau_abs_ic = _LW_ABS_TO_EXT_RATIO * tau_ic
    tau_abs_grid = cf * tau_abs_ic
    # ``(1-cf) + cf e^{-tau} == 1 + cf expm1(-tau)`` — the expm1/log1p pair
    # avoids the 1-minus-almost-1 cancellation in the THIN limit (the plain
    # log form loses ~4 digits of (1 - chi) at tau ~ 1e-12); the max() floors
    # the log1p argument at ``_LW_EMISS_ARG_FLOOR - 1`` against e^{-tau}
    # underflow at overcast+thick (see the constant's comment).
    tau_eff = -jnp.log1p(
        jnp.maximum(cf * jnp.expm1(-tau_abs_ic), _LW_EMISS_ARG_FLOOR - 1.0)
    )
    # 0/0 only at tau_abs_grid = 0 (clear or zero-depth layer), where the
    # exact limit is 1; the untaken branch is finite (denominator floored).
    return jnp.where(
        tau_abs_grid > 1.0e-30,
        tau_eff / jnp.maximum(tau_abs_grid, 1.0e-30),
        jnp.ones_like(tau_abs_grid),
    )


def compute_cloud_properties(
    T: jnp.ndarray,
    p_full: jnp.ndarray,
    q_v: jnp.ndarray,
    dp: jnp.ndarray,
    config: CloudConfig,
    q_cloud: jnp.ndarray | None = None,
    q_ice: jnp.ndarray | None = None,
    n_ice: jnp.ndarray | None = None,
    n_cloud: jnp.ndarray | None = None,
    conv_precip: jnp.ndarray | None = None,
    cloud_fraction_override: jnp.ndarray | None = None,
) -> CloudProperties:
    """Compute diagnostic cloud fraction and cloud optical properties.

    Parameters
    ----------
    T : jnp.ndarray
        Temperature [K], shape (ncol, nlev).
    p_full : jnp.ndarray
        Pressure at full levels [Pa], shape (ncol, nlev).
    q_v : jnp.ndarray
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    dp : jnp.ndarray
        Layer pressure thickness [Pa], shape (ncol, nlev).
        Computed as ``p_half[..., 1:] - p_half[..., :-1]`` (positive).
    config : CloudConfig
    q_cloud : jnp.ndarray or None
        Explicit cloud liquid water [kg/kg] from microphysics.
    q_ice : jnp.ndarray or None
        Explicit cloud ice [kg/kg] from microphysics.
    cloud_fraction_override : jnp.ndarray or None
        Optional sub-grid cloud fraction [-], shape (ncol, nlev), from a moist
        higher-order turbulence closure (CLUBB's PDF).  When supplied it REPLACES
        the RH-diagnosed ``cf`` (before the convective overlap), so the sub-grid
        condensate floor and hence radiation reflect the moist closure's
        less-overcast marine BL.  ``None`` (default) keeps the RH grid-scale
        fraction (byte-identical).

    Returns
    -------
    CloudProperties
        Cloud fraction and water/ice paths for radiation.
    """
    # Saturation mixing ratio and relative humidity.
    # ``saturation_scheme`` selects the curve RH is measured against
    # (dispatch-hardening: unknown value raises at fn entry on the static
    # config string, never a silent default):
    #   * "liquid" (legacy): liquid (Tetens) saturation at ALL temperatures.
    #     Genuinely ICE-saturated cold air (TTL / tropical anvil, ~205-245 K,
    #     where the liquid curve sits up to ~60% above the ice curve) then
    #     reads RH ~ 0.55-0.75 — below any rh_crit — so the RH-based schemes
    #     diagnose NO stratiform cloud exactly where the model carries
    #     substantial detrained ice (#1521: production day-365 anvil had
    #     Sundqvist cf = 0.000 at every level against RH_liq while 77% of the
    #     anvil cells were super-saturated over the mixed-phase curve).
    #   * "mixed_phase": w(T)-blended liquid/ice saturation, weighted by the
    #     SAME linear ice-fraction ramp ``_ice_fraction`` (T_freeze ->
    #     T_ice_only) that partitions this scheme's condensate — so the RH
    #     criterion and the diagnosed condensate PHASE agree by construction
    #     (a cloud diagnosed via the ice curve gets ice condensate).  This is
    #     the standard mixed-phase saturation convention (ECMWF IFS alpha(T)
    #     blend, Tiedtke 1993; Morrison M2005 partition ramp), and at the
    #     default ``T_ice_only = 233.15 K = constants.T_hom_freeze`` it equals
    #     the shared ``microphysics._warm_rain.mixed_phase_saturation_mixing_
    #     ratio`` curve used by the hard-saturation drain (verified equal to
    #     1.7e-18 kg/kg over 190-310 K at the CloudConfig defaults).  At and
    #     above T_freeze the ICE weight ``_f_ice_sat`` is exactly 0, so the
    #     blend collapses to ``1.0*q_sat_liq + 0.0*q_sat_ice`` = q_sat_liq
    #     BIT-identically (the ice curve is bounded, never inf/NaN, so the
    #     0.0*x term is exactly 0.0) => warm cloud is unchanged.
    if config.saturation_scheme == "liquid":
        q_sat = saturation_mixing_ratio(T, p_full)
    elif config.saturation_scheme == "mixed_phase":
        _f_ice_sat = _ice_fraction(T, config)
        q_sat = ((1.0 - _f_ice_sat) * saturation_mixing_ratio(T, p_full)
                 + _f_ice_sat * saturation_mixing_ratio_ice(T, p_full))
    else:
        raise ValueError(
            f"Unknown cloud saturation_scheme: {config.saturation_scheme!r}. "
            f"Valid schemes: 'liquid' (legacy, liquid saturation at all T), "
            f"'mixed_phase' (ice-fraction-blended liquid/ice saturation)."
        )
    RH = q_v / jnp.maximum(q_sat, 1.0e-10)

    # --- Cloud fraction ---
    if config.scheme == "xu_randall":
        q_c = jnp.zeros_like(T) if q_cloud is None else q_cloud
        q_i = jnp.zeros_like(T) if q_ice is None else q_ice
        q_condensate = q_c + q_i
        cf = xu_randall_cloud_fraction(RH, q_condensate, q_sat, config)
    elif config.scheme == "sundqvist":
        cf = sundqvist_cloud_fraction(RH, config)
    elif config.scheme == "resolved":
        # Cloud-resolving cloud fraction: a grid cell is (smoothly) FULLY
        # cloudy where it holds resolved condensate — SAM's CRM convention
        # (cf = 1 wherever qn = qcl+qci > 0), unlike Xu-Randall's sub-grid
        # fraction that under-represents the cloud-radiative effect at CRM
        # resolution.  Smooth saturating surrogate keeps it AD-safe.
        if q_cloud is None and q_ice is None:
            raise ValueError(
                "cloud scheme 'resolved' requires explicit q_cloud/q_ice "
                "from microphysics (it cannot diagnose condensate from RH "
                "like 'sundqvist')."
            )
        q_c = jnp.zeros_like(T) if q_cloud is None else jnp.maximum(q_cloud, 0.0)
        q_i = jnp.zeros_like(T) if q_ice is None else jnp.maximum(q_ice, 0.0)
        q_condensate = q_c + q_i
        cf = q_condensate / (q_condensate + config.q_cloud_resolved_ref)
    else:
        raise ValueError(
            f"Unknown cloud scheme: {config.scheme!r}. "
            f"Valid schemes: 'sundqvist', 'xu_randall', 'resolved'. "
            f"(Use cloud_scheme='none' upstream to skip clouds entirely.)"
        )

    # --- Moist-closure (CLUBB) cloud-fraction override ---
    # A moist higher-order closure (CLUBB) diagnoses a sub-grid cloud fraction
    # from its assumed PDF that is physically LESS overcast than the RH-diagnosed
    # grid-scale ``cf`` above over a saturated marine boundary layer.  When the
    # caller routes CLUBB's fraction here (via PhysicsState; gated to the
    # clubb-active path by ``RadiationConfig.use_clubb_cloud_fraction``), it
    # SUPERSEDES the RH ``cf`` so the sub-grid condensate floor
    # (``cf * q_c_diagnostic``, below) — which sets the marine-Sc LWP and hence
    # the planetary albedo — reflects the moist closure instead.  The override is
    # CLUBB's LIQUID fraction; the floor splits it by ``f_ice_diag(T)``, exact for
    # the warm-liquid marine BL target (cold-cloud ice reuse is a minor
    # approximation).  Convective overlap below still adds cumulus on top.
    # AD-safe: a plain clip, no NaN sentinel — the caller supplies a real
    # fraction; on the first step before turbulence has run it is the zero-init
    # carry, which merely drops the floor for that single step.
    if cloud_fraction_override is not None:
        _cf_clubb = jnp.clip(cloud_fraction_override, 0.0, 1.0)
        # STRENGTH: partial blend toward CLUBB rather than a full replacement
        # (``strength*CLUBB + (1-strength)*RH``).  Full replacement removed enough
        # low cloud under real forcing to drive a surface-heating runaway (day
        # 15-20 blowup, not fixed by halving dt); a gentler blend still lowers
        # albedo while keeping the column stable.  Static config branch; the blend
        # is on traced arrays (AD-safe).
        _strength = config.clubb_cf_override_strength
        if _strength < 1.0:
            _cf_clubb = _strength * _cf_clubb + (1.0 - _strength) * cf
        # FLOOR: keep a minimum BL cloud fraction so the override cannot collapse
        # the low cloud to ~0 (which triggered the cloud-temperature runaway).
        # Breaks the runaway while still allowing a bounded reduction.  Static
        # config branch (JIT-safe); jnp.maximum is AD-safe.
        _floor = config.clubb_cf_override_floor
        if _floor > 0.0:
            _cf_clubb = jnp.maximum(_cf_clubb, _floor)
        # LEVEL GATE (real-SST A/B fix): apply CLUBB's cf only in the boundary
        # layer / low cloud (p_full >= clubb_cf_override_p_min_pa, the marine-Sc
        # target) and keep the RH grid-scale fraction ALOFT.  A full-column
        # override over-clouds at altitude (CLUBB's PDF over-diagnoses high/mid
        # cloud -> OLR collapse + albedo RISE, the real-SST backfire).  The gate
        # is a static Python branch on the config (p_full is traced; the
        # threshold is a compile-time float): 0.0 restores the full-column
        # override (the analytical-A/B behaviour).
        _p_min = config.clubb_cf_override_p_min_pa
        _ramp = config.clubb_cf_override_ramp_pa
        if _p_min > 0.0 and _ramp > 0.0:
            # SMOOTH gate: linear weight w=1 in the BL (p_full >= p_min), 0 aloft
            # (p_full <= p_min - ramp), ramping between — a blend rather than a
            # step so the cloud/heating field has no discontinuity at the gate
            # (the sharp step seeded a late blowup in the real-SST A/B).  p_full is
            # traced; p_min/ramp are compile-time config floats.
            _w = jnp.clip((p_full - (_p_min - _ramp)) / _ramp, 0.0, 1.0)
            cf = _w * _cf_clubb + (1.0 - _w) * cf
        elif _p_min > 0.0:
            cf = jnp.where(p_full >= _p_min, _cf_clubb, cf)
        else:
            cf = _cf_clubb

    # --- Opt-in convective (cumulus) cloud, MAXIMUM-overlap combined ---
    # The stratiform RH/condensate fractions above miss convective cloud when an
    # adjustment scheme (sbm) holds the column subsaturated, so the convecting
    # tropics get cf≈0 and leak surface LW.  When enabled, add a bounded
    # Slingo(1987) cumulus cover tied to the convective precip rate; the
    # condensate floor below makes it radiatively active.  Default-off /
    # ``conv_precip=None`` ⇒ ``cf`` unchanged.  ``cf_strat`` is the stratiform
    # fraction BEFORE the convective overlap; the convective EXCESS
    # (``cf - cf_strat``) gets the optically-THIN ``conv_cloud_condensate``
    # (anvil cirrus: LW-active, SW-transparent) instead of the thick
    # ``q_c_diagnostic`` stratiform floor — without this, the high anvil
    # over-reflects SW (validation: planetary albedo ~42% vs Earth 30%).
    cf_strat = cf
    # Only the diagnostic-fraction schemes have the cf·condensate floor that
    # makes the added convective cover radiatively active; for 'resolved' (CRM,
    # explicit condensate is the truth) raising cf would leave the convective
    # excess optically inert, so restrict the feature to sundqvist/xu_randall
    # (codex review).
    if config.convective_cloud and config.scheme in ("sundqvist", "xu_randall"):
        if conv_precip is None:
            # Loud misconfiguration: the feature was requested but the caller
            # never plumbed the convective precip, so it would silently be a
            # no-op (dispatch-hardening — never a silent default).
            raise ValueError(
                "CloudConfig.convective_cloud=True requires conv_precip to be "
                "passed to compute_cloud_properties (the column convective "
                "precipitation rate [kg/m^2/s]); got None.  Wire the "
                "convection scheme's precip into the radiation cloud call."
            )
        cf_conv = convective_cloud_fraction(conv_precip, p_full, config)
        cf = jnp.maximum(cf, cf_conv)

    # --- Diagnostic in-cloud condensate scale, split stratiform/convective ---
    # Stratiform fraction carries the thick ``q_c_diagnostic``; the convective
    # EXCESS carries the thin ``conv_cloud_condensate`` so the anvil is LW-active
    # but SW-transparent.  When the convective cloud is off, ``cf_strat == cf``
    # ⇒ the excess is 0 ⇒ this reduces EXACTLY to ``cf * q_c_diagnostic``
    # (value-identical legacy behaviour — same numbers; the extra max/add ops
    # constant-fold but are not byte-identical HLO).
    _conv_excess = jnp.maximum(cf - cf_strat, 0.0)
    # In-cloud condensate for the STRATIFORM floor: a flat calibrated value
    # ("constant", the validated default) or a depth-scaled adiabatic LWC
    # ("adiabatic") that dims thin low clouds while leaving deep clouds at the
    # cap (see ``_adiabatic_incloud_condensate`` / CloudConfig docstring).  The
    # convective EXCESS keeps the thin anvil condensate regardless.
    if config.diagnostic_condensate_scheme == "constant":
        q_liq_incloud = config.q_c_diagnostic
    elif config.diagnostic_condensate_scheme == "adiabatic":
        # LIQUID in-cloud value only; the ICE floor stays q_c_diagnostic below.
        q_liq_incloud = _adiabatic_incloud_condensate(
            cf_strat, dp, T, p_full, config
        )
    else:
        # Dispatch-hardening: a typo must never silently fall back to the flat
        # floor and run different cloud physics (validated at fn entry on the
        # static config value).
        raise ValueError(
            f"Unknown diagnostic_condensate_scheme: "
            f"{config.diagnostic_condensate_scheme!r}; choose 'constant' "
            f"(flat q_c_diagnostic floor) or 'adiabatic' (depth-scaled "
            f"adiabatic in-cloud LWC)."
        )
    # PHASE-AWARE stratiform + convective-anvil floor (grid-mean), with ice
    # fraction applied EXACTLY ONCE: the LIQUID part carries the (adiabatic or
    # constant) in-cloud value, the ICE part ALWAYS the calibrated q_c_diagnostic
    # — so the liquid adiabatic gradient never contaminates IWP.  For
    # scheme='constant' both use q_c_diagnostic, so these reduce EXACTLY to the
    # legacy ``q_total_diag`` split by ice fraction (byte-identical).
    f_ice_diag = _ice_fraction(T, config)
    _conv_floor = _conv_excess * config.conv_cloud_condensate
    q_floor_liq = (cf_strat * q_liq_incloud + _conv_floor) * (1.0 - f_ice_diag)
    q_floor_ice = (cf_strat * config.q_c_diagnostic + _conv_floor) * f_ice_diag
    q_floor_total = q_floor_liq + q_floor_ice

    # --- Cloud condensate ---
    has_explicit_condensate = q_cloud is not None or q_ice is not None
    if has_explicit_condensate:
        q_c = jnp.zeros_like(T) if q_cloud is None else jnp.maximum(q_cloud, 0.0)
        q_i = jnp.zeros_like(T) if q_ice is None else jnp.maximum(q_ice, 0.0)
        # Coarse-GCM cloud-radiation floor (planetary-albedo fix).  The
        # prognostic GRID-MEAN condensate from microphysics under-represents the
        # radiatively-active SUB-GRID in-cloud water: a coarse grid-mean rarely
        # reaches saturation, so the grid-mean saturation-adjustment condenses
        # ~0 (q_c->0) even where the DIAGNOSTIC cloud fraction cf>0.  Feeding
        # that q_c~=0 to the cloud optics makes clouds OPTICALLY INERT and the
        # planetary albedo collapses to the clear-sky value (~12% vs Earth ~30%;
        # rrtmgp 8525261 measured rsut 35 W/m^2 with q_c~=0).  For the
        # DIAGNOSTIC-fraction schemes (sundqvist / xu_randall — sub-grid by
        # construction) floor the RADIATIVE condensate with ``cf *
        # q_c_diagnostic`` (the SAME calibrated in-cloud value the no-microphysics
        # path below uses), temperature-partitioned.  ``jnp.maximum`` so a scheme
        # carrying genuinely resolved/large q_c is unchanged (no double-count),
        # and condensate stays proportional to cf (aligned with the rh_crit=0.8
        # onset — does NOT re-introduce the cf/condensate MISMATCH that drove the
        # earlier overcast cold-drift; see CloudConfig.rh_crit).  Radiation-only:
        # the prognostic q_c and the water/energy budget are untouched.
        # NOT applied to 'resolved' (CRM: q_c IS the truth; a floor would inject
        # spurious cloud water).
        if config.scheme in ("sundqvist", "xu_randall"):
            if config.diagnostic_condensate_scheme == "constant":
                # EXACT legacy: ONE total floor F = cf*q_c_diagnostic + conv split
                # by ice fraction — no reconstruction, no division, so BYTE-
                # IDENTICAL (incl fp32) to the pre-feature path.  (Static branch on
                # the compile-time scheme string, not a traced value.)
                _F = cf_strat * config.q_c_diagnostic + _conv_floor
                deficit = jnp.maximum(_F - (q_c + q_i), 0.0)
                q_c = q_c + deficit * (1.0 - f_ice_diag)
                q_i = q_i + deficit * f_ice_diag
            else:
                # Adiabatic: deficit on the TOTAL (adds nothing when the prognostic
                # TOTAL already meets the floor — preserves the explicit phase
                # split there), apportioned by the floor's OWN phase ratio so ICE
                # gets the constant-floor share and LIQUID the dimmed share.  NB
                # for MIXED-PHASE explicit-condensate cells the ice deficit weakly
                # depends on the dimmed liquid TOTAL (a bounded coupling inherent
                # to the total-deficit form); the marine-BL target is warm-liquid,
                # where it is exact.
                deficit = jnp.maximum(q_floor_total - (q_c + q_i), 0.0)
                _liq_frac = q_floor_liq / jnp.maximum(q_floor_total, 1.0e-30)
                q_c = q_c + deficit * _liq_frac
                q_i = q_i + deficit * (1.0 - _liq_frac)
    else:
        # Diagnose condensate directly from the phase-aware floor (liquid part
        # carries the adiabatic/constant value, ice part the constant floor).
        q_c = q_floor_liq
        q_i = q_floor_ice

    # --- Cloud water/ice paths [kg/m^2] ---
    # Grid-mean water/ice paths: q * dp / g
    # These are grid-mean (not in-cloud) values, which is what RRTMGP expects
    # when treating each layer independently (no overlap assumption).
    # Grid-mean water/ice paths [kg/m^2].  The sub-grid inhomogeneity
    # correction (Cahalan scalar OR the tau-dependent two_region optic) is
    # applied AFTER the effective radii below, because the two_region scheme
    # needs the in-cloud optical depth = f(water path, r_eff).
    lwp = q_c * dp / constants.g
    iwp = q_i * dp / constants.g

    # Effective radii (constant for now): ``broadcast_to`` produces a
    # zero-copy logical view, whereas ``jnp.full_like(T, scalar)``
    # materialises a fresh ``(ncol, nlev)`` constant buffer every
    # physics step.  XLA folds the broadcast at trace time but the
    # broadcast form keeps the HLO graph small and avoids two
    # allocator round-trips per cloud_optics call.
    _scalar_dtype = T.dtype
    if n_cloud is not None:
        # RAD-1-liq: SAM+Morrison M2005 cloud-water effective RADIUS from the
        # gamma PSD, ``reffc = (PGAM+3)/(2·LAMC)`` (module_mp_graupel.f90:495).
        # PGAM = Martin et al. (1994) shape from the droplet number (:1676-1680);
        # LAMC the PSD slope (:1691) with Γ(PGAM+4)/Γ(PGAM+1) =
        # (PGAM+1)(PGAM+2)(PGAM+3). N_c is per-VOLUME [#/m³] in legoESM (≠ the
        # per-mass N_i), so the #/cm³ for PGAM is N_c/1e6 and the per-MASS number
        # for LAMC is N_c/ρ. Replaces the fixed 14 µm (= SAM's CAM OCEAN fallback;
        # SAM-M2005 uses this PSD reffc by default, douse_reffc=.true.).
        # Where the prognostic droplet number is 0/garbage (SAM specified-Nc
        # Morrison, dopredictNc=.false., keeps the Nc slot at 0), fall back to the
        # specified Nc_default so r_eff is the SAM constant-Nc value, not 35 um.
        # The PSD ratio N_c/q_c must pair LIKE WITH LIKE.  ``N_c`` is a tracer,
        # hence a GRID-MEAN number density, and ``q_c`` above is a grid-mean
        # mixing ratio (:833-836), so where the prognostic tracer is live the
        # ratio is already the in-cloud ratio and needs no cf.  ``Nc_default``
        # is NOT a grid mean: it is SAM's specified IN-CLOUD concentration
        # (dopredictNc=.false.), exact there because a CRM cell is either fully
        # cloudy or fully clear.  Pairing that in-cloud number with a grid-mean
        # q_c makes LAMC too large by cf^(-1/3), so reffc = (PGAM+3)/(2 LAMC)
        # comes out too SMALL by cf^(1/3) and the in-cloud tau below too LARGE
        # by cf^(-1/3), a spurious brightening that is ~1 for overcast layers
        # and grows without limit as the layer breaks up, i.e. it lands hardest
        # on exactly the subsidence regimes that should be the DARK end of the
        # shortwave contrast.  Reconstruct the in-cloud condensate there, with
        # the same floor the in-cloud water path uses for ``_cf_safe`` below so
        # the two agree on what "in-cloud" means; the LAMMIN clip below bounds
        # the result for vanishing cf (r_eff saturates near 36 um, well inside
        # the 60 um DIAMETER bound), so no column can run away as cover goes to
        # zero.  NB ``has_liq`` below still gates on the GRID-MEAN q_c against
        # SAM's QSMALL, unchanged: a cell whose grid mean is under 1e-14 but
        # whose reconstructed in-cloud value is above it takes the constant
        # r_eff_liq rather than the PSD.  Radiatively irrelevant at those water
        # paths, but the gate and the PSD do look at different condensate.
        _nc_is_specified = n_cloud <= 1.0
        n_cloud = jnp.where(n_cloud > 1.0, n_cloud, config.Nc_default)
        _cf_psd = jnp.clip(cf, _INHOM_CF_FLOOR, 1.0)
        q_c_psd = jnp.where(_nc_is_specified, q_c / _cf_psd, q_c)
        rho_air = p_full / (constants.R_d * jnp.maximum(T, 1.0))
        nc_cm3 = jnp.maximum(jnp.clip(n_cloud, 0.0), 0.0) / 1.0e6
        pgam = config.martin_pgam_slope * nc_cm3 + config.martin_pgam_intercept
        pgam = jnp.clip(
            1.0 / jnp.maximum(pgam, 1.0e-12) ** 2 - 1.0,
            config.pgam_min, config.pgam_max,
        )
        cons26 = jnp.pi * constants.rho_water / 6.0
        q_c_pos = jnp.maximum(jnp.clip(q_c_psd, 0.0), 1.0e-15)
        nc_permass = (jnp.maximum(jnp.clip(n_cloud, 0.0), 1.0e-15)
                      / jnp.maximum(rho_air, 0.1))  # coeff-ok: density floor [kg/m^3]
        lamc = (cons26 * nc_permass * (pgam + 1.0) * (pgam + 2.0) * (pgam + 3.0)
                / q_c_pos) ** (1.0 / 3.0)
        # SAM LAMMIN/LAMMAX (1-60 µm DIAMETER, :1697-1698) bound LAMC ⇒ reffc.
        lamc = jnp.clip(lamc, (pgam + 1.0) / _CLOUD_R_EFF_MAX_M, (pgam + 1.0) / 1.0e-6)
        r_eff_liq_psd = (pgam + 3.0) / (2.0 * lamc)
        has_liq = jnp.clip(q_c, 0.0) > 1.0e-14            # SAM QSMALL
        r_eff_liq = jnp.where(
            has_liq, r_eff_liq_psd,
            jnp.asarray(config.r_eff_liq, dtype=_scalar_dtype))
    else:
        r_eff_liq = jnp.broadcast_to(
            jnp.asarray(config.r_eff_liq, dtype=_scalar_dtype), T.shape,
        )
    if n_ice is not None:
        # RAD-1-ice: SAM+Morrison M2005 ice effective RADIUS from the ice
        # PSD, ``EFFI = 1.5/LAMI`` (module_mp_graupel.f90:4866), with
        # ``LAMI = (CONS12·N_i/q_i)^(1/3)``, CONS12 = ρ_ci·π (DI=3) and N_i
        # per-mass [1/kg] — the SAME PSD as the M2005 deposition.  So
        # ``r_eff_ice = 1.5·(q_i/(CONS12·N_i))^(1/3)`` [m].  EFFI = 25 µm in
        # ice-free cells (SAM uses 25 µm when q_i < QSMALL = 1e-14).
        # The radius is returned in metres; the RRTMGP cloud-optics
        # (cloud_optics.py) converts radius→generalized DIAMETER (×2) and
        # clamps to the ice lookup-table validity range, so NO explicit
        # bound is imposed here. (NB: legoESM's RRTMGP omits SAM's
        # ``ρ_ci/917`` solid-ice density rescale that its RRTM ice table
        # needs — an accepted RRTMG↔RRTMGP generation difference.)
        #
        # PAIR LIKE WITH LIKE (#1520): ``n_ice`` is the PROGNOSTIC (tracer)
        # ice number, so the M2005 PSD is meaningful only for the PROGNOSTIC
        # ice mass.  The ``q_i`` at this point additionally carries the
        # DIAGNOSTIC sub-grid condensate floor (``cf·q_c_diagnostic·f_ice``
        # + convective-anvil floor, :768-831), which has NO number of its
        # own.  Pairing that floor mass with the tracer ``N_i`` produced PSD
        # radii of metres wherever the floor dominates and ``N_i`` is
        # 0/tiny — on the #1520 production checkpoint ~34% of the radiative
        # IWP was pinned at the RRTMGP ice-LUT 180 µm diameter ceiling
        # (92% of that pinned mass floor-injected), leaving the ice ~3x
        # optically too thin.  Mirror of the liquid branch's dead-``N_c``
        # ``Nc_default`` fallback: the PSD radius applies to the TRACER
        # mass only, the floor-injected mass carries the configured
        # constant ``r_eff_ice`` (its own calibration knob), and the two
        # populations combine by EXTINCTION (τ ∝ IWP/r_eff ⇒ mass-weighted
        # HARMONIC mean):
        #   r_eff = (m_psd + m_flr) / (m_psd/r_psd + m_flr/r_const).
        # A cell with no floor mass (``m_flr = 0``: any 'resolved'-scheme
        # cell, or a cell whose prognostic condensate already exceeds the
        # floor) keeps the PSD radius EXACTLY (the ``jnp.where`` selects
        # the unblended value, so pre-#1520 behaviour is bit-preserved
        # there).
        cons12 = config.rho_cloud_ice * jnp.pi
        q_i_trc = (jnp.maximum(q_ice, 0.0) if q_ice is not None
                   else jnp.zeros_like(T))
        q_i_pos = jnp.maximum(q_i_trc, 1.0e-15)
        n_i_pos = jnp.maximum(jnp.clip(n_ice, 0.0), 1.0e-15)
        lami = (cons12 * n_i_pos / q_i_pos) ** (1.0 / 3.0)
        r_eff_ice_psd = _R_EFF_ICE_PSD_COEFF / jnp.clip(lami, 1.0e-30)
        has_trc_ice = q_i_trc > 1.0e-14                   # SAM QSMALL
        m_psd = jnp.where(has_trc_ice, q_i_trc, 0.0)
        m_flr = jnp.maximum(q_i - m_psd, 0.0)
        r_const = jnp.asarray(config.r_eff_ice, dtype=_scalar_dtype)
        # Extinction sum: both denominators are strictly positive by the
        # clips above (r_psd ≥ 1.5e-30 via the lami clip; r_const a config
        # float), so the untaken ``where`` branch stays finite (AD-safe).
        _ext = (m_psd / jnp.maximum(r_eff_ice_psd, 1.0e-30)
                + m_flr / jnp.maximum(r_const, 1.0e-30))
        r_eff_ice_blend = jnp.where(
            m_flr > 0.0,
            (m_psd + m_flr) / jnp.maximum(_ext, 1.0e-30),
            r_eff_ice_psd,
        )
        has_ice = jnp.clip(q_i, 0.0) > 1.0e-14            # SAM QSMALL
        r_eff_ice = jnp.where(has_ice, r_eff_ice_blend, _R_EFF_ICE_DEFAULT_M)
    else:
        r_eff_ice = jnp.broadcast_to(
            jnp.asarray(config.r_eff_ice, dtype=_scalar_dtype), T.shape,
        )

    # --- Sub-grid cloud-optics inhomogeneity (applied to the radiative paths) -
    # Real clouds are horizontally PATCHY, so a plane-parallel HOMOGENEOUS layer
    # carrying the same mean water is too reflective (the plane-parallel albedo
    # bias).  This THINS the radiative lwp/iwp; SIGN: chi <= 1 (never brightens).
    # Dispatch raises on an unknown scheme (fn-entry, static config value).
    # In-cloud optical depth tau = (3 Q_ext / 4) * WP_incloud / (rho_p r_eff)
    # with Q_ext≈2 => coeff 1.5; WP_incloud = grid-mean WP / cf.  Computed ONCE
    # here from the UNCORRECTED paths so the fsd and coverage factors below are
    # both functions of the same physical tau_ic and cannot double-discount.
    _cf_safe = jnp.clip(cf, _INHOM_CF_FLOOR, 1.0)
    _r_liq = jnp.maximum(r_eff_liq, _INHOM_R_EFF_FLOOR_M)
    _r_ice = jnp.maximum(r_eff_ice, _INHOM_R_EFF_FLOOR_M)
    _tau_liq = _TAU_GEOMETRIC_COEFF * (lwp / _cf_safe) / (constants.rho_water * _r_liq)
    _tau_ice = _TAU_GEOMETRIC_COEFF * (iwp / _cf_safe) / (config.rho_cloud_ice * _r_ice)
    _g = config.cloud_optics_asymmetry_g

    _inhom_scheme = getattr(config, "cloud_optics_inhomogeneity", "constant")
    if _inhom_scheme == "constant":
        # Cahalan et al. (1994) fixed scalar (legacy; chi=1 => byte-identical).
        _chi = getattr(config, "cloud_inhomogeneity_factor", 1.0)
        lwp = lwp * _chi
        iwp = iwp * _chi
        # tau is linear in the path, so the in-cloud tau the COVER correction
        # must see is scaled by the same factor.
        _tau_liq = _tau_liq * _chi
        _tau_ice = _tau_ice * _chi
    elif _inhom_scheme == "two_region":
        # Apply the inhomogeneity factor PER PHASE, each from its OWN optical
        # depth (a patchy LIQUID cloud must not thin a horizontally-uniform ICE
        # layer).  tau -> 0 => chi_eff -> 1 (no change).
        _fsd = config.cloud_fsd
        _chi_liq = _two_region_inhomogeneity_factor(_tau_liq, _fsd, _g)
        _chi_ice = _two_region_inhomogeneity_factor(_tau_ice, _fsd, _g)
        lwp = lwp * _chi_liq
        iwp = iwp * _chi_ice
        # SEQUENTIAL, not a product of two factors of the raw tau: the
        # three-region ICA reflectance is cf*R(tau*chi_fsd), so the coverage
        # inversion below must be evaluated at the fsd-THINNED optical depth.
        # Evaluating both at the raw tau over-thins (codex review, 2026-07-31).
        _tau_liq = _tau_liq * _chi_liq
        _tau_ice = _tau_ice * _chi_ice
    else:
        raise ValueError(
            f"unknown cloud_optics_inhomogeneity {_inhom_scheme!r}; "
            "expected 'constant' or 'two_region'"
        )

    # --- Partial cloud COVER (the CLEAR fraction, not the in-cloud variance) -
    # The radiative solver has no McICA / subcolumn / overlap machinery: it sees
    # ONE homogeneous column at the grid-mean path, which is too bright whenever
    # cf < 1.  SIGN: chi_cover <= 1 (never brightens).  Unknown scheme => raise.
    _cover_scheme = getattr(config, "cloud_partial_coverage_optics", "none")
    _overlap_scheme = getattr(config, "cloud_vertical_overlap_optics", "none")
    if _cover_scheme != "none" and _overlap_scheme != "none":
        # Both correct PARTIAL COVERAGE -- two_column horizontally per layer,
        # max_random with real subcolumns. Enabling both discounts the cloud
        # twice, which would read as a bigger "fix" while being wrong.
        raise ValueError(
            "cloud_partial_coverage_optics and cloud_vertical_overlap_optics "
            "are mutually exclusive (both correct partial cloud coverage); "
            f"got {_cover_scheme!r} and {_overlap_scheme!r}"
        )
    lwp_lw = iwp_lw = None
    if _cover_scheme == "none":
        pass                       # legacy path; byte-identical to before
    elif _cover_scheme == "two_column":
        # ONE factor for the layer, from the COMBINED grid-mean optical depth of
        # both phases (RRTMGP sums tau_liq + tau_ice into a single layer total),
        # and from the TRUE cf -- _cf_safe would plateau the correction at the
        # 1e-3 floor instead of vanishing as cf -> 0.  _tau_* are in-cloud, so
        # cf_safe * tau_ic recovers the grid-mean depth the solver will see.
        _tau_ic_tot = _tau_liq + _tau_ice
        _tau_grid = _cf_safe * _tau_ic_tot
        _chi_cover = _partial_coverage_factor(_tau_grid, cf, _g)
        # SEPARATE LW paths: the coverage inversion is stream-specific —
        # reflectance space for SW (above), emissivity space for LW.  The
        # SW-space factor applied to the LW path over-trapped OLR by ~10 W/m2
        # globally (small-cf/thick tropical anvil worst; #1521 measurement vs
        # the McICA subcolumn treatment).  Same composition rule: chi_lw is a
        # function of the fsd-THINNED in-cloud tau (sequential, not raw).
        _chi_cover_lw = _partial_coverage_factor_lw(_tau_ic_tot, cf)
        lwp_lw = lwp * _chi_cover_lw
        iwp_lw = iwp * _chi_cover_lw
        lwp = lwp * _chi_cover
        iwp = iwp * _chi_cover
    else:
        raise ValueError(
            f"unknown cloud_partial_coverage_optics {_cover_scheme!r}; "
            "expected 'none' or 'two_column'"
        )

    return CloudProperties(
        cloud_fraction=cf,
        lwp=lwp,
        iwp=iwp,
        r_eff_liq=r_eff_liq,
        r_eff_ice=r_eff_ice,
        lwp_lw=lwp_lw,
        iwp_lw=iwp_lw,
    )
