"""Diagnostic cloud fraction and cloud optical property computation.

Implements two cloud fraction schemes:

1. **Sundqvist, Berge & Kristjansson (1989)**: RH-based, simple and robust.
   The sqrt-form ``(1 − b)² = (1 − RH)/(1 − RH_crit)``, i.e.
   ``cf = 1 − sqrt((1 − RH)/(1 − RH_crit))`` for RH ≥ RH_crit, else 0
   (clamped to [0, 1]).  See ``sundqvist_cloud_fraction`` for the
   AD-safe double-``where`` implementation.

2. **Xu-Randall (1996)**: RH + condensate-based, more physical.
   ``cf = RH^p * [1 - exp(-alpha * q_c / ((1 - RH) * q_s))]``

Both schemes compute cloud fraction per column per layer and derive
cloud liquid/ice water paths for RRTMGP cloud optics.

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
from jax import lax

from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.thermo import saturation_mixing_ratio
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

        Since τ is linear in LWP, "grid-mean LWP, no cf scaling" is
        mathematically identical to the correct "in-cloud LWP × cf
        scaling".  This helper centralises the right behaviour so the
        bug cannot resurface at a third call site (iter-15 restored
        it in ``physics_pipeline.py``, iter-16 fixed an independent
        copy in ``integration.py``).
        """
        return {
            "cloud_path_liq": self.lwp,
            "cloud_path_ice": self.iwp,
            "cloud_r_eff_liq": self.r_eff_liq,
            "cloud_r_eff_ice": self.r_eff_ice,
        }


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
    ``(RH−RH_crit)/(1−RH_crit)`` mislabeled as Sundqvist — the √-form
    rises faster just above RH_crit (e.g. 0.29 vs 0.5 at the midpoint).

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
    # upper stratosphere / dry init).  The forward is unaffected — cf -> 0 there
    # anyway via the (1 - exp) factor — and the clip zeroes the gradient chain
    # below the floor.
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


# Cloud-fraction above which a layer counts as (fully) part of a cloudy DECK for
# the geometric-depth integral in ``_adiabatic_incloud_condensate``.  This is a
# deck-ISOLATION scale, not a tunable closure: it turns cf into a saturating
# 0/1-ish indicator so ``D`` is the true GEOMETRIC cloudy depth and cloud
# COVERAGE enters exactly ONCE (via the outer ``cf_strat`` in ``q_total_diag``).
# Weighting the depth by cf itself would make the grid-mean floor scale as cf^2
# (double-counting coverage) and detach ``adiabatic_lwc_rate`` from the physical
# adiabatic gradient.  0.05 => any layer with cf >= 5% contributes its full dz.
_ADIAB_DECK_CF_FLOOR = 0.05


def _adiabatic_incloud_condensate(cf, dp, T, p_full, config):
    """Capped adiabatic IN-CLOUD liquid-water content [kg/kg] for the stratiform
    radiative floor of WARM (liquid) clouds.

    Real in-cloud LWC grows ~linearly with height above cloud base (adiabatic
    ascent), so a THIN low cloud holds far less water than a deep one.  The flat
    ``q_c_diagnostic`` floor ignores this and over-brightens shallow marine
    stratocumulus (measured: the floor is the radiative q_c there, ~11x the
    prognostic).  Returns, for warm cells, ``q_ad = min(adiabatic_lwc_rate * D,
    q_c_diagnostic)`` where ``D`` is the LAYER-MEAN cloudy geometric depth above
    cloud base.  This is the IN-CLOUD value; cloud COVERAGE is applied once by
    the caller (``q_total_diag = cf_strat * q_ad``), so ``adiabatic_lwc_rate`` is
    the physical adiabatic LWC gradient, not a coverage-entangled effective rate.

    Depth integral (surface-last: index 0 = model top, -1 = surface; height
    increases toward index 0):

    * a SATURATING indicator ``clip(cf/_ADIAB_DECK_CF_FLOOR, 0, 1)`` (not cf, so
      coverage is not double-counted) marks cloudy layers;
    * a RESET cumulative sum over the reversed (surface->up) axis accumulates the
      cloudy geometric depth but RESTARTS at every clear gap (``cr`` minus the
      running-max of ``cr`` sampled at clear layers), so a stacked upper deck
      does NOT inherit a lower deck's depth (codex review: the inherited
      condensate would be phase-partitioned to ice and perturb IWP/OLR);
    * the value is taken at the layer MIDPOINT (``- 0.5*dz`` of the current
      layer): a linear base->top profile has its layer-average at the midpoint,
      so the layer top would over-count a 1-layer cloud by 2x (codex review).

    A WARM/liquid gate (``T > T_freeze``) restricts the liquid adiabatic gradient
    to warm boundary-layer cloud; ice / mixed-phase cells keep the calibrated
    constant floor (do not dim cirrus with a liquid gradient and repartition to
    ice).  ``q_ad`` is capped at ``q_c_diagnostic`` so it can only DIM, never
    exceed the validated floor — deep warm clouds are ~unchanged above the cap
    depth (``q_c_diagnostic/rate`` ~ 667 m); only near-base layers dim.

    Subdifferentiable / transform-compatible (cumsum, clip, cummax, minimum are
    kinked and JAX supplies subgradients); no Python control flow on traced
    values.
    """
    # Layer geometric thickness dz = dp / (rho g), with rho = p / (R_d T) [m].
    rho = p_full / (constants.R_d * T)
    dz = dp / jnp.maximum(rho * constants.g, 1.0e-12)
    # Saturating cloudy-deck indicator so D is the GEOMETRIC cloudy depth and
    # coverage (cf) is applied exactly once downstream (not squared).
    cloud_indicator = jnp.clip(cf / _ADIAB_DECK_CF_FLOOR, 0.0, 1.0)
    cloudy_dz = cloud_indicator * dz
    # Reset cumulative sum over the reversed (surface -> up) axis: restart the
    # depth integral at every clear gap so decks are isolated.  ``cr`` is
    # non-decreasing (cloudy_dz >= 0), so the most-recent clear-gap depth is the
    # running MAX of ``cr`` sampled at clear layers.
    xr = cloudy_dz[..., ::-1]
    clear = cloud_indicator[..., ::-1] < 0.5
    _vax = xr.ndim - 1  # vertical axis (last); lax.cummax rejects a negative axis
    cr = jnp.cumsum(xr, axis=_vax)
    last_gap = lax.cummax(jnp.where(clear, cr, 0.0), axis=_vax)
    depth_top = cr - last_gap                       # depth above base at layer TOP
    depth_mid = jnp.maximum(depth_top - 0.5 * xr, 0.0)  # layer-MEAN (midpoint)
    depth_from_base = depth_mid[..., ::-1]          # back to surface-last
    q_ad = jnp.minimum(config.adiabatic_lwc_rate * depth_from_base,
                       config.q_c_diagnostic)
    # Warm/liquid gate: ice & mixed-phase cells keep the constant floor.
    warm = T > constants.T_freeze
    return jnp.where(warm, q_ad, config.q_c_diagnostic)


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

    Returns
    -------
    CloudProperties
        Cloud fraction and water/ice paths for radiation.
    """
    # Saturation mixing ratio and relative humidity
    q_sat = saturation_mixing_ratio(T, p_full)
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
        q_c_incloud = config.q_c_diagnostic
    elif config.diagnostic_condensate_scheme == "adiabatic":
        q_c_incloud = _adiabatic_incloud_condensate(
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
    q_total_diag = (
        cf_strat * q_c_incloud
        + _conv_excess * config.conv_cloud_condensate
    )

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
            # Floor on TOTAL condensate, then add only the DEFICIT, partitioned
            # by temperature.  Per-phase maxima would over-floor a layer whose
            # explicit condensate already meets the floor but sits in one phase
            # (e.g. all-ice: the liquid max would still inject liquid),
            # inflating total condensate (codex review).  The deficit form adds
            # nothing when the prognostic TOTAL already meets the floor, so the
            # explicit phase split is preserved EXACTLY there.
            deficit = jnp.maximum(q_total_diag - (q_c + q_i), 0.0)
            f_ice_diag = _ice_fraction(T, config)
            q_c = q_c + deficit * (1.0 - f_ice_diag)
            q_i = q_i + deficit * f_ice_diag
    else:
        # Diagnose condensate from cloud fraction and a typical in-cloud value
        # (stratiform thick + convective-excess thin), partitioned by temperature.
        f_ice = _ice_fraction(T, config)
        q_c = q_total_diag * (1.0 - f_ice)
        q_i = q_total_diag * f_ice

    # --- Cloud water/ice paths [kg/m^2] ---
    # Grid-mean water/ice paths: q * dp / g
    # These are grid-mean (not in-cloud) values, which is what RRTMGP expects
    # when treating each layer independently (no overlap assumption).
    # The Cahalan et al. (1994) inhomogeneity factor scales the RADIATIVE water
    # path down to account for horizontal cloud-water variability: a patchy real
    # cloud is less reflective than a plane-parallel homogeneous layer of the
    # same mean water (the plane-parallel albedo bias).  chi = 1.0 (default) is
    # the legacy homogeneous path (byte-identical); chi < 1 thins the optics.
    _chi = getattr(config, "cloud_inhomogeneity_factor", 1.0)
    lwp = q_c * dp / constants.g * _chi
    iwp = q_i * dp / constants.g * _chi

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
        n_cloud = jnp.where(n_cloud > 1.0, n_cloud, config.Nc_default)
        rho_air = p_full / (constants.R_d * jnp.maximum(T, 1.0))
        nc_cm3 = jnp.maximum(jnp.clip(n_cloud, 0.0), 0.0) / 1.0e6
        pgam = config.martin_pgam_slope * nc_cm3 + config.martin_pgam_intercept
        pgam = jnp.clip(
            1.0 / jnp.maximum(pgam, 1.0e-12) ** 2 - 1.0,
            config.pgam_min, config.pgam_max,
        )
        cons26 = jnp.pi * constants.rho_water / 6.0
        q_c_pos = jnp.maximum(jnp.clip(q_c, 0.0), 1.0e-15)
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
        cons12 = config.rho_cloud_ice * jnp.pi
        q_i_pos = jnp.maximum(jnp.clip(q_i, 0.0), 1.0e-15)
        n_i_pos = jnp.maximum(jnp.clip(n_ice, 0.0), 1.0e-15)
        lami = (cons12 * n_i_pos / q_i_pos) ** (1.0 / 3.0)
        r_eff_ice_psd = _R_EFF_ICE_PSD_COEFF / jnp.clip(lami, 1.0e-30)
        has_ice = jnp.clip(q_i, 0.0) > 1.0e-14            # SAM QSMALL
        r_eff_ice = jnp.where(has_ice, r_eff_ice_psd, _R_EFF_ICE_DEFAULT_M)
    else:
        r_eff_ice = jnp.broadcast_to(
            jnp.asarray(config.r_eff_ice, dtype=_scalar_dtype), T.shape,
        )

    return CloudProperties(
        cloud_fraction=cf,
        lwp=lwp,
        iwp=iwp,
        r_eff_liq=r_eff_liq,
        r_eff_ice=r_eff_ice,
    )
