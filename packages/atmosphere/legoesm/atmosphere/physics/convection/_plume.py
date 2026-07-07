"""Column physics primitives shared across convection schemes.

Smooth, fully-differentiable helpers consumed by Zhang-McFarlane (PR 1),
Kain-Fritsch (PR 2), Emanuel (PR 3), Tiedtke (PR 4), and Bechtold/IFS
(PR 5).  The goal of this module is to provide a small set of
well-tested column workhorses so that no convection scheme has to
re-implement an LCL, LFC, plume integrator, or CMT closure.

This module pairs with :mod:`._triggers` (sigmoid math) — together they
are the only place these primitives live.

What's in this PR (PR 0):

* :func:`compute_lcl`           — Bolton (1980) lifting condensation level.
* :func:`compute_lfc_lnb`       — smooth fractional levels of free
                                  convection and neutral buoyancy.
* :func:`compute_cin`           — CIN [J/kg] from the buoyancy profile.
* :func:`entraining_detraining_plume` — vmappable updraft integrator
                                  (``jax.lax.scan`` over levels).
* :func:`cmt_gregory_1997`      — Gregory et al. 1997 convective
                                  momentum transport closure.

Deferred to later scheme PRs (each lands alongside its first consumer):

* ``diagnose_grid_w_from_omega``     — Kain-Fritsch trigger (PR 2).
* ``buoyancy_sort_emanuel``           — Emanuel mixing ensemble (PR 3).
* ``downdraft_thermo``                — Tiedtke / Bechtold downdrafts
                                         (PRs 4–5).

This deferral was a pragmatic scope reduction within PR 0; each helper
is small (~50–200 LOC) and is documented in
``add-more-complex-convection-fluttering-quasar.md`` under its scheme.

Conventions
-----------
* Column shape ``(ncol, nlev)``.  **Surface at the last index**
  ``[:, -1]``; the model top is at ``[:, 0]``.  Pressure ``p_full``
  *increases* with the array index (TOA ≪ surface).
* Half levels ``p_half`` have shape ``(ncol, nlev+1)`` with ``p_half[:, 0]``
  at TOA and ``p_half[:, -1]`` at the surface.
* All temperatures in Kelvin, pressures in Pa, mass flux in kg/m²/s.
* Reused upstream — never re-implement: ``legoesm.constants`` for
  physical constants; ``legoesm.thermo.saturation_mixing_ratio`` and
  ``saturation_vapor_pressure`` for thermodynamics; the ``moist_adiabat``
  / ``moist_adiabat_lapse_rate`` / ``compute_cape`` family from
  ``legoesm.atmosphere.physics.thermodynamics``; column geometry
  helpers (``compute_heights_from_sigma``, ``compute_layer_dz``,
  ``compute_rho``) from ``legoesm.atmosphere.physics._shared``.

References
----------
* Bolton, D. (1980). The computation of equivalent potential
  temperature.  *Mon. Wea. Rev.*, 108(7), 1046–1053.
* Gregory, D., Kershaw, R., & Inness, P. M. (1997). Parametrization of
  momentum transport by convection. II: Tests in single-column and
  general circulation models.  *Quart. J. Roy. Meteor. Soc.*, 123,
  1153–1183.
* Zhang, G. J., & McFarlane, N. A. (1995). Sensitivity of climate
  simulations to the parameterization of cumulus convection in the
  Canadian Climate Centre general circulation model.  *Atmos.-Ocean*,
  33(3), 407–446.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics._shared import compute_rho, virtual_temperature
from legoesm.atmosphere.physics.thermodynamics import (
    bolton_lcl_temperature,
    moist_adiabat_lapse_rate,
)

from legoesm.atmosphere.physics.convection._triggers import (
    smooth_level_indicator,
    smooth_lowest_crossing_index,
)

__all__ = (
    "LCL",
    "Plume",
    "compute_lcl",
    "compute_lfc_lnb",
    "compute_cin",
    "entraining_detraining_plume",
    "cmt_gregory_1997",
)


# ---------------------------------------------------------------------------
# Lifting condensation level (Bolton 1980)
# ---------------------------------------------------------------------------

# --- pspec autoblock
_CROSSING_SHARPNESS = 0.001
_PLUME_GATE_SHARPNESS = 20.0
_PLUME_C_U = 0.55
_PLUME_C_D = 0.55
# Cloud-base gate sharpness [1/level index] for the plume's
# ``above_base_weight`` sigmoid on the (integer) level-index difference
# ``k_rev - k_base_rev``.  4.0 puts the gate at ~0.02 one level below the
# cloud base and ~3e-4 two levels below, so the reported updraft mass
# flux effectively vanishes in the sub-cloud layer.  This is a LEVEL-INDEX
# sharpness — deliberately independent of the Kelvin ``buoyancy_sharpness``
# taper (same-name/different-units bug class; ``compute_cin`` keeps the
# analogous separation with ``indicator_sharpness``).
_ABOVE_BASE_SHARPNESS = 4.0

class LCL(NamedTuple):
    """LCL diagnostics for one column.

    Fields
    ------
    p_lcl : jax.Array, shape (ncol,)
        LCL pressure [Pa].
    T_lcl : jax.Array, shape (ncol,)
        LCL temperature [K].
    k_lcl_smooth : jax.Array, shape (ncol,)
        Smooth fractional level index (surface-last convention) at
        which ``p_full`` crosses ``p_lcl``.  Used by downstream
        helpers that need to localize the LCL on the model grid.
    """
    p_lcl: jax.Array
    T_lcl: jax.Array
    k_lcl_smooth: jax.Array


def compute_lcl(
    T_parcel: jax.Array,
    q_parcel: jax.Array,
    p_parcel: jax.Array,
    p_full: jax.Array,
    *,
    crossing_sharpness: float = _CROSSING_SHARPNESS,
) -> LCL:
    """Lifting condensation level via Bolton (1980) Eq. 22.

    Bolton's empirical formula::

        T_LCL = 1 / [ 1/(T - 55) - ln(RH)/2840 ] + 55

    where ``T`` is parcel temperature [K], ``RH`` is relative humidity
    in ``(0, 1]``, and the result is the LCL temperature [K].  The
    LCL pressure follows from Poisson's equation along a dry adiabat
    from parcel level::

        p_LCL = p * (T_LCL / T)^(c_pd / R_d)

    Parameters
    ----------
    T_parcel : jax.Array, shape (ncol,)
        Parcel temperature at the launch level [K].
    q_parcel : jax.Array, shape (ncol,)
        Parcel water-vapor specific humidity [kg/kg].
    p_parcel : jax.Array, shape (ncol,)
        Parcel launch pressure [Pa] (typically the lowest model
        full-level pressure).
    p_full : jax.Array, shape (ncol, nlev)
        Full-level pressure [Pa] for the column, surface-last.  Used
        only to compute ``k_lcl_smooth``.
    crossing_sharpness : float
        Sharpness for the soft fractional level diagnosis.  Units of
        [1/Pa]; ``0.001`` sharpens to ~95%/5% transition over a
        ~1000 Pa pressure range, which is finer than typical model
        layer thickness in the boundary layer.

    Returns
    -------
    LCL
        ``p_lcl, T_lcl, k_lcl_smooth``.
    """
    # Bolton (1980) Eq. 22 via the canonical shared implementation
    # (``thermodynamics.bolton_lcl_temperature`` — RH clip, the 55 K
    # offset and the 2840 K denominator live only there).
    T_lcl = bolton_lcl_temperature(T_parcel, p_parcel, q_parcel)

    # Poisson: dry-adiabatic descent from parcel to LCL.
    p_lcl = p_parcel * (T_lcl / T_parcel) ** (constants.c_pd / constants.R_d)

    # Soft fractional level index where p_full = p_lcl.  Pressure
    # *decreases* with altitude (surface-first), so we apply
    # smooth_lowest_crossing_index to negated pressure to convert the
    # downward-with-altitude crossing into an upward one.
    k_lcl_smooth = smooth_lowest_crossing_index(
        -p_full, -p_lcl[:, None], crossing_sharpness,
    )

    return LCL(p_lcl=p_lcl, T_lcl=T_lcl, k_lcl_smooth=k_lcl_smooth)


# ---------------------------------------------------------------------------
# LFC and LNB
# ---------------------------------------------------------------------------

def compute_lfc_lnb(
    T_env: jax.Array,
    T_parcel_ma: jax.Array,
    *,
    sharpness: float = 1.0,
    q_v_env: jax.Array | None = None,
    q_v_parcel: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Smooth fractional levels of free convection and neutral buoyancy.

    The buoyancy proxy is the **virtual-temperature** difference
    ``T_v_parcel - T_v_env`` (positive where the parcel is lighter than
    the environment) when both humidity profiles are supplied — the same
    buoyancy definition as the virtual-T CAPE
    (:func:`~legoesm.atmosphere.physics.thermodynamics.compute_cape`) and
    the plume's ``B_u``, via the shared ``_shared.virtual_temperature``
    helper.  Without humidity the legacy dry-temperature proxy
    ``T_parcel_ma - T_env`` is used (the virtual-T correction is ~0.6 K
    for a 10 g/kg moist parcel — schemes gating a virtual-T CAPE should
    pass humidity so the LFC/LNB use the same buoyancy as the energy
    they bound).  The LFC is the lowest level where the proxy turns from
    negative to positive going upward, and the LNB is the lowest level
    above the LFC where the proxy turns back from positive to negative.

    Both are returned as smooth fractional indices in surface-last
    convention (``ncol, nlev`` indexing).  For columns with no clear
    crossing the no-crossing fallback in
    :func:`._triggers.smooth_lowest_crossing_index` returns a value
    near the surface (LFC) or near the top (LNB).

    Parameters
    ----------
    T_env : jax.Array, shape (ncol, nlev)
        Environmental temperature [K].
    T_parcel_ma : jax.Array, shape (ncol, nlev)
        Moist-adiabatic parcel temperature [K] from
        :func:`legoesm.atmosphere.physics.thermodynamics.compute_moist_adiabat`.
    sharpness : float
        Sigmoid sharpness in [1/K] on the buoyancy threshold.
    q_v_env, q_v_parcel : jax.Array, shape (ncol, nlev) or None
        Optional environment / parcel water-vapor mixing ratios [kg/kg].
        Pass BOTH for the virtual-temperature buoyancy.

    Returns
    -------
    k_lfc_smooth, k_lnb_smooth : jax.Array, shape (ncol,)
        Smooth fractional level indices.
    """
    # Sign convention (z up, surface-last): buoyancy > 0 where the
    # (virtual) parcel is warmer/lighter than the environment.
    if q_v_env is not None and q_v_parcel is not None:
        buoyancy = (
            virtual_temperature(T_parcel_ma, q_v_parcel)
            - virtual_temperature(T_env, q_v_env)
        )
    else:
        buoyancy = T_parcel_ma - T_env
    nlev = buoyancy.shape[-1]

    # LFC: lowest UPWARD crossing of buoyancy = 0.
    k_lfc = smooth_lowest_crossing_index(buoyancy, 0.0, sharpness)

    # LNB: lowest UPWARD crossing of negative-buoyancy.  Equivalently,
    # the lowest level (above the LFC) where the parcel turns from
    # positive to negative buoyancy.  Implementation: apply the same
    # primitive to ``-buoyancy`` AFTER the LFC is reached; the soft
    # gating-by-LFC weight ensures we don't pick up sub-cloud layers
    # where the parcel was negatively buoyant.
    #
    # Convention: ``profile`` and ``k_lfc`` are both surface-last indices
    # (surface at index ``nlev-1``, top at index 0).  "Above LFC altitude"
    # means *smaller* surface-last index than ``k_lfc``.  ``direction="below"``
    # in ``smooth_level_indicator`` gives ``sigmoid(threshold - profile)``
    # which is ~1 where ``profile < threshold`` — so the threshold IS
    # ``k_lfc`` itself.  The earlier formula ``(nlev - 1) - k_lfc`` flipped
    # ``k_lfc`` into surface-first space and then compared against the
    # surface-last ``profile``, marking the wrong levels as "above LFC"
    # and collapsing the LNB onto the LFC for many columns.
    # Sharp gate (independent of outer ``sharpness``) so the ``LARGE``
    # guard is essentially binary in the level axis: ~0 above LFC, ~1
    # below.  Re-using the K^-1 outer ``sharpness`` (typically O(1)) for
    # the level-axis indicator gives a sigmoid scale of ~1 *level*,
    # which combined with ``LARGE = 1e6`` leaks the guard far above LFC
    # and monotonises the profile so no upward-crossing is detected
    # (LNB then collapses onto the surface fallback).  Use ~20 so the
    # transition spans ~0.1 level.
    GATE_SHARPNESS = _PLUME_GATE_SHARPNESS
    above_lfc_weight = smooth_level_indicator(
        jnp.broadcast_to(jnp.arange(nlev, dtype=buoyancy.dtype), buoyancy.shape),
        threshold=k_lfc[:, None],
        sharpness=GATE_SHARPNESS,
        direction="below",
    )
    LARGE = jnp.asarray(1.0e6, dtype=buoyancy.dtype)
    guarded_neg_buoyancy = -buoyancy - LARGE * (1.0 - above_lfc_weight)
    k_lnb = smooth_lowest_crossing_index(guarded_neg_buoyancy, 0.0, sharpness)
    return k_lfc, k_lnb


# ---------------------------------------------------------------------------
# CIN
# ---------------------------------------------------------------------------

def compute_cin(
    T_env: jax.Array,
    T_parcel_ma: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    k_lcl_smooth: jax.Array,
    k_lfc_smooth: jax.Array,
    *,
    indicator_sharpness: float = 1.0,
    q_v_env: jax.Array | None = None,
    q_v_parcel: jax.Array | None = None,
) -> jax.Array:
    """Convective Inhibition (CIN) [J/kg].

    Integrates the negative buoyancy between the LCL and the LFC::

        CIN = R_d * ∫_{LCL}^{LFC} max(0, T_v_env - T_v_parcel) * dp/p

    When ``q_v_env`` and ``q_v_parcel`` are both supplied the deficit is
    the **virtual-temperature** difference — the same buoyancy
    definition as the virtual-T CAPE
    (:func:`~legoesm.atmosphere.physics.thermodynamics.compute_cape`)
    and the plume's ``B_u`` (shared ``_shared.virtual_temperature``
    helper), so the inhibition uses the same buoyancy as the energy it
    gates.  Without humidity the legacy dry-temperature form
    ``T_env - T_parcel`` is used.

    The bounds of integration are encoded as a smooth window in
    surface-last index space: ``window[k] = above_LCL(k) * below_LFC(k)``,
    each factor a sigmoid on the level index relative to the
    fractional indices.  Smooth bounds preserve gradients.

    Convention check: for a positively-CAPE column where the parcel
    is warmer than the environment between the LCL and the LFC,
    ``T_v_env - T_v_parcel`` is negative and the ``max(., 0)`` clamp
    gives zero.  CIN therefore counts only the genuinely-inhibiting
    layers.

    Parameters
    ----------
    T_env, T_parcel_ma : jax.Array, shape (ncol, nlev)
        Environmental and moist-adiabatic parcel temperatures [K].
    p_full, p_half : jax.Array
        Full-level (``ncol, nlev``) and half-level (``ncol, nlev+1``)
        pressures [Pa].
    k_lcl_smooth, k_lfc_smooth : jax.Array, shape (ncol,)
        Smooth fractional level indices from :func:`compute_lcl` and
        :func:`compute_lfc_lnb` (surface-last convention).
    indicator_sharpness : float
        Sigmoid sharpness on the level-window bounds, in [1/level].
    q_v_env, q_v_parcel : jax.Array, shape (ncol, nlev) or None
        Optional environment / parcel water-vapor mixing ratios [kg/kg].
        Pass BOTH for the virtual-temperature inhibition.

    Returns
    -------
    jax.Array, shape (ncol,)
        CIN [J/kg].  Non-negative.
    """
    nlev = T_env.shape[-1]
    levels = jnp.arange(nlev, dtype=T_env.dtype)
    levels = jnp.broadcast_to(levels, T_env.shape)

    # The integration window covers altitudes BETWEEN the LCL and the
    # LFC.  In surface-last index space (surface at the LARGEST index)
    # the LCL has a larger index than the LFC, so the CIN layer is at
    # indices ``k_lfc < k < k_lcl``.  The smooth window is the product
    # of two sigmoids: ``below LFC altitude`` (index larger than
    # ``k_lfc``) AND ``above LCL altitude`` (index smaller than
    # ``k_lcl``).  Earlier this product was the WRONG intersection
    # (``above_LFC AND below_LCL``) which is empty for the natural
    # ordering ``k_lnb < k_lfc < k_lcl`` — CIN was suppressed by ~93%
    # in straightforward test columns.
    window_below_lfc = jax.nn.sigmoid(
        indicator_sharpness * (levels - (k_lfc_smooth[:, None] + 0.5))
    )  # 1 at indices below LFC altitude (larger index), 0 above.
    window_above_lcl = jax.nn.sigmoid(
        indicator_sharpness * (k_lcl_smooth[:, None] - 0.5 - levels)
    )  # 1 at indices above LCL altitude (smaller index), 0 below.
    window = window_below_lfc * window_above_lcl

    dp = p_half[:, 1:] - p_half[:, :-1]
    # Sign convention (z up, surface-last): inhibition where the
    # ENVIRONMENT is (virtually) warmer than the parcel — the deficit
    # ``T_v_env - T_v_parcel`` is positive exactly where the parcel is
    # negatively buoyant, and the ``max(., 0)`` keeps CIN >= 0.
    if q_v_env is not None and q_v_parcel is not None:
        buoyancy_deficit = (
            virtual_temperature(T_env, q_v_env)
            - virtual_temperature(T_parcel_ma, q_v_parcel)
        )
    else:
        buoyancy_deficit = T_env - T_parcel_ma
    inhibiting_buoyancy = jnp.maximum(0.0, buoyancy_deficit)

    # Use the half-level midpoint pressure for the discrete ``∫ dlnp``
    # approximation (matches ``compute_cape`` after audit cycle iter-39
    # HIGH #1 fix and every sister physics helper —
    # ``_shared.compute_layer_dz``, ``mass_flux``, ``dca``).  The earlier
    # code used ``p_full`` (a layer-mean pressure on hybrid-sigma grids)
    # which produced a 0.5–2 % CIN bias relative to the matching CAPE.
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    return constants.R_d * jnp.sum(
        window * inhibiting_buoyancy * dp / p_mid, axis=-1,
    )


# ---------------------------------------------------------------------------
# Entraining-detraining updraft plume
# ---------------------------------------------------------------------------

class Plume(NamedTuple):
    """Output of :func:`entraining_detraining_plume`.

    All arrays are surface-last with shape ``(ncol, nlev)``.

    Fields
    ------
    M_u : jax.Array
        Updraft mass flux [kg/m²/s].  Non-negative; zero below the
        cloud base and at/above the level of neutral buoyancy.
    T_u : jax.Array
        Updraft temperature [K].
    q_u : jax.Array
        Updraft water-vapor specific humidity [kg/kg].
    q_c_u : jax.Array
        Updraft cloud-water mixing ratio [kg/kg].  Non-negative.
    B_u : jax.Array
        Updraft buoyancy ``T_u - T_env`` [K].  Negative aloft is the
        signal for the plume to terminate.
    """
    M_u: jax.Array
    T_u: jax.Array
    q_u: jax.Array
    q_c_u: jax.Array
    B_u: jax.Array


def entraining_detraining_plume(
    T_env: jax.Array,
    q_v_env: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    T_parcel_base: jax.Array,
    q_parcel_base: jax.Array,
    k_base_smooth: jax.Array,
    epsilon_profile: jax.Array,
    delta_profile: jax.Array,
    M_b: jax.Array,
    *,
    buoyancy_sharpness: float = 0.5,
    above_base_sharpness: float = _ABOVE_BASE_SHARPNESS,
    buoyancy_death_memory: bool = False,
    filter_negative_buoyancy: bool = True,
) -> Plume:
    """Bulk entraining-detraining updraft from cloud base to LNB.

    Integrates the standard plume budget upward from the cloud base
    using :func:`jax.lax.scan` over levels (surface-first ordering
    inside the scan).  At each layer the plume entrains environmental
    air at fractional rate ``epsilon`` and detrains plume air at
    rate ``delta`` (both with units [1/m]):

    .. math::

        \\frac{1}{M_u} \\frac{\\partial M_u}{\\partial z} = \\epsilon - \\delta

        \\frac{\\partial T_u}{\\partial z} = -\\frac{g}{c_{pd}} - \\epsilon (T_u - T_{env})

        \\frac{\\partial q_u}{\\partial z} = -\\epsilon (q_u - q_{v,env}) - C

    where ``C`` is the condensation rate (the surplus of the parcel's
    water vapor over its saturation value at the current level).
    Cloud water accumulates at rate ``C - precip_rate`` (precipitation
    is delegated to the calling scheme).

    The plume's effective extent is encoded smoothly: the mass-flux
    profile is multiplied by a sigmoid on the buoyancy ``T_u - T_env``
    so that the plume tapers off (rather than being abruptly
    truncated) once the parcel becomes negatively buoyant aloft.

    Parameters
    ----------
    T_env, q_v_env : jax.Array, shape (ncol, nlev)
        Environmental temperature [K] and vapor specific humidity
        [kg/kg].
    p_full, p_half : jax.Array
        Pressures [Pa] at full and half levels.
    z_full : jax.Array, shape (ncol, nlev)
        Geopotential heights [m] at full levels.
    T_parcel_base, q_parcel_base : jax.Array, shape (ncol,)
        Parcel temperature [K] and specific humidity [kg/kg] at the
        cloud-base launch level.
    k_base_smooth : jax.Array, shape (ncol,)
        Smooth fractional level index of the cloud base
        (surface-last).  Levels below the cloud base contribute
        zero plume mass flux.
    epsilon_profile, delta_profile : jax.Array, shape (ncol, nlev)
        Per-level fractional entrainment / detrainment rates [1/m].
    M_b : jax.Array, shape (ncol,)
        Cloud-base mass flux [kg/m²/s].
    buoyancy_sharpness : float
        Sharpness on the buoyancy-based plume-tapering sigmoid in
        [1/K].  Default ``0.5`` per Kelvin of buoyancy means the
        plume is at half mass flux when ``T_u - T_env`` reaches the
        modest negative value of about ``-1.4 K``.
    above_base_sharpness : float
        Sharpness of the sub-cloud (below cloud base) mass-flux gate in
        [1/level index] — applied to the level-index difference
        ``k_rev - k_base_rev``, NOT to a temperature.  Default
        ``_ABOVE_BASE_SHARPNESS`` (4.0): the reported mass flux is ~2 %
        one level below the cloud base and ~3e-4 two levels below.
        Deliberately a SEPARATE parameter from the Kelvin
        ``buoyancy_sharpness``: an earlier implementation reused the
        [1/K] value here, which made the level-space gate ~9 levels
        wide and leaked 27-38 % of ``M_b`` into the sub-cloud layer for
        every caller (same-name/different-units defect; mirrors
        ``compute_cin``'s dedicated ``indicator_sharpness`` [1/level]).
    buoyancy_death_memory : bool
        Whether the buoyancy-tapering filter has carry-state memory.
        Default ``False`` (legacy local filter): each level applies
        its own ``sigmoid(B_u)`` independently — a plume that is
        killed by negative buoyancy at one level can resume reporting
        nonzero ``M_u`` above an inversion, which can be physically
        inappropriate for single-plume schemes (audit Codex cycle 2
        P2: "plume terminated by negative buoyancy can revive above
        an inversion").

        When ``True``, the carry tracks two slots — ``launched``
        (cumulative max of a buoyancy-ramp gated by ``abv²``) and
        ``alive_min`` (cumulative min of ``launched·plume_alive +
        (1-launched)``) — so once the plume reaches its CAPE region
        and is then killed by an inversion, ``alive_min`` ratchets
        down and stays low even if buoyancy recovers above the
        inversion.

        **The True branch is opt-in and not enabled in any production
        scheme by default.**  Iterative review during cycle-3
        identified six independent edge-case failure modes (sub-LCL
        warm bubble, weak-CAPE miss, sub-LCL leak, missed cloud-base
        launch, revival-after-inversion still leaks).  The current
        ``buoyancy_ramp · abv²`` design satisfies all six in static
        traces, but the strongest cycle-3 finding ("cloud-base launch
        gate still does not block revival") suggests the cumulative-
        max + cumulative-min algorithm needs a more discrete state-
        machine treatment to fully suppress weak-CAPE revival.  The
        opt-in interface is preserved so a future PR can land a
        validated implementation; current callers stay on the
        legacy behaviour.
    filter_negative_buoyancy : bool
        Whether to apply the local ``sigmoid(B_u)`` reporting taper.
        Bulk schemes use the default ``True``.  Kain-Fritsch disables it
        because KF-Eta carries an explicit updraft vertical-velocity
        budget (``WTW``) and exits on ``WTW < 1e-3`` rather than on
        instantaneous negative buoyancy at a single level.

    Returns
    -------
    Plume
        Per-level mass flux, plume temperature, plume vapor, plume
        cloud water, and plume buoyancy.
    """
    ncol, nlev = T_env.shape

    # Pin everything to the input precision so saturation_mixing_ratio
    # (which promotes f32 → f64 via its Clausius-Clapeyron literal
    # constants) doesn't break ``jax.lax.scan``'s "carry-in dtype must
    # equal carry-out dtype" invariant.
    _dtype = T_env.dtype

    # Reverse to surface-first for the scan.
    T_env_rev = T_env[:, ::-1].astype(_dtype)
    q_v_env_rev = q_v_env[:, ::-1].astype(_dtype)
    p_full_rev = p_full[:, ::-1].astype(_dtype)
    z_full_rev = z_full[:, ::-1].astype(_dtype)
    eps_rev = epsilon_profile[:, ::-1].astype(_dtype)
    del_rev = delta_profile[:, ::-1].astype(_dtype)

    # Per-level above-base weight.  In surface-last indexing the cloud
    # base is at ``k_base_smooth``; levels with surface-last index
    # smaller than ``k_base_smooth`` are above the base.  In
    # surface-first reversed indexing the relationship inverts:
    # surface-first index ``k_rev`` corresponds to surface-last index
    # ``nlev - 1 - k_rev``, and "above cloud base" means
    # ``k_rev > (nlev - 1 - k_base_smooth)``.
    k_rev = jnp.arange(nlev, dtype=T_env.dtype)
    k_rev = jnp.broadcast_to(k_rev, T_env.shape)
    k_base_rev = (nlev - 1.0) - k_base_smooth
    # LEVEL-INDEX gate (``above_base_sharpness`` [1/level]), independent
    # of the Kelvin ``buoyancy_sharpness`` taper — see the kwarg doc.
    above_base_weight = jax.nn.sigmoid(
        above_base_sharpness * (k_rev - k_base_rev[:, None])
    )

    # Initial plume state at the surface-first index 0 (which is the
    # actual surface).  We launch with the parcel values; the
    # ``above_base_weight`` mask will suppress mass flux below cloud
    # base.  All carry components are pinned to ``_dtype``.
    #
    # ``alive_min`` tracks the cumulative MIN of ``plume_alive_local``
    # once the plume has reached its CAPE region (gated by
    # ``launched``).  ``launched`` is the cumulative MAX of a sharp
    # detector ``sigmoid(launched_sharpness · (plume_alive_local -
    # launched_threshold))`` — it ramps from 0 to 1 the first time
    # plume_alive_local clearly exceeds the threshold, then stays at 1.
    # Both start at 0/1 respectively (plume not launched, fully alive).
    init_carry = (
        T_parcel_base.astype(_dtype),                       # T_u_prev
        q_parcel_base.astype(_dtype),                       # q_u_prev
        jnp.zeros_like(T_parcel_base, dtype=_dtype),        # q_c_u_prev
        M_b.astype(_dtype),                                 # M_u_prev
        z_full_rev[:, 0],                                   # z_prev (already cast)
        jnp.ones_like(T_parcel_base, dtype=_dtype),         # alive_min
        jnp.zeros_like(T_parcel_base, dtype=_dtype),        # launched
    )

    # Per-level inputs to the scan.  Transpose to (nlev, ncol).
    inputs = (
        jnp.moveaxis(T_env_rev, 1, 0),
        jnp.moveaxis(q_v_env_rev, 1, 0),
        jnp.moveaxis(p_full_rev, 1, 0),
        jnp.moveaxis(z_full_rev, 1, 0),
        jnp.moveaxis(eps_rev, 1, 0),
        jnp.moveaxis(del_rev, 1, 0),
        jnp.moveaxis(above_base_weight, 1, 0),
    )

    g = constants.g

    def step(carry, layer_inputs):
        (T_u_prev, q_u_prev, q_c_u_prev, M_u_raw_prev,
         z_prev, alive_min_prev, launched_prev) = carry
        T_e, q_e, p_e, z_e, eps, dlt, abv = layer_inputs

        dz = jnp.maximum(z_e - z_prev, 1.0)  # ascending; floor to avoid div-by-zero

        # Raw plume mass flux: dM/dz = (epsilon - delta) * M.  Use the
        # exact integration ``M(z+dz) = M(z) * exp((eps - dlt) * dz)``
        # for this linear ODE — always positive, AD-safe everywhere,
        # and exact when ``(eps - dlt)`` is constant over the layer.
        # An earlier explicit-Euler form ``M * (1 + (eps - dlt) * dz)``
        # could go negative for strong detrainment + thick layers
        # (e.g. ``dlt = 5e-3 /m``, ``dz = 2000 m`` ⇒ multiplier =
        # ``-7``); the subsequent ``jnp.maximum(..., 0)`` clipped the
        # mass flux to 0 AND *zeroed the gradient* w.r.t. ``dlt`` /
        # ``eps``, breaking AD-based sensitivity studies through the
        # plume integrator (audit Codex finding: "plume mass flux uses
        # explicit Euler plus a hard nonnegative clip ... after
        # which jnp.maximum kills both mass flux and gradients").
        # We intentionally do NOT bake the buoyancy / sub-cloud masks
        # into the carry — those are reporting filters, not dynamics.
        # Folding them into the carry would compound across levels and
        # destroy the cloud-base-to-LNB profile that consumers expect.
        M_u_raw = M_u_raw_prev * jnp.exp((eps - dlt) * dz)

        # Entrainment of environmental T, q via the analytic relaxation
        # ``X(z+dz) = X_e + (X_prev - X_e) * exp(-eps · dz)`` — exact for
        # the linear ODE ``dX/dz = -eps · (X - X_e)`` and always bounded
        # between ``X_prev`` and ``X_e``.  An earlier explicit-Euler form
        # ``X_prev + eps · dz · (X_e - X_prev)`` overshoots past ``X_e``
        # for ``eps · dz > 1`` (e.g. Bechtold ``epsilon_shallow=3e-3``
        # with a 500–1000 m layer gives ``eps·dz ∈ [1.5, 3]``) — making
        # ``q_u_ent`` go negative and zeroing the gradient w.r.t. ``eps``.
        # The exponential form is dimensionally identical and AD-safe
        # everywhere (Codex stop-time review cycle 2: "plume entrainment
        # still uses unstable explicit Euler").
        decay_eps = jnp.exp(-eps * dz)
        T_u_ent = T_e + (T_u_prev - T_e) * decay_eps
        q_u_ent = q_e + (q_u_prev - q_e) * decay_eps

        # Use the analytic moist-adiabatic lapse rate from
        # ``moist_adiabat_lapse_rate`` (Iribarne–Godson) at the
        # entrained parcel state.  The function evaluates dT/dp on
        # the moist adiabat assuming the parcel is saturated; for
        # *unsaturated* parcels this is approximate but matches the
        # behavior of the existing ``compute_moist_adiabat`` helper
        # that we're benchmarked against.  Latent heating is already
        # baked into the lapse rate, so the post-hoc condensation
        # step below does NOT add an additional ``L_v/c_pd *
        # condensate`` correction — that would double-count.
        # Shared ideal-gas dry density (``_shared.compute_rho``; clips T
        # at 1 K — the previous inline form floored at 100 K, identical
        # for any physical plume temperature).
        rho_u_ent = compute_rho(T_u_ent, p_e)
        # ``moist_adiabat_lapse_rate`` returns dT/dp [K/Pa]; convert
        # to dT/dz [K/m] via dp/dz = -rho*g.
        Gamma_moist_per_pa = moist_adiabat_lapse_rate(T_u_ent, p_e)
        dT_dz = Gamma_moist_per_pa * (-rho_u_ent * g)

        # Entrainment-damped moist-adiabat source.  The plume temperature
        # obeys ``dT_u/dz = Γ_moist − ε·(T_u − T_e)``.  Integrating this
        # linear ODE across the layer (with ``T_e`` and ``Γ_moist`` held
        # constant) gives the entrainment relaxation ``T_u_ent`` PLUS the
        # source contribution ``Γ·(1 − e^{−ε·dz})/ε`` — **not** ``Γ·dz``.
        # The naive ``Γ·dz`` is correct only when ``ε·dz → 0``; for
        # deep-convection entrainment on a coarse grid ``ε·dz ≈ 1–2``, so
        # it over-applies the moist-adiabatic cooling by a factor
        # ``ε·dz/(1 − e^{−ε·dz}) ≈ 1.6–2``.  That over-cooled the updraft
        # until it lost buoyancy in the lower troposphere, leaving the
        # free troposphere unheated — the entraining-plume schemes (ZM,
        # KF, Emanuel, Bechtold) then collapsed to an anti-convective,
        # super-adiabatic, ~30 K-too-cold RCE.  The damped factor reduces
        # to ``dz`` as ``ε → 0`` (weakly-entraining behaviour preserved).
        eps_dz = eps * dz
        source_factor = jnp.where(
            eps_dz > 1e-6,
            -jnp.expm1(-eps_dz) / jnp.maximum(eps, 1e-30),
            dz,
        )
        T_u = T_u_ent + dT_dz * source_factor

        # Condense any super-saturation into cloud water.  This is the
        # diagnostic that resolves the q-budget; the temperature
        # already incorporates the latent heat from condensation via
        # the moist lapse rate.
        q_sat_new = saturation_mixing_ratio(T_u, p_e).astype(_dtype)
        condensate = jnp.maximum(q_u_ent - q_sat_new, 0.0).astype(_dtype)
        q_u = (q_u_ent - condensate).astype(_dtype)
        # Dilute plume cloud water by entrainment.  The continuity
        # equation for an intensive quantity in an entraining-
        # detraining plume is ``dq_c/dz = -eps · q_c + cond/M`` —
        # environmental air carries q_c=0 so entrainment uniformly
        # decreases ``q_c_u`` while detrainment is intensively
        # neutral (it removes mass but not the per-kg amount).  An
        # earlier formulation ``q_c_u = q_c_u_prev + condensate``
        # carried ``q_c_u_prev`` forward unchanged and the plume's
        # total water grew unphysically aloft (audit Codex finding:
        # "plume cloud water is accumulated but not diluted by
        # entrainment").
        # Exponential dilution: ``q_c_u_ent = q_c_u_prev * exp(-eps·dz)``
        # — exact analytic solution to ``dq_c/dz = -eps · q_c`` for an
        # entraining plume with environment q_c=0.  Always non-negative,
        # AD-safe everywhere.  An earlier explicit-Euler form
        # ``max(q_c_u_prev * (1 - eps·dz), 0)`` zeroed the gradient
        # whenever ``eps·dz > 1`` (corner case at Bechtold's
        # ``epsilon_shallow=3e-3`` × dz=500 m and thicker — the clip
        # branch dominated and made the test case ineffective for AD-
        # tuning of ``eps`` in shallow convection).
        q_c_u_ent = q_c_u_prev * decay_eps
        q_c_u = (q_c_u_ent + condensate).astype(_dtype)

        T_u = T_u.astype(_dtype)

        # Buoyancy at this level — use virtual temperature so that the
        # vapor-loading effect (moister plumes are lighter at fixed T)
        # is captured.  Both parcel and environment use the same
        # ``virtual_temperature`` helper from ``_shared.py`` so the ε
        # convention is centralised and audit-consistent with the
        # turbulence / PBL paths.  The plume cloud-water mass loading
        # ``q_c_u`` is folded in as a density-temperature correction
        # ``T_v_u·(1 − q_c_u)``; environmental ``q_c = 0`` so the
        # correction is one-sided (positive ``q_c_u`` reduces plume
        # buoyancy ~0.1–0.3 K, the canonical "water loading" effect
        # known to suppress weak-CAPE convection).  Previous form
        # ``T_u − T_e`` used dry temperature and missed both effects;
        # virtual-T contribution alone is ~0.6 K for a 10 g/kg moist
        # parcel, i.e. comparable to the ``1/buoyancy_sharpness = 2 K``
        # plume-alive threshold.
        T_v_u = virtual_temperature(T_u, q_u) * (1.0 - q_c_u)
        T_v_e = virtual_temperature(T_e, q_e)
        B_u = T_v_u - T_v_e

        # Reporting filters: smoothly suppress the mass flux below
        # cloud base (``abv``) and where the plume has lost buoyancy
        # (``plume_alive``).
        #
        # When ``buoyancy_death_memory`` is False (default), the
        # filter is purely local: each level applies its own
        # sigmoid(B_u), so a plume killed by an inversion can revive
        # above it.  The carry does NOT track the filter.
        #
        # When True, the death-memory carry has TWO slots:
        #
        # 1. ``launched``: cumulative max of a smooth ramp on
        #    ``plume_alive_local`` that engages PROPORTIONALLY to
        #    positive buoyancy AND only above cloud base:
        #
        #        launched_local = abv · clamp(2·max(pl - 0.5, 0), 0, 1)
        #
        #    The two factors gate independently:
        #    - ``2 · max(pl - 0.5, 0)`` is the buoyancy ramp: 0 for
        #      ``B_u ≤ 0``, scaling linearly with positive B_u up to
        #      saturation at ``B_u → ∞``.  Captures ALL positive-CAPE
        #      regions, including weak CAPE (an earlier sharp
        #      threshold required ``B_u > 1.7 K`` to engage at all,
        #      missing weak-CAPE columns — Codex cycle-3 follow-up
        #      "launched gate misses weak positive CAPE").
        #    - ``abv`` is the above-cloud-base weight: 0 below LCL,
        #      1 above.  Without this factor, a sub-LCL warm bubble
        #      (e.g. a cooler-air-aloft-over-warm-BL profile) could
        #      drive ``B_u > 0`` briefly at the surface and trigger
        #      ``launched`` prematurely, causing the LCL-to-LFC CIN
        #      passage to ratchet alive_min before the plume reaches
        #      its actual CAPE region (Codex cycle-3 follow-up:
        #      "launch gate is not masked to cloud base/LFC").
        #
        #    Cumulative max ⇒ once the plume reaches its CAPE region,
        #    ``launched_new`` stays at the highest value seen.
        #
        # 2. ``alive_min``: cumulative min of ``plume_alive_local``,
        #    GATED by ``launched`` so it only tracks deaths AFTER the
        #    plume has reached its CAPE region.  Pre-LFC the gate is
        #    0 ⇒ ratchet target = 1 ⇒ alive_min preserved.  Above
        #    LFC the gate is launched_new ∈ (0, 1] ⇒ ratchet target
        #    interpolates between ``plume_alive`` (full ratchet) and
        #    1 (no ratchet) by buoyancy strength.  Once an above-LFC
        #    inversion kills the plume (alive_min drops), it stays
        #    dead even if buoyancy recovers above the inversion
        #    (audit Codex cycle 2 P2: "plume terminated by negative
        #    buoyancy can revive above an inversion").
        if filter_negative_buoyancy:
            plume_alive_local = jax.nn.sigmoid(buoyancy_sharpness * B_u)
        else:
            plume_alive_local = jnp.ones_like(B_u)
        if buoyancy_death_memory:
            # Buoyancy ramp: 0 for B_u ≤ 0, scales linearly above.
            # ``relu`` is a smooth-enough subgradient for AD.
            buoyancy_ramp = jnp.minimum(
                2.0 * jax.nn.relu(plume_alive_local - 0.5), 1.0,
            )
            # Above-cloud-base weighting for the launched gate.  The
            # plain ``abv`` factor leaks ~2 % of the buoyancy ramp at
            # the sub-LCL transition zone (Codex cycle-3 "sub-LCL
            # launch gate is still leaky"), but the harder
            # ``max(2·abv - 1, 0)`` zeros the gate exactly AT cloud
            # base and misses genuine cloud-base launches (Codex
            # cycle-3 "the new launch mask can miss a real cloud-
            # base launch").  ``abv²`` is the right trade-off:
            #
            #   abv = 0.1  → abv² = 0.01   (very small sub-LCL leak)
            #   abv = 0.3  → abv² = 0.09   (still small)
            #   abv = 0.5  → abv² = 0.25   (cloud-base launch engages
            #                                at 25 % — partial but
            #                                non-zero, captures the
            #                                launch event)
            #   abv = 0.7  → abv² = 0.49   (half engagement)
            #   abv = 1.0  → abv² = 1.0    (full above cloud base)
            #
            # This gives a quadratic suppression of sub-LCL leaks
            # while preserving non-zero engagement at cloud base
            # itself.  Cumulative max ⇒ once a level engages the
            # launched gate, subsequent levels can only hold or
            # increase it.
            launched_abv_mask = abv * abv
            launched_local = buoyancy_ramp * launched_abv_mask
            launched_new = jnp.maximum(launched_prev, launched_local)
            # Ratchet target: blend plume_alive_local (when launched)
            # with 1 (when not launched).
            ratchet_target = (
                launched_new * plume_alive_local + (1.0 - launched_new)
            )
            alive_min_new = jnp.minimum(alive_min_prev, ratchet_target)
        else:
            alive_min_new = jnp.ones_like(alive_min_prev)
            launched_new = jnp.zeros_like(launched_prev)
        plume_alive_filter = (
            alive_min_new if buoyancy_death_memory else plume_alive_local
        )
        M_u_reported = M_u_raw * plume_alive_filter * abv

        new_carry = (T_u, q_u, q_c_u, M_u_raw, z_e, alive_min_new, launched_new)
        outputs = (T_u, q_u, q_c_u, M_u_reported, B_u)
        return new_carry, outputs

    _, scan_out = jax.lax.scan(step, init_carry, inputs)

    T_u_rev, q_u_rev, q_c_u_rev, M_u_rev, B_u_rev = scan_out  # (nlev, ncol)

    # Move axis back and reverse to surface-last.
    T_u = jnp.moveaxis(T_u_rev, 0, 1)[:, ::-1]
    q_u = jnp.moveaxis(q_u_rev, 0, 1)[:, ::-1]
    q_c_u = jnp.moveaxis(q_c_u_rev, 0, 1)[:, ::-1]
    M_u = jnp.moveaxis(M_u_rev, 0, 1)[:, ::-1]
    B_u = jnp.moveaxis(B_u_rev, 0, 1)[:, ::-1]

    return Plume(M_u=M_u, T_u=T_u, q_u=q_u, q_c_u=q_c_u, B_u=B_u)


# ---------------------------------------------------------------------------
# Convective momentum transport (Gregory et al. 1997)
# ---------------------------------------------------------------------------

def cmt_gregory_1997(
    u_env: jax.Array,
    v_env: jax.Array,
    M_u: jax.Array,
    M_d: jax.Array | None,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    *,
    c_u: float = _PLUME_C_U,
    c_d: float = _PLUME_C_D,
) -> tuple[jax.Array, jax.Array]:
    """Gregory et al. 1997 convective momentum transport closure.

    The eddy-flux closure carries plume momentum aloft minus a
    pressure-gradient correction that limits the upward transport in
    sheared environments.  In bulk-plume form ::

        F_u = M_u * (u_u - u_env) - c_u * M_u * (du/dz)_layer * dz_layer

    and the resulting tendency is ``du/dt = -(1/rho) * dF/dz``.  The
    same form applies to the downdraft with sign convention
    ``M_d < 0`` and parameter ``c_d``.

    The implementation here is the **simplified bulk closure**: we
    approximate ``u_u`` by the environmental wind at the cloud base
    plus a fraction of the layer-by-layer environmental shear,
    yielding a numerically stable form that does not require a
    separate plume-momentum integrator.  This is the formulation
    used in CESM/CAM with the ZM scheme and follows Gregory et al.
    1997 Eq. 14.

    Parameters
    ----------
    u_env, v_env : jax.Array, shape (ncol, nlev)
        Environmental zonal / meridional wind [m/s].
    M_u : jax.Array, shape (ncol, nlev)
        Updraft mass flux [kg/m²/s].  Non-negative.
    M_d : jax.Array or None, shape (ncol, nlev)
        Downdraft mass flux [kg/m²/s] (negative by convention).
        ``None`` skips the downdraft contribution.
    p_full, p_half : jax.Array
        Pressures [Pa] at full / half levels.  Used to compute layer
        thickness ``dp = p_half[1:] - p_half[:-1]``.
    rho : jax.Array, shape (ncol, nlev)
        Air density [kg/m³].
    c_u, c_d : float
        Pressure-gradient correction coefficients in Gregory et al.
        1997 Eq. 12.  ``0.55`` is the canonical value (range
        ``0.3–0.7`` in the literature).

    Returns
    -------
    du_dt, dv_dt : jax.Array, shape (ncol, nlev)
        Convective momentum tendencies [m/s²].
    """
    # Layer pressure thickness; with surface-last convention dp > 0.
    dp = p_half[:, 1:] - p_half[:, :-1]

    # Stratospheric mass-flux gate — same factor the kernel applies to
    # T/q_v tendencies.  Without this, the CMT path detrains
    # convective momentum into the model top (where the plume should
    # already be dead), producing wind-driven dycore blowups (e.g.
    # KF at day 10 in 1-year lat-lon FV RCE).  Imported lazily to
    # avoid a circular import (`mass_flux` imports from `_plume`).
    from legoesm.atmosphere.physics.convection.mass_flux import (
        stratosphere_mass_flux_gate,
    )
    p_gate = stratosphere_mass_flux_gate(p_full)
    M_u = M_u * p_gate
    if M_d is not None:
        M_d = M_d * p_gate

    # Environmental shear (forward difference per layer).  Edge layers
    # use one-sided differences via padded edges to keep shape
    # ``(ncol, nlev)``.
    du_layer = jnp.diff(u_env, axis=-1, prepend=u_env[:, :1])
    dv_layer = jnp.diff(v_env, axis=-1, prepend=v_env[:, :1])

    # Eddy momentum flux from Gregory et al. 1997 §3.  In the bulk
    # plume approximation:
    #     M_u (u_u - u_env) ≈ -c_u * M_u * du/dz_layer
    # i.e. only the pressure-gradient correction term contributes —
    # the leading ``M_u * u_env`` mass-transport term cancels when
    # the mass-flux divergence is also accounted for in the
    # large-scale momentum equation.  As a consequence a uniform
    # wind column produces zero CMT regardless of mass flux.
    flux_u_up = -c_u * M_u * du_layer
    flux_v_up = -c_u * M_u * dv_layer

    if M_d is not None:
        flux_u_down = -c_d * M_d * du_layer
        flux_v_down = -c_d * M_d * dv_layer
    else:
        flux_u_down = jnp.zeros_like(flux_u_up)
        flux_v_down = jnp.zeros_like(flux_v_up)

    flux_u = flux_u_up + flux_u_down
    flux_v = flux_v_up + flux_v_down

    # Vertical divergence of the flux: ``du/dt = -(1/rho) * dF/dz``.
    # Using hydrostatic ``dz = -dp/(rho*g)`` gives
    # ``du/dt = -(g) * dF/dp`` after the rho cancels.
    dflux_u = jnp.diff(flux_u, axis=-1, append=flux_u[:, -1:])
    dflux_v = jnp.diff(flux_v, axis=-1, append=flux_v[:, -1:])

    du_dt = -constants.g * dflux_u / dp
    dv_dt = -constants.g * dflux_v / dp

    return du_dt, dv_dt
