"""Mechanical ridging — convergence-driven ITD redistribution.

Implements the Lipscomb (2007) ridging closure: when the strain-rate
deformation invariant ``Δ`` indicates net convergence (negative
divergence), thin ice "participates" in ridging and is converted
into ridged ice that lands in the thicker portion of the ITD.

Participation function (exponential, Lipscomb 2007 eq. 22):

    b(h) ∝ exp(−h / e*)     with e* ≈ 0.36 m

normalised so that ``∫ b · g_cat dh = 1`` over thin / unridged
categories.

Ridge transfer function (Hibler 1980 / Lipscomb 2007 eq. 26):
ridged ice from participating categories is spread uniformly in
``h`` between

    H_min = 2 · h_part                     (minimum thickness multiplier)
    H_max = min(μ_rdg · √(h_part), H_star) (Hibler scaling, capped at H_star)

Snow on participating ice is partly retained in ridges
(``snow_fraction_retained``); the rest is dropped to the ocean as
freshwater.

The full Lipscomb 2007 algorithm tracks every (donor, receiver)
category pair.  The implementation below operates column-wise via
``jax.vmap`` and is conservation-exact for ice area and ice
volume (snow + salt mass conservation: see ``apply_ridging``
docstring).

Faithfulness
------------
``tests/ice/unit/test_ice_ridging_faithful.py`` pins:

  * :func:`participation_weights` to round-off (rel 1e-9) against an independent
    reimplementation of Lipscomb 2007 eq. 22 (``b_k = a_k exp(-h_k/e*) / sum``),
    plus its defining properties (normalised to 1 with ice, thin-preferential
    ``w/a`` strictly decreasing in ``h``, zero when ice-free, the ``e*`` divide
    floor);
  * the Hibler 1980 / Lipscomb eq. 26 ridge-thickness range via a single-donor
    column with analytic ``h_part``: the per-receiver-bin area and volume match
    the independent uniform-``g`` overlap integral on ``[H_min, H_max] =
    [2 h_part, min(mu_rdg sqrt(h_part), H_star)]`` (mean thickness ``H_mean``),
    including the ``H_star`` cap and the two numerical REGULARIZATIONS: when
    ``mu sqrt(h_part) < 2 h_part`` (i.e. ``h_part > (mu/2)^2``) ``H_max`` is
    floored to ``H_min + width``, and for an over-thick donor
    (``H_min > hi[-1]``) the range collapses into the top category — both remain
    volume-conserving;
  * the DEFINING conservation invariants of :func:`apply_ridging` (the truth
    tier): ice volume and bulk salt mass conserved, total area reduced by exactly
    ``closing_rate*dt`` (CICE aksum) in the un-capped regime, saturating when the
    total area is exhausted, donor snow deficit reported to the ocean (retained
    fraction = ``snow_fraction_retained``), ridging pond water fully drained,
    divergent columns a no-op; the effective production defaults are exercised by
    a default-vs-explicit equivalence test.

Closure constants (e_star, mu_rdg, H_star, snow_fraction_retained) are canaried
against :class:`RidgingConfig` defaults (0.36 m, 4.0, 100 m, 0.5).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ice.config import RidgingConfig
from legoesm.ice.itd import category_bounds, upper_bounds

from legoesm import constants

# Canonical ridging defaults live on RidgingConfig (single source of truth);
# kwarg signatures default to these. Salt mass uses S [PSU = g/kg] x volume x
# rho_ice [kg/m^3]; the 1e-3 converts g/kg -> kg/kg so salt is in kg.
_RIDGE_DEFAULTS = RidgingConfig()
_PSU_TO_FRACTION = 1.0e-3
# Minimum ridge-thickness range width [m]: keeps H_max strictly above H_min so
# the uniform-g overlap integral has a finite, well-defined support.
_MIN_RIDGE_WIDTH_M = 1.0e-3

# Selectable SI3 ORCA1/jpl=1 arm.  Every value is the resolved rung-3.4 deck,
# not a new production default: namelist_ice_cfg:52-66 over
# namelist_ice_ref:87-102.  The fixed factors are SI3 source constants at
# icedyn_rdgrft.F90:77-78; tolerances and iteration limit are :169 and the
# NEMO epsi10/epsi20 kinds used at :429-430,562,572,594,692.
_SI3_JPL1_SCHEME = "si3_orca1_jpl1"
_SI3_JPL1_CATEGORY_COUNT = 1
_SI3_JPL1_CS_RIDGING = 0.5
_SI3_JPL1_ASTAR = 0.03
_SI3_JPL1_HSTAR_M = 25.0
_SI3_JPL1_MU_RIDGING = 3.0
_SI3_JPL1_HRAFT_M = 0.75
_SI3_JPL1_CRAFT = 5.0
_SI3_JPL1_POROSITY = 0.0
# The deck supplies 0.5, but this rung resolves ln_icethd=F.  SI3 therefore
# overwrites all four factors to one at icedyn_rdgrft.F90:1244-1247 before the
# first call.  These constants name the executed selector state.
_SI3_JPL1_SNOW_RIDGE_RETENTION = 1.0
_SI3_JPL1_SNOW_RAFT_RETENTION = 1.0
_SI3_JPL1_POND_RIDGE_RETENTION = 1.0
_SI3_JPL1_POND_RAFT_RETENTION = 1.0
_SI3_JPL1_RIDGE_MIN_MULTIPLIER = 1.1
_SI3_JPL1_RAFT_AREA_MULTIPLIER = 0.5
_SI3_JPL1_MAX_ITERATIONS = 19
_SI3_JPL1_EPSI10 = 1.0e-10
_SI3_JPL1_EPSI20 = 1.0e-20
_SI3_JPL1_ZERO = 0.0
_SI3_JPL1_ONE = 1.0
_SI3_JPL1_TWO = 2.0


@jax.custom_jvp
def _si3_jpl1_sqrt(value: jnp.ndarray) -> jnp.ndarray:
    """Exact-forward square root with a finite zero-state tangent."""

    return jnp.sqrt(value)


@_si3_jpl1_sqrt.defjvp
def _si3_jpl1_sqrt_jvp(primals, tangents):
    (value,), (value_tangent,) = primals, tangents
    root = jnp.sqrt(value)
    safe_root = jnp.where(root > _SI3_JPL1_ZERO, root, _SI3_JPL1_ONE)
    root_tangent = jnp.where(
        root > _SI3_JPL1_ZERO,
        _SI3_JPL1_RAFT_AREA_MULTIPLIER * value_tangent / safe_root,
        _SI3_JPL1_ZERO,
    )
    return root, root_tangent


class SI3JPL1RidgingConfig(NamedTuple):
    """Only the ORCA1-resolved SI3 redistribution selector composition."""

    scheme: str = _SI3_JPL1_SCHEME
    category_count: int = _SI3_JPL1_CATEGORY_COUNT
    exponential_distribution: bool = True
    exponential_participation: bool = True
    ridging: bool = True
    rafting: bool = True
    cs_ridging: float = _SI3_JPL1_CS_RIDGING
    astar: float = _SI3_JPL1_ASTAR
    hstar_m: float = _SI3_JPL1_HSTAR_M
    mu_ridging: float = _SI3_JPL1_MU_RIDGING
    hraft_m: float = _SI3_JPL1_HRAFT_M
    craft: float = _SI3_JPL1_CRAFT
    porosity: float = _SI3_JPL1_POROSITY
    snow_ridge_retention: float = _SI3_JPL1_SNOW_RIDGE_RETENTION
    snow_raft_retention: float = _SI3_JPL1_SNOW_RAFT_RETENTION
    pond_ridge_retention: float = _SI3_JPL1_POND_RIDGE_RETENTION
    pond_raft_retention: float = _SI3_JPL1_POND_RAFT_RETENTION


class SI3JPL1RidgingState(NamedTuple):
    """SI3 jpl=1 carried redistribution state, all on T points."""

    ice_area: jnp.ndarray
    open_water_area: jnp.ndarray
    ice_volume: jnp.ndarray
    snow_volume: jnp.ndarray
    age_content: jnp.ndarray
    pond_area: jnp.ndarray
    pond_volume: jnp.ndarray
    pond_lid_volume: jnp.ndarray
    snow_enthalpy: jnp.ndarray
    ice_enthalpy: jnp.ndarray
    ice_salt_content: jnp.ndarray


class SI3JPL1RidgingLosses(NamedTuple):
    """Per-cell material sent to the ocean by the pinned SI3 arm."""

    snow_volume: jnp.ndarray
    snow_enthalpy: jnp.ndarray
    pond_volume: jnp.ndarray
    pond_lid_volume: jnp.ndarray
    iterations: jnp.ndarray
    excessive_removal_clamp: jnp.ndarray
    open_water_correction: jnp.ndarray


def participation_weights(
    a_cat: jnp.ndarray,
    h_cat: jnp.ndarray,
    e_star: float,
) -> jnp.ndarray:
    """Exponential ridging participation, normalised.

    Returns the fraction of each category that participates in
    ridging this step (per-category, summing to one over the
    column).  Thin categories are ridged preferentially; the
    e-folding thickness ``e_star`` ≈ 0.36 m gives a > 80 %
    participation of < 1 m ice (Lipscomb 2007 default).

    Parameters
    ----------
    a_cat : array ``(n_cat,)`` or ``(..., n_cat)``
        Per-category area fraction.
    h_cat : array
        Per-category mean thickness [m].
    e_star : float
        e-folding thickness for participation [m].

    Returns
    -------
    weights : array (same shape as input)
        Per-category participation fraction (sums to 1 along the
        trailing axis when there is any ice; zero everywhere when
        the column is ice-free).
    """
    raw = a_cat * jnp.exp(
        -h_cat / jnp.maximum(e_star, 1e-3)  # coeff-ok: e_star divide-safety floor [m]
    )
    total = jnp.sum(raw, axis=-1, keepdims=True)
    safe_total = jnp.where(total > 1e-30, total, 1.0)
    return jnp.where(total > 1e-30, raw / safe_total, 0.0)


def _ridging_column_kernel(
    a_cat: jnp.ndarray,
    h_cat: jnp.ndarray,
    V_snow_cat: jnp.ndarray,
    S_ice_cat: jnp.ndarray,
    V_pond_cat: jnp.ndarray,
    closing_rate: jnp.ndarray,
    lo: jnp.ndarray,
    hi: jnp.ndarray,
    dt: float,
    e_star: float,
    mu_rdg: float,
    H_star: float,
    snow_fraction_retained: float,
) -> tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    """Per-column ridging kernel.

    Returns updated ``(a_cat, h_cat, V_snow_cat, S_ice_cat, V_pond_cat,
    snow_to_ocean_kg_m2_per_s_cell, pond_to_ocean_kg_m2_per_s_cell)`` arrays.
    Melt-pond water on ridging ice DRAINS to the ocean (deformation destroys
    the pond surface; CICE convention) — conserving water, so ridges carry no
    pond and the drained volume is reported as a freshwater flux.
    """
    # Volume per category.
    V_cat = h_cat * a_cat

    # Closing rate must be non-negative for ridging to fire; if the
    # column is purely divergent there is no ridging.
    closing_rate = jnp.maximum(closing_rate, 0.0)

    # Area available to ridge this step (cap at total area present).
    a_total = jnp.sum(a_cat)
    da_ridged_request = closing_rate * dt

    # Participation weights (per category) — fraction of total
    # ridging area drawn from each category (normalised, sum to 1).
    weights = participation_weights(a_cat, h_cat, e_star)

    # aksum normalization (audit): the redistribution compresses the
    # participating area ``a_part`` into a SMALLER ridge area ``a_ridge =
    # a_part * h_part / H_mean`` (ridges are thicker than donors), so the NET
    # area removed is only ``a_part * (1 - h_part/H_mean)``.  Without correcting
    # for this the net closing under-delivers (50-100% of the requested
    # ``closing_rate*dt``) and the documented ``ΔA = -closing_rate*dt`` invariant
    # is false.  Scale the participating draw up by ``1/(1 - h_part/H_mean)`` so
    # the net closing MATCHES the dynamics-requested value (CICE ``aksum``).
    # ``h_part`` / ``H_mean`` depend only on the participation weights and
    # category thicknesses (NOT on the draw magnitude), so the compression
    # factor is computed here from the weights before the draw; the exact
    # per-cat-capped ``h_part`` / ``H_mean`` are recomputed below for the ridge
    # distribution.  ``1 - h_part/H_mean`` is bounded in [0.5, 1] since
    # ``H_mean >= H_min = 2*h_part`` (clipped for the over-thick collapse case).
    # float32 AD safety: ``participation_weights`` returns EXACTLY zero for an
    # ice-free column (its own two-sided where, L111-113), so this
    # UNCONDITIONAL floored divide produced a NaN adjoint in float32
    # (``integer_pow(1e-30, -2) == inf``; ``-0 * inf == NaN``).  Safe
    # denominator inside the branch; fallback 0.0 == the old floored value
    # (0/1e-30).  Every OTHER divide in this kernel already uses this idiom.
    _w_sum = jnp.sum(weights)
    _has_participation = _w_sum > 1e-30
    _w_sum_safe = jnp.where(_has_participation, _w_sum, 1.0)
    h_part_est = jnp.where(
        _has_participation,
        jnp.sum(weights * h_cat) / _w_sum_safe,
        0.0,
    )
    H_min_est = 2.0 * h_part_est
    H_max_est = jnp.minimum(mu_rdg * jnp.sqrt(jnp.maximum(h_part_est, 1e-6)), H_star)
    H_max_est = jnp.minimum(H_max_est, hi[-1])
    H_max_est = jnp.maximum(H_max_est, H_min_est + _MIN_RIDGE_WIDTH_M)
    H_mean_est = 0.5 * (jnp.minimum(H_min_est, hi[-1]) + jnp.minimum(H_max_est, hi[-1]))
    comp = jnp.clip(
        1.0 - h_part_est / jnp.maximum(H_mean_est, 1e-6),
        0.5,
        1.0,
    )
    da_ridged = jnp.minimum(da_ridged_request / comp, a_total)

    da_per_cat = da_ridged * weights
    # Limit per-cat draw to its available area.
    da_per_cat = jnp.minimum(da_per_cat, a_cat)

    # Volume + snow removed per donor cat.
    fraction_taken = jnp.where(
        a_cat > 1e-30,
        da_per_cat / jnp.where(a_cat > 1e-30, a_cat, 1.0),
        0.0,
    )
    dV_per_cat = V_cat * fraction_taken
    dVsnow_per_cat = V_snow_cat * fraction_taken
    # Melt-pond water on the ridging ice drains entirely to the ocean
    # (ridging deformation destroys the pond surface); ridges carry no pond.
    dVpond_per_cat = V_pond_cat * fraction_taken
    # Carry salt mass into ridged ice: per-cat salt mass = S_ice * V * rho_ice * 1e-3
    # Salt mass moves with the ice — keep S_ice on transfer (mass moves, S unchanged).
    # No explicit salt term needed here; S_ice is reconstructed below.

    # Compute participating mean thickness h_part (volume-weighted).
    V_part_total = jnp.sum(dV_per_cat)
    a_part_total = jnp.sum(da_per_cat)
    h_part = jnp.where(
        a_part_total > 1e-30,
        V_part_total / jnp.where(a_part_total > 1e-30, a_part_total, 1.0),
        0.0,
    )

    # Ridge thickness range (Hibler / Lipscomb).
    # The fixed categories contiguously tile ``[lo[0], hi[-1]] = [0, 100] m``.
    # The overlap integral below conserves ridge area+volume ONLY when the WHOLE
    # range ``[H_min, H_max]`` lies inside that support, so ``sum_j overlap_frac
    # == 1``.  Clamp BOTH endpoints into ``[lo[0], hi[-1]]`` (finding #7 + codex
    # R2-1): clamping only the CEILING was insufficient — when participating ice
    # is so thick that ``H_min = 2*h_part > hi[-1]`` (and the ``max(H_max,
    # H_min+width)`` floor then lifts H_max back above hi[-1]), the range sits
    # entirely ABOVE every category and ALL the ridged volume is dropped.
    # Clamping H_min down into the top bin routes such an over-thick ridge into
    # the top category while preserving ``sum(dV_ridge_to_cat) == V_part_total``.
    H_min = 2.0 * h_part
    H_max = jnp.minimum(mu_rdg * jnp.sqrt(jnp.maximum(h_part, 1e-6)), H_star)
    H_max = jnp.minimum(H_max, hi[-1])  # ceiling at top bound
    # NORMAL case (H_min + width <= hi[-1]): keep the original Lipscomb range,
    # flooring H_max above the physical H_min = 2*h_part.  This preserves the
    # ridge thickness distribution for ordinary thin/mixed/thick states (the
    # round-2 unconditional ``clip(H_min, lo[0], H_max-width)`` wrongly thinned
    # the ridge and inflated ridge area whenever mu*sqrt(h_part) < 2*h_part,
    # i.e. h_part > (mu/2)^2; codex R3-1).
    H_max_normal = jnp.maximum(H_max, H_min + _MIN_RIDGE_WIDTH_M)
    # OVER-THICK defensive case (H_min would exceed the top bound hi[-1]): the
    # whole physical range sits above every category, so collapse it into the
    # TOP bin [hi[-1]-width, hi[-1]] -> the entire ridge routes to the top
    # category while ``sum(dV_ridge_to_cat) == V_part_total`` (codex R2-1).
    over_thick = H_min > (hi[-1] - _MIN_RIDGE_WIDTH_M)
    H_min = jnp.where(over_thick, hi[-1] - _MIN_RIDGE_WIDTH_M, H_min)
    H_max = jnp.where(over_thick, hi[-1], H_max_normal)
    H_width = H_max - H_min

    # Snow donated by participating ice.  Fraction retained in
    # ridge, the rest to ocean as freshwater.
    Vsnow_donated = jnp.sum(dVsnow_per_cat)
    Vsnow_in_ridge = snow_fraction_retained * Vsnow_donated
    snow_to_ocean_m_snow_per_m2 = (1.0 - snow_fraction_retained) * Vsnow_donated
    # Convert snow-depth volume [m of snow per m²] to mass [kg/m²] via
    # rho_snow (caller divides by dt for a flux):
    snow_to_ocean_kg_m2 = snow_to_ocean_m_snow_per_m2 * constants.rho_snow

    # All ridged pond water drains to the ocean as fresh liquid water.
    pond_to_ocean_m_per_m2 = jnp.sum(dVpond_per_cat)
    pond_to_ocean_kg_m2 = pond_to_ocean_m_per_m2 * constants.rho_water

    # Ridged volume — area is compressed by factor h_part / H_mean,
    # where H_mean = (H_min + H_max) / 2 — preserves volume since
    # ridges are thicker than donors.
    H_mean = 0.5 * (H_min + H_max)
    a_ridge_total = jnp.where(
        H_mean > 1e-6,
        V_part_total / jnp.where(H_mean > 1e-6, H_mean, 1.0),
        0.0,
    )

    # Distribute the ridge into FIXED categories using the
    # uniform-in-h transfer function on [H_min, H_max] (Lipscomb
    # 2007).  Per fixed cat j with bounds [lo[j], hi[j]]:
    #   area share = overlap(H_min..H_max, lo[j]..hi[j]) / H_width
    a_over = jnp.maximum(lo, H_min)
    b_over = jnp.minimum(hi, H_max)
    overlap = jnp.maximum(b_over - a_over, 0.0)
    overlap_frac = overlap / jnp.maximum(H_width, 1e-6)
    da_ridge_to_cat = a_ridge_total * overlap_frac

    # Mean thickness inside the j-th fixed cat under uniform g(h):
    # h_mean_j = 0.5 * (a_over + b_over)
    h_mean_in_cat = 0.5 * (a_over + b_over)
    dV_ridge_to_cat = da_ridge_to_cat * h_mean_in_cat
    # Snow that survives the ridge spreads in proportion to V.
    V_ridge_total = jnp.sum(dV_ridge_to_cat)
    Vsnow_to_cat = jnp.where(
        V_ridge_total > 1e-30,
        Vsnow_in_ridge * dV_ridge_to_cat / jnp.where(V_ridge_total > 1e-30, V_ridge_total, 1.0),
        0.0,
    )
    # Salt mass goes with V (per-cat ridge salt = S_part_mean * dV_ridge_to_cat).
    salt_donated_total = (
        jnp.sum(
            S_ice_cat * dV_per_cat,
        )
        * constants.rho_ice
        * _PSU_TO_FRACTION
    )
    salt_to_cat = jnp.where(
        V_ridge_total > 1e-30,
        salt_donated_total * dV_ridge_to_cat / jnp.where(V_ridge_total > 1e-30, V_ridge_total, 1.0),
        0.0,
    )

    # Update donor-side state (subtract per-cat draws).
    a_after_donate = jnp.maximum(a_cat - da_per_cat, 0.0)
    V_after_donate = jnp.maximum(V_cat - dV_per_cat, 0.0)
    Vsnow_after_donate = jnp.maximum(V_snow_cat - dVsnow_per_cat, 0.0)
    # Donor pond water is drained (not redistributed into the ridge).
    Vpond_new = jnp.maximum(V_pond_cat - dVpond_per_cat, 0.0)
    # Salt remains with the ice mass that stays in donor cat:
    salt_old_cat = S_ice_cat * V_cat * constants.rho_ice * _PSU_TO_FRACTION
    salt_donated_per_cat = S_ice_cat * dV_per_cat * constants.rho_ice * _PSU_TO_FRACTION
    salt_remaining = jnp.maximum(salt_old_cat - salt_donated_per_cat, 0.0)

    # Add ridge contributions to fixed cats.
    a_new = a_after_donate + da_ridge_to_cat
    V_new = V_after_donate + dV_ridge_to_cat
    Vsnow_new = Vsnow_after_donate + Vsnow_to_cat
    salt_new = salt_remaining + salt_to_cat

    # Recover h, S.
    a_safe = jnp.where(a_new > 1e-30, a_new, 1.0)
    h_new = jnp.where(a_new > 1e-30, V_new / a_safe, 0.0)
    V_safe = jnp.where(V_new > 1e-30, V_new, 1.0)
    S_new = jnp.where(
        V_new > 1e-30,
        salt_new / (V_safe * constants.rho_ice * _PSU_TO_FRACTION),
        0.0,
    )

    # Clamp to physical ranges.
    a_new = jnp.clip(a_new, 0.0, 1.0)

    return (
        a_new,
        h_new,
        Vsnow_new,
        S_new,
        Vpond_new,
        snow_to_ocean_kg_m2 / dt,
        pond_to_ocean_kg_m2 / dt,
    )


def apply_ridging(
    a_cat: jnp.ndarray,
    h_cat: jnp.ndarray,
    V_snow_cat: jnp.ndarray,
    S_ice_cat: jnp.ndarray,
    closing_rate: jnp.ndarray,
    n_cat: int,
    dt: float,
    *,
    V_pond_cat: jnp.ndarray | None = None,
    e_star: float = _RIDGE_DEFAULTS.e_star,
    mu_rdg: float = _RIDGE_DEFAULTS.mu_rdg,
    H_star: float = _RIDGE_DEFAULTS.H_star,
    snow_fraction_retained: float = _RIDGE_DEFAULTS.snow_fraction_retained,
) -> dict:
    """Apply Lipscomb 2007 mechanical ridging to a multi-category state.

    Conservation invariants (per column):
        * Total ice area is *reduced* by net convergence: ΔA = −closing_rate·dt
          (the aksum normalization scales the participating draw so the NET
          closing matches the requested rate) in the un-capped regime, UNLESS
          limited by available area — either the total ``a_total`` (all ice
          ridged) or a participating category's per-cat draw cap ``da <= a_cat``
          (the compression factor is estimated before per-cat clipping) — where
          the net closing saturates below the requested rate.
        * Total ice volume is conserved (donor volume = ridge volume).
        * Total snow volume is **not** conserved when
          ``snow_fraction_retained < 1`` — the difference is reported
          as ``snow_to_ocean`` (freshwater).
        * Bulk salt mass is conserved across the redistribution.

    Parameters
    ----------
    a_cat, h_cat : arrays ``(..., n_cat)``
        Per-category area fraction and mean thickness.
    V_snow_cat : array ``(..., n_cat)``
        Per-category snow volume ``h_snow · a`` [m].
    S_ice_cat : array ``(..., n_cat)``
        Per-category bulk ice salinity [PSU].
    closing_rate : array ``(...)``
        Net convergence rate ``max(0, −div(u_ice))`` [1/s].
    n_cat : int
    dt : float
    e_star, mu_rdg, H_star, snow_fraction_retained : float
        Closure parameters (see :class:`RidgingConfig`).

    Returns
    -------
    result : dict
        ``"a"``, ``"h"``, ``"V_snow"``, ``"S_ice"`` —
        post-ridging per-category fields.
        ``"snow_to_ocean"`` — freshwater mass flux to ocean from
        ridge-shedded snow [kg/m²/s].
    """
    lo = category_bounds(n_cat)
    hi = upper_bounds(n_cat)

    spatial_shape = a_cat.shape[:-1]
    n_cells = 1
    for s in spatial_shape:
        n_cells *= s

    def _flat(x):
        return x.reshape((n_cells, n_cat))

    closing_flat = closing_rate.reshape((n_cells,))

    if V_pond_cat is None:
        V_pond_cat = jnp.zeros_like(V_snow_cat)

    vmapped = jax.vmap(
        _ridging_column_kernel,
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None),
    )

    a_f, h_f, Vsnow_f, S_f, Vpond_f, snow_to_ocean_f, pond_to_ocean_f = vmapped(
        _flat(a_cat),
        _flat(h_cat),
        _flat(V_snow_cat),
        _flat(S_ice_cat),
        _flat(V_pond_cat),
        closing_flat,
        lo,
        hi,
        dt,
        e_star,
        mu_rdg,
        H_star,
        snow_fraction_retained,
    )

    full_shape = spatial_shape + (n_cat,)
    return {
        "a": a_f.reshape(full_shape),
        "h": h_f.reshape(full_shape),
        "V_snow": Vsnow_f.reshape(full_shape),
        "S_ice": S_f.reshape(full_shape),
        "V_pond": Vpond_f.reshape(full_shape),
        "snow_to_ocean": snow_to_ocean_f.reshape(spatial_shape),
        "pond_to_ocean": pond_to_ocean_f.reshape(spatial_shape),
    }


def _validate_si3_jpl1_ridging_config(config: SI3JPL1RidgingConfig) -> None:
    """Reject every unmeasured redistribution selector combination."""

    expected = SI3JPL1RidgingConfig()
    if config != expected:
        raise ValueError(
            "SI3 jpl=1 ridging selector composition "
            f"{config!r} != {expected!r}; no Frankenstein fallback"
        )


def apply_si3_jpl1_ridging(
    state: SI3JPL1RidgingState,
    divergence: jnp.ndarray,
    deformation: jnp.ndarray,
    dt: float,
    *,
    config: SI3JPL1RidgingConfig = SI3JPL1RidgingConfig(),
) -> tuple[SI3JPL1RidgingState, SI3JPL1RidgingLosses]:
    """Apply SI3's exact ORCA1 exponential ridge/raft arm for ``jpl=1``.

    This is the selectable SI3 sibling of :func:`apply_ridging`; that existing
    Lipscomb-2007 path is unchanged.  The transcription follows
    ``icedyn_rdgrft.F90:209-341`` (closing and iteration), ``:398-624``
    (cumulative-area participation, ridge/raft split and normalization), and
    ``:667-903`` (ordered donor/receiver ledger and roundoff cleanup).

    For one category the exponential receiver is necessarily the last
    category, so SI3's own last-bin branch makes both ridge fractions exactly
    one (``:843-852``); rafting likewise lands in that category or takes the
    conservation fallback (``:857-870``).  The layer axes are trailing and
    are carried with their parent ice/snow inventory.  ORCA1's zero ridge
    porosity makes ice volume, ice enthalpy, and option-4 layer salt invariant
    under this redistribution.  Because this rung has ``ln_icethd=F``, SI3's
    initialization also forces the snow/pond retention factors to one
    (``icedyn_rdgrft.F90:1244-1247``), so those inventories remain carried.
    """

    _validate_si3_jpl1_ridging_config(config)
    if not math.isfinite(dt) or dt <= _SI3_JPL1_ZERO:
        raise ValueError("SI3 jpl=1 ridging requires finite positive dt")
    base_shape = state.ice_area.shape
    if len(base_shape) > 2:
        raise ValueError(
            "SI3 jpl=1 ridging requires category-collapsed state; "
            "a trailing category axis is outside this measured arm"
        )
    scalar_fields = state[:8]
    if any(value.shape != base_shape for value in scalar_fields):
        raise ValueError("SI3 jpl=1 scalar state leaves must share one T-grid shape")
    if divergence.shape != base_shape or deformation.shape != base_shape:
        raise ValueError("SI3 jpl=1 deformation fields must match the T grid")
    if state.snow_enthalpy.shape[:-1] != base_shape:
        raise ValueError("SI3 jpl=1 snow enthalpy must use trailing layers")
    if state.ice_enthalpy.shape[:-1] != base_shape:
        raise ValueError("SI3 jpl=1 ice enthalpy must use trailing layers")
    if state.ice_salt_content.shape != state.ice_enthalpy.shape:
        raise ValueError("SI3 jpl=1 salt and ice enthalpy layers must agree")

    area = state.ice_area
    open_water = state.open_water_area
    ice_volume = state.ice_volume
    snow_volume = state.snow_volume
    age = state.age_content
    pond_area = state.pond_area
    pond_volume = state.pond_volume
    pond_lid_volume = state.pond_lid_volume
    snow_enthalpy = state.snow_enthalpy
    ice_enthalpy = state.ice_enthalpy
    salt_content = state.ice_salt_content

    snow_loss = jnp.zeros_like(snow_volume)
    snow_enthalpy_loss = jnp.zeros_like(snow_enthalpy)
    pond_loss = jnp.zeros_like(pond_volume)
    pond_lid_loss = jnp.zeros_like(pond_lid_volume)
    iteration_count = jnp.zeros(base_shape, dtype=jnp.int32)
    excessive_removal_clamp = jnp.zeros(base_shape, dtype=bool)
    open_water_correction = jnp.zeros(base_shape, dtype=bool)

    closing = config.cs_ridging * _SI3_JPL1_RAFT_AREA_MULTIPLIER * (
        deformation - jnp.abs(divergence)
    ) - jnp.minimum(divergence, _SI3_JPL1_ZERO)
    closing = jnp.where(
        divergence < _SI3_JPL1_ZERO,
        jnp.maximum(closing, -divergence),
        closing,
    )
    opening = closing + divergence
    work = area > _SI3_JPL1_EPSI10

    # NEMO initializes iter=1 and loops while iter<20 (:289-292): at most 19
    # ordered shifts.  A fixed loop plus the per-cell ``work`` mask is the JAX
    # equivalent of its shrinking packed-cell list and remains JIT/grad safe.
    for _ in range(_SI3_JPL1_MAX_ITERATIONS):
        total_area = open_water + area
        safe_total_area = jnp.where(total_area > _SI3_JPL1_EPSI10, total_area, _SI3_JPL1_ONE)
        inverse_total = jnp.where(
            total_area > _SI3_JPL1_EPSI10,
            _SI3_JPL1_ONE / safe_total_area,
            _SI3_JPL1_ZERO,
        )
        g_open = open_water * inverse_total
        inverse_astar = _SI3_JPL1_ONE / config.astar
        exponential_scale = _SI3_JPL1_ONE / (_SI3_JPL1_ONE - jnp.exp(-inverse_astar))
        g_minus_one = exponential_scale
        g_zero = jnp.exp(-g_open * inverse_astar) * exponential_scale
        g_one = jnp.exp(-inverse_astar) * exponential_scale
        participation_open = g_minus_one - g_zero
        participation_ice = g_zero - g_one

        has_area = area > _SI3_JPL1_EPSI10
        safe_area = jnp.where(has_area, area, _SI3_JPL1_ONE)
        thickness = jnp.where(has_area, ice_volume / safe_area, _SI3_JPL1_ZERO)
        ridge_fraction = (
            (_SI3_JPL1_ONE + jnp.tanh(config.craft * (thickness - config.hraft_m)))
            * _SI3_JPL1_RAFT_AREA_MULTIPLIER
            * participation_ice
        )
        raft_fraction = participation_ice - ridge_fraction
        mean_ridge_thickness = jnp.maximum(
            _si3_jpl1_sqrt(config.hstar_m * thickness),
            thickness * _SI3_JPL1_RIDGE_MIN_MULTIPLIER,
        )
        ridge_minimum = jnp.minimum(
            _SI3_JPL1_TWO * thickness,
            _SI3_JPL1_RAFT_AREA_MULTIPLIER * (mean_ridge_thickness + thickness),
        )
        ridge_exponential = config.mu_ridging * _si3_jpl1_sqrt(thickness)
        ridge_area_ratio = thickness / jnp.maximum(
            _SI3_JPL1_EPSI20, ridge_minimum + ridge_exponential
        )
        normalization = (
            participation_open
            + ridge_fraction * (_SI3_JPL1_ONE - ridge_area_ratio)
            + raft_fraction * (_SI3_JPL1_ONE - _SI3_JPL1_RAFT_AREA_MULTIPLIER)
        )
        safe_normalization = jnp.where(
            normalization > _SI3_JPL1_EPSI10,
            normalization,
            _SI3_JPL1_ONE,
        )
        gross = jnp.where(
            normalization > _SI3_JPL1_EPSI10,
            closing / safe_normalization,
            _SI3_JPL1_ZERO,
        )
        requested_area = participation_ice * gross * dt
        safe_participation = jnp.where(
            participation_ice != _SI3_JPL1_ZERO,
            participation_ice,
            _SI3_JPL1_ONE,
        )
        excessive = work & (requested_area > area) & (participation_ice != _SI3_JPL1_ZERO)
        excessive_removal_clamp = excessive_removal_clamp | excessive
        gross = jnp.where(
            excessive,
            area / safe_participation / dt,
            gross,
        )
        corrected_open = open_water + (opening - participation_open * gross) * dt
        open_water_correction = open_water_correction | (
            work & ((corrected_open < _SI3_JPL1_ZERO) | (corrected_open > total_area))
        )
        opening = jnp.where(
            corrected_open < _SI3_JPL1_ZERO,
            participation_open * gross - open_water / dt,
            opening,
        )
        opening = jnp.where(
            corrected_open > total_area,
            participation_open * gross + (total_area - open_water) / dt,
            opening,
        )
        shift = work & (participation_ice > _SI3_JPL1_ZERO) & (gross > _SI3_JPL1_ZERO)

        new_open_water = jnp.maximum(
            _SI3_JPL1_ZERO,
            open_water + (opening - participation_open * gross) * dt,
        )
        # SI3 writes every donor/receiver ledger operation separately at
        # icedyn_rdgrft.F90:697-705,715-720,734-741,779-792,874-890.  Preserve
        # those fp64 statement boundaries under compiled execution: a plain
        # optimization_barrier is not an arithmetic association guard in HLO.
        sr = nemo_source_round
        ridge_area_removed = sr(sr(ridge_fraction * gross) * dt)
        raft_area_removed = sr(sr(raft_fraction * gross) * dt)
        inverse_area = jnp.where(
            area > _SI3_JPL1_EPSI10,
            _SI3_JPL1_ONE / safe_area,
            _SI3_JPL1_ZERO,
        )
        ridged_fraction = sr(ridge_area_removed * inverse_area)
        rafted_fraction = sr(raft_area_removed * inverse_area)
        retained_fraction = sr(sr(_SI3_JPL1_ONE - ridged_fraction) - rafted_fraction)
        new_ridge_area = sr(ridge_area_removed * ridge_area_ratio)
        new_raft_area = sr(raft_area_removed * _SI3_JPL1_RAFT_AREA_MULTIPLIER)

        area_after_donation = sr(sr(area - ridge_area_removed) - raft_area_removed)
        new_area = sr(area_after_donation + sr(new_ridge_area + new_raft_area))
        # rn_porordg=0 on this arm: SI3's donor volume/ice-energy/salt removed
        # at :779-793 returns in full through the last-category receiver at
        # :843-852,874-890.  Keep the written expressions (rather than simply
        # copying) so a future selector cannot silently reuse this arm.
        ridge_ice_volume = sr(ice_volume * ridged_fraction)
        raft_ice_volume = sr(ice_volume * rafted_fraction)
        ice_volume_after_donation = sr(ice_volume * retained_fraction)
        new_ice_volume = sr(
            ice_volume_after_donation + sr(ridge_ice_volume + raft_ice_volume)
        )
        ridge_age = sr(sr(age * ridged_fraction) * ridge_area_ratio)
        raft_age = sr(sr(age * rafted_fraction) * _SI3_JPL1_RAFT_AREA_MULTIPLIER)
        age_after_donation = sr(age * retained_fraction)
        new_age = sr(age_after_donation + sr(ridge_age + raft_age))
        ridge_snow_volume = sr(snow_volume * ridged_fraction)
        raft_snow_volume = sr(snow_volume * rafted_fraction)
        snow_volume_after_donation = sr(snow_volume * retained_fraction)
        new_snow_volume = sr(
            snow_volume_after_donation
            + sr(
                sr(ridge_snow_volume * config.snow_ridge_retention)
                + sr(raft_snow_volume * config.snow_raft_retention)
            )
        )
        ridge_pond_area = sr(sr(pond_area * ridged_fraction) * ridge_area_ratio)
        raft_pond_area = sr(
            sr(pond_area * rafted_fraction) * _SI3_JPL1_RAFT_AREA_MULTIPLIER
        )
        pond_area_after_donation = sr(pond_area * retained_fraction)
        new_pond_area = sr(
            pond_area_after_donation
            + sr(
                sr(ridge_pond_area * config.pond_ridge_retention)
                + sr(raft_pond_area * config.pond_raft_retention)
            )
        )
        ridge_pond_volume = sr(pond_volume * ridged_fraction)
        raft_pond_volume = sr(pond_volume * rafted_fraction)
        pond_volume_after_donation = sr(pond_volume * retained_fraction)
        new_pond_volume = sr(
            pond_volume_after_donation
            + sr(
                sr(ridge_pond_volume * config.pond_ridge_retention)
                + sr(raft_pond_volume * config.pond_raft_retention)
            )
        )
        ridge_lid_volume = sr(pond_lid_volume * ridged_fraction)
        raft_lid_volume = sr(pond_lid_volume * rafted_fraction)
        lid_volume_after_donation = sr(pond_lid_volume * retained_fraction)
        new_pond_lid_volume = sr(
            lid_volume_after_donation
            + sr(
                sr(ridge_lid_volume * config.pond_ridge_retention)
                + sr(raft_lid_volume * config.pond_raft_retention)
            )
        )
        layer_fraction = retained_fraction[..., None]
        ridge_layer_fraction = ridged_fraction[..., None]
        raft_layer_fraction = rafted_fraction[..., None]
        ridge_snow_enthalpy = sr(snow_enthalpy * ridge_layer_fraction)
        raft_snow_enthalpy = sr(snow_enthalpy * raft_layer_fraction)
        snow_enthalpy_after_donation = sr(snow_enthalpy * layer_fraction)
        new_snow_enthalpy = sr(
            snow_enthalpy_after_donation
            + sr(
                sr(ridge_snow_enthalpy * config.snow_ridge_retention)
                + sr(raft_snow_enthalpy * config.snow_raft_retention)
            )
        )
        ridge_ice_enthalpy = sr(ice_enthalpy * ridge_layer_fraction)
        raft_ice_enthalpy = sr(ice_enthalpy * raft_layer_fraction)
        ice_enthalpy_after_donation = sr(ice_enthalpy * layer_fraction)
        new_ice_enthalpy = sr(
            ice_enthalpy_after_donation + sr(ridge_ice_enthalpy + raft_ice_enthalpy)
        )
        ridge_salt = sr(salt_content * ridge_layer_fraction)
        raft_salt = sr(salt_content * raft_layer_fraction)
        salt_after_donation = sr(salt_content * layer_fraction)
        new_salt_content = sr(salt_after_donation + sr(ridge_salt + raft_salt))

        lost_snow_fraction = ridged_fraction * (
            _SI3_JPL1_ONE - config.snow_ridge_retention
        ) + rafted_fraction * (_SI3_JPL1_ONE - config.snow_raft_retention)
        lost_pond_fraction = ridged_fraction * (
            _SI3_JPL1_ONE - config.pond_ridge_retention
        ) + rafted_fraction * (_SI3_JPL1_ONE - config.pond_raft_retention)
        snow_loss = snow_loss + jnp.where(shift, snow_volume * lost_snow_fraction, _SI3_JPL1_ZERO)
        snow_enthalpy_loss = snow_enthalpy_loss + jnp.where(
            shift[..., None],
            snow_enthalpy * lost_snow_fraction[..., None],
            _SI3_JPL1_ZERO,
        )
        pond_loss = pond_loss + jnp.where(shift, pond_volume * lost_pond_fraction, _SI3_JPL1_ZERO)
        pond_lid_loss = pond_lid_loss + jnp.where(
            shift, pond_lid_volume * lost_pond_fraction, _SI3_JPL1_ZERO
        )
        iteration_count = iteration_count + shift.astype(jnp.int32)

        area = jnp.where(shift, new_area, area)
        open_water = jnp.where(shift, new_open_water, open_water)
        ice_volume = jnp.where(shift, new_ice_volume, ice_volume)
        snow_volume = jnp.where(shift, new_snow_volume, snow_volume)
        age = jnp.where(shift, new_age, age)
        pond_area = jnp.where(shift, new_pond_area, pond_area)
        pond_volume = jnp.where(shift, new_pond_volume, pond_volume)
        pond_lid_volume = jnp.where(shift, new_pond_lid_volume, pond_lid_volume)
        snow_enthalpy = jnp.where(shift[..., None], new_snow_enthalpy, snow_enthalpy)
        ice_enthalpy = jnp.where(shift[..., None], new_ice_enthalpy, ice_enthalpy)
        salt_content = jnp.where(shift[..., None], new_salt_content, salt_content)

        residual = _SI3_JPL1_ONE - (open_water + area)
        converged = jnp.abs(residual) < _SI3_JPL1_EPSI10
        open_water = jnp.where(
            shift & converged,
            jnp.maximum(_SI3_JPL1_ZERO, _SI3_JPL1_ONE - area),
            open_water,
        )
        work = shift & ~converged
        divergence = jnp.where(work, residual / dt, divergence)
        closing = jnp.where(work, jnp.maximum(_SI3_JPL1_ZERO, -divergence), closing)
        opening = jnp.where(work, jnp.maximum(_SI3_JPL1_ZERO, divergence), opening)

    # icedyn_rdgrft.F90:900-903 calls ice_var_roundoff.  Open water is absent
    # from that call, and icevar.F90:871 clips lid volume only in the narrow
    # -epsi10 < v_il < 0 roundoff interval.  Preserve those exact exclusions.
    lid_roundoff = (pond_lid_volume < _SI3_JPL1_ZERO) & (pond_lid_volume > -_SI3_JPL1_EPSI10)
    result = SI3JPL1RidgingState(
        ice_area=jnp.maximum(area, _SI3_JPL1_ZERO),
        open_water_area=open_water,
        ice_volume=jnp.maximum(ice_volume, _SI3_JPL1_ZERO),
        snow_volume=jnp.maximum(snow_volume, _SI3_JPL1_ZERO),
        age_content=jnp.maximum(age, _SI3_JPL1_ZERO),
        pond_area=jnp.maximum(pond_area, _SI3_JPL1_ZERO),
        pond_volume=jnp.maximum(pond_volume, _SI3_JPL1_ZERO),
        pond_lid_volume=jnp.where(lid_roundoff, _SI3_JPL1_ZERO, pond_lid_volume),
        snow_enthalpy=jnp.maximum(snow_enthalpy, _SI3_JPL1_ZERO),
        ice_enthalpy=jnp.maximum(ice_enthalpy, _SI3_JPL1_ZERO),
        ice_salt_content=jnp.maximum(salt_content, _SI3_JPL1_ZERO),
    )
    losses = SI3JPL1RidgingLosses(
        snow_volume=snow_loss,
        snow_enthalpy=snow_enthalpy_loss,
        pond_volume=pond_loss,
        pond_lid_volume=pond_lid_loss,
        iterations=iteration_count,
        excessive_removal_clamp=excessive_removal_clamp,
        open_water_correction=open_water_correction,
    )
    return result, losses


# ==============================================================================
# SI3 ORCA2 (rung O6): five-category ridging/rafting, NEMO-literal
# Cited build: ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo/icedyn_rdgrft.f90 (byte-identical
# to the O1/O2/O5_REF/O7_REF builds).  Arrays use NEMO's index order:
# (jpi, jpj, jpl) and (jpi, jpj, nlay, jpl).  Every operation is rounded in source
# order (DECISION 158); exp/tanh go through the precision policy (DECISION 162).
# ==============================================================================

_RDG_EPSI10 = 1.0e-10  # ice.F90 epsi10
_RDG_EPSI20 = 1.0e-20  # ice.F90 epsi20
_RDG_HRDG_HI_MIN = 1.1  # icedyn_rdgrft.f90:87
_RDG_HI_HRFT = 0.5  # icedyn_rdgrft.f90:88
_RDG_ITERMAX = 20  # icedyn_rdgrft.f90:228 (DO WHILE iter < 20: at most 19 passes)
_RDG_POROSITY_FULL_M = 10.0  # icedyn_rdgrft.f90:749
_RDG_POROSITY_ZERO_M = 20.0  # icedyn_rdgrft.f90:750
_RDG_POROSITY_SLOPE = 0.1  # icedyn_rdgrft.f90:751


class SI3RdgrftConfig(NamedTuple):
    """Resolved ``namdyn_rdgrft`` + the selectors ``ice_dyn_rdgrft`` reads; no defaults.

    Build it from the run's resolved namelists, then pass it through
    :func:`si3_rdgrft_config_after_init` with the run's ``ln_icethd``.
    """

    rn_csrdg: float
    ln_partf_exp: bool
    rn_astar: float
    ln_distf_lin: bool
    ln_distf_exp: bool
    rn_murdg: float
    ln_ridging: bool
    rn_hstar: float
    rn_porordg: float
    rn_fsnwrdg: float
    rn_fpndrdg: float
    ln_rafting: bool
    rn_hraft: float
    rn_craft: float
    rn_fsnwrft: float
    rn_fpndrft: float
    ln_rhg_evp_or_vp: bool
    ln_pnd: bool  # ln_pnd_LEV .OR. ln_pnd_TOPO
    ln_pnd_lids: bool
    nn_icesal: int
    rdt_ice: float
    rhoi: float
    rhos: float
    rhow: float
    rcp: float


class SI3RdgrftState(NamedTuple):
    """NEMO's arrays at ``ice_dyn_rdgrft`` entry that it modifies (also outputs)."""

    a_i: jnp.ndarray
    v_i: jnp.ndarray
    v_s: jnp.ndarray
    oa_i: jnp.ndarray
    a_ip: jnp.ndarray
    v_ip: jnp.ndarray
    v_il: jnp.ndarray
    e_s: jnp.ndarray
    e_i: jnp.ndarray
    szv_i: jnp.ndarray
    at_i: jnp.ndarray
    ato_i: jnp.ndarray
    wfx_dyn: jnp.ndarray
    sfx_dyn: jnp.ndarray
    hfx_dyn: jnp.ndarray
    wfx_snw_dyn: jnp.ndarray
    wfx_pnd: jnp.ndarray


class SI3RdgrftResult(NamedTuple):
    state: SI3RdgrftState
    selected: jnp.ndarray  # cells kept by the :311-323 selection
    n_passes: jnp.ndarray  # ridging passes executed (<= 19)


def si3_rdgrft_config_after_init(config: SI3RdgrftConfig, *, ln_icethd: bool) -> SI3RdgrftConfig:
    """``ice_dyn_rdgrft_init``'s override, ``icedyn_rdgrft.f90:1273-1276``."""
    if ln_icethd:
        return config
    return config._replace(rn_porordg=0.0, rn_fsnwrdg=1.0, rn_fsnwrft=1.0,
                           rn_fpndrdg=1.0, rn_fpndrft=1.0)


def _validate_rdgrft_config(cfg: SI3RdgrftConfig) -> None:
    if not isinstance(cfg, SI3RdgrftConfig):
        raise TypeError("SI3 ridging requires an explicit SI3RdgrftConfig")
    if not cfg.ln_partf_exp:
        raise ValueError("SI3 O6 transcribes ln_partf_exp only (ln_partf_lin is not executed)")
    if cfg.ln_distf_lin == cfg.ln_distf_exp:
        raise ValueError("choose one and only one redistribution function (icedyn_rdgrft.f90:1262-1265)")
    if not cfg.ln_rhg_evp_or_vp:
        raise ValueError("SI3 O6 transcribes the EVP/VP closing rate only (EAP rdg_conv absent)")
    if cfg.nn_icesal != 4:
        raise ValueError("SI3 O6 transcribes nn_icesal == 4 only")
    if not (math.isfinite(cfg.rdt_ice) and cfg.rdt_ice > 0.0):
        raise ValueError("SI3 ridging requires finite positive rdt_ice")


def _rdgrft_prep(a, v, ato, closing_net, opning, cfg: SI3RdgrftConfig):
    """``rdgrft_prep``, ``icedyn_rdgrft.f90:444-665`` (per cell; cells are the 2-D grid)."""
    from legoesm.core import transcendentals as tr
    from legoesm.ice.itd import si3_safe_div
    from legoesm.ice.si3_surface import fortran_max_tie_first as fmax
    from legoesm.ice.si3_surface import fortran_min_tie_first as fmin
    from legoesm.ice.si3_surface import fortran_sum0

    sr = nemo_source_round
    jpl = a.shape[-1]
    rdt = cfg.rdt_ice
    r1 = 1.0 / rdt
    z1_astar = 1.0 / cfg.rn_astar  # :445
    ok = a > _RDG_EPSI10
    zhi = jnp.where(ok, sr(si3_safe_div(v, a, ok)), 0.0)  # :451-453
    zasum = sr(ato + fortran_sum0(a))  # :468
    oks = zasum > _RDG_EPSI10
    z1_asum = jnp.where(oks, sr(si3_safe_div(1.0, zasum, oks)), 0.0)  # :470-472
    zg = [jnp.zeros_like(ato), sr(ato * z1_asum)]  # :478-479
    for jl in range(1, jpl + 1):  # :480-482
        zg.append(sr(sr(ato + fortran_sum0(a[..., :jl])) * z1_asum))
    zfac = sr(1.0 / sr(1.0 - tr.exp(jnp.asarray(-z1_astar, jnp.float64))))  # :501
    ge = [sr(sr(tr.exp(-sr(g * z1_astar))) * zfac) for g in zg]  # :502-506
    apartf = jnp.stack([sr(ge[k] - ge[k + 1]) for k in range(jpl + 1)], axis=-1)  # :507-511
    ap = apartf[..., 1:]
    if cfg.ln_rafting and cfg.ln_ridging:  # :516-522
        tanh_term = sr(1.0 + tr.tanh(sr(cfg.rn_craft * sr(zhi - cfg.rn_hraft))))
        aridge = sr(sr(tanh_term * 0.5) * ap)
        araft = sr(ap - aridge)
    elif cfg.ln_ridging:  # :523-529
        aridge, araft = ap, jnp.zeros_like(ap)
    elif cfg.ln_rafting:  # :530-536
        aridge, araft = jnp.zeros_like(ap), ap
    else:  # :537-543
        aridge, araft = jnp.zeros_like(ap), jnp.zeros_like(ap)

    pos = ap > 0.0  # :603
    sq = jnp.sqrt(sr(cfg.rn_hstar * zhi))
    zhmean = fmax(sq, sr(zhi * _RDG_HRDG_HI_MIN))  # :604, MAXSD tie order PLAUSIBLE
    hrmin = fmin(sr(2.0 * zhi), sr(0.5 * sr(zhmean + zhi)))  # :605
    hraft = sr(zhi * (1.0 / _RDG_HI_HRFT))  # :598,606
    if cfg.ln_distf_lin:  # :608-610
        hrmax = sr(sr(2.0 * zhmean) - hrmin)
        hrexp = jnp.zeros_like(zhi)
        phi = sr(zhi / jnp.maximum(zhmean, _RDG_EPSI20))
    else:  # :611-613
        hrmax = jnp.zeros_like(zhi)
        hrexp = sr(cfg.rn_murdg * jnp.sqrt(zhi))
        phi = sr(zhi / jnp.maximum(_RDG_EPSI20, sr(hrmin + hrexp)))
    hrmin = jnp.where(pos, hrmin, 0.0)  # :619-625
    hrmax = jnp.where(pos, hrmax, 0.0)
    hrexp = jnp.where(pos, hrexp, 0.0)
    hraft = jnp.where(pos, hraft, 0.0)
    phi = jnp.where(pos, phi, 1.0)
    zaksum = apartf[..., 0]  # :599
    for jl in range(jpl):  # :617-618, (zaksum + ridge term) + raft term
        term = sr(sr(zaksum + sr(aridge[..., jl] * sr(1.0 - phi[..., jl])))
                  + sr(araft[..., jl] * (1.0 - _RDG_HI_HRFT)))
        zaksum = jnp.where(pos[..., jl], term, zaksum)

    okk = zaksum > _RDG_EPSI10  # :635-637
    cg = jnp.where(okk, sr(si3_safe_div(closing_net, zaksum, okk)), 0.0)
    for jl in range(jpl):  # :643-650, sequential in jl
        p = ap[..., jl]
        excess = (sr(sr(p * cg) * rdt) > a[..., jl]) & (p != 0.0)
        cg = jnp.where(excess, sr(sr(si3_safe_div(a[..., jl], p, p != 0.0)) * r1), cg)
    p0cg = sr(apartf[..., 0] * cg)  # :656-663
    zf = sr(ato + sr(sr(opning - p0cg) * rdt))
    opning = jnp.where(zf < 0.0, sr(p0cg - sr(ato * r1)),
                       jnp.where(zf > zasum, sr(p0cg + sr(sr(zasum - ato) * r1)), opning))
    return apartf, aridge, araft, phi, hraft, hrmin, hrmax, hrexp, cg, opning


def _rdgrft_shift(s: dict, cells, prep, sst, sss, hi_max, cfg: SI3RdgrftConfig) -> dict:
    """``rdgrft_shift``, ``icedyn_rdgrft.f90:708-944`` on the selected ``cells``."""
    from legoesm.core import transcendentals as tr
    from legoesm.ice.itd import si3_safe_div, si3_var_roundoff
    from legoesm.ice.si3_surface import fortran_max_tie_first as fmax
    from legoesm.ice.si3_surface import fortran_sum0

    sr = nemo_source_round
    apartf, aridge, araft, phi, hraft, hrmin, hrmax, hrexp, cg, opning = prep
    s = dict(s)
    jpl = s["a_i"].shape[-1]
    rdt = cfg.rdt_ice
    r1 = 1.0 / rdt
    r1_nlay = 1.0 / s["e_i"].shape[-2]
    zvti = fortran_sum0(s["v_i"])  # :708
    new_ato = fmax(0.0, sr(s["ato_i"] + sr(sr(opning - sr(apartf[..., 0] * cg)) * rdt)))  # :713
    s["ato_i"] = jnp.where(cells, new_ato, s["ato_i"])

    def col(name, jl):
        return s[name][..., jl]

    def put(name, jl, value, mask):
        x = s[name]
        m = mask[..., None] if x.ndim == 4 else mask
        s[name] = x.at[..., jl].set(jnp.where(m, value, x[..., jl]))

    for jl1 in range(jpl):  # :722
        ll = cells & (apartf[..., jl1 + 1] > 0.0) & (cg > 0.0)  # :727-729
        a1 = col("a_i", jl1)
        oka = a1 > _RDG_EPSI10
        z1_ai = jnp.where(oka, sr(si3_safe_div(1.0, a1, oka)), 0.0)  # :733-735
        airdg1 = sr(sr(aridge[..., jl1] * cg) * rdt)  # :738
        airft1 = sr(sr(araft[..., jl1] * cg) * rdt)  # :739
        airdg2 = sr(airdg1 * phi[..., jl1])  # :741
        airft2 = sr(airft1 * _RDG_HI_HRFT)  # :742
        afrdg = sr(airdg1 * z1_ai)  # :745
        afrft = sr(airft1 * z1_ai)  # :746
        v1, vs1, oa1 = col("v_i", jl1), col("v_s", jl1), col("oa_i", jl1)
        vsw_full = sr(sr(v1 * afrdg) * cfg.rn_porordg)  # :749
        ramp = fmax(0.0, sr(2.0 - sr(_RDG_POROSITY_SLOPE * zvti)))
        vsw = jnp.where(zvti <= _RDG_POROSITY_FULL_M, vsw_full,
                        jnp.where(zvti >= _RDG_POROSITY_ZERO_M, 0.0, sr(vsw_full * ramp)))  # :749-752
        ersw = -sr(sr(sr(cfg.rhoi * vsw) * cfg.rcp) * sst)  # :753
        virdg = sr(sr(v1 * afrdg) + vsw)  # :756
        vsrdg = sr(vs1 * afrdg)  # :757
        oirdg = sr(sr(oa1 * afrdg) * phi[..., jl1])  # :758
        virft = sr(v1 * afrft)  # :760
        vsrft = sr(vs1 * afrft)  # :761
        oirft = sr(sr(oa1 * afrft) * _RDG_HI_HRFT)  # :762
        aip1, vip1, vil1 = col("a_ip", jl1), col("v_ip", jl1), col("v_il", jl1)
        aprdg = sr(sr(aip1 * afrdg) * phi[..., jl1])  # :765
        vprdg = sr(vip1 * afrdg)  # :766
        aprft = sr(sr(aip1 * afrft) * _RDG_HI_HRFT)  # :767
        vprft = sr(vip1 * afrft)  # :768
        vlrdg = sr(vil1 * afrdg)  # :770
        vlrft = sr(vil1 * afrft)  # :771
        es1, ei1, sz1 = col("e_s", jl1), col("e_i", jl1), col("szv_i", jl1)
        esrdg = sr(es1 * afrdg[..., None])  # :775
        esrft = sr(es1 * afrft[..., None])  # :776
        eirdg = sr(sr(ei1 * afrdg[..., None]) + sr(ersw * r1_nlay)[..., None])  # :777
        eirft = sr(ei1 * afrft[..., None])  # :778
        sirdg = sr(sr(sz1 * afrdg[..., None]) + sr(sr(vsw * sss) * r1_nlay)[..., None])  # :781
        sirft = sr(sz1 * afrft[..., None])  # :782

        def acc(name, value):
            s[name] = jnp.where(ll, value, s[name])

        acc("wfx_dyn", sr(s["wfx_dyn"] - sr(sr(vsw * cfg.rhoi) * r1)))  # :789
        acc("sfx_dyn", sr(s["sfx_dyn"] - sr(sr(sr(vsw * sss) * cfg.rhoi) * r1)))  # :790
        acc("hfx_dyn", sr(s["hfx_dyn"] + sr(ersw * r1)))  # :791
        snw_r, snw_f = 1.0 - cfg.rn_fsnwrdg, 1.0 - cfg.rn_fsnwrft
        acc("wfx_snw_dyn", sr(s["wfx_snw_dyn"] + sr(sr(sr(sr(cfg.rhos * vsrdg) * snw_r)
                                                       + sr(sr(cfg.rhos * vsrft) * snw_f)) * r1)))  # :795-796
        for jk in range(esrdg.shape[-1]):  # :797-800
            acc("hfx_dyn", sr(s["hfx_dyn"] + sr(sr(-sr(esrdg[..., jk] * snw_r)
                                                   - sr(esrft[..., jk] * snw_f)) * r1)))
        pnd_r, pnd_f = 1.0 - cfg.rn_fpndrdg, 1.0 - cfg.rn_fpndrft
        if cfg.ln_pnd:  # :802-809
            acc("wfx_pnd", sr(s["wfx_pnd"] + sr(sr(sr(sr(cfg.rhow * vprdg) * pnd_r)
                                                 + sr(sr(cfg.rhow * vprft) * pnd_f)) * r1)))
            if cfg.ln_pnd_lids:
                acc("wfx_pnd", sr(s["wfx_pnd"] + sr(sr(sr(sr(cfg.rhow * vlrdg) * pnd_r)
                                                     + sr(sr(cfg.rhow * vlrft) * pnd_f)) * r1)))

        keep = sr(sr(1.0 - afrdg) - afrft)  # :821
        put("a_i", jl1, sr(sr(a1 - airdg1) - airft1), ll)  # :820
        put("v_i", jl1, sr(v1 * keep), ll)  # :821
        put("v_s", jl1, sr(vs1 * keep), ll)  # :822
        put("oa_i", jl1, sr(oa1 * keep), ll)  # :823
        if cfg.ln_pnd:  # :824-828
            put("a_ip", jl1, sr(aip1 * keep), ll)
            put("v_ip", jl1, sr(vip1 * keep), ll)
            if cfg.ln_pnd_lids:
                put("v_il", jl1, sr(vil1 * keep), ll)
        put("e_s", jl1, sr(es1 * keep[..., None]), ll)  # :830
        put("e_i", jl1, sr(ei1 * keep[..., None]), ll)  # :831
        put("szv_i", jl1, sr(sz1 * keep[..., None]), ll)  # :833

        itest_rdg = jnp.zeros_like(ll)  # :844-845
        itest_rft = jnp.zeros_like(ll)
        hmin1, hmax1, hexp1, hrft1 = hrmin[..., jl1], hrmax[..., jl1], hrexp[..., jl1], hraft[..., jl1]
        for jl2 in range(jpl):  # :846, Fortran jl2 = jl2 + 1
            lo, hi = hi_max[jl2], hi_max[jl2 + 1]
            if cfg.ln_distf_lin:  # :853-865
                ov = (hmin1 <= hi) & (hmax1 > lo)
                hL = jnp.maximum(hmin1, lo)
                hR = jnp.minimum(hmax1, hi)
                farea = jnp.where(ov, sr(si3_safe_div(sr(hR - hL), sr(hmax1 - hmin1), ov)), 0.0)
                fvol = jnp.where(ov, sr(si3_safe_div(sr(sr(hR * hR) - sr(hL * hL)),
                                                      sr(sr(hmax1 * hmax1) - sr(hmin1 * hmin1)), ov)), 0.0)
                itest_rdg = itest_rdg | ov
            else:  # :867-893
                den = jnp.maximum(_RDG_EPSI20, hexp1)
                den2 = jnp.maximum(_RDG_EPSI20, sr(hmin1 + hexp1))
                hL = jnp.maximum(hmin1, lo)
                expL = sr(tr.exp(sr(-sr(hL - hmin1) / den)))
                if jl2 < jpl - 1:
                    ov = hmin1 <= hi
                    expR = sr(tr.exp(sr(-sr(hi - hmin1) / den)))
                    farea = jnp.where(ov, sr(expL - expR), 0.0)
                    fvol = jnp.where(ov, sr(sr(sr(sr(hL + hexp1) * expL) - sr(sr(hi + hexp1) * expR)) / den2), 0.0)
                else:
                    farea = expL
                    fvol = sr(sr(sr(hL + hexp1) * expL) / den2)
                itest_rdg = jnp.ones_like(ll)
            hit = (hrft1 <= hi) & (hrft1 > lo)  # :898-904
            zsw = jnp.where(hit, 1.0, 0.0)
            itest_rft = itest_rft | hit
            if jl2 == jpl - 1:  # :910-911
                farea = jnp.where(itest_rdg, farea, 1.0)
                fvol = jnp.where(itest_rdg, fvol, 1.0)
                zsw = jnp.where(itest_rft, zsw, 1.0)

            def add(name, x_rdg, f, x_rft):
                cur = col(name, jl2)
                fb = f[..., None] if cur.ndim == 3 else f
                zb = zsw[..., None] if cur.ndim == 3 else zsw
                put(name, jl2, sr(cur + sr(sr(x_rdg * fb) + sr(x_rft * zb))), ll)

            add("a_i", airdg2, farea, airft2)  # :915
            add("oa_i", oirdg, farea, oirft)  # :916
            add("v_i", virdg, fvol, virft)  # :917
            add("v_s", sr(vsrdg * cfg.rn_fsnwrdg), fvol, sr(vsrft * cfg.rn_fsnwrft))  # :918
            if cfg.ln_pnd:  # :919-925
                add("v_ip", sr(vprdg * cfg.rn_fpndrdg), fvol, sr(vprft * cfg.rn_fpndrft))
                add("a_ip", sr(aprdg * cfg.rn_fpndrdg), farea, sr(aprft * cfg.rn_fpndrft))
                if cfg.ln_pnd_lids:
                    add("v_il", sr(vlrdg * cfg.rn_fpndrdg), fvol, sr(vlrft * cfg.rn_fpndrft))
            add("e_s", sr(esrdg * cfg.rn_fsnwrdg), fvol, sr(esrft * cfg.rn_fsnwrft))  # :926
            add("e_i", eirdg, fvol, eirft)  # :927
            add("szv_i", sirdg, fvol, sirft)  # :929

    return si3_var_roundoff(s, cells, ln_pnd=cfg.ln_pnd, ln_pnd_lids=cfg.ln_pnd_lids,
                            nn_icesal=cfg.nn_icesal)  # :944


def si3_ice_dyn_rdgrft(
    state: SI3RdgrftState,
    *,
    divu_i,
    delta_i,
    sst_m,
    sss_m,
    hi_max,
    nn_hls: int,
    config: SI3RdgrftConfig,
) -> SI3RdgrftResult:
    """``ice_dyn_rdgrft``, ``icedyn_rdgrft.f90:268-392``, NEMO-literal.

    Inputs are NEMO's arrays at the call (``icedyn.f90:193``): the category
    state, ``divu_i``/``delta_i`` from the rheology, ``sst_m``/``sss_m`` and the
    category bounds ``hi_max(0:jpl)``.  Every one is required.  The packed
    cell list is NEMO's: interior ``A2D(0)`` cells with ``at_i > epsi10``, then
    those with ``SUM(apartf(1:jpl)) > 0`` and ``closing_gross > 0``
    (``:311-323``).  ``iterate_ridging`` is one flag over that list (one MPI
    rank's subdomain), so converged cells keep being re-visited, as in NEMO.
    The ridging diagnostics (``ll_diag_rdg``) feed only ``iom_put`` and are
    not transcribed.
    """
    from legoesm.ice.itd import si3_interior_mask
    from legoesm.ice.si3_surface import fortran_max_tie_first as fmax
    from legoesm.ice.si3_surface import fortran_min_tie_first as fmin
    from legoesm.ice.si3_surface import fortran_sum0

    _validate_rdgrft_config(config)
    for name, value in (("divu_i", divu_i), ("delta_i", delta_i), ("sst_m", sst_m),
                        ("sss_m", sss_m), ("hi_max", hi_max)):
        if value is None:
            raise ValueError(f"SI3 ridging requires the NEMO array {name!r}; no default")
    if any(v is None for v in state):
        raise ValueError("SI3 ridging requires every SI3RdgrftState array; no default")
    sr = nemo_source_round
    s = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in state._asdict().items()}
    zdivu = jnp.asarray(divu_i, jnp.float64)
    zdelt = jnp.asarray(delta_i, jnp.float64)
    sst = jnp.asarray(sst_m, jnp.float64)
    sss = jnp.asarray(sss_m, jnp.float64)
    hi_max = jnp.asarray(hi_max, jnp.float64)
    if hi_max.shape != (s["a_i"].shape[-1] + 1,):
        raise ValueError("hi_max must be hi_max(0:jpl)")
    interior = si3_interior_mask(s["at_i"].shape, nn_hls)
    r1 = 1.0 / config.rdt_ice

    s["at_i"] = jnp.where(interior, fortran_sum0(s["a_i"]), s["at_i"])  # :268
    c0 = interior & (s["at_i"] > _RDG_EPSI10)  # :271-276
    closing_net = sr(sr(sr(config.rn_csrdg * 0.5) * sr(zdelt - jnp.abs(zdivu)))
                     - fmin(0.0, zdivu))  # :295, MINSD keeps the 0 on a tie
    closing_net = jnp.where(zdivu < 0.0, fmax(closing_net, -zdivu), closing_net)  # :298
    opning = sr(closing_net + zdivu)  # :301
    prep = _rdgrft_prep(s["a_i"], s["v_i"], s["ato_i"], closing_net, opning, config)  # :307-309
    opning = jnp.where(c0, prep[-1], opning)
    selected = c0 & (fortran_sum0(prep[0][..., 1:]) > 0.0) & (prep[-2] > 0.0)  # :311-323

    def one_pass(_, carry):
        st, cn, op, dv, active, n = carry
        pr = _rdgrft_prep(st["a_i"], st["v_i"], st["ato_i"], cn, op, config)  # :346-348
        op_p = jnp.where(selected, pr[-1], op)
        new = _rdgrft_shift(st, selected, pr[:-1] + (op_p,), sst, sss, hi_max, config)  # :351-352
        zfac = sr(1.0 - sr(new["ato_i"] + fortran_sum0(new["a_i"])))  # :371
        conv = jnp.abs(zfac) < _RDG_EPSI10
        dv_new = jnp.where(selected & ~conv, sr(zfac * r1), dv)  # :378
        cn_new = jnp.where(selected, jnp.where(conv, 0.0, fmax(0.0, -dv_new)), cn)  # :373,379
        op_new = jnp.where(selected, jnp.where(conv, 0.0, fmax(0.0, dv_new)), op_p)  # :374,380
        ato_conv = fmax(0.0, sr(1.0 - fortran_sum0(new["a_i"])))  # :375
        new["ato_i"] = jnp.where(selected & conv, ato_conv, new["ato_i"])
        keep = {k: jnp.where(active, new[k], st[k]) for k in st}
        nxt = active & jnp.any(selected & ~conv)  # :358,377 iterate_ridging
        return (keep, jnp.where(active, cn_new, cn), jnp.where(active, op_new, op),
                jnp.where(active, dv_new, dv), nxt, n + active.astype(jnp.int32))

    active0 = jnp.any(selected)  # :334 IF( npti > 0 )
    carry = (s, closing_net, opning, zdivu, active0, jnp.asarray(0, jnp.int32))
    s, _, _, _, _, n_passes = jax.lax.fori_loop(0, _RDG_ITERMAX - 1, one_pass, carry)  # :341
    return SI3RdgrftResult(SI3RdgrftState(**{k: s[k] for k in SI3RdgrftState._fields}),
                           selected, n_passes)
