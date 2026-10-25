"""Ice Thickness Distribution (ITD) — multi-category sea ice.

Implements an N-category ice model following the CICE framework
(Hunke & Dukowicz 1997).  Each category carries its own thickness,
temperature, and (optionally) bulk salinity, snow volume, and pond
volume.

Two remapping schemes are provided:

- ``linear_remap`` — simplified category transfer (legacy).  Moves
  volume and area between adjacent categories when the post-thermo
  mean thickness exits the category bounds, with a local
  volume-conserving rescale.  Cheap, monotone, but does not
  represent the sub-category thickness distribution.
- ``lipscomb_2001_remap`` — true piecewise-linear remapping
  (Lipscomb 2001).  Fits a linear sub-distribution ``g(h)`` within
  each category — anchored at the BIN CENTRE, so in the central third
  (``|η| ≤ H/6``) both the area and volume moments are exact and
  ``g ≥ 0`` — displaces the inter-category boundaries by the
  interpolated growth rate, and re-integrates ``g`` over the FIXED
  category bins.  Conserves ice area, ice volume, snow volume, and salt
  mass to machine precision (a per-source-category renormalisation
  restores the moments where the ±H/6 η-clip, needed to keep
  ``g(h) ≥ 0``, engages — a robust approximation of Lipscomb's exact
  cutoff-support outer-third form; see the Faithfulness note).
  Temperature and salinity are carried as volume-weighted intensive
  tracers (not an explicit enthalpy variable).

All functions are JAX-compatible (differentiable, JIT-friendly).

Faithfulness
------------
``tests/ice/unit/test_ice_itd_lipscomb_faithful.py`` pins ``lipscomb_2001_remap``
(previously untested).  In order of authority: (1) the TRUTH-TIER conservation of
ice area, ice volume, salt mass, snow volume, and pond volume to rel 1e-11..1e-12
across grow/melt/mixed/clip-saturating growth (Lipscomb's central claim; outranks
form-matching); (2) the zero-growth identity; (3) a separate NumPy reference
re-derivation of the column kernel pinning all six per-FIXED-bin outputs (the
per-bin SPLIT, which global conservation alone cannot verify — a degenerate
single-bin dump also conserves), plus a directional pin that growth/melt move
areal mass up/down the thickness axis; (4) the published slope coefficient
``_LIPSCOMB_G1_COEFF == 12`` and the exact-Lipscomb two-moment property.

Reconstruction: the linear ``g(h) = a/H + g1·(h − centre)`` is anchored at the
BIN CENTRE with ``g1 = 12·a·η/H³`` (Lipscomb 2001).  In the CENTRAL third
(``|η| ≤ H/6``) this is exact Lipscomb — both the zeroth (area) and first (volume)
moments are preserved analytically and ``g ≥ 0`` (it touches zero at ``η = ±H/6``,
never negative), pinned by
``test_lipscomb_reconstruction_moments_and_positivity``.

DEPARTURE (documented): outside the central third the implementation CLIPS ``η`` to
``±H/6`` and restores the moments with a per-source-category renormalisation
(``A_k_scale`` / ``V_k_scale``), rather than Lipscomb's exact cutoff-support
triangle (eqs. 14-15) that shrinks the support to keep a single non-negative
``g(h)`` matching both moments.  The RETAINED clip+renormalisation conserves total
area and volume to machine precision (truth tier) on every state, but in
saturation the area and volume transfers are not moments of one distribution.  The
clip is retained deliberately for robustness: the interpolated boundary
displacement can collapse a displaced bin (``H → 0``) or leave ``h_new`` outside
it — exactly where the ideal cutoff triangle's ``g1 ~ 1/H³`` blows up and LOSES
mass (the cutoff form conserves only when its support assumptions hold), whereas
the eta clip keeps ``g1 ~ 1/H²`` finite and conserving.  Exact cutoff-support with
robust degenerate-bin handling (e.g. a delta/two-point conservative deposition
fallback for collapsed/out-of-support bins) is a documented follow-up.  (A still
earlier mean-anchored ``G0 = a/H`` at ``h̄`` left the zeroth moment
``a·(1 − 12η²/H²) ≠ a`` and let ``g`` go negative near ``η = H/6``; fixed to exact
center anchoring during codex review of this suite.)  ``_lipscomb_ref`` in the
test is a cross-implementation (NumPy) regression oracle for this kernel — exact
Lipscomb in the central third — not a full published-spec oracle in saturation.

References
----------
- Hunke, E. C. & Dukowicz, J. K. (1997): An elastic-viscous-plastic model
  for sea ice dynamics. J. Phys. Oceanogr., 27, 1849-1867.
- Lipscomb, W. H. (2001): Remapping the thickness distribution in sea ice
  models. J. Geophys. Res., 106(C7), 13989-14000.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.source_rounding import nemo_source_round

# Lipscomb (2001) piecewise-linear g(h) slope coefficient: the first-moment term
# of the linear thickness reconstruction is 12*eta/H^3 (fixed published).
_LIPSCOMB_G1_COEFF = 12.0


# ==============================================================================
# Category bounds
# ==============================================================================

# Standard CICE bounds (lower limits in meters).
# Upper bound of last category is infinity (represented by a large number).
CICE_BOUNDS = {
    1: (0.0,),
    3: (0.0, 0.6, 2.4),
    5: (0.0, 0.6, 1.4, 2.4, 3.6),
    7: (0.0, 0.3, 0.7, 1.2, 2.0, 3.0, 4.5),
}


@lru_cache(maxsize=8)
def category_bounds(n_cat: int) -> jnp.ndarray:
    """Return lower thickness bounds for *n_cat* categories.

    Uses CICE-standard bounds when available (1, 3, 5, 7 categories).
    For other values, returns evenly spaced bounds from 0 to 4 m.

    The result is cached per ``n_cat`` so repeated calls share the
    same JAX device array instead of building a fresh ``jnp.array``
    each time (which used to allocate from a Python tuple every step
    when called from inside the ITD pipeline).

    Returns
    -------
    bounds : jnp.ndarray, shape (n_cat,)
        Lower thickness bound for each category [m].
    """
    if n_cat in CICE_BOUNDS:
        return jnp.array(CICE_BOUNDS[n_cat])
    # Fallback: evenly spaced
    return jnp.linspace(0.0, 4.0, n_cat)


def upper_bounds(n_cat: int) -> jnp.ndarray:
    """Return upper thickness bounds for *n_cat* categories.

    The last category has an upper bound of 100 m (effectively infinite).

    Returns
    -------
    bounds : jnp.ndarray, shape (n_cat,)
    """
    lo = category_bounds(n_cat)
    hi = jnp.concatenate([lo[1:], jnp.array([100.0])])
    return hi


# ==============================================================================
# Multi-category state helpers
# ==============================================================================

def aggregate_state(
    h_ice: jnp.ndarray,
    T_ice: jnp.ndarray,
    concentration: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Aggregate multi-category fields to single-category equivalents.

    Parameters
    ----------
    h_ice : array (..., n_cat)
        Per-category ice thickness [m].
    T_ice : array (..., n_cat)
        Per-category surface temperature [K].
    concentration : array (..., n_cat)
        Per-category areal fraction.

    Returns
    -------
    h_agg : array (...)
        Area-weighted mean thickness.
    T_agg : array (...)
        Area-weighted mean temperature.
    conc_agg : array (...)
        Total concentration.
    """
    # All three reductions share the ``concentration`` weight on the
    # category axis — fuse into one stacked sum.
    _stack = jnp.stack(
        [jnp.ones_like(h_ice), h_ice, T_ice], axis=-1,
    ) * concentration[..., None]
    _agg = jnp.sum(_stack, axis=-2)
    conc_total = _agg[..., 0]
    # float32 AD safety: ``jnp.maximum(x, TINY)`` is NOT a sufficient guard.
    # JAX's divide-JVP w.r.t. the denominator is
    # ``mul(mul(neg(g), num), integer_pow(den, -2))`` -- it forms ``den**-2``
    # DIRECTLY, and ``integer_pow(1e-20, -2) == 1e40`` overflows float32 to
    # ``inf``.  With an ice-free numerator that residual is ``0 * inf == NaN``.
    # The NaN survives BOTH guards: the outer ``jnp.where`` transposes to a
    # select that feeds cotangent 0.0 into the divide (0.0 * NaN == NaN), and
    # ``jnp.maximum``'s JVP is a MULTIPLY by a 0/1 mask, not a select
    # (NaN * 0 == NaN).  Use the two-sided idiom (safe denominator INSIDE the
    # branch) already used by ``lipscomb_2001_remap`` below (L581/598).
    # The predicate is deliberately TIGHTENED from ``> 0.0`` to ``> 1e-20``:
    # with ``> 0.0`` a sliver denominator (e.g. 1e-25) still gives
    # ``integer_pow(1e-25, -2) == inf`` and an ``inf`` adjoint.  Forward delta
    # is confined to conc_total in (0, 1e-20], i.e. <= 1e-20 m of discarded
    # ice volume per cell -- below float32 resolution of any budget.
    has_ice = conc_total > 1e-20
    conc_safe = jnp.where(has_ice, conc_total, 1.0)

    # Volume-conserving mean thickness: sum(h_k * a_k) / sum(a_k)
    h_agg = jnp.where(has_ice, _agg[..., 1] / conc_safe, 0.0)

    # Area-weighted mean temperature
    T_agg = jnp.where(has_ice, _agg[..., 2] / conc_safe, constants.T_freeze_ocean)

    return h_agg, T_agg, conc_total


def distribute_to_categories(
    h_ice: jnp.ndarray,
    T_ice: jnp.ndarray,
    concentration: jnp.ndarray,
    n_cat: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Distribute single-category ice into multi-category ITD.

    Places all ice into the category whose bounds contain the given thickness.
    If thickness exceeds all upper bounds, it goes in the last category.

    Parameters
    ----------
    h_ice : array (...)
        Mean ice thickness [m].
    T_ice : array (...)
        Surface temperature [K].
    concentration : array (...)
        Areal fraction.
    n_cat : int
        Number of categories.

    Returns
    -------
    h_mc : array (..., n_cat)
    T_mc : array (..., n_cat)
    conc_mc : array (..., n_cat)
    """
    lo = category_bounds(n_cat)
    hi = upper_bounds(n_cat)

    # Determine which category h_ice belongs to
    # h_exp: (..., 1), lo/hi: (n_cat,) → broadcast to (..., n_cat)
    h_exp = h_ice[..., jnp.newaxis]  # (..., 1)
    # Broadcast comparison: (..., 1) vs (n_cat,) → (..., n_cat)
    in_cat = (h_exp >= lo) & (h_exp < hi)

    # For h_ice >= upper_bounds[-1], force into last category
    in_cat = in_cat.at[..., -1].set(
        in_cat[..., -1] | (h_ice >= hi[-1])
    )

    in_cat_f = in_cat.astype(h_ice.dtype)

    h_mc = h_exp * in_cat_f          # (..., n_cat)
    T_mc = T_ice[..., jnp.newaxis] * in_cat_f + constants.T_freeze_ocean * (1.0 - in_cat_f)
    conc_mc = concentration[..., jnp.newaxis] * in_cat_f

    return h_mc, T_mc, conc_mc


# ==============================================================================
# Linear remapping (Lipscomb 2001)
# ==============================================================================

def linear_remap(
    h_old: jnp.ndarray,
    a_old: jnp.ndarray,
    h_new: jnp.ndarray,
    a_new: jnp.ndarray,
    n_cat: int,
    T_new: jnp.ndarray | None = None,
    T_ice_min: float = 180.0,
    T_freeze_ocean: float = constants.T_freeze_ocean,
    T_max: float = constants.T_freeze,
) -> tuple[jnp.ndarray, jnp.ndarray] | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Redistribute ice across categories after thermodynamic changes.

    **Simplified category transfer** (not full Lipscomb 2001): ice that
    has grown beyond its category upper bound is moved to the next
    category; ice that has melted below its lower bound is moved to the
    previous category.  Post-remap thickness is clamped to category
    bounds to prevent drift.  Conservation is exact when no clamping
    activates, and only approximate otherwise.

    Parameters
    ----------
    h_old : array (..., n_cat)
        Pre-thermodynamics thickness.
    a_old : array (..., n_cat)
        Pre-thermodynamics concentration.
    h_new : array (..., n_cat)
        Post-thermodynamics thickness.
    a_new : array (..., n_cat)
        Post-thermodynamics concentration.
    n_cat : int
    T_new : array (..., n_cat) or None
        Post-thermodynamics temperature. If provided, enthalpy is remapped
        alongside volume and the remapped temperature is returned.
    T_ice_min : float
        Lower temperature bound [K] (default 180).
    T_freeze_ocean : float
        Upper temperature bound [K] (default 271.35).

    Returns
    -------
    h_remap : array (..., n_cat)
        Remapped thickness (clamped to category bounds).
    a_remap : array (..., n_cat)
        Remapped concentration (clamped to [0, 1]).
    T_remap : array (..., n_cat)
        Remapped temperature (only if T_new is provided;
        clamped to [T_ice_min, T_freeze_ocean]).
    """
    lo = category_bounds(n_cat)
    hi = upper_bounds(n_cat)

    # Volume = h * a for each category
    vol_new = h_new * a_new

    # Vectorized transfer: compute excess/deficit fractions for all categories
    # excess_frac[..., k]: fraction of category k's ice above its upper bound
    excess_frac = jnp.where(
        h_new > hi,
        jnp.clip((h_new - hi) / jnp.maximum(h_new, 1e-10), 0.0, 1.0),
        0.0,
    )
    # deficit_frac[..., k]: fraction below lower bound
    deficit_frac = jnp.where(
        (h_new < lo) & (h_new > 0.0),
        jnp.clip((lo - h_new) / jnp.maximum(lo, 1e-10), 0.0, 1.0),
        0.0,
    )

    # Volume and area transfers
    vol_excess = vol_new * excess_frac
    a_excess = a_new * excess_frac
    vol_deficit = vol_new * deficit_frac
    a_deficit = a_new * deficit_frac

    # Remove excess and deficit from each category
    vol_remap = vol_new - vol_excess - vol_deficit
    a_remap = a_new - a_excess - a_deficit

    # Add excess to next category (shift left); last category retains
    # its own excess to avoid non-conservative volume loss.  Use
    # ``jnp.pad`` along the trailing axis instead of constructing a
    # zero strip and concatenating — single ``Pad`` HLO op vs
    # alloc + concat.
    pad_axes = ((0, 0),) * (vol_excess.ndim - 1)
    vol_receive = jnp.pad(vol_excess[..., :-1], (*pad_axes, (1, 0)))
    a_receive = jnp.pad(a_excess[..., :-1], (*pad_axes, (1, 0)))
    # Last category: re-add its own excess (nowhere to promote)
    vol_receive = vol_receive.at[..., -1].add(vol_excess[..., -1])
    a_receive = a_receive.at[..., -1].add(a_excess[..., -1])
    vol_remap = vol_remap + vol_receive
    a_remap = a_remap + a_receive

    # Add deficit to previous category (shift right, pad first with zero)
    vol_remap = vol_remap + jnp.pad(vol_deficit[..., 1:], (*pad_axes, (0, 1)))
    a_remap = a_remap + jnp.pad(a_deficit[..., 1:], (*pad_axes, (0, 1)))

    # Recover thickness from volume
    a_remap = jnp.clip(a_remap, 0.0, 1.0)
    # float32 AD safety: safe denominator INSIDE the branch (see aggregate_state
    # above for why the outer where + jnp.maximum do NOT stop the NaN).
    # Predicate deliberately tightened >0.0 -> >1e-20 to also remove the
    # sliver-denominator ``inf`` adjoint; forward delta <= 1e-20 m volume/cell.
    has_area = a_remap > 1e-20
    a_safe = jnp.where(has_area, a_remap, 1.0)
    h_remap = jnp.where(has_area, vol_remap / a_safe, 0.0)
    h_remap = jnp.maximum(h_remap, 0.0)

    # Volume-conserving clamp.  An earlier implementation clamped
    # h_remap to [lo, hi] without adjusting a_remap, leaking
    # 1–4 % volume per call.  The local fix below rescales the
    # area so ``a_remap * h_remap`` (volume per cell) is preserved
    # whenever the thickness is moved into the category bin:
    #     a_post · h_post = a_pre · h_pre   (V invariant)
    # The Lipscomb piecewise-linear g(h) redistribution (CICE
    # convention) is the long-term fix; until that lands, the
    # local rescale eliminates the systematic mass drift while
    # remaining differentiable and JIT-friendly.  Codex finding #9.
    h_pre = h_remap
    h_clamped_lo = jnp.where(
        (a_remap > 0.0) & (h_pre < lo),
        lo,
        h_pre,
    )
    h_clamped = jnp.where(
        (a_remap > 0.0) & (h_clamped_lo > hi),
        hi,
        h_clamped_lo,
    )
    # Rescale concentration to preserve volume.  Where no clamp
    # fired ``h_clamped == h_pre`` so the ratio is 1.0.  Two regimes:
    #   1. ``a_rescaled ≤ 1``: ordinary case, keep clamped thickness.
    #   2. ``a_rescaled > 1``: would-be excess area is folded back into
    #      the thickness (``h_final = a_pre · h_pre``) so volume is
    #      conserved exactly.  This locally violates the upper bin
    #      bound when concentration saturates, but preserves mass —
    #      the proper Lipscomb redistribution to the next category is
    #      the long-term structural fix.  Codex iter-3 finding #4.
    # float32 AD safety: this divide was UNCONDITIONAL, so the floored
    # denominator NaN'd the adjoint for every ice-free category.  Physical
    # predicate: the volume-conserving area rescale ``a_post = a_pre*h_pre/h_post``
    # is only defined for a category that HAS a post-clamp thickness; with
    # ``h_clamped <= 0`` there is no ice to hold area, so the rescaled area is 0.
    # Byte-identity check: ``h_clamped == 0`` implies ``h_pre == 0`` (when
    # a_remap > 0, h_clamped = max(h_pre, lo) >= h_pre; when a_remap == 0,
    # h_pre == 0 identically from the where above), so the old floored value
    # was ``a_pre*0/1e-20 == 0`` -- exactly the new fallback.
    has_thickness = h_clamped > 1e-20
    h_safe = jnp.where(has_thickness, h_clamped, 1.0)
    a_pre = a_remap
    vol_pre = a_pre * h_pre
    a_rescaled = jnp.where(has_thickness, a_pre * h_pre / h_safe, 0.0)
    saturated = a_rescaled > 1.0
    a_remap = jnp.clip(a_rescaled, 0.0, 1.0)
    # When saturated: put the residual volume back into h.
    h_remap = jnp.where(saturated, vol_pre, h_clamped)

    if T_new is None:
        return h_remap, a_remap

    # Remap enthalpy (E = T * vol) alongside volume to conserve energy.
    E_new = T_new * vol_new
    E_excess = E_new * excess_frac
    E_deficit = E_new * deficit_frac
    E_remap = E_new - E_excess - E_deficit

    # Add excess enthalpy to next category (same pattern as volume)
    e_pad_axes = ((0, 0),) * (E_excess.ndim - 1)
    E_recv = jnp.pad(E_excess[..., :-1], (*e_pad_axes, (1, 0)))
    E_recv = E_recv.at[..., -1].add(E_excess[..., -1])
    E_remap = E_remap + E_recv

    # Add deficit enthalpy to previous category
    E_remap = E_remap + jnp.pad(E_deficit[..., 1:], (*e_pad_axes, (0, 1)))

    # Recover temperature from enthalpy
    # float32 AD safety: safe denominator INSIDE the branch.
    # ``integer_pow(1e-30, -2) == 1e60`` overflows float32 to ``inf``, so the
    # zero-numerator adjoint was NaN despite the outer where.
    has_vol = vol_remap > 1e-30
    vol_safe = jnp.where(has_vol, vol_remap, 1.0)
    T_remap = jnp.where(has_vol, E_remap / vol_safe, T_new)

    # Clamp temperature to physical bounds.  Upper bound is the SURFACE
    # melt point (T_max = T_freeze = 273.15 K), not the saline basal
    # freezing point T_freeze_ocean (271.35 K) — otherwise the remap
    # would re-clamp a melting surface back below 0 C every step and
    # undo the surface-vs-basal melt-point split.  ``T_freeze_ocean`` is
    # kept as the empty-cell fill value above.
    T_remap = jnp.clip(T_remap, T_ice_min, T_max)

    return h_remap, a_remap, T_remap


# ==============================================================================
# True Lipscomb (2001) piecewise-linear remapping
# ==============================================================================

def _lipscomb_column_kernel(
    h_old: jnp.ndarray,
    a_old: jnp.ndarray,
    h_new: jnp.ndarray,
    a_new: jnp.ndarray,
    T_new: jnp.ndarray,
    S_new: jnp.ndarray,
    V_snow_new: jnp.ndarray,
    V_pond_new: jnp.ndarray,
    lo: jnp.ndarray,
    hi: jnp.ndarray,
    dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-column Lipscomb 2001 piecewise-linear remap kernel.

    Operates on 1-D ``(n_cat,)`` arrays.  Builds a linear g(h) within
    each displaced category, then re-integrates over the FIXED bins.
    """
    n_cat = h_new.shape[0]

    # Per-category post-thermo volume and growth rate.
    V_new_cat = h_new * a_new
    dh_dt_cat = (h_new - h_old) / dt

    has_ice = a_new > 1e-10
    centers_fixed = 0.5 * (lo + hi)
    h_bar = jnp.where(has_ice, h_new, centers_fixed)

    # Growth rate at interior boundaries h = lo[1:].
    h_bar_left = h_bar[:-1]
    h_bar_right = h_bar[1:]
    dh_dt_left = dh_dt_cat[:-1]
    dh_dt_right = dh_dt_cat[1:]
    has_ice_left = has_ice[:-1]
    has_ice_right = has_ice[1:]
    denom = h_bar_right - h_bar_left
    safe_denom = jnp.where(jnp.abs(denom) > 1e-10, denom, 1.0)
    w_interp = jnp.where(
        jnp.abs(denom) > 1e-10,
        (lo[1:] - h_bar_left) / safe_denom,
        0.5,
    )
    w_interp = jnp.clip(w_interp, 0.0, 1.0)
    both = has_ice_left & has_ice_right
    only_left = has_ice_left & (~has_ice_right)
    only_right = (~has_ice_left) & has_ice_right
    dh_dt_b_int = jnp.where(
        both,
        (1.0 - w_interp) * dh_dt_left + w_interp * dh_dt_right,
        jnp.where(
            only_left,
            dh_dt_left,
            jnp.where(only_right, dh_dt_right, jnp.zeros_like(dh_dt_left)),
        ),
    )
    # Full boundary growth array of length (n_cat + 1).
    dh_dt_b = jnp.concatenate(
        [jnp.zeros((1,)), dh_dt_b_int, jnp.zeros((1,))],
        axis=0,
    )

    # Displaced bin boundaries for each source category.
    h_L_disp = lo + dt * dh_dt_b[:n_cat]
    h_R_disp = hi + dt * dh_dt_b[1:]
    h_L_disp = h_L_disp.at[0].set(0.0)
    h_R_disp = h_R_disp.at[-1].set(jnp.maximum(h_R_disp[-1], hi[-1]))
    # Clamp displaced bins to the global remap domain so that a very
    # large positive (or negative) growth rate cannot push an interior
    # bin entirely past the FIXED outer bounds [lo[0], hi[-1]] and have
    # its contents silently dropped during the overlap integral.
    # Both bounds are clipped *into* the domain, and the upper bound
    # is then forced to lie strictly *inside* it (otherwise a bin
    # collapsed at the upper edge has zero overlap with the last
    # FIXED bin since its right boundary equals ``hi[-1]``).
    domain_lo = lo[0]
    domain_hi = hi[-1]
    eps_width = 1.0e-6
    h_L_disp = jnp.clip(h_L_disp, domain_lo, domain_hi - eps_width)
    h_R_disp = jnp.clip(h_R_disp, domain_lo + eps_width, domain_hi)
    # Numerical guard — keep displaced bin strictly positive width.
    h_R_disp = jnp.maximum(h_R_disp, h_L_disp + eps_width)
    h_L_disp = jnp.minimum(h_L_disp, h_R_disp - eps_width)

    H = h_R_disp - h_L_disp
    centers_disp = 0.5 * (h_L_disp + h_R_disp)
    eta_raw = h_new - centers_disp
    # Positivity of the BIN-CENTRE-anchored linear g(h) = a/H + g1*(h - centre)
    # constrains |eta| <= H/6 (Lipscomb 2001, g1 = 12*a*eta/H^3).  In the CENTRAL
    # third the reconstruction below is exact Lipscomb (both moments preserved, g
    # >= 0).  Outside it (saturation) eta is CLIPPED to +-H/6 and the per-source
    # renormalisation restores the moments — a robust APPROXIMATION of Lipscomb's
    # cutoff-support triangle (eqs. 14-15), NOT the paper's exact outer-third form.
    # The eta clip (rather than the cutoff triangle) is retained deliberately: it
    # keeps g1 ~ 1/H^2 finite even when the interpolated boundary displacement
    # collapses a bin (H -> 0) or leaves h_new outside [h_L_disp, h_R_disp] — a
    # cutoff triangle there has g1 ~ 1/H^3 and loses mass.  Exact cutoff-support
    # (with robust degenerate-bin handling) is a documented follow-up.
    eta = jnp.clip(eta_raw, -H / 6.0, H / 6.0)
    # Anchor the reconstruction at the BIN CENTRE (exact Lipscomb): the density at
    # the centre is a/H and the slope g1 carries the mean displacement, so BOTH
    # the zeroth (area) and first (volume) moments are preserved exactly for
    # unclipped eta.  (Anchoring at the mean h_bar instead would leave the zeroth
    # moment a*(1 - 12*eta^2/H^2) != a — forcing the rescale to do real work and
    # letting g(h) go negative near |eta| = H/6.)
    anchor_h = centers_disp
    G0 = jnp.where(has_ice, a_new / H, 0.0)
    G1 = jnp.where(has_ice, _LIPSCOMB_G1_COEFF * eta * a_new / (H ** 3), 0.0)

    # Overlap of displaced bin k (source) with FIXED bin j (target).
    lo_j = lo[:, None]
    hi_j = hi[:, None]
    h_L_k = h_L_disp[None, :]
    h_R_k = h_R_disp[None, :]
    a_over = jnp.maximum(lo_j, h_L_k)
    b_over = jnp.minimum(hi_j, h_R_k)
    overlap_w = jnp.maximum(b_over - a_over, 0.0)

    G0_k = G0[None, :]
    G1_k = G1[None, :]
    anchor_k = anchor_h[None, :]

    int_area = (
        G0_k * overlap_w
        + G1_k * ((b_over - anchor_k) ** 2 - (a_over - anchor_k) ** 2) / 2.0
    )
    int_area = jnp.where(overlap_w > 0.0, jnp.maximum(int_area, 0.0), 0.0)
    int_vol = (
        (G0_k - G1_k * anchor_k) * (b_over ** 2 - a_over ** 2) / 2.0
        + G1_k * (b_over ** 3 - a_over ** 3) / 3.0
    )
    int_vol = jnp.where(overlap_w > 0.0, jnp.maximum(int_vol, 0.0), 0.0)

    # Per-source-cat rescale to recover the exact a_new[k], V_new_cat[k] under the
    # eta clip.  In the central third both scales are 1 (moments already exact);
    # in saturation they restore total area and volume conservation.
    V_k_disp_sum = jnp.sum(int_vol, axis=0)
    A_k_disp_sum = jnp.sum(int_area, axis=0)
    V_k_scale = jnp.where(
        (V_k_disp_sum > 1e-30) & has_ice,
        V_new_cat / jnp.where(V_k_disp_sum > 1e-30, V_k_disp_sum, 1.0),
        0.0,
    )
    A_k_scale = jnp.where(
        (A_k_disp_sum > 1e-30) & has_ice,
        a_new / jnp.where(A_k_disp_sum > 1e-30, A_k_disp_sum, 1.0),
        0.0,
    )
    int_vol = int_vol * V_k_scale[None, :]
    int_area = int_area * A_k_scale[None, :]

    a_remap = jnp.sum(int_area, axis=1)
    V_remap = jnp.sum(int_vol, axis=1)
    a_remap = jnp.clip(a_remap, 0.0, 1.0)
    V_remap = jnp.maximum(V_remap, 0.0)
    a_safe = jnp.where(a_remap > 1e-30, a_remap, 1.0)
    h_remap = jnp.where(a_remap > 1e-30, V_remap / a_safe, 0.0)

    inv_V_remap = jnp.where(
        V_remap > 1e-30,
        1.0 / jnp.where(V_remap > 1e-30, V_remap, 1.0),
        0.0,
    )

    T_weighted = jnp.sum(T_new[None, :] * int_vol, axis=1)
    T_remap = jnp.where(
        V_remap > 1e-30,
        T_weighted * inv_V_remap,
        constants.T_freeze_ocean,
    )
    S_weighted = jnp.sum(S_new[None, :] * int_vol, axis=1)
    S_remap = jnp.where(V_remap > 1e-30, S_weighted * inv_V_remap, 0.0)
    V_safe = jnp.where(V_new_cat > 1e-30, V_new_cat, 1.0)
    snow_factor = jnp.where(V_new_cat > 1e-30, V_snow_new / V_safe, 0.0)
    V_snow_remap = jnp.sum(int_vol * snow_factor[None, :], axis=1)
    pond_factor = jnp.where(V_new_cat > 1e-30, V_pond_new / V_safe, 0.0)
    V_pond_remap = jnp.sum(int_vol * pond_factor[None, :], axis=1)

    return h_remap, a_remap, T_remap, S_remap, V_snow_remap, V_pond_remap


def lipscomb_2001_remap(
    h_old: jnp.ndarray,
    a_old: jnp.ndarray,
    h_new: jnp.ndarray,
    a_new: jnp.ndarray,
    n_cat: int,
    dt: float,
    T_new: jnp.ndarray | None = None,
    S_new: jnp.ndarray | None = None,
    V_snow_new: jnp.ndarray | None = None,
    V_pond_new: jnp.ndarray | None = None,
    T_ice_min: float = 180.0,
    T_freeze_ocean: float = constants.T_freeze_ocean,
    T_max: float = constants.T_freeze,
) -> dict:
    """Lipscomb (2001) piecewise-linear ITD remapping.

    Conservation-form redistribution of ice across the FIXED category
    bins after thermodynamic growth/melt has moved the per-category
    mean thickness.  The algorithm:

    1. Compute the growth rate ``dh/dt`` of each category from
       ``(h_new - h_old) / dt``.
    2. Linearly interpolate ``dh/dt`` to the interior category
       boundaries in ``h``-space (using neighbour mean thicknesses).
    3. Displace each interior boundary by ``dt · dh/dt(boundary)``.
       The displaced (h_L_disp_k, h_R_disp_k) range is where the ice
       that was in cat k actually lives after the step.
    4. Fit a bin-centre-anchored linear thickness distribution
       ``g_k(h)`` on [h_L_disp_k, h_R_disp_k].  In the central third
       (``|η| ≤ H/6``) it matches BOTH ``a_new[k]`` and ``h_new[k]``
       (exact Lipscomb).  Outside it, ``η`` is clipped to ±H/6 (to keep
       ``g ≥ 0``) and a per-source renormalisation restores the total
       moments — a robust approximation of Lipscomb's cutoff-support
       (see the module Faithfulness note); the area/volume transfers are
       then not moments of a single ``g``.
    5. Integrate ``g_k`` over each FIXED bin ``j`` to obtain the
       remapped per-bin area, volume, and intensive tracers.

    All input arrays have a trailing category axis ``n_cat``.  The
    spatial axes ahead of it are arbitrary and processed in parallel
    via ``jax.vmap``.

    Parameters
    ----------
    h_old, a_old : arrays ``(..., n_cat)``
        Pre-thermodynamics thickness and concentration.
    h_new, a_new : arrays ``(..., n_cat)``
        Post-thermodynamics thickness and concentration.
    n_cat : int
        Number of categories.
    dt : float
        Time step over which thermodynamics ran [s].  Used to recover
        boundary displacements ``Δh = dt · dh/dt``.
    T_new : array ``(..., n_cat)`` or None
        Per-category temperature [K].  Remapped as a
        volume-weighted intensive tracer.
    S_new : array ``(..., n_cat)`` or None
        Per-category bulk salinity [g/kg].  Volume-weighted.
    V_snow_new : array ``(..., n_cat)`` or None
        Per-category snow volume ``h_snow * a`` [m].  Extensive —
        transferred in proportion to the ice-volume transfer between
        source-displaced and fixed bins.
    V_pond_new : array ``(..., n_cat)`` or None
        Per-category pond volume ``h_pond * pond_area * a`` [m].
        Same convention as ``V_snow_new``.
    T_ice_min, T_freeze_ocean : float
        Temperature bounds applied to ``T_remap``.

    Returns
    -------
    result : dict
        ``"h"`` : remapped thickness ``(..., n_cat)``
        ``"a"`` : remapped concentration ``(..., n_cat)``
        ``"T"`` : remapped temperature (only if T_new supplied)
        ``"S"`` : remapped salinity (only if S_new supplied)
        ``"V_snow"`` : remapped snow volume (only if V_snow_new supplied)
        ``"V_pond"`` : remapped pond volume (only if V_pond_new supplied)
    """
    lo = category_bounds(n_cat)
    hi = upper_bounds(n_cat)

    spatial_shape = h_new.shape[:-1]
    n_cells = 1
    for s in spatial_shape:
        n_cells *= s

    def _flat(x):
        return x.reshape((n_cells, n_cat))

    # Replace absent inputs with zero arrays of matching shape — the
    # kernel always evaluates the four extras but unused branches in
    # the returned dict are dropped at the Python level below.
    zero_like_cat = jnp.zeros_like(_flat(h_new))
    T_in = _flat(T_new) if T_new is not None else zero_like_cat
    S_in = _flat(S_new) if S_new is not None else zero_like_cat
    Vs_in = _flat(V_snow_new) if V_snow_new is not None else zero_like_cat
    Vp_in = _flat(V_pond_new) if V_pond_new is not None else zero_like_cat

    vmapped = jax.vmap(
        _lipscomb_column_kernel,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None, None),
    )
    h_f, a_f, T_f, S_f, Vs_f, Vp_f = vmapped(
        _flat(h_old), _flat(a_old), _flat(h_new), _flat(a_new),
        T_in, S_in, Vs_in, Vp_in,
        lo, hi, dt,
    )

    full_shape = spatial_shape + (n_cat,)
    result = {
        "h": h_f.reshape(full_shape),
        "a": a_f.reshape(full_shape),
    }
    if T_new is not None:
        # Upper clamp is the SURFACE melt point (T_max = T_freeze =
        # 273.15 K), not the basal freezing point T_freeze_ocean
        # (271.35 K); clamping to the latter would re-freeze a melting
        # surface every remap and defeat the surface-melt-point split.
        result["T"] = jnp.clip(T_f.reshape(full_shape), T_ice_min, T_max)
    if S_new is not None:
        result["S"] = jnp.clip(S_f.reshape(full_shape), 0.0, None)
    if V_snow_new is not None:
        result["V_snow"] = jnp.maximum(Vs_f.reshape(full_shape), 0.0)
    if V_pond_new is not None:
        result["V_pond"] = jnp.maximum(Vp_f.reshape(full_shape), 0.0)
    return result


def _nemo_minsd(first: jax.Array, second: jax.Array) -> jax.Array:
    """gfortran ``MINSD``: choose the second operand on a tie/unordered pair."""
    return jnp.where(first < second, first, second)


def _nemo_maxsd(first: jax.Array, second: jax.Array) -> jax.Array:
    """gfortran ``MAXSD``: choose the second operand on a tie/unordered pair."""
    return jnp.where(first > second, first, second)


def _nemo_minsd_src_first(first: jax.Array, second: jax.Array) -> jax.Array:
    """``MINSD`` site whose emitted source register holds the first operand."""
    return jnp.where(second < first, second, first)


def _nemo_maxsd_src_first(first: jax.Array, second: jax.Array) -> jax.Array:
    """``MAXSD`` site whose emitted source register holds the first operand."""
    return jnp.where(second > first, second, first)


def _si3_itd_glinear(
    hb_l: jax.Array,
    hb_r: jax.Array,
    h_ice: jax.Array,
    a_ice: jax.Array,
    *,
    epsi10: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Exact ``itd_glinear`` transcription (compiled ``iceitd.f90:368-429``)."""
    sr = nemo_source_round
    active = (a_ice > epsi10) & (h_ice > epsi10)
    one_third = sr(jnp.asarray(1.0, h_ice.dtype) / jnp.asarray(3.0, h_ice.dtype))
    h_l0 = jnp.where(active, hb_l, 0.0)
    h_r0 = jnp.where(active, hb_r, 0.0)
    h_13 = sr(one_third * sr(sr(2.0 * h_l0) + h_r0))
    h_23 = sr(one_third * sr(h_l0 + sr(2.0 * h_r0)))
    h_r = jnp.where(active & (h_ice < h_13), sr(sr(3.0 * h_ice) - sr(2.0 * h_l0)), h_r0)
    h_l = jnp.where(active & ~(h_ice < h_13) & (h_ice > h_23), sr(sr(3.0 * h_ice) - sr(2.0 * h_r)), h_l0)
    width = sr(h_r - h_l)
    width_safe = jnp.where(width > 0.0, width, 1.0)
    inv_width = jnp.where(width > 0.0, sr(1.0 / width_safe), 0.0)
    work1 = sr(sr(6.0 * a_ice) * inv_width)
    work2 = sr(sr(h_ice - h_l) * inv_width)
    g0 = sr(work1 * sr(sr(2.0 * one_third) - work2))
    g1 = sr(sr(sr(2.0 * inv_width) * work1) * sr(work2 - 0.5))
    return (
        jnp.where(active, g0, 0.0),
        jnp.where(active, g1, 0.0),
        jnp.where(active, h_l, 0.0),
        jnp.where(active, h_r, 0.0),
    )


def si3_ice_itd_rem(
    *,
    h_i_b: jax.Array,
    a_i_b: jax.Array,
    h_i: jax.Array,
    a_i: jax.Array,
    v_i: jax.Array,
    v_s: jax.Array,
    sv_i: jax.Array,
    oa_i: jax.Array,
    a_ip: jax.Array,
    v_ip: jax.Array,
    v_il: jax.Array,
    t_su: jax.Array,
    e_s: jax.Array,
    e_i: jax.Array,
    szv_i: jax.Array,
    hi_max: jax.Array,
    rn_amax_2d: jax.Array,
    nn_icesal: int,
    ln_pnd: bool,
    ln_pnd_lids: bool,
    epsi10: float,
    epsi20: float,
    rt0: float,
) -> dict[str, jax.Array]:
    """NEMO 5.0.2 SI3 ``ice_itd_rem`` at its production array boundary.

    This is the single exact implementation used by the ORCA2 O13 scorer and
    by the production multi-category SI3-LEV remap branch.  Category is the
    last axis; snow/ice layer is the penultimate axis of ``e_s/e_i/szv_i``.
    The transcription follows the executing reference-build source at
    ``ORCA2_SAS_ICE_O8O16_REF/BLD/ppsrc/nemo/iceitd.f90:118-597``.
    """
    sr = nemo_source_round
    dtype = h_i.dtype
    eps10 = jnp.asarray(epsi10, dtype=dtype)
    eps20 = jnp.asarray(epsi20, dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)
    jpl = h_i.shape[-1]
    if jpl < 2:
        raise ValueError("si3_ice_itd_rem requires at least two categories")
    if hi_max.shape != (jpl + 1,):
        raise ValueError(f"hi_max must have shape ({jpl + 1},)")
    if nn_icesal not in (1, 2, 3, 4):
        raise ValueError("nn_icesal must be an explicit NEMO selector in 1..4")

    originals = {
        "h_i": h_i, "a_i": a_i, "v_i": v_i, "v_s": v_s,
        "sv_i": sv_i, "oa_i": oa_i, "a_ip": a_ip, "v_ip": v_ip,
        "v_il": v_il, "t_su": t_su, "e_s": e_s, "e_i": e_i,
        "szv_i": szv_i,
    }

    # iceitd.f90:164-237 -- identify ice and construct displaced boundaries.
    from legoesm.ice.si3_orca2_state import fortran_sum_categories
    at_i = fortran_sum_categories(a_i, axis=-1)
    remap = at_i > eps10
    zdh = jnp.where(a_i > eps10, sr(h_i - h_i_b), zero)
    bounds = [jnp.zeros_like(at_i)]
    for jl in range(jpl - 1):
        left_before = a_i_b[..., jl] > eps10
        right_before = a_i_b[..., jl + 1] > eps10
        denom = sr(h_i_b[..., jl + 1] - h_i_b[..., jl])
        denom_safe = jnp.where(left_before & right_before, denom, 1.0)
        slope = sr(sr(zdh[..., jl + 1] - zdh[..., jl]) / denom_safe)
        both = sr(sr(hi_max[jl + 1] + zdh[..., jl]) + sr(slope * sr(hi_max[jl + 1] - h_i_b[..., jl])))
        left_only = sr(hi_max[jl + 1] + zdh[..., jl])
        right_only = sr(hi_max[jl + 1] + zdh[..., jl + 1])
        boundary = jnp.where(
            left_before & right_before,
            both,
            jnp.where(left_before, left_only, jnp.where(right_before, right_only, hi_max[jl + 1])),
        )
        bounds.append(boundary)
        remap = remap & ~((a_i[..., jl] > eps10) & (h_i[..., jl] > sr(boundary - eps10)))
        remap = remap & ~((a_i[..., jl + 1] > eps10) & (h_i[..., jl + 1] < sr(boundary + eps10)))
        remap = remap & ~(boundary < hi_max[jl]) & ~(boundary > hi_max[jl + 2])

    last_expr = sr(sr(3.0 * h_i[..., -1]) - sr(2.0 * bounds[-1]))
    last = jnp.where(
        a_i[..., -1] > eps10,
        _nemo_maxsd_src_first(hi_max[-2], last_expr),
        hi_max[-1],
    )
    bounds.append(last)
    remap = remap & ~(h_i_b[..., 0] < sr(hi_max[0] + eps10))
    remap = remap & ~(h_i_b[..., 0] > sr(hi_max[1] - eps10))
    zhbnew = jnp.stack(bounds, axis=-1)

    # iceitd.f90:252-312 -- thin-ice loss and g(h) for every category.
    g0_first, g1_first, h_l_first, h_r_first = _si3_itd_glinear(
        jnp.broadcast_to(hi_max[0], at_i.shape),
        jnp.broadcast_to(hi_max[1], at_i.shape),
        h_i_b[..., 0], a_i[..., 0], epsi10=eps10,
    )
    dh0 = sr(h_i[..., 0] - h_i_b[..., 0])
    melt = remap & (a_i[..., 0] > eps10) & (dh0 < 0.0)
    dh0_melt = _si3_o13_fused_melt_min(
        neg_dh0=sr(-dh0), hi_max=hi_max[1], h_r=h_r_first
    )
    eta_max = sr(dh0_melt - h_l_first)
    do_loss = melt & (eta_max > 0.0)
    x1 = eta_max
    x2 = sr(sr(0.5 * eta_max) * eta_max)
    da0_raw = sr(sr(g1_first * x2) + sr(g0_first * x1))
    hb_safe = jnp.where(jnp.abs(h_i_b[..., 0]) > eps20, h_i_b[..., 0], 1.0)
    da_max = sr(a_i[..., 0] * sr(1.0 - sr(h_i[..., 0] / hb_safe)))
    da0 = _nemo_minsd(da0_raw, da_max)
    a0_after = sr(a_i[..., 0] - da0)
    a0_safe = jnp.where(do_loss & (jnp.abs(a0_after) > eps20), a0_after, 1.0)
    h0_after = sr(sr(h_i[..., 0] * a_i[..., 0]) / a0_safe)
    v0_after = sr(a0_after * h0_after)
    h_i = h_i.at[..., 0].set(jnp.where(do_loss, h0_after, h_i[..., 0]))
    a_i = a_i.at[..., 0].set(jnp.where(do_loss, a0_after, a_i[..., 0]))
    v_i = v_i.at[..., 0].set(jnp.where(do_loss, v0_after, v_i[..., 0]))
    grow = remap & (a_i[..., 0] > eps10) & ~(dh0 < 0.0)
    zhbnew = zhbnew.at[..., 0].set(jnp.where(grow, _nemo_minsd_src_first(dh0, hi_max[1]), zhbnew[..., 0]))
    area_sum_after_thin_loss = fortran_sum_categories(a_i, axis=-1)

    g0s = []
    g1s = []
    h_ls = []
    h_rs = []
    for jl in range(jpl):
        g0_j, g1_j, h_l_j, h_r_j = _si3_itd_glinear(
            zhbnew[..., jl], zhbnew[..., jl + 1], h_i[..., jl], a_i[..., jl], epsi10=eps10,
        )
        g0s.append(g0_j)
        g1s.append(g1_j)
        h_ls.append(h_l_j)
        h_rs.append(h_r_j)
    g0 = jnp.stack(g0s, axis=-1)
    g1 = jnp.stack(g1s, axis=-1)
    h_l = jnp.stack(h_ls, axis=-1)
    h_r = jnp.stack(h_rs, axis=-1)

    # iceitd.f90:317-342 -- integrations across each fixed boundary.
    donors_up = []
    da_transfers = []
    dv_transfers = []
    for jl in range(jpl - 1):
        up = zhbnew[..., jl + 1] > hi_max[jl + 1]
        eta_min_up = sr(_nemo_maxsd(hi_max[jl + 1], h_l[..., jl]) - h_l[..., jl])
        eta_max_up = sr(_nemo_minsd(zhbnew[..., jl + 1], h_r[..., jl]) - h_l[..., jl])
        eta_min = jnp.where(up, eta_min_up, zero)
        eta_max_down = sr(_nemo_minsd_src_first(hi_max[jl + 1], h_r[..., jl + 1]) - h_l[..., jl + 1])
        eta_max = jnp.where(up, eta_max_up, eta_max_down)
        eta_max = _nemo_maxsd(eta_max, eta_min)
        x1 = sr(eta_max - eta_min)
        work1 = sr(eta_min * eta_min)
        work2 = sr(eta_max * eta_max)
        x2 = sr(0.5 * sr(work2 - work1))
        work1 = sr(work1 * eta_min)
        work2 = sr(work2 * eta_max)
        x3 = sr(sr(1.0 / 3.0) * sr(work2 - work1))
        donor_g0 = jnp.where(up, g0[..., jl], g0[..., jl + 1])
        donor_g1 = jnp.where(up, g1[..., jl], g1[..., jl + 1])
        donor_hl = jnp.where(up, h_l[..., jl], h_l[..., jl + 1])
        da = sr(sr(donor_g1 * x2) + sr(donor_g0 * x1))
        dv = sr(sr(sr(donor_g1 * x3) + sr(donor_g0 * x2)) + sr(da * donor_hl))
        donors_up.append(up)
        da_transfers.append(jnp.where(remap, da, zero))
        dv_transfers.append(jnp.where(remap, dv, zero))

    # iceitd.f90:432-597 -- sequential category shifts of every extensive field.
    za_tsfn = sr(a_i * t_su)

    def shift_pair(field, jl, transfer, up):
        left = field[..., jl]
        right = field[..., jl + 1]
        return field.at[..., jl].set(jnp.where(up, sr(left - transfer), sr(left + transfer))).at[..., jl + 1].set(
            jnp.where(up, sr(right + transfer), sr(right - transfer)))

    def shift_ratio(field, jl, ratio, up):
        donor = jnp.where(up, field[..., jl], field[..., jl + 1])
        transfer = sr(donor * ratio)
        return shift_pair(field, jl, transfer, up)

    def shift_ratio_layered(field, jl, ratio, up):
        donor = jnp.where(up[..., None], field[..., :, jl], field[..., :, jl + 1])
        transfer = sr(donor * ratio[..., None])
        left = field[..., :, jl]
        right = field[..., :, jl + 1]
        return field.at[..., :, jl].set(jnp.where(up[..., None], sr(left - transfer), sr(left + transfer))).at[..., :, jl + 1].set(
            jnp.where(up[..., None], sr(right + transfer), sr(right - transfer)))

    for jl in range(jpl - 1):
        up = donors_up[jl]
        da = da_transfers[jl]
        dv = dv_transfers[jl]
        donor_v = jnp.where(up, v_i[..., jl], v_i[..., jl + 1])
        donor_a = jnp.where(up, a_i[..., jl], a_i[..., jl + 1])
        donor_v_safe = jnp.where(donor_v >= eps10, donor_v, 1.0)
        donor_a_safe = jnp.where(donor_a >= eps10, donor_a, 1.0)
        work_v = jnp.where(donor_v >= eps10, sr(dv / donor_v_safe), zero)
        work_a = jnp.where(donor_a >= eps10, sr(da / donor_a_safe), zero)
        a_i = shift_pair(a_i, jl, da, up)
        v_i = shift_pair(v_i, jl, dv, up)
        v_s = shift_ratio(v_s, jl, work_v, up)
        oa_i = shift_ratio(oa_i, jl, work_a, up)
        za_tsfn = shift_ratio(za_tsfn, jl, work_a, up)
        if ln_pnd:
            a_ip = shift_ratio(a_ip, jl, work_a, up)
            v_ip = shift_ratio(v_ip, jl, work_v, up)
            if ln_pnd_lids:
                v_il = shift_ratio(v_il, jl, work_v, up)
        e_s = shift_ratio_layered(e_s, jl, work_v, up)
        e_i = shift_ratio_layered(e_i, jl, work_v, up)
        if nn_icesal == 4:
            szv_i = shift_ratio_layered(szv_i, jl, work_v, up)
        else:
            sv_i = shift_ratio(sv_i, jl, work_v, up)

    # ice_var_roundoff, then the NEMO per-cell concentration cap.
    a_i = jnp.where(a_i < 0.0, zero, a_i)
    v_i = jnp.where(v_i < 0.0, zero, v_i)
    v_s = jnp.where(v_s < 0.0, zero, v_s)
    oa_i = jnp.where(oa_i < 0.0, zero, oa_i)
    e_i = jnp.where(e_i < 0.0, zero, e_i)
    e_s = jnp.where(e_s < 0.0, zero, e_s)
    if ln_pnd:
        a_ip = jnp.where(a_ip < 0.0, zero, a_ip)
        v_ip = jnp.where(v_ip < 0.0, zero, v_ip)
        if ln_pnd_lids:
            v_il = jnp.where((v_il < 0.0) & (v_il > -eps10), zero, v_il)
    if nn_icesal == 4:
        szv_i = jnp.where(szv_i < 0.0, zero, szv_i)
    else:
        sv_i = jnp.where(sv_i < 0.0, zero, sv_i)

    total_a = fortran_sum_categories(a_i, axis=-1)
    cap = total_a > rn_amax_2d
    total_safe = jnp.where(cap, total_a, 1.0)
    for jl in range(jpl):
        capped = sr(sr(a_i[..., jl] * rn_amax_2d) / total_safe)
        a_i = a_i.at[..., jl].set(jnp.where(cap, capped, a_i[..., jl]))

    a_safe = jnp.where(a_i >= eps20, a_i, 1.0)
    h_i = jnp.where(a_i >= eps20, sr(v_i / a_safe), zero)
    t_su = jnp.where(a_i >= eps20, sr(za_tsfn / a_safe), jnp.asarray(rt0, dtype=dtype))

    # Packed NEMO tables scatter only cells surviving the remap checks.
    result = {"at_i": at_i, "remap_mask": remap, "a_i_sum_target": jnp.where(
        remap, jnp.minimum(area_sum_after_thin_loss, rn_amax_2d), at_i)}
    updates = {
        "h_i": h_i, "a_i": a_i, "v_i": v_i, "v_s": v_s,
        "sv_i": sv_i, "oa_i": oa_i, "a_ip": a_ip, "v_ip": v_ip,
        "v_il": v_il, "t_su": t_su, "e_s": e_s, "e_i": e_i,
        "szv_i": szv_i,
    }
    for name, value in updates.items():
        mask = remap
        while mask.ndim < value.ndim:
            mask = mask[..., None]
        result[name] = jnp.where(mask, value, originals[name])
    return result


from legoesm.ice.si3_orca2_state import (  # noqa: E402,F401
    SI3CategoryBounds,
    si3_hfn_category_bounds,
)


def _si3_o13_fused_melt_min(
    *, neg_dh0: jax.Array, hi_max: jax.Array, h_r: jax.Array
) -> jax.Array:
    """Execute fused lines 280/282 in the record object's ``MINSD`` order."""
    return _nemo_minsd(_nemo_minsd(hi_max, h_r), neg_dh0)

# ==============================================================================
# SI3 ORCA2 (rung O6): roundoff, rebinning, zapsmall and ice_cor(kt, 1)
# Cited build: ORCA2_SAS_ICE_O7_REC/BLD/ppsrc/nemo (icevar.f90, iceitd.f90,
# icecor.f90 byte-identical to the O1/O2/O5_REF/O7_REF builds).
# Arrays use NEMO's own index order: (jpi, jpj, jpl) and (jpi, jpj, nlay, jpl).
# ==============================================================================

from typing import NamedTuple  # noqa: E402

_SI3_EPSI10 = 1.0e-10  # ice.F90 epsi10
_SI3_EPSI20 = 1.0e-20  # ice.F90 epsi20
_SI3_HALF = 0.5


class SI3IceCorConfig(NamedTuple):
    """Resolved selectors and constants ``ice_cor(kt, 1)`` reads; no defaults."""

    rn_himin: float
    rn_simin: float
    rn_sinew: float
    nn_icesal: int
    ln_pnd: bool  # ln_pnd_LEV .OR. ln_pnd_TOPO
    ln_pnd_lids: bool
    rdt_ice: float
    rhoi: float
    rhos: float
    rhow: float
    rt0: float


class SI3IceCorState(NamedTuple):
    """NEMO's ice arrays at ``ice_cor`` entry (also its outputs)."""

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
    sz_i: jnp.ndarray
    t_su: jnp.ndarray
    t_i: jnp.ndarray
    t_s: jnp.ndarray
    h_i: jnp.ndarray
    h_s: jnp.ndarray
    h_ip: jnp.ndarray
    h_il: jnp.ndarray
    at_i: jnp.ndarray
    vt_i: jnp.ndarray
    ato_i: jnp.ndarray
    sfx_res: jnp.ndarray
    hfx_res: jnp.ndarray
    wfx_res: jnp.ndarray


def si3_interior_mask(shape2d, nn_hls: int) -> jnp.ndarray:
    mask = jnp.zeros(shape2d, dtype=bool)
    return mask.at[nn_hls:-nn_hls, nn_hls:-nn_hls].set(True)


def si3_safe_div(num, den, ok):
    """``num / den`` where ``ok``; the masked denominator is a gradient-only guard."""
    return num / jnp.where(ok, den, 1.0)


def si3_var_roundoff(fields: dict, cells: jnp.ndarray, *, ln_pnd: bool, ln_pnd_lids: bool,
                     nn_icesal: int) -> dict:
    """``ice_var_roundoff``, ``icevar.f90:875-892``, on the packed ``cells`` only.

    ``WHERE( x < 0 ) x = 0``: a ``-0`` is not ``< 0`` and survives.  ``v_il`` is
    clipped only in ``(-epsi10, 0)`` (``:885``).  ``nn_icesal == 4`` clips
    ``szv_i`` and leaves ``sv_i`` (``:888-892``).
    """
    if nn_icesal != 4:
        raise ValueError("SI3 O6 transcribes nn_icesal == 4 only")
    out = dict(fields)
    c2 = cells[..., None]
    c3 = cells[..., None, None]

    def clip(name, cmask):
        x = out[name]
        out[name] = jnp.where(cmask & (x < 0.0), 0.0, x)

    for name in ("a_i", "v_i", "v_s", "oa_i"):
        clip(name, c2)
    clip("e_i", c3)
    clip("e_s", c3)
    if ln_pnd:
        clip("a_ip", c2)
        clip("v_ip", c2)
        if ln_pnd_lids:
            x = out["v_il"]
            out["v_il"] = jnp.where(c2 & (x < 0.0) & (x > -_SI3_EPSI10), 0.0, x)
    clip("szv_i", c3)
    return out


def _si3_itd_shiftice(s: dict, cells, jl, donor_upper: bool, dai, dvi, amax, cfg):
    """``itd_shiftice`` for one active boundary ``jl`` (``iceitd.f90:432-597``).

    ``ice_itd_reb`` resets every other boundary's donor to 0 between calls
    (``:656-658``, ``:685-687``), so exactly one transfer row is live.
    """
    from legoesm.core.source_rounding import nemo_source_round as sr
    from legoesm.ice.si3_surface import fortran_sum0

    jl1, jl2 = (jl + 1, jl) if donor_upper else (jl, jl + 1)
    c = cells
    s = dict(s)
    zaTsfn = sr(s["a_i"] * s["t_su"])  # :469-473
    v1 = s["v_i"][..., jl1]
    a1 = s["a_i"][..., jl1]
    okv = v1 >= _SI3_EPSI10
    oka = a1 >= _SI3_EPSI10
    zworkv = jnp.where(okv, si3_safe_div(dvi, v1, okv), 0.0)  # :489-491
    zworka = jnp.where(oka, si3_safe_div(dai, a1, oka), 0.0)  # :492-494

    def move(name, amount, layered=False):
        x = s[name]
        cm = c[..., None] if layered else c
        x = x.at[..., jl1].set(jnp.where(cm, sr(x[..., jl1] - amount), x[..., jl1]))
        x = x.at[..., jl2].set(jnp.where(cm, sr(x[..., jl2] + amount), x[..., jl2]))
        s[name] = x

    move("a_i", dai)  # :496-497
    move("v_i", dvi)  # :499-500
    move("v_s", sr(s["v_s"][..., jl1] * zworkv))  # :502-504
    move("oa_i", sr(s["oa_i"][..., jl1] * zworka))  # :506-508
    s["_zaTsfn"] = zaTsfn
    move("_zaTsfn", sr(zaTsfn[..., jl1] * zworka))  # :510-512
    if cfg.ln_pnd:
        move("a_ip", sr(s["a_ip"][..., jl1] * zworka))  # :515-517
        move("v_ip", sr(s["v_ip"][..., jl1] * zworkv))  # :519-521
        if cfg.ln_pnd_lids:
            move("v_il", sr(s["v_il"][..., jl1] * zworkv))  # :524-526
    move("e_s", sr(s["e_s"][..., jl1] * zworkv[..., None]), layered=True)  # :530-534
    move("e_i", sr(s["e_i"][..., jl1] * zworkv[..., None]), layered=True)  # :535-539
    move("szv_i", sr(s["szv_i"][..., jl1] * zworkv[..., None]), layered=True)  # :541-546
    zaTsfn = s.pop("_zaTsfn")

    s = si3_var_roundoff(s, c, ln_pnd=cfg.ln_pnd, ln_pnd_lids=cfg.ln_pnd_lids,
                         nn_icesal=cfg.nn_icesal)  # :563
    ztmp = fortran_sum0(s["a_i"])  # :566
    over = c & (ztmp > amax)
    clamp = sr(sr(s["a_i"] * amax[..., None]) / jnp.where(over, ztmp, 1.0)[..., None])
    s["a_i"] = jnp.where(over[..., None], clamp, s["a_i"])  # :567-570
    ok = c[..., None] & (s["a_i"] >= _SI3_EPSI20)  # :575-581
    s["h_i"] = jnp.where(c[..., None], jnp.where(ok, sr(si3_safe_div(s["v_i"], s["a_i"], ok)), 0.0), s["h_i"])
    s["t_su"] = jnp.where(c[..., None], jnp.where(ok, sr(si3_safe_div(zaTsfn, s["a_i"], ok)), cfg.rt0), s["t_su"])
    return s


def si3_ice_itd_reb(s: dict, *, interior, hi_max, hi_mean, amax, cfg: SI3IceCorConfig) -> dict:
    """``ice_itd_reb``, ``iceitd.f90:623-690``: rebin too-thick then too-thin categories."""
    from legoesm.core.source_rounding import nemo_source_round as sr

    jpl = s["a_i"].shape[-1]
    for jl in range(jpl - 1):  # :628 DO jl = 1, jpl-1 (0-based jl)
        a, v = s["a_i"][..., jl], s["v_i"][..., jl]
        cells = interior & (a > 0.0) & (v > sr(a * hi_max[jl + 1]))  # :632
        dai = sr(_SI3_HALF * a)  # :650
        dvi = sr(v - sr(sr((1.0 - _SI3_HALF) * a) * hi_mean[jl]))  # :651
        s = _si3_itd_shiftice(s, cells, jl, False, dai, dvi, amax, cfg)  # :654
    for jl in range(jpl - 2, -1, -1):  # :664 DO jl = jpl-1, 1, -1
        a, v = s["a_i"][..., jl + 1], s["v_i"][..., jl + 1]
        cells = interior & (a > 0.0) & (v <= sr(a * hi_max[jl + 1]))  # :668
        s = _si3_itd_shiftice(s, cells, jl, True, a, v, amax, cfg)  # :678-683
    return s


def si3_var_zapsmall(s: dict, *, interior, sss_m, sst_m, cfg: SI3IceCorConfig) -> dict:
    """``ice_var_zapsmall``, ``icevar.f90:625-735`` (``nn_icesal == 4`` arm)."""
    from legoesm.core.source_rounding import nemo_source_round as sr
    from legoesm.ice.si3_surface import fortran_sum0

    s = dict(s)
    r1 = 1.0 / cfg.rdt_ice
    i2 = interior[..., None]
    ok = s["a_i"] > _SI3_EPSI10
    s["h_i"] = jnp.where(i2, jnp.where(ok, sr(si3_safe_div(s["v_i"], s["a_i"], ok)), 0.0), s["h_i"])  # :625-627
    small = i2 & (jnp.minimum(jnp.minimum(s["a_i"], s["v_i"]), s["h_i"]) < _SI3_EPSI10)  # :636
    jpl = s["a_i"].shape[-1]
    sfx, hfx, wfx = s["sfx_res"], s["hfx_res"], s["wfx_res"]
    sz_new = jnp.minimum(cfg.rn_simin, sr(cfg.rn_sinew * sss_m))  # :642
    for jl in range(jpl):  # :633-645
        for jk in range(s["szv_i"].shape[-2]):
            m = small[..., jl]
            sfx = jnp.where(m, sr(sfx + sr(sr(s["szv_i"][..., jk, jl] * cfg.rhoi) * r1)), sfx)
    s["szv_i"] = jnp.where(small[..., None, :], 0.0, s["szv_i"])
    s["sz_i"] = jnp.where(small[..., None, :], sz_new[..., None, None], s["sz_i"])
    for name, t_name in (("e_i", "t_i"), ("e_s", "t_s")):  # :664-690, e_i before e_s
        for jl in range(jpl):
            for jk in range(s[name].shape[-2]):
                hfx = jnp.where(small[..., jl], sr(hfx - sr(s[name][..., jk, jl] * r1)), hfx)
        s[name] = jnp.where(small[..., None, :], 0.0, s[name])
        s[t_name] = jnp.where(small[..., None, :], cfg.rt0, s[t_name])
    for jl in range(jpl):  # :695-722
        m = small[..., jl]
        wfx = jnp.where(m, sr(wfx + sr(sr(s["v_i"][..., jl] * cfg.rhoi) * r1)), wfx)
        wfx = jnp.where(m, sr(wfx + sr(sr(s["v_s"][..., jl] * cfg.rhos) * r1)), wfx)
        wfx = jnp.where(m, sr(wfx + sr(sr(sr(s["v_ip"][..., jl] + s["v_il"][..., jl]) * cfg.rhow) * r1)), wfx)
    for name in ("a_i", "v_i", "v_s", "oa_i", "h_i", "h_s", "a_ip", "v_ip", "v_il", "h_ip", "h_il"):
        s[name] = jnp.where(small, 0.0, s[name])
    s["t_su"] = jnp.where(small, sr(sst_m + cfg.rt0)[..., None], s["t_su"])  # :709
    s["sfx_res"], s["hfx_res"], s["wfx_res"] = sfx, hfx, wfx
    s["at_i"] = jnp.where(interior, fortran_sum0(s["a_i"]), s["at_i"])  # :725
    s["vt_i"] = jnp.where(interior, fortran_sum0(s["v_i"]), s["vt_i"])  # :726
    s["ato_i"] = jnp.where(interior & (s["at_i"] == 0.0), 1.0, s["ato_i"])  # :735
    return s


def si3_ice_cor_dyn(state: SI3IceCorState, *, rn_amax_2d, sss_m, sst_m, hi_max, hi_mean,
                    nn_hls: int, config: SI3IceCorConfig) -> SI3IceCorState:
    """``ice_cor(kt, kn=1)``, ``icecor.f90:81-130``, the post-dynamics correction.

    Interior ``A2D(0)`` only, as NEMO; halos are returned unchanged.  Inputs are
    NEMO's arrays at the call (``icedyn.f90:194``); every one is required.
    """
    from legoesm.core.source_rounding import nemo_source_round as sr
    from legoesm.ice.si3_surface import fortran_sum0

    if not isinstance(config, SI3IceCorConfig):
        raise TypeError("ice_cor requires an explicit SI3IceCorConfig")
    if config.nn_icesal != 4:
        raise ValueError("SI3 O6 transcribes nn_icesal == 4 only")
    for name, value in (("rn_amax_2d", rn_amax_2d), ("sss_m", sss_m), ("sst_m", sst_m),
                        ("hi_max", hi_max), ("hi_mean", hi_mean)):
        if value is None:
            raise ValueError(f"ice_cor requires the NEMO array {name!r}; no default")
    if any(v is None for v in state):
        raise ValueError("ice_cor requires every SI3IceCorState array; no default")
    s = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in state._asdict().items()}
    amax = jnp.asarray(rn_amax_2d, jnp.float64)
    sss_m = jnp.asarray(sss_m, jnp.float64)
    sst_m = jnp.asarray(sst_m, jnp.float64)
    hi_max = jnp.asarray(hi_max, jnp.float64)
    hi_mean = jnp.asarray(hi_mean, jnp.float64)
    interior = si3_interior_mask(s["at_i"].shape, nn_hls)
    i2 = interior[..., None]

    ok = s["a_i"] >= _SI3_EPSI20  # :81-83
    s["h_i"] = jnp.where(i2, jnp.where(ok, sr(si3_safe_div(s["v_i"], s["a_i"], ok)), 0.0), s["h_i"])
    thin = i2 & (s["h_i"] < config.rn_himin)
    # Materialized divisor: XLA otherwise rewrites x / constant as x * (1 / constant).
    himin = sr(jnp.asarray(config.rn_himin, jnp.float64))
    if config.ln_pnd:  # :84-86
        s["a_ip"] = jnp.where(thin, sr(sr(s["a_ip"] * s["h_i"]) / himin), s["a_ip"])
    s["a_i"] = jnp.where(thin, sr(sr(s["a_i"] * s["h_i"]) / himin), s["a_i"])  # :87
    at_i = fortran_sum0(s["a_i"])  # :92
    s["at_i"] = jnp.where(interior, at_i, s["at_i"])
    over = interior & (s["at_i"] > amax)  # :93-95
    clamp = sr(sr(s["a_i"] * amax[..., None]) / jnp.where(over, s["at_i"], 1.0)[..., None])
    s["a_i"] = jnp.where(over[..., None], clamp, s["a_i"])
    if s["a_i"].shape[-1] > 1:  # :99
        s = si3_ice_itd_reb(s, interior=interior, hi_max=hi_max, hi_mean=hi_mean, amax=amax, cfg=config)
    r1 = 1.0 / config.rdt_ice
    r1_nlay = 1.0 / s["szv_i"].shape[-2]
    sfx = s["sfx_res"]
    for jl in range(s["a_i"].shape[-1]):  # :114-122
        v = s["v_i"][..., jl]
        lo = sr(sr(config.rn_simin * v) * r1_nlay)
        hi = sr(sr(sr(config.rn_sinew * sss_m) * v) * r1_nlay)
        for jk in range(s["szv_i"].shape[-2]):
            zsal = s["szv_i"][..., jk, jl]
            new = jnp.minimum(jnp.maximum(zsal, lo), hi)  # :117-118, zero ties PLAUSIBLE
            new = jnp.where(interior, new, zsal)
            sfx = jnp.where(interior, sr(sfx - sr(sr(sr(new - zsal) * config.rhoi) * r1)), sfx)
            s["szv_i"] = s["szv_i"].at[..., jk, jl].set(new)
    s["sfx_res"] = sfx
    s = si3_var_zapsmall(s, interior=interior, sss_m=sss_m, sst_m=sst_m, cfg=config)  # :128
    return SI3IceCorState(**{k: s[k] for k in SI3IceCorState._fields})
