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
    conc_safe = jnp.maximum(conc_total, 1e-20)

    # Volume-conserving mean thickness: sum(h_k * a_k) / sum(a_k)
    h_agg = _agg[..., 1] / conc_safe
    h_agg = jnp.where(conc_total > 0.0, h_agg, 0.0)

    # Area-weighted mean temperature
    T_agg = _agg[..., 2] / conc_safe
    T_agg = jnp.where(conc_total > 0.0, T_agg, constants.T_freeze_ocean)

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
    a_safe = jnp.maximum(a_remap, 1e-20)
    h_remap = jnp.where(a_remap > 0.0, vol_remap / a_safe, 0.0)
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
    h_safe = jnp.maximum(h_clamped, 1e-20)
    a_pre = a_remap
    vol_pre = a_pre * h_pre
    a_rescaled = a_pre * h_pre / h_safe
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
    vol_safe = jnp.maximum(vol_remap, 1e-30)
    T_remap = jnp.where(vol_remap > 0.0, E_remap / vol_safe, T_new)

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
