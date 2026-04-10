"""Ice Thickness Distribution (ITD) — multi-category sea ice.

Implements an N-category ice model following the CICE framework
(Hunke & Dukowicz 1997).  Each category carries its own thickness,
temperature, and areal fraction.  (Snow depth is **not** tracked.)

The ``linear_remap`` function is a **simplified category transfer
scheme** that moves ice volume and area between adjacent categories
when thickness exceeds category bounds.  It is *not* a faithful
implementation of the Lipscomb (2001) linear remapping (which fits
a piecewise-linear g(h) within each category).  Post-remap clamping
ensures category means stay within bounds, but conservation is only
approximate when clamping activates.

All functions are JAX-compatible (differentiable, JIT-friendly).

References
----------
- Hunke, E. C. & Dukowicz, J. K. (1997): An elastic-viscous-plastic model
  for sea ice dynamics. J. Phys. Oceanogr., 27, 1849-1867.
- Lipscomb, W. H. (2001): Remapping the thickness distribution in sea ice
  models. J. Geophys. Res., 106(C7), 13989-14000.  *(Cited for context;
  the full algorithm is not implemented here.)*
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field


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


def category_bounds(n_cat: int) -> jnp.ndarray:
    """Return lower thickness bounds for *n_cat* categories.

    Uses CICE-standard bounds when available (1, 3, 5, 7 categories).
    For other values, returns evenly spaced bounds from 0 to 4 m.

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
    conc_total = jnp.sum(concentration, axis=-1)
    conc_safe = jnp.maximum(conc_total, 1e-20)

    # Volume-conserving mean thickness: sum(h_k * a_k) / sum(a_k)
    h_agg = jnp.sum(h_ice * concentration, axis=-1) / conc_safe
    h_agg = jnp.where(conc_total > 0.0, h_agg, 0.0)

    # Area-weighted mean temperature
    T_agg = jnp.sum(T_ice * concentration, axis=-1) / conc_safe
    T_agg = jnp.where(conc_total > 0.0, T_agg, 271.35)

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
    T_mc = T_ice[..., jnp.newaxis] * in_cat_f + 271.35 * (1.0 - in_cat_f)
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
    T_freeze_ocean: float = 271.35,
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
    # its own excess to avoid non-conservative volume loss.
    z_pad = jnp.zeros_like(vol_excess[..., :1])
    vol_receive = jnp.concatenate([z_pad, vol_excess[..., :-1]], axis=-1)
    a_receive = jnp.concatenate([z_pad, a_excess[..., :-1]], axis=-1)
    # Last category: re-add its own excess (nowhere to promote)
    vol_receive = vol_receive.at[..., -1].add(vol_excess[..., -1])
    a_receive = a_receive.at[..., -1].add(a_excess[..., -1])
    vol_remap = vol_remap + vol_receive
    a_remap = a_remap + a_receive

    # Add deficit to previous category (shift right, pad first with zero)
    vol_remap = vol_remap + jnp.concatenate([vol_deficit[..., 1:], z_pad], axis=-1)
    a_remap = a_remap + jnp.concatenate([a_deficit[..., 1:], z_pad], axis=-1)

    # Recover thickness from volume
    a_remap = jnp.clip(a_remap, 0.0, 1.0)
    a_safe = jnp.maximum(a_remap, 1e-20)
    h_remap = jnp.where(a_remap > 0.0, vol_remap / a_safe, 0.0)
    h_remap = jnp.maximum(h_remap, 0.0)

    # Post-remap: clamp category mean thickness to bounds.
    # Last category has no finite upper bound (100 m sentinel).
    h_remap = jnp.where(
        (a_remap > 0.0) & (h_remap < lo),
        lo,
        h_remap,
    )
    h_remap = jnp.where(
        (a_remap > 0.0) & (h_remap > hi),
        hi,
        h_remap,
    )

    if T_new is None:
        return h_remap, a_remap

    # Remap enthalpy (E = T * vol) alongside volume to conserve energy.
    E_new = T_new * vol_new
    E_excess = E_new * excess_frac
    E_deficit = E_new * deficit_frac
    E_remap = E_new - E_excess - E_deficit

    # Add excess enthalpy to next category (same pattern as volume)
    z_E = jnp.zeros_like(E_excess[..., :1])
    E_recv = jnp.concatenate([z_E, E_excess[..., :-1]], axis=-1)
    E_recv = E_recv.at[..., -1].add(E_excess[..., -1])
    E_remap = E_remap + E_recv

    # Add deficit enthalpy to previous category
    E_remap = E_remap + jnp.concatenate([E_deficit[..., 1:], z_E], axis=-1)

    # Recover temperature from enthalpy
    vol_safe = jnp.maximum(vol_remap, 1e-30)
    T_remap = jnp.where(vol_remap > 0.0, E_remap / vol_safe, T_new)

    # Clamp temperature to physical bounds
    T_remap = jnp.clip(T_remap, T_ice_min, T_freeze_ocean)

    return h_remap, a_remap, T_remap
