"""Ice Thickness Distribution (ITD) — multi-category sea ice.

Implements an N-category ice model following the CICE framework
(Hunke & Dukowicz 1997; Lipscomb 2001). Each category carries its own
thickness, snow depth, temperature, and areal fraction.

The linear remapping scheme (Lipscomb 2001) redistributes ice across
categories after thermodynamic growth/melt so that each category's mean
thickness stays within its prescribed bounds.

All functions are JAX-compatible (differentiable, JIT-friendly).

References
----------
- Hunke, E. C. & Dukowicz, J. K. (1997): An elastic-viscous-plastic model
  for sea ice dynamics. J. Phys. Oceanogr., 27, 1849-1867.
- Lipscomb, W. H. (2001): Remapping the thickness distribution in sea ice
  models. J. Geophys. Res., 106(C7), 13989-14000.
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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Redistribute ice across categories after thermodynamic changes.

    Simplified linear remapping following Lipscomb (2001): ice that has
    grown beyond its category upper bound is moved to the next category;
    ice that has melted below its lower bound is moved to the previous
    category.

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

    Returns
    -------
    h_remap : array (..., n_cat)
        Remapped thickness.
    a_remap : array (..., n_cat)
        Remapped concentration.
    """
    lo = category_bounds(n_cat)
    hi = upper_bounds(n_cat)

    # Volume = h * a for each category
    vol_new = h_new * a_new

    # Transfer ice that exceeds upper bound to next category
    # Transfer ice that falls below lower bound to previous category
    vol_remap = vol_new.copy()
    a_remap = a_new.copy()

    for k in range(n_cat):
        # Fraction of this category's ice that exceeds its upper bound
        excess_frac = jnp.where(
            h_new[..., k] > hi[k],
            jnp.clip((h_new[..., k] - hi[k]) / jnp.maximum(h_new[..., k], 1e-10), 0.0, 1.0),
            0.0,
        )
        # Fraction below lower bound
        deficit_frac = jnp.where(
            (h_new[..., k] < lo[k]) & (h_new[..., k] > 0.0),
            jnp.clip((lo[k] - h_new[..., k]) / jnp.maximum(lo[k], 1e-10), 0.0, 1.0),
            0.0,
        )

        # Move excess volume to next category (if exists)
        vol_excess = vol_remap[..., k] * excess_frac
        a_excess = a_remap[..., k] * excess_frac
        vol_remap = vol_remap.at[..., k].add(-vol_excess)
        a_remap = a_remap.at[..., k].add(-a_excess)
        if k < n_cat - 1:
            vol_remap = vol_remap.at[..., k + 1].add(vol_excess)
            a_remap = a_remap.at[..., k + 1].add(a_excess)

        # Move deficit volume to previous category (if exists)
        vol_deficit = vol_remap[..., k] * deficit_frac
        a_deficit = a_remap[..., k] * deficit_frac
        vol_remap = vol_remap.at[..., k].add(-vol_deficit)
        a_remap = a_remap.at[..., k].add(-a_deficit)
        if k > 0:
            vol_remap = vol_remap.at[..., k - 1].add(vol_deficit)
            a_remap = a_remap.at[..., k - 1].add(a_deficit)

    # Recover thickness from volume
    a_remap = jnp.clip(a_remap, 0.0, 1.0)
    a_safe = jnp.maximum(a_remap, 1e-20)
    h_remap = jnp.where(a_remap > 0.0, vol_remap / a_safe, 0.0)
    h_remap = jnp.maximum(h_remap, 0.0)

    return h_remap, a_remap
