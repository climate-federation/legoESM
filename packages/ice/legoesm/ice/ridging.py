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

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.ice.config import RidgingConfig
from legoesm.ice.itd import category_bounds, upper_bounds

# Canonical ridging defaults live on RidgingConfig (single source of truth);
# kwarg signatures default to these. Salt mass uses S [PSU = g/kg] x volume x
# rho_ice [kg/m^3]; the 1e-3 converts g/kg -> kg/kg so salt is in kg.
_RIDGE_DEFAULTS = RidgingConfig()
_PSU_TO_FRACTION = 1.0e-3
# Minimum ridge-thickness range width [m]: keeps H_max strictly above H_min so
# the uniform-g overlap integral has a finite, well-defined support.
_MIN_RIDGE_WIDTH_M = 1.0e-3


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
    raw = a_cat * jnp.exp(-h_cat / jnp.maximum(e_star, 1e-3))  # coeff-ok: e_star divide-safety floor [m]
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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
           jnp.ndarray, jnp.ndarray]:
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
        _has_participation, jnp.sum(weights * h_cat) / _w_sum_safe, 0.0,
    )
    H_min_est = 2.0 * h_part_est
    H_max_est = jnp.minimum(mu_rdg * jnp.sqrt(jnp.maximum(h_part_est, 1e-6)), H_star)
    H_max_est = jnp.minimum(H_max_est, hi[-1])
    H_max_est = jnp.maximum(H_max_est, H_min_est + _MIN_RIDGE_WIDTH_M)
    H_mean_est = 0.5 * (jnp.minimum(H_min_est, hi[-1]) + jnp.minimum(H_max_est, hi[-1]))
    comp = jnp.clip(
        1.0 - h_part_est / jnp.maximum(H_mean_est, 1e-6), 0.5, 1.0,
    )
    da_ridged = jnp.minimum(da_ridged_request / comp, a_total)

    da_per_cat = da_ridged * weights
    # Limit per-cat draw to its available area.
    da_per_cat = jnp.minimum(da_per_cat, a_cat)

    # Volume + snow removed per donor cat.
    fraction_taken = jnp.where(
        a_cat > 1e-30, da_per_cat / jnp.where(a_cat > 1e-30, a_cat, 1.0), 0.0,
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
    H_max = jnp.minimum(H_max, hi[-1])                       # ceiling at top bound
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
    snow_to_ocean_kg_m2 = (
        snow_to_ocean_m_snow_per_m2 * constants.rho_snow
    )

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
    salt_donated_total = jnp.sum(
        S_ice_cat * dV_per_cat,
    ) * constants.rho_ice * _PSU_TO_FRACTION
    salt_to_cat = jnp.where(
        V_ridge_total > 1e-30,
        salt_donated_total * dV_ridge_to_cat
        / jnp.where(V_ridge_total > 1e-30, V_ridge_total, 1.0),
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

    return (a_new, h_new, Vsnow_new, S_new, Vpond_new,
            snow_to_ocean_kg_m2 / dt, pond_to_ocean_kg_m2 / dt)


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
        _flat(a_cat), _flat(h_cat), _flat(V_snow_cat), _flat(S_ice_cat),
        _flat(V_pond_cat),
        closing_flat,
        lo, hi, dt,
        e_star, mu_rdg, H_star, snow_fraction_retained,
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
