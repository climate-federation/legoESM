"""Grid-neutral building blocks for the Shchepetkin & McWilliams 2003
density-Jacobian pressure-gradient force.

These two helpers operate on per-column arrays of shape ``(..., nlev)``
and make no assumption about the leading horizontal dimension(s) — the
internal rolls touch only the level axis.  They are therefore reusable
across every grid type (logically-rectangular lat-lon, cubed-sphere,
unstructured Voronoi, ...).

Lat-lon-specific edge wrappers (``density_jacobian_pgf_smc03_x`` /
``_y``) live in ``latlon_cgrid_operators``; the MPAS Voronoi wrapper
``density_jacobian_pgf_smc03_mpas`` lives below.

Reference: Shchepetkin & McWilliams (2003), JGR Oceans 108(C9), §4.

See ``docs/ocean/experiments/density_jacobian_pgf_plan.md`` (lat-lon
design) and ``docs/ocean/experiments/density_jacobian_pgf_mpas.md``
(MPAS port) for context.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def reconstruct_harmonic_slopes(
    rho_per_cell: jnp.ndarray,
    z_centroid: jnp.ndarray,
    is_active: jnp.ndarray,
    eps: float = 1e-30,
    bottom_slope_2nd_order: bool = False,
) -> jnp.ndarray:
    """Per-cell harmonic-mean monotonized density slopes ``σ_k``.

    For Shchepetkin & McWilliams 2003 density-Jacobian PGF.  Within
    each cell ``k`` of a column we represent ``ρ(z) = ρ_k + σ_k · (z −
    z_centroid_k)``.  The slope ``σ_k`` is the harmonic mean of the
    one-sided slopes computed from the cell-centroid finite differences

        Δρ_top_k = (ρ_{k-1} − ρ_k) / (z_{k-1} − z_k)
        Δρ_bot_k = (ρ_k − ρ_{k+1}) / (z_k − z_{k+1})
        σ_k      = 2 · Δρ_top · Δρ_bot / (Δρ_top + Δρ_bot)

    monotonized to zero at extrema (signs differ).  Two key properties:

    1. For linear ρ(z), ``Δρ_top = Δρ_bot = a`` and ``σ_k = a`` exactly
       in every column, regardless of where the centroids sit.  This
       makes adjacent columns reconstruct ρ at intermediate depths
       identically — the property that lets the rest-state PGF vanish
       on partial cells with shifted centroids.
    2. At local extrema the limiter sets ``σ_k = 0`` (flat-top), so the
       reconstruction is monotone (no overshoots).

    Boundary handling:
    - Top cell (no neighbour above): ``σ_0 = Δρ_bot_0`` (one-sided).
    - Bottom-active cell (no active neighbour below — the partial
      seafloor): ``σ_{bot} = Δρ_top_{bot}`` (one-sided), OR — when
      ``bottom_slope_2nd_order`` — a 3-point 2nd-order backward
      derivative through the bottom three cell centroids.
    - Inactive cells (below seafloor): ``σ = 0``.

    ``bottom_slope_2nd_order`` (default ``False`` → bit-exact legacy):
    the one-sided ``Δρ_top`` estimates dρ/dz at the FACE above the bottom
    centroid, not at the centroid ``z_k``, so it is O(Δz) biased whenever
    ρ(z) is CURVED — which it is under a pressure-dependent EOS (Wright)
    even for horizontally-uniform T,S.  Adjacent columns whose bottom cell
    sits at level ``k`` (shallow, one-sided) vs interior at ``k`` (deep,
    two-sided harmonic) then reconstruct ρ differently → a spurious
    rest-state horizontal PGF at steep partial-cell topography (the
    cubed-sphere cold-start seed).  The 2nd-order backward derivative
    evaluates dρ/dz AT ``z_k`` and is EXACT (exact arithmetic) for linear
    AND quadratic ρ, so it removes the curvature bias.  Needs the cell two
    levels up (``k-2``) active; otherwise falls back to the one-sided
    ``Δρ_top``.  The DEFAULT (``False``) path is BITWISE unchanged (the
    branch is never entered), so the proven lat-lon / tripole / MPAS callers
    — which never pass the kwarg — are exactly untouched.  With the flag ON,
    linear ρ matches the one-sided slope only to round-off (different FP op
    sequence), not bitwise.

    Parameters
    ----------
    rho_per_cell : array, shape (..., nlev)
        Cell-mean density [kg/m³] (often the baroclinic anomaly
        ``ρ'`` from ``iterate_eos_and_pressure_anomaly``).
    z_centroid : array, shape (..., nlev)
        Per-cell centroid depth [m], positive downward.
    is_active : array, shape (..., nlev)
        1.0 for wet cells, 0.0 below the partial seafloor.
    eps : float
        Safety floor for the harmonic-mean denominator.

    Returns
    -------
    sigma : array, shape (..., nlev)
        Per-cell density slope [kg/m⁴] (dρ/dz, positive z downward).

    References
    ----------
    Shchepetkin & McWilliams (2003), JGR Oceans 108(C9), §4.
    """
    rho = rho_per_cell
    z = z_centroid
    active_f = is_active.astype(rho.dtype)

    # Roll along the cell axis to get neighbour values.  Boundary slots
    # (k=0 above, k=nlev-1 below) are filled with the cell's own values
    # so that "Δρ" at the boundary safely evaluates to zero — the
    # boundary mask below selects the correct one-sided fall-back.
    rho_above = jnp.concatenate([rho[..., :1], rho[..., :-1]], axis=-1)
    rho_below = jnp.concatenate([rho[..., 1:], rho[..., -1:]], axis=-1)
    z_above = jnp.concatenate([z[..., :1], z[..., :-1]], axis=-1)
    z_below = jnp.concatenate([z[..., 1:], z[..., -1:]], axis=-1)

    # Has-active-neighbour masks.  The slot at k=0 has no upper
    # neighbour by construction; same for k=nlev-1 below.
    is_active_above = jnp.concatenate(
        [jnp.zeros_like(active_f[..., :1]), active_f[..., :-1]], axis=-1,
    )
    is_active_below = jnp.concatenate(
        [active_f[..., 1:], jnp.zeros_like(active_f[..., -1:])], axis=-1,
    )
    has_top = (active_f * is_active_above) > 0.5
    has_bot = (active_f * is_active_below) > 0.5

    # Safe-divide one-sided slopes.  When there is no active neighbour
    # the denominator can be zero; we substitute 1 to keep gradients
    # finite and zero out the result via ``jnp.where``.
    dz_top = z_above - z
    dz_bot = z - z_below
    safe_dz_top = jnp.where(has_top, dz_top, 1.0)
    safe_dz_bot = jnp.where(has_bot, dz_bot, 1.0)
    delta_top = jnp.where(has_top, (rho_above - rho) / safe_dz_top, 0.0)
    delta_bot = jnp.where(has_bot, (rho - rho_below) / safe_dz_bot, 0.0)

    # Harmonic mean of one-sided slopes (when both signs agree).
    sum_slopes = delta_top + delta_bot
    safe_sum = jnp.where(jnp.abs(sum_slopes) > eps, sum_slopes, eps)
    sigma_harm = 2.0 * delta_top * delta_bot / safe_sum
    same_sign = (delta_top * delta_bot) > 0.0
    sigma_interior = jnp.where(same_sign, sigma_harm, 0.0)

    if bottom_slope_2nd_order:
        # 3-point 2nd-order backward dρ/dz AT the bottom centroid z_k (bottom
        # three centroids k, k-1, k-2), written as a small CORRECTION to the
        # one-sided Δρ_top:
        #     σ_bot = Δρ_top + (Δρ_top − d_up)·h1/(h1+h2)
        # with h1 = z_k−z_{k-1}, h2 = z_{k-1}−z_{k-2}, and d_up the slope of the
        # segment one level up.  Exact for linear AND quadratic ρ(z); removes the
        # O(Δz) curvature bias of Δρ_top under a pressure-dependent EOS.  This
        # delta-form is well-conditioned (it adds a SMALL curvature term to the
        # already-correct one-sided slope instead of cancelling large ρ≈1027
        # values) and → Δρ_top exactly when the curvature vanishes.  Falls back
        # to Δρ_top where the cell two levels up is inactive (shallow columns).
        rho_above2 = jnp.concatenate([rho[..., :2], rho[..., :-2]], axis=-1)
        z_above2 = jnp.concatenate([z[..., :2], z[..., :-2]], axis=-1)
        active_above2 = jnp.concatenate(
            [jnp.zeros_like(active_f[..., :2]), active_f[..., :-2]], axis=-1,
        )
        has_top2 = has_top & (active_above2 > 0.5)
        h1 = z - z_above            # z_k - z_{k-1} > 0
        h2 = z_above - z_above2     # z_{k-1} - z_{k-2} > 0
        safe_h2 = jnp.where(has_top2, h2, 1.0)
        d_up = jnp.where(has_top2, (rho_above - rho_above2) / safe_h2, 0.0)
        safe_h12 = jnp.where(has_top2, h1 + h2, 1.0)
        curv = jnp.where(has_top2, (delta_top - d_up) * h1 / safe_h12, 0.0)
        bottom_slope = delta_top + curv
    else:
        bottom_slope = delta_top

    sigma = jnp.where(
        has_top & has_bot, sigma_interior,
        jnp.where(has_top, bottom_slope,
                  jnp.where(has_bot, delta_bot, 0.0)),
    )
    return jnp.where(active_f > 0.5, sigma, 0.0)


def compute_pressure_at_target_smc03(
    rho_per_cell: jnp.ndarray,
    h_partial: jnp.ndarray,
    z_centroid: jnp.ndarray,
    sigma: jnp.ndarray,
    z_target: jnp.ndarray,
    g: float,
) -> jnp.ndarray:
    """Per-column pressure at arbitrary target depths, evaluated from
    the harmonic-slope piecewise-linear ρ(z) reconstruction.

    Sign convention: all depths are **positive downward** [m].

    Algorithm:

    1. Cell-top interface depths and pressures by cumulative sum:

       ``z_top_0   = 0,                  P_top_0   = 0``
       ``z_top_k   = z_top_{k-1} + h_{k-1}``
       ``P_top_k   = P_top_{k-1} + g · h_{k-1} · ρ_{k-1}``

       (Cell-mean integral of the linear deviation ``σ_k · (z' − z_c)``
       across a full cell vanishes because ``z_c`` is the geometric
       centroid — so ``P_top_{k+1} − P_top_k = g · h_k · ρ_k`` exactly.)

    2. For each target ``z_t`` find its enclosing cell ``k_t`` such
       that ``z_top_{k_t} ≤ z_t ≤ z_top_{k_t}+h_{k_t}``.  Within that
       cell the analytic linear-deviation integral gives

       ``P(z_t) = P_top_{k_t}
                  + g · (z_t − z_top_{k_t})
                      · [ρ_{k_t}
                         + 0.5 · σ_{k_t}
                              · (z_t + z_top_{k_t} − 2 · z_c_{k_t})]``

    Parameters
    ----------
    rho_per_cell : array, shape (..., nlev)
        Cell-mean density [kg/m³].
    h_partial : array, shape (..., nlev)
        Per-cell layer thickness [m].  Inactive cells (below the
        partial seafloor) have ``h = 0`` and contribute nothing to the
        integral.
    z_centroid : array, shape (..., nlev)
        Per-cell centroid depth [m, positive downward].  Inactive
        cells inherit the seafloor depth from above (``h = 0`` cells
        have ``z_centroid`` at the seafloor; inert).
    sigma : array, shape (..., nlev)
        Per-cell density slope [kg/m⁴] from
        ``reconstruct_harmonic_slopes``.  Inactive cells: 0.
    z_target : array, shape (..., n_targets)
        Target depths [m, positive downward].  Targets outside the
        column ``[0, sum h_partial]`` are clamped — the resulting
        pressure equals zero (above surface) or the seafloor pressure
        (below).  Phase 3 face-mask logic should keep that branch
        from materially affecting answers, but the clamp ensures
        finite output and stable AD.
    g : float
        Gravitational acceleration [m/s²].

    Returns
    -------
    P : array, shape (..., n_targets)
        Hydrostatic pressure [Pa] at each target depth.
    """
    # 1. Cell-top depths and pressures (cumulative).
    z_bot_per_cell = jnp.cumsum(h_partial, axis=-1)
    z_top_per_cell = z_bot_per_cell - h_partial

    cell_dP = g * h_partial * rho_per_cell
    P_bot_per_cell = jnp.cumsum(cell_dP, axis=-1)
    P_top_per_cell = P_bot_per_cell - cell_dP

    # 2. Clamp z_target to the column's valid range.  Targets above the
    # surface saturate to z=0 (P=0); targets below the column-bottom
    # saturate to the seafloor depth (P = column-integrated weight).
    z_seafloor = z_bot_per_cell[..., -1:]                  # (..., 1)
    z_t_clamped = jnp.clip(z_target, min=0.0, max=z_seafloor)

    # 3. Find the enclosing cell per target.  z_bot is nondecreasing down a
    # column (thicknesses are >= 0), so the first cell whose BOTTOM reaches
    # the target is the enclosing one, and a sorted search finds it without
    # ever forming the (..., nlev, n_t) comparison table the obvious
    # broadcast builds.  That table is what made this scheme unusable on a
    # fine mesh: on the ico7 Voronoi grid it is 491520 x 75 x 75 entries,
    # about 22 GB of float temporaries per call, so the step asked for a
    # single 33 GiB buffer and no GPU on the cluster could run it.
    #
    # side="left" reproduces the previous first-True argmax exactly,
    # including the interface tie: a target sitting exactly on z_bot_{k-1}
    # belongs to cell k-1 under both rules.
    _nlev = z_bot_per_cell.shape[-1]
    k_t = jax.vmap(
        lambda bot, tgt: jnp.searchsorted(bot, tgt, side="left")
    )(
        z_bot_per_cell.reshape(-1, _nlev),
        z_t_clamped.reshape(-1, z_t_clamped.shape[-1]),
    ).reshape(z_t_clamped.shape)                            # (..., n_t)

    # 4. Gather per-cell quantities at k_t and evaluate the in-cell integral.
    rho_kt = jnp.take_along_axis(rho_per_cell, k_t, axis=-1)
    sigma_kt = jnp.take_along_axis(sigma, k_t, axis=-1)
    z_top_kt = jnp.take_along_axis(z_top_per_cell, k_t, axis=-1)
    z_c_kt = jnp.take_along_axis(z_centroid, k_t, axis=-1)
    P_top_kt = jnp.take_along_axis(P_top_per_cell, k_t, axis=-1)

    dz = z_t_clamped - z_top_kt
    rho_eff = rho_kt + 0.5 * sigma_kt * (z_t_clamped + z_top_kt - 2.0 * z_c_kt)
    return P_top_kt + g * dz * rho_eff
