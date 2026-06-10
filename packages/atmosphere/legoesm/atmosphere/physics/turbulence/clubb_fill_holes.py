"""CLUBB mass-conserving vertical hole-filling (``fill_holes_type = 2``).

Faithful port of ``fill_holes.F90:fill_holes_vertical`` for the CAM-default
configuration ``clubb_fill_holes_type = 2`` (sliding-window fill with a global
fallback). Hole-filling restores positive-definiteness of a clipped field (a
variance, or a positive scalar) by redistributing mass from levels above the
``threshold`` into the holes, conserving the density-weighted column integral
``sum(rho_ds * dz * field)``.

The ``fill_holes_type = 1`` global path is retained because it is the fallback
invoked inside the type-2 path when holes survive the sliding window; the other
types are out of the CAM-default tree and not ported.

All routines are pure, JIT-safe, and differentiable (the only control flow is
``lax.fori_loop`` / ``lax.cond`` over static bounds and ``jnp.where`` masks).
``num_hf_draw_points = 2`` and the divide-by-zero guard ``eps = 1e-10`` are the
CLUBB ``constants_clubb`` values; ``eps`` is a numerical safety floor.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_NUM_HF_DRAW = 2       # num_hf_draw_points (constants_clubb): sliding-window half-width
_EPS = 1.0e-10         # constants_clubb eps = max(1e-10, machine eps): divide-by-zero guard


def fill_holes_global(field, rho_dz, threshold, lower_k, upper_k):
    """Mass-conserving global hole-fill over ``[lower_k, upper_k]`` (``fill_holes_global``).

    Redistributes mass across the whole column-interval so every level reaches
    at least ``threshold`` while preserving ``sum(rho_dz * field)``. ``field``,
    ``rho_dz`` are ``(ngrdcol, nz)``; ``lower_k``/``upper_k`` are static 0-based
    inclusive bounds. Returns the filled field (unchanged in columns with no
    hole).
    """
    nz = field.shape[1]
    k_idx = jnp.arange(nz)[None, :]
    mask = (k_idx >= lower_k) & (k_idx <= upper_k)

    rho_dz_m = jnp.where(mask, rho_dz, 0.0)
    denom = jnp.sum(rho_dz_m, axis=1, keepdims=True)
    field_avg = jnp.sum(rho_dz_m * field, axis=1, keepdims=True) / denom

    field_clipped = jnp.where(
        field_avg >= threshold,
        jnp.maximum(threshold, field),
        jnp.minimum(threshold, field),
    )
    field_clipped_avg = jnp.sum(rho_dz_m * field_clipped, axis=1, keepdims=True) / denom

    safe = (jnp.abs(field_clipped_avg - threshold)
            > jnp.abs(field_clipped_avg + threshold) * _EPS / 2.0)
    mass_frac = jnp.where(
        safe,
        (field_avg - threshold) / jnp.where(safe, field_clipped_avg - threshold, 1.0),
        1.0,
    )
    field_new = jnp.where(mask,
                          threshold + mass_frac * (field_clipped - threshold),
                          field)

    any_hole = jnp.any(jnp.where(mask, field < threshold, False),
                       axis=1, keepdims=True)
    return jnp.where(any_hole, field_new, field)


def fill_holes_sliding_window(field, rho_dz, threshold, lower_k, upper_k,
                              num_draw=_NUM_HF_DRAW):
    """Sliding-window fill with global fallback (``fill_holes_type = 2``).

    Sweeps a window of width ``2*num_draw + 1`` over the interior, locally
    redistributing mass to fill holes; if any hole survives the sweep the
    mass-conserving :func:`fill_holes_global` runs over the full interval. The
    window length is static (compile-time) so the ``fori_loop`` body has a fixed
    ``dynamic_slice`` shape. ``lower_k``/``upper_k`` are static 0-based inclusive
    bounds.
    """
    wlen = 2 * num_draw + 1

    def body(k, field_carry):
        start = k - num_draw
        field_win = jax.lax.dynamic_slice(
            field_carry, (0, start), (field_carry.shape[0], wlen))
        rho_dz_win = jax.lax.dynamic_slice(
            rho_dz, (0, start), (rho_dz.shape[0], wlen))

        denom = jnp.sum(rho_dz_win, axis=1, keepdims=True)
        field_avg = jnp.sum(rho_dz_win * field_win, axis=1, keepdims=True) / denom
        any_hole = jnp.any(field_win < threshold, axis=1, keepdims=True)

        field_clipped = jnp.where(
            field_avg >= threshold,
            jnp.maximum(threshold, field_win),
            jnp.minimum(threshold, field_win),
        )
        field_clipped_avg = jnp.sum(rho_dz_win * field_clipped, axis=1, keepdims=True) / denom

        safe = (jnp.abs(field_clipped_avg - threshold)
                > jnp.abs(field_clipped_avg + threshold) * _EPS / 2.0)
        mass_frac = jnp.where(
            safe,
            (field_avg - threshold) / jnp.where(safe, field_clipped_avg - threshold, 1.0),
            1.0,
        )
        field_win_new = threshold + mass_frac * (field_clipped - threshold)
        field_win_out = jnp.where(any_hole, field_win_new, field_win)
        return jax.lax.dynamic_update_slice(field_carry, field_win_out, (0, start))

    start_k = lower_k + num_draw
    end_k = upper_k - num_draw + 1
    field_sw = jax.lax.fori_loop(start_k, end_k, body, field)

    return jax.lax.cond(
        jnp.any(field_sw < threshold),
        lambda f: fill_holes_global(f, rho_dz, threshold, lower_k, upper_k),
        lambda f: f,
        field_sw,
    )


def fill_holes_vertical(field, rho_ds, dz, threshold, lower_k, upper_k,
                        fill_holes_type, grid_dir_indx=1):
    """Mass-conserving vertical hole-fill (``fill_holes_vertical_api``).

    Dispatches on the static ``fill_holes_type``: 1 = global, 2 = sliding-window
    + global fallback (CAM default). ``field``/``rho_ds``/``dz`` are
    ``(ngrdcol, nz)``; ``lower_k``/``upper_k`` are static 0-based inclusive
    bounds. Returns a filled copy (input not mutated). Unknown types raise
    (no silent default).

    JIT contract: ``fill_holes_type``, ``lower_k``, ``upper_k`` and
    ``grid_dir_indx`` are **compile-time static** (they drive Python branching
    and the window/slice shapes). In normal use they come from the static
    ``CLUBBFlags``/grid config and are closed over by the enclosing ``jax.jit``;
    if this function is jitted directly they must be passed via
    ``static_argnums=(4, 5, 6, 7)`` (or the matching ``static_argnames``). The
    array inputs (``field``/``rho_ds``/``dz``) and ``threshold`` are traced and
    differentiable. ``grid_dir_indx`` is accepted for reference-signature parity
    but currently unused (the CAM-default ascending grid is grid_dir = +1).
    """
    rho_dz = rho_ds * dz
    if fill_holes_type == 1:
        return fill_holes_global(field, rho_dz, threshold, lower_k, upper_k)
    elif fill_holes_type == 2:
        return fill_holes_sliding_window(field, rho_dz, threshold, lower_k, upper_k)
    raise ValueError(f"fill_holes_type={fill_holes_type} not supported "
                     "(CAM-default tree implements 1 and 2)")


_F64_EPS = jnp.finfo(jnp.float64).eps


def fill_holes_wp2_from_horz_tke(wp2, up2, vp2, threshold, lower_k, upper_k):
    """TKE-conserving wp2 hole-fill from the horizontal variances (CAM default).

    Faithful port of ``fill_holes.F90:fill_holes_wp2_from_horz_tke``
    (``l_wp2_fill_holes_tke = .true.``): where ``wp2 < threshold`` and there is
    available TKE in ``up2``/``vp2`` (``> threshold``), borrow from up2/vp2 to
    fill the wp2 hole, conserving total ``wp2 + up2 + vp2``. If the available
    TKE is insufficient (case 1) wp2 takes all of it (up2/vp2 floored to
    ``threshold``); otherwise (case 2) the deficit is drawn proportionally, with
    one-sided fallbacks when a component has no surplus. Only levels in the
    static ``[lower_k, upper_k]`` range are modified. ``wp2``/``up2``/``vp2`` are
    ``(ncol, nzm)``. Returns ``(wp2, up2, vp2)``.
    """
    nzm = wp2.shape[1]
    k_idx = jnp.arange(nzm)[None, :]
    in_range = (k_idx >= lower_k) & (k_idx <= upper_k)

    do_fill = in_range & (wp2 < threshold) & ((up2 > threshold) | (vp2 > threshold))
    missing = threshold - wp2
    up2_avail = jnp.maximum(up2 - threshold, 0.0)
    vp2_avail = jnp.maximum(vp2 - threshold, 0.0)
    total_avail = up2_avail + vp2_avail

    case1 = do_fill & (missing >= total_avail)        # not enough TKE
    wp2_c1 = wp2 + total_avail
    up2_c1 = jnp.minimum(up2, threshold)
    vp2_c1 = jnp.minimum(vp2, threshold)

    case2 = do_fill & (missing < total_avail)          # enough TKE
    eps_thr = _F64_EPS * 1000.0
    case2a = case2 & (jnp.abs(up2_avail) < eps_thr)    # take all from vp2
    case2b = case2 & (~case2a) & (jnp.abs(vp2_avail) < eps_thr)  # take all from up2
    ratio = jnp.where(total_avail > 0.0, missing / jnp.where(total_avail > 0.0, total_avail, 1.0), 0.0)

    up2_2c = threshold + up2_avail * (1.0 - ratio)
    vp2_2c = threshold + vp2_avail * (1.0 - ratio)
    up2_c2 = jnp.where(case2a, up2, jnp.where(case2b, up2 - missing, up2_2c))
    vp2_c2 = jnp.where(case2a, vp2 - missing, jnp.where(case2b, vp2, vp2_2c))

    wp2_new = jnp.where(case1, wp2_c1, jnp.where(case2, threshold, wp2))
    up2_new = jnp.where(case1, up2_c1, jnp.where(case2, up2_c2, up2))
    vp2_new = jnp.where(case1, vp2_c1, jnp.where(case2, vp2_c2, vp2))
    return wp2_new, up2_new, vp2_new


__all__ = ["fill_holes_global", "fill_holes_sliding_window", "fill_holes_vertical",
           "fill_holes_wp2_from_horz_tke"]
