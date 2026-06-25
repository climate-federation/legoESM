"""PPM-based conservative vertical remapping.

**NOT YET INTEGRATED** — this module is staged for future use in a
vertically Lagrangian time-stepping mode.  It is not imported by any
production code path.  Wire it into the PE model's step() method
before moving it back to core/.

Implements the vertically Lagrangian approach of Lin (2004):
1. Advance dynamics without vertical advection (layers deform).
2. After the step, remap prognostic fields back to reference levels
   using PPM reconstruction, conserving the integral ∫ q dp.

The remapping is 1D along the vertical axis at each column, using the
same PPM edge reconstruction and Colella-Woodward limiter shared by
the horizontal transport operators.

References
----------
- Lin (2004): A "Vertically Lagrangian" Finite-Volume Dynamical Core.
- Colella & Woodward (1984): The Piecewise Parabolic Method (PPM).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ==============================================================================
# 1D PPM vertical reconstruction
# ==============================================================================

def _ppm_edge_values_vertical(q: jnp.ndarray) -> jnp.ndarray:
    """4th-order edge values along the vertical (last) axis.

    Produces nlev+1 interface values from nlev cell averages:
    - Edge 0: top boundary (zero-gradient, = q[0])
    - Edge 1: 2nd-order, 0.5*(q[0]+q[1])
    - Edges 2..nlev-2: 4th-order interior (nlev-3 values)
    - Edge nlev-1: 2nd-order, 0.5*(q[-2]+q[-1])
    - Edge nlev: bottom boundary (zero-gradient, = q[-1])

    Parameters
    ----------
    q : (..., nlev)  Cell averages.

    Returns
    -------
    q_hat : (..., nlev+1)  Edge values at level interfaces.
    """
    nlev = q.shape[-1]

    if nlev >= 4:
        # 4th-order interior edges (nlev-3 values, between cells 1..nlev-2)
        q_hat_inner = (
            (7.0 / 12.0) * (q[..., 1:-2] + q[..., 2:-1])
            - (1.0 / 12.0) * (q[..., :-3] + q[..., 3:])
        )  # (..., nlev-3)

        # 2nd-order boundary edges
        q_hat_1 = 0.5 * (q[..., 0:1] + q[..., 1:2])       # edge 1
        q_hat_nm1 = 0.5 * (q[..., -2:-1] + q[..., -1:])    # edge nlev-1

        # Zero-gradient boundary values
        q_hat_top = q[..., 0:1]       # edge 0 (top)
        q_hat_bot = q[..., -1:]        # edge nlev (bottom)

        # Assemble: 1 + 1 + (nlev-3) + 1 + 1 = nlev+1
        q_hat = jnp.concatenate([
            q_hat_top, q_hat_1, q_hat_inner, q_hat_nm1, q_hat_bot,
        ], axis=-1)
    else:
        # For very few levels (< 4), use 2nd-order everywhere
        interior = [
            0.5 * (q[..., k:k+1] + q[..., k+1:k+2])
            for k in range(nlev - 1)
        ]
        q_hat = jnp.concatenate(
            [q[..., 0:1]] + interior + [q[..., -1:]],
            axis=-1,
        )

    # Monotonicity clamp: each edge between its two flanking cell values
    q_lo = jnp.minimum(
        jnp.concatenate([q[..., 0:1], q], axis=-1),
        jnp.concatenate([q, q[..., -1:]], axis=-1),
    )
    q_hi = jnp.maximum(
        jnp.concatenate([q[..., 0:1], q], axis=-1),
        jnp.concatenate([q, q[..., -1:]], axis=-1),
    )
    q_hat = jnp.clip(q_hat, q_lo, q_hi)

    return q_hat


def _ppm_limit_vertical(q_bar, q_L, q_R):
    """Colella-Woodward limiter along vertical axis.

    Parameters
    ----------
    q_bar, q_L, q_R : (..., nlev)

    Returns
    -------
    q_L_lim, q_R_lim : (..., nlev)
    """
    is_extremum = (q_R - q_bar) * (q_bar - q_L) <= 0
    dm = q_R - q_L
    d6 = 6.0 * (q_bar - 0.5 * (q_L + q_R))

    over_L = dm * d6 > dm**2
    q_L_lim = jnp.where(over_L, 3.0 * q_bar - 2.0 * q_R, q_L)

    over_R = dm * d6 < -(dm**2)
    q_R_lim = jnp.where(over_R, 3.0 * q_bar - 2.0 * q_L, q_R)

    q_L_lim = jnp.where(is_extremum, q_bar, q_L_lim)
    q_R_lim = jnp.where(is_extremum, q_bar, q_R_lim)

    return q_L_lim, q_R_lim


# ==============================================================================
# Conservative vertical remapping
# ==============================================================================

def vertical_remap_ppm(
    q: jnp.ndarray,
    dp_old: jnp.ndarray,
    dp_new: jnp.ndarray,
    limiter: bool = True,
) -> jnp.ndarray:
    """Conservative PPM vertical remapping.

    Remaps a field from deformed (Lagrangian) layers back to reference
    layers, conserving the mass-weighted integral: ∫ q dp.

    The algorithm sweeps from the top boundary downward, accumulating
    the PPM sub-cell integral within each new layer.

    Parameters
    ----------
    q : jax.Array, shape (..., nlev)
        Field values on the old (deformed) grid.
    dp_old : jax.Array, shape (..., nlev)
        Layer pressure thicknesses on the old grid [Pa].
    dp_new : jax.Array, shape (..., nlev_new)
        Layer pressure thicknesses on the new (target) grid [Pa]. ``nlev_new``
        may differ from the old layer count; the mass-weighted integral ∫ q dp
        is conserved as long as Σ dp_new == Σ dp_old per column.
    limiter : bool
        Apply Colella-Woodward monotonicity limiter.

    Returns
    -------
    q_new : jax.Array, shape (..., nlev_new)
        Field values on the new grid, conserving ∫ q dp.
    """
    # PPM reconstruction of q within each old cell
    q_hat = _ppm_edge_values_vertical(q)  # (..., nlev+1)
    q_L = q_hat[..., :-1]   # top edge of each cell
    q_R = q_hat[..., 1:]    # bottom edge of each cell

    if limiter:
        q_L, q_R = _ppm_limit_vertical(q, q_L, q_R)

    # PPM parabola coefficients: q(xi) = q_L + xi*(q_R - q_L + d6*(1-xi))
    # where xi ∈ [0,1] is fractional position within the cell,
    # d6 = 6*(q_bar - 0.5*(q_L + q_R))
    d6 = 6.0 * (q - 0.5 * (q_L + q_R))

    # Integral of q(xi) from xi=a to xi=b within one cell:
    # ∫_a^b q(xi) dxi = (b-a)*q_L + (b²-a²)/2*(q_R-q_L+d6) - (b³-a³)/3*d6
    # Multiply by dp_old[k] to get mass-weighted integral.

    nlev = q.shape[-1]
    nlev_new = dp_new.shape[-1]  # target grid may have a different layer count
    batch_shape = q.shape[:-1]

    # Compute pressure interfaces on old and new grids
    p_old_iface = jnp.concatenate([
        jnp.zeros((*batch_shape, 1), dtype=dp_old.dtype),
        jnp.cumsum(dp_old, axis=-1),
    ], axis=-1)  # (..., nlev+1)

    p_new_iface = jnp.concatenate([
        jnp.zeros((*batch_shape, 1), dtype=dp_new.dtype),
        jnp.cumsum(dp_new, axis=-1),
    ], axis=-1)  # (..., nlev+1)

    # For each new cell j, find the integral of q*dp from p_new[j] to p_new[j+1]
    # This integral may span multiple old cells.
    # We use a vectorized approach: for each old cell k, compute its
    # contribution to each new cell j via overlap.

    def _ppm_integral(a, b, q_L_k, q_R_k, d6_k, dp_k):
        """PPM sub-cell integral: ∫_a^b q(xi) dxi * dp_k."""
        return dp_k * (
            (b - a) * q_L_k
            + (b**2 - a**2) / 2.0 * (q_R_k - q_L_k + d6_k)
            - (b**3 - a**3) / 3.0 * d6_k
        )

    def _remap_column(q_L_col, q_R_col, d6_col, dp_old_col, p_old_col, p_new_col, dp_new_col):
        """Remap a single column in O(nlev) with a monotone sweep.

        Uses a single lax.scan over new cells, carrying a pointer into
        the old grid.  Each old cell is visited at most twice (partial
        contributions to two adjacent new cells), giving O(nlev) total.
        """
        nlev_new = dp_new_col.shape[0]
        nlev_old = dp_old_col.shape[0]  # OLD-grid length: the pointer k indexes
        #                                 dp_old_col/q_L_col/... so it must clamp
        #                                 to nlev_old-1, NOT nlev_new-1 (else a
        #                                 grid with nlev_new != nlev_old loses
        #                                 the deepest source layers / goes OOB).

        def _scan_fn(carry, j):
            # carry = (k, frac_consumed)
            # k: current old cell index
            # frac_consumed: fraction of old cell k already used [0, 1]
            k, frac_consumed = carry

            dp_needed = dp_new_col[j]
            dp_k = dp_old_col[k]

            # Available pressure in current old cell
            p_avail = dp_k * (1.0 - frac_consumed)
            p_take = jnp.minimum(p_avail, dp_needed)

            # PPM integral for the first contributing old cell
            a = frac_consumed
            b = frac_consumed + p_take / jnp.maximum(dp_k, 1e-30)
            accum = _ppm_integral(a, b, q_L_col[k], q_R_col[k], d6_col[k], dp_k)

            # Update remaining and advance old-cell pointer if exhausted.
            # When the LAST old cell is exhausted we must NOT reset frac to 0
            # (that would re-consume the deepest source layer on the next new
            # cell, fabricating mass for a Σdp_new > Σdp_old column); keep
            # frac=1 so no further source is drawn and the excess target space
            # stays empty (still conservative: all source is distributed once).
            remaining = dp_needed - p_take
            exhausted = (dp_k * (1.0 - b)) < 1e-20
            at_last = k >= jnp.int32(nlev_old - 1)
            advance = exhausted & jnp.logical_not(at_last)
            k = jnp.where(advance, k + 1, k)
            frac_consumed = jnp.where(advance, 0.0, jnp.where(exhausted, 1.0, b))

            # Continue consuming additional old cells while mass remains AND the
            # source is not exhausted (the deepest old cell fully consumed) —
            # the source-availability guard prevents non-termination / re-draw
            # when a new cell needs more thickness than the old grid provides.
            def _while_cond(state):
                _, kk, fc, rem, _ = state
                source_left = (fc < 1.0) | (kk < jnp.int32(nlev_old - 1))
                return (rem > 1e-20) & source_left

            def _while_body(state):
                acc, kk, fc, rem, _ = state
                dp_kk = dp_old_col[kk]
                p_avail2 = dp_kk * (1.0 - fc)
                p_take2 = jnp.minimum(p_avail2, rem)
                a2 = fc
                b2 = fc + p_take2 / jnp.maximum(dp_kk, 1e-30)
                acc = acc + _ppm_integral(
                    a2, b2, q_L_col[kk], q_R_col[kk], d6_col[kk], dp_kk,
                )
                rem = rem - p_take2
                ex = (dp_kk * (1.0 - b2)) < 1e-20
                at_last2 = kk >= jnp.int32(nlev_old - 1)
                advance2 = ex & jnp.logical_not(at_last2)
                kk = jnp.where(advance2, kk + 1, kk)
                fc = jnp.where(advance2, 0.0, jnp.where(ex, 1.0, b2))
                return (acc, kk, fc, rem, True)

            accum, k, frac_consumed, _, _ = jax.lax.while_loop(
                _while_cond, _while_body,
                (accum, k, frac_consumed, remaining, True),
            )

            q_j = accum / jnp.maximum(dp_needed, 1e-30)
            return (k, frac_consumed), q_j

        (_, _), q_new_col = jax.lax.scan(
            _scan_fn, (jnp.int32(0), 0.0), jnp.arange(nlev_new),
        )
        return q_new_col

    # Vectorize over all columns
    # Flatten batch dimensions
    flat_shape = (-1, nlev)
    q_L_flat = q_L.reshape(flat_shape)
    q_R_flat = q_R.reshape(flat_shape)
    d6_flat = d6.reshape(flat_shape)
    dp_old_flat = dp_old.reshape(flat_shape)
    dp_new_flat = dp_new.reshape(-1, nlev_new)
    p_old_flat = p_old_iface.reshape(-1, nlev + 1)
    p_new_flat = p_new_iface.reshape(-1, nlev_new + 1)

    q_new_flat = jax.vmap(_remap_column)(
        q_L_flat, q_R_flat, d6_flat,
        dp_old_flat, p_old_flat, p_new_flat, dp_new_flat,
    )

    return q_new_flat.reshape(*batch_shape, nlev_new)
