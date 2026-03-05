"""Finite-volume transport operators on the cubed-sphere grid.

Provides upwind finite-volume operators using MUSCL, PPM, or WENO5
reconstruction and Rusanov approximate Riemann solver.

Key functions
-------------
fv_flux_divergence : -div(h*v) using Rusanov flux (for mass equations)
fv_scalar_advection : -div(q*v) using upwind FV (for tracer equations)

References
----------
- Toro (2009): Riemann Solvers and Numerical Methods for Fluid Dynamics
- LeVeque (2002): Finite Volume Methods for Hyperbolic Problems
- Colella & Woodward (1984): The Piecewise Parabolic Method (PPM)
- Jiang & Shu (1996): Efficient Implementation of Weighted ENO Schemes
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import pad_halo, pad_halo_vector


# ==============================================================================
# Slope limiters
# ==============================================================================

def minmod(a, b):
    """Minmod limiter: most dissipative TVD limiter."""
    return jnp.where(a * b > 0,
                     jnp.where(jnp.abs(a) < jnp.abs(b), a, b),
                     0.0)


def mc_limiter(a, b):
    """Monotonized central limiter: less dissipative, still TVD."""
    c = 0.5 * (a + b)
    return minmod(minmod(2.0 * a, 2.0 * b), c)


def _get_limiter(name):
    """Get limiter function by name."""
    if name == "mc":
        return mc_limiter
    elif name == "minmod":
        return minmod
    else:
        raise ValueError(f"Unknown limiter: {name!r}")


# ==============================================================================
# Reconstruction halo requirements and dispatcher
# ==============================================================================

_RECON_HALO = {"mc": 1, "minmod": 1, "ppm": 2, "weno5": 2}


def _get_halo_width(limiter):
    """Return required halo width for the given reconstruction scheme."""
    try:
        return _RECON_HALO[limiter]
    except KeyError:
        raise ValueError(f"Unknown limiter: {limiter!r}")


def _reconstruct_x(q_pad, limiter):
    """Dispatch x-direction reconstruction to the appropriate scheme."""
    if limiter in ("mc", "minmod"):
        return _reconstruct_edges_x(q_pad, _get_limiter(limiter))
    elif limiter == "ppm":
        return _reconstruct_edges_x_ppm(q_pad)
    elif limiter == "weno5":
        return _reconstruct_edges_x_weno5(q_pad)
    else:
        raise ValueError(f"Unknown limiter: {limiter!r}")


def _reconstruct_y(q_pad, limiter):
    """Dispatch y-direction reconstruction to the appropriate scheme."""
    if limiter in ("mc", "minmod"):
        return _reconstruct_edges_y(q_pad, _get_limiter(limiter))
    elif limiter == "ppm":
        return _reconstruct_edges_y_ppm(q_pad)
    elif limiter == "weno5":
        return _reconstruct_edges_y_weno5(q_pad)
    else:
        raise ValueError(f"Unknown limiter: {limiter!r}")


# ==============================================================================
# MUSCL reconstruction to cell edges (halo=1)
# ==============================================================================

def _reconstruct_edges_x(q_pad, limiter_fn):
    """Reconstruct cell-center values to x-direction edge midpoints.

    Parameters
    ----------
    q_pad : jax.Array, shape (6, n+2, n+2)
        Padded scalar field (with halo data from neighbors).
    limiter_fn : callable
        Slope limiter function (minmod or mc_limiter).

    Returns
    -------
    q_left, q_right : each shape (6, n+1, n)
        Left and right states at x-direction edges.
        Edge index i corresponds to the face between padded cells i and i+1.
    """
    # Strip y-halos; keep x-padding
    q_y = q_pad[:, :, 1:-1]  # (6, n+2, n)

    # Limited slopes at interior cells (padded indices 1..n)
    dq_fwd = q_y[:, 2:, :] - q_y[:, 1:-1, :]    # (6, n, n)
    dq_bwd = q_y[:, 1:-1, :] - q_y[:, :-2, :]    # (6, n, n)
    slopes_int = limiter_fn(dq_fwd, dq_bwd)       # (6, n, n)

    # Zero slope at halo cells (first-order at face boundaries)
    zero = jnp.zeros_like(slopes_int[:, :1, :])    # (6, 1, n)
    slopes = jnp.concatenate([zero, slopes_int, zero], axis=1)  # (6, n+2, n)

    # Edge states: edge i is between padded cells i and i+1
    q_left  = q_y[:, :-1, :] + 0.5 * slopes[:, :-1, :]   # (6, n+1, n)
    q_right = q_y[:, 1:, :]  - 0.5 * slopes[:, 1:, :]    # (6, n+1, n)
    return q_left, q_right


def _reconstruct_edges_y(q_pad, limiter_fn):
    """Reconstruct cell-center values to y-direction edge midpoints.

    Parameters
    ----------
    q_pad : jax.Array, shape (6, n+2, n+2)
        Padded scalar field.
    limiter_fn : callable
        Slope limiter function.

    Returns
    -------
    q_left, q_right : each shape (6, n, n+1)
        Left and right states at y-direction edges.
        Edge index j corresponds to the face between padded cells j and j+1.
    """
    # Strip x-halos; keep y-padding
    q_x = q_pad[:, 1:-1, :]  # (6, n, n+2)

    # Limited slopes at interior cells (padded indices 1..n)
    dq_fwd = q_x[:, :, 2:] - q_x[:, :, 1:-1]     # (6, n, n)
    dq_bwd = q_x[:, :, 1:-1] - q_x[:, :, :-2]     # (6, n, n)
    slopes_int = limiter_fn(dq_fwd, dq_bwd)        # (6, n, n)

    # Zero slope at halo cells
    zero = jnp.zeros_like(slopes_int[:, :, :1])     # (6, n, 1)
    slopes = jnp.concatenate([zero, slopes_int, zero], axis=2)  # (6, n, n+2)

    # Edge states
    q_left  = q_x[:, :, :-1] + 0.5 * slopes[:, :, :-1]   # (6, n, n+1)
    q_right = q_x[:, :, 1:]  - 0.5 * slopes[:, :, 1:]    # (6, n, n+1)
    return q_left, q_right


# ==============================================================================
# PPM reconstruction (halo=2, Colella & Woodward 1984)
# ==============================================================================

def _ppm_1d(q):
    """PPM reconstruction along the last-but-one axis.

    Parameters
    ----------
    q : shape (..., M, K) where M = n+4 (halo=2 stripped of transverse halos)

    Returns
    -------
    q_left, q_right : each shape (..., n+1, K)
    """
    # 4th-order edge values at interior interfaces (n+1 values)
    q_hat_inner = ((7.0 / 12.0) * (q[..., 1:-2, :] + q[..., 2:-1, :])
                   - (1.0 / 12.0) * (q[..., :-3, :] + q[..., 3:, :]))

    # Boundary edges (2nd-order fallback for outermost interfaces)
    q_hat_lo = 0.5 * (q[..., 0:1, :] + q[..., 1:2, :])
    q_hat_hi = 0.5 * (q[..., -2:-1, :] + q[..., -1:, :])

    # All n+3 edge values at positions 0.5, 1.5, ..., (n+2).5
    q_hat = jnp.concatenate([q_hat_lo, q_hat_inner, q_hat_hi], axis=-2)

    # Monotonicity constraint: clamp each edge value between its two cells
    q_lo = jnp.minimum(q[..., :-1, :], q[..., 1:, :])
    q_hi = jnp.maximum(q[..., :-1, :], q[..., 1:, :])
    q_hat = jnp.clip(q_hat, q_lo, q_hi)

    # Parabola for cells 1..n+2 (n+2 cells)
    a_L = q_hat[..., :-1, :]     # left edge of each cell
    a_R = q_hat[..., 1:, :]      # right edge of each cell
    q_c = q[..., 1:-1, :]        # cell-center values

    # Colella-Woodward limiting
    is_extremum = (a_R - q_c) * (q_c - a_L) <= 0
    dm = a_R - a_L
    d6 = 6.0 * (q_c - 0.5 * (a_L + a_R))

    # Limit left edge when parabola overshoots on the left
    over_L = dm * d6 > dm ** 2
    a_L_lim = jnp.where(over_L, 3.0 * q_c - 2.0 * a_R, a_L)

    # Limit right edge when parabola overshoots on the right
    over_R = dm * d6 < -(dm ** 2)
    a_R_lim = jnp.where(over_R, 3.0 * q_c - 2.0 * a_L, a_R)

    # Flatten at local extrema (overrides overshoot corrections)
    a_L_lim = jnp.where(is_extremum, q_c, a_L_lim)
    a_R_lim = jnp.where(is_extremum, q_c, a_R_lim)

    # Extract states at n+1 interior edges
    # q_left[edge k] = right-edge value of cell to the left of edge k
    # q_right[edge k] = left-edge value of cell to the right of edge k
    q_left = a_R_lim[..., :-1, :]     # (n+1 values)
    q_right = a_L_lim[..., 1:, :]     # (n+1 values)

    return q_left, q_right


def _reconstruct_edges_x_ppm(q_pad_h2):
    """PPM reconstruction in x-direction.

    Parameters
    ----------
    q_pad_h2 : shape (6, n+4, n+4)

    Returns
    -------
    q_left, q_right : each shape (6, n+1, n)
    """
    q = q_pad_h2[:, :, 2:-2]   # (6, n+4, n) — strip y-halos
    return _ppm_1d(q)


def _reconstruct_edges_y_ppm(q_pad_h2):
    """PPM reconstruction in y-direction.

    Parameters
    ----------
    q_pad_h2 : shape (6, n+4, n+4)

    Returns
    -------
    q_left, q_right : each shape (6, n, n+1)
    """
    q = q_pad_h2[:, 2:-2, :]   # (6, n, n+4) — strip x-halos
    # Transpose last two axes to reuse _ppm_1d, then transpose back
    q_t = jnp.swapaxes(q, -2, -1)       # (6, n+4, n)
    qL_t, qR_t = _ppm_1d(q_t)           # each (6, n+1, n)
    return jnp.swapaxes(qL_t, -2, -1), jnp.swapaxes(qR_t, -2, -1)


# ==============================================================================
# WENO5 reconstruction (halo=2, Jiang & Shu 1996)
# ==============================================================================

def _weno5_1d(q):
    """WENO5 reconstruction along the last-but-one axis.

    Uses the full WENO5 stencil (6 points) for interior edges, and
    falls back to MC-limited MUSCL at the two boundary edges where
    the 6-point stencil extends beyond the halo=2 data.

    Parameters
    ----------
    q : shape (..., M, K) where M = n+4 (halo=2 on each side)

    Returns
    -------
    q_left, q_right : each shape (..., n+1, K)
    """
    # --- Interior WENO5: n-1 edges (full 6-point stencil available) ---
    # Edge k between cells (k+2) and (k+3) in q, for k = 0..n-2
    a = q[..., :-5, :]     # q_{j-2}: indices 0..n-2
    b = q[..., 1:-4, :]    # q_{j-1}: indices 1..n-1
    c = q[..., 2:-3, :]    # q_j:     indices 2..n
    d = q[..., 3:-2, :]    # q_{j+1}: indices 3..n+1
    e = q[..., 4:-1, :]    # q_{j+2}: indices 4..n+2
    f = q[..., 5:, :]      # q_{j+3}: indices 5..n+3

    eps = 1e-6

    # Left-biased (uses a, b, c, d, e) → q_L at interface c|d
    p0_L = (2.0 * a - 7.0 * b + 11.0 * c) / 6.0
    p1_L = (-b + 5.0 * c + 2.0 * d) / 6.0
    p2_L = (2.0 * c + 5.0 * d - e) / 6.0

    b0_L = (13.0 / 12.0) * (a - 2.0 * b + c) ** 2 + 0.25 * (a - 4.0 * b + 3.0 * c) ** 2
    b1_L = (13.0 / 12.0) * (b - 2.0 * c + d) ** 2 + 0.25 * (b - d) ** 2
    b2_L = (13.0 / 12.0) * (c - 2.0 * d + e) ** 2 + 0.25 * (3.0 * c - 4.0 * d + e) ** 2

    # WENO-Z weights (Borges et al. 2008): better near smooth extrema
    tau5_L = jnp.abs(b0_L - b2_L)
    w0_L = 0.1 * (1.0 + (tau5_L / (b0_L + eps)) ** 2)
    w1_L = 0.6 * (1.0 + (tau5_L / (b1_L + eps)) ** 2)
    w2_L = 0.3 * (1.0 + (tau5_L / (b2_L + eps)) ** 2)
    ws_L = w0_L + w1_L + w2_L

    q_left_int = (w0_L * p0_L + w1_L * p1_L + w2_L * p2_L) / ws_L

    # Right-biased (uses b, c, d, e, f) → q_R at interface c|d
    p0_R = (2.0 * f - 7.0 * e + 11.0 * d) / 6.0
    p1_R = (-e + 5.0 * d + 2.0 * c) / 6.0
    p2_R = (2.0 * d + 5.0 * c - b) / 6.0

    b0_R = (13.0 / 12.0) * (f - 2.0 * e + d) ** 2 + 0.25 * (f - 4.0 * e + 3.0 * d) ** 2
    b1_R = (13.0 / 12.0) * (e - 2.0 * d + c) ** 2 + 0.25 * (e - c) ** 2
    b2_R = (13.0 / 12.0) * (d - 2.0 * c + b) ** 2 + 0.25 * (3.0 * d - 4.0 * c + b) ** 2

    tau5_R = jnp.abs(b0_R - b2_R)
    w0_R = 0.1 * (1.0 + (tau5_R / (b0_R + eps)) ** 2)
    w1_R = 0.6 * (1.0 + (tau5_R / (b1_R + eps)) ** 2)
    w2_R = 0.3 * (1.0 + (tau5_R / (b2_R + eps)) ** 2)
    ws_R = w0_R + w1_R + w2_R

    q_right_int = (w0_R * p0_R + w1_R * p1_R + w2_R * p2_R) / ws_R

    # Monotonicity clamp: prevent oscillations at cube edges
    # Clip to range of the 3-cell neighborhood around each edge
    q_min_L = jnp.minimum(jnp.minimum(b, c), d)
    q_max_L = jnp.maximum(jnp.maximum(b, c), d)
    q_left_int = jnp.clip(q_left_int, q_min_L, q_max_L)

    q_min_R = jnp.minimum(jnp.minimum(c, d), e)
    q_max_R = jnp.maximum(jnp.maximum(c, d), e)
    q_right_int = jnp.clip(q_right_int, q_min_R, q_max_R)

    # --- Boundary edges: MC-limited MUSCL fallback ---
    # Left boundary: edge between cells 1 and 2 (halo|interior)
    slope_L0 = mc_limiter(q[..., 2:3, :] - q[..., 1:2, :],
                          q[..., 1:2, :] - q[..., 0:1, :])
    slope_R0 = mc_limiter(q[..., 3:4, :] - q[..., 2:3, :],
                          q[..., 2:3, :] - q[..., 1:2, :])
    q_left_lo = q[..., 1:2, :] + 0.5 * slope_L0
    q_right_lo = q[..., 2:3, :] - 0.5 * slope_R0

    # Right boundary: edge between cells n+1 and n+2 (interior|halo)
    slope_Ln = mc_limiter(q[..., -2:-1, :] - q[..., -3:-2, :],
                          q[..., -3:-2, :] - q[..., -4:-3, :])
    slope_Rn = mc_limiter(q[..., -1:, :] - q[..., -2:-1, :],
                          q[..., -2:-1, :] - q[..., -3:-2, :])
    q_left_hi = q[..., -3:-2, :] + 0.5 * slope_Ln
    q_right_hi = q[..., -2:-1, :] - 0.5 * slope_Rn

    # Concatenate: [left_boundary, interior, right_boundary]
    q_left = jnp.concatenate([q_left_lo, q_left_int, q_left_hi], axis=-2)
    q_right = jnp.concatenate([q_right_lo, q_right_int, q_right_hi], axis=-2)

    return q_left, q_right


def _reconstruct_edges_x_weno5(q_pad_h2):
    """WENO5 reconstruction in x-direction.

    Parameters
    ----------
    q_pad_h2 : shape (6, n+4, n+4)

    Returns
    -------
    q_left, q_right : each shape (6, n+1, n)
    """
    q = q_pad_h2[:, :, 2:-2]   # (6, n+4, n)
    return _weno5_1d(q)


def _reconstruct_edges_y_weno5(q_pad_h2):
    """WENO5 reconstruction in y-direction.

    Parameters
    ----------
    q_pad_h2 : shape (6, n+4, n+4)

    Returns
    -------
    q_left, q_right : each shape (6, n, n+1)
    """
    q = q_pad_h2[:, 2:-2, :]   # (6, n, n+4)
    q_t = jnp.swapaxes(q, -2, -1)       # (6, n+4, n)
    qL_t, qR_t = _weno5_1d(q_t)         # each (6, n+1, n)
    return jnp.swapaxes(qL_t, -2, -1), jnp.swapaxes(qR_t, -2, -1)


# ==============================================================================
# FV flux operators
# ==============================================================================

def _halo_fields(grid, halo):
    """Return (interp_offsets, cos_angle_padded, sin_angle_padded) for halo."""
    if halo == 1:
        return (grid.halo_interp_offsets,
                grid.cos_angle_padded, grid.sin_angle_padded)
    else:
        return (grid.halo_interp_offsets_h2,
                grid.cos_angle_padded_h2, grid.sin_angle_padded_h2)


def fv_flux_divergence(h, u, v, grid, g=9.80616, limiter="mc", h_s=None):
    """Compute -div(h*v) using finite volume with Rusanov flux.

    Uses MUSCL, PPM, or WENO5 reconstruction to obtain edge states and
    the Rusanov approximate Riemann solver for upwind dissipation.

    When *h_s* (topography) is provided, the scheme becomes **well-balanced**:
    it reconstructs the free surface eta = h + h_s at cell edges, then
    recovers h_L = eta_L - h_s_edge, h_R = eta_R - h_s_edge.

    Parameters
    ----------
    h : jax.Array, shape (6, n, n)
        Scalar field (fluid depth, surface pressure, density, ...).
    u, v : jax.Array, shape (6, n, n)
        Velocity components (grid-aligned).
    grid : CubedSphereGrid
    g : float
        Gravity for gravity-wave speed (set 0 for pure transport).
    limiter : str
        Reconstruction scheme: "mc", "minmod", "ppm", or "weno5".
    h_s : jax.Array or None, shape (6, n, n)
        Surface topography for well-balanced reconstruction.

    Returns
    -------
    jax.Array, shape (6, n, n)
        Mass tendency: -div(h*v).
    """
    halo = _get_halo_width(limiter)
    interp_off, cos_ap, sin_ap = _halo_fields(grid, halo)

    # Pad velocity with appropriate halo
    u_pad, v_pad = pad_halo_vector(
        u, v,
        grid.cos_angle, grid.sin_angle,
        cos_ap, sin_ap,
        interp_offsets=interp_off, halo=halo,
    )

    # Well-balanced: reconstruct eta = h + h_s
    if h_s is not None:
        eta = h + h_s
        eta_pad = pad_halo(eta, halo=halo, interp_offsets=interp_off)
        # Topography edges always use halo=1 (only need cell-pair average)
        hs_pad_h1 = pad_halo(h_s, interp_offsets=grid.halo_interp_offsets)
    else:
        eta_pad = pad_halo(h, halo=halo, interp_offsets=interp_off)

    # ---- X-direction Rusanov flux ----
    eta_L_x, eta_R_x = _reconstruct_x(eta_pad, limiter)   # (6, n+1, n)
    u_L_x, u_R_x = _reconstruct_x(u_pad, limiter)

    if h_s is not None:
        hs_x = hs_pad_h1[:, :, 1:-1]   # (6, n+2, n)
        hs_edge_x = 0.5 * (hs_x[:, :-1, :] + hs_x[:, 1:, :])   # (6, n+1, n)
        h_L_x = jnp.maximum(eta_L_x - hs_edge_x, 0.0)
        h_R_x = jnp.maximum(eta_R_x - hs_edge_x, 0.0)
    else:
        h_L_x = eta_L_x
        h_R_x = eta_R_x

    if g > 0:
        c_L_x = jnp.sqrt(jnp.maximum(g * h_L_x, 0.0))
        c_R_x = jnp.sqrt(jnp.maximum(g * h_R_x, 0.0))
        alpha_x = jnp.maximum(jnp.abs(u_L_x) + c_L_x, jnp.abs(u_R_x) + c_R_x)
    else:
        alpha_x = jnp.maximum(jnp.abs(u_L_x), jnp.abs(u_R_x))

    F_x = (0.5 * (h_L_x * u_L_x + h_R_x * u_R_x)
           - 0.5 * alpha_x * (h_R_x - h_L_x))

    # Edge metric (always halo=1)
    hy_edge = 0.5 * (grid.hy_ext[:, :-1, 1:-1] + grid.hy_ext[:, 1:, 1:-1])
    Phi_x = F_x * hy_edge

    # ---- Y-direction Rusanov flux ----
    eta_L_y, eta_R_y = _reconstruct_y(eta_pad, limiter)   # (6, n, n+1)
    v_L_y, v_R_y = _reconstruct_y(v_pad, limiter)

    if h_s is not None:
        hs_y = hs_pad_h1[:, 1:-1, :]   # (6, n, n+2)
        hs_edge_y = 0.5 * (hs_y[:, :, :-1] + hs_y[:, :, 1:])   # (6, n, n+1)
        h_L_y = jnp.maximum(eta_L_y - hs_edge_y, 0.0)
        h_R_y = jnp.maximum(eta_R_y - hs_edge_y, 0.0)
    else:
        h_L_y = eta_L_y
        h_R_y = eta_R_y

    if g > 0:
        c_L_y = jnp.sqrt(jnp.maximum(g * h_L_y, 0.0))
        c_R_y = jnp.sqrt(jnp.maximum(g * h_R_y, 0.0))
        alpha_y = jnp.maximum(jnp.abs(v_L_y) + c_L_y, jnp.abs(v_R_y) + c_R_y)
    else:
        alpha_y = jnp.maximum(jnp.abs(v_L_y), jnp.abs(v_R_y))

    G_y = (0.5 * (h_L_y * v_L_y + h_R_y * v_R_y)
           - 0.5 * alpha_y * (h_R_y - h_L_y))

    hx_edge = 0.5 * (grid.hx_ext[:, 1:-1, :-1] + grid.hx_ext[:, 1:-1, 1:])
    Phi_y = G_y * hx_edge

    # ---- Net flux divergence ----
    net_x = Phi_x[:, 1:, :] - Phi_x[:, :-1, :]
    net_y = Phi_y[:, :, 1:] - Phi_y[:, :, :-1]

    return -(net_x + net_y) / grid.area


def fv_scalar_advection(q, u, v, grid, limiter="mc"):
    """Compute upwind FV advection of scalar q by velocity (u, v).

    Returns -div(q*v), which approximates -v*grad(q) for nearly
    divergence-free flow.

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
        Scalar field to advect.
    u, v : jax.Array, shape (6, n, n)
        Velocity components (grid-aligned).
    grid : CubedSphereGrid
    limiter : str
        Reconstruction scheme: "mc", "minmod", "ppm", or "weno5".

    Returns
    -------
    jax.Array, shape (6, n, n)
        Advective tendency: -div(q*v).
    """
    halo = _get_halo_width(limiter)
    interp_off, cos_ap, sin_ap = _halo_fields(grid, halo)

    # q padded with reconstruction halo
    q_pad = pad_halo(q, halo=halo, interp_offsets=interp_off)

    # Velocity always halo=1 (only used for upwind direction, not reconstructed)
    u_pad, v_pad = pad_halo_vector(
        u, v,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )

    # ---- X-direction upwind flux ----
    q_L_x, q_R_x = _reconstruct_x(q_pad, limiter)   # (6, n+1, n)

    # Edge velocity (average of adjacent cells, halo=1)
    u_edge = 0.5 * (u_pad[:, :-1, 1:-1] + u_pad[:, 1:, 1:-1])

    q_edge_x = jnp.where(u_edge > 0, q_L_x, q_R_x)

    hy_edge = 0.5 * (grid.hy_ext[:, :-1, 1:-1] + grid.hy_ext[:, 1:, 1:-1])
    Phi_x = q_edge_x * u_edge * hy_edge

    # ---- Y-direction upwind flux ----
    q_L_y, q_R_y = _reconstruct_y(q_pad, limiter)    # (6, n, n+1)

    v_edge = 0.5 * (v_pad[:, 1:-1, :-1] + v_pad[:, 1:-1, 1:])

    q_edge_y = jnp.where(v_edge > 0, q_L_y, q_R_y)

    hx_edge = 0.5 * (grid.hx_ext[:, 1:-1, :-1] + grid.hx_ext[:, 1:-1, 1:])
    Phi_y = q_edge_y * v_edge * hx_edge

    # ---- Net flux divergence ----
    net_x = Phi_x[:, 1:, :] - Phi_x[:, :-1, :]
    net_y = Phi_y[:, :, 1:] - Phi_y[:, :, :-1]

    return -(net_x + net_y) / grid.area
