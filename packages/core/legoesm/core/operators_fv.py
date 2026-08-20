"""Finite-volume transport operators on the cubed-sphere grid.

Implements PPM (Piecewise Parabolic Method) reconstruction with
Colella-Woodward monotonicity limiting for conservative and advective
2D transport.

Key design:
- PPM reconstruction is purely 1D along grid lines
- At cube edges, pad_halo(halo=2) provides neighbor data
- The 1D stencil never encounters coordinate discontinuities
- Both x and y fluxes are computed on the SAME unmodified field
  (no directional splitting) to preserve geostrophic balance

Key functions
-------------
fv_flux_divergence : Conservative flux-form transport using PPM
fv_scalar_advection : Advective (non-conservative) transport using PPM

References
----------
- Colella & Woodward (1984): The Piecewise Parabolic Method (PPM)
- Lin (2004): A "Vertically Lagrangian" Finite-Volume Dynamical Core (FV3)
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.halo import pad_halo, pad_halo_vector


# ==============================================================================
# PPM edge reconstruction
# ==============================================================================

def ppm_edge_values_axis0(q):
    """Axis-0 twin of the DEFAULT-path :func:`ppm_edge_values`.

    Same 4th-order interior + 2nd-order boundary edge formula, swept
    along axis 0 instead of axis -2 — so callers whose swept axis is
    leading (the lat-lon latitude PPM) need no ``moveaxis`` round-trip.
    On the LL2048@64 trace those transposes were the single largest
    compute-kernel family (input_transpose_fusion, 370 us/step, 11% of
    compute; codex fusion consult 2026-08-11).

    Deliberately supports ONLY the default options (no ``blend_edges``,
    no Fortran xppm boundary — the latitude path never used them).
    Coupled to ``ppm_edge_values`` by a parity unit test
    (tests/unit/test_ppm_axis0_parity.py): change one formula, the test
    forces the other.
    """
    q_hat_inner = ((7.0 / 12.0) * (q[1:-2] + q[2:-1])
                   - (1.0 / 12.0) * (q[:-3] + q[3:]))
    q_hat_lo = 0.5 * (q[0:1] + q[1:2])
    q_hat_hi = 0.5 * (q[-2:-1] + q[-1:])
    return jnp.concatenate([q_hat_lo, q_hat_inner, q_hat_hi], axis=0)


def ppm_edge_values(q_1d, blend_edges=False,
                     apply_fortran_xppm_boundary=False,
                     n_interior=None):
    """4th-order edge values from cell averages along last-but-one axis.

    When ``blend_edges=True``, the boundary edges (inner-most halo to
    first interior cell) are blended with one-sided 3rd-order
    extrapolation from the interior — the FV3 approach of Putman &
    Lin (2007).  With Duo-Grid halo interpolation, this blending is
    unnecessary and disabled by default.

    Parameters
    ----------
    q_1d : jax.Array, shape (..., M, K) where M = n+4
        Cell averages with halo=2, transverse halo stripped.
    blend_edges : bool
        If True, blend boundary edges with one-sided extrapolation.
        Default False (full 4th-order everywhere with Duo-Grid halo).
    apply_fortran_xppm_boundary : bool, default False
        Iter-891 (parallel to iter-889 in `operators_cdgrid.py:_ppm_reconstruct_1d`).
        When True AND ``n_interior`` provided, overwrite the 5
        cube-edge ``q_hat`` indices [1, 2, 3, n_interior+1,
        n_interior+2] with Fortran's iord<7 boundary formulas from
        ``tp_core.F90:357-369``:
        - Left side: ``q_hat[1] = c1*q1(-2) + c2*q1(-1) + c3*q1(0)``
          (Fortran al(0)), ``q_hat[2] = uniform 4-point xt`` (Fortran
          al(1) — cube-face edge), ``q_hat[3] = c3*q1(1) + c2*q1(2)
          + c1*q1(3)`` (Fortran al(2)).
        - Right side: ``q_hat[n_interior+1] = c1*q1(npx-3) + c2*q1(npx-2)
          + c3*q1(npx-1)`` (Fortran al(npx-1)), ``q_hat[n_interior+2]
          = uniform 4-point xt`` clipped (Fortran al(npx) — right
          cube-face edge).
        Constants from ``tp_core.F90:63-65``: ``c1 = -2/14, c2 = 11/14,
        c3 = 5/14``.  4-point xt uses uniform-grid simplification
        ``0.75*(q1(0)+q1(1)) - 0.25*(q1(-1)+q1(2))`` (and right mirror)
        clipped to ``min/max(q1(-1..2))`` (and right mirror).
        Default False preserves prior behaviour bit-for-bit.
    n_interior : int or None, default None
        Number of interior cells along the swept axis when the strip
        has halo=2 padding (so ``M = n_interior + 4``).  Required when
        ``apply_fortran_xppm_boundary=True``.  Ignored otherwise.

    Returns
    -------
    q_hat : jax.Array, shape (..., M-1, K)
        Edge values at interfaces between cells.
    """
    M = q_1d.shape[-2]

    # 4th-order interior edges: M-3 values (indices 1 to M-3 in the full array)
    q_hat_inner = ((7.0 / 12.0) * (q_1d[..., 1:-2, :] + q_1d[..., 2:-1, :])
                   - (1.0 / 12.0) * (q_1d[..., :-3, :] + q_1d[..., 3:, :]))

    if blend_edges and M >= 7:
        # One-sided 3rd-order extrapolation from interior at cube-face boundaries.
        q_os_lo = (
            15.0 * q_1d[..., 2:3, :]
            - 10.0 * q_1d[..., 3:4, :]
            + 3.0 * q_1d[..., 4:5, :]
        ) / 8.0
        q_os_hi = (
            15.0 * q_1d[..., -3:-2, :]
            - 10.0 * q_1d[..., -4:-3, :]
            + 3.0 * q_1d[..., -5:-4, :]
        ) / 8.0

        # Blend: average of 4th-order (which uses halo cells) and one-sided
        # interior extrapolation (no halo dependence).
        q_hat_inner = q_hat_inner.at[..., 0:1, :].set(
            0.5 * (q_hat_inner[..., 0:1, :] + q_os_lo)
        )
        q_hat_inner = q_hat_inner.at[..., -1:, :].set(
            0.5 * (q_hat_inner[..., -1:, :] + q_os_hi)
        )

    # 2nd-order boundary edges (outermost — fully in halo)
    q_hat_lo = 0.5 * (q_1d[..., 0:1, :] + q_1d[..., 1:2, :])
    q_hat_hi = 0.5 * (q_1d[..., -2:-1, :] + q_1d[..., -1:, :])

    # All M-1 = n+3 edge values
    q_hat = jnp.concatenate([q_hat_lo, q_hat_inner, q_hat_hi], axis=-2)

    # Iter-880: pre-iter-880 we clipped each edge value into the
    # ``[min(q[i],q[i+1]), max(q[i],q[i+1])]`` range here.  Fortran's
    # ``xppm`` (tp_core.F90:353-355) does NOT clip the 4th-order
    # edge values; it passes them directly to the CW84 ``pert_ppm``
    # constraint (which our caller applies via ``ppm_limit``).  The
    # extra clip step in our Python made the limiter MORE diffusive
    # than Fortran by pre-flattening edge overshoots before the CW84
    # constraint could see them.  Removed for Fortran fidelity.

    # Iter-891b (Codex iter-891 stop-time fix — off-by-one in q_hat
    # placement).  Apply Fortran's iord<7 cube-edge boundary overrides
    # at tp_core.F90:357-369 behind a default-OFF kwarg.  Constants
    # from tp_core.F90:63-65: c1 = -2/14, c2 = 11/14, c3 = 5/14.
    #
    # Correct index mapping (Codex iter-891 stop-time correction):
    #   q_1d[k] = Fortran q1(k-1) for k=0..n+3   (q_1d[0]=q1(-1)=halo
    #     depth 1, q_1d[1]=q1(0)=halo depth 0, q_1d[2]=q1(1)=first
    #     interior).
    #   q_hat[k] for k=1..n_int+1 is the inner 4th-order face between
    #     q_1d[k] and q_1d[k+1] = face between q1(k-1) and q1(k) =
    #     Fortran al(k).  i.e. al(i) → q_hat[i].
    # Pre-iter-891b iter-891 used `al(i) → q_hat[i+1]`, off-by-one.
    #
    # Fortran al(0) needs q1(-2) and Fortran al(npx+1) needs q1(npx+2);
    # both are halo depth 2 cells unavailable with our halo=2 input.
    # iter-891b therefore overrides ONLY the 4 cube-edge faces that
    # are computable Fortran-faithfully with halo=2:
    #   q_hat[1]            ← Fortran al(1)     left 4-point xt clipped
    #   q_hat[2]            ← Fortran al(2)     left c3/c2/c1 mirror
    #   q_hat[n_int]        ← Fortran al(npx-1) right c1/c2/c3 formula
    #   q_hat[n_int+1]      ← Fortran al(npx)   right 4-point xt clipped
    # Fortran al(0)/al(npx+1) need halo=3 for true Fortran fidelity;
    # those slots are LEFT UNTOUCHED on the standard 4th-order /
    # boundary-halo path.
    if apply_fortran_xppm_boundary and n_interior is not None:
        c1 = -2.0 / 14.0
        c2 = 11.0 / 14.0
        c3 = 5.0 / 14.0
        n_int = int(n_interior)

        # LEFT cube-edge overrides (al(1) and al(2), placed at q_hat[1]
        # and q_hat[2]).  q_hat[0] (= q_hat_lo, al(0) position) is left
        # untouched because Fortran al(0) needs q1(-2) which our halo=2
        # input lacks.
        # al(1) = uniform 4-pt xt = 0.75*(q1(0)+q1(1)) - 0.25*(q1(-1)+q1(2))
        # clipped to min/max(q1(-1..2)).
        # In our q_1d: q1(-1)=q_1d[0], q1(0)=q_1d[1], q1(1)=q_1d[2],
        # q1(2)=q_1d[3].
        xt_L = (0.75 * (q_1d[..., 1, :] + q_1d[..., 2, :])
                - 0.25 * (q_1d[..., 0, :] + q_1d[..., 3, :]))
        q_lo_L = jnp.minimum(jnp.minimum(q_1d[..., 0, :], q_1d[..., 1, :]),
                              jnp.minimum(q_1d[..., 2, :], q_1d[..., 3, :]))
        q_hi_L = jnp.maximum(jnp.maximum(q_1d[..., 0, :], q_1d[..., 1, :]),
                              jnp.maximum(q_1d[..., 2, :], q_1d[..., 3, :]))
        face_al1 = jnp.clip(xt_L, q_lo_L, q_hi_L)
        # al(2) = c3*q1(1) + c2*q1(2) + c1*q1(3)
        # In our q_1d: q1(1)=q_1d[2], q1(2)=q_1d[3], q1(3)=q_1d[4].
        face_al2 = (c3 * q_1d[..., 2, :] + c2 * q_1d[..., 3, :]
                    + c1 * q_1d[..., 4, :])

        q_hat = q_hat.at[..., 1, :].set(face_al1)
        q_hat = q_hat.at[..., 2, :].set(face_al2)

        # RIGHT cube-edge overrides (al(npx-1) and al(npx), placed at
        # q_hat[n_int] and q_hat[n_int+1]).  q_hat[n_int+2] (=
        # q_hat_hi, al(npx+1) position) is left untouched because
        # Fortran al(npx+1) needs q1(npx+2) which our halo=2 input
        # lacks.
        #
        # Index map: q_1d[k] = q1(k-1) for k=0..n_int+3, so q1(j) =
        # q_1d[j+1].  With npx = n_int + 1 (n_int interior cells in
        # q_1d at indices 2..n_int+1):
        #   q1(npx-3) = q1(n_int-2) = q_1d[n_int-1]
        #   q1(npx-2) = q1(n_int-1) = q_1d[n_int]
        #   q1(npx-1) = q1(n_int)   = q_1d[n_int+1]   (last interior)
        #   q1(npx)   = q1(n_int+1) = q_1d[n_int+2]   (right halo depth 0)
        #   q1(npx+1) = q1(n_int+2) = q_1d[n_int+3]   (right halo depth 1)
        # al(npx-1) = c1*q1(npx-3) + c2*q1(npx-2) + c3*q1(npx-1)
        face_alnm1 = (c1 * q_1d[..., n_int - 1, :]
                      + c2 * q_1d[..., n_int, :]
                      + c3 * q_1d[..., n_int + 1, :])
        # al(npx) = uniform 4-pt xt clipped to min/max(q1(npx-2..npx+1)).
        xt_R = (0.75 * (q_1d[..., n_int + 1, :] + q_1d[..., n_int + 2, :])
                - 0.25 * (q_1d[..., n_int, :] + q_1d[..., n_int + 3, :]))
        q_lo_R = jnp.minimum(
            jnp.minimum(q_1d[..., n_int, :], q_1d[..., n_int + 1, :]),
            jnp.minimum(q_1d[..., n_int + 2, :], q_1d[..., n_int + 3, :]))
        q_hi_R = jnp.maximum(
            jnp.maximum(q_1d[..., n_int, :], q_1d[..., n_int + 1, :]),
            jnp.maximum(q_1d[..., n_int + 2, :], q_1d[..., n_int + 3, :]))
        face_aln = jnp.clip(xt_R, q_lo_R, q_hi_R)

        q_hat = q_hat.at[..., n_int, :].set(face_alnm1)
        q_hat = q_hat.at[..., n_int + 1, :].set(face_aln)

    return q_hat


def ppm_limit(q_bar, q_L, q_R):
    """Colella-Woodward monotonicity limiter for PPM.

    Limits left/right parabola edge values to prevent new extrema.

    Parameters
    ----------
    q_bar : jax.Array, shape (...)
        Cell averages.
    q_L, q_R : jax.Array, shape (...)
        Left and right edge values per cell.

    Returns
    -------
    q_L_lim, q_R_lim : jax.Array
        Limited edge values.
    """
    # Detect local extrema: parabola should be flat
    is_extremum = (q_R - q_bar) * (q_bar - q_L) <= 0

    dm = q_R - q_L
    d6 = 6.0 * (q_bar - 0.5 * (q_L + q_R))

    # Overshoot on the left: parabola peak/trough outside [q_L, q_R]
    over_L = dm * d6 > dm ** 2
    q_L_lim = jnp.where(over_L, 3.0 * q_bar - 2.0 * q_R, q_L)

    # Overshoot on the right
    over_R = dm * d6 < -(dm ** 2)
    q_R_lim = jnp.where(over_R, 3.0 * q_bar - 2.0 * q_L, q_R)

    # At local extrema, flatten to cell average
    q_L_lim = jnp.where(is_extremum, q_bar, q_L_lim)
    q_R_lim = jnp.where(is_extremum, q_bar, q_R_lim)

    return q_L_lim, q_R_lim


def _ppm_reconstruct_x(q_pad_h2, limiter=True,
                       apply_fortran_xppm_boundary=False):
    """PPM reconstruction in x-direction.

    Parameters
    ----------
    q_pad_h2 : jax.Array, shape (6, n+4, n+4)
        Scalar padded with halo=2.
    limiter : bool
        Apply Colella-Woodward limiter.
    apply_fortran_xppm_boundary : bool, default False
        Iter-891: forwards through to ``ppm_edge_values`` so the
        Fortran iord<7 cube-edge boundary formulas
        (`tp_core.F90:357-369`) are reachable from
        ``fv_flux_divergence`` callers.  Default False preserves
        prior behaviour bit-for-bit.

    Returns
    -------
    q_left, q_right : each shape (6, n+1, n)
        Left and right states at x-direction interfaces.
        q_left[i] = right-edge of cell i (cell to left of interface i)
        q_right[i] = left-edge of cell i+1 (cell to right of interface i)
    """
    q = q_pad_h2[:, :, 2:-2]  # (6, n+4, n) — strip transverse halo
    n = q.shape[-2] - 4  # interior cell count along the swept axis

    # Edge values: (6, n+3, n) at all M-1 interfaces
    q_hat = ppm_edge_values(
        q,
        apply_fortran_xppm_boundary=apply_fortran_xppm_boundary,
        n_interior=n)

    # Parabola for cells 1..n+2 (interior cells in padded array)
    a_L = q_hat[..., :-1, :]   # left edge of each cell, (6, n+2, n)
    a_R = q_hat[..., 1:, :]    # right edge
    q_c = q[..., 1:-1, :]       # cell centers, (6, n+2, n)

    if limiter:
        a_L, a_R = ppm_limit(q_c, a_L, a_R)

    # Extract n+1 interior interface states
    q_left = a_R[..., :-1, :]   # right-edge of cell to left of interface
    q_right = a_L[..., 1:, :]   # left-edge of cell to right of interface

    return q_left, q_right


def _ppm_reconstruct_y(q_pad_h2, limiter=True,
                       apply_fortran_xppm_boundary=False):
    """PPM reconstruction in y-direction.

    Iter-891: forwards ``apply_fortran_xppm_boundary`` to
    ``ppm_edge_values``.  Default False preserves prior behaviour.

    Parameters
    ----------
    q_pad_h2 : jax.Array, shape (6, n+4, n+4)

    Returns
    -------
    q_left, q_right : each shape (6, n, n+1)
    """
    q = q_pad_h2[:, 2:-2, :]  # (6, n, n+4) — strip transverse halo
    n = q.shape[-1] - 4  # interior cell count along the swept axis
    # Transpose to reuse x-direction logic
    q_t = jnp.swapaxes(q, -2, -1)  # (6, n+4, n)
    q_hat = ppm_edge_values(
        q_t,
        apply_fortran_xppm_boundary=apply_fortran_xppm_boundary,
        n_interior=n)

    a_L = q_hat[..., :-1, :]
    a_R = q_hat[..., 1:, :]
    q_c = q_t[..., 1:-1, :]

    if limiter:
        a_L, a_R = ppm_limit(q_c, a_L, a_R)

    q_left_t = a_R[..., :-1, :]
    q_right_t = a_L[..., 1:, :]

    return jnp.swapaxes(q_left_t, -2, -1), jnp.swapaxes(q_right_t, -2, -1)


# ==============================================================================
# Unsplit 2D flux-form transport
# ==============================================================================

def fv_flux_divergence(q, u, v, grid, limiter=True,
                       apply_fortran_xppm_boundary=False):
    """Conservative flux-form 2D transport using PPM (unsplit).

    Both x and y fluxes are computed on the SAME unmodified field q.
    This is critical for maintaining discrete geostrophic balance when
    the momentum equation uses centered differences.

    Each directional flux is a telescoping sum, so the global sum
    of the total divergence is zero by construction.

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
        Scalar field to transport (e.g. fluid depth h, density rho).
    u, v : jax.Array, shape (6, n, n)
        Velocity components (grid-aligned).
    grid : CubedSphereGrid
    limiter : bool
        Apply Colella-Woodward monotonicity limiter.
    apply_fortran_xppm_boundary : bool, default False
        Iter-891 (parallel to iter-889 production-path plumbing).
        Forwards through to ``_ppm_reconstruct_x`` / ``_ppm_reconstruct_y``
        / ``ppm_edge_values`` so Fortran's iord<7 cube-edge boundary
        formulas (`tp_core.F90:357-369`) become reachable from
        ``fv_flux_divergence``.  Iter-891 also matches iter-889b's
        bounded_domain gate: we only fire the override when the grid
        is a global cubed sphere (`not grid.bounded_domain`).
        Default False preserves prior behaviour bit-for-bit.

    Returns
    -------
    jax.Array, shape (6, n, n)
        Flux divergence tendency: dq/dt = -div(q * v).
    """
    # Iter-891b (matching iter-889b's bounded_domain gate pattern):
    # the iord<7 boundary formulas are gated on `not bounded_domain`
    # in Fortran (`tp_core.F90:333/357`).  Compute the effective flag
    # here so the leaf `ppm_edge_values` receives a pre-gated boolean.
    effective_xppm_boundary = (
        apply_fortran_xppm_boundary
        and not bool(getattr(grid, "bounded_domain", False)))

    # Single halo exchange for both directions
    q_pad = pad_halo(q, halo=2, interp_offsets=grid.halo_interp_offsets_h2)
    u_pad, v_pad = pad_halo_vector(
        u, v,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded_h2, grid.sin_angle_padded_h2,
        interp_offsets=grid.halo_interp_offsets_h2, halo=2,
    )

    # --- X-direction flux ---
    q_L_x, q_R_x = _ppm_reconstruct_x(
        q_pad, limiter,
        apply_fortran_xppm_boundary=effective_xppm_boundary)  # each (6, n+1, n)

    u_strip = u_pad[:, :, 2:-2]  # (6, n+4, n)
    u_iface = 0.5 * (u_strip[:, 1:-2, :] + u_strip[:, 2:-1, :])  # (6, n+1, n)

    q_face_x = jnp.where(u_iface > 0, q_L_x, q_R_x)

    hy = grid.hy_ext_h2[:, :, 2:-2]  # (6, n+4, n)
    hy_iface = 0.5 * (hy[:, 1:-2, :] + hy[:, 2:-1, :])  # (6, n+1, n)

    Phi_x = u_iface * hy_iface * q_face_x  # (6, n+1, n)

    # --- Y-direction flux ---
    q_L_y, q_R_y = _ppm_reconstruct_y(
        q_pad, limiter,
        apply_fortran_xppm_boundary=effective_xppm_boundary)  # each (6, n, n+1)

    v_strip = v_pad[:, 2:-2, :]  # (6, n, n+4)
    v_iface = 0.5 * (v_strip[:, :, 1:-2] + v_strip[:, :, 2:-1])  # (6, n, n+1)

    q_face_y = jnp.where(v_iface > 0, q_L_y, q_R_y)

    hx = grid.hx_ext_h2[:, 2:-2, :]  # (6, n, n+4)
    hx_iface = 0.5 * (hx[:, :, 1:-2] + hx[:, :, 2:-1])  # (6, n, n+1)

    Phi_y = v_iface * hx_iface * q_face_y  # (6, n, n+1)

    # --- Net flux divergence (both directions, same field) ---
    net_x = Phi_x[:, 1:, :] - Phi_x[:, :-1, :]
    net_y = Phi_y[:, :, 1:] - Phi_y[:, :, :-1]

    return -(net_x + net_y) / grid.area


def fv_scalar_advection(q, u, v, grid, limiter=True):
    """PPM advection of scalar q by (u,v) in advective form.

    Computes:
        -v·∇q = -div(q v) + q div(v)

    using the same unsplit FV transport operator for both terms, which
    preserves constant-field invariance even for divergent flow.

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
        Scalar field to advect.
    u, v : jax.Array, shape (6, n, n)
        Velocity components.
    grid : CubedSphereGrid
    limiter : bool
        Apply Colella-Woodward monotonicity limiter.

    Returns
    -------
    jax.Array, shape (6, n, n)
        Advective tendency: approximately -v·grad(q).
    """
    flux_form = fv_flux_divergence(q, u, v, grid, limiter)
    div_v = -fv_flux_divergence(jnp.ones_like(q), u, v, grid, limiter=False)
    return flux_form + q * div_v


# ==============================================================================
# PPM-compatible gradients
# ==============================================================================

def fv_gradient_x(q, grid):
    """PPM-compatible x-gradient using 4th-order edge values.

    Computes dq/dx at cell centers by differencing PPM edge values
    at the left and right cell boundaries.  This makes the gradient
    operator compatible with the FV mass flux, preserving discrete
    geostrophic balance.

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
    grid : CubedSphereGrid

    Returns
    -------
    jax.Array, shape (6, n, n)
    """
    q_pad = pad_halo(q, halo=2, interp_offsets=grid.halo_interp_offsets_h2)
    q_strip = q_pad[:, :, 2:-2]   # (6, n+4, n) — strip transverse halo

    # 4th-order edge values along x: (6, n+3, n)
    q_hat = ppm_edge_values(q_strip)

    # Interior edges for n cells: need n+1 edges (indices 1..n+1)
    q_edges = q_hat[:, 1:-1, :]   # (6, n+1, n)

    # Gradient: (right edge - left edge) / cell width
    # grid.dx spans 2 cells, so single-cell width = dx/2
    return (q_edges[:, 1:, :] - q_edges[:, :-1, :]) / (grid.dx / 2.0)


def fv_gradient_y(q, grid):
    """PPM-compatible y-gradient using 4th-order edge values.

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
    grid : CubedSphereGrid

    Returns
    -------
    jax.Array, shape (6, n, n)
    """
    q_pad = pad_halo(q, halo=2, interp_offsets=grid.halo_interp_offsets_h2)
    q_strip = q_pad[:, 2:-2, :]   # (6, n, n+4) — strip transverse halo

    # Transpose to reuse x-direction PPM edge values
    q_t = jnp.swapaxes(q_strip, -2, -1)  # (6, n+4, n)
    q_hat_t = ppm_edge_values(q_t)       # (6, n+3, n)
    q_edges_t = q_hat_t[:, 1:-1, :]       # (6, n+1, n)

    dq_t = q_edges_t[:, 1:, :] - q_edges_t[:, :-1, :]  # (6, n, n)

    # Swap back to standard (face, x, y) layout and divide by cell width
    # grid.dy spans 2 cells, so single-cell width = dy/2
    return jnp.swapaxes(dq_t, -2, -1) / (grid.dy / 2.0)
