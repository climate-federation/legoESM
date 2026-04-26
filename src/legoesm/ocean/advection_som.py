"""SOM (Second Order Moments / Prather 1986) tracer advection.

Tracks 10 moments per cell (mean + 9 polynomial coefficients) to achieve
near-zero spurious diapycnal mixing on z-coordinate grids.  Fully smooth
(no min/max limiter), making it ideal for jax.grad.

The sub-cell distribution is a degree-2 polynomial in normalised
coordinates xi = x/dx in [-1/2, 1/2]:

    q(xi_x, xi_y, xi_z) = sm_o/V
        + sx  * (2 xi_x)
        + sy  * (2 xi_y)
        + sz  * (2 xi_z)
        + sxx * (6 xi_x^2 - 1/2)
        + syy * (6 xi_y^2 - 1/2)
        + szz * (6 xi_z^2 - 1/2)
        + sxy * (4 xi_x xi_y)
        + sxz * (4 xi_x xi_z)
        + syz * (4 xi_y xi_z)

Reference:
    Prather, M. (1986). "Numerical advection by conservation of
    second-order moments." J. Geophys. Res., 91, 6671-6681.

    Hill et al. (2012). "Controlling spurious diapycnal mixing in
    eddy-resolving height-coordinate ocean models."
    Ocean Modelling, 45-46, 14-26.
"""

import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid


# Moment indices in the (…, 9) moment array.
IX, IY, IZ = 0, 1, 2
IXX, IYY, IZZ = 3, 4, 5
IXY, IXZ, IYZ = 6, 7, 8

_EPS = 1e-20  # Safe for both float32 and float64; 1e-30 overflows in float32 AD


# =============================================================================
# Moment limiter
# =============================================================================

def _limit_moments(sm_o, mom):
    """Limit moments to prevent the sub-cell polynomial from producing
    extreme (especially negative) tracer values.

    For our basis {1, 2ξ, 6ξ²-1/2}, the edge values (ξ = ±1/2) are::

        q_edge = sm_o ± s_d + s_dd

    (since 2*(±1/2)=±1 and 6*(1/4)-1/2=1).  We scale down all moments
    uniformly if the edge value would go negative::

        |s_d| + |s_dd| <= sm_o

    The limiter has a single kink at ``max(0, ...)`` — smooth enough for
    ``jax.grad`` (one dead-gradient point per cell, much better than
    Superbee's four kinks).
    """
    abs_sm_o = jnp.abs(sm_o) + _EPS

    # For each direction d, the edge values (in volume-weighted form) are:
    #   q(+1/2) = sm_o + s_d + s_dd
    #   q(-1/2) = sm_o - s_d + s_dd
    # For non-negativity at edges when s_dd < 0:
    #   sm_o - |s_d| - |s_dd| >= 0  →  |s_d| + |s_dd| <= sm_o
    # When s_dd > 0 the edges are always larger than the interior min,
    # but the interior parabola minimum may go negative; this bound
    # is sufficient for both cases (verified numerically).
    excess_x = (jnp.abs(mom[..., IX]) + jnp.abs(mom[..., IXX])) / abs_sm_o
    excess_y = (jnp.abs(mom[..., IY]) + jnp.abs(mom[..., IYY])) / abs_sm_o
    excess_z = (jnp.abs(mom[..., IZ]) + jnp.abs(mom[..., IZZ])) / abs_sm_o

    max_excess = jnp.maximum(jnp.maximum(excess_x, excess_y), excess_z)

    # Scale factor: if max_excess > 1, scale all moments by 1/max_excess.
    # Use max(excess, 1) instead of where(excess>1, 1/excess, 1) to avoid
    # 1/0 = inf in the gradient when moments are zero (JAX traces both branches).
    scale = 1.0 / jnp.maximum(max_excess, 1.0)

    return mom * scale[..., jnp.newaxis]


# =============================================================================
# Flux extraction helpers
# =============================================================================

def _extract_flux(sm_o, mom, alpha, sign_edge):
    """Compute flux moments leaving through one edge of a cell.

    Parameters
    ----------
    sm_o : array (...,)
        Volume-weighted tracer mean of donor cell.
    mom : array (..., 9)
        Higher-order moments of donor cell.
    alpha : array (...,)
        Courant fraction (>= 0, < 1) — fraction of cell volume leaving.
    sign_edge : float or array (...,)
        +1.0 for extraction from the RIGHT/NORTH/TOP edge (positive flow),
        -1.0 for extraction from the LEFT/SOUTH/BOTTOM edge (negative flow).
        The sign_edge multiplies coupling terms between the sweep direction
        and transverse directions.

    Returns
    -------
    fp_o : array (...,)
        Volume-weighted mean of the outgoing slab (unsigned, >= 0).
    fp_mom : array (..., 9)
        Higher-order moments of the outgoing slab.

    Notes
    -----
    Formulas assume the sweep direction maps to the x-index in the moment
    array.  Callers for y- and z-sweeps must permute the moment indices
    before calling (see ``_permuted_extract``).
    """
    alp1 = 1.0 - alpha
    se = sign_edge

    sx = mom[..., IX]
    sy = mom[..., IY]
    sz = mom[..., IZ]
    sxx = mom[..., IXX]
    syy = mom[..., IYY]
    szz = mom[..., IZZ]
    sxy = mom[..., IXY]
    sxz = mom[..., IXZ]
    syz = mom[..., IYZ]

    # Mean of outgoing slab
    fp_o = alpha * (sm_o + se * alp1 * sx + alp1 * (alp1 - alpha) * sxx)

    # Sweep-direction moments
    fp_sx = alpha ** 2 * (sx + se * 3.0 * alp1 * sxx)
    fp_sxx = alpha ** 3 * sxx

    # Cross-term moments (sweep × transverse)
    fp_sxy = alpha ** 2 * sxy
    fp_sxz = alpha ** 2 * sxz

    # Transverse slopes (coupled through cross terms)
    fp_sy = alpha * (sy + se * alp1 * sxy)
    fp_sz = alpha * (sz + se * alp1 * sxz)

    # Pure transverse moments (simple scaling)
    fp_syy = alpha * syy
    fp_szz = alpha * szz
    fp_syz = alpha * syz

    fp_mom = jnp.stack(
        [fp_sx, fp_sy, fp_sz, fp_sxx, fp_syy, fp_szz, fp_sxy, fp_sxz, fp_syz],
        axis=-1,
    )
    return fp_o, fp_mom


def _permute_moments(mom, sweep):
    """Re-order a (..., 9) moment array so that the sweep direction maps to x.

    After permutation the generic flux/merge helpers operate correctly.
    ``_unpermute_moments`` reverses this.

    For the z-sweep the array index direction (k increases downward) is
    opposite to the physical z-direction (positive = upward).  The flux
    sign is flipped in ``_som_z_sweep`` so that positive = downward
    (increasing index), but the moments must also be sign-flipped:
    odd-in-z moments (sz, sxz, syz) are negated so that the generic
    helpers see a consistent "positive = rightward = downward" convention.
    """
    if sweep == "x":
        return mom
    if sweep == "y":
        # Rename y→x, x→y in the moment slots
        return mom[..., [IY, IX, IZ, IYY, IXX, IZZ, IXY, IYZ, IXZ]]
    # sweep == "z" — permute AND negate odd-in-z (slots IZ, IXZ, IYZ).
    # After permutation these map to slots 0 (sweep slope), 7 (sweep×trans2),
    # 6 (sweep×trans1), i.e. the generic "odd-in-x" positions.
    out = mom[..., [IZ, IY, IX, IZZ, IYY, IXX, IYZ, IXZ, IXY]]
    # Negate the slots that came from odd-in-z moments:
    #   slot 0 ← IZ,  slot 6 ← IYZ,  slot 7 ← IXZ
    return out * jnp.array([-1, 1, 1, 1, 1, 1, -1, -1, 1], dtype=mom.dtype)


def _unpermute_moments(mom, sweep):
    """Inverse of ``_permute_moments``."""
    if sweep == "x":
        return mom
    if sweep == "y":
        return mom[..., [1, 0, 2, 4, 3, 5, 6, 8, 7]]
    # sweep == "z" — undo sign flip, then unpermute
    mom = mom * jnp.array([-1, 1, 1, 1, 1, 1, -1, -1, 1], dtype=mom.dtype)
    return mom[..., [2, 1, 0, 5, 4, 3, 8, 7, 6]]


# =============================================================================
# Donor update (remove outgoing slabs from both faces)
# =============================================================================

def _donor_update(sm_o, mom, fp_o_left, fp_mom_left, fp_o_right, fp_mom_right,
                  alpha_L, alpha_R):
    """Update a cell after outgoing flux removal from both faces.

    Works in permuted coordinates where the sweep direction is mapped to x.

    Parameters
    ----------
    sm_o, mom : current cell content
    fp_o_left, fp_mom_left : flux leaving through the LEFT face (unsigned)
    fp_o_right, fp_mom_right : flux leaving through the RIGHT face (unsigned)
    alpha_L : Courant fraction leaving leftward (>= 0)
    alpha_R : Courant fraction leaving rightward (>= 0)

    Returns
    -------
    sm_o_d, mom_d : donor-updated cell content
    """
    alf1 = jnp.maximum(1.0 - alpha_L - alpha_R, _EPS)
    alpmn = alpha_R - alpha_L  # asymmetry

    # Total tracer update: subtract both outgoing slabs
    sm_o_d = sm_o - fp_o_left - fp_o_right

    sx = mom[..., IX]
    sxx = mom[..., IXX]
    sxy = mom[..., IXY]
    sxz = mom[..., IXZ]

    # Sweep-direction moments are rescaled (internal distribution shrinks)
    sx_d = alf1 ** 2 * (sx - 3.0 * alpmn * sxx)
    sxx_d = alf1 ** 3 * sxx
    sxy_d = alf1 ** 2 * sxy
    sxz_d = alf1 ** 2 * sxz

    # Transverse moments: subtract outgoing contributions
    sy_d = mom[..., IY] - fp_mom_left[..., IY] - fp_mom_right[..., IY]
    sz_d = mom[..., IZ] - fp_mom_left[..., IZ] - fp_mom_right[..., IZ]
    syy_d = mom[..., IYY] - fp_mom_left[..., IYY] - fp_mom_right[..., IYY]
    szz_d = mom[..., IZZ] - fp_mom_left[..., IZZ] - fp_mom_right[..., IZZ]
    syz_d = mom[..., IYZ] - fp_mom_left[..., IYZ] - fp_mom_right[..., IYZ]

    mom_d = jnp.stack(
        [sx_d, sy_d, sz_d, sxx_d, syy_d, szz_d, sxy_d, sxz_d, syz_d],
        axis=-1,
    )
    return sm_o_d, mom_d


# =============================================================================
# Receiver merge (add incoming slab to cell)
# =============================================================================

def _receiver_merge(sm_o, mom, fp_o, fp_mom, vol_cell, vol_flux, from_left):
    """Merge incoming flux slab into a cell.

    Works in permuted coordinates where the sweep direction is mapped to x.

    Parameters
    ----------
    sm_o, mom : current cell content (after donor update)
    fp_o, fp_mom : incoming flux (unsigned)
    vol_cell : cell volume BEFORE this merge
    vol_flux : incoming volume (unsigned, >= 0)
    from_left : bool or array
        True if the slab enters from the LEFT face, False if from the RIGHT.

    Returns
    -------
    sm_o_new, mom_new, vol_new : merged cell content and new volume
    """
    vol_new = vol_cell + vol_flux
    alf = vol_flux / jnp.maximum(vol_new, _EPS)   # fraction from incoming
    alf1 = 1.0 - alf                               # fraction from remaining

    # Signed displacement moment:
    # from_left  → remaining at +x, incoming at -x → d0 = alf*sm_o - alf1*fp_o
    # from_right → remaining at -x, incoming at +x → d0 = alf1*fp_o - alf*sm_o
    sign = jnp.where(from_left, 1.0, -1.0)
    d0 = sign * (alf * sm_o - alf1 * fp_o)

    sx_cell = mom[..., IX]
    sxx_cell = mom[..., IXX]
    fp_sx = fp_mom[..., IX]
    fp_sxx = fp_mom[..., IXX]

    d1 = sign * (alf ** 2 * sx_cell - alf1 ** 2 * fp_sx)

    sm_o_new = sm_o + fp_o

    # Sweep-direction moments
    #
    # MITgcm (GAD_SOM_ADV_R.F) uses the same {2ξ, 6ξ²-1/2} basis and
    # factors (3, 5).  The weighted combination alf1*cell + alf*flux
    # accounts for the sub-cell position of each body in the merged cell.
    sx_new = alf1 * sx_cell + alf * fp_sx + 3.0 * d0
    sxx_new = (alf1 ** 2 * sxx_cell + alf ** 2 * fp_sxx
               + 5.0 * sign * alf * alf1 * (sx_cell - fp_sx)
               + 5.0 * (alf - alf1) * d0 * sign)

    # Cross terms: simple addition (same as transverse moments).
    sxy_new = mom[..., IXY] + fp_mom[..., IXY]
    sxz_new = mom[..., IXZ] + fp_mom[..., IXZ]

    # Transverse moments: simple addition (no sweep-direction coupling)
    sy_new = mom[..., IY] + fp_mom[..., IY]
    sz_new = mom[..., IZ] + fp_mom[..., IZ]
    syy_new = mom[..., IYY] + fp_mom[..., IYY]
    szz_new = mom[..., IZZ] + fp_mom[..., IZZ]
    syz_new = mom[..., IYZ] + fp_mom[..., IYZ]

    mom_new = jnp.stack(
        [sx_new, sy_new, sz_new, sxx_new, syy_new, szz_new,
         sxy_new, sxz_new, syz_new],
        axis=-1,
    )
    return sm_o_new, mom_new, vol_new


# =============================================================================
# X-sweep (periodic in longitude)
# =============================================================================

def _som_x_sweep(sm_o, moments, vol_flux_x, vol):
    """SOM x-sweep with periodic longitude boundary.

    Parameters
    ----------
    sm_o : (n_lat, n_lon, nlev)
        Volume-weighted tracer mean.
    moments : (n_lat, n_lon, nlev, 9)
        Higher-order moments.
    vol_flux_x : (n_lat, n_lon, nlev)
        SIGNED volume transport at interior x-faces, positive = eastward.
        Face j sits between cell (j-1) mod n_lon and cell j.
        Only n_lon faces needed (periodic wrap: face n_lon = face 0).
    vol : (n_lat, n_lon, nlev)
        Cell volume.

    Returns
    -------
    sm_o_new, moments_new, vol_new
    """
    # The x-sweep works directly in the canonical moment ordering (no permute needed).
    mom = moments

    # --- Step 1: Compute face fluxes ---
    vf = vol_flux_x  # (n_lat, n_lon, nlev), face j between cell j-1 and j

    # Donor for positive flow: cell j-1 (left of face)
    sm_o_left = jnp.roll(sm_o, 1, axis=1)
    mom_left = jnp.roll(mom, 1, axis=1)
    vol_left = jnp.roll(vol, 1, axis=1)

    # Donor for negative flow: cell j (right of face)
    sm_o_right = sm_o
    mom_right = mom
    vol_right = vol

    is_pos = vf >= 0
    vol_donor = jnp.where(is_pos, vol_left, vol_right)
    alpha = jnp.minimum(jnp.abs(vf) / jnp.maximum(vol_donor, _EPS), 1.0)
    sign_edge = jnp.where(is_pos, 1.0, -1.0)

    # Select donor moments
    sm_o_donor = jnp.where(is_pos, sm_o_left, sm_o_right)
    mom_donor = jnp.where(is_pos[..., jnp.newaxis], mom_left, mom_right)

    fp_o, fp_mom = _extract_flux(sm_o_donor, mom_donor, alpha, sign_edge)

    # --- Step 2: Per-cell incoming/outgoing identification ---
    # For cell j: left face = face j, right face = face (j+1) % n_lon
    vf_left = vf                            # face j
    vf_right = jnp.roll(vf, -1, axis=1)    # face j+1

    # Outgoing Courant fractions from this cell
    alpha_out_L = jnp.where(vf_left < 0,
                            jnp.minimum(-vf_left / jnp.maximum(vol, _EPS), 1.0),
                            0.0)
    alpha_out_R = jnp.where(vf_right > 0,
                            jnp.minimum(vf_right / jnp.maximum(vol, _EPS), 1.0),
                            0.0)

    # Outgoing flux moments (from this cell)
    # When vf_left < 0, this cell is the donor at face j (left-edge extraction)
    # The face flux at face j was computed from this cell when is_pos[face j] = False
    # fp at face j corresponds to extraction from cell j = this cell
    fp_o_out_L = jnp.where(vf_left < 0, fp_o, 0.0)
    fp_mom_out_L = jnp.where(vf_left[..., jnp.newaxis] < 0, fp_mom, 0.0)

    # When vf_right > 0, this cell is the donor at face j+1 (right-edge extraction)
    # fp at face j+1 was computed from this cell when is_pos[face j+1] = True
    fp_o_at_right_face = jnp.roll(fp_o, -1, axis=1)
    fp_mom_at_right_face = jnp.roll(fp_mom, -1, axis=1)
    fp_o_out_R = jnp.where(vf_right > 0, fp_o_at_right_face, 0.0)
    fp_mom_out_R = jnp.where(vf_right[..., jnp.newaxis] > 0, fp_mom_at_right_face, 0.0)

    # --- Step 3: Donor update ---
    sm_o_d, mom_d = _donor_update(
        sm_o, mom, fp_o_out_L, fp_mom_out_L, fp_o_out_R, fp_mom_out_R,
        alpha_out_L, alpha_out_R,
    )

    # Update volume after outgoing
    vol_d = vol - jnp.abs(jnp.where(vf_left < 0, vf_left, 0.0)) \
                - jnp.where(vf_right > 0, vf_right, 0.0)
    vol_d = jnp.maximum(vol_d, _EPS)

    # --- Step 4: Receiver merges ---
    # Incoming from LEFT (positive flux at left face)
    vol_in_L = jnp.maximum(vf_left, 0.0)
    has_in_L = vol_in_L > 0
    fp_o_in_L = jnp.where(has_in_L, fp_o, 0.0)
    fp_mom_in_L = jnp.where(has_in_L[..., jnp.newaxis], fp_mom, 0.0)

    sm_o_m1, mom_m1, vol_m1 = _receiver_merge(
        sm_o_d, mom_d, fp_o_in_L, fp_mom_in_L, vol_d, vol_in_L,
        from_left=True,
    )

    # Incoming from RIGHT (negative flux at right face)
    vol_in_R = jnp.maximum(-vf_right, 0.0)
    has_in_R = vol_in_R > 0
    fp_o_in_R = jnp.where(has_in_R, fp_o_at_right_face, 0.0)
    fp_mom_in_R = jnp.where(has_in_R[..., jnp.newaxis], fp_mom_at_right_face, 0.0)

    sm_o_new, mom_new, vol_new = _receiver_merge(
        sm_o_m1, mom_m1, fp_o_in_R, fp_mom_in_R, vol_m1, vol_in_R,
        from_left=False,
    )

    return sm_o_new, mom_new, vol_new


# =============================================================================
# Y-sweep (solid wall at poles)
# =============================================================================

def _som_y_sweep(sm_o, moments, vol_flux_y, vol):
    """SOM y-sweep with solid-wall boundary at j=0 and j=n_lat.

    Parameters
    ----------
    sm_o : (n_lat, n_lon, nlev)
    moments : (n_lat, n_lon, nlev, 9)
    vol_flux_y : (n_lat-1, n_lon, nlev)
        SIGNED volume transport at interior y-faces (n_lat-1 interior faces),
        positive = northward.  Face j sits between cell j (south) and cell j+1.
        Note: we use n_lat-1 interior faces; the wall faces at j=-1 and j=n_lat-1
        have zero flux.
    vol : (n_lat, n_lon, nlev)

    Returns
    -------
    sm_o_new, moments_new, vol_new
    """
    n_lat, n_lon, nlev = sm_o.shape

    # Permute moments: y→x so generic helpers work
    mom = _permute_moments(moments, "y")

    # Pad vol_flux_y with zeros at wall boundaries:
    # face -1 (south wall) and face n_lat-1 (north wall) are zero.
    # Interior faces: 0..n_lat-2 → vol_flux_y[j] between cell j and cell j+1.
    # Total faces including walls: n_lat+1, but we index as:
    #   wall_south (0), interior 0..n_lat-2, wall_north (n_lat-1)
    # For cell j: left face = face j-1, right face = face j
    # Using padded array: face index i between cell i and cell i+1 (0-based)

    # Pad with zero-flux walls.  Single Pad HLO op replaces alloc-zeros
    # + concatenate-of-three.
    vf_padded = jnp.pad(vol_flux_y, ((1, 1), (0, 0), (0, 0)))
    # vf_padded shape: (n_lat+1, n_lon, nlev)
    # vf_padded[0] = south wall (zero), vf_padded[n_lat] = north wall (zero)
    # vf_padded[j] = face between cell j-1 and cell j (for j=1..n_lat-1)

    # For cell j (j=0..n_lat-1):
    #   left face = vf_padded[j]     (face between cell j-1 and cell j)
    #   right face = vf_padded[j+1]  (face between cell j and cell j+1)
    vf_left = vf_padded[:-1, :, :]   # (n_lat, n_lon, nlev), left face of each cell
    vf_right = vf_padded[1:, :, :]   # (n_lat, n_lon, nlev), right face of each cell

    # --- Step 1: Compute face fluxes at all n_lat+1 faces ---
    # But wall faces have zero flux, so only compute interior faces.
    # Interior face j (1..n_lat-1) sits between cell j-1 and cell j.
    # vf_padded[j] for j=1..n_lat-1 → vol_flux_y[j-1]

    # For all interior faces (vf_padded[1:n_lat]):
    vf_int = vf_padded[1:n_lat, :, :]  # (n_lat-1, n_lon, nlev)

    # Donor for positive flow (northward): cell j-1 (south of face)
    sm_o_south = sm_o[:-1, :, :]   # cells 0..n_lat-2
    mom_south = mom[:-1, :, :, :]
    vol_south = vol[:-1, :, :]

    # Donor for negative flow (southward): cell j (north of face)
    sm_o_north = sm_o[1:, :, :]    # cells 1..n_lat-1
    mom_north = mom[1:, :, :, :]
    vol_north = vol[1:, :, :]

    is_pos = vf_int >= 0
    vol_donor = jnp.where(is_pos, vol_south, vol_north)
    alpha = jnp.minimum(jnp.abs(vf_int) / jnp.maximum(vol_donor, _EPS), 1.0)
    sign_edge = jnp.where(is_pos, 1.0, -1.0)
    sm_o_donor = jnp.where(is_pos, sm_o_south, sm_o_north)
    mom_donor = jnp.where(is_pos[..., jnp.newaxis], mom_south, mom_north)

    fp_o_int, fp_mom_int = _extract_flux(sm_o_donor, mom_donor, alpha, sign_edge)

    # Pad face fluxes with zeros at walls.  Single Pad HLO op each
    # replaces alloc-zeros + concatenate-of-three.
    fp_o_all = jnp.pad(fp_o_int, ((1, 1), (0, 0), (0, 0)))
    fp_mom_all = jnp.pad(fp_mom_int, ((1, 1), (0, 0), (0, 0), (0, 0)))
    alpha_all = jnp.pad(alpha, ((1, 1), (0, 0), (0, 0)))

    # --- Step 2: Per-cell incoming/outgoing ---
    # For cell j: left face = index j, right face = index j+1
    fp_o_at_left = fp_o_all[:-1, :, :]       # (n_lat, ...)
    fp_mom_at_left = fp_mom_all[:-1, :, :, :]
    fp_o_at_right = fp_o_all[1:, :, :]
    fp_mom_at_right = fp_mom_all[1:, :, :, :]

    # Outgoing fractions
    alpha_out_L = jnp.where(vf_left < 0,
                            jnp.minimum(-vf_left / jnp.maximum(vol, _EPS), 1.0),
                            0.0)
    alpha_out_R = jnp.where(vf_right > 0,
                            jnp.minimum(vf_right / jnp.maximum(vol, _EPS), 1.0),
                            0.0)

    fp_o_out_L = jnp.where(vf_left < 0, fp_o_at_left, 0.0)
    fp_mom_out_L = jnp.where(vf_left[..., jnp.newaxis] < 0, fp_mom_at_left, 0.0)
    fp_o_out_R = jnp.where(vf_right > 0, fp_o_at_right, 0.0)
    fp_mom_out_R = jnp.where(vf_right[..., jnp.newaxis] > 0, fp_mom_at_right, 0.0)

    # --- Step 3: Donor update ---
    sm_o_d, mom_d = _donor_update(
        sm_o, mom, fp_o_out_L, fp_mom_out_L, fp_o_out_R, fp_mom_out_R,
        alpha_out_L, alpha_out_R,
    )
    vol_d = vol - jnp.abs(jnp.where(vf_left < 0, vf_left, 0.0)) \
                - jnp.where(vf_right > 0, vf_right, 0.0)
    vol_d = jnp.maximum(vol_d, _EPS)

    # --- Step 4: Receiver merges ---
    vol_in_L = jnp.maximum(vf_left, 0.0)
    fp_o_in_L = jnp.where(vol_in_L > 0, fp_o_at_left, 0.0)
    fp_mom_in_L = jnp.where((vol_in_L > 0)[..., jnp.newaxis], fp_mom_at_left, 0.0)

    sm_o_m1, mom_m1, vol_m1 = _receiver_merge(
        sm_o_d, mom_d, fp_o_in_L, fp_mom_in_L, vol_d, vol_in_L,
        from_left=True,
    )

    vol_in_R = jnp.maximum(-vf_right, 0.0)
    fp_o_in_R = jnp.where(vol_in_R > 0, fp_o_at_right, 0.0)
    fp_mom_in_R = jnp.where((vol_in_R > 0)[..., jnp.newaxis], fp_mom_at_right, 0.0)

    sm_o_new, mom_new, vol_new = _receiver_merge(
        sm_o_m1, mom_m1, fp_o_in_R, fp_mom_in_R, vol_m1, vol_in_R,
        from_left=False,
    )

    # Unpermute moments back to canonical ordering
    mom_new = _unpermute_moments(mom_new, "y")
    return sm_o_new, mom_new, vol_new


# =============================================================================
# Z-sweep (closed surface and bottom)
# =============================================================================

def _som_z_sweep(sm_o, moments, vol_flux_z, vol):
    """SOM z-sweep with zero flux at surface and bottom.

    Parameters
    ----------
    sm_o : (n_lat, n_lon, nlev)
    moments : (n_lat, n_lon, nlev, 9)
    vol_flux_z : (n_lat, n_lon, nlev-1)
        SIGNED volume transport at interior z-faces, positive = upward.
        Face k sits between level k (above) and level k+1 (below).
        k=0..nlev-2 interior faces; surface (above k=0) and bottom
        (below k=nlev-1) are zero-flux.
    vol : (n_lat, n_lon, nlev)

    Returns
    -------
    sm_o_new, moments_new, vol_new
    """
    n_lat, n_lon, nlev = sm_o.shape

    # Permute moments: z→x
    mom = _permute_moments(moments, "z")

    # For z, "positive" = upward.  Level 0 = surface (top), level nlev-1 = bottom.
    # Face k between level k (above) and level k+1 (below).
    # Positive flux at face k: transport from level k+1 (below) to level k (above).
    # So for the SOM sweep in "x" convention:
    #   "left" = above (level k), "right" = below (level k+1)
    #   positive vf → flow from right to left → donor = below
    #
    # But our generic helpers assume positive = rightward (south→north for y, etc).
    # For z: we want positive = "downward" (increasing index) to match x-convention.
    # FLIP the sign so positive = downward:
    vf_int = -vol_flux_z  # (n_lat, n_lon, nlev-1), positive = downward

    # Pad with zero-flux boundaries at surface and bottom.
    # Single Pad HLO op replaces alloc-zeros + concatenate-of-three.
    vf_padded = jnp.pad(vf_int, ((0, 0), (0, 0), (1, 1)))
    # vf_padded[:, :, k] = face between level k-1 and level k (k=1..nlev-1 interior)
    # vf_padded[:, :, 0] = surface wall, vf_padded[:, :, nlev] = bottom wall

    # For level k: left face = vf_padded[:,:,k], right face = vf_padded[:,:,k+1]
    vf_left = vf_padded[:, :, :-1]   # (n_lat, n_lon, nlev)
    vf_right = vf_padded[:, :, 1:]   # (n_lat, n_lon, nlev)

    # --- Step 1: Face fluxes at interior faces ---
    # Interior face k (1..nlev-1) between level k-1 (left) and level k (right)
    # Donor for positive (downward) flow: level k-1 (above)
    sm_o_above = sm_o[:, :, :-1]
    mom_above = mom[:, :, :-1, :]
    vol_above = vol[:, :, :-1]

    sm_o_below = sm_o[:, :, 1:]
    mom_below = mom[:, :, 1:, :]
    vol_below = vol[:, :, 1:]

    is_pos = vf_int >= 0  # positive = downward in flipped convention
    vol_donor = jnp.where(is_pos, vol_above, vol_below)
    alpha = jnp.minimum(jnp.abs(vf_int) / jnp.maximum(vol_donor, _EPS), 1.0)
    sign_edge = jnp.where(is_pos, 1.0, -1.0)
    sm_o_donor = jnp.where(is_pos, sm_o_above, sm_o_below)
    mom_donor = jnp.where(is_pos[..., jnp.newaxis], mom_above, mom_below)

    fp_o_int, fp_mom_int = _extract_flux(sm_o_donor, mom_donor, alpha, sign_edge)

    # Pad with zeros at surface/bottom walls.  Single Pad HLO op each.
    fp_o_all = jnp.pad(fp_o_int, ((0, 0), (0, 0), (1, 1)))
    fp_mom_all = jnp.pad(fp_mom_int, ((0, 0), (0, 0), (1, 1), (0, 0)))

    fp_o_at_left = fp_o_all[:, :, :-1]
    fp_mom_at_left = fp_mom_all[:, :, :-1, :]
    fp_o_at_right = fp_o_all[:, :, 1:]
    fp_mom_at_right = fp_mom_all[:, :, 1:, :]

    # --- Step 2-4: Same pattern as y-sweep ---
    alpha_out_L = jnp.where(vf_left < 0,
                            jnp.minimum(-vf_left / jnp.maximum(vol, _EPS), 1.0),
                            0.0)
    alpha_out_R = jnp.where(vf_right > 0,
                            jnp.minimum(vf_right / jnp.maximum(vol, _EPS), 1.0),
                            0.0)

    fp_o_out_L = jnp.where(vf_left < 0, fp_o_at_left, 0.0)
    fp_mom_out_L = jnp.where(vf_left[..., jnp.newaxis] < 0, fp_mom_at_left, 0.0)
    fp_o_out_R = jnp.where(vf_right > 0, fp_o_at_right, 0.0)
    fp_mom_out_R = jnp.where(vf_right[..., jnp.newaxis] > 0, fp_mom_at_right, 0.0)

    sm_o_d, mom_d = _donor_update(
        sm_o, mom, fp_o_out_L, fp_mom_out_L, fp_o_out_R, fp_mom_out_R,
        alpha_out_L, alpha_out_R,
    )
    vol_d = vol - jnp.abs(jnp.where(vf_left < 0, vf_left, 0.0)) \
                - jnp.where(vf_right > 0, vf_right, 0.0)
    vol_d = jnp.maximum(vol_d, _EPS)

    vol_in_L = jnp.maximum(vf_left, 0.0)
    fp_o_in_L = jnp.where(vol_in_L > 0, fp_o_at_left, 0.0)
    fp_mom_in_L = jnp.where((vol_in_L > 0)[..., jnp.newaxis], fp_mom_at_left, 0.0)

    sm_o_m1, mom_m1, vol_m1 = _receiver_merge(
        sm_o_d, mom_d, fp_o_in_L, fp_mom_in_L, vol_d, vol_in_L,
        from_left=True,
    )

    vol_in_R = jnp.maximum(-vf_right, 0.0)
    fp_o_in_R = jnp.where(vol_in_R > 0, fp_o_at_right, 0.0)
    fp_mom_in_R = jnp.where((vol_in_R > 0)[..., jnp.newaxis], fp_mom_at_right, 0.0)

    sm_o_new, mom_new, vol_new = _receiver_merge(
        sm_o_m1, mom_m1, fp_o_in_R, fp_mom_in_R, vol_m1, vol_in_R,
        from_left=False,
    )

    mom_new = _unpermute_moments(mom_new, "z")
    return sm_o_new, mom_new, vol_new


# =============================================================================
# Public orchestrator
# =============================================================================

def som_advect_tracers(
    tracer: jnp.ndarray,
    moments: jnp.ndarray,
    mass_flux_u: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
    w_half: jnp.ndarray,
    h_k_old: jnp.ndarray,
    h_k_new: jnp.ndarray,
    grid: LatLonGrid,
    dt: float,
    land_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """SOM (Prather 1986) tracer advection with directional splitting.

    Performs sequential x→y→z sweeps.  Each sweep advects the full
    polynomial distribution (mean + 9 moments), preserving sub-cell
    structure and achieving near-zero spurious diapycnal mixing.

    Parameters
    ----------
    tracer : (n_lat, n_lon, nlev)
        Cell-mean tracer (e.g. temperature or salinity).
    moments : (n_lat, n_lon, nlev, 9)
        Higher-order SOM moments.
    mass_flux_u : (n_lat, n_lon+1, nlev)
        Thickness-weighted zonal velocity at u-faces [m²/s].
    mass_flux_v : (n_lat+1, n_lon, nlev)
        Thickness-weighted meridional velocity at v-faces [m²/s].
    w_half : (n_lat, n_lon, nlev+1)
        Vertical velocity at half-levels [m/s], positive = upward.
    h_k_old : (n_lat, n_lon, nlev)
        Layer thickness before the step.
    h_k_new : (n_lat, n_lon, nlev)
        Layer thickness after the step (from continuity).
    grid : LatLonGrid
    dt : float
        Timestep [s].
    land_mask : (n_lat, n_lon)
        1 = ocean, 0 = land.

    Returns
    -------
    tracer_new : (n_lat, n_lon, nlev)
    moments_new : (n_lat, n_lon, nlev, 9)
    """
    area = grid.area[..., jnp.newaxis]  # (n_lat, n_lon, 1)
    vol = h_k_old * area  # (n_lat, n_lon, nlev)

    # Convert to volume-weighted mean
    sm_o = vol * tracer

    # --- Reconcile moments with physics-modified tracer ---
    # Between steps, physics tendencies (K_v diffusion, KPP, GM/Redi)
    # modify the cell-mean tracer but NOT the SOM moments.  The moments
    # describe the sub-cell distribution of the PREVIOUS step's tracer.
    # If sm_o changed significantly, the moments can be out of proportion
    # and the extraction formula fp_o = alpha*(sm_o + alp1*sx + ...) can
    # overshoot, draining more tracer than available.
    #
    # Fix: scale moments so that the ratio |moments|/|sm_o| doesn't grow
    # from the physics step.  This is equivalent to MITgcm's approach of
    # applying diffusion to sm_o directly (which preserves moment ratios).
    moments = _limit_moments(sm_o, moments)

    # --- Volume transports ---
    # X: mass_flux_u is h*u at u-faces [m²/s].  Volume transport = h*u * dy_face * dt.
    # dy at u-face = R * dlat (uniform in latitude for regular grids)
    face_dy = grid.radius * grid.dlat
    # Use interior faces only (periodic: n_lon faces)
    vol_flux_x = mass_flux_u[:, :grid.n_lon, :] * face_dy * dt

    # Y: mass_flux_v is h*v at v-faces [m²/s].  Volume transport = h*v * dx_face * dt.
    # dx at v-face j = R * dlon * cos(lat_v_j)
    # v-faces: n_lat+1 total, interior = 1..n_lat-1
    # Use cos(average latitude) to match the divergence operator exactly
    # (divergence_cgrid uses cos(0.5*(lat[i]+lat[i+1])), not avg(cos)).
    # cos(±π/2) ≈ 0 analytically; build cos_lat_v directly via Pad of
    # cos(lat_interior) (single Pad HLO op vs alloc-2-singletons +
    # concatenate-of-three + cos tower).
    lat = grid.lat  # (n_lat,)
    lat_interior_v = 0.5 * (lat[:-1] + lat[1:])
    cos_lat_v = jnp.pad(jnp.cos(lat_interior_v), (1, 1))  # (n_lat+1,)
    face_dx_v = grid.radius * grid.dlon * cos_lat_v  # (n_lat+1,)
    vol_flux_v_all = mass_flux_v * face_dx_v[:, jnp.newaxis, jnp.newaxis] * dt
    # Interior v-faces only (1..n_lat-1), excluding wall boundaries
    vol_flux_y = vol_flux_v_all[1:-1, :, :]  # (n_lat-1, n_lon, nlev)

    # Z: w_half at half-levels [m/s].  Volume transport = w * area * dt.
    # Interior half-levels: 1..nlev-1 (between level k-1 and level k)
    vol_flux_z = w_half[:, :, 1:-1] * area * dt  # (n_lat, n_lon, nlev-1), positive = upward

    # --- Sequential sweeps with pre-sweep limiting ---
    # Apply the moment limiter before EACH sweep (not just after all three).
    # This prevents corrupted moments from one sweep propagating into the
    # next, matching MITgcm's approach (GAD_SOM_LIM_R.F called per sweep).
    # Note: the reconciliation limiter above already limited before the
    # first sweep, so the x-sweep limiter below is for the y/z sweeps.
    sm_o, mom, vol_after_x = _som_x_sweep(sm_o, moments, vol_flux_x, vol)

    mom = _limit_moments(sm_o, mom)
    sm_o, mom, vol_after_xy = _som_y_sweep(sm_o, mom, vol_flux_y, vol_after_x)

    mom = _limit_moments(sm_o, mom)
    sm_o, mom, vol_after_xyz = _som_z_sweep(sm_o, mom, vol_flux_z, vol_after_xy)

    # Final limiter after all sweeps
    mom = _limit_moments(sm_o, mom)

    # Recover tracer from volume-weighted mean using h_k_new for
    # consistency with the rest of the model (continuity equation).
    vol_new = h_k_new * area
    tracer_new = sm_o / jnp.maximum(vol_new, _EPS)

    # Zero out land cells
    mask_3d = land_mask[..., jnp.newaxis]
    tracer_new = jnp.where(mask_3d > 0.5, tracer_new, tracer)
    mom = jnp.where(mask_3d[..., jnp.newaxis] > 0.5, mom, 0.0)

    return tracer_new, mom
