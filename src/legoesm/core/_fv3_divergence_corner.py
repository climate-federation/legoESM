"""Faithful port of FV3 ``divergence_corner`` from ``sw_core.F90:2124``.

Computes B-grid corner divergence with the FV3-specific edge stencils
and cube-vertex corner-removal terms.  This is the exact arithmetic
used by GFDL FV3 in ``d_sw5`` (sw_core.F90:1641-1719) before the
adaptive Smagorinsky-style damping is applied to the kinetic-energy
field.

Why this exists
---------------

Our 3D atmospheric path computes divergence at CELL CENTRES via
``cgrid_divergence`` (a simple flux-form ``net_x + net_y / area``)
that does NOT include the FV3-specific:

* sin_sg metric averaging at the j==1 / j==npy boundary rows (which
  picks up the sub-grid-cell metric at panel boundaries instead of
  the cell-edge average);

* cosa-cross-correction ``- 0.25*(va_W+va_E)*(cos_sg_W + cos_sg_E)``
  for the non-orthogonal cubed-sphere grid (the contravariant
  velocity correction);

* sw/se/ne/nw corner removal terms at the 4 cube vertices that
  subtract the spurious 4th-cell contribution where only 3 faces
  meet.

These three FV3 mechanisms give the corner-staggered ``divg_d`` its
faithful magnitude at panel-boundary halo cells — the same cells that
drive the cube-imprint feedback diagnosed in iter-2.

This module provides:

- :func:`fv3_divergence_corner_2d`: the 2D port, faithful to FV3
  ``sw_core.F90:2124-2229`` (non-bounded-domain, ``grid_type < 3``
  branch — i.e., the full sin_sg + cosa-correction + corner-removal
  formulation).

- :func:`fv3_divergence_corner_3d`: 3D wrapper that vmaps over the
  trailing level axis.

The returned ``divg_d`` is at B-grid corners with shape
``(6, n+1, n+1[, nlev])`` — ready to be used by an adaptive damping
that targets cube-vertex amplification.

Reference
---------

GFDL FV3 ``atmos_cubed_sphere-symmetryclean/model/sw_core.F90``
lines 2124-2229 (subroutine ``divergence_corner``).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.grids.halo import pad_halo


def _to_fv3_normal_dgrid_2d(
    u_corner: jnp.ndarray, v_corner: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert C-D corner winds (n+1, n+1) to FV3 normal D-grid layout.

    FV3's ``divergence_corner`` expects:

    - ``u`` shape ``(isd:ied, jsd:jed+1)`` — x-velocity at v-interface
      midpoints, our convention ``(6, n, n+1)``.
    - ``v`` shape ``(isd:ied+1, jsd:jed)`` — y-velocity at u-interface
      midpoints, our convention ``(6, n+1, n)``.

    Our C-D layout has both at corners ``(6, n+1, n+1)``.  The
    FV3 v-interface midpoint at ``(i+0.5, j)`` corresponds to the
    average of corners ``(i, j)`` and ``(i+1, j)``.

    Parameters
    ----------
    u_corner : (6, n+1, n+1) — C-D u_d at corners.
    v_corner : (6, n+1, n+1) — C-D v_d at corners.

    Returns
    -------
    u_fv3 : (6, n, n+1) — FV3 normal D-grid u.
    v_fv3 : (6, n+1, n) — FV3 normal D-grid v.
    """
    u_fv3 = 0.5 * (u_corner[:, :-1, :] + u_corner[:, 1:, :])
    v_fv3 = 0.5 * (v_corner[:, :, :-1] + v_corner[:, :, 1:])
    return u_fv3, v_fv3


def _to_agrid_cell_centre_2d(
    u_corner: jnp.ndarray, v_corner: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert C-D corner winds to A-grid cell-centre (ua, va) for FV3.

    A-grid cell-centre value at (i, j) = average of the 4 surrounding
    corner values.  This is exactly ``_interp_corner_to_center``.
    """
    ua = 0.25 * (
        u_corner[:, :-1, :-1] + u_corner[:, 1:, :-1]
        + u_corner[:, :-1, 1:] + u_corner[:, 1:, 1:]
    )
    va = 0.25 * (
        v_corner[:, :-1, :-1] + v_corner[:, 1:, :-1]
        + v_corner[:, :-1, 1:] + v_corner[:, 1:, 1:]
    )
    return ua, va


def fv3_divergence_corner_2d(
    u_corner: jnp.ndarray,
    v_corner: jnp.ndarray,
    cdgrid: CubedSphereCDGrid,
) -> jnp.ndarray:
    """Faithful port of FV3 ``sw_core.F90:divergence_corner`` (2D).

    Direct port of the non-bounded-domain, ``grid_type < 3`` branch
    (lines 2181-2225 in the Fortran source).  Computes B-grid corner
    divergence using the FV3 sin_sg edge metrics, cosa cross-
    correction for non-orthogonality, and the four cube-vertex
    corner-removal terms.

    Parameters
    ----------
    u_corner : (6, n+1, n+1) — C-D u_d at D-grid corners.
    v_corner : (6, n+1, n+1) — C-D v_d at D-grid corners.
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    divg_d : (6, n+1, n+1) — corner-staggered B-grid divergence
        with the FV3-faithful edge metrics and corner-removal terms.

    Notes
    -----
    - ``sin_sg[..., 1]`` (Fortran ``sin_sg(i,j,2)``) is the south-edge
      metric at cell (i, j); ``[..., 3]`` is the north-edge metric;
      ``[..., 0]`` is the west-edge; ``[..., 2]`` is the east-edge.
      We map FV3's 1-based index to our 0-based at the call site.

    - The cosa cross-correction reads ``va`` at cell (i, j-1) and
      (i, j) — both A-grid cell-centre values.  Periodic-aware halo
      padding via ``pad_halo`` is required so that the j-1 read at
      j=0 picks up the south-neighbour-panel cell.

    - Corner-removal: at the 4 cube vertices (sw, se, ne, nw of each
      face), one of the four cells in the divergence stencil
      doesn't physically exist.  FV3 subtracts that cell's
      contribution from the raw difference, giving the correct
      3-cell convergence at the cube vertex.
    """
    # Step 1: bring inputs into FV3 normal D-grid + A-grid layouts.
    u_fv3, v_fv3 = _to_fv3_normal_dgrid_2d(u_corner, v_corner)  # (6, n, n+1), (6, n+1, n)
    ua, va = _to_agrid_cell_centre_2d(u_corner, v_corner)        # (6, n, n)

    n = cdgrid.n

    # Step 2: pad ua, va with halo=1 so the cosa cross-correction at
    # j-1 / i-1 reads neighbour-panel cells (cross-face values).
    ua_pad = pad_halo(ua)   # (6, n+2, n+2)
    va_pad = pad_halo(va)

    # Step 3: pad u_fv3, v_fv3 with halo=1 in the cell-axis.  u_fv3
    # has axis-1 length n (cells), axis-2 length n+1 (faces); we need
    # u_fv3 at i ∈ [-1, n+1] for the corner stencil.  Pad axis 1 by 1
    # via mode='edge' (halo handled by sin_sg pad below).
    # For our purposes the FV3 formula reads u(is-1:ie+1, js:je+1) which
    # in 0-based is u(0:n+1, 0:n+1) — exactly the (n+2, n+1) range.
    # u_fv3 (n, n+1) padded to (n+2, n+1) covers this.
    u_fv3_pad = jnp.pad(
        u_fv3, [(0, 0), (1, 1), (0, 0)], mode="edge",
    )                                                # (6, n+2, n+1)
    v_fv3_pad = jnp.pad(
        v_fv3, [(0, 0), (0, 0), (1, 1)], mode="edge",
    )                                                # (6, n+1, n+2)

    # Step 4: extract sin_sg / cos_sg at the 4 sub-grid positions used.
    # FV3 indexing convention (0-based here): sg[0]=west, sg[1]=south,
    # sg[2]=east, sg[3]=north.  See cubed_sphere_cdgrid.py:_compute_sin_cos_sg
    # for the derivation.  The Fortran 1-based ``(i, j, 2)`` → south
    # → our index 1; ``(i, j, 4)`` → north → our index 3.
    sin_s = cdgrid.sin_sg[..., 1]     # south edge sin
    sin_n = cdgrid.sin_sg[..., 3]     # north edge sin
    sin_w = cdgrid.sin_sg[..., 0]     # west edge sin
    sin_e = cdgrid.sin_sg[..., 2]     # east edge sin
    cos_s = cdgrid.cos_sg[..., 1]
    cos_n = cdgrid.cos_sg[..., 3]
    cos_w = cdgrid.cos_sg[..., 0]
    cos_e = cdgrid.cos_sg[..., 2]

    sin_s_pad = pad_halo(sin_s)
    sin_n_pad = pad_halo(sin_n)
    sin_w_pad = pad_halo(sin_w)
    sin_e_pad = pad_halo(sin_e)
    cos_s_pad = pad_halo(cos_s)
    cos_n_pad = pad_halo(cos_n)
    cos_w_pad = pad_halo(cos_w)
    cos_e_pad = pad_halo(cos_e)

    # Step 5: ``uf`` at v-interface (i, j) for i ∈ [0, n], j ∈ [0, n].
    # j == 0 (FV3 j==1) and j == n (FV3 j==npy): boundary rows use the
    # simpler sin-only formula.  Other j: full cosa-corrected formula.
    #
    # FV3 line 2190 (j==1 or j==npy):
    #   uf(i,j) = u(i,j) * dyc(i,j) * 0.5*(sin_sg(i,j-1,4)+sin_sg(i,j,2))
    # FV3 line 2194 (interior j):
    #   uf(i,j) = (u(i,j) - 0.25*(va(i,j-1)+va(i,j)) *
    #              (cos_sg(i,j-1,4) + cos_sg(i,j,2)))
    #             * dyc(i,j) * 0.5*(sin_sg(i,j-1,4)+sin_sg(i,j,2))
    #
    # In our 0-based arrays:
    #  - u_fv3_pad[:, i+1, j] is u_FV3(i, j) — added 1 to i for the pad.
    #    Wait, u_fv3 has shape (n, n+1) → axis 1 is i ∈ [0, n-1].
    #    After pad axis 1 to (n+2,): index 0 = west halo, 1..n = interior.
    #    But the Fortran reads u(i, j) for i ∈ [is-1, ie+1] = [-1, n+1] in
    #    0-based, which is the full padded range [0, n+1].
    #
    # Simpler: the divergence_corner output is at j ∈ [0, n] (n+1 corners).
    # Build uf at i ∈ [0, n], j ∈ [0, n] (also n+1 in i).  uf shape: (n+1, n+1).
    # j_face = 0..n; cell-i for sin_sg lookups: i_cell = i_face but we need
    # cell (i, j-1) and cell (i, j) where j is the FACE index — FV3 uses
    # j-1 → south cell, j → north cell of the v-edge at j-face.
    #
    # The FV3 ``uf`` is actually at v-interfaces (j-faces), shape (i, j_face).
    # For our 0-based j_face ∈ [0, n], "south cell" (j-1 in FV3) is cell
    # j_face-1 with index j_face_pad-1 in the halo'd array.
    #
    # For the 6 face dimension we vectorise via leading axis.

    # Build sin/cos at v-edges via south + north cell averages.
    # Padded south cell at j_face = j-1 in FV3 → padded j (when j_face=0).
    # Padded north cell at j_face = j in FV3 → padded j+1 (when j_face=0).
    # In 0-based with halo padded shape (n+2, n+2): cell (i, j_face-1) →
    # padded[i+1, j_face]; cell (i, j_face) → padded[i+1, j_face+1].
    # Wait — `j-1` in FV3 corresponds to "the cell to the south of the
    # j-face".  In our convention, the j-face at index j_face sits BETWEEN
    # cells at row j_face-1 (south) and row j_face (north).  Padded
    # indexing: cell j_face-1 = padded[1+j_face-1] = padded[j_face]; cell
    # j_face = padded[j_face+1].
    # So at j_face = 0: south cell padded index is 0 (the south halo);
    # north cell padded index is 1 (first interior).
    # At j_face = n: south = padded[n] (last interior); north = padded[n+1]
    # (north halo).

    # j_face range: 0..n (n+1 values).  i_face range: 0..n (n+1 values).
    # Use slicing [j_face_idx_start: j_face_idx_end] = [0: n+1].
    # For interior i: cell at column i_cell, padded index i_cell+1.
    # uf(i_face, j_face) reads u_fv3_pad[..., i_face?, j_face? ]
    #
    # u_FV3 at v-interface (i, j_face) — i is which COLUMN of cells you're
    # between, j_face is the face index in j.  u shape (n, n+1) means
    # i ∈ [0, n-1], j_face ∈ [0, n].  Halo-padded to (n+2, n+1) (axis 1 padding):
    # u_fv3_pad[i_axis_1, j_face] for i_axis_1 ∈ [0, n+1].  Map FV3's i
    # cell index to padded: i_cell padded = i_cell + 1.
    #
    # Oh wait: u(i, j) in FV3 with shape (isd:ied, jsd:jed+1) at i=is-1
    # means halo cell.  But it's a v-interface NOT a cell — there's a
    # subtle distinction.  Following the existing legoESM SW path
    # (fv3_d_sw5_corner_divergence.py uses u_d shape (6, n, n+1)
    # without padding because it computes interior corners only).
    #
    # For a faithful divergence_corner port that includes BOUNDARY corners
    # (i_face = 0 or n, j_face = 0 or n), we need u_fv3 at i = -1 (halo).
    # u_fv3_pad provides that via ``mode='edge'`` — same-face extension.

    # Build south/north cell indices in the halo'd cell-array (n+2 axis)
    # for j_face = 0..n:
    #   south_cell_pad = j_face       (j_face=0 → padded[0] = south halo)
    #   north_cell_pad = j_face + 1   (j_face=n → padded[n+1] = north halo)
    # Use slicing windows.

    # Interpolate sin/cos at v-edges (j-faces) via 0.5 * (south + north).
    sin_s_at_j1 = sin_n_pad[:, 1:n + 1, 0:n + 1]   # cell j-1 at j-face j: padded col j
    sin_n_at_j  = sin_s_pad[:, 1:n + 1, 1:n + 2]   # cell j   at j-face j: padded col j+1
    sin_v_face = 0.5 * (sin_s_at_j1 + sin_n_at_j)  # (6, n, n+1)
    cos_s_at_j1 = cos_n_pad[:, 1:n + 1, 0:n + 1]
    cos_n_at_j  = cos_s_pad[:, 1:n + 1, 1:n + 2]
    cos_v_face = 0.5 * (cos_s_at_j1 + cos_n_at_j)  # (6, n, n+1)

    # va at j-face (south + north cell, axis 2 is j).
    va_s = va_pad[:, 1:n + 1, 0:n + 1]   # (6, n, n+1) cell j-1
    va_n = va_pad[:, 1:n + 1, 1:n + 2]   # (6, n, n+1) cell j
    va_at_j_face = 0.5 * (va_s + va_n)   # (6, n, n+1) — this is 0.25*(va(j-1)+va(j))*2

    # j-face mask: True where j == 0 or j == n (boundary), False elsewhere.
    # j_face indices 0..n.
    j_face_idx = jnp.arange(n + 1)
    is_boundary_j = (j_face_idx == 0) | (j_face_idx == n)        # (n+1,)
    boundary_j_3d = is_boundary_j[None, None, :]                  # (1, 1, n+1)

    # Now compute uf.  But note FV3's uf has shape (i_face, j_face) with
    # i_face ∈ [is-1, ie+1] (so halo+1).  Our u_fv3 (n, n+1) in axis 1
    # is cell-i.  We need uf at i_face, but actually FV3's uf is at
    # i_cell — confusingly named.  Reading the formula again:
    #   uf(i, j) = u(i, j) * stuff
    # where u has shape (isd:ied, jsd:jed+1).  So u(i, j) at i ∈ [-1, n+1]
    # means cell-i halo coverage.  uf has shape (is-2:ie+2, js-1:je+2)
    # so uf is at cell-i, j-face.
    #
    # For divg_d(i_face, j_face) = vf(j-1) - vf(j) + uf(i-1) - uf(i) the
    # uf indices are i-1 and i where i = i_face.  So uf is at cell-i =
    # i_face-1 and i_face.  The output divg_d at i_face uses uf at
    # i_face-1 (left cell) and i_face (right cell).
    #
    # In our convention, uf is at cell-i, j-face → shape (n, n+1) BEFORE
    # halo extension.  Pad axis 1 to (n+2, n+1) for the i_face-1 access.

    # u_fv3 at v-interface (cell-i, j-face) — directly available, but we
    # have it after pad as u_fv3_pad shape (6, n+2, n+1).  Axis 1 padded
    # index 0 = west halo, n+1 = east halo, 1..n = interior cells 0..n-1.

    # Slice u at all relevant (cell-i, j-face) for the boundary-aware
    # formula.  Use the full (n+2, n+1) range — interior cell-i ∈ [0, n-1]
    # in 0-based maps to padded index 1..n; halo at 0 and n+1.

    # Build uf with both branches and select via boundary mask.
    u_full = u_fv3_pad[:, :, :]                   # (6, n+2, n+1)
    # va at v-interfaces in cell-i coords: va_at_j_face shape (6, n, n+1).
    # Pad axis 1 (cell-i) to (n+2, n+1).
    va_at_jface_pad = jnp.pad(
        va_at_j_face, [(0, 0), (1, 1), (0, 0)], mode="edge",
    )
    cos_v_pad = jnp.pad(
        cos_v_face, [(0, 0), (1, 1), (0, 0)], mode="edge",
    )
    sin_v_pad = jnp.pad(
        sin_v_face, [(0, 0), (1, 1), (0, 0)], mode="edge",
    )

    # FV3's dyc has shape (6, n, n+1) per ``cdgrid.dyc`` definition.
    # Pad cell-i to (n+2, n+1) via mode='edge'.
    dyc_pad = jnp.pad(cdgrid.dyc, [(0, 0), (1, 1), (0, 0)], mode="edge")

    # Boundary uf:  uf = u * dyc * sin_v_face
    # Interior uf:  uf = (u - 0.5*va_at_j_face*(2*cos_v_face)) * dyc * sin_v_face
    # FV3's 0.25*(va_W+va_E)*(cos_W+cos_E) = 0.5*va_avg*2*cos_avg = va_avg * 2*cos_avg.
    # Wait, va_at_j_face = 0.5*(va_W + va_E) so 0.25*(va_W+va_E) = 0.5*va_at_j_face.
    # Likewise (cos_W + cos_E) = 2*cos_v_face.  So the cross term = 0.5*va_at_j_face * 2*cos_v_face = va_at_j_face * cos_v_face.
    #
    # No wait — Fortran: 0.25 * (va(i,j-1)+va(i,j)) * (cos_sg(i,j-1,4)+cos_sg(i,j,2))
    # = 0.25 * 2*va_avg * 2*cos_avg = va_avg * cos_avg.
    # Where va_avg = 0.5*(va(j-1)+va(j)) and cos_avg = 0.5*(cos(j-1)+cos(j)).
    # So Fortran cross term = va_avg * cos_avg.

    cross = va_at_jface_pad * cos_v_pad           # (6, n+2, n+1)
    uf_interior = (u_full - cross) * dyc_pad * sin_v_pad
    uf_boundary = u_full * dyc_pad * sin_v_pad
    uf = jnp.where(boundary_j_3d, uf_boundary, uf_interior)  # (6, n+2, n+1)

    # Same construction for vf.  FV3 lines 2200-2207:
    #   vf(i,j) = (v(i,j) - 0.25*(ua(i-1,j)+ua(i,j))*(cos_sg(i-1,j,3)+cos_sg(i,j,1))) * dxc(i,j) * 0.5*(sin_sg(i-1,j,3)+sin_sg(i,j,1))
    # at non-edge i (i_face != 0, npx), and a boundary form at i==1 or i==npx.
    # Interpolate ua at u-edges (i-faces) using west + east cell.
    ua_w = ua_pad[:, 0:n + 1, 1:n + 1]            # (6, n+1, n) cell i-1
    ua_e = ua_pad[:, 1:n + 2, 1:n + 1]            # (6, n+1, n) cell i
    ua_at_iface = 0.5 * (ua_w + ua_e)              # (6, n+1, n)

    # cos_sg at i-face: 0.5 * (east_of_i-1, west_of_i).
    cos_e_w = cos_e_pad[:, 0:n + 1, 1:n + 1]      # cell i-1, east edge
    cos_w_e = cos_w_pad[:, 1:n + 2, 1:n + 1]      # cell i, west edge
    cos_u_face = 0.5 * (cos_e_w + cos_w_e)        # (6, n+1, n)
    sin_e_w = sin_e_pad[:, 0:n + 1, 1:n + 1]
    sin_w_e = sin_w_pad[:, 1:n + 2, 1:n + 1]
    sin_u_face = 0.5 * (sin_e_w + sin_w_e)        # (6, n+1, n)

    # Pad to (n+1, n+2) for the cell-j halo on j-axis.
    cos_u_pad = jnp.pad(
        cos_u_face, [(0, 0), (0, 0), (1, 1)], mode="edge",
    )
    sin_u_pad = jnp.pad(
        sin_u_face, [(0, 0), (0, 0), (1, 1)], mode="edge",
    )
    ua_at_iface_pad = jnp.pad(
        ua_at_iface, [(0, 0), (0, 0), (1, 1)], mode="edge",
    )
    dxc_pad = jnp.pad(cdgrid.dxc, [(0, 0), (0, 0), (1, 1)], mode="edge")

    # i-face boundary mask: i_face = 0 or n.
    i_face_idx = jnp.arange(n + 1)
    is_boundary_i = (i_face_idx == 0) | (i_face_idx == n)
    boundary_i_3d = is_boundary_i[None, :, None]

    cross_v = ua_at_iface_pad * cos_u_pad
    vf_full = v_fv3_pad                            # (6, n+1, n+2) already
    vf_interior = (vf_full - cross_v) * dxc_pad * sin_u_pad
    vf_boundary = vf_full * dxc_pad * sin_u_pad
    vf = jnp.where(boundary_i_3d, vf_boundary, vf_interior)  # (6, n+1, n+2)

    # Step 6: assemble divg_d at corners (n+1, n+1).  FV3 line 2211:
    #   divg_d(i, j) = vf(i, j-1) - vf(i, j) + uf(i-1, j) - uf(i, j)
    # In our notation: divg_d at corner (i_face, j_face).
    #   uf at (cell_i, j_face) — for divg_d at i_face we use cell_i = i_face-1 (left)
    #   and cell_i = i_face (right).
    #   In padded indexing: padded_i = cell_i + 1, so left = i_face, right = i_face+1.
    #   vf at (i_face, cell_j) — for divg_d at j_face we use cell_j = j_face-1
    #   (south) and cell_j = j_face (north).  Padded j: south = j_face,
    #   north = j_face+1.

    uf_left  = uf[:, 0:n + 1, :]      # cell_i = i_face-1 → padded i_face
    uf_right = uf[:, 1:n + 2, :]      # cell_i = i_face   → padded i_face+1
    vf_south = vf[:, :, 0:n + 1]      # cell_j = j_face-1 → padded j_face
    vf_north = vf[:, :, 1:n + 2]      # cell_j = j_face   → padded j_face+1

    divg_d_raw = vf_south - vf_north + uf_left - uf_right        # (6, n+1, n+1)

    # Step 7: corner-removal at the 4 cube vertices of each face.  FV3
    # lines 2216-2219:
    #   if (sw_corner) divg_d(1, 1)     -= vf(1, 0)         ! SW
    #   if (se_corner) divg_d(npx, 1)   -= vf(npx, 0)       ! SE
    #   if (ne_corner) divg_d(npx, npy) += vf(npx, npy)     ! NE
    #   if (nw_corner) divg_d(1, npy)   += vf(1, npy)       ! NW
    # In our 0-based: sw=(0, 0), se=(n, 0), ne=(n, n), nw=(0, n).  vf at the
    # corresponding (i_face, padded j) = (0, 0), (n, 0), (n, n+1), (0, n+1).
    # All face cube vertices are corner cells in our (6, n+1, n+1)
    # representation (every face has all four).
    sw = vf[:, 0, 0]                        # (6,)
    se = vf[:, n, 0]
    ne = vf[:, n, n + 1]
    nw = vf[:, 0, n + 1]

    divg_d = divg_d_raw
    divg_d = divg_d.at[:, 0, 0].add(-sw)
    divg_d = divg_d.at[:, n, 0].add(-se)
    divg_d = divg_d.at[:, n, n].add(ne)
    divg_d = divg_d.at[:, 0, n].add(nw)

    # Step 8: divide by area_corner (Fortran rarea_c).
    rarea_c = cdgrid.rarea_c                    # (6, n+1, n+1)
    return divg_d * rarea_c


def fv3_divergence_corner_3d(
    u_corner_3d: jnp.ndarray,
    v_corner_3d: jnp.ndarray,
    cdgrid: CubedSphereCDGrid,
) -> jnp.ndarray:
    """3D wrapper that vmaps :func:`fv3_divergence_corner_2d` over levels.

    Parameters
    ----------
    u_corner_3d : (6, n+1, n+1, nlev)
    v_corner_3d : (6, n+1, n+1, nlev)
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    divg_d_3d : (6, n+1, n+1, nlev)
    """
    u_t = jnp.moveaxis(u_corner_3d, -1, 0)   # (nlev, 6, n+1, n+1)
    v_t = jnp.moveaxis(v_corner_3d, -1, 0)
    divg_t = jax.vmap(
        lambda args: fv3_divergence_corner_2d(args[0], args[1], cdgrid),
    )((u_t, v_t))
    return jnp.moveaxis(divg_t, 0, -1)


def fv3_corner_laplacian_iteration(
    divg_d: jnp.ndarray,
    cdgrid: CubedSphereCDGrid,
    apply_vector_corner_fill: bool = False,
) -> jnp.ndarray:
    """Apply one Laplacian iteration on a B-grid corner-staggered field.

    Faithful port of FV3 ``sw_core.F90:1746-1782`` inner block of the
    higher-order ``nord > 0`` divergence-damping path.  One iteration
    (``do n=1, nord`` body) on the corner-staggered ``divg_d`` field.

    The Fortran sequence (sw_core.F90:1748-1785, 1-based)::

        do j=js-nt,je+1+nt
           do i=is-1-nt,ie+1+nt
              vc(i,j) = (divg_d(i+1,j)-divg_d(i,j))*divg_u(i,j)   ! x-flux
           enddo
        enddo
        do j=js-1-nt,je+1+nt
           do i=is-nt,ie+1+nt
              uc(i,j) = (divg_d(i,j+1)-divg_d(i,j))*divg_v(i,j)   ! y-flux
           enddo
        enddo
        do j=js-nt,je+1+nt
           do i=is-nt,ie+1+nt
              divg_d(i,j) = uc(i,j-1) - uc(i,j) + vc(i-1,j) - vc(i,j)
           enddo
        enddo
        ! Cube-vertex corner-removal (sw_core.F90:1773-1776):
        if (sw_corner) divg_d(1,    1)   -= uc(1,    0)     ! uc south halo
        if (se_corner) divg_d(npx,  1)   -= uc(npx,  0)     ! uc south halo
        if (ne_corner) divg_d(npx,npy)   += uc(npx,npy)     ! uc interior NE
        if (nw_corner) divg_d(1,  npy)   += uc(1,  npy)     ! uc interior NW
        divg_d(:,:) = divg_d(:,:) * rarea_c(:,:)

    Parameters
    ----------
    divg_d : jnp.ndarray, shape ``(6, n+1, n+1)``
        Corner-staggered divergence (or its iterated Laplacian).
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    lap_divg : jnp.ndarray, same shape — Laplacian of the input at
        corner staggering.

    Notes
    -----
    Halo provenance:

    * ``divg_d`` is padded with the legoESM halo (``pad_halo``) to
      provide cross-panel values at i = -1 / i = n+1 / j = -1 / j = n+1
      (in our 0-based shifted convention).  The default
      ``set_corner_fill_mode = "avg"`` (iter-7) handles the cube-vertex
      halo of the scalar.

    * ``vc`` is computed at i ∈ [-1, n] (shape ``(6, n+2, n+1)``) so
      that the divergence at the WEST corner (i = 0) sees the
      cross-panel value at i = -1.

    * ``uc`` is computed at j ∈ [-1, n] (shape ``(6, n+1, n+2)``) so
      that the divergence at the SOUTH corner (j = 0) AND the cube-
      vertex corner-removal at SW / SE both pick up the cross-panel
      south-halo value (FV3's ``uc(:, 0)`` row in 1-based).

    Documented fidelity gaps relative to FV3:

    * FV3 calls ``fill_corners(divg_d, ..., FILL=XDir, BGRID=.true.)``
      and ``YDir`` separately before each gradient (sw_core.F90:1746,
      1754) for direction-aware cube-vertex completion.  We use a
      single ``pad_halo`` with the iter-7 average corner mode; this is
      direction-agnostic but consistent with the iter-15 / iter-16
      port philosophy.

    Optional FV3 vector cube-vertex fill (iter 20):

    * ``apply_vector_corner_fill = True`` triggers FV3's
      ``fill_corners(vc, uc, VECTOR=true, DGRID=true)`` between
      gradient and divergence (sw_core.F90:1762, ported in
      ``legoesm.grids._fv3_dgrid_corner_fill``).  This affects the
      4 cube-vertex halo cells of vc / uc.

    * Code-level audit verified the cells written by the vector fill
      are NOT read by the divergence operator OR the corner-removal
      step at ``nt = 0`` (the only iteration when ``nord = 1``).  In
      that regime ``apply_vector_corner_fill = True`` produces output
      bit-for-bit identical to ``apply_vector_corner_fill = False`` —
      see the regression test ``test_corner_laplacian_vector_fill_is_noop_for_nord1``.

    * For ``nord >= 2``, the inner iterations have ``nt > 0`` and the
      divergence operator extends into halo rows that DO read the
      fill-written cells.  In that regime the flag changes output.
      Default off to preserve iter-18 bit-for-bit nord=1 behaviour
      until the wider nord >= 2 restructure (iter 21+) is in place.
    """
    n = cdgrid.n

    # Step 1: cross-panel halo of divg_d.  divg_pad shape (6, n+3, n+3).
    # Padded index 0 = west halo (i = -1 in shifted 0-based); padded
    # index n+2 = east halo (i = n+1).  Cube-vertex halo follows
    # ``set_corner_fill_mode`` (default "avg" since iter-7).
    from legoesm.grids.halo import pad_halo
    divg_pad = pad_halo(divg_d)                            # (6, n+3, n+3)

    if apply_vector_corner_fill:
        # FV3-faithful path with full halo'd vc / uc (matches FV3
        # D-grid layout) and ``fill_corners(vc, uc, VECTOR=true,
        # DGRID=true)`` applied between gradient and divergence.
        return _laplacian_iteration_with_vector_fill(
            divg_pad, cdgrid, n,
        )

    # Default iter-18 path: vc / uc with the minimum halo needed by
    # the divergence operator + corner removal at nt = 0.

    # Step 2: x-flux ``vc(i, j) = (divg_d(i+1, j) - divg_d(i, j)) * divg_u``
    # at i ∈ [-1, n], j ∈ [0, n] — shape (6, n+2, n+1).
    # Padded index translation: i (shifted) = -1..n → padded index 0..n+1.
    # divg_d(i+1) at i = -1..n → padded 1..n+2.
    # divg_d(i)   at i = -1..n → padded 0..n+1.
    vc_raw = (
        divg_pad[:, 1:n + 3, 1:n + 2]   # divg_d(i+1, j), i ∈ [-1, n]
        - divg_pad[:, 0:n + 2, 1:n + 2]  # divg_d(i,   j)
    )                                                       # (6, n+2, n+1)
    # FV3 ``divg_u = dy / dxc`` at u-face.  Our equivalent:
    # ``dy_edge_x * rdxc``, shape (6, n+1, n) at u-face.  Pad to
    # (6, n+2, n+1) via edge mode (the metric is geometric and varies
    # smoothly across panel boundaries; the i = -1 west-halo strip is
    # well-approximated by the i = 0 interior strip).
    divg_u = cdgrid.dy_edge_x * cdgrid.rdxc                  # (6, n+1, n)
    divg_u_pad = jnp.pad(
        divg_u, [(0, 0), (1, 0), (0, 1)], mode="edge",
    )                                                       # (6, n+2, n+1)
    vc = vc_raw * divg_u_pad                                # (6, n+2, n+1)

    # Step 3: y-flux ``uc(i, j) = (divg_d(i, j+1) - divg_d(i, j)) * divg_v``
    # at i ∈ [0, n], j ∈ [-1, n] — shape (6, n+1, n+2).
    uc_raw = (
        divg_pad[:, 1:n + 2, 1:n + 3]   # divg_d(i, j+1), j ∈ [-1, n]
        - divg_pad[:, 1:n + 2, 0:n + 2]  # divg_d(i, j)
    )                                                       # (6, n+1, n+2)
    divg_v = cdgrid.dx_edge_y * cdgrid.rdyc                  # (6, n, n+1)
    divg_v_pad = jnp.pad(
        divg_v, [(0, 0), (0, 1), (1, 0)], mode="edge",
    )                                                       # (6, n+1, n+2)
    uc = uc_raw * divg_v_pad                                # (6, n+1, n+2)

    # Step 4: divergence of (vc, uc) back at corners.
    #   divg_d_new(i, j) = uc(i, j-1) - uc(i, j) + vc(i-1, j) - vc(i, j)
    # In our extended uc (j ∈ [-1, n], padded slice 0..n+1):
    #   uc(i, j-1) for j ∈ [0, n] → uc[..., 0:n+1]
    #   uc(i, j)   for j ∈ [0, n] → uc[..., 1:n+2]
    # In our extended vc (i ∈ [-1, n], padded slice 0..n+1):
    #   vc(i-1, j) for i ∈ [0, n] → vc[..., 0:n+1, :]
    #   vc(i,   j) for i ∈ [0, n] → vc[..., 1:n+2, :]
    lap_divg_raw = (
        uc[:, :, 0:n + 1]
        - uc[:, :, 1:n + 2]
        + vc[:, 0:n + 1, :]
        - vc[:, 1:n + 2, :]
    )                                                       # (6, n+1, n+1)

    # Step 5: corner-removal at the 4 cube vertices — FV3 lines
    # 1773-1776.  In our extended uc (j ∈ [-1, n], shape (n+1, n+2)):
    #   FV3 uc(:, j=js-1=0) = SW/SE south-halo row → our uc[..., 0]
    #   FV3 uc(:, j=npy)    = NE/NW interior row  → our uc[..., n+1]
    sw = uc[:, 0, 0]            # uc at (i = 0,  j = -1) — south halo
    se = uc[:, n, 0]            # uc at (i = n,  j = -1) — south halo
    ne = uc[:, n, n + 1]        # uc at (i = n,  j = n)  — interior
    nw = uc[:, 0, n + 1]        # uc at (i = 0,  j = n)  — interior
    lap_divg = lap_divg_raw
    lap_divg = lap_divg.at[:, 0, 0].add(-sw)
    lap_divg = lap_divg.at[:, n, 0].add(-se)
    lap_divg = lap_divg.at[:, n, n].add(ne)
    lap_divg = lap_divg.at[:, 0, n].add(nw)

    # Step 6: normalise by rarea_c (Fortran line 1782).
    return lap_divg * cdgrid.rarea_c


def fv3_laplacian_step_from_pad_h1(
    divg_pad: jnp.ndarray,
    cdgrid: "CubedSphereCDGrid",
) -> jnp.ndarray:
    """FV3_3D iter 897: one Laplacian step on h1-padded input.

    Consumes pre-padded ``divg_pad`` of shape ``(6, n+3, n+3)``
    (halo=1 around canonical ``(n+1, n+1)`` corner-staggered field)
    and returns the Laplacian of shape ``(6, n+1, n+1)``.

    Bit-for-bit equivalent to the post-pad arithmetic (steps 2-6)
    of ``fv3_corner_laplacian_iteration``.  Foundation for the
    FV3-faithful expanding-halo nord>=2 pattern: chaining
    ``pad_halo(divg, halo=nord)`` once + nord calls without re-pad
    avoids the JAX-level corner-fill discrepancy across iterations.

    For nord=1: equivalent to existing ``fv3_corner_laplacian_iteration``
    with default ``apply_vector_corner_fill=False``.

    For nord>=2 (future iter-898+): caller will pad with wider halo
    and call this helper / a wider variant in a shrinking-interior
    loop matching FV3 sw_core.F90:1748-1785.

    Parameters
    ----------
    divg_pad : jnp.ndarray, shape ``(6, n+3, n+3)``
        Pre-padded corner-staggered divergence.
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    lap_divg : jnp.ndarray, shape ``(6, n+1, n+1)``
        Laplacian at canonical corner-staggered interior.
    """
    n = cdgrid.n

    # Step 2: x-flux vc.
    vc_raw = (
        divg_pad[:, 1:n + 3, 1:n + 2]
        - divg_pad[:, 0:n + 2, 1:n + 2]
    )
    divg_u = cdgrid.dy_edge_x * cdgrid.rdxc
    divg_u_pad = jnp.pad(
        divg_u, [(0, 0), (1, 0), (0, 1)], mode="edge",
    )
    vc = vc_raw * divg_u_pad

    # Step 3: y-flux uc.
    uc_raw = (
        divg_pad[:, 1:n + 2, 1:n + 3]
        - divg_pad[:, 1:n + 2, 0:n + 2]
    )
    divg_v = cdgrid.dx_edge_y * cdgrid.rdyc
    divg_v_pad = jnp.pad(
        divg_v, [(0, 0), (0, 1), (1, 0)], mode="edge",
    )
    uc = uc_raw * divg_v_pad

    # Step 4: divergence at corner.
    lap_divg_raw = (
        uc[:, :, 0:n + 1]
        - uc[:, :, 1:n + 2]
        + vc[:, 0:n + 1, :]
        - vc[:, 1:n + 2, :]
    )

    # Step 5: cube-vertex corner removal.
    sw = uc[:, 0, 0]
    se = uc[:, n, 0]
    ne = uc[:, n, n + 1]
    nw = uc[:, 0, n + 1]
    lap_divg = lap_divg_raw
    lap_divg = lap_divg.at[:, 0, 0].add(-sw)
    lap_divg = lap_divg.at[:, n, 0].add(-se)
    lap_divg = lap_divg.at[:, n, n].add(ne)
    lap_divg = lap_divg.at[:, 0, n].add(nw)

    # Step 6: normalise by rarea_c.
    return lap_divg * cdgrid.rarea_c


def fv3_laplacian_step_from_pad_h2(
    divg_pad: jnp.ndarray,
    cdgrid: "CubedSphereCDGrid",
) -> jnp.ndarray:
    """FV3_3D iter 898: one Laplacian step on h2-padded input.

    Consumes pre-padded ``divg_pad`` of shape ``(6, n+5, n+5)``
    (halo=2 around canonical ``(n+1, n+1)`` corner-staggered field)
    and returns the Laplacian on the SHRUNK halo=1 interior of shape
    ``(6, n+3, n+3)``.

    Each iteration consumes one halo cell per side.  Foundation
    for the FV3-faithful expanding-halo nord>=2 pattern
    (sw_core.F90:1748-1785, ``nt = nord - n`` decreasing).

    Sequence for nord=2 single-pad-multi-step pattern:

        divg_pad = pad_halo(divg_d, halo=2)        # (6, n+5, n+5)
        intermediate = fv3_laplacian_step_from_pad_h2(...)  # (6, n+3, n+3)
        out = fv3_laplacian_step_from_pad_h1(intermediate, ...)  # (6, n+1, n+1)

    Metric arrays (``divg_u``, ``divg_v``, ``rarea_c``) are
    canonical-sized and edge-padded to match the wider write region
    — same approximation as iter-18's halo=1 step, just applied
    over a wider stencil.

    Parameters
    ----------
    divg_pad : jnp.ndarray, shape ``(6, n+5, n+5)``
        Pre-halo-2-padded corner-staggered divergence.
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    lap_divg_h1 : jnp.ndarray, shape ``(6, n+3, n+3)``
        Laplacian on halo=1 interior (consumed by subsequent
        ``fv3_laplacian_step_from_pad_h1`` call).

    Notes
    -----
    Index translation for divg_pad shape (6, n+5, n+5):

      Padded index 0 = i = -2 (outer west halo)
      Padded index 1 = i = -1 (inner west halo)
      Padded indices 2..n+2 = canonical interior i = 0..n
      Padded index n+3 = i = n+1 (inner east halo)
      Padded index n+4 = i = n+2 (outer east halo)

    Output covers i = -1..n+1 (n+3 cells) — the halo=1 interior.
    """
    n = cdgrid.n

    # Step 2: x-flux vc at i ∈ [-2, n+1] (shape n+4), j ∈ [-1, n+1]
    # (shape n+3).  Padded index translation:
    #   divg_pad(i+1) for i ∈ [-2, n+1] → padded 1..n+4 → slice [1:n+5]
    #   divg_pad(i)   for i ∈ [-2, n+1] → padded 0..n+3 → slice [0:n+4]
    #   j ∈ [-1, n+1] → padded 1..n+3 → slice [1:n+4]
    vc_raw = (
        divg_pad[:, 1:n + 5, 1:n + 4]
        - divg_pad[:, 0:n + 4, 1:n + 4]
    )                                                       # (6, n+4, n+3)
    divg_u = cdgrid.dy_edge_x * cdgrid.rdxc                  # (6, n+1, n)
    divg_u_pad = jnp.pad(
        divg_u, [(0, 0), (2, 1), (1, 2)], mode="edge",
    )                                                       # (6, n+4, n+3)
    vc = vc_raw * divg_u_pad

    # Step 3: y-flux uc at i ∈ [-1, n+1] (shape n+3), j ∈ [-2, n+1]
    # (shape n+4).
    uc_raw = (
        divg_pad[:, 1:n + 4, 1:n + 5]
        - divg_pad[:, 1:n + 4, 0:n + 4]
    )                                                       # (6, n+3, n+4)
    divg_v = cdgrid.dx_edge_y * cdgrid.rdyc                  # (6, n, n+1)
    divg_v_pad = jnp.pad(
        divg_v, [(0, 0), (1, 2), (2, 1)], mode="edge",
    )                                                       # (6, n+3, n+4)
    uc = uc_raw * divg_v_pad

    # Step 4: divergence at corner for output i, j ∈ [-1, n+1]
    # (write shape n+3).  Local indexing into extended uc / vc:
    #   uc(i, j-1) for j ∈ [-1, n+1] → uc[..., 0:n+3]
    #   uc(i, j)   for j ∈ [-1, n+1] → uc[..., 1:n+4]
    #   vc(i-1, j) for i ∈ [-1, n+1] → vc[..., 0:n+3, :]
    #   vc(i,   j) for i ∈ [-1, n+1] → vc[..., 1:n+4, :]
    lap_divg_raw = (
        uc[:, :, 0:n + 3]
        - uc[:, :, 1:n + 4]
        + vc[:, 0:n + 3, :]
        - vc[:, 1:n + 4, :]
    )                                                       # (6, n+3, n+3)

    # Step 5: cube-vertex corner removal.  Cube vertices are at the
    # CANONICAL corners of the original (n+1, n+1) field, which sit
    # at INTERIOR indices of the wider (n+3, n+3) write region:
    #   (i=0, j=0) → local (1, 1)
    #   (i=n, j=0) → local (n+1, 1)
    #   (i=n, j=n) → local (n+1, n+1)
    #   (i=0, j=n) → local (1, n+1)
    # uc indices: south-halo row j = -1 → uc[..., 0]; interior j = n
    # → uc[..., n+2]
    sw = uc[:, 1, 0]            # uc at (i = 0,  j = -1)
    se = uc[:, n + 1, 0]        # uc at (i = n,  j = -1)
    ne = uc[:, n + 1, n + 2]    # uc at (i = n,  j = n)
    nw = uc[:, 1, n + 2]        # uc at (i = 0,  j = n)
    lap_divg = lap_divg_raw
    lap_divg = lap_divg.at[:, 1, 1].add(-sw)
    lap_divg = lap_divg.at[:, n + 1, 1].add(-se)
    lap_divg = lap_divg.at[:, n + 1, n + 1].add(ne)
    lap_divg = lap_divg.at[:, 1, n + 1].add(nw)

    # Step 6: normalise by rarea_c (extended via edge mode).
    rarea_c_pad = jnp.pad(
        cdgrid.rarea_c, [(0, 0), (1, 1), (1, 1)], mode="edge",
    )                                                       # (6, n+3, n+3)
    return lap_divg * rarea_c_pad


def fv3_corner_laplacian_nord(
    divg_d: jnp.ndarray,
    cdgrid: "CubedSphereCDGrid",
    nord: int,
    apply_vector_corner_fill: bool = False,
) -> jnp.ndarray:
    """FV3_3D iter 892: nord-fold Laplacian wrapper.

    Apply ``fv3_corner_laplacian_iteration`` ``nord`` times in
    sequence, equivalent to the loops at PE
    ``primitive_eq_cdgrid.py:508-510`` and NH
    ``compressible_euler_cdgrid.py:631-632``:

        for _ in range(nord):
            divg_d = fv3_corner_laplacian_iteration(divg_d, cdgrid, ...)

    Extracting this into a named helper:

      1. Provides a single-source-of-truth point for the nord-loop
         pattern (currently duplicated across PE and NH paths).
      2. Enables iter-893+ to swap in the expanding-halo
         implementation behind one function rather than two
         independent loop refactors.
      3. Decouples the unit-test for the higher-order Laplacian
         stack from the full PE/NH model dispatch (faster CI).

    Bit-for-bit equivalent to the inline loop — no behaviour change
    in this iteration.  Foundation for iter-893's expanding-halo
    replacement.

    Faithful to FV3 ``sw_core.F90:1746-1782``::

        do n=1, nord
           ! ... inner Laplacian arithmetic ...
        enddo

    Parameters
    ----------
    divg_d : jnp.ndarray, shape ``(6, n+1, n+1)``
        Corner-staggered divergence (or its iterated Laplacian).
    cdgrid : CubedSphereCDGrid
    nord : int
        Number of Laplacian iterations.  Must be in {0, 1, 2, 3}
        per FV3 namelist range; caller should validate via
        ``validate_corner_div_damp_nord`` (iter-890).  Pass-through
        of nord=0 returns input unchanged.
    apply_vector_corner_fill : bool
        FV3-faithful vector cube-vertex fill (sw_core.F90:1762).
        Default False (no-op at nord=1, changes output at nord>=2).

    Returns
    -------
    lap_nord_divg : jnp.ndarray, same shape as ``divg_d``.
        nord-fold iterated Laplacian.

    Notes
    -----
    nord=0: returns input unchanged (degenerate).
    nord=1: equivalent to single ``fv3_corner_laplacian_iteration``.
    nord>=2: each iteration re-pads via ``pad_halo`` (current
        legoESM behaviour; iter-893 replaces with FV3-faithful
        expanding-halo pattern).
    """
    result = divg_d
    for _ in range(nord):
        result = fv3_corner_laplacian_iteration(
            result, cdgrid, apply_vector_corner_fill=apply_vector_corner_fill,
        )
    return result


def _laplacian_iteration_with_vector_fill(
    divg_pad: jnp.ndarray,
    cdgrid: CubedSphereCDGrid,
    n: int,
) -> jnp.ndarray:
    """FV3-fully-faithful Laplacian iteration with vector cube-vertex fill.

    Differs from the default path: vc / uc are computed on the FV3
    D-grid layout shape ((6, n+2, n+3) and (6, n+3, n+2) respectively),
    and ``fv3_fill_corners_dgrid_vector`` is applied between gradient
    and divergence (matching FV3 ``sw_core.F90:1762``).

    For nord = 1 (single iteration with nt = 0) the divergence
    operator and corner-removal access cells that are unaffected by
    the vector fill — the output is bit-for-bit identical to the
    default path.  Verified by ``test_corner_laplacian_vector_fill_is_noop_for_nord1``.

    For nord >= 2, the inner iterations (nt > 0) extend into halo
    rows that DO read fill-written cells, so the flag is genuinely
    effective there.  This function is preparation for a future
    nord >= 2 fidelity fix.
    """
    from legoesm.grids._fv3_dgrid_corner_fill import (
        fv3_fill_corners_dgrid_vector,
    )

    # vc at i ∈ [-1, n], j ∈ [-1, n+1] — shape (6, n+2, n+3).
    # Padded index translation in our shifted convention:
    #   i (shifted) = -1..n   → padded i = 0..n+1
    #   j (shifted) = -1..n+1 → padded j = 0..n+2
    # vc(i, j) = divg_d(i+1, j) - divg_d(i, j)
    # divg_d(i+1) → padded i = 1..n+2
    # divg_d(i)   → padded i = 0..n+1
    vc_raw = (
        divg_pad[:, 1:n + 3, 0:n + 3]
        - divg_pad[:, 0:n + 2, 0:n + 3]
    )                                                       # (6, n+2, n+3)
    divg_u = cdgrid.dy_edge_x * cdgrid.rdxc                  # (6, n+1, n)
    # Pad to (6, n+2, n+3): (1, 0) on i (add west halo), (1, 2) on j
    # (add 1 south halo + 2 north halo to cover j ∈ [-1, n+1]).
    divg_u_pad = jnp.pad(
        divg_u, [(0, 0), (1, 0), (1, 2)], mode="edge",
    )                                                       # (6, n+2, n+3)
    vc = vc_raw * divg_u_pad                                # (6, n+2, n+3)

    # uc at i ∈ [-1, n+1], j ∈ [-1, n] — shape (6, n+3, n+2).
    uc_raw = (
        divg_pad[:, 0:n + 3, 1:n + 3]
        - divg_pad[:, 0:n + 3, 0:n + 2]
    )                                                       # (6, n+3, n+2)
    divg_v = cdgrid.dx_edge_y * cdgrid.rdyc                  # (6, n, n+1)
    divg_v_pad = jnp.pad(
        divg_v, [(0, 0), (1, 2), (1, 0)], mode="edge",
    )                                                       # (6, n+3, n+2)
    uc = uc_raw * divg_v_pad                                # (6, n+3, n+2)

    # Apply FV3 vector cube-vertex fill (sw_core.F90:1762).
    # This overwrites the 4 cube-vertex halo cells of vc / uc with the
    # sign-flipped diagonal mirror of the OTHER component.
    vc, uc = fv3_fill_corners_dgrid_vector(vc, uc, n)

    # Divergence at interior corners (i, j) ∈ [0, n] × [0, n].
    # In the wider vc shape (n+2, n+3), padded i = 0..n+1 = i_logical = -1..n;
    # padded j = 0..n+2 = j_logical = -1..n+1.
    # vc(i-1, j) for (i, j) ∈ [0, n]² → padded i = 0..n, padded j = 1..n+1.
    # vc(i,   j) for (i, j) ∈ [0, n]² → padded i = 1..n+1, padded j = 1..n+1.
    # uc(i, j-1) for (i, j) ∈ [0, n]² → padded i = 1..n+1, padded j = 0..n.
    # uc(i,   j) for (i, j) ∈ [0, n]² → padded i = 1..n+1, padded j = 1..n+1.
    lap_divg_raw = (
        uc[:, 1:n + 2, 0:n + 1]      # uc(i, j-1)
        - uc[:, 1:n + 2, 1:n + 2]    # uc(i, j)
        + vc[:, 0:n + 1, 1:n + 2]    # vc(i-1, j)
        - vc[:, 1:n + 2, 1:n + 2]    # vc(i, j)
    )                                                       # (6, n+1, n+1)

    # Corner removal — same logical positions as the default path
    # but adjusted indices for the wider arrays.
    # SW: uc(i_logical = 0, j_logical = -1) → padded i = 1, j = 0.
    # SE: uc(i_logical = n, j_logical = -1) → padded i = n+1, j = 0.
    # NE: uc(i_logical = n, j_logical = n)  → padded i = n+1, j = n+1.
    # NW: uc(i_logical = 0, j_logical = n)  → padded i = 1,   j = n+1.
    sw = uc[:, 1, 0]
    se = uc[:, n + 1, 0]
    ne = uc[:, n + 1, n + 1]
    nw = uc[:, 1, n + 1]
    lap_divg = lap_divg_raw
    lap_divg = lap_divg.at[:, 0, 0].add(-sw)
    lap_divg = lap_divg.at[:, n, 0].add(-se)
    lap_divg = lap_divg.at[:, n, n].add(ne)
    lap_divg = lap_divg.at[:, 0, n].add(nw)

    return lap_divg * cdgrid.rarea_c
