"""FV3-style C-D grid on the cubed-sphere.

Extends the cell-centre ``CubedSphereGrid`` with staggered metric fields
needed for the C-D grid discretisation:

* **D-grid** (cell corners): prognostic wind components ``u_d, v_d``
  and absolute vorticity live at shape ``(6, n+1, n+1)``.
* **C-grid** (cell edges): normal velocity used for mass flux.
  ``u_c`` at x-interfaces: ``(6, n+1, n)``; ``v_c`` at y-interfaces:
  ``(6, n, n+1)``.

The key advantage is that vorticity on the D-grid is computed directly
from corner wind values without spatial averaging, eliminating the
Hollingsworth-Kallberg instability that plagues collocated schemes.

References
----------
- Lin (2004): A "Vertically Lagrangian" Finite-Volume Dynamical Core
- Harris & Lin (2013): A Two-Way Nested Global-Regional Dynamical Core
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

_TINY = float(jnp.finfo(jnp.float32).tiny)  # Smallest normal float32 (~1.18e-38)
_EPS = float(jnp.finfo(jnp.float32).eps)    # Float32 machine epsilon (~1.19e-7)

from legoesm import constants
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import (
    CONNECTIVITY,
    EAST,
    NORTH,
    SOUTH,
    WEST,
    face_gnomonic_to_lonlat,
    fill_corners_h1,
    pad_halo,
)


class CubedSphereCDGrid(NamedTuple):
    """C-D grid metrics for a cubed-sphere grid.

    Wraps a base ``CubedSphereGrid`` (cell-centre grid) and adds the staggered
    metric arrays needed by the C-D grid discretisation.

    Attributes
    ----------
    base : CubedSphereGrid
        The underlying cell-centre grid with cell-center metrics.
    lon_corner : jax.Array, shape (6, n+1, n+1)
        Longitude at cell corners (D-grid positions).
    lat_corner : jax.Array, shape (6, n+1, n+1)
        Latitude at cell corners.
    f_corner : jax.Array, shape (6, n+1, n+1)
        Coriolis parameter at cell corners.
    angle_corner : jax.Array, shape (6, n+1, n+1)
        Grid rotation angle at cell corners.
    cos_angle_corner : jax.Array, shape (6, n+1, n+1)
    sin_angle_corner : jax.Array, shape (6, n+1, n+1)
    dx_edge_y : jax.Array, shape (6, n, n+1)
        Length of cell edges in y-direction (v_c positions).
    dy_edge_x : jax.Array, shape (6, n+1, n)
        Length of cell edges in x-direction (u_c positions).
    area_corner : jax.Array, shape (6, n+1, n+1)
        Dual-cell area at corners (for vorticity equation).
    """
    base: CubedSphereGrid
    lon_corner: jax.Array
    lat_corner: jax.Array
    f_corner: jax.Array
    angle_corner: jax.Array
    cos_angle_corner: jax.Array
    sin_angle_corner: jax.Array
    dx_edge_y: jax.Array
    dy_edge_x: jax.Array
    area_corner: jax.Array
    # Non-orthogonality metrics (FV3 cos_sg / sin_sg)
    cosa_corner: jax.Array   # (6, n+1, n+1) cos(angle between i and j tangents)
    rsin2_corner: jax.Array  # (6, n+1, n+1) 1/sin²(angle) for gradient correction
    z21_corner: jax.Array    # (6, n+1, n+1) ec2·east — corner c2l z-matrix row (j-tangent)
    z22_corner: jax.Array    # (6, n+1, n+1) ec2·north — corner c2l z-matrix row (j-tangent)
    cosa_u: jax.Array        # (6, n+1, n) non-orthogonality at u-interfaces
    cosa_v: jax.Array        # (6, n, n+1) non-orthogonality at v-interfaces
    rsin_u: jax.Array        # (6, n+1, n) 1/sin² interior; 1/sin at panel edges (non-duogrid)
    rsin_v: jax.Array        # (6, n, n+1) 1/sin² interior; 1/sin at panel edges (non-duogrid)
    # --- FV3 edge-midpoint D-grid metrics ---
    # Edge-midpoint positions
    lon_edge_x: jax.Array    # (6, n, n+1) lon at x-edge midpoints
    lat_edge_x: jax.Array    # (6, n, n+1) lat at x-edge midpoints
    lon_edge_y: jax.Array    # (6, n+1, n) lon at y-edge midpoints
    lat_edge_y: jax.Array    # (6, n+1, n) lat at y-edge midpoints
    # Grid angle at edge midpoints
    angle_edge_x: jax.Array      # (6, n, n+1)
    cos_angle_edge_x: jax.Array  # (6, n, n+1)
    sin_angle_edge_x: jax.Array  # (6, n, n+1)
    angle_edge_y: jax.Array      # (6, n+1, n)
    cos_angle_edge_y: jax.Array  # (6, n+1, n)
    sin_angle_edge_y: jax.Array  # (6, n+1, n)
    # Coriolis at edges
    f_edge_x: jax.Array      # (6, n, n+1)
    f_edge_y: jax.Array      # (6, n+1, n)
    # Cell-centre non-orthogonality metrics (for d2a2c)
    cosa_cell: jax.Array     # (6, n, n) cos(angle between i and j tangents)
    sina_cell: jax.Array     # (6, n, n) sin(angle)
    rsin2_cell: jax.Array    # (6, n, n) 1/sin²(angle)
    # Precomputed Arakawa-Lamb gradient transformation matrix.
    # Maps 4-point finite-difference quantities (ΔB_x, ΔB_y) directly to
    # physical gradient (dB/dx along e_i, dB/dy_perp ⊥ e_i).  Derived from
    # 3D Cartesian geometry of haloed cell centers → correct at face
    # boundaries and cube vertices where face-local metrics are inconsistent.
    grad_c00: jax.Array    # (6, n+1, n+1)
    grad_c01: jax.Array    # (6, n+1, n+1)
    grad_c10: jax.Array    # (6, n+1, n+1)
    grad_c11: jax.Array    # (6, n+1, n+1)
    # --- Duo-Grid sub-grid metrics (Mouallem, Harris & Chen 2023) ---
    # 9-point supergrid layout per cell:
    #      9---4---8
    #      |       |
    #      1   5   3
    #      |       |
    #      6---2---7
    # 0-indexed: 0=W, 1=S, 2=E, 3=N edge midpoints; 4=center;
    #            5=SW, 6=SE, 7=NE, 8=NW corners
    sin_sg: jax.Array      # (6, n, n, 9) sin(angle) at 9 sub-grid positions
    cos_sg: jax.Array      # (6, n, n, 9) cos(angle) at 9 sub-grid positions
    # --- FV3 c_sw grid metrics ---
    # Center-to-center distances across C-grid faces, used for the FV3
    # 2-point Bernoulli/KE gradient (replaces Arakawa-Lamb 4-point stencil).
    dxc: jax.Array         # (6, n+1, n) distance between cell (i-1,j) and (i,j)
    dyc: jax.Array         # (6, n, n+1) distance between cell (i,j-1) and (i,j)
    rdxc: jax.Array        # (6, n+1, n) 1/dxc
    rdyc: jax.Array        # (6, n, n+1) 1/dyc
    rarea_c: jax.Array     # (6, n+1, n+1) 1/area_corner
    # FV3 cell-width metrics (fv_grid_tools.F90): dxa = x-width of cell (i,j)
    # = distance between u-face midpoints i and i+1 along row j.
    # Used for Courant number: crx = dt*ut*rdxa(upwind_cell) (sw_core.F90:850).
    rdxa: jax.Array        # (6, n, n) 1/dxa at cell centres
    rdya: jax.Array        # (6, n, n) 1/dya at cell centres

    @property
    def n(self) -> int:
        return self.base.n

    @property
    def radius(self) -> float:
        return self.base.radius

    @property
    def gnomonic_form(self) -> str:
        """Static grid provenance, delegated to the base grid."""
        return self.base.gnomonic_form

    @property
    def fv3_grid_type(self) -> int:
        """Static FV3 grid_type provenance, delegated to the base grid."""
        return self.base.fv3_grid_type


def _compute_sin_cos_sg(n, padded_supergrid_lon, padded_supergrid_lat):
    """Compute Duo-Grid sub-grid metrics at 9 positions per cell.

    Uses a supergrid (half the cell spacing) to evaluate the angle between
    i-tangent and j-tangent at each sub-grid position via centred
    differences of 3D Cartesian positions.

    **Iter-678 note**: an attempt to replace this with Fortran's exact
    ``cos_angle`` + ``mid_pt3_cart`` formula from
    ``fv_grid_utils.F90:324-355`` (pointwise Fortran-faithful) WORSENED
    Williamson 2 alpha=0 C36 1-day L2 by 2.2× (1.098e-3 vs iter-505 lock
    ceiling 5.0e-4) and broke 9 regression tests.  Downstream operators
    (c_sw, d2a2c_vect, deln flux, KE, vorticity) are tuned against this
    tangent-vector cos_sg.  Switching cos_sg formulas without updating
    the downstream operators in tandem breaks numerical consistency.

    The ~O(1/N) discretization difference between Python's tangent method
    and Fortran's cos_angle formula (verified by
    ``TestCosSgFortranFormulaIter678`` at atol 2e-2 for C8) reflects a
    CONSISTENT numerical choice: Python uses centred differences; Fortran
    uses arc-projection.  Both are valid metrics; replacing one without
    the other creates a MIXED scheme that neither Python nor Fortran has
    tuned against.  Revisiting this requires a coordinated rewrite of
    downstream operators, which is deferred architectural work (see
    review doc item #1).

    Parameters
    ----------
    n : int
        Number of cells per face edge.
    face_gnomonic_to_lonlat : callable
        ``face_gnomonic_to_lonlat(face, ax, ay) -> (lon, lat)``

    Returns
    -------
    sin_sg : jax.Array, shape (6, n, n, 9)
    cos_sg : jax.Array, shape (6, n, n, 9)
    """
    jnp.pi / (2 * n)

    # Supergrid: 2n+3 positions per axis.
    # Index mapping (0-based):
    #   0: -pi/4 - dalpha/2   (padding for centred diff)
    #   1: -pi/4              (corner 0)
    #   2: -pi/4 + dalpha/2   (cell centre 0)
    #   2k+1: corner k        for k = 0..n
    #   2k+2: cell centre k   for k = 0..n-1
    #   2n+2: pi/4 + dalpha/2 (padding)
    # Padded supergrid (2n+3 per axis) supplied precomputed (grid-type-agnostic,
    # iter71); index per face instead of the equiangular parametric map.
    padded_supergrid_lon = jnp.asarray(padded_supergrid_lon)
    padded_supergrid_lat = jnp.asarray(padded_supergrid_lat)

    all_cos_sg = []
    for face in range(6):
        lon_sg, lat_sg = padded_supergrid_lon[face], padded_supergrid_lat[face]
        cos_lat = jnp.cos(lat_sg)
        px = cos_lat * jnp.cos(lon_sg)
        py = cos_lat * jnp.sin(lon_sg)
        pz = jnp.sin(lat_sg)

        # Centre positions for tangent-plane projection: (2n+1, 2n+1)
        cx = px[1:-1, 1:-1]
        cy = py[1:-1, 1:-1]
        cz = pz[1:-1, 1:-1]

        # i-tangent via centred diff in axis 0: (2n+1, 2n+1)
        ti_x = px[2:, 1:-1] - px[:-2, 1:-1]
        ti_y = py[2:, 1:-1] - py[:-2, 1:-1]
        ti_z = pz[2:, 1:-1] - pz[:-2, 1:-1]
        dot_i = ti_x * cx + ti_y * cy + ti_z * cz
        ti_x = ti_x - dot_i * cx
        ti_y = ti_y - dot_i * cy
        ti_z = ti_z - dot_i * cz
        norm_i = jnp.sqrt(ti_x ** 2 + ti_y ** 2 + ti_z ** 2 + _TINY)
        ti_x = ti_x / norm_i
        ti_y = ti_y / norm_i
        ti_z = ti_z / norm_i

        # j-tangent via centred diff in axis 1: (2n+1, 2n+1)
        tj_x = px[1:-1, 2:] - px[1:-1, :-2]
        tj_y = py[1:-1, 2:] - py[1:-1, :-2]
        tj_z = pz[1:-1, 2:] - pz[1:-1, :-2]
        dot_j = tj_x * cx + tj_y * cy + tj_z * cz
        tj_x = tj_x - dot_j * cx
        tj_y = tj_y - dot_j * cy
        tj_z = tj_z - dot_j * cz
        norm_j = jnp.sqrt(tj_x ** 2 + tj_y ** 2 + tj_z ** 2 + _TINY)
        tj_x = tj_x / norm_j
        tj_y = tj_y / norm_j
        tj_z = tj_z / norm_j

        # cos(angle) at all supergrid interior positions
        cosa = ti_x * tj_x + ti_y * tj_y + ti_z * tj_z  # (2n+1, 2n+1)

        # Extract 9 sub-grid positions for each cell (i,j), i,j = 0..n-1.
        # In the cosa array (indexed 0..2n):
        #   corner (i,j)       → (2i, 2j)
        #   cell centre (i)    → (2i+1, ...)
        #   W edge of cell i,j → (2i, 2j+1)      [corner-i, centre-j]
        #   E edge             → (2i+2, 2j+1)    [corner-i+1, centre-j]
        #   S edge             → (2i+1, 2j)      [centre-i, corner-j]
        #   N edge             → (2i+1, 2j+2)    [centre-i, corner-j+1]
        #   Centre             → (2i+1, 2j+1)

        m = 2 * n  # shorthand
        cos_sg_face = jnp.stack([
            cosa[0:m:2,   1:m + 1:2],   # 0: W edge
            cosa[1:m + 1:2, 0:m:2],     # 1: S edge
            cosa[2:m + 1:2, 1:m + 1:2], # 2: E edge
            cosa[1:m + 1:2, 2:m + 1:2], # 3: N edge
            cosa[1:m + 1:2, 1:m + 1:2], # 4: Centre
            cosa[0:m:2,   0:m:2],       # 5: SW corner
            cosa[2:m + 1:2, 0:m:2],     # 6: SE corner
            cosa[2:m + 1:2, 2:m + 1:2], # 7: NE corner
            cosa[0:m:2,   2:m + 1:2],   # 8: NW corner
        ], axis=-1)  # (n, n, 9)

        all_cos_sg.append(cos_sg_face)

    cos_sg = jnp.stack(all_cos_sg, axis=0)  # (6, n, n, 9)
    sin_sg = jnp.sqrt(jnp.maximum(1.0 - cos_sg ** 2, 0.0))

    return sin_sg, cos_sg


def _compute_supergrid_metrics(n, supergrid_lon, supergrid_lat, radius):
    """Compute area_c and dxc/dyc from the FV3 supergrid.

    Uses the SAME 2x-refined supergrid as sin_sg/cos_sg to ensure all
    metrics are mutually consistent (discrete Stokes theorem).

    ``supergrid_lon/lat`` are the precomputed ``(6, 2n+1, 2n+1)`` supergrid
    node positions — equiangular or gnomonic_ed — so this metric computation
    is grid-type-agnostic (iter71: was the equiangular parametric map).

    FV3 convention (fv_grid_tools.F90 line 1459):
        area_c(i,j) = sum of 4 supergrid cell areas around dual-cell corner

    FV3 convention (fv_grid_tools.F90 line 883):
        dxc(i,j) = 2 * great_circle_dist(edge_midpoint, cell_center)

    Returns
    -------
    area_c_all : (6, n+1, n+1) dual cell areas
    dxc_all : (6, n+1, n) center-to-center distances in x
    dyc_all : (6, n, n+1) center-to-center distances in y
    dxa_all : (6, n, n) cell widths in x (face-to-face)
    dya_all : (6, n, n) cell widths in y (face-to-face)
    """
    import numpy as np

    supergrid_lon = np.asarray(supergrid_lon)
    supergrid_lat = np.asarray(supergrid_lat)

    all_area_c = []
    all_dxc = []
    all_dyc = []
    all_dxa = []
    all_dya = []

    for face in range(6):
        lon_sg = supergrid_lon[face]
        lat_sg = supergrid_lat[face]
        cos_lat = np.cos(lat_sg)
        px = cos_lat * np.cos(lon_sg)
        py = cos_lat * np.sin(lon_sg)
        pz = np.sin(lat_sg)

        # Supergrid indices:
        #   Corner (i,j) at (2i, 2j)          i,j = 0..n
        #   Cell center (i,j) at (2i+1, 2j+1) i,j = 0..n-1
        #   x-edge midpoint at (2i, 2j+1)     between corners (2i,2j) and (2i,2j+2)
        #   y-edge midpoint at (2i+1, 2j)

        # --- area_c at dual-cell corners = cell centers ---
        # Dual cell corner at cell center (i,j) = supergrid (2i+1, 2j+1)
        # Sum of 4 surrounding supergrid quadrilateral areas.
        # The 4 supergrid cells around (2i+1, 2j+1):
        #   SW: (2i, 2j)-(2i+1, 2j)-(2i+1, 2j+1)-(2i, 2j+1)
        #   SE: (2i+1, 2j)-(2i+2, 2j)-(2i+2, 2j+1)-(2i+1, 2j+1)
        #   NE: (2i+1, 2j+1)-(2i+2, 2j+1)-(2i+2, 2j+2)-(2i+1, 2j+2)
        #   NW: (2i, 2j+1)-(2i+1, 2j+1)-(2i+1, 2j+2)-(2i, 2j+2)

        # Compute ALL supergrid quadrilateral areas: (2n, 2n)
        # Each quad (si, sj) has corners at (si,sj),(si+1,sj),(si+1,sj+1),(si,sj+1)
        # NOTE: this is the PLANAR chord-cross-product area (0.5*|d1×d2|*R²), a
        # deliberate O(dx²) approximation to FV3's spherical-excess `get_area`
        # (fv_grid_utils.F90).  The two agree to ~1e-4 at C36 / ~1e-2 at C8 and
        # converge as the grid refines; the entire SW-core gold-file fingerprint
        # surface (cosine-bell, d_sw_native, production-tendencies, corner-
        # vorticity, ...) is pinned to this chord convention.  The FV3-faithful
        # part of `area_corner` is the (#faces)-junction SCALING below (edges ×2,
        # vertices ×3); switching the absolute per-quadrant area to spherical
        # get_area is a tracked follow-up (see fv3_faithful.md, requires
        # regenerating all chord-era gold files) — NOT done here to keep the C1
        # guard fix regression-isolated.  At C1 this chord area gives a
        # cube-vertex area of 0.659*cell vs the spherical 0.75*cell.
        d1x = px[1:, 1:] - px[:-1, :-1]; d1y = py[1:, 1:] - py[:-1, :-1]; d1z = pz[1:, 1:] - pz[:-1, :-1]
        d2x = px[1:, :-1] - px[:-1, 1:]; d2y = py[1:, :-1] - py[:-1, 1:]; d2z = pz[1:, :-1] - pz[:-1, 1:]
        cx = d1y*d2z - d1z*d2y; cy = d1z*d2x - d1x*d2z; cz = d1x*d2y - d1y*d2x
        sg_area = 0.5 * np.sqrt(cx**2 + cy**2 + cz**2) * radius**2  # (2n, 2n)

        # Sum 4 supergrid cells at each dual-cell corner (cell center position)
        # Cell center (i,j) → supergrid (2i+1, 2j+1), i,j = 0..n-1
        # But dual cell CORNER at grid corner (i,j), i,j = 0..n
        # Grid corner (i,j) = supergrid (2i, 2j)
        # The 4 supergrid cells around grid corner (2i, 2j):
        #   SW: sg_area[2i-1, 2j-1], SE: sg_area[2i, 2j-1]
        #   NW: sg_area[2i-1, 2j],   NE: sg_area[2i, 2j]
        # For i,j = 0..n: need 2i-1 >= 0 and 2i <= 2n-1 → i = 1..n-1 for interior
        # Boundaries use available cells only

        # Iter-670 fix: at cube boundaries/vertices, the 4-quadrant
        # supergrid sum misses contributions from neighbour faces
        # (1 or 3 of the 4 quadrants are off-face).  This gave
        # area_corner ≈ 0.22× interior at cube vertices and 0.43× at
        # cube edges — which amplified rarea_corner by 2-4× at
        # boundaries and propagated into every operator that uses
        # `cdgrid.rarea_c` (e.g., `cdgrid_momentum_tendencies` at
        # `operators_cdgrid.py:1090`).
        #
        # Fortran oracle (`tools/fv_grid_tools.F90:1084-1087, 1561-1564`)
        # extrapolates the metric at cube boundaries:
        #     area_c(isd,j)         = area_c(isd+1,j)          ! west edge
        #     area_c(ied+1,j)       = area_c(ied,j)            ! east edge
        #     area_c(i,jsd)         = area_c(i,jsd+1)          ! south edge
        #     area_c(i,jed+1)       = area_c(i,jed)            ! north edge
        #     area_c(isd,jsd)       = area_c(isd+1,jsd+1)      ! SW corner
        #     area_c(isd,jed+1)     = area_c(isd+1,jed)        ! NW corner
        #     area_c(ied+1,jsd)     = area_c(ied,jsd+1)        ! SE corner
        #     area_c(ied+1,jed+1)   = area_c(ied,jed)          ! NE corner
        # (exact match at leading order for uniform cubed-sphere; the
        # neighbour-face area is identical.)
        #
        # For n >= 1, scale the boundary corners by the number of faces meeting
        # at the node (edges ×2, vertices ×3); see the FV3 grid_area block below.
        area_c = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            for j in range(n + 1):
                si, sj = 2 * i, 2 * j
                total = 0.0
                if si > 0 and sj > 0:
                    total += sg_area[si - 1, sj - 1]
                if si < 2 * n and sj > 0:
                    total += sg_area[si, sj - 1]
                if si > 0 and sj < 2 * n:
                    total += sg_area[si - 1, sj]
                if si < 2 * n and sj < 2 * n:
                    total += sg_area[si, sj]
                area_c[i, j] = total
        if n >= 1:
            # FV3 boundary corner control-volume area = (# faces meeting at the
            # node) × (the ON-FACE sub-quadrant sum already accumulated in area_c
            # by the loop above).  fv_grid_tools.F90:980-1067:
            #   - interior corner: full 4-quadrant dual cell (loop value, no scale);
            #   - cube EDGE node (2 faces meet): loop = 2 on-face quadrants → ×2
            #     (FV3 2*get_area = 2*(sg_area[0,2j-1]+sg_area[0,2j]) etc.);
            #   - cube VERTEX (3-face junction): loop = 1 on-face quadrant → ×3
            #     (FV3 3*get_area, lines 1036-1067).
            # iter84 fixed the vertices (×3); iter89 (codex review of 4ad2fea0)
            # extends the same (#faces)-scaling to the EDGES — the prior inward
            # interior-copy extrapolation was ~1.2% off at C96 on O(n) boundary
            # corners (rarea_c silently low → boundary vorticity/divergence bias).
            # The edge slices are empty no-ops at n=1 (1:n is empty); the four
            # cube vertices still need the ×3 there (all C1 corners are 3-face
            # junctions), so the whole block runs for n >= 1 — iter90 (codex
            # review of d7108d48), guarding C1 against a 3× corner under-count.
            area_c[0, 1:n] = 2.0 * area_c[0, 1:n]    # west edge  (i=0)
            area_c[n, 1:n] = 2.0 * area_c[n, 1:n]    # east edge  (i=n)
            area_c[1:n, 0] = 2.0 * area_c[1:n, 0]    # south edge (j=0)
            area_c[1:n, n] = 2.0 * area_c[1:n, n]    # north edge (j=n)
            area_c[0, 0] = 3.0 * area_c[0, 0]        # SW vertex (= 3*sg_area[0,0])
            area_c[0, n] = 3.0 * area_c[0, n]        # NW vertex
            area_c[n, 0] = 3.0 * area_c[n, 0]        # SE vertex
            area_c[n, n] = 3.0 * area_c[n, n]        # NE vertex
        all_area_c.append(area_c)

        # --- dxc: distance between cell centers (i-1,j) and (i,j) ---
        # Cell center (i,j) at supergrid (2i+1, 2j+1)
        # dxc at x-face (i,j): dist from (2(i-1)+1, 2j+1) to (2i+1, 2j+1)
        # = dist from (2i-1, 2j+1) to (2i+1, 2j+1) for i=1..n-1, j=0..n-1
        # Iter-666/667/669 fix: at i=0 and i=n (cube boundary u-faces),
        # the pre-iter-666 `max(2*i-1, 0)` / `min(2*i+1, 2*n)`
        # clamping gave sj spans of 1 supergrid cell = HALF the
        # interior cell width.  That made rdxc at cube boundaries 2×
        # the interior value, which amplified PGF by 2× at cube
        # edges — the direct cause of the FB-chain step-1 residual
        # (measured 2.93 interior vs 5.87 boundary; ratio exactly
        # 2.0, iter-665).
        #
        # Iter-669 verification: Fortran oracle at
        # `atmos_cubed_sphere-symmetryclean/tools/fv_grid_tools.F90:
        # 894-914` does exactly this — compute great-circle distance
        # for interior i only (`do i=isd+1,ied`), then extrapolate:
        #     dxc(isd,j)   = dxc(isd+1,j)
        #     dxc(ied+1,j) = dxc(ied,j)
        # The iter-666 fix below is the direct Python equivalent.
        #
        # Iter-667 (Codex correction on iter-666): the extrapolation
        # requires adjacent interior cells (i=1, i=n-1) to exist,
        # which needs n >= 2.  For n == 1 there IS no interior
        # u-face, so fall back to the original clamped values — not
        # Fortran-faithful but non-zero, which is what downstream
        # code expects.  n == 1 is only used in toy regional tests.
        dxc = np.zeros((n + 1, n))
        if n >= 2:
            for i in range(1, n):  # interior u-faces only
                for j in range(n):
                    si0 = 2 * i - 1
                    si1 = 2 * i + 1
                    sj = 2 * j + 1
                    chord = np.sqrt((px[si0, sj] - px[si1, sj])**2
                                    + (py[si0, sj] - py[si1, sj])**2
                                    + (pz[si0, sj] - pz[si1, sj])**2)
                    dxc[i, j] = radius * 2.0 * np.arcsin(min(chord / 2.0, 1.0))
            # Extrapolate at cube boundaries (i=0 from i=1, i=n from i=n-1).
            dxc[0, :] = dxc[1, :]
            dxc[n, :] = dxc[n - 1, :]
        else:
            # n == 1: no interior u-face.  Fall back to the original
            # clamped-supergrid computation so dxc is non-zero.
            for i in range(n + 1):
                for j in range(n):
                    si0 = max(2 * i - 1, 0)
                    si1 = min(2 * i + 1, 2 * n)
                    sj = 2 * j + 1
                    chord = np.sqrt((px[si0, sj] - px[si1, sj])**2
                                    + (py[si0, sj] - py[si1, sj])**2
                                    + (pz[si0, sj] - pz[si1, sj])**2)
                    dxc[i, j] = radius * 2.0 * np.arcsin(min(chord / 2.0, 1.0))
        all_dxc.append(dxc)

        # --- dyc: distance between cell centers (i,j-1) and (i,j) ---
        # Iter-666/667 fix: same clamping bug as dxc.  Extrapolate at
        # cube-boundary v-faces (j=0, j=n) from adjacent interior when
        # n >= 2.  For n == 1 fall back to original clamped behaviour.
        dyc = np.zeros((n, n + 1))
        if n >= 2:
            for i in range(n):
                for j in range(1, n):  # interior v-faces only
                    si = 2 * i + 1
                    sj0 = 2 * j - 1
                    sj1 = 2 * j + 1
                    chord = np.sqrt((px[si, sj0] - px[si, sj1])**2
                                    + (py[si, sj0] - py[si, sj1])**2
                                    + (pz[si, sj0] - pz[si, sj1])**2)
                    dyc[i, j] = radius * 2.0 * np.arcsin(min(chord / 2.0, 1.0))
            # Extrapolate at cube boundaries.
            dyc[:, 0] = dyc[:, 1]
            dyc[:, n] = dyc[:, n - 1]
        else:
            # n == 1: fall back to original clamped computation.
            for i in range(n):
                for j in range(n + 1):
                    si = 2 * i + 1
                    sj0 = max(2 * j - 1, 0)
                    sj1 = min(2 * j + 1, 2 * n)
                    chord = np.sqrt((px[si, sj0] - px[si, sj1])**2
                                    + (py[si, sj0] - py[si, sj1])**2
                                    + (pz[si, sj0] - pz[si, sj1])**2)
                    dyc[i, j] = radius * 2.0 * np.arcsin(min(chord / 2.0, 1.0))
        all_dyc.append(dyc)

        # --- dxa: cell width in x = face-to-face distance (FV3 fv_grid_tools.F90) ---
        # dxa(i,j) = dist from u-face i to u-face i+1 at cell centre row j
        # On supergrid: u-face i at column 2i, cell-centre row j at row 2j+1
        dxa = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                si0 = 2 * i        # u-face i
                si1 = 2 * (i + 1)  # u-face i+1
                sj = 2 * j + 1     # cell-centre row j
                chord = np.sqrt((px[si0, sj] - px[si1, sj])**2
                                + (py[si0, sj] - py[si1, sj])**2
                                + (pz[si0, sj] - pz[si1, sj])**2)
                dxa[i, j] = radius * 2.0 * np.arcsin(min(chord / 2.0, 1.0))
        all_dxa.append(dxa)

        # --- dya: cell width in y = face-to-face distance ---
        dya = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                si = 2 * i + 1     # cell-centre column i
                sj0 = 2 * j        # v-face j
                sj1 = 2 * (j + 1)  # v-face j+1
                chord = np.sqrt((px[si, sj0] - px[si, sj1])**2
                                + (py[si, sj0] - py[si, sj1])**2
                                + (pz[si, sj0] - pz[si, sj1])**2)
                dya[i, j] = radius * 2.0 * np.arcsin(min(chord / 2.0, 1.0))
        all_dya.append(dya)

    area_c_all = jnp.array(np.stack(all_area_c, axis=0))
    dxc_all = jnp.array(np.stack(all_dxc, axis=0))
    dyc_all = jnp.array(np.stack(all_dyc, axis=0))
    dxa_all = jnp.array(np.stack(all_dxa, axis=0))
    dya_all = jnp.array(np.stack(all_dya, axis=0))
    return area_c_all, dxc_all, dyc_all, dxa_all, dya_all


def _remap_fv3_metrics_to_create(m: dict) -> tuple:
    """Remap FV3-numbered metric fields onto create_cubed_sphere's layout.

    Pure index shuffling (face permutation + rot90) via the phase-1
    ``_GNOMONIC_ED_FACE_PERM/_ROT`` table; rot90 by an odd k maps
    x-staggered fields onto y-staggered positions, so staggered pairs swap.
    Numerics live entirely in :mod:`legoesm.grids.fv3_native_metrics`;
    the oracle test performs the same remap independently on the REFERENCE
    side, so an error here cannot self-cancel.
    """
    import numpy as np

    from legoesm.grids.cubed_sphere import (
        _GNOMONIC_ED_FACE_PERM,
        _GNOMONIC_ED_FACE_ROT,
    )

    def _square(a):
        return np.stack([
            np.rot90(a[_GNOMONIC_ED_FACE_PERM[F]], _GNOMONIC_ED_FACE_ROT[F])
            for F in range(6)
        ])

    def _pair(ax, ay):
        out_x, out_y = [], []
        for F in range(6):
            g = _GNOMONIC_ED_FACE_PERM[F]
            k = _GNOMONIC_ED_FACE_ROT[F]
            if k % 2 == 0:
                out_x.append(np.rot90(ax[g], k))
                out_y.append(np.rot90(ay[g], k))
            else:
                out_x.append(np.rot90(ay[g], k))
                out_y.append(np.rot90(ax[g], k))
        return np.stack(out_x), np.stack(out_y)

    area_c = _square(m["area_c"])
    dxc, dyc = _pair(m["dxc"], m["dyc"])
    dxa, dya = _pair(m["dxa"], m["dya"])
    return (jnp.asarray(area_c), jnp.asarray(dxc), jnp.asarray(dyc),
            jnp.asarray(dxa), jnp.asarray(dya))


def create_cubed_sphere_cdgrid(
    base: CubedSphereGrid,
    omega: float | None = None,
    metric_dtype=None,
    gnomonic: str = "auto",
) -> CubedSphereCDGrid:
    """Create a C-D grid from an existing cell-centre grid.

    Parameters
    ----------
    base : CubedSphereGrid
        cell-centre cubed-sphere with cell-center metrics.
    omega : float or None
        Planetary rotation rate [rad/s].  If ``None`` (default), the
        effective omega is inferred from ``base.f`` and ``base.sin_lat``
        so that ``cdgrid.f_corner`` is consistent with ``base.f`` under
        any rescaling (small-earth, non-rotating §3-1, …).  Pass an
        explicit float only when you specifically need to *override*
        the base grid's Coriolis at the corners (rare).
    metric_dtype : dtype or None
        Dtype for corner-critical metrics (gradient matrix, rsin, rarea,
        cosa, sin_sg, dxc/dyc).  Defaults to float32.
    gnomonic : str, default "auto"
        C/D metric family.  ``"auto"`` follows the base grid's static
        provenance (``base.gnomonic_form``) — the only self-consistent
        choice.  An explicit ``"ed"``/``"equiangular"`` is accepted only
        when it MATCHES that provenance; a contradicting request raises
        ``ValueError`` (mixing metric families across staggers silently
        runs different numerics).

    Returns
    -------
    CubedSphereCDGrid
    """
    if metric_dtype is None:
        try:
            from legoesm.core.precision import get_policy
            metric_dtype = get_policy().storage
        except Exception:
            metric_dtype = jnp.float32
    n = base.n
    radius = base.radius

    # Phase-1 FV3-native grid work: the base CubedSphereGrid now carries
    # STATIC provenance (`gnomonic_form`, pytree aux_data — see the explicit
    # register_pytree_node in cubed_sphere.py), so the C/D metric family is
    # READ from the base grid instead of the old iter72 dx/dy aspect-ratio
    # inference (which was provably blind at n=2 and fragile by construction).
    # "auto" = follow the base grid's provenance.  An explicit request that
    # CONTRADICTS the provenance is the silent-metric-mixing bug class the
    # inference was built to avoid — now a hard error.
    if gnomonic == "auto":
        gnomonic = base.gnomonic_form
    elif gnomonic in ("ed", "equiangular") and gnomonic != base.gnomonic_form:
        raise ValueError(
            f"create_cubed_sphere_cdgrid: requested gnomonic={gnomonic!r} but "
            f"the base grid's provenance is gnomonic_form="
            f"{base.gnomonic_form!r}. Mixing metric families silently ran "
            "different numerics per stagger; rebuild the base grid with the "
            "matching create_cubed_sphere(..., gnomonic=...) instead.")

    # ------------------------------------------------------------------
    # The C-D supergrid metrics all derive from 4 node-grids: the 2n+1
    # supergrid, the n+1 corners, the n+3 extended-corner grid (corner/edge
    # angles), and the 2n+3 padded supergrid (sin_sg).  Build all four for the
    # requested grid type (iter71 cdgrid ed rework) — equiangular default
    # BYTE-IDENTICAL; gnomonic_ed via the construct→mirror→remap helpers.
    # ------------------------------------------------------------------
    if gnomonic == "ed":
        from legoesm.grids.cubed_sphere import (
            gnomonic_ed_supergrid_lonlat, gnomonic_ed_corner_ext_lonlat,
            gnomonic_ed_padded_supergrid_lonlat, make_fv3_native_grid,
            gnomonic_ed_remap_to_create)
        _sg_lon, _sg_lat = gnomonic_ed_supergrid_lonlat(n)            # (6,2n+1,2n+1)
        corner_lon, corner_lat = gnomonic_ed_remap_to_create(
            *make_fv3_native_grid(n, grid_type=0))                   # (6,n+1,n+1)
        corner_ext_lon, corner_ext_lat = gnomonic_ed_corner_ext_lonlat(n)  # (6,n+3,n+3)
        _psg_lon, _psg_lat = gnomonic_ed_padded_supergrid_lonlat(n)  # (6,2n+3,2n+3)
    elif gnomonic == "equiangular":
        _alpha_sg = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, 2 * n + 1)
        _ax_sg, _ay_sg = jnp.meshgrid(_alpha_sg, _alpha_sg, indexing='ij')
        _sg = [face_gnomonic_to_lonlat(f, _ax_sg, _ay_sg) for f in range(6)]
        _sg_lon = jnp.stack([s[0] for s in _sg])
        _sg_lat = jnp.stack([s[1] for s in _sg])
        _alpha_edges = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, n + 1)
        _ax_e, _ay_e = jnp.meshgrid(_alpha_edges, _alpha_edges, indexing='ij')
        _cg = [face_gnomonic_to_lonlat(f, _ax_e, _ay_e) for f in range(6)]
        corner_lon = jnp.stack([c[0] for c in _cg])
        corner_lat = jnp.stack([c[1] for c in _cg])
        _dalpha = jnp.pi / (2 * n)
        _alpha_ext = jnp.linspace(-jnp.pi / 4 - _dalpha, jnp.pi / 4 + _dalpha, n + 3)
        _axe, _aye = jnp.meshgrid(_alpha_ext, _alpha_ext, indexing='ij')
        _ceg = [face_gnomonic_to_lonlat(f, _axe, _aye) for f in range(6)]
        corner_ext_lon = jnp.stack([c[0] for c in _ceg])
        corner_ext_lat = jnp.stack([c[1] for c in _ceg])
        _dasg = jnp.pi / (2 * n)
        _alpha_psg = jnp.linspace(
            -jnp.pi / 4 - _dasg / 2, jnp.pi / 4 + _dasg / 2, 2 * n + 3)
        _axp, _ayp = jnp.meshgrid(_alpha_psg, _alpha_psg, indexing='ij')
        _psg = [face_gnomonic_to_lonlat(f, _axp, _ayp) for f in range(6)]
        _psg_lon = jnp.stack([s[0] for s in _psg])
        _psg_lat = jnp.stack([s[1] for s in _psg])
    else:
        raise ValueError(
            f"gnomonic must be 'equiangular' or 'ed', got {gnomonic!r}")

    if gnomonic == "ed":
        # Phase-2 FV3-native metrics: the exact init_grid/grid_area
        # construction (spherical-excess areas, agrid distances, the ×2
        # half-dual edge conventions — oracle-pinned against the verbatim
        # Fortran extraction in tests/grids/test_fv3_native_metrics_phase2.py)
        # built in FV3 face numbering and remapped to create's layout.  The
        # legacy chord/supergrid approximation below is NEVER used on this
        # path (guard test monkeypatches it to raise).
        import numpy as _np

        from legoesm.grids.cubed_sphere import make_fv3_native_grid
        from legoesm.grids.fv3_native_metrics import (
            compute_fv3_native_metrics,
        )

        _lon6, _lat6 = make_fv3_native_grid(n, grid_type=0)
        _m = compute_fv3_native_metrics(
            _np.asarray(_lon6), _np.asarray(_lat6), radius=radius)
        area_c_sg, dxc_sg, dyc_sg, dxa_sg, dya_sg = (
            _remap_fv3_metrics_to_create(_m))
    else:
        # Legacy supergrid metrics: area_c, dxc, dyc from the 2x-refined
        # supergrid (same supergrid as sin_sg/cos_sg → mutual consistency).
        area_c_sg, dxc_sg, dyc_sg, dxa_sg, dya_sg = _compute_supergrid_metrics(
            n, _sg_lon, _sg_lat, radius)

    all_lon_c, all_lat_c = [], []
    all_angle_c = []
    all_dx_ey, all_dy_ex = [], []

    for face in range(6):
        lon_c, lat_c = corner_lon[face], corner_lat[face]
        all_lon_c.append(lon_c)
        all_lat_c.append(lat_c)

        # Cartesian coordinates at corners for metric computation
        cos_lat_c = jnp.cos(lat_c)
        x_c = cos_lat_c * jnp.cos(lon_c)
        y_c = cos_lat_c * jnp.sin(lon_c)
        z_c = jnp.sin(lat_c)

        # Edge lengths in x-direction (dy_edge_x): distance between
        # consecutive corners in the x-direction → shape (n+1, n)
        # (length of the cell edge crossed by u_c)
        chord_x = jnp.sqrt(
            (x_c[1:, :] - x_c[:-1, :]) ** 2
            + (y_c[1:, :] - y_c[:-1, :]) ** 2
            + (z_c[1:, :] - z_c[:-1, :]) ** 2
        )  # (n, n+1) — between consecutive x-corners, all y-corners
        # Actually: dy_edge_x has shape (n+1, n) — the edge between
        # corner (i,j) and (i,j+1), which is an x-oriented edge
        # u_c lives on x-oriented edges: (n+1, n)
        # These edges connect (i,j) to (i,j+1) — y-direction span
        # Wait, let me reconsider the staggering:
        # u_c at x-interfaces: between cell (i,j) and cell (i+1,j)
        #   The interface has corners at (i+1,j) and (i+1,j+1)
        #   Its length is the great-circle distance between those corners
        # v_c at y-interfaces: between cell (i,j) and cell (i,j+1)
        #   The interface has corners at (i,j+1) and (i+1,j+1)
        #   Its length is the great-circle distance between those corners

        # dy_edge_x: length of x-interface (u_c position), shape (n+1, n)
        # = distance between corner (i, j) and corner (i, j+1)
        chord_dy = jnp.sqrt(
            (x_c[:, 1:] - x_c[:, :-1]) ** 2
            + (y_c[:, 1:] - y_c[:, :-1]) ** 2
            + (z_c[:, 1:] - z_c[:, :-1]) ** 2
        )  # (n+1, n)
        dy_ex = radius * 2.0 * jnp.arcsin(jnp.clip(chord_dy / 2.0, 0.0, 1.0))
        all_dy_ex.append(dy_ex)

        # dx_edge_y: length of y-interface (v_c position), shape (n, n+1)
        # = distance between corner (i, j) and corner (i+1, j)
        chord_dx = jnp.sqrt(
            (x_c[1:, :] - x_c[:-1, :]) ** 2
            + (y_c[1:, :] - y_c[:-1, :]) ** 2
            + (z_c[1:, :] - z_c[:-1, :]) ** 2
        )  # (n, n+1)
        dx_ey = radius * 2.0 * jnp.arcsin(jnp.clip(chord_dx / 2.0, 0.0, 1.0))
        all_dx_ey.append(dx_ey)

        # Grid angle at corners: computed from Cartesian tangent vectors
        # on the precomputed extended grid (analytical, no centred-diff error).
        lon_ext, lat_ext = corner_ext_lon[face], corner_ext_lat[face]
        cos_lat_ext_full = jnp.cos(lat_ext)
        # Cartesian positions on unit sphere (extended grid)
        px_ext = cos_lat_ext_full * jnp.cos(lon_ext)
        py_ext = cos_lat_ext_full * jnp.sin(lon_ext)
        pz_ext = jnp.sin(lat_ext)

        # i-tangent vector at each corner via centred diff of Cartesian
        # positions → (n+1, n+1) after stripping 1 on each side
        ti_x = px_ext[2:, 1:-1] - px_ext[:-2, 1:-1]
        ti_y = py_ext[2:, 1:-1] - py_ext[:-2, 1:-1]
        ti_z = pz_ext[2:, 1:-1] - pz_ext[:-2, 1:-1]
        # Project onto tangent plane at corner
        cx = px_ext[1:-1, 1:-1]
        cy = py_ext[1:-1, 1:-1]
        cz = pz_ext[1:-1, 1:-1]
        dot_i = ti_x * cx + ti_y * cy + ti_z * cz
        ti_x -= dot_i * cx; ti_y -= dot_i * cy; ti_z -= dot_i * cz
        norm_i = jnp.sqrt(ti_x**2 + ti_y**2 + ti_z**2 + _TINY)
        ti_x /= norm_i; ti_y /= norm_i; ti_z /= norm_i

        # Grid angle = angle between i-tangent and geographic east
        sin_lon_c = jnp.sin(lon_ext[1:-1, 1:-1])
        cos_lon_c = jnp.cos(lon_ext[1:-1, 1:-1])
        sin_lat_c = jnp.sin(lat_ext[1:-1, 1:-1])
        cos_lat_c = jnp.cos(lat_ext[1:-1, 1:-1])
        # e_east = (-sin_lon, cos_lon, 0)
        # e_north = (-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat)
        ti_dot_east = -sin_lon_c * ti_x + cos_lon_c * ti_y
        ti_dot_north = (-sin_lat_c * cos_lon_c * ti_x
                        - sin_lat_c * sin_lon_c * ti_y
                        + cos_lat_c * ti_z)
        face_angle = jnp.arctan2(ti_dot_north, ti_dot_east)
        all_angle_c.append(face_angle)

    lon_corner = jnp.stack(all_lon_c, axis=0)
    lat_corner = jnp.stack(all_lat_c, axis=0)
    angle_corner = jnp.stack(all_angle_c, axis=0)
    dx_edge_y = jnp.stack(all_dx_ey, axis=0)
    dy_edge_x = jnp.stack(all_dy_ex, axis=0)

    # ------------------------------------------------------------------
    # Owner-based sync of corner lon/lat at shared edges/vertices.
    # Each shared edge is owned by the lower face index; each shared
    # vertex is owned by the lowest face index among the 3 faces
    # meeting there.  This ensures that lon_corner and lat_corner are
    # bitwise identical on both sides of every face boundary.
    #
    # NOTE: angle_corner is NOT synced because it is face-local (the
    # angle between that face's i-tangent and geographic east), and is
    # therefore inherently different on each face even at shared points.
    # ------------------------------------------------------------------
    def _get_strip_corner(arr, face, edge, n_):
        if edge == WEST:    return arr[face, 0, :]
        elif edge == EAST:  return arr[face, n_, :]
        elif edge == SOUTH: return arr[face, :, 0]
        else:               return arr[face, :, n_]

    for face in range(6):
        for edge in [WEST, EAST, SOUTH, NORTH]:
            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]
            if nbr_face >= face:
                continue  # only non-owner faces copy from owner
            for arr_name in ['lon_corner', 'lat_corner']:
                arr = locals()[arr_name]
                owner_strip = _get_strip_corner(arr, nbr_face, nbr_edge, n)
                if is_reversed:
                    owner_strip = owner_strip[::-1]
                if edge == WEST:
                    arr = arr.at[face, 0, :].set(owner_strip)
                elif edge == EAST:
                    arr = arr.at[face, n, :].set(owner_strip)
                elif edge == SOUTH:
                    arr = arr.at[face, :, 0].set(owner_strip)
                else:
                    arr = arr.at[face, :, n].set(owner_strip)
                if arr_name == 'lon_corner':
                    lon_corner = arr
                else:
                    lat_corner = arr

    # Vertex sync: lowest face index owns
    _vtx_corners = [
        [(0, 0, 0), (3, n, 0), (5, 0, n)],
        [(0, n, 0), (1, 0, 0), (5, n, n)],
        [(0, 0, n), (3, n, n), (4, 0, 0)],
        [(0, n, n), (1, 0, n), (4, n, 0)],
        [(1, n, 0), (2, 0, 0), (5, n, 0)],
        [(1, n, n), (2, 0, n), (4, n, n)],
        [(2, n, 0), (3, 0, 0), (5, 0, 0)],
        [(2, n, n), (3, 0, n), (4, 0, n)],
    ]
    for vtx in _vtx_corners:
        owner = min(vtx, key=lambda x: x[0])
        for arr_name in ['lon_corner', 'lat_corner']:
            arr = locals()[arr_name]
            val = arr[owner[0], owner[1], owner[2]]
            for f, i, j in vtx:
                if (f, i, j) != owner:
                    arr = arr.at[f, i, j].set(val)
            if arr_name == 'lon_corner':
                lon_corner = arr
            else:
                lat_corner = arr

    # --- area_corner from FV3 supergrid (sum of 4 supergrid quadrilaterals) ---
    area_corner = area_c_sg  # (6, n+1, n+1)

    # Infer omega from base.f when not explicitly provided so that
    # `cdgrid.f_corner` remains consistent with `base.f` under any
    # rescaling (e.g. `small_earth_factor` in NH model tests).  Use
    # the cell with the largest |sin(lat)| for numerical stability.
    # Falls back to Earth's default if the inference is not reliable
    # (e.g. base.f/sin_lat cannot be reduced to a concrete scalar,
    # or the base grid has sin_lat effectively zero everywhere).
    if omega is None:
        try:
            flat_abs_sl = jnp.abs(base.sin_lat).reshape(-1)
            idx = int(jnp.argmax(flat_abs_sl))
            sl_probe = float(base.sin_lat.reshape(-1)[idx])
            f_probe = float(base.f.reshape(-1)[idx])
            if abs(sl_probe) > 1e-6:
                omega = f_probe / (2.0 * sl_probe)
            else:
                omega = constants.Omega  # fallback: sin_lat ~ 0 everywhere
        except (TypeError, jax.errors.ConcretizationTypeError):
            # base.f or base.sin_lat is a JAX tracer (e.g. JIT-time
            # grid construction). Fall back to the Earth default.
            omega = constants.Omega
    f_corner = 2.0 * omega * jnp.sin(lat_corner)
    cos_angle_corner = jnp.cos(angle_corner)
    sin_angle_corner = jnp.sin(angle_corner)

    # ------------------------------------------------------------------
    # Non-orthogonality metrics (FV3 cos_sg / sin_sg)
    # At each D-grid corner, compute the cosine of the angle between
    # the i-tangent and j-tangent vectors using Cartesian positions on
    # the unit sphere from an extended gnomonic grid.
    # ------------------------------------------------------------------
    # Reuse the precomputed (6, n+3, n+3) extended corner grid (same n+3
    # layout as the corner grid-angle); grid-type-agnostic (iter71).
    all_cosa_c = []
    all_z21_c = []
    all_z22_c = []
    for face in range(6):
        lon_ext, lat_ext = corner_ext_lon[face], corner_ext_lat[face]
        cos_lat_ext = jnp.cos(lat_ext)
        # Cartesian positions on unit sphere
        px = cos_lat_ext * jnp.cos(lon_ext)
        py = cos_lat_ext * jnp.sin(lon_ext)
        pz = jnp.sin(lat_ext)

        # Centre positions (n+1 × n+1) at indices [1:-1, 1:-1]
        cx = px[1:-1, 1:-1]
        cy = py[1:-1, 1:-1]
        cz = pz[1:-1, 1:-1]

        # i-tangent: centred diff in axis-0 → (n+1, n+1)
        ti_x = px[2:, 1:-1] - px[:-2, 1:-1]
        ti_y = py[2:, 1:-1] - py[:-2, 1:-1]
        ti_z = pz[2:, 1:-1] - pz[:-2, 1:-1]
        # Project onto tangent plane: t -= (t·P)*P
        dot_i = ti_x * cx + ti_y * cy + ti_z * cz
        ti_x = ti_x - dot_i * cx
        ti_y = ti_y - dot_i * cy
        ti_z = ti_z - dot_i * cz
        norm_i = jnp.sqrt(ti_x**2 + ti_y**2 + ti_z**2 + _TINY)
        ti_x /= norm_i; ti_y /= norm_i; ti_z /= norm_i

        # j-tangent: centred diff in axis-1 → (n+1, n+1)
        tj_x = px[1:-1, 2:] - px[1:-1, :-2]
        tj_y = py[1:-1, 2:] - py[1:-1, :-2]
        tj_z = pz[1:-1, 2:] - pz[1:-1, :-2]
        dot_j = tj_x * cx + tj_y * cy + tj_z * cz
        tj_x = tj_x - dot_j * cx
        tj_y = tj_y - dot_j * cy
        tj_z = tj_z - dot_j * cz
        norm_j = jnp.sqrt(tj_x**2 + tj_y**2 + tj_z**2 + _TINY)
        tj_x /= norm_j; tj_y /= norm_j; tj_z /= norm_j

        # cosa = dot(t_i, t_j)
        cosa_face = ti_x * tj_x + ti_y * tj_y + ti_z * tj_z
        all_cosa_c.append(cosa_face)

        # iter3: corner z-matrix rows for the j-tangent (FV3 c2l z21/z22).
        # z21 = ec2·east, z22 = ec2·north — the exact non-orthogonal
        # covariant→geographic coefficients used by the BGRID_NE corner sync
        # (synchronize_bgrid_ne_corner_geo).  (z11=ec1·east=cos_angle_corner,
        # z12=ec1·north=sin_angle_corner already stored.)  Geographic unit
        # vectors at the corner (lon,lat) = lon_ext/lat_ext[1:-1,1:-1].
        clon = lon_ext[1:-1, 1:-1]; clat = lat_ext[1:-1, 1:-1]
        s_lon = jnp.sin(clon); c_lon = jnp.cos(clon)
        s_lat = jnp.sin(clat); c_lat = jnp.cos(clat)
        # east = (-sin_lon, cos_lon, 0); north = (-sin_lat cos_lon, -sin_lat sin_lon, cos_lat)
        z21_face = -s_lon * tj_x + c_lon * tj_y
        z22_face = (-s_lat * c_lon * tj_x - s_lat * s_lon * tj_y + c_lat * tj_z)
        all_z21_c.append(z21_face)
        all_z22_c.append(z22_face)

    cosa_corner = jnp.stack(all_cosa_c, axis=0)
    z21_corner = jnp.stack(all_z21_c, axis=0)
    z22_corner = jnp.stack(all_z22_c, axis=0)
    sina_corner = jnp.sqrt(jnp.maximum(1.0 - cosa_corner**2, _EPS))
    rsin2_corner = 1.0 / jnp.maximum(sina_corner**2, _EPS)

    # ------------------------------------------------------------------
    # Duo-Grid sub-grid metrics (sin_sg, cos_sg) at 9 positions per cell.
    # Computed BEFORE C-grid face metrics so cosa_u/rsin_u can be derived
    # from sin_sg/cos_sg following FV3 fv_grid_utils.F90:505-518.
    # 0-indexed: 0=W, 1=S, 2=E, 3=N (edge midpoints); 4=center;
    #            5=SW, 6=SE, 7=NE, 8=NW (corners)
    # ------------------------------------------------------------------
    # Padded supergrid (2n+3 per axis) for sin_sg/cos_sg — built above per
    # grid type (`_psg_lon/_psg_lat`).
    sin_sg, cos_sg = _compute_sin_cos_sg(n, _psg_lon, _psg_lat)

    # ------------------------------------------------------------------
    # C-grid face metrics from sin_sg/cos_sg (FV3 fv_grid_utils.F90:505-518)
    #
    # FV3 convention:
    #   cosa_u(i,j) = 0.5*(cos_sg(i-1,j,E) + cos_sg(i,j,W))
    #   sina_u(i,j) = 0.5*(sin_sg(i-1,j,E) + sin_sg(i,j,W))
    #   rsin_u = 1/sin² (interior), 1/sin (panel edges)
    # ------------------------------------------------------------------
    cos_sg_W = cos_sg[:, :, :, 0]  # (6, n, n)
    cos_sg_E = cos_sg[:, :, :, 2]
    cos_sg_S = cos_sg[:, :, :, 1]
    cos_sg_N = cos_sg[:, :, :, 3]
    sin_sg_W = sin_sg[:, :, :, 0]
    sin_sg_E = sin_sg[:, :, :, 2]
    sin_sg_S = sin_sg[:, :, :, 1]
    sin_sg_N = sin_sg[:, :, :, 3]

    # u-faces (6, n+1, n): average E-edge of left cell + W-edge of right cell
    # FV3 fv_grid_utils.F90:507-508: cosa_u(i,j) = 0.5*(cos_sg(i-1,j,3)+cos_sg(i,j,1))
    # Interior: both cells on same face → direct average.
    # Boundary: cos_sg sub-grid positions are face-local, so cross-face halo
    # gives wrong sub-grid values. Use local cell edge value (geometrically
    # exact: both sides of the face boundary measure the same angle).
    cosa_u_int = 0.5 * (cos_sg_E[:, :-1, :] + cos_sg_W[:, 1:, :])  # (6, n-1, n)
    sina_u_int = 0.5 * (sin_sg_E[:, :-1, :] + sin_sg_W[:, 1:, :])
    cosa_u = jnp.concatenate([
        cos_sg_W[:, :1, :], cosa_u_int, cos_sg_E[:, -1:, :]
    ], axis=1)  # (6, n+1, n)
    sina_u = jnp.concatenate([
        sin_sg_W[:, :1, :], sina_u_int, sin_sg_E[:, -1:, :]
    ], axis=1)

    # v-faces (6, n, n+1): average N-edge of bottom cell + S-edge of top cell
    cosa_v_int = 0.5 * (cos_sg_N[:, :, :-1] + cos_sg_S[:, :, 1:])  # (6, n, n-1)
    sina_v_int = 0.5 * (sin_sg_N[:, :, :-1] + sin_sg_S[:, :, 1:])
    cosa_v = jnp.concatenate([
        cos_sg_S[:, :, :1], cosa_v_int, cos_sg_N[:, :, -1:]
    ], axis=2)  # (6, n, n+1)
    sina_v = jnp.concatenate([
        sin_sg_S[:, :, :1], sina_v_int, sin_sg_N[:, :, -1:]
    ], axis=2)

    # rsin_u/rsin_v follow FV3 fv_grid_utils.F90:509-561:
    #   - Interior u/v faces: rsin_u = 1/sina_u² (line 509, 517)
    #   - Panel edges (i=1 or i=npx for rsin_u; j=1 or j=npy for rsin_v):
    #     rsin_u = 1/sina_u (lines 548-561) — ONLY when .not. bounded_domain
    # Gating matches Fortran bounded_domain = (regional .or. nested .or.
    # duogrid) at fv_arrays.F90:1512.  In legoESM:
    #   - duogrid → base.duogrid is not None
    #   - regional/nested → single-face panel (base.lat.shape[0] == 1; see
    #     create_cubed_sphere_panel and the `data.shape[0] == 1` branch in
    #     pad_halo).  For a single-face panel the outer perimeter is a
    #     physical wall, not a cube seam, and panel-edge rsin is not used.
    rsin_u = 1.0 / jnp.maximum(sina_u**2, _EPS)
    rsin_v = 1.0 / jnp.maximum(sina_v**2, _EPS)
    # Use the backend-aware property (single source of truth) rather than
    # re-deriving from ``lat.shape[0] == 1``: under cubed-sphere MPI
    # face-scatter a rank can own a SINGLE global cube face (shape[0] == 1)
    # that is NOT a regional panel — its panel edges ARE cube seams handled by
    # MPI halo exchange, so the rsin_u/rsin_v seam override must still apply.
    _bounded_domain = base.bounded_domain
    if not _bounded_domain:
        # Panel-edge override: replace 1/sin² with 1/sin at i==0 and i==n
        # for rsin_u, and j==0 and j==n for rsin_v.
        rsin_u_edge = 1.0 / jnp.sign(sina_u) / jnp.maximum(jnp.abs(sina_u), _EPS)
        rsin_u = rsin_u.at[:, 0, :].set(rsin_u_edge[:, 0, :])
        rsin_u = rsin_u.at[:, -1, :].set(rsin_u_edge[:, -1, :])
        rsin_v_edge = 1.0 / jnp.sign(sina_v) / jnp.maximum(jnp.abs(sina_v), _EPS)
        rsin_v = rsin_v.at[:, :, 0].set(rsin_v_edge[:, :, 0])
        rsin_v = rsin_v.at[:, :, -1].set(rsin_v_edge[:, :, -1])

    # ------------------------------------------------------------------
    # FV3 edge-midpoint D-grid metrics
    # ------------------------------------------------------------------
    # Edge midpoints are computed via Cartesian averaging on the unit
    # sphere (avoids longitude wrapping issues at the dateline).
    # x-edge midpoint: midpoint of corners (i,j) and (i+1,j) → (6,n,n+1)
    # y-edge midpoint: midpoint of corners (i,j) and (i,j+1) → (6,n+1,n)

    cos_lat_corner = jnp.cos(lat_corner)
    xc = cos_lat_corner * jnp.cos(lon_corner)
    yc = cos_lat_corner * jnp.sin(lon_corner)
    zc = jnp.sin(lat_corner)

    # --- x-edge midpoints (6, n, n+1) ---
    mx_x = 0.5 * (xc[:, :-1, :] + xc[:, 1:, :])
    mx_y = 0.5 * (yc[:, :-1, :] + yc[:, 1:, :])
    mx_z = 0.5 * (zc[:, :-1, :] + zc[:, 1:, :])
    # Re-project onto sphere
    mx_norm = jnp.sqrt(mx_x**2 + mx_y**2 + mx_z**2 + _TINY)
    mx_x /= mx_norm; mx_y /= mx_norm; mx_z /= mx_norm
    lat_edge_x = jnp.arcsin(jnp.clip(mx_z, -1.0, 1.0))
    lon_edge_x = jnp.mod(jnp.arctan2(mx_y, mx_x), 2.0 * jnp.pi)  # [0, 2π)

    # --- y-edge midpoints (6, n+1, n) ---
    my_x = 0.5 * (xc[:, :, :-1] + xc[:, :, 1:])
    my_y = 0.5 * (yc[:, :, :-1] + yc[:, :, 1:])
    my_z = 0.5 * (zc[:, :, :-1] + zc[:, :, 1:])
    my_norm = jnp.sqrt(my_x**2 + my_y**2 + my_z**2 + _TINY)
    my_x /= my_norm; my_y /= my_norm; my_z /= my_norm
    lat_edge_y = jnp.arcsin(jnp.clip(my_z, -1.0, 1.0))
    lon_edge_y = jnp.mod(jnp.arctan2(my_y, my_x), 2.0 * jnp.pi)  # [0, 2π)

    # --- Grid angle at edge midpoints ---
    # Use the Cartesian tangent-vector approach on the extended gnomonic
    # grid.  For x-edges we need i-tangent vectors at (n, n+1) positions,
    # for y-edges at (n+1, n) positions.
    all_angle_ex = []
    all_angle_ey = []
    # iter71: the edge-midpoint grid angles are computed directly from the
    # Cartesian corner positions (xc/yc/zc, from the precomputed corner grid)
    # and edge midpoints (mx/my) — grid-type-agnostic.  The former equiangular
    # `alpha_ext2`/`alpha_half_*` (n+4 / half-integer linspace) were DEAD CODE
    # (computed, never used) and are removed.

    for face in range(6):
        # --- Grid angle at x-edge midpoints (n, n+1) ---
        # Use finite-difference of Cartesian positions along gnomonic i
        # at the edge midpoint.  The i-tangent at edge midpoint (i+0.5, j)
        # is proportional to corner(i+1,j) - corner(i,j).
        # For the grid angle we need the i-tangent projected onto the
        # tangent plane, then its angle with geographic east.

        # x-edge: i-tangent from the two adjacent corners
        ti_xe = xc[face, 1:, :] - xc[face, :-1, :]  # (n, n+1)
        ti_ye = yc[face, 1:, :] - yc[face, :-1, :]
        ti_ze = zc[face, 1:, :] - zc[face, :-1, :]
        # Project onto tangent plane at midpoint
        dot_ie = ti_xe * mx_x[face] + ti_ye * mx_y[face] + ti_ze * mx_z[face]
        ti_xe = ti_xe - dot_ie * mx_x[face]
        ti_ye = ti_ye - dot_ie * mx_y[face]
        ti_ze = ti_ze - dot_ie * mx_z[face]
        norm_ie = jnp.sqrt(ti_xe**2 + ti_ye**2 + ti_ze**2 + _TINY)
        ti_xe /= norm_ie; ti_ye /= norm_ie; ti_ze /= norm_ie

        # Grid angle: angle between i-tangent and geographic east
        sin_lon_ex = jnp.sin(lon_edge_x[face])
        cos_lon_ex = jnp.cos(lon_edge_x[face])
        sin_lat_ex = jnp.sin(lat_edge_x[face])
        cos_lat_ex = jnp.cos(lat_edge_x[face])
        ti_dot_east_ex = -sin_lon_ex * ti_xe + cos_lon_ex * ti_ye
        ti_dot_north_ex = (-sin_lat_ex * cos_lon_ex * ti_xe
                           - sin_lat_ex * sin_lon_ex * ti_ye
                           + cos_lat_ex * ti_ze)
        angle_ex = jnp.arctan2(ti_dot_north_ex, ti_dot_east_ex)
        all_angle_ex.append(angle_ex)

        # --- Grid angle at y-edge midpoints (n+1, n) ---
        # i-tangent at y-edge midpoint (i, j+0.5): use centred diff from
        # corners (i-1, j+0.5) and (i+1, j+0.5).  Approximate by averaging
        # the i-tangent from the two adjacent corner rows.
        # A cleaner approach: use the extended gnomonic grid evaluated at
        # half-integer j positions.  But for simplicity and consistency,
        # use the same Cartesian tangent approach with corner differences.

        # For y-edge midpoint (i, j+0.5), the i-tangent can be approximated
        # from the average of the two neighboring x-edge tangent directions.
        # However, the most direct approach: use centred diff on corners.
        # i-tangent at (i, j+0.5) ≈ avg of (corner(i+1,j)-corner(i-1,j))
        # at j and j+1.  We need padding for the boundary.

        # Use the precomputed (6, n+3, n+3) extended grid for y-edge tangents
        lon_ext_f, lat_ext_f = corner_ext_lon[face], corner_ext_lat[face]
        cos_lat_extf = jnp.cos(lat_ext_f)
        px_e = cos_lat_extf * jnp.cos(lon_ext_f)
        py_e = cos_lat_extf * jnp.sin(lon_ext_f)
        pz_e = jnp.sin(lat_ext_f)

        # i-tangent at extended corners via centred diff: (n+1, n+1)
        # Extended grid indices [1:-1, 1:-1] correspond to corner grid.
        # For y-edge midpoints (n+1, n) we average j and j+1 slices.
        # i-tangent at extended: diff in axis-0 → (n+1, n+3)
        ti_ext_x = px_e[2:, :] - px_e[:-2, :]  # (n+1, n+3)
        ti_ext_y = py_e[2:, :] - py_e[:-2, :]
        ti_ext_z = pz_e[2:, :] - pz_e[:-2, :]
        # Take columns [1:-1] to get (n+1, n+1), then average j, j+1
        ti_ey_x = 0.5 * (ti_ext_x[:, 1:-2] + ti_ext_x[:, 2:-1])  # (n+1, n)
        ti_ey_y = 0.5 * (ti_ext_y[:, 1:-2] + ti_ext_y[:, 2:-1])
        ti_ey_z = 0.5 * (ti_ext_z[:, 1:-2] + ti_ext_z[:, 2:-1])

        # Project onto tangent plane at y-edge midpoint
        dot_jey = (ti_ey_x * my_x[face] + ti_ey_y * my_y[face]
                   + ti_ey_z * my_z[face])
        ti_ey_x = ti_ey_x - dot_jey * my_x[face]
        ti_ey_y = ti_ey_y - dot_jey * my_y[face]
        ti_ey_z = ti_ey_z - dot_jey * my_z[face]
        norm_jey = jnp.sqrt(ti_ey_x**2 + ti_ey_y**2 + ti_ey_z**2 + _TINY)
        ti_ey_x /= norm_jey; ti_ey_y /= norm_jey; ti_ey_z /= norm_jey

        sin_lon_ey = jnp.sin(lon_edge_y[face])
        cos_lon_ey = jnp.cos(lon_edge_y[face])
        sin_lat_ey = jnp.sin(lat_edge_y[face])
        cos_lat_ey = jnp.cos(lat_edge_y[face])
        ti_dot_east_ey = -sin_lon_ey * ti_ey_x + cos_lon_ey * ti_ey_y
        ti_dot_north_ey = (-sin_lat_ey * cos_lon_ey * ti_ey_x
                           - sin_lat_ey * sin_lon_ey * ti_ey_y
                           + cos_lat_ey * ti_ey_z)
        angle_ey = jnp.arctan2(ti_dot_north_ey, ti_dot_east_ey)
        all_angle_ey.append(angle_ey)

    angle_edge_x = jnp.stack(all_angle_ex, axis=0)   # (6, n, n+1)
    angle_edge_y = jnp.stack(all_angle_ey, axis=0)   # (6, n+1, n)
    cos_angle_edge_x = jnp.cos(angle_edge_x)
    sin_angle_edge_x = jnp.sin(angle_edge_x)
    cos_angle_edge_y = jnp.cos(angle_edge_y)
    sin_angle_edge_y = jnp.sin(angle_edge_y)

    # Coriolis at edge midpoints
    f_edge_x = 2.0 * omega * jnp.sin(lat_edge_x)  # (6, n, n+1)
    f_edge_y = 2.0 * omega * jnp.sin(lat_edge_y)  # (6, n+1, n)

    # Cell-centre non-orthogonality from sin_sg centre (FV3 cosa_s/rsin2)
    cosa_cell = cos_sg[:, :, :, 4]   # (6, n, n) — cell centre angle
    sina_cell = sin_sg[:, :, :, 4]
    rsin2_cell = 1.0 / jnp.maximum(sina_cell**2, _EPS)

    # ------------------------------------------------------------------
    # Precompute Arakawa-Lamb gradient transformation matrix.
    #
    # At each D-grid corner the 4-point stencil produces raw differences
    #   ΔB_x = (B_se + B_ne) - (B_sw + B_nw)
    #   ΔB_y = (B_nw + B_ne) - (B_sw + B_se)
    # The physical gradient is then:
    #   dB/dx     = grad_c00 * ΔB_x + grad_c01 * ΔB_y
    #   dB/d_perp = grad_c10 * ΔB_x + grad_c11 * ΔB_y
    #
    # The matrix is the inverse of the 2×2 system formed by projecting
    # the 3D displacement vectors of the haloed cell-center stencil onto
    # the face-local tangent directions (e_x, e_perp) at each corner.
    # This gives a physically correct gradient everywhere — including at
    # face boundaries and cube vertices where face-local dx/dy metrics
    # are inconsistent across faces.
    # ------------------------------------------------------------------

    # Cell-centre 3D Cartesian positions on the unit sphere
    cos_lat_cc = jnp.cos(base.lat)
    x_cc = cos_lat_cc * jnp.cos(base.lon)   # (6, n, n)
    y_cc = cos_lat_cc * jnp.sin(base.lon)
    z_cc = jnp.sin(base.lat)

    # Pad with the same halo exchange used for field values at runtime.
    # Interpolation of positions is required: without it, the nearest-
    # neighbor positional mismatch (up to 0.5 cells near cube vertices)
    # creates O(1) errors in the gradient transformation matrix.
    # When duogrid is active on the base grid, route through the duogrid
    # kinked-to-extended remap so the precomputed `grad_c00..c11` matrix
    # is consistent with `pad_halo_auto`-routed field halos used by the
    # runtime `arakawa_lamb_gradient` operator.  Previously these
    # positions were pinned to `interp_offsets` even in duogrid mode,
    # creating a silent inconsistency between grid-build and runtime halos.
    _base_dg = base.duogrid
    _pos_offs = None if _base_dg is not None else base.halo_interp_offsets
    x_pad = pad_halo(x_cc, interp_offsets=_pos_offs, duogrid=_base_dg)
    y_pad = pad_halo(y_cc, interp_offsets=_pos_offs, duogrid=_base_dg)
    z_pad = pad_halo(z_cc, interp_offsets=_pos_offs, duogrid=_base_dg)
    x_pad = fill_corners_h1(x_pad)
    y_pad = fill_corners_h1(y_pad)
    z_pad = fill_corners_h1(z_pad)

    # ------------------------------------------------------------------
    # FV3 c_sw metrics: center-to-center distances from supergrid.
    # ------------------------------------------------------------------
    dxc = dxc_sg   # (6, n+1, n) from supergrid
    dyc = dyc_sg   # (6, n, n+1) from supergrid

    rdxc = 1.0 / jnp.maximum(dxc, _TINY)
    rdyc = 1.0 / jnp.maximum(dyc, _TINY)
    rarea_c = 1.0 / jnp.maximum(area_corner, _TINY)

    # FV3 cell-width metrics: exact face-to-face distances from supergrid
    rdxa = 1.0 / jnp.maximum(dxa_sg, _TINY)  # (6, n, n)
    rdya = 1.0 / jnp.maximum(dya_sg, _TINY)  # (6, n, n)

    # 4-point stencil cell positions at each corner
    x_sw, x_se = x_pad[:, :-1, :-1], x_pad[:, 1:, :-1]
    x_nw, x_ne = x_pad[:, :-1, 1:], x_pad[:, 1:, 1:]
    y_sw, y_se = y_pad[:, :-1, :-1], y_pad[:, 1:, :-1]
    y_nw, y_ne = y_pad[:, :-1, 1:], y_pad[:, 1:, 1:]
    z_sw, z_se = z_pad[:, :-1, :-1], z_pad[:, 1:, :-1]
    z_nw, z_ne = z_pad[:, :-1, 1:], z_pad[:, 1:, 1:]

    # Displacement vectors (east−west, north−south) in 3D Cartesian
    drx_x = (x_se + x_ne) - (x_sw + x_nw)
    drx_y = (y_se + y_ne) - (y_sw + y_nw)
    drx_z = (z_se + z_ne) - (z_sw + z_nw)
    dry_x = (x_nw + x_ne) - (x_sw + x_se)
    dry_y = (y_nw + y_ne) - (y_sw + y_se)
    dry_z = (z_nw + z_ne) - (z_sw + z_se)

    # Corner unit normal = position on unit sphere (reuse xc, yc, zc
    # computed above for edge-midpoint metrics)
    _nx, _ny, _nz = xc, yc, zc

    # e_x: face-local i-tangent direction at each corner
    # e_x = cos(angle) * ê_east + sin(angle) * ê_north
    _sl = jnp.sin(lon_corner)
    _cl = jnp.cos(lon_corner)
    _slat = jnp.sin(lat_corner)
    _ca = cos_angle_corner
    _sa = sin_angle_corner
    _ex_x = _ca * (-_sl) + _sa * (-_slat * _cl)
    _ex_y = _ca * _cl + _sa * (-_slat * _sl)
    _ex_z = _sa * cos_lat_corner

    # e_perp = n̂ × e_x  (tangent direction perpendicular to e_x)
    _ep_x = _ny * _ex_z - _nz * _ex_y
    _ep_y = _nz * _ex_x - _nx * _ex_z
    _ep_z = _nx * _ex_y - _ny * _ex_x

    # 2×2 matrix elements: a_ij = Δr_i · e_j
    # (dot product with tangent vectors auto-projects onto tangent plane)
    _a11 = drx_x * _ex_x + drx_y * _ex_y + drx_z * _ex_z
    _a12 = drx_x * _ep_x + drx_y * _ep_y + drx_z * _ep_z
    _a21 = dry_x * _ex_x + dry_y * _ex_y + dry_z * _ex_z
    _a22 = dry_x * _ep_x + dry_y * _ep_y + dry_z * _ep_z

    # Inverse matrix scaled by 1/radius → gradient coefficients
    _det = _a11 * _a22 - _a12 * _a21
    _inv_Rd = 1.0 / (radius * jnp.maximum(_det, 1e-20))
    grad_c00 = _a22 * _inv_Rd
    grad_c01 = -_a12 * _inv_Rd
    grad_c10 = -_a21 * _inv_Rd
    grad_c11 = _a11 * _inv_Rd

    # Base dtype follows the parent grid. Metric dtype may be higher
    # precision for corner-critical operators (gradient matrix, rsin, etc).
    _base = base.lon.dtype if hasattr(base.lon, 'dtype') else jnp.float32
    _prec = metric_dtype
    return CubedSphereCDGrid(
        base=base,
        lon_corner=lon_corner.astype(_base),
        lat_corner=lat_corner.astype(_base),
        f_corner=f_corner.astype(_base),
        angle_corner=angle_corner.astype(_base),
        cos_angle_corner=cos_angle_corner.astype(_base),
        sin_angle_corner=sin_angle_corner.astype(_base),
        dx_edge_y=dx_edge_y.astype(_base),
        dy_edge_x=dy_edge_x.astype(_base),
        area_corner=area_corner.astype(_prec),
        cosa_corner=cosa_corner.astype(_prec),
        rsin2_corner=rsin2_corner.astype(_prec),
        z21_corner=z21_corner.astype(_prec),
        z22_corner=z22_corner.astype(_prec),
        cosa_u=cosa_u.astype(_prec),
        cosa_v=cosa_v.astype(_prec),
        rsin_u=rsin_u.astype(_prec),
        rsin_v=rsin_v.astype(_prec),
        lon_edge_x=lon_edge_x.astype(_base),
        lat_edge_x=lat_edge_x.astype(_base),
        lon_edge_y=lon_edge_y.astype(_base),
        lat_edge_y=lat_edge_y.astype(_base),
        angle_edge_x=angle_edge_x.astype(_base),
        cos_angle_edge_x=cos_angle_edge_x.astype(_base),
        sin_angle_edge_x=sin_angle_edge_x.astype(_base),
        angle_edge_y=angle_edge_y.astype(_base),
        cos_angle_edge_y=cos_angle_edge_y.astype(_base),
        sin_angle_edge_y=sin_angle_edge_y.astype(_base),
        f_edge_x=f_edge_x.astype(_base),
        f_edge_y=f_edge_y.astype(_base),
        cosa_cell=cosa_cell.astype(_prec),
        sina_cell=sina_cell.astype(_prec),
        rsin2_cell=rsin2_cell.astype(_prec),
        grad_c00=grad_c00.astype(_prec),
        grad_c01=grad_c01.astype(_prec),
        grad_c10=grad_c10.astype(_prec),
        grad_c11=grad_c11.astype(_prec),
        sin_sg=sin_sg.astype(_prec),
        cos_sg=cos_sg.astype(_prec),
        dxc=dxc.astype(_prec),
        dyc=dyc.astype(_prec),
        rdxc=rdxc.astype(_prec),
        rdyc=rdyc.astype(_prec),
        rarea_c=rarea_c.astype(_prec),
        rdxa=rdxa.astype(_prec),
        rdya=rdya.astype(_prec),
    )


# ==============================================================================
# Diagnostic helper: 4-edge angle averaging for D-grid → geographic regrid
# ==============================================================================

def cell_centre_angles_from_4edge(cdgrid):
    """4-surrounding-edge mean of grid angle for cell-centre D→geo regrid.

    Returns (cos_alpha, sin_alpha) of shape (6, n, n) where each
    cell-centre angle is the average of the 4 surrounding edge-midpoint
    angles (2 x-edges + 2 y-edges).  This is the matrix's W2 v-wind
    regrid path (see `run_atmosphere_test_matrix.py:1213-1233`):
    using cell-centre angles instead introduces an O(dx) v_north
    residual on Williamson 2 (~0.39 m/s); the 4-edge mean reduces
    this to ~0.008 m/s at t=0 (47x improvement, see iter-25/26 of
    fv3_fortran_fidelity_review.md).

    Iter-528: extracted from the matrix's inline `extract_fn` so any
    diagnostic code that converts D-grid winds to geographic on the
    cubed sphere can use the same higher-fidelity rotation without
    duplicating the code.

    Parameters
    ----------
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    cos_alpha : jax.Array, shape (6, n, n)
    sin_alpha : jax.Array, shape (6, n, n)
    """
    cax = cdgrid.cos_angle_edge_x  # (6, n, n+1)
    sax = cdgrid.sin_angle_edge_x  # (6, n, n+1)
    cay = cdgrid.cos_angle_edge_y  # (6, n+1, n)
    say = cdgrid.sin_angle_edge_y  # (6, n+1, n)
    ca = 0.25 * (cax[:, :, :-1] + cax[:, :, 1:]
                 + cay[:, :-1, :] + cay[:, 1:, :])
    sa = 0.25 * (sax[:, :, :-1] + sax[:, :, 1:]
                 + say[:, :-1, :] + say[:, 1:, :])
    # Renormalize to unit magnitude (4-point average on a unit
    # circle does not preserve length exactly; the matrix does this
    # at run_atmosphere_test_matrix.py:1230-1231).
    norm = jnp.sqrt(ca ** 2 + sa ** 2)
    return ca / norm, sa / norm
