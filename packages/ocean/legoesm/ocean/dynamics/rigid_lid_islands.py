"""Static island machinery for the rigid-lid streamfunction solver (lat-lon C-grid).

Builds the time-independent ``RigidLidStaticData`` once at model construction
from the fixed bathymetry + land mask:

  1. Flood-fill the land cells into disconnected land masses (``islands``),
     respecting the east-west periodic wrap of a re-entrant channel.
  2. Per-island vertex masks (the vertices that are corners of each island's
     land) + the interior-solve vertex mask (all 4 surrounding cells ocean).
  3. The per-island streamfunction basis functions ``psin`` — each solves the
     homogeneous elliptic problem ``L(ψ)=0`` with ψ=1 on its island's vertices,
     0 on the others (Veros ``streamfunction_init.solve_streamfunction``).
  4. The island circulation-coupling matrix ``line_psin`` — the line integral
     of each basis function's velocity around each island.

This is the multiply-connected machinery: for a periodic channel (the ACC) the
single free island constant is the net channel transport.  All of this is
STATIC (built from the fixed topology), computed eagerly at construction; it is
never differentiated through and is captured by the model as a compile-time
constant (not carried in the ``lax.scan`` state).

Reference: Veros ``veros/core/external/{island,streamfunction_init,
line_integrals}.py`` — adapted from the Arakawa-B grid to the C-grid, with the
line integral expressed via discrete Stokes (``rigid_lid_latlon_cgrid.
island_line_integrals``) rather than directional edge masks.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    streamfunction_vorticity_operator,
    recover_velocity_from_streamfunction,
    vertex_area_cgrid,
)
from legoesm.ocean.dynamics.rigid_lid_latlon_cgrid import (
    RigidLidStaticData,
    assert_rigid_lid_single_rank,
    barotropic_face_depths,
    solve_streamfunction_interior,
    island_line_integrals,
    compute_operator_inv_diag,
)


def _label_islands(cell_land: np.ndarray, periodic_x: bool) -> tuple[np.ndarray, int]:
    """Label disconnected land masses (4-connectivity, optional periodic x).

    Parameters
    ----------
    cell_land : (n_lat, n_lon) bool — True on land cells.
    periodic_x : bool — merge land masses connected across the lon wrap.

    Returns
    -------
    labels : (n_lat, n_lon) int — 0 on ocean, 1..nisle on land masses (ordered by
        descending cell count, so label 1 is the largest — the reference island).
    nisle : int — number of land masses.
    """
    from scipy.ndimage import label

    # 8-CONNECTIVITY (merge diagonally connected land masses) — Veros/pyOM
    # faithful: ``island._compute_isleperim`` uses ``structure = ones((3,3))``.
    # This is also REQUIRED for well-posedness of the island system here:
    # two land masses touching only at a corner SHARE that vertex, so with
    # 4-connectivity they become two islands whose vertex masks OVERLAP —
    # the basis-function Dirichlet conditions (ψ=1 on one island's vertices,
    # 0 on the others') are then CONTRADICTORY at the shared vertex and the
    # island coupling matrix goes ill-conditioned (global_4deg: 11 islands
    # 4-connected vs Veros's 5, ψ-solve NaN by step 4).  Flat-wall domains
    # without diagonal land contacts (the ACC channel) are unchanged.
    structure = np.ones((3, 3))
    raw, n_raw = label(cell_land, structure=structure)

    if periodic_x and n_raw > 1:
        # Union-find merge across the x-wrap: land in column 0 adjacent
        # (same row, 8-connected ⇒ also row±1) to land in column -1 belongs
        # to the same mass.
        parent = list(range(n_raw + 1))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        col0 = raw[:, 0]
        coln = raw[:, -1]
        n_lat = cell_land.shape[0]
        for r in range(n_lat):
            if col0[r] > 0 and coln[r] > 0:
                union(int(col0[r]), int(coln[r]))
            # Diagonal wrap contacts (8-connectivity across the seam).
            if col0[r] > 0:
                if r > 0 and coln[r - 1] > 0:
                    union(int(col0[r]), int(coln[r - 1]))
                if r + 1 < n_lat and coln[r + 1] > 0:
                    union(int(col0[r]), int(coln[r + 1]))
        # Relabel by representative.
        rep = np.array([find(i) for i in range(n_raw + 1)])
        raw = rep[raw]

    # Compact labels 1..nisle, ordered by descending cell count (label 1 = largest).
    present = [lab for lab in np.unique(raw) if lab != 0]
    counts = {lab: int(np.sum(raw == lab)) for lab in present}
    ordered = sorted(present, key=lambda lab: counts[lab], reverse=True)
    labels = np.zeros_like(raw)
    for new_lab, old_lab in enumerate(ordered, start=1):
        labels[raw == old_lab] = new_lab
    return labels, len(ordered)


def _island_vertex_masks(labels: np.ndarray, nisle: int) -> np.ndarray:
    """Per-island vertex masks: vertex (i,j) belongs to island k if any of the 4
    surrounding cells is island-k land.  Shape (nisle, n_lat+1, n_lon+1)."""
    n_lat, n_lon = labels.shape
    masks = np.zeros((nisle, n_lat + 1, n_lon + 1), dtype=np.float64)
    for k in range(1, nisle + 1):
        isl = (labels == k).astype(np.float64)               # (n_lat, n_lon)
        # A vertex touches a cell if the cell is one of its 4 corners.  Scatter
        # each cell to its 4 surrounding vertices: vertex (i,j) corners are cells
        # (i-1,j-1),(i-1,j),(i,j-1),(i,j).  Periodic wrap in lon (col n_lon==0).
        vtx = np.zeros((n_lat + 1, n_lon + 1), dtype=np.float64)
        # cell (a,b) is a corner of vertices (a,b),(a,b+1),(a+1,b),(a+1,b+1)
        vtx[0:n_lat, 0:n_lon] += isl     # vertex (a,b)
        vtx[0:n_lat, 1:n_lon + 1] += isl  # vertex (a,b+1)
        vtx[1:n_lat + 1, 0:n_lon] += isl  # vertex (a+1,b)
        vtx[1:n_lat + 1, 1:n_lon + 1] += isl  # vertex (a+1,b+1)
        # Periodic wrap: the wrap column n_lon mirrors column 0; fold the
        # contributions that landed on the wrap column back onto column 0 and
        # copy column 0 -> wrap column so both carry the same value.
        vtx[:, 0] += vtx[:, n_lon]
        vtx[:, n_lon] = vtx[:, 0]
        masks[k - 1] = (vtx > 0.5).astype(np.float64)
    return masks


def _solve_vertex_mask(cell_land: np.ndarray) -> np.ndarray:
    """Interior-solve vertex mask: 1 where all 4 surrounding cells are ocean.

    These are the vertices where ψ is solved; coast/land vertices (≥1 land
    neighbour) are pinned (carry the island constant, = 0 for the interior
    particular solve).  Walls/poles (boundary vertex rows) are 0.  Shape
    (n_lat+1, n_lon+1)."""
    n_lat, n_lon = cell_land.shape
    ocean = (~cell_land).astype(np.float64)
    # Count ocean cells around each vertex; require all 4 (== 4.0).
    cnt = np.zeros((n_lat + 1, n_lon + 1), dtype=np.float64)
    cnt[0:n_lat, 0:n_lon] += ocean
    cnt[0:n_lat, 1:n_lon + 1] += ocean
    cnt[1:n_lat + 1, 0:n_lon] += ocean
    cnt[1:n_lat + 1, 1:n_lon + 1] += ocean
    # Periodic wrap in lon: vertex col 0 and col n_lon share the same 4 cells
    # (cols n_lon-1 and 0).  Re-evaluate the wrap columns with the wrapped count.
    # Column 0 corners include cells at lon -1 (== n_lon-1) and 0:
    cnt0 = np.zeros((n_lat + 1,), dtype=np.float64)
    cnt0 += np.concatenate([[0.0], ocean[:, n_lon - 1]]) + np.concatenate([ocean[:, n_lon - 1], [0.0]])
    cnt0 += np.concatenate([[0.0], ocean[:, 0]]) + np.concatenate([ocean[:, 0], [0.0]])
    cnt[:, 0] = cnt0
    cnt[:, n_lon] = cnt0
    solve = (cnt > 3.5).astype(np.float64)
    # Pole/wall vertex rows are walls (ψ pinned).
    solve[0, :] = 0.0
    solve[-1, :] = 0.0
    return solve


def build_psin_basis(rl_partial: RigidLidStaticData, grid: LatLonGrid,
                     *, tol: float, maxiter: int) -> jnp.ndarray:
    """Per-island streamfunction basis ``psin`` (n_lat+1, n_lon+1, nisle).

    ``psin[...,k]`` solves ``L(ψ)=0`` on the wet interior with ψ=1 on island k's
    vertices and ψ=0 on all other islands (homogeneous Dirichlet decomposition:
    ψ = ψ_BC + ψ_int with ψ_int=0 on land and ``L(ψ_int) = -L(ψ_BC)``).
    """
    nisle = rl_partial.nisle
    cols = []
    for k in range(nisle):
        psi_bc = rl_partial.island_vertex_masks[k]                 # 1 on island k
        L_bc = streamfunction_vorticity_operator(
            psi_bc, rl_partial.inv_H_u, rl_partial.inv_H_v, grid,
            u_mask=rl_partial.u_mask, v_mask=rl_partial.v_mask,
        )
        rhs = -L_bc
        x0 = jnp.zeros_like(psi_bc)
        psi_int = solve_streamfunction_interior(
            rhs, rl_partial, grid, x0, tol=tol, maxiter=maxiter)
        cols.append(psi_bc + psi_int)
    return jnp.stack(cols, axis=-1)                                # (..., nisle)


def build_line_psin(psin: jnp.ndarray, rl_partial: RigidLidStaticData,
                    grid: LatLonGrid) -> jnp.ndarray:
    """Island circulation-coupling matrix ``line_psin`` (nisle, nisle).

    ``line_psin[i,k]`` = circulation of basis function k's velocity around
    island i (discrete-Stokes line integral)."""
    nisle = rl_partial.nisle
    cols = []
    for k in range(nisle):
        u_k, v_k = recover_velocity_from_streamfunction(
            psin[..., k], rl_partial.inv_H_u, rl_partial.inv_H_v, grid,
            u_mask=rl_partial.u_mask, v_mask=rl_partial.v_mask,
        )
        cols.append(island_line_integrals(u_k, v_k, rl_partial, grid))  # (nisle,)
    # cols[k] is column k (circulation around each island i of basis k):
    return jnp.stack(cols, axis=1)                                  # line_psin[i,k]


def build_rigid_lid_data(H_bathy, land_mask, u_mask, v_mask, config, grid,
                         *, periodic_x: bool = True) -> RigidLidStaticData:
    """Assemble the static rigid-lid data from the fixed bathymetry + masks.

    Parameters
    ----------
    H_bathy : (n_lat, n_lon) bathymetry depth [m] (positive down).
    land_mask : (n_lat, n_lon) cell-center ocean mask (1=ocean, 0=land).
    u_mask, v_mask : C-grid face masks.
    config : LatLonCGridOceanConfig (rigid_lid_cg_tol/maxiter).
    grid : LatLonGrid.
    periodic_x : whether the domain wraps in longitude (re-entrant channel).

    Returns
    -------
    RigidLidStaticData with islands, basis functions and the coupling matrix.
    """
    # Fail fast BEFORE any island labelling / basis solve: the whole build is
    # single-rank only (the psin basis calls the global streamfunction CG, and
    # the line-integral coupling matrix sums over the whole domain). Guards
    # direct callers; the model's _ensure_rigid_lid_data also guards upstream.
    assert_rigid_lid_single_rank()

    np.asarray(H_bathy)
    land_np = np.asarray(land_mask)
    cell_land = land_np < 0.5

    labels, nisle = _label_islands(cell_land, periodic_x)
    if nisle == 0:
        # A rigid lid needs at least one land boundary (coast/wall) to pin the
        # streamfunction; a fully-open domain leaves ψ undetermined (pure-
        # constant null space, no transport constraint) — ill-posed.
        raise ValueError(
            "rigid-lid barotropic solver requires at least one land mass "
            "(coast/wall) to pin the streamfunction; the domain has no land. "
            "Use a free-surface solver for a land-free domain.")
    island_vertex_masks = _island_vertex_masks(labels, nisle)      # (nisle, V, V)
    solve_mask = _solve_vertex_mask(cell_land)                      # (V, V)

    _, _, inv_H_u, inv_H_v = barotropic_face_depths(
        jnp.asarray(H_bathy), jnp.asarray(land_mask), grid)
    A_vertex = jnp.asarray(vertex_area_cgrid(grid))
    inv_diag = compute_operator_inv_diag(
        inv_H_u, inv_H_v, jnp.asarray(u_mask), jnp.asarray(v_mask),
        jnp.asarray(solve_mask), A_vertex, grid)

    # Partial data sufficient for the interior solve + line integrals; psin /
    # line_psin filled in below.
    n_lat, n_lon = land_np.shape
    placeholder_psin = jnp.zeros((n_lat + 1, n_lon + 1, nisle))
    placeholder_line = jnp.zeros((nisle, nisle))
    rl_partial = RigidLidStaticData(
        inv_H_u=inv_H_u, inv_H_v=inv_H_v,
        u_mask=jnp.asarray(u_mask), v_mask=jnp.asarray(v_mask),
        solve_mask=jnp.asarray(solve_mask), A_vertex=A_vertex, inv_diag=inv_diag,
        psin=placeholder_psin, line_psin=placeholder_line,
        island_vertex_masks=jnp.asarray(island_vertex_masks), nisle=nisle,
    )

    psin = build_psin_basis(
        rl_partial, grid,
        tol=config.barotropic.rigid_lid_cg_tol, maxiter=config.barotropic.rigid_lid_cg_maxiter)
    line_psin = build_line_psin(psin, rl_partial, grid)
    return rl_partial._replace(psin=psin, line_psin=line_psin)
