"""GM/Redi isopycnal mixing on the MPAS Voronoi mesh — API skeleton.

**Status (2026-04-28):** This module defines the public API for the
MPAS port of GM/Redi but does not yet implement it.  Calling any
function below raises ``NotImplementedError`` with a pointer to the
implementation plan at ``docs/ocean_experiments/gm_redi_mpas_plan.md``.

The signatures mirror ``gm_redi_tracer_tendency_latlon`` in
``gm_redi_latlon_cgrid.py`` so the dycore hook in
``ocean_model_mpas.py`` is mechanical once the body is filled in:

    if self.config.gm_redi is not None:
        from legoesm.ocean.physics.lateral_mixing.gm_redi_mpas import (
            gm_redi_tracer_tendency_mpas,
        )
        dT_gm, dS_gm = gm_redi_tracer_tendency_mpas(...)
        T_mid = T_mid + dt * dT_gm * mask
        S_mid = S_mid + dt * dS_gm * mask

The skeleton intentionally raises rather than silently no-op'ing —
this matches the lesson learned with the surface-forcing scheme dropout
(`make_mpas_ocean_physics` used to silently no-op on
``scheme='combined'``, producing a bit-frozen 1-yr global overturning
run before the bug was caught).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from legoesm.core.operators_voronoi import (
    cell_to_edge_avg_3d,
    divergence_cell_3d,
    gradient_edge_3d,
)
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    dm95_taper_scalar,
    vertical_flux_divergence,
)

if TYPE_CHECKING:
    from legoesm.grids.voronoi import VoronoiMesh
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.vertical import OceanZStarCoordinate


_PLAN = "docs/ocean_experiments/gm_redi_mpas_plan.md"

_EPS_DIV = 1e-30


def _not_implemented(name: str) -> None:
    raise NotImplementedError(
        f"GM/Redi on MPAS is not yet implemented (called: {name}). "
        f"See {_PLAN} for the design and phasing."
    )


def _voronoi_neumann_fill(
    f: jnp.ndarray,
    mask: jnp.ndarray,
    mesh: "VoronoiMesh",
    n_passes: int = 3,
) -> jnp.ndarray:
    """Extend a cell-centred field into land via 1-cell-per-pass averaging.

    Land cells with at least one ocean neighbour take the
    ocean-neighbour mean; the (now-filled) cells become "ocean" for
    subsequent passes.  Three passes mirror the lat-lon
    ``_neumann_fill_cgrid`` reach so multi-cell-deep land columns are
    populated to the same depth before any gradient is taken.

    The fill is essential for slope computation: an unfilled coastal
    edge sees ``rho[ocean] - rho[0]`` (sentinel) producing a huge
    spurious horizontal gradient, which on the lat-lon C-grid clipped
    the slope to ``S_max`` and broke the per-triad cancellation of
    the Redi off-diagonal flux (see lat-lon GM/Redi memo, lesson #1).

    Parameters
    ----------
    f : array (nCells,) or (nCells, nlev)
        Field to fill.
    mask : array (nCells,)
        1 = ocean, 0 = land.
    mesh : VoronoiMesh
    n_passes : int

    Returns
    -------
    array same shape as ``f``.
    """
    coc = mesh.cellsOnCell                       # (maxEdges, nCells)
    valid = (coc >= 0).astype(f.dtype)           # (maxEdges, nCells)
    coc_safe = jnp.maximum(coc, 0)

    filled = f
    m = mask.astype(f.dtype)

    for _ in range(n_passes):
        # Gather neighbour values and ocean-mask along (maxEdges, nCells, ...).
        f_nbr = filled[coc_safe]                 # (maxEdges, nCells) or (..., nlev)
        m_nbr = m[coc_safe] * valid              # (maxEdges, nCells)

        if filled.ndim == 1:
            nbr_sum = jnp.sum(f_nbr * m_nbr, axis=0)         # (nCells,)
            nbr_count = jnp.sum(m_nbr, axis=0)               # (nCells,)
        else:
            nbr_sum = jnp.sum(f_nbr * m_nbr[:, :, None], axis=0)  # (nCells, nlev)
            nbr_count = jnp.sum(m_nbr, axis=0)               # (nCells,)
            nbr_count_e = nbr_count[:, None]
        nbr_avg = nbr_sum / jnp.maximum(
            nbr_count if filled.ndim == 1 else nbr_count_e, 1.0
        )

        is_land = m < 0.5                                    # (nCells,)
        has_any_nbr = nbr_count > 0.0                        # (nCells,)
        update_cell = is_land & has_any_nbr                  # (nCells,)
        if filled.ndim == 1:
            filled = jnp.where(update_cell, nbr_avg, filled)
        else:
            filled = jnp.where(update_cell[:, None], nbr_avg, filled)
        # The cells we just filled count as ocean for subsequent passes.
        m = jnp.where(update_cell, 1.0, m)

    return filled


def compute_isopycnal_slopes_mpas(
    rho: jnp.ndarray,
    mask: jnp.ndarray,
    z_coord: "OceanZStarCoordinate",
    jacobian: jnp.ndarray,
    mesh: "VoronoiMesh",
    cfg: "GMRediConfig",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Edge-normal isopycnal slope at interior interfaces (Phase 1).

    Implements Option (A) from ``docs/ocean_experiments/gm_redi_mpas_plan.md``:
    a single scalar slope along each edge's own normal, no cell-centred
    reconstruction.

    Algorithm
    ---------
    1. Neumann-fill ``rho`` from ocean cells into land neighbours so
       coastal edges see a real horizontal gradient rather than a
       sentinel value.
    2. Compute the edge-normal density gradient at full levels via
       ``gradient_edge_3d`` (TRiSK primitive).
    3. Compute the vertical density gradient at cell-centred interfaces
       and clamp to negative values (stable stratification) using the
       same convention as the lat-lon code.
    4. Average the vertical gradient onto edges (``cell_to_edge_avg_3d``).
    5. Average the horizontal edge-normal gradient onto interfaces
       (simple half-sum across the two adjacent full levels).
    6. Form ``S_n = -drho_dn / drho_dz``, clip to ``[-S_max, S_max]``,
       then apply the DM95 scalar taper.

    Parameters
    ----------
    rho : (nCells, nlev)
        In-situ density at cell centres.
    mask : (nCells,)
        Ocean mask (1 = ocean, 0 = land).
    z_coord : OceanZStarCoordinate
    jacobian : (nCells,)
        z-star Jacobian (eta + H) / H.
    mesh : VoronoiMesh
    cfg : GMRediConfig

    Returns
    -------
    S_n : (nEdges, nlev-1)
        Tapered edge-normal isopycnal slope at interior interfaces.
    taper : (nEdges, nlev-1)
        Danabasoglu-McWilliams 1995 taper factor in [0, 1].
    """
    rho_filled = _voronoi_neumann_fill(rho, mask, mesh)

    # --- Horizontal edge-normal density gradient at full levels ---
    drho_dn = gradient_edge_3d(rho_filled, mesh)             # (nEdges, nlev)

    # --- Vertical density gradient at cell-centred interfaces ---
    # dz_half_ref shape (nlev-1,); jacobian shape (nCells,).
    dz_half = z_coord.dz_half_ref * jacobian[:, jnp.newaxis]  # (nCells, nlev-1)
    drho_dz = (rho_filled[:, :-1] - rho_filled[:, 1:]) / jnp.maximum(
        dz_half, _EPS_DIV
    )                                                          # (nCells, nlev-1)
    # Stable stratification: density increases with depth ⇒ drho/dz < 0
    # because z increases upward.  Floor to a small negative value to
    # avoid sign flips in unstable columns (mirrors lat-lon convention).
    drho_dz_safe = jnp.minimum(drho_dz, -_EPS_DIV)             # (nCells, nlev-1)

    # --- Move vertical gradient onto edges ---
    drho_dz_edge = cell_to_edge_avg_3d(drho_dz_safe, mesh)     # (nEdges, nlev-1)

    # --- Average horizontal gradient from full levels to interfaces ---
    drho_dn_half = 0.5 * (drho_dn[:, :-1] + drho_dn[:, 1:])   # (nEdges, nlev-1)

    # --- Slope along the edge normal ---
    S_n_raw = -drho_dn_half / drho_dz_edge                    # (nEdges, nlev-1)
    S_n_clipped = jnp.clip(S_n_raw, -cfg.S_max, cfg.S_max)

    # DM95 scalar taper (single-component variant — see _gm_redi_common).
    return dm95_taper_scalar(S_n_clipped, cfg.S_max)


def _perot_inner_product_cell(
    a_edge: jnp.ndarray,
    b_edge: jnp.ndarray,
    mesh: "VoronoiMesh",
) -> jnp.ndarray:
    """Cell-centre Perot reconstruction of an inner product of two
    edge-normal vector fields.

    For two vector fields whose normal components live at edges, the
    Perot reconstruction of their cell-centred inner product is::

        (a · b)_cell = (1 / A_c) · Σ_e (dc_e · dv_e / 2) · a_e · b_e

    summed over the edges of cell ``c`` (with implicit `sign² = 1`).
    The same dual-area weight ``dc·dv/2`` underlies
    ``kinetic_energy_cell`` (Ringler 2010, Eq. 63).  Used here for the
    cell-averaged ``S · ∇q`` and ``|S|²`` that enter the vertical Redi
    flux on Voronoi.

    Parameters
    ----------
    a_edge, b_edge : array (nEdges, ...) or (nEdges,)
        Edge-normal components.  Trailing axes are broadcast.
    mesh : VoronoiMesh

    Returns
    -------
    array (nCells, ...) — leading axis replaces nEdges.
    """
    eoc = mesh.edgesOnCell                        # (maxEdges, nCells)
    valid = (eoc >= 0).astype(a_edge.dtype)       # (maxEdges, nCells)
    eoc_safe = jnp.maximum(eoc, 0)

    # Gather (a, b) and edge dual area onto (maxEdges, nCells, ...).
    a_g = a_edge[eoc_safe]                        # (maxEdges, nCells, ...)
    b_g = b_edge[eoc_safe]
    dc = mesh.dcEdge[eoc_safe]                    # (maxEdges, nCells)
    dv = mesh.dvEdge[eoc_safe]
    area_e = 0.5 * dc * dv                        # (maxEdges, nCells)

    if a_edge.ndim == 1:
        contrib = a_g * b_g * area_e * valid
        return jnp.sum(contrib, axis=0) / mesh.areaCell
    # Trailing-axis broadcast for (nEdges, nlev[-1]).
    contrib = a_g * b_g * (area_e * valid)[:, :, None]
    return jnp.sum(contrib, axis=0) / mesh.areaCell[:, None]


def gm_redi_tracer_tendency_centered_mpas(
    q: jnp.ndarray,
    S_n: jnp.ndarray,
    mask: jnp.ndarray,
    edge_mask: jnp.ndarray,
    z_coord: "OceanZStarCoordinate",
    jacobian: jnp.ndarray,
    mesh: "VoronoiMesh",
    kappa_GM,
    kappa_Redi: float,
) -> jnp.ndarray:
    """Centred GM+Redi tracer tendency on Voronoi (Phase 2).

    Mirrors ``gm_redi_tracer_tendency_latlon_cgrid`` with edge-normal
    primitives.  All slope×gradient combinations are evaluated at
    INTERFACE level so that the algebraic identity
    ``∂_n q + S_n · ∂_z q = 0`` (which holds when ``q = f(ρ)``) holds
    at every edge × interface, giving exact cancellation of the
    off-diagonal Redi flux to within centred-averaging error.

    Algorithm
    ---------
    Horizontal:
        F_n[edge, k_int] = κ_R · ∂_n q + (κ_R − κ_GM) · S_n · ∂_z q
        F_n[edge, k_full] = ½ · (F_n[..., k_int-1] + F_n[..., k_int])
                            (zero pad at surface/bottom)
        F_n[edge, *]      *= edge_mask
        dq_h               = divergence_cell_3d(F_n, mesh)

    Vertical (cell-centred via Perot reconstruction):
        (S · ∇q)_c[k_int]  = Perot Σ over cell edges of (S_n · ∂_n q)
        (|S|²)_c[k_int]    = Perot Σ over cell edges of (S_n²)
        F_z[c, k_int]      = (κ_R + κ_GM) · (S · ∇q)_c
                             + κ_R · |S|²_c · ∂_z q[c]
        dq_vert            = vertical_flux_divergence(F_z, dz_actual)

    Parameters
    ----------
    q : (nCells, nlev)
        Tracer field at cell centres.
    S_n : (nEdges, nlev-1)
        Tapered edge-normal isopycnal slope at interior interfaces.
    mask : (nCells,)
    edge_mask : (nEdges,)
        ``mask[c1] · mask[c2]`` — zero on coastline edges so no flux
        is fed through land boundaries.
    z_coord, jacobian, mesh
    kappa_GM : float or (nCells,)
        GM transport coefficient.  Per-cell values are broadcast to
        edges for the horizontal flux and used directly at cell centres
        for the vertical flux.
    kappa_Redi : float

    Returns
    -------
    tendency : (nCells, nlev)
    """
    dz_actual = z_coord.dz_ref * jacobian[:, jnp.newaxis]    # (nCells, nlev)
    dz_half = z_coord.dz_half_ref * jacobian[:, jnp.newaxis]  # (nCells, nlev-1)

    # --- Per-cell vs scalar kappa_GM ---
    if isinstance(kappa_GM, jnp.ndarray) and kappa_GM.ndim == 1:
        kappa_GM_cell = kappa_GM                              # (nCells,)
        kappa_GM_edge = cell_to_edge_avg_3d(
            kappa_GM[:, None], mesh
        )[:, 0]                                                # (nEdges,)
    else:
        kappa_GM_cell = kappa_GM
        kappa_GM_edge = kappa_GM

    # --- Neumann-fill q before differencing across coastlines ---
    q_filled = _voronoi_neumann_fill(q, mask, mesh)

    # --- Edge-normal tracer gradient at full levels ---
    dq_dn = gradient_edge_3d(q_filled, mesh)                  # (nEdges, nlev)

    # --- Vertical tracer gradient at cell-centred interfaces ---
    dq_dz_cell = (q_filled[:, :-1] - q_filled[:, 1:]) / jnp.maximum(
        dz_half, _EPS_DIV
    )                                                          # (nCells, nlev-1)

    # Move dq/dz onto edges for the horizontal flux.
    dq_dz_edge = cell_to_edge_avg_3d(dq_dz_cell, mesh)         # (nEdges, nlev-1)

    # Average dq/dn from full levels to interfaces.
    dq_dn_half = 0.5 * (dq_dn[:, :-1] + dq_dn[:, 1:])         # (nEdges, nlev-1)

    # ------------------------------------------------------------------
    # Horizontal flux at edge-interface, then averaged to edge-full
    # ------------------------------------------------------------------
    # κ_GM may be (nEdges,) or scalar; promote shape for broadcast.
    if isinstance(kappa_GM_edge, jnp.ndarray) and kappa_GM_edge.ndim == 1:
        kappa_GM_edge_b = kappa_GM_edge[:, None]              # (nEdges, 1)
    else:
        kappa_GM_edge_b = kappa_GM_edge

    F_n_half = (
        kappa_Redi * dq_dn_half
        + (kappa_Redi - kappa_GM_edge_b) * S_n * dq_dz_edge
    )                                                          # (nEdges, nlev-1)

    # Average interface fluxes to full levels (zero at surface/bottom).
    z_pad = jnp.zeros((mesh.nEdges, 1), dtype=F_n_half.dtype)
    F_n_full = 0.5 * (
        jnp.concatenate([z_pad, F_n_half], axis=-1)
        + jnp.concatenate([F_n_half, z_pad], axis=-1)
    )                                                          # (nEdges, nlev)

    # Land-boundary flux is zero.
    F_n_full = F_n_full * edge_mask[:, None]

    # Horizontal flux divergence: standard TRiSK divergence_cell.
    dq_h = divergence_cell_3d(F_n_full, mesh)                  # (nCells, nlev)

    # ------------------------------------------------------------------
    # Vertical flux at cell-centred interfaces (Perot reconstruction)
    # ------------------------------------------------------------------
    # (S · ∇q)_cell at interfaces.
    # NOTE: zero out S_n on land-adjacent edges so Perot doesn't carry
    # spurious slope·gradient products into the cell-mean from edges
    # where the flux must be zero.
    S_n_oc = S_n * edge_mask[:, None]                          # (nEdges, nlev-1)
    dq_dn_half_oc = dq_dn_half * edge_mask[:, None]
    S_dot_grad_cell = _perot_inner_product_cell(
        S_n_oc, dq_dn_half_oc, mesh,
    )                                                          # (nCells, nlev-1)
    S_sq_cell = _perot_inner_product_cell(
        S_n_oc, S_n_oc, mesh,
    )                                                          # (nCells, nlev-1)

    # κ_GM may be (nCells,) — broadcast to interface axis.
    if isinstance(kappa_GM_cell, jnp.ndarray) and kappa_GM_cell.ndim == 1:
        kappa_GM_cell_b = kappa_GM_cell[:, None]               # (nCells, 1)
    else:
        kappa_GM_cell_b = kappa_GM_cell

    F_z = (
        (kappa_Redi + kappa_GM_cell_b) * S_dot_grad_cell
        + kappa_Redi * S_sq_cell * dq_dz_cell
    )                                                          # (nCells, nlev-1)

    dq_vert = vertical_flux_divergence(F_z, dz_actual)         # (nCells, nlev)

    # ------------------------------------------------------------------
    # Total, masked
    # ------------------------------------------------------------------
    return (dq_h + dq_vert) * mask[:, None]


def gm_redi_tracer_tendency_triads_mpas(
    q: jnp.ndarray,
    rho: jnp.ndarray,
    mask: jnp.ndarray,
    edge_mask: jnp.ndarray,
    z_coord: "OceanZStarCoordinate",
    jacobian: jnp.ndarray,
    mesh: "VoronoiMesh",
    kappa_GM,
    kappa_Redi: float,
    S_max: float,
) -> jnp.ndarray:
    """Triad-discretised GM+Redi tracer tendency (Phase 5).

    The Griffies 1998 triad scheme is the open scientific subproblem
    on Voronoi geometry — see the dedicated phase in the plan doc.
    """
    _not_implemented("gm_redi_tracer_tendency_triads_mpas")


def gm_redi_tracer_tendency_mpas(
    T: jnp.ndarray,
    S: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    mesh: "VoronoiMesh",
    z_coord: "OceanZStarCoordinate",
    cfg: "GMRediConfig",
    *,
    eos: str = "wright",
    eos_linear=None,
    mask: jnp.ndarray | None = None,
    edge_mask: jnp.ndarray | None = None,
    f_coriolis: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Top-level MPAS GM/Redi entry point.

    Composes density → slopes → optional Visbeck → tracer tendency.
    Signature mirrors ``gm_redi_tracer_tendency_latlon`` so the
    dycore hook in ``ocean_model_mpas.py`` is mechanical once the
    body is filled in.

    Returns
    -------
    dT_dt, dS_dt : (nCells, nlev)
    """
    _not_implemented("gm_redi_tracer_tendency_mpas")
