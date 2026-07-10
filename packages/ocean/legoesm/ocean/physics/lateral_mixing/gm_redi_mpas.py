"""GM/Redi isopycnal mixing on the MPAS Voronoi mesh.

**Status:** The centered GM/Redi scheme is implemented —
``compute_isopycnal_slopes_mpas``, ``gm_redi_tracer_tendency_centered_mpas``
and ``gm_redi_tracer_tendency_mpas`` are live.  Only the TRIAD slope-limited
path is still pending and raises ``NotImplementedError`` with a pointer to
the implementation plan at ``docs/ocean/experiments/gm_redi_mpas_plan.md``.

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

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.operators_voronoi import (
    cell_to_edge_avg_3d,
    divergence_cell_3d,
    gradient_edge_3d,
)
from legoesm.ocean.dynamics.ocean_tendency_common import (
    iterate_eos_and_pressure_anomaly,
)
from legoesm.ocean.eos import make_eos_fn, rho_0 as _RHO_0
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    EPS_DIV as _EPS_DIV,
    compute_visbeck_kappa_gm,
    dm95_taper_scalar,
    gm_resolution_scaled_kappa,
    validate_adjoint_stabilization,
    vertical_flux_divergence,
)
from legoesm.ocean.vertical import compute_ocean_jacobian

if TYPE_CHECKING:
    from legoesm.grids.voronoi import VoronoiMesh
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.vertical import OceanZStarCoordinate


__physics_contract__ = {
    "summary": (
        "GM/Redi isopycnal mixing on the MPAS Voronoi mesh (centred small-slope "
        "scheme): density -> isoneutral slopes -> optional Visbeck kappa -> "
        "adiabatic eddy advection + isoneutral diffusion tendencies for T and S "
        "via edge fluxes and a cell divergence."
    ),
    "inputs": {
        "T": "degC", "S": "psu", "eta": "m", "H_bathy": "m",
        "cfg.kappa_GM": "m^2/s", "cfg.kappa_Redi": "m^2/s",
    },
    "outputs": {"dT_dt": "degC/s", "dS_dt": "psu/s"},
    "sign_convention": (
        "kappa_GM, kappa_Redi >= 0; slopes DM95-tapered; GM skew flux adiabatic "
        "+ Redi along-isopycnal down-gradient; edge fluxes summed into a cell "
        "divergence so the cell-area/volume-integrated tracer is conserved; "
        "edge/land masks give no-flux boundaries; z positive up. (The triad "
        "slope-limited path raises NotImplementedError.)"
    ),
    # Adiabatic tracer redistribution: conserves cell-volume-integrated tracer.
    "conserves": ["tracer"],
    "differentiable": True,
    "reference": (
        "Griffies (1998) JPO 28, 831-841; Gent & McWilliams (1990); Redi "
        "(1982); MPAS-Ocean (Ringler et al. 2013, Ocean Modelling 69)"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_gm_redi_mpas.py — the centred MPAS tendency "
        "matches the lat-lon C-grid path on an equivalent slope and conserves "
        "cell-area-integrated T, S; the triads path raises NotImplementedError."
    ),
}


_PLAN = "docs/ocean/experiments/gm_redi_mpas_plan.md"


def _not_implemented(name: str) -> None:
    raise NotImplementedError(
        f"GM/Redi on MPAS is not yet implemented (called: {name}). "
        f"See {_PLAN} for the design and phasing."
    )


def _validate_slope_density(cfg: "GMRediConfig") -> None:
    """Fail fast if a caller requests an unsupported ``slope_density``.

    ``GMRediConfig.slope_density`` offers ``"neutral"`` (Veros-faithful
    locally-referenced neutral-density slope gradients) on the lat-lon
    C-grid path only.  The MPAS/Voronoi path builds the isoneutral slopes
    from the IN-SITU density ``rho`` exclusively (see
    ``compute_isopycnal_slopes_mpas``).  A ``"neutral"`` request here would
    otherwise be SILENTLY ignored and run in-situ physics — dispatch
    hardening: an unsupported option must raise, never no-op.  Validated on
    the static Python config value at function entry (not in a traced
    branch).
    """
    slope_density = getattr(cfg, "slope_density", "in_situ")
    if slope_density != "in_situ":
        raise NotImplementedError(
            "gm_redi MPAS path only supports slope_density='in_situ'; got "
            f"{slope_density!r} (neutral-density slopes are not implemented "
            "on the MPAS/Voronoi grid)."
        )


def voronoi_neumann_fill(
    f: jnp.ndarray,
    mask: jnp.ndarray,
    mesh: "VoronoiMesh",
    n_passes: int = 3,
) -> jnp.ndarray:
    """Extend a cell-centred field into land via 1-cell-per-pass averaging.

    Land cells with at least one ocean neighbour take the
    ocean-neighbour mean; the (now-filled) cells become "ocean" for
    subsequent passes.  Three passes mirror the lat-lon
    ``neumann_fill_cgrid`` reach so multi-cell-deep land columns are
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
            # Both reductions share ``m_nbr`` weight on axis 0 — fuse.
            _pair = jnp.sum(jnp.stack([f_nbr * m_nbr, m_nbr], axis=-1), axis=0)
            nbr_sum = _pair[..., 0]                         # (nCells,)
            nbr_count = _pair[..., 1]                       # (nCells,)
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

    Implements Option (A) from ``docs/ocean/experiments/gm_redi_mpas_plan.md``:
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
    # Dispatch hardening: the MPAS slope build only supports in-situ-density
    # slopes; a 'neutral' request must raise rather than silently run in-situ.
    _validate_slope_density(cfg)

    rho_filled = voronoi_neumann_fill(rho, mask, mesh)

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

    # Adjoint stabilization (primal-invisible; see _gm_redi_common note and
    # the GMRediConfig field doc). Static Python gating on the config literal.
    adj_stab = getattr(cfg, "adjoint_stabilization", "none")
    validate_adjoint_stabilization(adj_stab)
    if adj_stab == "stop_gradient_slopes":
        S_n_clipped = jax.lax.stop_gradient(S_n_clipped)

    # DM95 scalar taper (single-component variant — see _gm_redi_common).
    # ``taper_width_frac`` maps to Veros's ``iso_dslope / iso_slopec``.
    return dm95_taper_scalar(
        S_n_clipped, cfg.S_max,
        transition_width_frac=cfg.taper_width_frac,
        stop_gradient_taper=(adj_stab == "stop_gradient_taper"),
    )


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
    q_filled = voronoi_neumann_fill(q, mask, mesh)

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

    # The diagonal term ``kappa_Redi * dq_dn_half`` is UNTAPERED (the DM95 taper
    # enters only through the tapered slope ``S_n`` in the off-diagonal term), so
    # — as on the cubed-sphere path — the mixed-layer horizontal diffusivity is
    # always kappa_Redi: the Ferrari (2008) surface complement is inherent here,
    # not a separate term (see gm_redi.validate_cubed_sphere_gm_redi_config).
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

    # Per-level edge mask on partial-cell coordinates: zero F_n at
    # levels below the shallower neighbor's seafloor.  Without this,
    # the divergence at deep cells' deep levels would include spurious
    # flux from the (sub-seafloor, T_fill) side of the step edge.
    if hasattr(z_coord, 'bottom_level'):
        from legoesm.ocean.dynamics.mpas_partial_cell_helpers import (
            compute_max_level_edge_bot,
        )
        bot_e = compute_max_level_edge_bot(z_coord.bottom_level, mesh)
        nlev_loc = F_n_full.shape[1]
        k_idx = jnp.arange(nlev_loc, dtype=bot_e.dtype)
        edge_mask_3d = (
            (k_idx[None, :] <= bot_e[:, None]).astype(F_n_full.dtype)
            * edge_mask[:, None]
        )
        F_n_full = F_n_full * edge_mask_3d
    else:
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

    # No-flux SEAFLOOR boundary on partial-cell coordinates (z positive up).
    # F_z lives at the nlev-1 interior interfaces; interface j is the
    # interface between cells j and j+1.  Zero F_z on any interface whose
    # LOWER cell is below the seafloor (inactive) BEFORE it enters the
    # vertical divergence: interface j is active iff BOTH adjacent cells are
    # wet (mirrors the horizontal ``edge_mask_3d`` seafloor cut above and the
    # lat-lon centered path's zero-flux seafloor BC).  Without this, the
    # seafloor interface F_z[:, bottom_level] — reconstructed via Perot from
    # the sub-seafloor (filled) side of a step edge and from the S²·∂_z q
    # term — carries a spurious diapycnal flux into the DEEPEST ACTIVE cell
    # (k = bottom_level).  The caller's final-tendency active mask never
    # removes it because that cell IS active, so it must be cut here.
    if hasattr(z_coord, "is_active"):
        _active = z_coord.is_active                            # (nCells, nlev) bool
        interface_active = (
            _active[:, :-1] & _active[:, 1:]
        ).astype(F_z.dtype)                                    # (nCells, nlev-1)
        F_z = F_z * interface_active

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


def _visbeck_kappa_gm_mpas(
    rho: jnp.ndarray,
    S_n: jnp.ndarray,
    mesh: "VoronoiMesh",
    z_coord: "OceanZStarCoordinate",
    jacobian: jnp.ndarray,
    f_coriolis: jnp.ndarray,
    cfg_visbeck,
) -> jnp.ndarray:
    """Visbeck (1997) adaptive GM coefficient on Voronoi.

    Reuses the grid-agnostic ``compute_visbeck_kappa_gm`` from
    ``_gm_redi_common.py``.  That function only consumes the slope
    *magnitude* ``|S| = sqrt(S_x² + S_y²)``; on MPAS we reconstruct
    ``|S|²`` at cell centres via the Perot inner-product helper and
    pass the proxy ``(S_x, S_y) = (|S|, 0)``.

    Parameters
    ----------
    rho : (nCells, nlev) — in-situ density.
    S_n : (nEdges, nlev-1) — tapered edge-normal slope.
    mesh : VoronoiMesh
    z_coord, jacobian
    f_coriolis : (nCells,)
    cfg_visbeck : VisbeckConfig

    Returns
    -------
    kappa : (nCells,) clamped GM coefficient [m²/s].
    """
    S_sq_cell = _perot_inner_product_cell(S_n, S_n, mesh)         # (nCells, nlev-1)
    # Tiny positive floor before sqrt to avoid NaN gradients at S²=0.
    # See _gm_redi_common.py:154 for the analogous Visbeck N² fix.
    S_mag_cell = jnp.sqrt(jnp.maximum(S_sq_cell, 1e-30))           # (nCells, nlev-1)
    S_x_proxy = S_mag_cell
    S_y_proxy = jnp.zeros_like(S_mag_cell)
    return compute_visbeck_kappa_gm(
        rho, S_x_proxy, S_y_proxy,
        z_coord, jacobian, f_coriolis, cfg_visbeck,
    )


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

    Composes density → slopes → optional Visbeck → centred tracer
    tendency for ``T`` and ``S``.  Signature mirrors
    ``gm_redi_tracer_tendency_latlon`` so the dycore hook in
    ``ocean_model_mpas.py`` is mechanical.

    The triad scheme (``cfg.slope_scheme = "triads"``) is not yet
    implemented on MPAS — see Phase 5 in the plan doc.  Passing it
    here raises ``NotImplementedError``; callers must use ``"centered"``
    until the triad port lands.

    Parameters
    ----------
    T, S : (nCells, nlev)
    eta : (nCells,) — sea surface height.
    H_bathy : (nCells,) — bottom depth (positive).
    mesh : VoronoiMesh
    z_coord, cfg
    eos : str — "wright" or "linear".
    eos_linear : LinearEOSConfig or None.
    mask : (nCells,) — ocean mask (default all ocean).
    edge_mask : (nEdges,) — face mask, default = mask[c1] · mask[c2].
    f_coriolis : (nCells,) — default 2·Ω·sin(latCell).

    Returns
    -------
    dT_dt, dS_dt : (nCells, nlev)
    """
    # Dispatch hardening (static config value, at function entry): the MPAS
    # path builds isoneutral slopes from in-situ density only.  A 'neutral'
    # slope_density request would be silently ignored below, so raise.
    _validate_slope_density(cfg)

    if mask is None:
        mask = jnp.ones((mesh.nCells,), dtype=T.dtype)
    if edge_mask is None:
        c1 = mesh.cellsOnEdge[0]
        c2 = mesh.cellsOnEdge[1]
        edge_mask = mask[c1] * mask[c2]

    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)

    # Sub-seafloor fill on partial cells: extend bottom-cell T/S downward
    # into inactive levels.  Without this, EOS at sub-seafloor sees
    # T=0,S=35 (the zero-fill convention from model init) and produces a
    # spurious rho value that creates infinite vertical density gradients
    # at the seafloor interface — same root cause that crashed KPP at
    # day 35 before the 2026-05-10 fixes.  Apply the fill BEFORE EOS so
    # rho at every level represents a sensible water mass.
    T_fill = T
    S_fill = S
    if hasattr(z_coord, 'is_active') and hasattr(z_coord, 'bottom_level'):
        _active = z_coord.is_active  # (nCells, nlev)
        _bot_lev = jnp.clip(z_coord.bottom_level, 0, T.shape[1] - 1)
        _row_idx = jnp.arange(T.shape[0])
        _T_bot = T[_row_idx, _bot_lev]
        _S_bot = S[_row_idx, _bot_lev]
        T_fill = jnp.where(_active, T, _T_bot[:, None])
        S_fill = jnp.where(_active, S, _S_bot[:, None])

    # In-situ density via the grid-agnostic 2-pass EOS iteration.
    eos_fn = make_eos_fn(eos, eos_linear)
    fill_fn = lambda field: voronoi_neumann_fill(field, mask, mesh)
    rho, _rho_prime, _p_prime = iterate_eos_and_pressure_anomaly(
        T_fill, S_fill, mask, fill_fn, eos_fn,
        z_coord.dz_ref, _RHO_0, constants.g,
        n_iter=2,
    )

    # Edge-normal slopes (always — needed for Visbeck and for the
    # centred tracer tendency).
    S_n, _taper = compute_isopycnal_slopes_mpas(
        rho, mask, z_coord, jacobian, mesh, cfg,
    )

    # GM coefficient.
    if getattr(cfg, "treguier", None) is not None and cfg.treguier.enabled:
        raise NotImplementedError(
            "GMRediConfig.treguier (NEMO nn_aei_ijk_t=21 adaptive kappa) is "
            "implemented on the lat-lon C-grid path only; the MPAS GM/Redi "
            "would silently fall back. Use visbeck or constant kappa_GM here.")
    if cfg.visbeck.enabled:
        if f_coriolis is None:
            f_coriolis = 2.0 * constants.Omega * jnp.sin(mesh.latCell)
        kappa_GM = _visbeck_kappa_gm_mpas(
            rho, S_n, mesh, z_coord, jacobian, f_coriolis, cfg.visbeck,
        )
    else:
        kappa_GM = cfg.kappa_GM

    # Hallberg (2013) resolution taper of the GM coefficient (default off =>
    # byte-identical). Static Python gate on the config bool. GM-only: the Redi
    # diffusivity cfg.kappa_Redi passed below is left unscaled. dx = sqrt(cell
    # area) matches the lat-lon / cube local-grid-spacing convention.
    if getattr(cfg, "resolution_function", False):
        if f_coriolis is None:
            f_coriolis = 2.0 * constants.Omega * jnp.sin(mesh.latCell)
        kappa_GM = gm_resolution_scaled_kappa(
            kappa_GM, f_coriolis, jnp.sqrt(mesh.areaCell),
            cfg.resfn_gamma, cfg.resfn_cbcl_ms,
        )

    scheme = getattr(cfg, "slope_scheme", "centered")
    if scheme == "centered":
        # Pass the sub-seafloor-filled T/S to the tendency function so
        # the tracer gradients across step edges use sensible values on
        # the inactive side.  The active_3d mask zeros the tendency at
        # sub-seafloor levels before returning.
        dT_dt = gm_redi_tracer_tendency_centered_mpas(
            T_fill, S_n, mask, edge_mask, z_coord, jacobian, mesh,
            kappa_GM, cfg.kappa_Redi,
        )
        dS_dt = gm_redi_tracer_tendency_centered_mpas(
            S_fill, S_n, mask, edge_mask, z_coord, jacobian, mesh,
            kappa_GM, cfg.kappa_Redi,
        )
        # Per-level active mask: zero the tendency at inactive (sub-
        # seafloor) levels on partial-cell coords.  Without this, GM/
        # Redi imprints O(K_redi * gradient / dz) tendencies on cells
        # below the seafloor where there is no real water — the same
        # bug class that crashed KPP before the 2026-05-10 fixes.
        if hasattr(z_coord, 'is_active'):
            _active = z_coord.is_active.astype(dT_dt.dtype)
            dT_dt = dT_dt * _active
            dS_dt = dS_dt * _active
    elif scheme == "triads":
        raise NotImplementedError(
            "GM/Redi triad scheme is not yet implemented on MPAS. "
            f"See Phase 5 in {_PLAN}.  Use slope_scheme='centered' "
            "for now."
        )
    else:
        raise ValueError(
            f"Unknown GMRediConfig.slope_scheme={scheme!r}; "
            f"expected 'centered' or 'triads'."
        )

    return dT_dt, dS_dt
