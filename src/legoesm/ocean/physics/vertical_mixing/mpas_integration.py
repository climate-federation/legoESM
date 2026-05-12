"""MPAS Voronoi-mesh adapters for vertical mixing physics.

Mirrors the lat-lon ``integration.py`` factories but adapts to the MPAS
C-grid where:
- T, S, rho are at cells (nCells, nlev)
- u is at edges (nEdges, nlev) — edge-normal scalar

For tracers (T, S), KPP applies directly cell-by-cell.

For momentum (u), KPP needs (u_east, v_north) at cells to compute the
shear-driven Richardson number.  We reconstruct cell-centered velocity
via TRiSK (Perot 2000 reconstruction), pass it to KPP, take the returned
viscosity field A_v(nCells, nlev), interpolate to edges, and apply
``vertical_diffusion_variable_K`` to the edge-normal u directly.

This avoids reconstructing back from (du/dt_east, dv/dt_north) to the
edge-normal tendency, which would be more expensive and introduce
reconstruction errors.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.state import MPASOceanState, MPASOceanTendencies
from legoesm.ocean.eos import (
    compute_ocean_rho,
    rho_0 as _RHO_0,
    c_sw as _C_SW,
    thermal_expansion_coeff,
    haline_contraction_coeff,
)
from legoesm.ocean.init_mpas import reconstruct_cell_velocity
from legoesm.ocean.physics.mixing import vertical_diffusion_variable_K
from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
from legoesm.ocean.physics.vertical_mixing.kpp import kpp_vertical_mixing
from legoesm.ocean.vertical import (
    OceanPartialCellCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
)


def _vertical_diffusion_edge_partial(
    field: jnp.ndarray,
    h_e: jnp.ndarray,
    K_half: jnp.ndarray,
) -> jnp.ndarray:
    """Vertical diffusion with actual per-edge layer thicknesses.

    Same algorithm as ``vertical_diffusion_variable_K`` but uses
    explicit per-level layer thicknesses ``h_e`` (from min-rule on
    partial cells) instead of ``dz_ref * J``.  This correctly handles:

    - Thin partial bottom cells: tendency uses actual 5m thickness,
      not the 128m reference thickness.
    - Sub-seafloor levels: h_e = 0 there, so those levels get zero
      tendency (division by ``max(h_e, eps)`` produces a finite but
      irrelevant value that is zeroed by ``edge_mask_3d`` in the PE
      module).
    - Full cells: h_e = dz_ref * J, so this is bit-exact with the
      original operator on z-star (non-partial) coordinates.

    Parameters
    ----------
    field : array, shape (nEdges, nlev)
        Edge-normal velocity.
    h_e : array, shape (nEdges, nlev)
        Actual layer thickness at edges [m] (min-rule, from
        ``min_cell_to_edge(compute_layer_thickness(...))``).
        Zero at sub-seafloor levels.
    K_half : array, shape (nEdges, nlev-1)
        Diffusivity at interior interfaces [m^2/s].
        Pre-masked: zero at sub-seafloor interfaces.

    Returns
    -------
    array, shape (nEdges, nlev) : Vertical diffusion tendency [m/s^2].
    """
    nlev = field.shape[-1]
    if nlev < 2:
        return jnp.zeros_like(field)

    dtype = field.dtype
    h_e = h_e.astype(dtype)
    K_half = K_half.astype(dtype)

    # Guard against zero thickness (sub-seafloor levels).  The
    # resulting tendency is large but irrelevant — it gets multiplied
    # by edge_mask_3d=0 in the PE module.  Using a LARGE floor (1m
    # instead of 1e-10) prevents extreme values from affecting
    # numerical precision elsewhere.
    h_safe = jnp.maximum(h_e, 1.0)

    # Distance between cell centers: 0.5 * (h[k] + h[k+1]).
    # At the seafloor boundary (h[bot]=5m, h[bot+1]=0): dz_half =
    # 0.5*(5+1) = 3m (with the 1m floor).  But K_half is zero there
    # (masked), so the flux is zero regardless.  The only interface
    # where dz_half matters is between two active cells.
    dz_half = 0.5 * (h_safe[..., :-1] + h_safe[..., 1:])  # (nEdges, nlev-1)

    # Diffusive flux at interfaces: K * d(field)/dz.
    df_dz = (field[..., :-1] - field[..., 1:]) / dz_half
    flux = K_half * df_dz  # (nEdges, nlev-1)

    # Tendency at full levels: d(flux)/dz with zero-flux BCs.
    top = -flux[..., :1] / h_safe[..., :1]
    interior = (flux[..., :-1] - flux[..., 1:]) / h_safe[..., 1:-1]
    bottom = flux[..., -1:] / h_safe[..., -1:]
    return jnp.concatenate([top, interior, bottom], axis=-1)


def make_kpp_physics_mpas(config: VerticalMixingConfig) -> Callable:
    """Build KPP physics_fn for MPAS Voronoi mesh.

    Parameters
    ----------
    config : VerticalMixingConfig
        Must have ``scheme="kpp"`` and ``kpp`` sub-config populated.

    Returns
    -------
    Callable
        ``physics_fn(state, mesh, z_coord, surface_forcing=None)``
        returning (du_dt_edge, dT_dt_cell, dS_dt_cell, A_v_cell, K_v_cell).
        The model assembles these into MPASOceanTendencies.
    """
    cfg = config.kpp

    def physics_fn(
        state: MPASOceanState,
        mesh,
        z_coord,
        surface_forcing=None,
    ):
        T_3d = state.T.data       # (nCells, nlev)
        S_3d = state.S.data       # (nCells, nlev)
        u_edge = state.u.data     # (nEdges, nlev)
        eta = state.eta.data      # (nCells,)
        H_bathy = state.H_bathy.data
        mask = state.land_mask.data  # (nCells,) — 1=ocean, 0=land

        # Reconstruct cell-centered (u_east, v_north) from edge-normal u
        # via TRiSK/Perot.  KPP needs this to diagnose Richardson-number
        # shear instability and produce the enhanced viscosity that damps
        # the equatorial jet (the whole reason we want KPP).
        #
        # Earlier (2026-05-09), passing real velocities destabilized the
        # equator at day 1 due to sub-seafloor momentum leak: KPP saw
        # spurious velocity below the seafloor (from zero-fill mismatch
        # with the active mask) and produced bogus viscosity profiles.
        # With the 2026-05-10 fixes (sub-seafloor T/S fill, A_v masking
        # at inactive interfaces, edge_mask_3d on physics du_dt in the
        # PE module, per-edge CFL cap with actual h_e), the sub-seafloor
        # leak is closed and real velocities can be passed safely.
        u_east_raw, v_north_raw = reconstruct_cell_velocity(u_edge, mesh)

        # Density at cells.  KPP divides by Jacobian internally; the
        # Jacobian is 0 on land cells (where H_bathy=0), which produces
        # NaN.  Replace land Jacobian with 1.0 (reference) and density
        # with rho_0 — the resulting tendencies are masked out below.
        J_real = compute_ocean_jacobian(eta, H_bathy, z_coord)
        J = jnp.where(mask > 0.5, J_real, 1.0)
        rho_real = compute_ocean_rho(state, z_coord, J_real)
        rho = jnp.where(mask[:, None] > 0.5, rho_real, _RHO_0)

        # Surface forcing (KPP needs friction velocity and buoyancy flux).
        # Lat-lon code derives these from OceanSurfaceForcing — we mirror
        # that pattern.  When surface_forcing is None, KPP uses interior
        # proxies and still runs.
        tau_x = getattr(surface_forcing, "tau_x", None) if surface_forcing else None
        tau_y = getattr(surface_forcing, "tau_y", None) if surface_forcing else None
        q_net = getattr(surface_forcing, "q_net", None) if surface_forcing else None
        fw = getattr(surface_forcing, "freshwater", None) if surface_forcing else None

        Q_sfc_T = None
        B_f = None
        if q_net is not None:
            Q_sfc_T = q_net / (_RHO_0 * _C_SW)
            T_sfc = T_3d[..., 0]
            S_sfc = S_3d[..., 0]
            p_sfc = jnp.zeros_like(T_sfc)
            alpha = thermal_expansion_coeff(T_sfc, S_sfc, p_sfc)
            B_f = -constants.g * alpha * Q_sfc_T

        Q_sfc_S = None
        if fw is not None:
            S_sfc = S_3d[..., 0]
            T_sfc = T_3d[..., 0]
            p_sfc = jnp.zeros_like(T_sfc)
            beta = haline_contraction_coeff(T_sfc, S_sfc, p_sfc)
            Q_sfc_S = -S_sfc * fw / _RHO_0
            B_salt = constants.g * beta * Q_sfc_S
            B_f = B_salt if B_f is None else (B_f + B_salt)

        # Zero-out fields on land cells so KPP doesn't see junk values.
        # On partial-cell coordinates, also fill sub-seafloor levels with
        # the deepest active cell's value.  Without this, KPP sees T=0
        # below the seafloor and interprets it as a massive temperature
        # discontinuity → spurious mixing at the bottom active level.
        m3 = mask[:, None]
        u_east_w = jnp.where(m3 > 0.5, u_east_raw, 0.0)
        v_north_w = jnp.where(m3 > 0.5, v_north_raw, 0.0)
        T_w = jnp.where(m3 > 0.5, T_3d, 0.0)
        S_w = jnp.where(m3 > 0.5, S_3d, 35.0)  # safe S for EOS
        if hasattr(z_coord, 'is_active'):
            # Fill sub-seafloor T/S/u/v by extending the deepest active
            # value downward.  Without this, KPP sees a discontinuity at
            # the seafloor (T=0, u=0 below) → spurious large mixing.
            _active = z_coord.is_active  # (nCells, nlev) bool
            _bot_lev = z_coord.bottom_level  # (nCells,) int
            _bot_lev_safe = jnp.clip(_bot_lev, 0, T_3d.shape[1] - 1)
            _row_idx = jnp.arange(T_w.shape[0])
            _T_bot = T_w[_row_idx, _bot_lev_safe]
            _S_bot = S_w[_row_idx, _bot_lev_safe]
            _u_bot = u_east_w[_row_idx, _bot_lev_safe]
            _v_bot = v_north_w[_row_idx, _bot_lev_safe]
            T_w = jnp.where(_active, T_w, _T_bot[:, None])
            S_w = jnp.where(_active, S_w, _S_bot[:, None])
            u_east_w = jnp.where(_active, u_east_w, _u_bot[:, None])
            v_north_w = jnp.where(_active, v_north_w, _v_bot[:, None])

        # Run KPP on cell-centered fields.  KPP returns (du_dt, dv_dt) at
        # cells AND the viscosity field A_v(nCells, nlev-1) at half levels.
        # We discard the cell-centered velocity tendencies and re-apply
        # diffusion to edge-normal u using the interpolated A_v.
        kpp_out = kpp_vertical_mixing(
            u_east_w, v_north_w, T_w, S_w,
            rho, eta, z_coord, J, cfg,
            tau_x=tau_x, tau_y=tau_y, B_f=B_f,
            Q_sfc_T=Q_sfc_T, Q_sfc_S=Q_sfc_S,
        )

        # Mask tracer tendencies — zero on land AND shallow cells.
        # KPP is designed for open-ocean boundary layers (50-500m deep).
        # In cells with fewer than 5 active levels (shallow shelves,
        # isolated water bodies like the Caspian), KPP's diffusion is
        # physically inappropriate and creates O(1 K/s) tendencies in
        # thin cells that blow up the model (Finding 10: lat=48°N,
        # lon=49°E, H=50m).  Mask these cells entirely.
        _min_kpp_levels = 5
        if hasattr(z_coord, 'bottom_level'):
            _n_active = z_coord.bottom_level + 1  # (nCells,)
            _kpp_mask = (_n_active >= _min_kpp_levels).astype(T_3d.dtype)
        else:
            _kpp_mask = jnp.ones(mask.shape, dtype=T_3d.dtype)
        _kpp_mask_3d = (_kpp_mask * mask)[:, None]  # (nCells, 1)
        # Use per-level active mask on partial-cell coords to zero tracer
        # tendencies at sub-seafloor levels.  Without this, KPP computes
        # diffusion through all nlev levels (using incorrect dz_ref
        # thicknesses below the seafloor) and the resulting tendencies
        # corrupt sub-seafloor T/S values before the model's active_3d
        # guard can restore them.
        if hasattr(z_coord, 'is_active'):
            _active_3d = z_coord.is_active.astype(T_3d.dtype)  # (nCells, nlev)
            _kpp_mask_full = _kpp_mask_3d * _active_3d
        else:
            _kpp_mask_full = _kpp_mask_3d
        dT_dt = jnp.where(_kpp_mask_full > 0.5, kpp_out.dT_dt, 0.0)
        dS_dt = jnp.where(_kpp_mask_full > 0.5, kpp_out.dS_dt, 0.0)

        # A_v is at half-levels (nCells, nlev-1).  Mask land cells.
        # Cap A_v to CFL-safe maximum based on the thinner of the two
        # adjacent layers: A_v_max = 0.25 * min(dz_k, dz_k+1)^2 / dt.
        # Without this cap, KPP can produce O(1-10) m²/s in deep
        # boundary layers, violating explicit-diffusion stability on
        # thin stretched-grid surface cells (dz ~ 20m, dt ~ 300s).
        # Use 0.25 (not 0.5) for safety margin.
        _dt_phys = 300.0  # physics timestep [s] — conservative estimate
        _dz = z_coord.dz_ref  # (nlev,)
        _dz_min_half = jnp.minimum(_dz[:-1], _dz[1:])  # (nlev-1,)
        _Av_max = 0.25 * _dz_min_half**2 / _dt_phys  # (nlev-1,)
        m3_half = mask[:, None]  # broadcast to (nCells, nlev-1)
        A_v_raw = jnp.where(m3_half > 0.5, kpp_out.A_v, 0.0)
        A_v_cells = jnp.minimum(A_v_raw, _Av_max[None, :])

        # Interpolate A_v from cells to edges and apply vertical diffusion
        # to edge-normal u directly.  Also apply the shallow-cell mask so
        # edges adjacent to shallow cells don't get KPP momentum diffusion.
        c1 = mesh.cellsOnEdge[0]
        c2 = mesh.cellsOnEdge[1]
        # Zero A_v at edges where EITHER cell is too shallow for KPP.
        _edge_kpp_mask = (_kpp_mask[c1] * _kpp_mask[c2])[:, None]
        A_v_cells_masked = A_v_cells * _kpp_mask[:, None]
        A_v_edge = 0.5 * (A_v_cells_masked[c1] + A_v_cells_masked[c2])  # (nEdges, nlev-1)
        A_v_edge = A_v_edge * _edge_kpp_mask

        # On partial-cell coordinates, zero A_v at half-level interfaces
        # below the shallower neighbor's seafloor.  Without this, the
        # diffusion operator sees nonzero A_v at inactive interfaces and
        # computes spurious fluxes through the seafloor boundary.
        if hasattr(z_coord, 'bottom_level'):
            from legoesm.ocean.dynamics.mpas_partial_cell_helpers import (
                compute_max_level_edge_bot,
                min_cell_to_edge,
            )
            bot_e = compute_max_level_edge_bot(z_coord.bottom_level, mesh)
            # Half-level k sits between full levels k and k+1.
            # Active interfaces: k = 0, ..., maxLevelEdgeBot - 1.
            # (maxLevelEdgeBot is the deepest FULL level that is active
            # on BOTH cells; the interface below it is the seafloor.)
            nlev_half = A_v_edge.shape[1]
            k_idx = jnp.arange(nlev_half, dtype=bot_e.dtype)
            active_half = (k_idx[None, :] < bot_e[:, None]).astype(A_v_edge.dtype)
            A_v_edge = A_v_edge * active_half

        # --- Edge-side vertical diffusion ---
        # On partial-cell coordinates, ``vertical_diffusion_variable_K``
        # uses ``dz_ref * J`` for layer thicknesses, which is WRONG at
        # the partial bottom cell (can be 25x too thick) and at sub-
        # seafloor levels.  This causes the diffusion tendency at the
        # thin bottom cell to be underestimated by the same factor,
        # creating a depth-integrated momentum imbalance that acts as
        # spurious bottom drag and drives the SSH blow-up observed in
        # ETOPO runs.
        #
        # Fix: compute actual per-edge layer thicknesses using the same
        # min-rule as the PE module (``min_cell_to_edge``).  This gives
        # h_e[k] = min(h_k[c1,k], h_k[c2,k]) — zero below the
        # shallower cell's seafloor, and correct partial thickness at
        # the bottom level.  Use these directly in the second-order
        # diffusion stencil (same algorithm as
        # ``vertical_diffusion_variable_K`` but with per-level dz).
        if isinstance(z_coord, OceanPartialCellCoordinate):
            h_k = compute_layer_thickness(eta, H_bathy, z_coord)
            h_e = min_cell_to_edge(h_k, mesh)  # (nEdges, nlev)
            # Per-edge CFL cap using ACTUAL thicknesses (not dz_ref).
            # A_v_max = 0.25 * min(h_e[k], h_e[k+1])^2 / dt at each
            # interface.  h_e=0 at sub-seafloor gives A_v_max=0 there
            # (redundant with active_half mask but defense-in-depth).
            _h_min_half_e = jnp.minimum(
                jnp.maximum(h_e[:, :-1], 1e-10),
                jnp.maximum(h_e[:, 1:], 1e-10),
            )  # (nEdges, nlev-1)
            _Av_max_edge = 0.25 * _h_min_half_e**2 / _dt_phys
            A_v_edge = jnp.minimum(A_v_edge, _Av_max_edge)
            du_dt_edge = _vertical_diffusion_edge_partial(
                u_edge, h_e, A_v_edge,
            )
        else:
            J_edge = 0.5 * (J[c1] + J[c2])
            du_dt_edge = vertical_diffusion_variable_K(
                u_edge, z_coord, J_edge, A_v_edge,
            )

        return du_dt_edge, dT_dt, dS_dt

    return physics_fn


def make_kpp_profiles_mpas(config: VerticalMixingConfig) -> Callable:
    """Build KPP profile-only function for MPAS implicit vertical mixing.

    Returns ``(A_v_cells, K_v_cells)`` at half-levels (nCells, nlev-1)
    — the raw KPP-produced viscosity and diffusivity profiles — WITHOUT
    computing or returning any tendencies.  The caller
    (``MPASOceanModel.step``) passes these profiles to the backward-Euler
    implicit solver together with background and convective K.

    The existing ``make_kpp_physics_mpas`` is preserved for backward
    compatibility with ``implicit_vertical_mixing=False``.

    Parameters
    ----------
    config : VerticalMixingConfig
        Must have ``scheme="kpp"`` and ``kpp`` sub-config populated.

    Returns
    -------
    Callable
        ``profiles_fn(state, mesh, z_coord, surface_forcing=None)``
        returning ``(A_v_cells, K_v_cells)`` both shape (nCells, nlev-1).
    """
    cfg = config.kpp

    def profiles_fn(
        state: MPASOceanState,
        mesh,
        z_coord,
        surface_forcing=None,
    ):
        T_3d = state.T.data       # (nCells, nlev)
        S_3d = state.S.data       # (nCells, nlev)
        u_edge = state.u.data     # (nEdges, nlev)
        eta = state.eta.data      # (nCells,)
        H_bathy = state.H_bathy.data
        mask = state.land_mask.data  # (nCells,) — 1=ocean, 0=land

        # Reconstruct cell-centered (u_east, v_north) from edge-normal u.
        u_east_raw, v_north_raw = reconstruct_cell_velocity(u_edge, mesh)

        # Density at cells (land-safe Jacobian).
        J_real = compute_ocean_jacobian(eta, H_bathy, z_coord)
        J = jnp.where(mask > 0.5, J_real, 1.0)
        rho_real = compute_ocean_rho(state, z_coord, J_real)
        rho = jnp.where(mask[:, None] > 0.5, rho_real, _RHO_0)

        # Surface forcing (KPP needs friction velocity and buoyancy flux).
        tau_x = getattr(surface_forcing, "tau_x", None) if surface_forcing else None
        tau_y = getattr(surface_forcing, "tau_y", None) if surface_forcing else None
        q_net = getattr(surface_forcing, "q_net", None) if surface_forcing else None
        fw = getattr(surface_forcing, "freshwater", None) if surface_forcing else None

        Q_sfc_T = None
        B_f = None
        if q_net is not None:
            Q_sfc_T = q_net / (_RHO_0 * _C_SW)
            T_sfc = T_3d[..., 0]
            S_sfc = S_3d[..., 0]
            p_sfc = jnp.zeros_like(T_sfc)
            alpha = thermal_expansion_coeff(T_sfc, S_sfc, p_sfc)
            B_f = -constants.g * alpha * Q_sfc_T

        Q_sfc_S = None
        if fw is not None:
            S_sfc = S_3d[..., 0]
            T_sfc = T_3d[..., 0]
            p_sfc = jnp.zeros_like(T_sfc)
            beta = haline_contraction_coeff(T_sfc, S_sfc, p_sfc)
            Q_sfc_S = -S_sfc * fw / _RHO_0
            B_salt = constants.g * beta * Q_sfc_S
            B_f = B_salt if B_f is None else (B_f + B_salt)

        # Zero-out fields on land cells; fill sub-seafloor levels.
        m3 = mask[:, None]
        u_east_w = jnp.where(m3 > 0.5, u_east_raw, 0.0)
        v_north_w = jnp.where(m3 > 0.5, v_north_raw, 0.0)
        T_w = jnp.where(m3 > 0.5, T_3d, 0.0)
        S_w = jnp.where(m3 > 0.5, S_3d, 35.0)
        if hasattr(z_coord, 'is_active'):
            _active = z_coord.is_active
            _bot_lev = z_coord.bottom_level
            _bot_lev_safe = jnp.clip(_bot_lev, 0, T_3d.shape[1] - 1)
            _row_idx = jnp.arange(T_w.shape[0])
            _T_bot = T_w[_row_idx, _bot_lev_safe]
            _S_bot = S_w[_row_idx, _bot_lev_safe]
            _u_bot = u_east_w[_row_idx, _bot_lev_safe]
            _v_bot = v_north_w[_row_idx, _bot_lev_safe]
            T_w = jnp.where(_active, T_w, _T_bot[:, None])
            S_w = jnp.where(_active, S_w, _S_bot[:, None])
            u_east_w = jnp.where(_active, u_east_w, _u_bot[:, None])
            v_north_w = jnp.where(_active, v_north_w, _v_bot[:, None])

        # Run KPP to get viscosity/diffusivity profiles.
        kpp_out = kpp_vertical_mixing(
            u_east_w, v_north_w, T_w, S_w,
            rho, eta, z_coord, J, cfg,
            tau_x=tau_x, tau_y=tau_y, B_f=B_f,
            Q_sfc_T=Q_sfc_T, Q_sfc_S=Q_sfc_S,
        )

        # Shallow-cell mask: KPP is physically inappropriate for cells
        # with fewer than 5 active levels.
        _min_kpp_levels = 5
        if hasattr(z_coord, 'bottom_level'):
            _n_active = z_coord.bottom_level + 1
            _kpp_mask = (_n_active >= _min_kpp_levels).astype(T_3d.dtype)
        else:
            _kpp_mask = jnp.ones(mask.shape, dtype=T_3d.dtype)

        # A_v and K_v at half-levels (nCells, nlev-1).
        # Mask land cells and shallow cells.
        _kpp_mask_2d = _kpp_mask * mask  # (nCells,)
        m_half = _kpp_mask_2d[:, None]   # broadcast to (nCells, nlev-1)
        A_v_cells = jnp.where(m_half > 0.5, kpp_out.A_v, 0.0)
        K_v_cells = jnp.where(m_half > 0.5, kpp_out.K_v, 0.0)

        # Sub-seafloor interface masking: zero K at interfaces below the
        # deepest active full level.
        if hasattr(z_coord, 'bottom_level'):
            nlev_half = A_v_cells.shape[1]
            k_idx = jnp.arange(nlev_half, dtype=z_coord.bottom_level.dtype)
            bot_c = z_coord.bottom_level  # (nCells,)
            active_half_c = (k_idx[None, :] < bot_c[:, None]).astype(A_v_cells.dtype)
            A_v_cells = A_v_cells * active_half_c
            K_v_cells = K_v_cells * active_half_c

        return A_v_cells, K_v_cells

    return profiles_fn
