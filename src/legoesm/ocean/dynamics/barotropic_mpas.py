"""MPAS barotropic (free-surface) solver on Voronoi meshes.

Split-explicit forward-backward substeps for the barotropic mode.
Updates sea surface height (eta) and depth-averaged normal velocity
(u_bar) using the fast gravity-wave CFL.

Coriolis treatment: ``f·v_t(u_bar)`` is applied at every substep with
the *current-substep* ``u_bar`` (evolving, semi-implicit Heun option).
``F_slow_u`` passed in by the caller is the depth-mean of the full 3D
baroclinic tendency with the planetary-Coriolis part subtracted (see
``ocean_pe_mpas.py``), so the barotropic substep can add online
evolving Coriolis without double-counting. This matches the lat-lon
C-grid pattern. Without this split, a frozen-Coriolis F_slow_u
accumulates over 30 substeps per baroclinic step and grows a
near-inertial numerical mode (τ ~ 1/f, ~0.2 days at mid-latitudes).

References
----------
- Ringler, T. D., et al. (2013). Ocean Modelling, 69, 211-232.
- Higdon, R. L. (2005). J. Comput. Phys., 205(1), 194-224.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.operators_voronoi import (
    divergence_cell,
    gradient_edge,
    tangential_velocity,
    edge_thickness as _edge_avg,
)
from legoesm.ocean.vertical import compute_layer_thickness
from legoesm.ocean.dynamics.eta_floor import clamp_and_redistribute as _clamp_redistribute
from legoesm.ocean.dynamics.barotropic_common import (
    bebt_blend,
    compute_filter_weights,
    maxvel_clip,
)
from legoesm.ocean.dynamics.ocean_tendency_common import implicit_bottom_drag_factor


def barotropic_substeps_mpas(
    state,
    mesh,
    z_coord,
    config,
    dt_baro,
    n_substeps,
    F_slow_eta=None,
    F_slow_u=None,
):
    """Run barotropic substeps on MPAS Voronoi mesh.

    Forward-backward scheme:
        1. Forward:  eta^{n+1} = eta^n - dt * div(H_e * u_bar^n) + dt * F_slow_eta
        2. Backward: u_bar^{n+1} = u_bar^n + dt * (-g*grad(eta^{n+1}) + f*v_t)

    The 3D baroclinic tendency is already applied to u before calling
    this function, so u_bar is computed from the updated state.  No
    F_slow_u is needed (same as cubed-sphere barotropic.py).

    Parameters
    ----------
    state : MPASOceanState
        State with 3D velocity already updated by baroclinic tendency.
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    config : MPASOceanConfig
    dt_baro : float
        Barotropic substep size [s].
    n_substeps : int
        Number of barotropic substeps.
    F_slow_eta : jax.Array or None, shape (nCells,)
        Slow forcing for eta (e.g., freshwater mass flux) [m/s].

    Returns
    -------
    eta_avg : jax.Array, shape (nCells,)
        Time-averaged sea surface height over substeps [m].
    u_bar_avg : jax.Array, shape (nEdges,)
        Time-averaged depth-averaged velocity over substeps [m/s].
    Hu_avg : jax.Array, shape (nEdges,)
        Time-averaged depth-integrated edge transport [m²/s].
        All three are time-averaged to filter fast barotropic gravity
        waves from the baroclinic coupling (Higdon 2005, issue #149).
    """
    g = config.g
    mask = state.land_mask.data  # (nCells,)
    H_bathy = state.H_bathy.data
    eta = state.eta.data
    u_3d = state.u.data  # (nEdges, nlev)

    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]

    # Compute layer thickness and depth-averaged velocity
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )  # (nCells, nlev)

    # Edge layer thickness for each level
    h_e_k = 0.5 * (h_k[c1] + h_k[c2])  # (nEdges, nlev)

    # Depth-integrated transport: sum_k(u_k * h_e_k)
    Hu_bar = jnp.sum(u_3d * h_e_k, axis=1)  # (nEdges,)

    # Total water column at edges
    H_total_cell = eta + H_bathy  # (nCells,)
    H_total_cell = jnp.maximum(H_total_cell, config.min_water_column_m)
    H_e = _edge_avg(H_total_cell, mesh)  # (nEdges,)

    # Depth-averaged velocity (from state that already includes baroclinic tendency)
    u_bar = Hu_bar / jnp.maximum(H_e, 1e-10)

    if F_slow_eta is None:
        F_slow_eta = jnp.zeros_like(eta)
    # F_slow_u is PGF+KE+zeta-advection+viscosity+bottom-drag with the
    # planetary Coriolis contribution EXCLUDED (see ocean_pe_mpas.py).
    # The barotropic substep applies online evolving f·v_t(u_bar) below,
    # matching the lat-lon C-grid pattern.
    if F_slow_u is None:
        F_slow_u = jnp.zeros_like(u_bar)

    # --- Fix 1: Neumann fill for eta before gradient ---
    # Fill land-cell eta with nearest-ocean-neighbor average so that
    # gradient_edge sees smooth fields at coastlines instead of the
    # sharp ocean-to-zero jump from masking.
    from legoesm.ocean.dynamics.mpas_fill import fill_land_cells_mpas

    def _fill_land_cells_mpas(field_cell, mask_cell):
        return fill_land_cells_mpas(field_cell, mask_cell, c1, c2)

    # --- Fix 2: Barotropic Laplacian diffusion ---
    baro_alpha_val = config.barotropic_diffusion_alpha
    dt_ref = config.barotropic_diffusion_dt_ref
    use_baro_diffusion = baro_alpha_val > 0.0

    if use_baro_diffusion:
        # Edge-centered diffusion coefficient for flux-form diffusion:
        # div(nu_edge * grad(eta)) is exactly conservative (divergence
        # theorem), unlike nu_cell * div(grad(eta)) which leaks volume
        # when cell areas are non-uniform.
        nu_dt_edge = baro_alpha_val * (dt_baro / dt_ref) * (
            0.5 * (mesh.areaCell[c1] + mesh.areaCell[c2])
        )

    # Divergence damping on barotropic velocity: grad(div(u_bar)).
    # Targets the divergent mode that creates the eta checkerboard (#205).
    use_div_damp = config.barotropic_div_damp > 0.0
    if use_div_damp:
        div_damp_coeff = jnp.asarray(
            config.barotropic_div_damp, dtype=eta.dtype,
        ) * (dt_baro / jnp.asarray(config.barotropic_diffusion_dt_ref, dtype=eta.dtype))
        div_damp_area_edge = 0.5 * (mesh.areaCell[c1] + mesh.areaCell[c2])

    # --- Fix 3: Semi-implicit Coriolis (trapezoidal predictor-corrector) ---
    # On Voronoi meshes, the (u, v_tangential) decomposition doesn't
    # allow a direct Crank-Nicolson solve. Instead, use a trapezoidal
    # predictor-corrector: predict u_star with old v_t, recompute v_t
    # from u_star, and average. This is second-order and avoids the
    # explicit f*dt instability at high latitudes.
    use_semi_implicit = config.semi_implicit_coriolis

    # Eta floor: prevent water column from going below minimum depth.
    # Matches cubed-sphere and lat-lon barotropic solvers.
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta.dtype)
    eta_floor = (min_water_col - H_bathy) * mask
    _area_cell = mesh.areaCell  # for mass-conserving floor clamp (#176)

    # Cosine time filter and BEBT/MAXVEL parameters
    bebt = config.bebt
    _maxvel = config.maxvel_barotropic
    use_maxvel = _maxvel > 0.0

    use_cosine_filter = config.barotropic_time_filter == "cosine"
    w_filter, w_total = compute_filter_weights(
        n_substeps, eta.dtype, use_cosine=use_cosine_filter,
    )

    # Accumulators for time-averaged barotropic fields (issues #145, #149, #102).
    Hu_sum = jnp.zeros_like(u_bar)
    eta_sum = jnp.zeros_like(eta)
    ubar_sum = jnp.zeros_like(u_bar)

    # Forward-backward substeps via scan
    _eta_dtype = eta.dtype
    _ubar_dtype = u_bar.dtype

    def _substep(carry, w_i):
        eta_c, u_bar_c, Hu_sum_c, eta_sum_c, ubar_sum_c = carry

        # Total depth at edges (updated with current eta)
        H_c = jnp.maximum(eta_c + H_bathy, config.min_water_column_m)
        H_e_c = _edge_avg(H_c, mesh)

        # Forward: update eta (continuity + freshwater mass source)
        transport = H_e_c * u_bar_c * edge_mask

        # Accumulate transport for time-averaged tracer advection
        Hu_sum_new = Hu_sum_c + transport.astype(_eta_dtype)

        eta_next = eta_c - dt_baro * divergence_cell(transport, mesh) * mask + dt_baro * F_slow_eta * mask
        eta_next = _clamp_redistribute(eta_next, eta_floor, mask, _area_cell)

        # Backward: update u_bar using new eta
        # BEBT: blend new/old eta for semi-implicit PGF (#205)
        eta_pgf = bebt_blend(eta_next, eta_c, bebt)
        eta_filled = _fill_land_cells_mpas(eta_pgf, mask)
        grad_eta = gradient_edge(eta_filled, mesh)

        # PGF + evolving Coriolis + slow forcing.
        # F_slow_u now has the depth-mean planetary Coriolis subtracted
        # in ocean_pe_mpas.py, so we always apply online f·v_t(u_bar) here
        # — otherwise the barotropic u_bar loses its rotational restoring
        # torque inside the substep loop and a near-inertial numerical
        # mode (τ ~ 1/f) grows on the order of 0.2 days at mid-latitudes.
        if use_semi_implicit:
            # Heun predictor-corrector for Coriolis (#172 docs fix)
            v_t_old = tangential_velocity(u_bar_c, mesh)
            u_star = u_bar_c + dt_baro * (
                -g * grad_eta + mesh.fEdge * v_t_old + F_slow_u
            ) * edge_mask
            v_t_star = tangential_velocity(u_star, mesh)
            u_bar_next = u_bar_c + dt_baro * (
                -g * grad_eta + mesh.fEdge * 0.5 * (v_t_old + v_t_star)
                + F_slow_u
            ) * edge_mask
        else:
            v_t_old = tangential_velocity(u_bar_c, mesh)
            u_bar_next = u_bar_c + dt_baro * (
                -g * grad_eta + mesh.fEdge * v_t_old + F_slow_u
            ) * edge_mask

        # Divergence damping: add nu * grad(div(u_bar)) (#205).
        if use_div_damp:
            div_ubar = divergence_cell(u_bar_next * edge_mask, mesh) * mask
            div_filled = _fill_land_cells_mpas(div_ubar, mask)
            grad_div = gradient_edge(div_filled, mesh)
            u_bar_next = (
                u_bar_next + div_damp_coeff * div_damp_area_edge * grad_div
            ) * edge_mask

        # Optional barotropic damping (Rayleigh drag)
        if config.barotropic_damping > 0:
            u_bar_next = u_bar_next * (1.0 - dt_baro * config.barotropic_damping)

        # Bottom drag on barotropic velocity: -r * U_bar / H_total.
        # r is in [m/s] — resolution-independent bottom stress.
        if config.bottom_drag_r > 0:
            u_bar_next = u_bar_next * implicit_bottom_drag_factor(
                dt_baro, config.bottom_drag_r, H_e_c,
            )

        # MAXVEL clipping
        if use_maxvel:
            u_bar_next = maxvel_clip(u_bar_next, _maxvel)

        # Barotropic Laplacian diffusion on eta (flux-form: conservative).
        # Uses div(nu_edge * grad(eta)) instead of nu_cell * div(grad(eta))
        # so that volume is exactly conserved by the divergence theorem.
        if use_baro_diffusion:
            eta_filled = _fill_land_cells_mpas(eta_next, mask)
            grad_e = gradient_edge(eta_filled, mesh)
            diff_flux = nu_dt_edge * grad_e * edge_mask
            eta_next = (
                eta_next + divergence_cell(diff_flux, mesh)
            ) * mask
            eta_next = _clamp_redistribute(eta_next, eta_floor, mask, _area_cell)

        # Accumulate eta and u_bar with cosine filter weights
        eta_sum_new = eta_sum_c + w_i * eta_next.astype(_eta_dtype)
        ubar_sum_new = ubar_sum_c + w_i * u_bar_next.astype(_eta_dtype)

        # Cast back to input dtype (mesh ops may promote to float64)
        return (eta_next.astype(_eta_dtype), u_bar_next.astype(_ubar_dtype),
                Hu_sum_new.astype(_eta_dtype),
                eta_sum_new, ubar_sum_new), None

    (eta_new, u_bar_new, Hu_sum_f, eta_sum_f, ubar_sum_f), _ = jax.lax.scan(
        _substep, (eta, u_bar, Hu_sum, eta_sum, ubar_sum),
        w_filter, length=n_substeps,
    )

    # Time-averaged barotropic fields
    Hu_avg = Hu_sum_f / n_substeps  # transport: always box-filtered
    eta_avg = eta_sum_f / w_total   # eta/velocity: cosine or box filtered
    u_bar_avg = ubar_sum_f / w_total

    return eta_avg, u_bar_avg, Hu_avg


def reconcile_3d_velocity(u_3d, u_bar_old, u_bar_new, mesh, mask):
    """Reconcile 3D velocity with updated barotropic velocity.

    Preserves baroclinic structure (deviations from depth-mean) while
    replacing the depth-mean with the barotropic solution:

        u_3d_new = (u_3d - u_bar_old) + u_bar_new

    This ensures depth_avg(u_3d_new) = u_bar_new exactly.

    Parameters
    ----------
    u_3d : jax.Array, shape (nEdges, nlev)
    u_bar_old : jax.Array, shape (nEdges,)
        Depth-averaged velocity BEFORE barotropic substeps.
    u_bar_new : jax.Array, shape (nEdges,)
        Depth-averaged velocity AFTER barotropic substeps.
    mesh : VoronoiMesh
    mask : jax.Array, shape (nCells,)

    Returns
    -------
    jax.Array, shape (nEdges, nlev)
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]

    # Baroclinic deviation + new barotropic mean
    u_prime = u_3d - u_bar_old[:, jnp.newaxis]
    return (u_prime + u_bar_new[:, jnp.newaxis]) * edge_mask[:, jnp.newaxis]
