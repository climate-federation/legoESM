"""MPAS barotropic (free-surface) solver on Voronoi meshes.

Split-explicit forward-backward substeps for the barotropic mode.
Updates sea surface height (eta) and depth-averaged normal velocity
(u_bar) using the fast gravity-wave CFL.

No explicit F_slow for velocity: the 3D baroclinic tendency has already
been applied to u before this function is called, so u_bar is computed
from the updated state. Including F_slow would double-count the
depth-averaged baroclinic tendency. (Same pattern as cubed-sphere
barotropic.py.)

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
    vector_laplacian_del2,
)
from legoesm.ocean.vertical import compute_layer_thickness


def barotropic_substeps_mpas(
    state,
    mesh,
    z_coord,
    config,
    dt_baro,
    n_substeps,
    F_slow_eta=None,
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
    eta_new : jax.Array, shape (nCells,)
    u_bar_new : jax.Array, shape (nEdges,)
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

    # --- Fix 1: Neumann fill for eta before gradient ---
    # Fill land-cell eta with nearest-ocean-neighbor average so that
    # gradient_edge sees smooth fields at coastlines instead of the
    # sharp ocean-to-zero jump from masking.
    def _fill_land_cells_mpas(field_cell, mask_cell):
        """Replace land values with ocean-neighbor average (Neumann BC)."""
        # Accumulate from all edges: for edge e connecting c1, c2,
        # add c2's contribution to c1 and vice versa.
        nbr_sum = jnp.zeros_like(field_cell)
        nbr_cnt = jnp.zeros_like(field_cell)
        nbr_sum = nbr_sum.at[c1].add(field_cell[c2] * mask_cell[c2])
        nbr_cnt = nbr_cnt.at[c1].add(mask_cell[c2])
        nbr_sum = nbr_sum.at[c2].add(field_cell[c1] * mask_cell[c1])
        nbr_cnt = nbr_cnt.at[c2].add(mask_cell[c1])
        nbr_avg = nbr_sum / jnp.maximum(nbr_cnt, 1.0)
        return jnp.where(mask_cell > 0.5, field_cell, nbr_avg)

    # --- Fix 2: Barotropic Laplacian diffusion ---
    # Scalar cell Laplacian: del2(phi) = div(grad(phi)) on Voronoi.
    # Edge-based velocity diffusion: vector_laplacian_del2(u, mesh).
    baro_alpha_val = config.barotropic_diffusion_alpha
    dt_ref = config.barotropic_diffusion_dt_ref
    use_baro_diffusion = baro_alpha_val > 0.0

    if use_baro_diffusion:
        nu_dt_cell = baro_alpha_val * (dt_baro / dt_ref) * mesh.areaCell
        # For edge diffusion, use average of adjacent cell areas
        nu_dt_edge = baro_alpha_val * (dt_baro / dt_ref) * (
            0.5 * (mesh.areaCell[c1] + mesh.areaCell[c2])
        )

    def _del2_cell(phi_cell, mask_cell):
        """Scalar Laplacian on Voronoi cells: div(grad(phi))."""
        phi_filled = _fill_land_cells_mpas(phi_cell, mask_cell)
        grad_e = gradient_edge(phi_filled, mesh)
        return divergence_cell(grad_e, mesh)

    # --- Fix 3: Semi-implicit Coriolis (trapezoidal predictor-corrector) ---
    # On Voronoi meshes, the (u, v_tangential) decomposition doesn't
    # allow a direct Crank-Nicolson solve. Instead, use a trapezoidal
    # predictor-corrector: predict u_star with old v_t, recompute v_t
    # from u_star, and average. This is second-order and avoids the
    # explicit f*dt instability at high latitudes.
    use_semi_implicit = config.semi_implicit_coriolis

    # Forward-backward substeps via scan
    _eta_dtype = eta.dtype
    _ubar_dtype = u_bar.dtype

    def _substep(carry, _):
        eta_c, u_bar_c = carry

        # Total depth at edges (updated with current eta)
        H_c = jnp.maximum(eta_c + H_bathy, config.min_water_column_m)
        H_e_c = _edge_avg(H_c, mesh)

        # Forward: update eta (continuity + freshwater mass source)
        transport = H_e_c * u_bar_c * edge_mask
        eta_next = eta_c - dt_baro * divergence_cell(transport, mesh) * mask + dt_baro * F_slow_eta * mask

        # Backward: update u_bar using new eta
        # Fill land cells before gradient to prevent spurious PGF
        eta_filled = _fill_land_cells_mpas(eta_next, mask)
        grad_eta = gradient_edge(eta_filled, mesh)

        # Coriolis + PGF
        v_t_old = tangential_velocity(u_bar_c, mesh)

        if use_semi_implicit:
            # Trapezoidal predictor-corrector:
            # 1. Predict with old Coriolis
            u_star = u_bar_c + dt_baro * (
                -g * grad_eta + mesh.fEdge * v_t_old
            ) * edge_mask
            # 2. Recompute tangential velocity from predicted u
            v_t_star = tangential_velocity(u_star, mesh)
            # 3. Correct with averaged Coriolis
            u_bar_next = u_bar_c + dt_baro * (
                -g * grad_eta + mesh.fEdge * 0.5 * (v_t_old + v_t_star)
            ) * edge_mask
        else:
            # Explicit Coriolis (original)
            u_bar_next = u_bar_c + dt_baro * (
                -g * grad_eta + mesh.fEdge * v_t_old
            ) * edge_mask

        # Optional barotropic damping (Rayleigh drag)
        if config.barotropic_damping > 0:
            u_bar_next = u_bar_next * (1.0 - dt_baro * config.barotropic_damping)

        # Barotropic Laplacian diffusion (matches CS and LL solvers)
        if use_baro_diffusion:
            # Eta: scalar Laplacian on cells
            eta_next = (
                eta_next + nu_dt_cell * _del2_cell(eta_next, mask)
            ) * mask
            # Velocity: vector Laplacian on edges
            del2_u = vector_laplacian_del2(u_bar_next, mesh)
            u_bar_next = (u_bar_next + nu_dt_edge * del2_u) * edge_mask

        # Cast back to input dtype (mesh ops may promote to float64)
        return (eta_next.astype(_eta_dtype), u_bar_next.astype(_ubar_dtype)), None

    (eta_new, u_bar_new), _ = jax.lax.scan(
        _substep, (eta, u_bar), None, length=n_substeps,
    )

    return eta_new, u_bar_new


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
