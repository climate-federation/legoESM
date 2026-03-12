"""MPAS barotropic (free-surface) solver on Voronoi meshes.

Split-explicit forward-backward substeps for the barotropic mode.
Updates sea surface height (eta) and depth-averaged normal velocity
(u_bar) using the fast gravity-wave CFL.

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


def barotropic_substeps_mpas(
    state,
    mesh,
    z_coord,
    config,
    dt_baro,
    n_substeps,
    F_slow_u=None,
    F_slow_eta=None,
):
    """Run barotropic substeps on MPAS Voronoi mesh.

    Forward-backward scheme:
        1. Forward:  eta^{n+1} = eta^n - dt * div(H_e * u_bar^n) + dt * F_slow_eta
        2. Backward: u_bar^{n+1} = u_bar^n + dt * (-g*grad(eta^{n+1}) + f*v_t + F_slow)

    Parameters
    ----------
    state : MPASOceanState
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    config : MPASOceanConfig
    dt_baro : float
        Barotropic substep size [s].
    n_substeps : int
        Number of barotropic substeps.
    F_slow_u : jax.Array or None, shape (nEdges,)
        Slow (baroclinic) forcing for u_bar.
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

    # Depth-averaged velocity
    u_bar = Hu_bar / jnp.maximum(H_e, 1e-10)

    # Slow forcing (default zero)
    if F_slow_u is None:
        F_slow_u = jnp.zeros_like(u_bar)
    if F_slow_eta is None:
        F_slow_eta = jnp.zeros_like(eta)

    # Forward-backward substeps via scan
    def _substep(carry, _):
        eta_c, u_bar_c = carry

        # Total depth at edges (updated with current eta)
        H_c = jnp.maximum(eta_c + H_bathy, config.min_water_column_m)
        H_e_c = _edge_avg(H_c, mesh)

        # Forward: update eta (continuity + freshwater mass source)
        transport = H_e_c * u_bar_c * edge_mask
        eta_next = eta_c - dt_baro * divergence_cell(transport, mesh) * mask + dt_baro * F_slow_eta * mask

        # Backward: update u_bar using new eta
        grad_eta = gradient_edge(eta_next, mesh)

        # Coriolis: f * v_tangential
        v_t = tangential_velocity(u_bar_c, mesh)
        coriolis = mesh.fEdge * v_t

        u_bar_next = u_bar_c + dt_baro * (
            -g * grad_eta + coriolis + F_slow_u
        ) * edge_mask

        # Optional barotropic damping
        if config.barotropic_damping > 0:
            u_bar_next = u_bar_next * (1.0 - dt_baro * config.barotropic_damping)

        return (eta_next, u_bar_next), None

    (eta_new, u_bar_new), _ = jax.lax.scan(
        _substep, (eta, u_bar), None, length=n_substeps,
    )

    return eta_new, u_bar_new


def reconcile_3d_velocity(u_3d, u_bar_old, u_bar_new, h_k, mesh, mask):
    """Reconcile 3D velocity with updated barotropic velocity.

    Preserves baroclinic structure while updating the depth-mean.

    u_3d_new = u_3d + (u_bar_new - u_bar_old) for each level.

    Parameters
    ----------
    u_3d : jax.Array, shape (nEdges, nlev)
    u_bar_old, u_bar_new : jax.Array, shape (nEdges,)
    h_k : jax.Array, shape (nCells, nlev)
    mesh : VoronoiMesh
    mask : jax.Array, shape (nCells,)

    Returns
    -------
    jax.Array, shape (nEdges, nlev)
    """
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]

    # Correction applied uniformly across levels
    delta_u_bar = (u_bar_new - u_bar_old) * edge_mask
    return u_3d + delta_u_bar[:, jnp.newaxis]
