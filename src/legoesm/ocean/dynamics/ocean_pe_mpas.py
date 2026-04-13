"""MPAS ocean baroclinic tendencies using TRiSK operators.

Boussinesq hydrostatic primitive equations in vector-invariant form
on MPAS Voronoi (C-grid) meshes. Uses the TRiSK discretization from
Ringler et al. (2010).

Equations (per layer k):
    du/dt = q_e * F_q - grad(KE + p'/ρ₀ + g·η) - w·du'/dz + A_h·del2(u) + A_v·d²u/dz²
    d(h·T)/dt = -div(h·u·T) + K_h·h·lap(T) + K_v·d²T/dz²
    d(h·S)/dt = -div(h·u·S) + K_h·h·lap(S) + K_v·d²S/dz²
    dη/dt = -Σ_k div(h_k · u_k)

References
----------
- Ringler, T. D., et al. (2010). J. Comput. Phys., 229(9), 3065-3090.
- Ringler, T. D., et al. (2013). Ocean Modelling, 69, 211-232.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import MPASOceanState, MPASOceanTendencies
from legoesm.core.operators_voronoi import (
    divergence_cell_3d,
    gradient_edge_3d,
    curl_vertex_3d,
    kinetic_energy_cell_3d,
    pv_flux_energy_conserving_3d,
    pv_flux_enstrophy_conserving_3d,
    vector_laplacian_del2_3d,
    vertex_thickness_3d,
)
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.eos import compute_hydrostatic_pressure, make_eos_fn
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
    diagnose_w_from_flux_div,
    vertical_advection_ocean,
)
from legoesm.ocean.freshwater import (
    FreshwaterForcing,
    freshwater_eta_tendency,
    virtual_salt_flux,
)


def mpas_ocean_baroclinic_tendencies(
    state: MPASOceanState,
    mesh,
    z_coord: OceanZStarCoordinate,
    config: MPASOceanConfig = MPASOceanConfig(),
    freshwater: FreshwaterForcing | None = None,
    physics_fn=None,
    surface_forcing=None,
) -> MPASOceanTendencies:
    """Compute baroclinic (slow) tendencies for MPAS ocean.

    Parameters
    ----------
    state : MPASOceanState
    mesh : VoronoiMesh
    z_coord : OceanZStarCoordinate
    config : MPASOceanConfig
    freshwater : FreshwaterForcing or None
        Freshwater forcing. If None, no freshwater terms are applied.

    Returns
    -------
    MPASOceanTendencies
    """
    g = config.g
    rho_0 = config.rho_0

    u_3d = state.u.data          # (nEdges, nlev)
    T_3d = state.T.data          # (nCells, nlev)
    S_3d = state.S.data          # (nCells, nlev)
    eta = state.eta.data         # (nCells,)
    H_bathy = state.H_bathy.data  # (nCells,)
    mask = state.land_mask.data  # (nCells,)

    c1 = mesh.cellsOnEdge[0]  # (nEdges,)
    c2 = mesh.cellsOnEdge[1]  # (nEdges,)

    from legoesm.ocean.dynamics.mpas_fill import fill_land_cells_mpas

    def _fill_land_cells_mpas(field_cell, mask_cell):
        return fill_land_cells_mpas(field_cell, mask_cell, c1, c2)

    # ---- Layer thickness and Jacobian ----
    jacobian = compute_ocean_jacobian(
        eta, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )  # (nCells, nlev)

    # ---- Density and hydrostatic pressure ----
    # Fill land-cell T/S with ocean-neighbor values before EOS so that
    # density on land ≈ ρ₀, preventing spurious ρ' at coastlines.
    T_filled = _fill_land_cells_mpas(T_3d, mask)
    S_filled = _fill_land_cells_mpas(S_3d, mask)
    eos_fn = make_eos_fn(config.eos, getattr(config, 'eos_linear', None))
    # Use REFERENCE Jacobian (J=1, eta=0) for the hydrostatic pressure
    # in the EOS iteration.  The barotropic solver handles the
    # free-surface pressure gradient g*grad(eta); using the actual J
    # here would create a spatially-varying pressure even for uniform
    # T/S, double-counting the barotropic forcing.
    # (Matches latlon C-grid: ocean_pe_latlon_cgrid.py:207-213)
    J_ref = jnp.ones_like(jacobian)
    eta_ref = jnp.zeros_like(eta)
    rho = eos_fn(T_filled, S_filled, jnp.zeros_like(T_3d))
    for _ in range(2):
        p_hydro = compute_hydrostatic_pressure(
            rho, eta_ref, z_coord.dz_ref, J_ref, rho_0, g,
        )
        rho = eos_fn(T_filled, S_filled, p_hydro)
    p_hydro = compute_hydrostatic_pressure(
        rho, eta_ref, z_coord.dz_ref, J_ref, rho_0, g,
    )  # (nCells, nlev)

    # Baroclinic pressure anomaly: built from rho' = rho - rho_0 only,
    # using REFERENCE layer thickness dz_ref (not actual dz = dz_ref*J).
    # This ensures the baroclinic PGF is independent of eta, avoiding
    # overlap with the barotropic solver's -g*grad(eta).
    rho_prime = rho - rho_0
    dp_layer = rho_prime * g * z_coord.dz_ref
    p_prime = jnp.cumsum(dp_layer, axis=-1) - dp_layer
    p_prime = p_prime + 0.5 * dp_layer  # (nCells, nlev)

    # Fill land cells in p_prime before gradient_edge so the 2-cell
    # stencil sees smooth values at coastlines.
    p_prime = _fill_land_cells_mpas(p_prime, mask)

    # ---- Edge mask for land boundaries ----
    edge_mask = mask[c1] * mask[c2]  # 1 only if both cells are ocean

    # ---- Depth-averaged velocity and perturbation ----
    # The baroclinic step must operate on PERTURBATION velocity
    # u' = u - u_bar to avoid double-counting with the barotropic
    # solver.  The barotropic solver handles the depth-mean Coriolis,
    # pressure gradient, and KE; the baroclinic step handles only the
    # vertical shear (perturbation) component.
    # (Matches latlon C-grid: ocean_pe_latlon_cgrid.py:256-266)
    h_e_3d = 0.5 * (h_k[c1] + h_k[c2])  # (nEdges, nlev)
    H_e = jnp.maximum(jnp.sum(h_e_3d, axis=1), config.min_water_column_m)
    u_bar = jnp.sum(u_3d * h_e_3d, axis=1) / jnp.maximum(H_e, 1e-10)
    u_bar = u_bar * edge_mask  # (nEdges,)
    u_prime_3d = u_3d - u_bar[:, jnp.newaxis]  # (nEdges, nlev)

    # ---- Thickness flux and vertical velocity ----
    # Compute flux divergence BEFORE momentum tendencies because we need
    # w for vertical advection of momentum (issue #152).
    thickness_flux = u_3d * h_e_3d * edge_mask[:, jnp.newaxis]  # (nEdges, nlev)
    div_flux = divergence_cell_3d(thickness_flux, mesh)  # (nCells, nlev)

    # Diagnose w from full-velocity flux divergence (matching latlon pattern:
    # ocean_pe_latlon_cgrid.py:302-305)
    w = diagnose_w_from_flux_div(
        div_flux, z_coord, thickness_weighted=True,
    )  # (nCells, nlev+1)

    # ---- Momentum and tracer tendencies (batched 3D) ----
    # Uses 3D operators that gather connectivity arrays once for all
    # levels, instead of per-level scan with repeated 1D gathers.
    # Coriolis is EXCLUDED — applied separately as forward-backward
    # (Matsuno) step in the model step function.
    # h_e_3d already computed above for u_bar; reuse it here.

    # Kinetic energy from perturbation velocity
    ke = kinetic_energy_cell_3d(u_prime_3d, mesh)  # (nCells, nlev)

    # Bernoulli function: KE(u') + p'/rho_0
    bernoulli = ke + p_prime / rho_0  # (nCells, nlev)

    # Pressure gradient + Bernoulli
    grad_B = gradient_edge_3d(bernoulli, mesh)  # (nEdges, nlev)

    # PV flux: RELATIVE VORTICITY from perturbation velocity ONLY.
    zeta_v = curl_vertex_3d(u_prime_3d, mesh)  # (nVertices, nlev)
    h_v = vertex_thickness_3d(h_k, mesh)  # (nVertices, nlev)
    h_v_safe = jnp.maximum(h_v, 1e-10)
    q_vort = zeta_v / h_v_safe  # PV without f
    if config.pv_scheme == "energy":
        pv_flux = pv_flux_energy_conserving_3d(u_prime_3d, h_k, q_vort, mesh)
    else:
        pv_flux = pv_flux_enstrophy_conserving_3d(u_prime_3d, h_k, q_vort, mesh)

    # Horizontal viscosity on perturbation velocity
    visc = config.A_h * vector_laplacian_del2_3d(u_prime_3d, mesh)

    # Vertical advection of perturbation momentum: -w * du'/dz
    # Interpolate w and Jacobian from cells to edges (same averaging as
    # _vertical_diffusion uses for edge Jacobian).
    # Matches latlon C-grid: ocean_pe_latlon_cgrid.py:342-347.
    w_e = 0.5 * (w[c1] + w[c2])  # (nEdges, nlev+1)
    J_e = 0.5 * (jacobian[c1] + jacobian[c2])  # (nEdges,)
    vert_adv_u = vertical_advection_ocean(u_prime_3d, w_e, z_coord, J_e)

    du_dt_3d = (-grad_B + pv_flux + visc + vert_adv_u) * edge_mask[:, jnp.newaxis]

    # ---- Tracer tendencies (diffusion + physics only) ----
    # Horizontal AND vertical tracer advection are handled in the step()
    # function using barotropic-averaged transport (Hallberg 1997, #102, #145).
    # This matches the latlon C-grid pattern (ocean_pe_latlon_cgrid.py).
    h_safe = jnp.maximum(h_k, 1e-10)  # (nCells, nlev)

    # Horizontal tracer diffusion: K_h * lap(T)
    grad_T = gradient_edge_3d(T_3d, mesh) * edge_mask[:, jnp.newaxis]
    dT_dt_3d = config.K_h * divergence_cell_3d(grad_T, mesh) / h_safe * h_k
    grad_S = gradient_edge_3d(S_3d, mesh) * edge_mask[:, jnp.newaxis]
    dS_dt_3d = config.K_h * divergence_cell_3d(grad_S, mesh) / h_safe * h_k

    # Mask land cells
    dT_dt_3d = dT_dt_3d * mask[:, jnp.newaxis]
    dS_dt_3d = dS_dt_3d * mask[:, jnp.newaxis]

    # ---- Vertical mixing ----
    dz_half = z_coord.dz_half_ref  # (nlev-1,)
    dz = z_coord.dz_ref  # (nlev,)

    # Vertical viscosity on perturbation velocity: d/dz(A_v * du'/dz)
    du_dt_3d = du_dt_3d + _vertical_diffusion(
        u_prime_3d, dz_half, dz, jacobian=jacobian, coeff=config.A_v, is_edge=True,
        mesh=mesh,
    )

    # Vertical tracer diffusion
    dT_dt_3d = dT_dt_3d + _vertical_diffusion(
        T_3d, dz_half, dz, jacobian=jacobian, coeff=config.K_v, is_edge=False,
        mesh=mesh,
    ) * mask[:, jnp.newaxis]
    dS_dt_3d = dS_dt_3d + _vertical_diffusion(
        S_3d, dz_half, dz, jacobian=jacobian, coeff=config.K_v, is_edge=False,
        mesh=mesh,
    ) * mask[:, jnp.newaxis]

    # ---- Physics (surface forcing, bottom drag, etc.) ----
    if physics_fn is not None:
        phys = physics_fn(state, mesh, z_coord, surface_forcing)
        du_dt_3d = du_dt_3d + phys.du_dt.data
        dT_dt_3d = dT_dt_3d + phys.dT_dt.data
        dS_dt_3d = dS_dt_3d + phys.dS_dt.data

    # ---- Free surface tendency ----
    # deta/dt = -sum_k div(u_k * h_e_k)
    deta_dt = -jnp.sum(div_flux, axis=1) * mask  # (nCells,)

    # ---- Freshwater forcing ----
    if freshwater is not None and config.freshwater_closure != "none":
        # Free-surface mass flux: deta/dt += F_fw / rho_0
        deta_dt = deta_dt + freshwater_eta_tendency(freshwater, config.rho_0) * mask

        # Virtual salt flux: dS/dt = -S_ref * F_fw / (rho_0 * dz_0)
        dz_0 = h_k[:, 0]  # top layer thickness (nCells,)
        dS_fw = virtual_salt_flux(freshwater, config.S_ref, dz_0, config.rho_0)
        dS_dt_3d = dS_dt_3d.at[:, 0].add(dS_fw * mask)

    return MPASOceanTendencies(
        du_dt=Field(data=du_dt_3d, name="du_dt",
                    dims=("nEdges", "nlev"), units="m/s²"),
        dT_dt=Field(data=dT_dt_3d, name="dT_dt",
                    dims=("nCells", "nlev"), units="degC/s"),
        dS_dt=Field(data=dS_dt_3d, name="dS_dt",
                    dims=("nCells", "nlev"), units="PSU/s"),
        deta_dt=Field(data=deta_dt, name="deta_dt",
                      dims=("nCells",), units="m/s"),
    )


def _vertical_diffusion(field_3d, dz_half, dz, jacobian, coeff, is_edge, mesh):
    """Compute vertical diffusion d/dz(coeff * df/dz).

    Parameters
    ----------
    field_3d : jax.Array, shape (n, nlev)
    dz_half : jax.Array, shape (nlev-1,)
    dz : jax.Array, shape (nlev,)
    jacobian : jax.Array or None, shape (nCells,) or None
    coeff : float
    is_edge : bool
        If True, field lives on edges (use edge-averaged jacobian).
    mesh : VoronoiMesh

    Returns
    -------
    jax.Array, shape (n, nlev)
    """
    nlev = field_3d.shape[1]
    if nlev < 2:
        return jnp.zeros_like(field_3d)

    # Scale dz by jacobian if available
    if jacobian is not None and not is_edge:
        J = jacobian[:, jnp.newaxis]  # (nCells, 1)
    elif jacobian is not None and is_edge:
        c1 = mesh.cellsOnEdge[0]
        c2 = mesh.cellsOnEdge[1]
        J = 0.5 * (jacobian[c1] + jacobian[c2])  # (nEdges,)
        J = J[:, jnp.newaxis]
    else:
        J = 1.0

    dz_half_actual = dz_half * J if jacobian is not None else jnp.broadcast_to(
        dz_half[jnp.newaxis, :], (field_3d.shape[0], nlev - 1),
    )
    dz_actual = dz * J if jacobian is not None else jnp.broadcast_to(
        dz[jnp.newaxis, :], field_3d.shape,
    )

    # Flux at interfaces: coeff * (f[k] - f[k+1]) / dz_half
    dz_half_safe = jnp.maximum(dz_half_actual, 1e-10)
    flux_interface = coeff * (field_3d[:, :-1] - field_3d[:, 1:]) / dz_half_safe

    # Tendency: (flux[k-1/2] - flux[k+1/2]) / dz[k]
    zeros = jnp.zeros((field_3d.shape[0], 1), dtype=field_3d.dtype)
    flux_above = jnp.concatenate([zeros, flux_interface], axis=1)  # (n, nlev)
    flux_below = jnp.concatenate([flux_interface, zeros], axis=1)  # (n, nlev)

    dz_safe = jnp.maximum(dz_actual, 1e-10)
    return (flux_above - flux_below) / dz_safe
