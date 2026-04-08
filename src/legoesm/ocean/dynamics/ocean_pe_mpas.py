"""MPAS ocean baroclinic tendencies using TRiSK operators.

Boussinesq hydrostatic primitive equations in vector-invariant form
on MPAS Voronoi (C-grid) meshes. Uses the TRiSK discretization from
Ringler et al. (2010).

Equations (per layer k):
    du/dt = q_e * F_q - grad(KE + p'/ρ₀ + g·η) + A_h·del2(u) + A_v·d²u/dz²
    d(h·T)/dt = -div(h·u·T) + K_h·h·lap(T) + K_v·d²T/dz²
    d(h·S)/dt = -div(h·u·S) + K_h·h·lap(S) + K_v·d²S/dz²
    dη/dt = -Σ_k div(h_k · u_k)

References
----------
- Ringler, T. D., et al. (2010). J. Comput. Phys., 229(9), 3065-3090.
- Ringler, T. D., et al. (2013). Ocean Modelling, 69, 211-232.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import MPASOceanState, MPASOceanTendencies
from legoesm.core.operators_voronoi import (
    divergence_cell,
    gradient_edge,
    curl_vertex,
    kinetic_energy_cell,
    potential_vorticity_vertex,
    pv_flux_energy_conserving,
    pv_flux_enstrophy_conserving,
    edge_thickness,
    cell_to_edge_avg,
    vector_laplacian_del2,
)
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.eos import wright_eos, compute_hydrostatic_pressure, make_eos_fn
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
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
    nlev = z_coord.n_levels

    u_3d = state.u.data          # (nEdges, nlev)
    T_3d = state.T.data          # (nCells, nlev)
    S_3d = state.S.data          # (nCells, nlev)
    eta = state.eta.data         # (nCells,)
    H_bathy = state.H_bathy.data  # (nCells,)
    mask = state.land_mask.data  # (nCells,)

    c1 = mesh.cellsOnEdge[0]  # (nEdges,)
    c2 = mesh.cellsOnEdge[1]  # (nEdges,)

    # Helper: fill land cells with ocean-neighbor average (Neumann BC).
    # Works for both 1D (nCells,) and 2D (nCells, nlev) arrays.
    def _fill_land_cells_mpas(field_cell, mask_cell):
        nbr_sum = jnp.zeros_like(field_cell)
        nbr_cnt = jnp.zeros_like(field_cell)
        if field_cell.ndim == 1:
            nbr_sum = nbr_sum.at[c1].add(field_cell[c2] * mask_cell[c2])
            nbr_cnt = nbr_cnt.at[c1].add(mask_cell[c2])
            nbr_sum = nbr_sum.at[c2].add(field_cell[c1] * mask_cell[c1])
            nbr_cnt = nbr_cnt.at[c2].add(mask_cell[c1])
        else:
            m2 = mask_cell[c2, jnp.newaxis]
            m1 = mask_cell[c1, jnp.newaxis]
            nbr_sum = nbr_sum.at[c1].add(field_cell[c2] * m2)
            nbr_cnt = nbr_cnt.at[c1].add(m2)
            nbr_sum = nbr_sum.at[c2].add(field_cell[c1] * m1)
            nbr_cnt = nbr_cnt.at[c2].add(m1)
        nbr_avg = nbr_sum / jnp.maximum(nbr_cnt, 1.0)
        mask_e = mask_cell if field_cell.ndim == 1 else mask_cell[:, jnp.newaxis]
        return jnp.where(mask_e > 0.5, field_cell, nbr_avg)

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
    rho = eos_fn(T_filled, S_filled, jnp.zeros_like(T_3d))
    for _ in range(2):
        p_hydro = compute_hydrostatic_pressure(
            rho, eta, z_coord.dz_ref, jacobian, rho_0, g,
        )
        rho = eos_fn(T_filled, S_filled, p_hydro)
    # Final p_hydro for EOS only (not used in Bernoulli)
    p_hydro = compute_hydrostatic_pressure(
        rho, eta, z_coord.dz_ref, jacobian, rho_0, g,
    )  # (nCells, nlev)

    # Baroclinic pressure anomaly: built from rho' = rho - rho_0 only.
    # This excludes the rho_0*g*eta surface term, which is handled by
    # the barotropic solver's -g*grad(eta).  Using full p_hydro would
    # double-count the barotropic pressure gradient.
    rho_prime = rho - rho_0
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    dp_layer = rho_prime * g * dz_actual
    p_prime = jnp.cumsum(dp_layer, axis=-1) - dp_layer
    p_prime = p_prime + 0.5 * dp_layer  # (nCells, nlev)

    # Fill land cells in p_prime before gradient_edge so the 2-cell
    # stencil sees smooth values at coastlines.
    p_prime = _fill_land_cells_mpas(p_prime, mask)

    # ---- Edge mask for land boundaries ----
    edge_mask = mask[c1] * mask[c2]  # 1 only if both cells are ocean

    # ---- Per-level momentum and tracer tendencies ----
    # Use vmap over vertical levels
    def _level_tendencies(k):
        """Compute tendencies for a single level."""
        u_k = u_3d[:, k]       # (nEdges,)
        T_k = T_3d[:, k]       # (nCells,)
        S_k = S_3d[:, k]       # (nCells,)
        h_k_level = h_k[:, k]  # (nCells,)
        p_k = p_prime[:, k]    # (nCells,) — baroclinic anomaly only

        # Edge layer thickness
        h_e = edge_thickness(h_k_level, mesh)  # (nEdges,)

        # ---- Momentum tendency ----
        # Kinetic energy
        ke = kinetic_energy_cell(u_k, mesh)  # (nCells,)

        # Bernoulli function: KE + p'/rho_0  (baroclinic only;
        # the barotropic pressure gradient -g*grad(eta) is handled
        # by the barotropic substeps to avoid double-counting)
        bernoulli = ke + p_k / rho_0  # (nCells,)

        # Pressure gradient + Bernoulli
        grad_B = gradient_edge(bernoulli, mesh)  # (nEdges,)

        # PV flux (Coriolis + vorticity)
        q_v = potential_vorticity_vertex(u_k, h_k_level, mesh.fVertex, mesh)
        if config.pv_scheme == "energy":
            pv_flux = pv_flux_energy_conserving(u_k, h_k_level, q_v, mesh)
        else:
            pv_flux = pv_flux_enstrophy_conserving(u_k, h_k_level, q_v, mesh)

        # Horizontal viscosity
        visc = config.A_h * vector_laplacian_del2(u_k, mesh)  # (nEdges,)

        du_dt_k = -grad_B + pv_flux + visc
        du_dt_k = du_dt_k * edge_mask  # zero on land edges

        # ---- Thickness flux for continuity ----
        thickness_flux_k = u_k * h_e * edge_mask  # (nEdges,) no-flux BC at coast

        # Continuity: dh_k/dt = -div(u * h_e)
        div_flux = divergence_cell(thickness_flux_k, mesh)  # (nCells,)

        # ---- Tracer tendencies (flux form) ----
        # Edge tracer values (centered)
        T_e = cell_to_edge_avg(T_k, mesh)  # (nEdges,)
        S_e = cell_to_edge_avg(S_k, mesh)

        # Tracer flux: u * h_e * T_e
        T_flux = thickness_flux_k * T_e
        S_flux = thickness_flux_k * S_e

        # Divergence of tracer flux — this is the term that telescopes
        # exactly on the Voronoi mesh: sum(div_T_flux * area) = 0.
        div_T_flux = divergence_cell(T_flux, mesh)  # (nCells,)
        div_S_flux = divergence_cell(S_flux, mesh)

        # Advective-form tracer tendency (kept for backward compatibility
        # and diagnostics): h * dT/dt = -div(h*u*T) + T*div(h*u)
        h_safe = jnp.maximum(h_k_level, 1e-10)
        dT_dt_k = (-div_T_flux + T_k * div_flux) / h_safe
        dS_dt_k = (-div_S_flux + S_k * div_flux) / h_safe

        # Non-advective source terms (horizontal diffusion).
        # These are separated from the advective flux for the flux-form
        # tracer update in ocean_model_mpas.py.
        grad_T = gradient_edge(T_k, mesh) * edge_mask
        dT_dt_src_k = config.K_h * divergence_cell(grad_T, mesh) / h_safe * h_k_level
        grad_S = gradient_edge(S_k, mesh) * edge_mask
        dS_dt_src_k = config.K_h * divergence_cell(grad_S, mesh) / h_safe * h_k_level

        # Add source to total tendency (backward compat)
        dT_dt_k = dT_dt_k + dT_dt_src_k
        dS_dt_k = dS_dt_k + dS_dt_src_k

        # Mask land cells
        dT_dt_k = dT_dt_k * mask
        dS_dt_k = dS_dt_k * mask
        div_T_flux = div_T_flux * mask
        div_S_flux = div_S_flux * mask
        dT_dt_src_k = dT_dt_src_k * mask
        dS_dt_src_k = dS_dt_src_k * mask

        return (du_dt_k, dT_dt_k, dS_dt_k, div_flux,
                div_T_flux, div_S_flux, dT_dt_src_k, dS_dt_src_k)

    # Vectorize over levels using scan for efficiency
    def _scan_fn(carry, k):
        (du, dT, dS, div_f,
         div_hut, div_hus, dT_src, dS_src) = _level_tendencies(k)
        return carry, (du, dT, dS, div_f,
                       div_hut, div_hus, dT_src, dS_src)

    _, (du_dt_all, dT_dt_all, dS_dt_all, div_flux_all,
        div_hut_all, div_hus_all,
        dT_src_all, dS_src_all) = jax.lax.scan(
        _scan_fn, None, jnp.arange(nlev),
    )
    # scan outputs: (nlev, nEdges), (nlev, nCells), etc.
    du_dt_3d = du_dt_all.T       # (nEdges, nlev)
    dT_dt_3d = dT_dt_all.T       # (nCells, nlev)
    dS_dt_3d = dS_dt_all.T       # (nCells, nlev)
    div_hut_3d = div_hut_all.T   # (nCells, nlev)
    div_hus_3d = div_hus_all.T   # (nCells, nlev)
    dT_dt_src_3d = dT_src_all.T  # (nCells, nlev)
    dS_dt_src_3d = dS_src_all.T  # (nCells, nlev)

    # ---- Vertical mixing ----
    dz_half = z_coord.dz_half_ref  # (nlev-1,)
    dz = z_coord.dz_ref  # (nlev,)

    # Vertical viscosity: d/dz(A_v * du/dz) at each edge
    du_dt_3d = du_dt_3d + _vertical_diffusion(
        u_3d, dz_half, dz, jacobian=jacobian, coeff=config.A_v, is_edge=True,
        mesh=mesh,
    )

    # Vertical tracer diffusion (conservative per-column; source term)
    vdiff_T = _vertical_diffusion(
        T_3d, dz_half, dz, jacobian=jacobian, coeff=config.K_v, is_edge=False,
        mesh=mesh,
    ) * mask[:, jnp.newaxis]
    vdiff_S = _vertical_diffusion(
        S_3d, dz_half, dz, jacobian=jacobian, coeff=config.K_v, is_edge=False,
        mesh=mesh,
    ) * mask[:, jnp.newaxis]
    dT_dt_3d = dT_dt_3d + vdiff_T
    dS_dt_3d = dS_dt_3d + vdiff_S
    dT_dt_src_3d = dT_dt_src_3d + vdiff_T
    dS_dt_src_3d = dS_dt_src_3d + vdiff_S

    # ---- Physics (surface forcing, bottom drag, etc.) ----
    if physics_fn is not None:
        phys = physics_fn(state, mesh, z_coord, surface_forcing)
        du_dt_3d = du_dt_3d + phys.du_dt.data
        dT_dt_3d = dT_dt_3d + phys.dT_dt.data
        dS_dt_3d = dS_dt_3d + phys.dS_dt.data
        dT_dt_src_3d = dT_dt_src_3d + phys.dT_dt.data
        dS_dt_src_3d = dS_dt_src_3d + phys.dS_dt.data

    # ---- Free surface tendency ----
    # deta/dt = -sum_k div(u_k * h_e_k)
    deta_dt = -jnp.sum(div_flux_all.T, axis=1) * mask  # (nCells,)

    # ---- Freshwater forcing ----
    if freshwater is not None and config.freshwater_closure != "none":
        # Free-surface mass flux: deta/dt += F_fw / rho_0
        deta_dt = deta_dt + freshwater_eta_tendency(freshwater, config.rho_0) * mask

        # Virtual salt flux: dS/dt = -S_ref * F_fw / (rho_0 * dz_0)
        dz_0 = h_k[:, 0]  # top layer thickness (nCells,)
        dS_fw = virtual_salt_flux(freshwater, config.S_ref, dz_0, config.rho_0)
        dS_dt_3d = dS_dt_3d.at[:, 0].add(dS_fw * mask)
        dS_dt_src_3d = dS_dt_src_3d.at[:, 0].add(dS_fw * mask)

    return MPASOceanTendencies(
        du_dt=Field(data=du_dt_3d, name="du_dt",
                    dims=("nEdges", "nlev"), units="m/s²"),
        dT_dt=Field(data=dT_dt_3d, name="dT_dt",
                    dims=("nCells", "nlev"), units="degC/s"),
        dS_dt=Field(data=dS_dt_3d, name="dS_dt",
                    dims=("nCells", "nlev"), units="PSU/s"),
        deta_dt=Field(data=deta_dt, name="deta_dt",
                      dims=("nCells",), units="m/s"),
        div_hut=Field(data=div_hut_3d, name="div_hut",
                      dims=("nCells", "nlev"), units="degC*m/s"),
        div_hus=Field(data=div_hus_3d, name="div_hus",
                      dims=("nCells", "nlev"), units="PSU*m/s"),
        dT_dt_source=Field(data=dT_dt_src_3d, name="dT_dt_source",
                           dims=("nCells", "nlev"), units="degC/s"),
        dS_dt_source=Field(data=dS_dt_src_3d, name="dS_dt_source",
                           dims=("nCells", "nlev"), units="PSU/s"),
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
