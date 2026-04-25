"""Boussinesq Hydrostatic Primitive Equations on the lat-lon C-grid (FV).

C-grid staggering:
  u at lon interfaces  (n_lat, n_lon+1, nlev)
  v at lat interfaces  (n_lat+1, n_lon, nlev)
  eta, T, S at cell centers (n_lat, n_lon [, nlev])

Key advantages over the A-grid formulation (ocean_pe_latlon.py):
- Pressure gradient uses compact 1-cell stencil -> no 2*dx null space
- Divergence sums actual face fluxes -> no checkerboard mode
- Coriolis coupling is on face-averaged velocities (forward-backward
  Matsuno step in the step function, not in this tendency)

Boundary conditions:
- Longitude: periodic (u wraps at j=0 and j=n_lon)
- Latitude: solid wall at poles (v=0 at i=0 and i=n_lat)

Vector-invariant momentum advection (issue #160, fixed)
-------------------------------------------------------
The nonlinear momentum advection uses TOTAL velocity throughout:
  KE = 0.5*(u² + v²)   (total velocity, section 6)
  q  = ζ(u_total) / h  (potential vorticity at vertices, section 7b)
  PV flux = q̄ · F      (Sadourny EC: PV times mass flux)
Planetary Coriolis (f × u) is handled separately in the step function
via forward-backward (Matsuno) stepping. The depth-mean of the full
nonlinear tendency enters the barotropic solver as F_slow; the
perturbation (full − depth-mean) is applied to 3D velocity. This is
the MOM6-style slow-forcing split that closes issue #160.

References
----------
- Griffies (2004): Fundamentals of Ocean Climate Models (MOM framework)
- Arakawa & Lamb (1977): Computational Design of the Basic Dynamical
  Processes of the UCLA GCM
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.eos import make_eos_fn
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
)
from legoesm.ocean.state import (
    LatLonCGridOceanState,
    LatLonCGridOceanTendencies,
    LatLonCGridOceanConfig,
)
from legoesm.ocean.dynamics.ocean_tendency_common import (
    apply_sponge_tracer_relaxation,
    iterate_eos_and_pressure_anomaly,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    biharmonic_scaling_factor,
    divergence_cgrid,
    gradient_x_cgrid,
    gradient_y_cgrid,
    bilaplacian_cgrid,
    laplacian_cgrid,
    vector_bilaplacian_cgrid,
    vector_laplacian_cgrid,
    interp_cell_to_uface,
    curl_vertex_cgrid,
    smagorinsky_biharmonic_tendency_cgrid,
    leith_biharmonic_tendency_cgrid,
)
from legoesm.ocean.vertical import (
    diagnose_w_from_flux_div as _diagnose_w_from_flux_div,
    vertical_advection_ocean as _vertical_advection_ocean,
    flux_form_vertical_momentum_advection as _flux_form_vertical_momentum_advection,
    flux_form_vertical_tracer_advection_tvd as _flux_form_vertical_advection_tvd,
)


# interp_cell_to_uface is imported from latlon_cgrid_operators (shared).


def _interp_to_v_points(f: jnp.ndarray) -> jnp.ndarray:
    """Interpolate cell-center field to v-points (lat interfaces).

    Parameters
    ----------
    f : array, shape (n_lat, n_lon, ...) at cell centers.

    Returns
    -------
    f_v : array, shape (n_lat+1, n_lon, ...) at v-points.
    """
    f_interior = 0.5 * (f[:-1] + f[1:])  # (n_lat-1, n_lon, ...)
    if f.ndim >= 3:
        zero = jnp.zeros((1, f.shape[1], f.shape[2]), dtype=f.dtype)
    else:
        zero = jnp.zeros((1, f.shape[1]), dtype=f.dtype)
    return jnp.concatenate([zero, f_interior, zero], axis=0)


def _van_leer_limiter(r: jnp.ndarray) -> jnp.ndarray:
    """Van Leer flux limiter: phi(r) = (r + |r|) / (1 + |r|). Differentiable, TVD."""
    return (r + jnp.abs(r)) / (1.0 + jnp.abs(r))


def _tvd_to_u_points(f: jnp.ndarray, mass_flux_u: jnp.ndarray) -> jnp.ndarray:
    """Van Leer TVD interpolation to u-points. Second-order, monotonic (#170)."""
    eps = 1e-30
    f_left = jnp.roll(f, 1, axis=1)
    f_right = f
    f_left2 = jnp.roll(f, 2, axis=1)
    f_right2 = jnp.roll(f, -1, axis=1)
    delta_pos = f_right - f_left
    r_pos = (f_left - f_left2) / jnp.where(jnp.abs(delta_pos) > eps, delta_pos, eps)
    delta_neg = f_left - f_right
    r_neg = (f_right2 - f_right) / jnp.where(jnp.abs(delta_neg) > eps, delta_neg, eps)
    f_pos = f_left + 0.5 * _van_leer_limiter(r_pos) * delta_pos
    f_neg = f_right + 0.5 * _van_leer_limiter(r_neg) * delta_neg
    n_lon = f.shape[1]
    mf = mass_flux_u[:, :n_lon]
    f_tvd = jnp.where(mf > 0, f_pos, f_neg)
    if f.ndim >= 3:
        return jnp.concatenate([f_tvd, f_tvd[:, 0:1, :]], axis=1)
    return jnp.concatenate([f_tvd, f_tvd[:, 0:1]], axis=1)


def _tvd_to_v_points(f: jnp.ndarray, mass_flux_v: jnp.ndarray) -> jnp.ndarray:
    """Van Leer TVD interpolation to v-points. Solid wall at poles (#170)."""
    eps = 1e-30
    f_south = f[:-1]; f_north = f[1:]
    f_south2 = jnp.concatenate([f[:1], f[:-2]], axis=0)
    f_north2 = jnp.concatenate([f[2:], f[-1:]], axis=0)
    delta_pos = f_north - f_south
    r_pos = (f_south - f_south2) / jnp.where(jnp.abs(delta_pos) > eps, delta_pos, eps)
    delta_neg = f_south - f_north
    r_neg = (f_north2 - f_north) / jnp.where(jnp.abs(delta_neg) > eps, delta_neg, eps)
    f_pos = f_south + 0.5 * _van_leer_limiter(r_pos) * delta_pos
    f_neg = f_north + 0.5 * _van_leer_limiter(r_neg) * delta_neg
    f_tvd = jnp.where(mass_flux_v[1:-1] > 0, f_pos, f_neg)
    if f.ndim >= 3:
        zero = jnp.zeros((1, f.shape[1], f.shape[2]), dtype=f.dtype)
    else:
        zero = jnp.zeros((1, f.shape[1]), dtype=f.dtype)
    return jnp.concatenate([zero, f_tvd, zero], axis=0)


def _upwind_to_u_points(
    f: jnp.ndarray,
    mass_flux_u: jnp.ndarray,
) -> jnp.ndarray:
    """First-order upwind interpolation of cell-center field to u-points.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon, ...) at cell centers.
    mass_flux_u : array, shape (n_lat, n_lon+1, ...) at u-points.
        Sign convention: positive = flow in +j (eastward) direction.

    Returns
    -------
    f_u : array, shape (n_lat, n_lon+1, ...) at u-points.
        Upwind value: uses the upstream cell based on mass_flux_u sign.
    """
    # Face j is between cell (j-1) mod n_lon and cell j.
    # Positive flux => flow from cell j-1 to cell j => upwind is cell j-1.
    # Negative flux => flow from cell j to cell j-1 => upwind is cell j.
    f_left = jnp.roll(f, 1, axis=1)   # f_left[:, j] = f[:, j-1]
    f_right = f                        # f_right[:, j] = f[:, j]

    # Build upwind at interior faces (n_lat, n_lon)
    f_upwind = jnp.where(mass_flux_u[:, :-1] > 0, f_left, f_right)

    # Wrap: face n_lon is the same as face 0 (periodic in longitude)
    if f.ndim >= 3:
        return jnp.concatenate([f_upwind, f_upwind[:, 0:1, :]], axis=1)
    else:
        return jnp.concatenate([f_upwind, f_upwind[:, 0:1]], axis=1)


def _upwind_to_v_points(
    f: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
) -> jnp.ndarray:
    """First-order upwind interpolation of cell-center field to v-points.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon, ...) at cell centers.
    mass_flux_v : array, shape (n_lat+1, n_lon, ...) at v-points.
        Sign convention: positive = flow in +i (northward) direction.

    Returns
    -------
    f_v : array, shape (n_lat+1, n_lon, ...) at v-points.
        Upwind value: uses the upstream cell based on mass_flux_v sign.
        Boundary faces (i=0 and i=n_lat) are zero (solid wall).
    """
    # Interior face i (for i=1..n_lat-1) sits between cell i-1 and cell i.
    # Positive flux => flow from cell i-1 to cell i => upwind is cell i-1.
    # Negative flux => flow from cell i to cell i-1 => upwind is cell i.
    f_south = f[:-1]   # cell i-1 for interior faces
    f_north = f[1:]    # cell i   for interior faces
    # Interior mass flux: faces 1..n_lat-1
    mf_interior = mass_flux_v[1:-1]
    f_upwind = jnp.where(mf_interior > 0, f_south, f_north)

    if f.ndim >= 3:
        zero = jnp.zeros((1, f.shape[1], f.shape[2]), dtype=f.dtype)
    else:
        zero = jnp.zeros((1, f.shape[1]), dtype=f.dtype)
    return jnp.concatenate([zero, f_upwind, zero], axis=0)


def _neumann_fill_cgrid(
    f: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Fill land cells with nearest ocean-neighbor (Neumann BC).

    Same algorithm as latlon_operators.neumann_fill_latlon.
    """
    m = mask
    filled = f
    for _ in range(3):
        f_s = jnp.concatenate([filled[0:1], filled[:-1]], axis=0)
        m_s = jnp.concatenate([m[0:1], m[:-1]], axis=0)
        f_n = jnp.concatenate([filled[1:], filled[-1:]], axis=0)
        m_n = jnp.concatenate([m[1:], m[-1:]], axis=0)
        f_w = jnp.roll(filled, 1, axis=1)
        m_w = jnp.roll(m, 1, axis=1)
        f_e = jnp.roll(filled, -1, axis=1)
        m_e = jnp.roll(m, -1, axis=1)

        is_land = m < 0.5

        if f.ndim > 2:
            m_s_e = m_s[..., jnp.newaxis]
            m_n_e = m_n[..., jnp.newaxis]
            m_w_e = m_w[..., jnp.newaxis]
            m_e_e = m_e[..., jnp.newaxis]
            is_land_e = is_land[..., jnp.newaxis]
        else:
            m_s_e = m_s
            m_n_e = m_n
            m_w_e = m_w
            m_e_e = m_e
            is_land_e = is_land

        nbr_sum = f_s * m_s_e + f_n * m_n_e + f_w * m_w_e + f_e * m_e_e
        nbr_count = m_s_e + m_n_e + m_w_e + m_e_e
        nbr_avg = nbr_sum / jnp.maximum(nbr_count, 1.0)

        has_any_nbr = (m_s + m_n + m_w + m_e) > 0.0
        if f.ndim > 2:
            has_any_nbr_e = has_any_nbr[..., jnp.newaxis]
        else:
            has_any_nbr_e = has_any_nbr

        filled = jnp.where(is_land_e & has_any_nbr_e, nbr_avg, filled)
        m = jnp.where(is_land & has_any_nbr, 1.0, m)

    return filled


# =============================================================================
# WENO momentum advection helpers (Phase 2b, Silvestri et al. 2024)
# =============================================================================

def _weno_zeta_at_u(
    phi: jnp.ndarray,
    v_smooth: jnp.ndarray,
    v_at_u: jnp.ndarray,
    order: int = 5,
) -> jnp.ndarray:
    """WENO reconstruction of a vertex field to u-faces (meridional).

    Uses the {φ; v} smoothness-optimised stencil (Silvestri et al. 2024):
    smoothness indicators computed from v (smoother velocity field),
    reconstruction applied to φ (e.g. potential vorticity q = ζ/h).

    Parameters
    ----------
    phi : (n_lat+1, n_lon+1, nlev) field at vertices to reconstruct.
    v_smooth : (n_lat+1, n_lon, nlev) at v-faces (smoothness field).
    v_at_u : (n_lat, n_lon+1, nlev) at u-faces (upwinding velocity).
    order : {5, 7}

    Returns
    -------
    phi_at_u : (n_lat, n_lon+1, nlev)
    """
    from legoesm.core.weno import weno_reconstruct_split, weno_upwind

    hw = {5: 3, 7: 4}[order]
    n_lat = phi.shape[0] - 1  # n_lat+1 vertices → n_lat u-faces

    # Interpolate v to vertex longitudes.
    # v: (n_lat+1, n_lon) at cell-center lons → vertex: (n_lat+1, n_lon+1) at interface lons
    v_w = jnp.roll(v_smooth, 1, axis=1)  # v[:, (j-1) % n_lon, :]
    v_at_vtx = 0.5 * (v_w + v_smooth)    # (n_lat+1, n_lon, nlev)
    v_at_vtx = jnp.concatenate(
        [v_at_vtx, v_at_vtx[:, 0:1, :]], axis=1)  # (n_lat+1, n_lon+1, nlev)

    # Ghost cells (Neumann BC) along axis 0 for the meridional stencil
    phi_ext = jnp.concatenate(
        [phi[:1, :, :]] * hw + [phi] + [phi[-1:, :, :]] * hw, axis=0)
    v_ext = jnp.concatenate(
        [v_at_vtx[:1, :, :]] * hw + [v_at_vtx] + [v_at_vtx[-1:, :, :]] * hw,
        axis=0)

    # Build stencil for all n_lat u-faces simultaneously.
    # Face i (i=0..n_lat-1) is between vertex i and vertex i+1.
    # WENO at I+1/2 where I=i: needs vertices i-hw+1 .. i+hw.
    # In ext: indices (i-hw+1)+hw .. (i+hw)+hw = i+1 .. i+2*hw.
    phi_stencil = [phi_ext[1 + j: n_lat + 1 + j, :, :]
                   for j in range(2 * hw)]
    psi_stencil = [v_ext[1 + j: n_lat + 1 + j, :, :]
                   for j in range(2 * hw)]

    phi_plus, phi_minus = weno_reconstruct_split(
        phi_stencil, psi_stencil, order=order)

    # Upwind: v > 0 ⟹ flow from south → use left-biased (f_plus)
    return weno_upwind(phi_plus, phi_minus, v_at_u)


def _weno_zeta_at_v(
    phi: jnp.ndarray,
    u_smooth: jnp.ndarray,
    u_at_v: jnp.ndarray,
    order: int = 5,
) -> jnp.ndarray:
    """WENO reconstruction of a vertex field to v-faces (zonal).

    Uses the {φ; u} smoothness-optimised stencil.
    Periodic in longitude.

    Parameters
    ----------
    phi : (n_lat+1, n_lon+1, nlev) field at vertices to reconstruct.
    u_smooth : (n_lat, n_lon+1, nlev) at u-faces (smoothness field).
    u_at_v : (n_lat+1, n_lon, nlev) at v-faces (upwinding velocity).
    order : {5, 7}

    Returns
    -------
    phi_at_v : (n_lat+1, n_lon, nlev)
    """
    from legoesm.core.weno import weno_reconstruct_split, weno_upwind

    hw = {5: 3, 7: 4}[order]
    n_lon = phi.shape[1] - 1   # n_lon+1 vertices → n_lon v-face longitudes
    nlev = phi.shape[2]

    # Interpolate u to vertex latitudes.
    # u: (n_lat, n_lon+1) at cell-center lats → vertex: (n_lat+1, n_lon+1) at interface lats
    n_lon_u = u_smooth.shape[1]  # n_lon+1
    zero_u = jnp.zeros((1, n_lon_u, nlev), dtype=u_smooth.dtype)
    u_ext_lat = jnp.concatenate(
        [zero_u, u_smooth, zero_u], axis=0)    # (n_lat+2, n_lon+1, nlev)
    u_at_vtx = 0.5 * (u_ext_lat[:-1, :, :] +
                       u_ext_lat[1:, :, :])   # (n_lat+1, n_lon+1, nlev)

    # Both φ and u_at_vtx are (n_lat+1, n_lon+1, nlev).
    # Reconstruct along axis 1 (longitude), periodic.
    # Use the first n_lon columns (column n_lon == column 0).
    phi_core = phi[:, :n_lon, :]      # (n_lat+1, n_lon, nlev)
    u_core = u_at_vtx[:, :n_lon, :]   # (n_lat+1, n_lon, nlev)

    # Face j (j=0..n_lon-1) between vertex j and vertex j+1.
    # WENO at I+1/2 where I=j: needs vertices j-hw+1 .. j+hw.
    # Roll offsets hw-1, hw-2, ..., -(hw) place vertex j-hw+1 .. j+hw
    # at position j in the rolled array.
    phi_stencil = [jnp.roll(phi_core, hw - 1 - j, axis=1)
                   for j in range(2 * hw)]
    psi_stencil = [jnp.roll(u_core, hw - 1 - j, axis=1)
                   for j in range(2 * hw)]

    phi_plus, phi_minus = weno_reconstruct_split(
        phi_stencil, psi_stencil, order=order)

    # Upwind: u > 0 ⟹ flow from west → use left-biased (f_plus)
    return weno_upwind(phi_plus, phi_minus, u_at_v)


def _flux_form_vertical_momentum_advection_weno(
    u: jnp.ndarray,
    w_half: jnp.ndarray,
    h_u: jnp.ndarray,
    order: int = 5,
) -> jnp.ndarray:
    """WENO vertical momentum advection as a per-thickness tendency.

    Same interface as ``flux_form_vertical_momentum_advection`` from
    ``vertical.py``, but uses WENO-Z reconstruction at vertical
    interfaces instead of first-order upwind.

    WARNING: WENO removes the implicit viscosity (~|w|*dz/2) that
    first-order upwind provides.  This may require compensating
    vertical viscosity (e.g. KPP, Richardson-dependent A_v).
    See issue #204.

    Parameters
    ----------
    u : (..., nlev)  velocity at momentum points.
    w_half : (..., nlev+1)  vertical velocity on interfaces.
    h_u : (..., nlev)  layer thickness at momentum points.
    order : {5, 7}

    Returns
    -------
    tendency : (..., nlev)
        ``-(F_top - F_bot) / h_u``
    """
    if order == 5:
        from legoesm.ocean.advection import (
            flux_form_vertical_tracer_advection_weno5,
        )
        vert_flux_div = flux_form_vertical_tracer_advection_weno5(
            u, w_half, h_u, dt=0.0)
    elif order == 7:
        from legoesm.ocean.advection import (
            flux_form_vertical_tracer_advection_weno7,
        )
        vert_flux_div = flux_form_vertical_tracer_advection_weno7(
            u, w_half, h_u, dt=0.0)
    else:
        raise ValueError(
            f"Unsupported WENO order {order} for vertical momentum")
    h_u_safe = jnp.maximum(h_u, 1.0e-10)
    return -vert_flux_div / h_u_safe


# =============================================================================
# WENO D-term (divergence flux) and K-term (KE) helpers — Phase 4b
# =============================================================================

def _weno_cell_to_uface(
    phi: jnp.ndarray,
    psi: jnp.ndarray,
    u_upwind: jnp.ndarray,
    order: int = 5,
) -> jnp.ndarray:
    """WENO reconstruction of a cell-center field to u-faces (zonal, periodic).

    Reconstructs *phi* from cell centers (n_lat, n_lon, nlev) to u-faces
    (n_lat, n_lon+1, nlev) using the {phi; psi} smoothness-optimised stencil.

    Used for the D term: divergence at cells → divergence at u-faces.

    Parameters
    ----------
    phi : (n_lat, n_lon, nlev)  field at cell centers to reconstruct.
    psi : (n_lat, n_lon, nlev)  field at cell centers for smoothness.
    u_upwind : (n_lat, n_lon+1, nlev)  velocity at u-faces (upwind sign).
    order : {5, 7}

    Returns
    -------
    phi_at_u : (n_lat, n_lon+1, nlev)
    """
    from legoesm.core.weno import weno_reconstruct_split, weno_upwind

    hw = {5: 3, 7: 4}[order]
    n_lon = phi.shape[1]

    # Periodic stencil along axis 1 (longitude).
    # U-face j is between cell j-1 and cell j.  WENO at the face between
    # cells (j-1) and j needs cells j-hw, ..., j+hw-1.
    # Roll offset for stencil position s: hw - s places cell j-hw+s at
    # position j.
    phi_stencil = [jnp.roll(phi, hw - s, axis=1)
                   for s in range(2 * hw)]
    psi_stencil = [jnp.roll(psi, hw - s, axis=1)
                   for s in range(2 * hw)]

    phi_plus, phi_minus = weno_reconstruct_split(
        phi_stencil, psi_stencil, order=order)

    # Upwind selection: u > 0 => flow from west => left-biased (f_plus)
    # n_lon output values (one per periodic u-face), then append wrap.
    phi_at_u_core = weno_upwind(
        phi_plus, phi_minus, u_upwind[:, :n_lon, :])
    return jnp.concatenate(
        [phi_at_u_core, phi_at_u_core[:, 0:1, :]], axis=1)


def _weno_cell_to_vface(
    phi: jnp.ndarray,
    psi: jnp.ndarray,
    v_upwind: jnp.ndarray,
    order: int = 5,
) -> jnp.ndarray:
    """WENO reconstruction of a cell-center field to v-faces (meridional, wall BC).

    Reconstructs *phi* from cell centers (n_lat, n_lon, nlev) to v-faces
    (n_lat+1, n_lon, nlev) using the {phi; psi} smoothness-optimised stencil.
    Boundary v-faces (south/north poles) are set to zero (wall BC: v=0).

    Used for the D term: divergence at cells → divergence at v-faces.

    Parameters
    ----------
    phi : (n_lat, n_lon, nlev)  field at cell centers to reconstruct.
    psi : (n_lat, n_lon, nlev)  field at cell centers for smoothness.
    v_upwind : (n_lat+1, n_lon, nlev)  velocity at v-faces (upwind sign).
    order : {5, 7}

    Returns
    -------
    phi_at_v : (n_lat+1, n_lon, nlev)
    """
    from legoesm.core.weno import weno_reconstruct_split, weno_upwind

    hw = {5: 3, 7: 4}[order]
    n_lat = phi.shape[0]
    nlev = phi.shape[2]
    n_lon = phi.shape[1]

    # Ghost cells (Neumann BC) along axis 0 for meridional stencil.
    phi_ext = jnp.concatenate(
        [phi[:1, :, :]] * hw + [phi] + [phi[-1:, :, :]] * hw, axis=0)
    psi_ext = jnp.concatenate(
        [psi[:1, :, :]] * hw + [psi] + [psi[-1:, :, :]] * hw, axis=0)

    # V-face i (i=1,...,n_lat-1) sits between cell i-1 and cell i.
    # WENO needs cells i-hw, ..., i+hw-1.
    # In extended array: cell c maps to ext[hw+c], so cell i-hw maps
    # to ext[i].  For all n_lat-1 interior faces (i=1,...,n_lat-1),
    # output index m = i-1, so stencil[s][m] = ext[m+1+s].
    phi_stencil = [phi_ext[1 + s: n_lat + s, :, :]
                   for s in range(2 * hw)]
    psi_stencil = [psi_ext[1 + s: n_lat + s, :, :]
                   for s in range(2 * hw)]

    phi_plus, phi_minus = weno_reconstruct_split(
        phi_stencil, psi_stencil, order=order)

    # Upwind: v > 0 => flow from south => left-biased (f_plus).
    # Interior faces only (n_lat-1 values).
    phi_at_v_interior = weno_upwind(
        phi_plus, phi_minus, v_upwind[1:-1, :, :])

    # Boundary v-faces = 0 (wall BC: v=0 at poles, so D*v=0 regardless).
    zero = jnp.zeros((1, n_lon, nlev), dtype=phi.dtype)
    return jnp.concatenate([zero, phi_at_v_interior, zero], axis=0)


def _weno_usq_to_cell(
    u: jnp.ndarray,
    order: int = 5,
) -> jnp.ndarray:
    """WENO reconstruction of u² from u-faces to cell centers (zonal, periodic).

    Uses {u²; u} smoothness-optimised stencil (Silvestri et al. 2024):
    smoothness indicators from velocity u, reconstruction of u².
    Replaces the centered ``(0.5*(u[j]+u[j+1]))²`` with WENO upwind
    ``avg(u²)`` for shock-capturing KE dissipation.

    Parameters
    ----------
    u : (n_lat, n_lon+1, nlev) velocity at u-faces.
    order : {5, 7}

    Returns
    -------
    u_sq_cell : (n_lat, n_lon, nlev)  u² reconstructed to cell centers.
    """
    from legoesm.core.weno import weno_reconstruct_split, weno_upwind

    hw = {5: 3, 7: 4}[order]
    n_lon = u.shape[1] - 1  # n_lon+1 u-faces => n_lon cells

    u_sq = u ** 2

    # Cell center j is between u-face j and u-face j+1.
    # Standard WENO at face j+1/2 uses cells j-hw+1, ..., j+hw.
    # Use n_lon core u-face values (periodic: face n_lon == face 0).
    u_sq_core = u_sq[:, :n_lon, :]
    u_core = u[:, :n_lon, :]

    # Roll offset hw-1-s: same pattern as _weno_zeta_at_v.
    phi_stencil = [jnp.roll(u_sq_core, hw - 1 - s, axis=1)
                   for s in range(2 * hw)]
    psi_stencil = [jnp.roll(u_core, hw - 1 - s, axis=1)
                   for s in range(2 * hw)]

    phi_plus, phi_minus = weno_reconstruct_split(
        phi_stencil, psi_stencil, order=order)

    # Upwind velocity at cell centers.
    u_at_cell = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
    return weno_upwind(phi_plus, phi_minus, u_at_cell)


def _weno_vsq_to_cell(
    v: jnp.ndarray,
    order: int = 5,
) -> jnp.ndarray:
    """WENO reconstruction of v² from v-faces to cell centers (meridional, wall BC).

    Uses {v²; v} smoothness-optimised stencil (Silvestri et al. 2024):
    smoothness indicators from velocity v, reconstruction of v².
    Replaces the centered ``(0.5*(v[i]+v[i+1]))²`` with WENO upwind
    ``avg(v²)`` for shock-capturing KE dissipation.

    Parameters
    ----------
    v : (n_lat+1, n_lon, nlev) velocity at v-faces.
    order : {5, 7}

    Returns
    -------
    v_sq_cell : (n_lat, n_lon, nlev)  v² reconstructed to cell centers.
    """
    from legoesm.core.weno import weno_reconstruct_split, weno_upwind

    hw = {5: 3, 7: 4}[order]
    n_lat = v.shape[0] - 1  # n_lat+1 v-faces => n_lat cells

    v_sq = v ** 2

    # Ghost cells (Neumann BC) along axis 0.
    v_sq_ext = jnp.concatenate(
        [v_sq[:1, :, :]] * hw + [v_sq] + [v_sq[-1:, :, :]] * hw, axis=0)
    v_ext = jnp.concatenate(
        [v[:1, :, :]] * hw + [v] + [v[-1:, :, :]] * hw, axis=0)

    # Cell center i is between v-face i and v-face i+1.
    # Standard WENO at face i+1/2 uses cells i-hw+1, ..., i+hw.
    # In ext: index hw+face => positions 1+s, ..., n_lat+s.
    # Same pattern as _weno_zeta_at_u.
    phi_stencil = [v_sq_ext[1 + s: n_lat + 1 + s, :, :]
                   for s in range(2 * hw)]
    psi_stencil = [v_ext[1 + s: n_lat + 1 + s, :, :]
                   for s in range(2 * hw)]

    phi_plus, phi_minus = weno_reconstruct_split(
        phi_stencil, psi_stencil, order=order)

    # Upwind velocity at cell centers.
    v_at_cell = 0.5 * (v[:-1, :, :] + v[1:, :, :])
    return weno_upwind(phi_plus, phi_minus, v_at_cell)


def latlon_cgrid_ocean_baroclinic_tendencies(
    state: LatLonCGridOceanState,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonCGridOceanConfig = LatLonCGridOceanConfig(),
    physics_fn=None,
    surface_forcing=None,
    sponge=None,
    dt: float = 300.0,
) -> LatLonCGridOceanTendencies:
    """Compute 3D baroclinic tendencies on a C-grid lat-lon grid.

    Parameters
    ----------
    state : LatLonCGridOceanState
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    config : LatLonCGridOceanConfig
    physics_fn : callable, optional
    surface_forcing : optional
    sponge : SpongeForcing, optional
        Sponge layer relaxation fields (gamma, T_ref, S_ref, u_ref, v_ref).

    Returns
    -------
    LatLonCGridOceanTendencies
    """
    u = state.u.data       # (n_lat, n_lon+1, nlev)
    v = state.v.data       # (n_lat+1, n_lon, nlev)
    T = state.T.data       # (n_lat, n_lon, nlev)
    S = state.S.data
    eta = state.eta.data   # (n_lat, n_lon)
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data
    u_mask = state.u_mask.data
    v_mask = state.v_mask.data
    mask_3d = mask[..., jnp.newaxis]
    u_mask_3d = u_mask[..., jnp.newaxis]
    v_mask_3d = v_mask[..., jnp.newaxis]

    g_val = config.g
    rho_0 = config.rho_0
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta.dtype)
    eta_floor = min_water_col - H_bathy
    eta_safe = jnp.maximum(eta, eta_floor) * mask

    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = z_coord.n_levels

    # --- 1. Layer thickness and Jacobian ---
    J = compute_ocean_jacobian(
        eta_safe, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )
    h_k = compute_layer_thickness(
        eta_safe, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )

    # --- 2. Density from EOS + 3. Baroclinic pressure anomaly ---
    # Reference Jacobian (J=1, eta=0): the barotropic solver handles
    # the free-surface gradient g*grad(eta) and using actual J here
    # would double-count it (would also create a spatially-varying
    # pressure even for uniform T/S).
    eos_fn = make_eos_fn(config.eos, getattr(config, 'eos_linear', None))
    rho, rho_prime, p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask,
        lambda field: _neumann_fill_cgrid(field, mask),
        eos_fn, z_coord.dz_ref, rho_0, g_val,
        n_iter=2,
    )

    p_prime_filled = _neumann_fill_cgrid(p_prime, mask)

    # C-grid gradient: compact stencil at face points
    # vmap over levels for 3D gradient
    p_t = jnp.moveaxis(p_prime_filled, -1, 0)  # (nlev, n_lat, n_lon)

    dp_dx_t = jax.vmap(lambda p2d: gradient_x_cgrid(p2d, grid))(p_t)
    dp_dy_t = jax.vmap(lambda p2d: gradient_y_cgrid(p2d, grid))(p_t)

    dp_dx = jnp.moveaxis(dp_dx_t, 0, -1)  # (n_lat, n_lon+1, nlev)
    dp_dy = jnp.moveaxis(dp_dy_t, 0, -1)  # (n_lat+1, n_lon, nlev)

    # --- 4. Vertical velocity from FV flux divergence ---
    # Divergence needs face fluxes: h*u at u-points, h*v at v-points.
    # Uses FULL velocity (barotropic + baroclinic) for mass transport.
    h_u = interp_cell_to_uface(h_k)
    h_v = _interp_to_v_points(h_k)
    flux_div_k = divergence_cgrid(
        h_u * u * u_mask_3d, h_v * v * v_mask_3d, grid,
    )
    w = _diagnose_w_from_flux_div(flux_div_k, z_coord, thickness_weighted=True)

    # --- 4b. Baroclinic perturbation velocity ---
    # The barotropic solver handles the depth-averaged momentum.
    # The baroclinic step must operate on the PERTURBATION velocity
    # u' = u - U_bar to avoid double-counting the barotropic tendency.
    U_bar = jnp.sum(u * h_u, axis=-1) / jnp.maximum(
        jnp.sum(h_u, axis=-1), 1e-10) * u_mask  # (n_lat, n_lon+1)
    V_bar = jnp.sum(v * h_v, axis=-1) / jnp.maximum(
        jnp.sum(h_v, axis=-1), 1e-10) * v_mask  # (n_lat+1, n_lon)
    u_prime = u - U_bar[..., jnp.newaxis]
    v_prime = v - V_bar[..., jnp.newaxis]

    # --- 5. Coriolis ---
    # Coriolis is NOT included in the returned momentum tendencies.
    # It is applied as a forward-backward (Matsuno) step in the step
    # function (ocean_model_latlon_cgrid.py), which is unconditionally
    # stable for inertial oscillations.  Forward Euler Coriolis amplifies
    # by sqrt(1 + (f*dt)^2) per step and blows up within ~1 day at
    # high latitudes.

    # --- 6. Kinetic energy gradient (from TOTAL velocity, #160) ---
    # MOM6-style: KE from total u, not perturbation u'. The depth-mean
    # contribution enters F_slow for the barotropic solver; the
    # perturbation is applied to 3D velocity in the step function.
    _mom_adv = config.momentum_advection
    if _mom_adv in ("weno5", "weno7"):
        # WENO5 {u²;u} and {v²;v} reconstruction (Silvestri et al. 2024,
        # Phase 4b K-term, Table 2: always WENO5 for K regardless of
        # config order).  Replaces centered interpolation with upwind
        # bias: computes avg(u²) not (avg(u))², adding O((ΔU)²) implicit
        # KE dissipation at velocity fronts.
        u_sq_cell = _weno_usq_to_cell(u, order=5)
        v_sq_cell = _weno_vsq_to_cell(v, order=5)
        KE = 0.5 * (u_sq_cell + v_sq_cell)
    else:
        u_cell = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
        v_cell = 0.5 * (v[:-1, :, :] + v[1:, :, :])
        KE = 0.5 * (u_cell**2 + v_cell**2)

    KE_t = jnp.moveaxis(KE, -1, 0)
    dKE_dx_t = jax.vmap(lambda ke2d: gradient_x_cgrid(ke2d, grid))(KE_t)
    dKE_dy_t = jax.vmap(lambda ke2d: gradient_y_cgrid(ke2d, grid))(KE_t)
    dKE_dx = jnp.moveaxis(dKE_dx_t, 0, -1)
    dKE_dy = jnp.moveaxis(dKE_dy_t, 0, -1)

    # --- 7. Momentum tendencies (non-Coriolis only) ---
    du_dt = -dKE_dx - dp_dx / rho_0
    dv_dt = -dKE_dy - dp_dy / rho_0

    # --- 7b. Potential vorticity flux (#160, Sadourny EC) ---
    # Vector-invariant advection: (u·∇)u = ∇(KE) + (f+ζ) × u.
    # Coriolis (f × u) handled in step function; here only ζ × u.
    #
    # MOM6-style total-velocity Sadourny EC scheme:
    #   ζ  = curl(u_total, v_total)    — total velocity vorticity
    #   q  = ζ / h_vertex              — potential vorticity at vertices
    #   Fv = h·v at v-faces            — thickness-weighted mass flux
    #   du/dt += q̄ · Fv_at_u           — PV times mass flux at u-faces
    #   dv/dt -= q̄ · Fu_at_v           — PV times mass flux at v-faces
    # Energy conserving (Sadourny 1975) and uses total velocity, fixing
    # the perturbation-only cross-term errors from issue #160.
    #
    # When momentum_advection == "weno5"/"weno7", the 2-point q average
    # is replaced by WENO-Z reconstruction (Silvestri et al. 2024).

    # Vorticity from TOTAL velocity (not perturbation u')
    zeta = curl_vertex_cgrid(u, v, grid)  # (n_lat+1, n_lon+1, nlev)

    # Layer thickness at vertices (4-cell average matching vertex layout)
    h_sw = jnp.roll(h_k, 1, axis=1)  # h_k[:, (j-1)%n_lon, :]
    h_vtx_interior = 0.25 * (h_k[:-1] + h_k[1:]
                              + h_sw[:-1] + h_sw[1:])  # (n_lat-1, n_lon, nlev)
    h_vtx_south = 0.5 * (h_k[0:1] + h_sw[0:1])        # (1, n_lon, nlev)
    h_vtx_north = 0.5 * (h_k[-1:] + h_sw[-1:])         # (1, n_lon, nlev)
    h_vtx = jnp.concatenate(
        [h_vtx_south, h_vtx_interior, h_vtx_north], axis=0,
    )  # (n_lat+1, n_lon, nlev)
    h_vtx = jnp.concatenate(
        [h_vtx, h_vtx[:, 0:1, :]], axis=1,
    )  # (n_lat+1, n_lon+1, nlev)

    # Potential vorticity q = ζ / h at vertices
    q = zeta / jnp.maximum(h_vtx, 1e-10)

    # Thickness-weighted mass fluxes at faces
    Fv = h_v * v * v_mask_3d  # (n_lat+1, n_lon, nlev)
    Fu = h_u * u * u_mask_3d  # (n_lat, n_lon+1, nlev)

    # Average Fv to u-points (4-point, periodic in lon)
    Fv_west = jnp.roll(Fv, 1, axis=1)
    Fv_at_u_core = 0.25 * (Fv[:-1] + Fv[1:]
                            + Fv_west[:-1] + Fv_west[1:])  # (n_lat, n_lon, nlev)
    Fv_at_u = jnp.concatenate(
        [Fv_at_u_core, Fv_at_u_core[:, 0:1, :]], axis=1,
    )  # (n_lat, n_lon+1, nlev)

    # Average Fu to v-points (4-point, zero-padded at poles)
    n_lon_loc = Fu.shape[1]   # n_lon+1
    nlev_loc = Fu.shape[2]
    zero_u = jnp.zeros((1, n_lon_loc, nlev_loc), dtype=u.dtype)
    Fu_ext = jnp.concatenate([zero_u, Fu, zero_u], axis=0)  # (n_lat+2, n_lon+1, nlev)
    Fu_at_v = 0.25 * (Fu_ext[:-1, :-1, :] + Fu_ext[:-1, 1:, :]
                       + Fu_ext[1:, :-1, :] + Fu_ext[1:, 1:, :])  # (n_lat+1, n_lon, nlev)

    # Total velocity at u/v faces (for WENO upwinding direction)
    v_west_total = jnp.roll(v, 1, axis=1)
    v_at_u_core = 0.25 * (v[:-1] + v[1:]
                           + v_west_total[:-1] + v_west_total[1:])
    v_at_u = jnp.concatenate(
        [v_at_u_core, v_at_u_core[:, 0:1, :]], axis=1,
    )  # (n_lat, n_lon+1, nlev)

    u_ext = jnp.concatenate([zero_u, u, zero_u], axis=0)
    u_at_v = 0.25 * (u_ext[:-1, :-1, :] + u_ext[:-1, 1:, :]
                      + u_ext[1:, :-1, :] + u_ext[1:, 1:, :])  # (n_lat+1, n_lon, nlev)

    # Reconstruct q from vertices to velocity points
    if _mom_adv in ("weno5", "weno7"):
        _weno_order = {"weno5": 5, "weno7": 7}[_mom_adv]
        q_at_u = _weno_zeta_at_u(
            q, v, v_at_u, order=_weno_order)
        q_at_v = _weno_zeta_at_v(
            q, u, u_at_v, order=_weno_order)
    else:
        # Sadourny EC: 2-point average of q to faces
        q_at_u = 0.5 * (q[:-1, :, :] + q[1:, :, :])   # (n_lat, n_lon+1, nlev)
        q_at_v = 0.5 * (q[:, :-1, :] + q[:, 1:, :])    # (n_lat+1, n_lon, nlev)

    du_dt = du_dt + q_at_u * Fv_at_u
    dv_dt = dv_dt - q_at_v * Fu_at_v

    # --- 7c. Divergence flux (D term, Silvestri et al. 2024 Eqs. 31-32) ---
    # WENO reconstruction of velocity divergence to faces adds implicit
    # dissipation of the divergent mode, complementing the Z-term's
    # rotational dissipation.  WENO5 always (Table 2).
    if _mom_adv in ("weno5", "weno7"):
        vel_div = divergence_cgrid(u * u_mask_3d, v * v_mask_3d, grid)
        D_at_u = _weno_cell_to_uface(vel_div, vel_div, u, order=5)
        D_at_v = _weno_cell_to_vface(vel_div, vel_div, v, order=5)
        du_dt = du_dt + D_at_u * u * u_mask_3d
        dv_dt = dv_dt + D_at_v * v * v_mask_3d

    # --- 8. Vertical advection of u, v (perturbation velocity) ---
    # Issue #171 Level-1 fix: use interface-upwind flux-form momentum
    # advection instead of the cell-centered upwind gradient form.
    # The flux-form helper returns -(F_top - F_bot) / h_u with
    # F = w_half * u_upwind_at_interface and F = 0 at top/bottom by
    # construction, eliminating the hard-zero gradient pathology at
    # k=0 / k=nlev-1 and matching the tracer-path interface upwind.
    # Full flux-form momentum update (Level 2) still requires step-
    # function restructuring; tracked on #171.
    J_u = interp_cell_to_uface(J)
    J_v = _interp_to_v_points(J)
    h_u_old = z_coord.dz_ref[jnp.newaxis, jnp.newaxis, :] * J_u[..., jnp.newaxis]
    h_v_old = z_coord.dz_ref[jnp.newaxis, jnp.newaxis, :] * J_v[..., jnp.newaxis]
    w_u = interp_cell_to_uface(w)
    w_v = _interp_to_v_points(w)
    if _mom_adv in ("weno5", "weno7"):
        # WENO vertical momentum advection removes the implicit viscosity
        # (~|w|*dz/2) that first-order upwind provides.  Requires
        # compensating vertical viscosity (KPP / Richardson-A_v, #204).
        du_dt = du_dt + _flux_form_vertical_momentum_advection_weno(
            u_prime, w_u, h_u_old, order=_weno_order)
        dv_dt = dv_dt + _flux_form_vertical_momentum_advection_weno(
            v_prime, w_v, h_v_old, order=_weno_order)
    else:
        # Default: 1st-order upwind.  The implicit viscosity (~|w|*dz/2)
        # damps baroclinic shear that explicit A_v=1e-5 cannot.
        du_dt = du_dt + _flux_form_vertical_momentum_advection(
            u_prime, w_u, h_u_old)
        dv_dt = dv_dt + _flux_form_vertical_momentum_advection(
            v_prime, w_v, h_v_old)

    # --- 9. Tracer tendencies (diffusion + physics only) ---
    # Horizontal AND vertical tracer advection are handled in the step()
    # function using barotropic-averaged transport (Hallberg 1997, #102).
    # Vertical velocity w is diagnosed from the barotropic-averaged
    # per-layer divergence, ensuring 3D transport consistency.
    #
    # The tendency here includes only: diffusion and physics.
    tracers = jnp.stack([T, S], axis=0)

    def tracer_tendency(tr: jnp.ndarray) -> jnp.ndarray:
        dtr_dt = jnp.zeros_like(tr)

        if config.K_h > 0:
            dtr_dt = dtr_dt + config.K_h * laplacian_cgrid(tr, grid, mask=mask)
        if config.K_bih > 0:
            dtr_dt = dtr_dt - config.K_bih * bilaplacian_cgrid(tr, grid, mask=mask)
        # Vertical tracer diffusion: always applied regardless of physics
        # pipeline state. The physics pipeline's vertical_mixing module is
        # a separate concept (e.g., KPP). Baseline K_v diffusion should
        # always be active when K_v > 0. (Fixes #150.)
        if config.K_v > 0 and tr.shape[-1] >= 2:
            jac_v = jnp.maximum(J[..., jnp.newaxis], 1e-10)
            dz_actual_loc = z_coord.dz_ref * jac_v
            dtr_dz_half = (tr[..., :-1] - tr[..., 1:]) / (
                z_coord.dz_half_ref * jac_v
            )
            flux = config.K_v * dtr_dz_half
            zeros_face = jnp.zeros(
                (*tr.shape[:-1], 1), dtype=tr.dtype,
            )
            flux_full = jnp.concatenate(
                [zeros_face, flux, zeros_face], axis=-1,
            )
            dtr_dt = dtr_dt + (
                flux_full[..., :-1] - flux_full[..., 1:]
            ) / dz_actual_loc
        return dtr_dt

    tracer_tend = jax.vmap(tracer_tendency, in_axes=0, out_axes=0)(tracers)
    dT_dt = tracer_tend[0]
    dS_dt = tracer_tend[1]

    # --- 10. Mixing (viscosity on perturbation velocity) ---
    # Uses the proper vector Laplacian grad(div) - k×grad(curl) directly
    # on face velocities, avoiding the lossy cell-center detour.
    # See issue #105 for details.
    if config.A_h > 0:
        vlap_u, vlap_v = vector_laplacian_cgrid(
            u_prime, v_prime, grid,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        du_dt = du_dt + config.A_h * vlap_u
        dv_dt = dv_dt + config.A_h * vlap_v

    if config.B_h > 0:
        bilap_u, bilap_v = vector_bilaplacian_cgrid(
            u_prime, v_prime, grid,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        # Scale biharmonic coefficient with (cos(lat)/cos_max)^4 to prevent
        # CFL violation near poles where dx shrinks (MOM6 convention).
        scale_u, scale_v = biharmonic_scaling_factor(grid)
        du_dt = du_dt - config.B_h * scale_u[:, None, None] * bilap_u
        dv_dt = dv_dt - config.B_h * scale_v[:, None, None] * bilap_v

    if config.C_smag > 0:
        smag_u, smag_v = smagorinsky_biharmonic_tendency_cgrid(
            u_prime, v_prime, grid, config.C_smag,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        du_dt = du_dt - smag_u
        dv_dt = dv_dt - smag_v

    if getattr(config, "C_leith", 0.0) > 0:
        leith_u, leith_v = leith_biharmonic_tendency_cgrid(
            u_prime, v_prime, grid, config.C_leith,
            modified=getattr(config, "C_leith_modified", False),
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        du_dt = du_dt - leith_u
        dv_dt = dv_dt - leith_v

    if config.bottom_drag_r > 0:
        # Drag acts on the full velocity (not perturbation) — the ocean
        # floor sees the total flow.  Consistent with MPAS and MOM6.
        # r is in [m/s]: du/dt = -r * u / dz_bottom  (resolution-independent stress).
        dz_bot_u = z_coord.dz_ref[-1] * jnp.maximum(interp_cell_to_uface(J), 1e-10)
        dz_bot_v = z_coord.dz_ref[-1] * jnp.maximum(_interp_to_v_points(J), 1e-10)
        du_dt = du_dt.at[..., -1].add(-config.bottom_drag_r * u[..., -1] / dz_bot_u)
        dv_dt = dv_dt.at[..., -1].add(-config.bottom_drag_r * v[..., -1] / dz_bot_v)

    if config.A_v > 0 and u.shape[-1] >= 2:
        jac_v_u = jnp.maximum(interp_cell_to_uface(J)[..., jnp.newaxis], 1e-10)
        jac_v_v = jnp.maximum(_interp_to_v_points(J)[..., jnp.newaxis], 1e-10)
        for vel, jac, is_u in [(u_prime, jac_v_u, True), (v_prime, jac_v_v, False)]:
            dv_dz_half = (vel[..., :-1] - vel[..., 1:]) / (
                z_coord.dz_half_ref * jac
            )
            flux = config.A_v * dv_dz_half
            zeros_face = jnp.zeros((*vel.shape[:-1], 1), dtype=vel.dtype)
            flux_full = jnp.concatenate([zeros_face, flux, zeros_face], axis=-1)
            vdiff = (flux_full[..., :-1] - flux_full[..., 1:]) / (
                z_coord.dz_ref * jac
            )
            if is_u:
                du_dt = du_dt + vdiff
            else:
                dv_dt = dv_dt + vdiff

    # --- 10b. Physics tendencies (surface forcing, bottom drag, etc.) ---
    # The physics pipeline expects cell-center u/v shapes (shared with
    # A-grid and cubed-sphere).  Create a cell-center proxy state so
    # the physics functions produce (n_lat, n_lon, nlev) output, then
    # interpolate momentum tendencies to C-grid face points.
    if physics_fn is not None:
        u_cell = 0.5 * (u[:, :-1, :] + u[:, 1:, :])  # (n_lat, n_lon, nlev)
        v_cell = 0.5 * (v[:-1, :, :] + v[1:, :, :])   # (n_lat, n_lon, nlev)
        cc_state = state._replace(
            u=state.u.replace(data=u_cell),
            v=state.v.replace(data=v_cell),
        )
        phys = physics_fn(cc_state, grid, z_coord, surface_forcing)
        du_dt = du_dt + interp_cell_to_uface(phys.du_dt.data)
        dv_dt = dv_dt + _interp_to_v_points(phys.dv_dt.data)
        dT_dt = dT_dt + phys.dT_dt.data
        dS_dt = dS_dt + phys.dS_dt.data

    # --- 10c. Sponge layer relaxation ---
    # Cast sponge arrays to state dtype to prevent float64 promotion when
    # the precision policy stores state in float32 (crashes barotropic scan).
    if sponge is not None:
        dT_dt, dS_dt = apply_sponge_tracer_relaxation(
            dT_dt, dS_dt, T, S, sponge, mask=None, expand_gamma_axis=-1,
        )
        _dt = T.dtype
        if sponge.u_ref is not None:
            gamma_u = interp_cell_to_uface(sponge.gamma.astype(_dt))[..., jnp.newaxis]
            du_dt = du_dt + gamma_u * (sponge.u_ref.astype(_dt) - u)
        if sponge.v_ref is not None:
            gamma_v = _interp_to_v_points(sponge.gamma.astype(_dt))[..., jnp.newaxis]
            dv_dt = dv_dt + gamma_v * (sponge.v_ref.astype(_dt) - v)

    # --- 11. Land masking ---
    du_dt = du_dt * u_mask_3d
    dv_dt = dv_dt * v_mask_3d
    dT_dt = dT_dt * mask_3d
    dS_dt = dS_dt * mask_3d

    # --- 12. Free-surface tendency ---
    deta_dt = -jnp.sum(flux_div_k, axis=-1) * mask

    dims_u = ("lat", "lon_u", "level")
    dims_v = ("lat_v", "lon", "level")
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    return LatLonCGridOceanTendencies(
        du_dt=Field(data=du_dt, name="du_dt", dims=dims_u, units="m/s^2"),
        dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims_v, units="m/s^2"),
        dT_dt=Field(data=dT_dt, name="dT_dt", dims=dims_3d, units="degC/s"),
        dS_dt=Field(data=dS_dt, name="dS_dt", dims=dims_3d, units="PSU/s"),
        deta_dt=Field(data=deta_dt, name="deta_dt", dims=dims_2d, units="m/s"),
        dH_bathy_dt=Field(
            data=jnp.zeros_like(H_bathy), name="dH_bathy_dt",
            dims=dims_2d, units="m/s",
        ),
        dland_mask_dt=Field(
            data=jnp.zeros_like(mask), name="dland_mask_dt",
            dims=dims_2d, units="1/s",
        ),
    )


