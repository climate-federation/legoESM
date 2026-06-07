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
    OceanPartialCellCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
)
from legoesm.ocean.state import (
    LatLonCGridOceanState,
    LatLonCGridOceanTendencies,
    MomentumTendencyDiagnostics,
    LatLonCGridOceanConfig,
    SurfaceTracerForcing,
)
from legoesm.ocean.dynamics.ocean_tendency_common import (
    apply_sponge_tracer_relaxation,
    iterate_eos_and_pressure_anomaly,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    biharmonic_scaling_factor,
    equatorial_boost_factor,
    polar_cap_boost_factor,
    laplacian_scaling_factor,
    divergence_cgrid,
    gradient_x_cgrid,
    gradient_y_cgrid,
    bilaplacian_cgrid,
    laplacian_cgrid,
    vector_bilaplacian_cgrid,
    vector_laplacian_cgrid,
    vector_laplacian_dissipation_cgrid,
    flux_divergence_viscosity_cgrid,
    interp_cell_to_uface,
    is_tripolar,
    min_cell_to_uface,
    min_cell_to_vface,
    curl_vertex_cgrid,
    pad_ns_scalar,
    pad_ns_zero,
    _fold_is_local,
    smagorinsky_biharmonic_tendency_cgrid,
    smagorinsky_viscosity_cgrid,
    smagorinsky_viscosity_q_cgrid,
    strain_rate_cgrid,
    viscous_tendency_cgrid,
    leith_biharmonic_tendency_cgrid,
    _compute_vertex_mask,
    compute_face_masks_3d,
    density_jacobian_pgf_smc03_x,
    density_jacobian_pgf_smc03_y,
    partial_cell_pgf_correction_x,
    partial_cell_pgf_correction_y,
    pv_flux_al81_partial_cell,
)
from legoesm.core.weno import weno_reconstruct_split, weno_upwind
from legoesm.ocean.advection import (
    flux_form_vertical_tracer_advection_weno5,
    flux_form_vertical_tracer_advection_weno7,
)
from legoesm.ocean.vertical import (
    diagnose_w_from_flux_div as _diagnose_w_from_flux_div,
    flux_form_vertical_momentum_advection as _flux_form_vertical_momentum_advection,
    compute_centroid_depth,
)


# interp_cell_to_uface is imported from latlon_cgrid_operators (shared).

# Single source of truth for the lat-lon C-grid horizontal momentum-advection
# dispatch literals (dispatch discipline: validated at config construction;
# unknown -> ValueError, never a silent fallthrough to vector-invariant).
VALID_MOMENTUM_ADVECTION = frozenset(
    {"vector_invariant", "weno5", "weno7", "flux_form"}
)
# Reconstruction schemes for the advected velocity in the flux-form path
# (config.momentum_flux_scheme; only used when momentum_advection="flux_form").
VALID_MOMENTUM_FLUX_SCHEME = frozenset({"upwind", "centered"})
# Lateral (harmonic) momentum-viscosity operator form (config.lateral_viscosity_operator):
# the default VECTOR Laplacian grad(div)−k×grad(curl), or Veros's component-wise
# FLUX-DIVERGENCE harmonic friction ∇·(A_h∇u). Validated at config construction;
# unknown -> ValueError (dispatch discipline).
VALID_LATERAL_VISCOSITY_OPERATOR = frozenset({"vector_laplacian", "flux_divergence"})


def _interp_to_v_points(f: jnp.ndarray, grid=None) -> jnp.ndarray:
    """Interpolate cell-center field to v-points (lat interfaces).

    Parameters
    ----------
    f : array, shape (n_lat, n_lon, ...) at cell centers.
    grid : optional LatLonGrid or LatLonCGridGeometry.
        When provided, ``pad_ns_zero`` is used (MPI-aware halo exchange
        fills neighbor rows at partition cuts), and the fold correction
        is applied on the northernmost rank via ``_fold_is_local()``.

    Returns
    -------
    f_v : array, shape (n_lat+1, n_lon, ...) at v-points.
    """
    f_padded = pad_ns_zero(f)
    f_v = 0.5 * (f_padded[:-1] + f_padded[1:])
    from legoesm.grids.halo_latlon import zero_polar_lat_ends
    f_v = zero_polar_lat_ends(f_v)
    fold = getattr(grid, "fold", None) if grid is not None else None
    if _fold_is_local(grid):
        n_cols = f_v.shape[1]
        n_lon = fold.perm_T.shape[0]
        last_interior_vface = 0.5 * (f[-2:-1] + f[-1:])
        if n_cols == n_lon:
            north = last_interior_vface[:, fold.perm_T]
        else:
            core = last_interior_vface[:, :n_lon][:, fold.perm_T]
            north = jnp.concatenate([core, core[:, 0:1]], axis=1)
        f_v = jnp.concatenate([f_v[:-1], north], axis=0)
    return f_v


# Canonical Van Leer limiter from core (redundancy audit), aliased to the local
# private name so call sites are unchanged.
from legoesm.core.flux_limiters import van_leer_limiter as _van_leer_limiter


def _tvd_to_u_points(
    f: jnp.ndarray,
    mass_flux_u: jnp.ndarray,
    limiter_fn=_van_leer_limiter,
) -> jnp.ndarray:
    """TVD interpolation to u-points. Second-order, monotonic (#170).

    ``limiter_fn`` selects the flux limiter family; defaults to Van Leer
    (used by ``tracer_advection="tvd"``). Pass
    :func:`legoesm.ocean.dynamics._flux_limiters.sweby_limiter` for
    Veros-compatible superbee (``tracer_advection="superbee"``).
    """
    eps = 1e-30
    f_left = jnp.roll(f, 1, axis=1)
    f_right = f
    f_left2 = jnp.roll(f, 2, axis=1)
    f_right2 = jnp.roll(f, -1, axis=1)
    delta_pos = f_right - f_left
    r_pos = (f_left - f_left2) / jnp.where(jnp.abs(delta_pos) > eps, delta_pos, eps)
    delta_neg = f_left - f_right
    r_neg = (f_right2 - f_right) / jnp.where(jnp.abs(delta_neg) > eps, delta_neg, eps)
    f_pos = f_left + 0.5 * limiter_fn(r_pos) * delta_pos
    f_neg = f_right + 0.5 * limiter_fn(r_neg) * delta_neg
    n_lon = f.shape[1]
    mf = mass_flux_u[:, :n_lon]
    f_tvd = jnp.where(mf > 0, f_pos, f_neg)
    if f.ndim >= 3:
        return jnp.concatenate([f_tvd, f_tvd[:, 0:1, :]], axis=1)
    return jnp.concatenate([f_tvd, f_tvd[:, 0:1]], axis=1)


def _tvd_to_v_points(
    f: jnp.ndarray,
    mass_flux_v: jnp.ndarray,
    grid=None,
    limiter_fn=_van_leer_limiter,
) -> jnp.ndarray:
    """TVD interpolation to v-points. Solid wall at poles (#170).

    ``limiter_fn`` selects the flux limiter; see :func:`_tvd_to_u_points`.
    """
    eps = 1e-30
    f_south = f[:-1]; f_north = f[1:]
    f_south2 = jnp.concatenate([f[:1], f[:-2]], axis=0)
    fold = getattr(grid, "fold", None) if grid is not None else None
    if fold is not None and fold.is_active and fold.fold_j >= 0:
        f_north2 = jnp.concatenate([f[2:], f[-1:, fold.perm_T]], axis=0)
    else:
        f_north2 = jnp.concatenate([f[2:], f[-1:]], axis=0)
    delta_pos = f_north - f_south
    r_pos = (f_south - f_south2) / jnp.where(jnp.abs(delta_pos) > eps, delta_pos, eps)
    delta_neg = f_south - f_north
    r_neg = (f_north2 - f_north) / jnp.where(jnp.abs(delta_neg) > eps, delta_neg, eps)
    f_pos = f_south + 0.5 * limiter_fn(r_pos) * delta_pos
    f_neg = f_north + 0.5 * limiter_fn(r_neg) * delta_neg
    f_tvd = jnp.where(mass_flux_v[1:-1] > 0, f_pos, f_neg)
    if grid is not None:
        return pad_ns_scalar(f_tvd, grid)
    return pad_ns_zero(f_tvd)


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
    grid=None,
) -> jnp.ndarray:
    """First-order upwind interpolation of cell-center field to v-points.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon, ...) at cell centers.
    mass_flux_v : array, shape (n_lat+1, n_lon, ...) at v-points.
        Sign convention: positive = flow in +i (northward) direction.
    grid : optional LatLonGrid or LatLonCGridGeometry.
        When provided and a tripolar fold is active, the north-boundary
        v-face value is computed from the fold-partner cells.

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

    if grid is not None:
        return pad_ns_scalar(f_upwind, grid)
    return pad_ns_zero(f_upwind)


def _neumann_fill_cgrid(
    f: jnp.ndarray,
    mask: jnp.ndarray,
    grid=None,
) -> jnp.ndarray:
    """Fill land cells with nearest ocean-neighbor (Neumann BC).

    Same algorithm as latlon_operators.neumann_fill_latlon.

    Parameters
    ----------
    f : (n_lat, n_lon, ...) field at cell centers.
    mask : (n_lat, n_lon) ocean mask (1 = wet, 0 = land).
    grid : optional LatLonGrid or LatLonCGridGeometry.
        When provided and a tripolar fold is active, the north neighbor
        of the fold row uses the fold-partner cell instead of repeating
        the last row.
    """
    fold = getattr(grid, "fold", None) if grid is not None else None
    use_fold = fold is not None and fold.is_active and fold.fold_j >= 0

    m = mask
    filled = f
    for _ in range(3):
        f_s = jnp.concatenate([filled[0:1], filled[:-1]], axis=0)
        m_s = jnp.concatenate([m[0:1], m[:-1]], axis=0)
        if use_fold:
            north_of_fold = filled[-1:, fold.perm_T]
            f_n = jnp.concatenate([filled[1:], north_of_fold], axis=0)
            m_n = jnp.concatenate([m[1:], m[-1:, fold.perm_T]], axis=0)
        else:
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


# Note: ``_neumann_fill_vertex`` was promoted to a public
# ``neumann_fill_vertex`` helper in ``latlon_cgrid_operators`` so it
# can be shared with the AL81 PV-flux scheme there.  Keep an alias
# for backward compatibility within this module.
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    neumann_fill_vertex as _neumann_fill_vertex,
)


# =============================================================================
# WENO momentum advection helpers (Phase 2b, Silvestri et al. 2024)
# =============================================================================

def _weno_zeta_at_u(
    phi: jnp.ndarray,
    v_smooth: jnp.ndarray,
    v_at_u: jnp.ndarray,
    order: int = 5,
    u_smooth: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """WENO reconstruction of a vertex field to u-faces (meridional).

    Uses Silvestri et al. (2024) Eq. 43 smoothness-optimised stencil:
    ``{ζ; u} = ({ζ; ⟨v⟩_i} + {ζ; ⟨u⟩_j}) / 2`` — average of two WENO
    reconstructions, one with ⟨v⟩_i smoothness and one with ⟨u⟩_j
    smoothness.  This makes the scheme less sensitive to noise in any
    single velocity component.  If *u_smooth* is None, falls back to
    the single ⟨v⟩_i reconstruction.

    Parameters
    ----------
    phi : (n_lat+1, n_lon+1, nlev) field at vertices to reconstruct.
    v_smooth : (n_lat+1, n_lon, nlev) at v-faces (⟨v⟩_i smoothness).
    v_at_u : (n_lat, n_lon+1, nlev) at u-faces (upwinding velocity).
    order : {5, 7}
    u_smooth : (n_lat, n_lon+1, nlev) or None
        u at u-faces for the ⟨u⟩_j smoothness path.

    Returns
    -------
    phi_at_u : (n_lat, n_lon+1, nlev)
    """
    hw = {5: 3, 7: 4}[order]
    n_lat = phi.shape[0] - 1  # n_lat+1 vertices → n_lat u-faces
    nlev = phi.shape[2]

    # ⟨v⟩_i: v averaged in longitude to vertex positions.
    v_w = jnp.roll(v_smooth, 1, axis=1)
    v_at_vtx = 0.5 * (v_w + v_smooth)
    v_at_vtx = jnp.concatenate(
        [v_at_vtx, v_at_vtx[:, 0:1, :]], axis=1)  # (n_lat+1, n_lon+1, nlev)

    # Convert point values to cell averages along the meridional
    # reconstruction axis before WENO.
    from legoesm.core.weno import point_to_cellavg_bounded
    conv_order = {5: 6, 7: 8}[order]
    phi_avg = point_to_cellavg_bounded(phi, axis=0, order=conv_order)
    v_at_vtx_avg = point_to_cellavg_bounded(v_at_vtx, axis=0, order=conv_order)

    # Ghost cells (Neumann BC) along axis 0 for the meridional stencil
    phi_ext = jnp.concatenate(
        [phi_avg[:1, :, :]] * hw + [phi_avg] + [phi_avg[-1:, :, :]] * hw, axis=0)
    v_ext = jnp.concatenate(
        [v_at_vtx_avg[:1, :, :]] * hw + [v_at_vtx_avg] + [v_at_vtx_avg[-1:, :, :]] * hw,
        axis=0)

    phi_stencil = [phi_ext[1 + j: n_lat + 1 + j, :, :]
                   for j in range(2 * hw)]
    psi_v_stencil = [v_ext[1 + j: n_lat + 1 + j, :, :]
                     for j in range(2 * hw)]

    phi_plus_v, phi_minus_v = weno_reconstruct_split(
        phi_stencil, psi_v_stencil, order=order)
    result_v = weno_upwind(phi_plus_v, phi_minus_v, v_at_u)

    if u_smooth is None:
        return result_v

    # ⟨u⟩_j: u averaged in latitude to vertex positions.
    n_lon_u = u_smooth.shape[1]  # n_lon+1
    zero_u = jnp.zeros((1, n_lon_u, nlev), dtype=u_smooth.dtype)
    u_ext_lat = jnp.concatenate([zero_u, u_smooth, zero_u], axis=0)
    u_at_vtx = 0.5 * (u_ext_lat[:-1, :, :] + u_ext_lat[1:, :, :])

    u_at_vtx_avg = point_to_cellavg_bounded(u_at_vtx, axis=0, order=conv_order)
    u_ext = jnp.concatenate(
        [u_at_vtx_avg[:1, :, :]] * hw + [u_at_vtx_avg] + [u_at_vtx_avg[-1:, :, :]] * hw,
        axis=0)
    psi_u_stencil = [u_ext[1 + j: n_lat + 1 + j, :, :]
                     for j in range(2 * hw)]

    phi_plus_u, phi_minus_u = weno_reconstruct_split(
        phi_stencil, psi_u_stencil, order=order)
    result_u = weno_upwind(phi_plus_u, phi_minus_u, v_at_u)

    return 0.5 * (result_v + result_u)


def _weno_zeta_at_v(
    phi: jnp.ndarray,
    u_smooth: jnp.ndarray,
    u_at_v: jnp.ndarray,
    order: int = 5,
    v_smooth: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """WENO reconstruction of a vertex field to v-faces (zonal).

    Uses Silvestri et al. (2024) Eq. 43 smoothness-optimised stencil:
    ``{ζ; u} = ({ζ; ⟨u⟩_j} + {ζ; ⟨v⟩_i}) / 2`` — average of two WENO
    reconstructions.  If *v_smooth* is None, falls back to the single
    ⟨u⟩_j reconstruction.

    Periodic in longitude.

    Parameters
    ----------
    phi : (n_lat+1, n_lon+1, nlev) field at vertices to reconstruct.
    u_smooth : (n_lat, n_lon+1, nlev) at u-faces (⟨u⟩_j smoothness).
    u_at_v : (n_lat+1, n_lon, nlev) at v-faces (upwinding velocity).
    order : {5, 7}
    v_smooth : (n_lat+1, n_lon, nlev) or None
        v at v-faces for the ⟨v⟩_i smoothness path.

    Returns
    -------
    phi_at_v : (n_lat+1, n_lon, nlev)
    """
    hw = {5: 3, 7: 4}[order]
    n_lon = phi.shape[1] - 1
    nlev = phi.shape[2]

    # ⟨u⟩_j: u averaged in latitude to vertex positions.
    n_lon_u = u_smooth.shape[1]  # n_lon+1
    zero_u = jnp.zeros((1, n_lon_u, nlev), dtype=u_smooth.dtype)
    u_ext_lat = jnp.concatenate(
        [zero_u, u_smooth, zero_u], axis=0)
    u_at_vtx = 0.5 * (u_ext_lat[:-1, :, :] + u_ext_lat[1:, :, :])

    # Convert point values to cell averages along zonal axis (periodic).
    from legoesm.core.weno import point_to_cellavg_periodic
    conv_order = {5: 6, 7: 8}[order]

    phi_core = phi[:, :n_lon, :]
    u_core = u_at_vtx[:, :n_lon, :]

    phi_core_avg = point_to_cellavg_periodic(phi_core, axis=1, order=conv_order)
    u_core_avg = point_to_cellavg_periodic(u_core, axis=1, order=conv_order)

    phi_stencil = [jnp.roll(phi_core_avg, hw - 1 - j, axis=1)
                   for j in range(2 * hw)]
    psi_u_stencil = [jnp.roll(u_core_avg, hw - 1 - j, axis=1)
                     for j in range(2 * hw)]

    phi_plus_u, phi_minus_u = weno_reconstruct_split(
        phi_stencil, psi_u_stencil, order=order)
    result_u = weno_upwind(phi_plus_u, phi_minus_u, u_at_v)

    if v_smooth is None:
        return result_u

    # ⟨v⟩_i: v averaged in longitude to vertex positions.
    v_w = jnp.roll(v_smooth, 1, axis=1)
    v_at_vtx = 0.5 * (v_w + v_smooth)
    v_at_vtx = jnp.concatenate(
        [v_at_vtx, v_at_vtx[:, 0:1, :]], axis=1)

    v_core = v_at_vtx[:, :n_lon, :]
    v_core_avg = point_to_cellavg_periodic(v_core, axis=1, order=conv_order)
    psi_v_stencil = [jnp.roll(v_core_avg, hw - 1 - j, axis=1)
                     for j in range(2 * hw)]

    phi_plus_v, phi_minus_v = weno_reconstruct_split(
        phi_stencil, psi_v_stencil, order=order)
    result_v = weno_upwind(phi_plus_v, phi_minus_v, u_at_v)

    return 0.5 * (result_u + result_v)


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
        vert_flux_div = flux_form_vertical_tracer_advection_weno5(
            u, w_half, h_u, dt=0.0)
    elif order == 7:
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

def _split_velocity_divergence(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Split velocity divergence into zonal and meridional components at cells.

    Returns the two cell-centered components of ∇·u separately:
      ``dU_di = (u[east] - u[west]) * face_dy / area``  (1/s)
      ``dV_dj = (v[north]*face_dx_n - v[south]*face_dx_s) / area``  (1/s)

    Used by the WENO D-term, which requires the matching-direction
    component to be WENO-upwinded and the cross-direction component to
    stay centered (Silvestri et al. 2024 Eqs. 31-32, Appendix C).

    Parameters
    ----------
    u : (n_lat, n_lon+1, nlev) zonal velocity at u-faces.
    v : (n_lat+1, n_lon, nlev) meridional velocity at v-faces.
    grid : LatLonGrid

    Returns
    -------
    dU_di_cell : (n_lat, n_lon, nlev)  zonal divergence component at cells.
    dV_dj_cell : (n_lat, n_lon, nlev)  meridional divergence component.
    """
    if is_tripolar(grid):
        # Tripolar: use full 2D metrics — column-0 extraction is NOT
        # valid on the bipolar cap where dy_u varies in longitude.
        face_dy = grid.dy_u                              # (n_lat, n_lon+1)
        _is_2d_dy = True
        face_dx = grid.dx_v                              # (n_lat+1, n_lon)
        fd = face_dx[:, :, jnp.newaxis]
    else:
        # Regular or Mercator: variable-dy safe.
        _is_2d_dy = False
        R = grid.radius
        dlon = grid.dlon
        lat = grid.lat
        face_dy = (grid.dy * 0.5)[:, jnp.newaxis, jnp.newaxis]  # (n_lat,1,1)
        lat_south_pole = jnp.array([-jnp.pi / 2], dtype=lat.dtype)
        lat_north_pole = jnp.array([jnp.pi / 2], dtype=lat.dtype)
        lat_interior = 0.5 * (lat[:-1] + lat[1:])
        lat_v = jnp.concatenate([lat_south_pole, lat_interior, lat_north_pole])
        cos_lat_v = jnp.cos(lat_v)
        face_dx = R * cos_lat_v * dlon                  # (n_lat+1,)
        fd = face_dx[:, jnp.newaxis, jnp.newaxis]

    # Zonal flux divergence at cells.
    if _is_2d_dy:
        # Per-face dy: each u-face has its own meridional extent.
        face_dy_e = face_dy[:, 1:, jnp.newaxis]         # (n_lat, n_lon, 1)
        face_dy_w = face_dy[:, :-1, jnp.newaxis]        # (n_lat, n_lon, 1)
        net_zonal = u[:, 1:, :] * face_dy_e - u[:, :-1, :] * face_dy_w
    else:
        net_zonal = (u[:, 1:, :] - u[:, :-1, :]) * face_dy

    # Meridional flux divergence at cells.
    net_merid = v[1:, :, :] * fd[1:] - v[:-1, :, :] * fd[:-1]

    area = grid.area[..., jnp.newaxis]                  # (n_lat, n_lon, 1)
    return net_zonal / area, net_merid / area


def _centered_cell_to_uface(phi: jnp.ndarray) -> jnp.ndarray:
    """Centered cell→u-face interpolation, periodic in longitude."""
    phi_west = jnp.roll(phi, 1, axis=1)
    phi_at_uface_core = 0.5 * (phi_west + phi)
    return jnp.concatenate([phi_at_uface_core, phi_at_uface_core[:, 0:1, :]], axis=1)


def _centered_cell_to_vface(phi: jnp.ndarray) -> jnp.ndarray:
    """Centered cell→v-face interpolation with wall BC (v=0 at poles)."""
    interior = 0.5 * (phi[:-1, :, :] + phi[1:, :, :])
    n_lon = phi.shape[1]
    nlev = phi.shape[2]
    zero = jnp.zeros((1, n_lon, nlev), dtype=phi.dtype)
    return jnp.concatenate([zero, interior, zero], axis=0)


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
    hw = {5: 3, 7: 4}[order]
    n_lon = phi.shape[1]

    # Convert point values to cell averages before WENO reconstruction.
    from legoesm.core.weno import point_to_cellavg_periodic
    conv_order = {5: 6, 7: 8}[order]
    phi_avg = point_to_cellavg_periodic(phi, axis=1, order=conv_order)
    psi_avg = point_to_cellavg_periodic(psi, axis=1, order=conv_order)

    # Periodic stencil along axis 1 (longitude).
    # U-face j is between cell j-1 and cell j.  WENO at the face between
    # cells (j-1) and j needs cells j-hw, ..., j+hw-1.
    # Roll offset for stencil position s: hw - s places cell j-hw+s at
    # position j.
    phi_stencil = [jnp.roll(phi_avg, hw - s, axis=1)
                   for s in range(2 * hw)]
    psi_stencil = [jnp.roll(psi_avg, hw - s, axis=1)
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
    hw = {5: 3, 7: 4}[order]
    n_lat = phi.shape[0]
    nlev = phi.shape[2]
    n_lon = phi.shape[1]

    # Convert point values to cell averages before WENO reconstruction.
    from legoesm.core.weno import point_to_cellavg_bounded
    conv_order = {5: 6, 7: 8}[order]
    phi_avg = point_to_cellavg_bounded(phi, axis=0, order=conv_order)
    psi_avg = point_to_cellavg_bounded(psi, axis=0, order=conv_order)

    # Ghost cells (Neumann BC) along axis 0 for meridional stencil.
    phi_ext = jnp.concatenate(
        [phi_avg[:1, :, :]] * hw + [phi_avg] + [phi_avg[-1:, :, :]] * hw, axis=0)
    psi_ext = jnp.concatenate(
        [psi_avg[:1, :, :]] * hw + [psi_avg] + [psi_avg[-1:, :, :]] * hw, axis=0)

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


def laplacian_smag_cfl_cap(grid, dt: float, safety: float):
    """Per-cell upper-bound (ceiling) on the Laplacian-Smagorinsky coefficient.

    Returns ``(cap_h, cap_q)`` = ``safety * area * cos^2(lat) / dt`` at cell
    centres and vertices -- the maximum STABILISING viscosity supplied to the
    under-resolved western-boundary-current jets during the cold start.

    This is a TUNED per-cell viscosity ceiling, deliberately NOT the strict
    rectangular explicit-diffusion CFL limit ``safety/(dt*(1/dx^2+1/dy^2))``.
    EMPIRICAL EVIDENCE (eORCA025 WOA cold-start): the rectangular/metric form is
    too conservative -- it STARVES the marginally-resolved WBC of the viscosity
    it needs and blew up by day 0.25 (job 8126978); ``area*cos^2(lat)/dt`` runs
    stably (the energy-stable stress-tensor operator tolerates this larger
    coefficient, which is ~20% above the rectangular limit at the WBC and more
    at high latitude). So the cold start needs the LARGER ceiling, not the
    smaller "correct" CFL bound. ``cos^2(lat)`` (floored at 0.04, ~78.5 deg)
    tapers the ceiling toward the poles so the small, metrically distorted
    high-latitude coastal cells keep the lighter default viscosity.

    Grid dispatch: ``LatLonCGridGeometry`` (tripole/Mercator) exposes
    ``area_T``/``lat_T``; the simpler ``LatLonGrid`` exposes ``area``/``lat2d``.
    ``cap_q`` replicates the centre ceiling onto the (n_lat+1, n_lon+1) vertex
    field, edge-padded in latitude and PERIODIC in longitude (append column 0 --
    the SAME cyclic wrap the q-point Laplacian uses), so the seam is continuous:
    ``cap_q[:, -1] == cap_q[:, 0]`` (no edge-replication discontinuity).
    """
    lat2d = getattr(grid, "lat_T", None)
    if lat2d is None:
        lat2d = grid.lat2d
    area_h = getattr(grid, "area_T", None)
    if area_h is None:
        area_h = grid.area
    cos2 = jnp.maximum(jnp.cos(lat2d) ** 2, 0.04)            # poleward floor
    cap_h = safety * area_h * cos2 / dt                      # (n_lat, n_lon)
    # Vertex ceiling: pad to (n_lat+1, n_lon+1) -- edge in lat, periodic in lon.
    cap_q = jnp.pad(cap_h, ((0, 1), (0, 0)), mode="edge")    # (n_lat+1, n_lon)
    cap_q = jnp.concatenate([cap_q, cap_q[:, :1]], axis=1)   # (n_lat+1, n_lon+1)
    return cap_h, cap_q


def _bc_geometry_and_density(
    eta_safe, H_bathy, z_coord, config, T, S, mask, grid, rho_0, g_val,
):
    """Stages 1-3 of the baroclinic tendencies: layer thickness + Jacobian,
    density from the EOS, and the baroclinic pressure anomaly (filled).

    Pure substage extracted verbatim from
    ``latlon_cgrid_ocean_baroclinic_tendencies`` (Q8 decomposition; same
    operators + order -> bit-identical). Returns ``(J, h_k, rho_prime,
    p_prime_filled)``; the in-situ ``rho`` and unfilled ``p_prime`` are not
    consumed downstream.
    """
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
    #
    # Partial-cell extension: when z_coord is an
    # OceanPartialCellCoordinate, pass z_coord.h_partial as h_actual so
    # the cumsum integrates to each cell's actual centroid depth,
    # accounting for the partial bottom cell.  Cells below the
    # seafloor have h_partial=0 and contribute zero pressure increment.
    # For pure z* coord (legacy), h_actual=None falls back to dz_ref.
    _h_actual_pprime = (
        z_coord.h_partial
        if isinstance(z_coord, OceanPartialCellCoordinate)
        else None
    )
    eos_fn = make_eos_fn(config.eos, getattr(config, 'eos_linear', None))
    rho, rho_prime, p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask,
        lambda field: _neumann_fill_cgrid(field, mask, grid=grid),
        eos_fn, z_coord.dz_ref, rho_0, g_val,
        n_iter=2,
        hi_precision_pressure=True,
        h_actual=_h_actual_pprime,
    )

    p_prime_filled = _neumann_fill_cgrid(p_prime, mask, grid=grid)
    return J, h_k, rho_prime, p_prime_filled


def _bc_vertical_and_depthmean_velocity(
    h_k, u, v, u_mask_3d, v_mask_3d, grid, z_coord, u_mask, v_mask,
):
    """Stages 4-4b: vertical velocity from the FV flux divergence + the
    depth-mean velocity, returning the perturbation velocities.

    Pure verbatim extraction (Q8). Returns ``(h_u, h_v, flux_div_k, w,
    u_prime, v_prime)``; ``U_bar``/``V_bar`` are local to the reduction.
    HORIZONTAL viscosity (stage 10) operates on TOTAL velocity (u, v); the
    perturbation ``u_prime``/``v_prime`` feed vertical momentum advection
    (stage 8) and the explicit ``A_v`` viscous block (stage 10).
    """
    # --- 4. Vertical velocity from FV flux divergence ---
    # Divergence needs face fluxes: h*u at u-points, h*v at v-points.
    # Uses FULL velocity (barotropic + baroclinic) for mass transport.
    # Min-rule (MOM6/MITgcm hFacW = min(hFacC_L, hFacC_R) convention,
    # Adcroft-Hill-Marshall 1997 eq. 11-13): the face's effective wet
    # thickness equals the shallower side's thickness.  For full cells
    # with same h, this reduces to the cell value (bit-exact unchanged
    # backwards-compat).  Consistency: same convention is used in the
    # barotropic Helmholtz solver, the slow-forcing depth-average, and
    # the tracer mass flux — required for the H&A 2009 column-sum
    # invariant to hold.
    h_u = min_cell_to_uface(h_k)
    h_v = min_cell_to_vface(h_k, grid)
    flux_div_k = divergence_cgrid(
        h_u * u * u_mask_3d, h_v * v * v_mask_3d, grid,
    )
    w = _diagnose_w_from_flux_div(flux_div_k, z_coord, thickness_weighted=True)

    # --- 4b. Depth-mean velocity (for diagnostics / KE gradient) ---
    # Fuse num/denom reductions per face — both share their h_u/h_v
    # weight on the level axis.
    _u_pair = jnp.sum(jnp.stack([u * h_u, h_u], axis=-1), axis=-2)
    U_bar = _u_pair[..., 0] / jnp.maximum(_u_pair[..., 1], 1e-10) * u_mask  # (n_lat, n_lon+1)
    _v_pair = jnp.sum(jnp.stack([v * h_v, h_v], axis=-1), axis=-2)
    V_bar = _v_pair[..., 0] / jnp.maximum(_v_pair[..., 1], 1e-10) * v_mask  # (n_lat+1, n_lon)
    u_prime = u - U_bar[..., jnp.newaxis]
    v_prime = v - V_bar[..., jnp.newaxis]
    return h_u, h_v, flux_div_k, w, u_prime, v_prime


def _bc_ke_and_pressure_gradients(
    u, v, p_prime_filled, rho_prime, grid, config, z_coord,
    eta_safe, H_bathy, g_val, mask,
):
    """Stages 6 / 6-7 / 6b: kinetic-energy gradient (centered, Hollingsworth,
    or WENO), baroclinic pressure gradient (batched with KE), and the
    Adcroft-Campin / SMC03 partial-cell PGF face correction.

    Pure verbatim extraction (Q8). Returns ``(dKE_dx, dp_dx, dKE_dy,
    dp_dy)``."""
    # --- 6. Kinetic energy gradient (from TOTAL velocity, #160) ---
    # MOM6-style: KE from total u, not perturbation u'. The depth-mean
    # contribution enters F_slow for the barotropic solver; the
    # perturbation is applied to 3D velocity in the step function.
    #
    # WENO K-term (Silvestri et al. 2024 Eq. 33): the substitution
    # ``<δ_i u²>_i ↦ {δ_i u²; <u>_i}_i`` applies ONLY to the matching-
    # direction component of the KE gradient.  For ∂K/∂x at u-faces:
    #   - u² part: WENO upwind reconstruction of cell-centered δ_i u²
    #     (small magnitude, smooth) back to u-faces with <u>_i smoothness
    #   - v² part: stays centered (the cross-direction term).
    # Symmetrically for ∂K/∂y at v-faces.  Reconstructing cell-centered
    # *differences* (small) is fundamentally less dissipative than
    # reconstructing u² (large): keeps WENO weights closer to optimal
    # in smooth regions, matching paper's design intent.
    _mom_adv = config.momentum_advection
    n_lat_g, n_lon_g, nlev_g = p_prime_filled.shape
    if _mom_adv in ("weno5", "weno7"):
        # u² at faces and cell-centered fields needed by Eq. 33.
        u_sq_face = u ** 2                                  # (n_lat, n_lon+1, nlev)
        v_sq_face = v ** 2                                  # (n_lat+1, n_lon, nlev)
        # δ_i u² at cell j = u²[face j+1] - u²[face j]      (small in smooth regions)
        delta_u_sq_cell = u_sq_face[:, 1:, :] - u_sq_face[:, :-1, :]   # (n_lat, n_lon, nlev)
        # δ_j v² at cell i = v²[face i+1] - v²[face i]
        delta_v_sq_cell = v_sq_face[1:, :, :] - v_sq_face[:-1, :, :]   # (n_lat, n_lon, nlev)
        # <u>_i, <v>_j at cells (smoothness fields per Eq. 41-42).
        u_avg_cell = 0.5 * (u[:, :-1, :] + u[:, 1:, :])               # (n_lat, n_lon, nlev)
        v_avg_cell = 0.5 * (v[:-1, :, :] + v[1:, :, :])               # (n_lat, n_lon, nlev)
        # <u²>_i, <v²>_j at cells (centered, used for cross terms in K).
        u_sq_cell_centered = 0.5 * (u_sq_face[:, :-1, :] + u_sq_face[:, 1:, :])
        v_sq_cell_centered = 0.5 * (v_sq_face[:-1, :, :] + v_sq_face[1:, :, :])

        # Fill land cells before WENO stencils so masked zeros don't
        # create false discontinuities near walls (Neumann extrapolation).
        delta_u_sq_cell = _neumann_fill_cgrid(delta_u_sq_cell, mask, grid=grid)
        u_avg_cell = _neumann_fill_cgrid(u_avg_cell, mask, grid=grid)
        delta_v_sq_cell = _neumann_fill_cgrid(delta_v_sq_cell, mask, grid=grid)
        v_avg_cell = _neumann_fill_cgrid(v_avg_cell, mask, grid=grid)

        # WENO upwind of δ_i u² to u-faces (gradient times dx_u_at_face).
        delta_u_sq_at_uface = _weno_cell_to_uface(
            delta_u_sq_cell, u_avg_cell, u, order=5)        # (n_lat, n_lon+1, nlev)
        # WENO upwind of δ_j v² to v-faces.
        delta_v_sq_at_vface = _weno_cell_to_vface(
            delta_v_sq_cell, v_avg_cell, v, order=5)        # (n_lat+1, n_lon, nlev)

        # Convert "δ across one cell" → "gradient at face" by dividing
        # by dx_u (cell width at u-face latitude) and dy_v (distance
        # between adjacent cell-centre latitudes at the v-face row).
        if is_tripolar(grid):
            # Tripolar: full 2D metrics.
            dx_u_at_face = grid.dx_u[:, :, jnp.newaxis]      # (n_lat, n_lon+1, 1)
            dy_v = grid.dy_v[1, 0]                            # scalar (interior row)
        else:
            # Regular or Mercator: variable-dy safe.
            R = grid.radius
            dx_u_at_face = (R * grid.dlon * grid.cos_lat)[:, jnp.newaxis, jnp.newaxis]  # (n_lat,1,1)
            dy_h_arr = grid.dy * 0.5                                          # (n_lat,)
            dy_v_int = 0.5 * (dy_h_arr[1:] + dy_h_arr[:-1])                    # (n_lat-1,)
            dy_v = jnp.pad(dy_v_int, (1, 1), mode='edge')[:, jnp.newaxis, jnp.newaxis]  # (n_lat+1,1,1)
        # Gradient of <u²>_i at u-face (WENO upwind version).
        dKE_u2_dx_at_uface = 0.5 * delta_u_sq_at_uface / dx_u_at_face
        # Gradient of <v²>_j at v-face (WENO upwind version).
        dKE_v2_dy_at_vface = 0.5 * delta_v_sq_at_vface / dy_v

        # Centered cross terms (v² for x-gradient, u² for y-gradient)
        # batched with pressure gradient: stack each cross-term with
        # p_prime_filled along trailing axis, fold into level dim, and
        # call each gradient operator once.  4 calls → 2.
        _vp_x_stack = jnp.stack([v_sq_cell_centered, p_prime_filled], axis=-1)
        _vp_x_flat = _vp_x_stack.reshape(n_lat_g, n_lon_g, nlev_g * 2)
        _dvp_dx_flat = gradient_x_cgrid(_vp_x_flat, grid)   # (n_lat, n_lon+1, nlev*2)
        _dvp_dx = _dvp_dx_flat.reshape(
            _dvp_dx_flat.shape[0], _dvp_dx_flat.shape[1], nlev_g, 2,
        )
        dvsq_dx = _dvp_dx[..., 0]                           # (n_lat, n_lon+1, nlev)
        dp_dx = _dvp_dx[..., 1]                              # (n_lat, n_lon+1, nlev)

        _up_y_stack = jnp.stack([u_sq_cell_centered, p_prime_filled], axis=-1)
        _up_y_flat = _up_y_stack.reshape(n_lat_g, n_lon_g, nlev_g * 2)
        _dup_dy_flat = gradient_y_cgrid(_up_y_flat, grid)    # (n_lat+1, n_lon, nlev*2)
        _dup_dy = _dup_dy_flat.reshape(
            _dup_dy_flat.shape[0], _dup_dy_flat.shape[1], nlev_g, 2,
        )
        dusq_dy = _dup_dy[..., 0]                            # (n_lat+1, n_lon, nlev)
        dp_dy = _dup_dy[..., 1]                               # (n_lat+1, n_lon, nlev)

        # Assemble: K_u = 0.5 * (WENO_upwind ∂_x <u²>_i + centered ∂_x <v²>_j)
        dKE_dx = dKE_u2_dx_at_uface + 0.5 * dvsq_dx
        # K_v = 0.5 * (centered ∂_y <u²>_i + WENO_upwind ∂_y <v²>_j)
        dKE_dy = 0.5 * dusq_dy + dKE_v2_dy_at_vface
    else:
        # --- 6/7. KE + pressure gradients (batched) ---
        #
        # The KE form is selected by ``config.ke_gradient_scheme``:
        #
        # ``"centered"`` (default, legacy):
        #   KE = 0.5 * ((<u>_i)² + (<v>_j)²)
        #   Standard C-grid centered KE. Has the Hollingsworth-Kallberg
        #   instability over stratified flow on sloping bathymetry.
        #
        # ``"hollingsworth"`` (NEMO ``nkeg_HW``, Hollingsworth-Kållberg-
        # Renner 1983 / Arakawa-Hsu 1990):
        #   K(i,j) = ( zu + zv ) / 48
        #     zu = 8*(u(i-1,j)² + u(i,j)²)
        #        + (u(i-1,j-1)+u(i-1,j+1))²
        #        + (u(i,  j-1)+u(i,  j+1))²
        #     zv = 8*(v(i,j-1)² + v(i,j)²)
        #        + (v(i-1,j-1)+v(i+1,j-1))²
        #        + (v(i-1,j)  +v(i+1,j))²
        #   3-row stencil widens the K computation, removing spurious
        #   vortex stretching from the geopotential coordinate's KE
        #   gradient near sloping bathymetry. Required for stable
        #   stratified flow on Mercator grids over realistic topography.
        if config.ke_gradient_scheme == "hollingsworth":
            # u shape (n_lat, n_lon+1, nlev) — u(i-1,j) = u[:, :-1, :], u(i,j) = u[:, 1:, :]
            u_l = u[:, :-1, :]                                  # (n_lat, n_lon, nlev)
            u_r = u[:, 1:, :]
            # j±1 with Neumann (edge) BC at south/north walls
            u_l_jm1 = jnp.concatenate([u_l[:1], u_l[:-1]], axis=0)
            u_l_jp1 = jnp.concatenate([u_l[1:], u_l[-1:]], axis=0)
            u_r_jm1 = jnp.concatenate([u_r[:1], u_r[:-1]], axis=0)
            u_r_jp1 = jnp.concatenate([u_r[1:], u_r[-1:]], axis=0)
            # v shape (n_lat+1, n_lon, nlev) — v(i,j-1) = v[:-1, :, :], v(i,j) = v[1:, :, :]
            v_s = v[:-1, :, :]                                  # (n_lat, n_lon, nlev)
            v_n = v[1:, :, :]
            # i±1 periodic in longitude (jnp.roll matches existing convention)
            v_s_im1 = jnp.roll(v_s, shift=+1, axis=1)
            v_s_ip1 = jnp.roll(v_s, shift=-1, axis=1)
            v_n_im1 = jnp.roll(v_n, shift=+1, axis=1)
            v_n_ip1 = jnp.roll(v_n, shift=-1, axis=1)
            zu = 8.0 * (u_l ** 2 + u_r ** 2) \
                 + (u_l_jm1 + u_l_jp1) ** 2 \
                 + (u_r_jm1 + u_r_jp1) ** 2
            zv = 8.0 * (v_s ** 2 + v_n ** 2) \
                 + (v_s_im1 + v_s_ip1) ** 2 \
                 + (v_n_im1 + v_n_ip1) ** 2
            KE = (zu + zv) / 48.0
        elif config.ke_gradient_scheme == "centered":
            u_cell = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
            v_cell = 0.5 * (v[:-1, :, :] + v[1:, :, :])
            KE = 0.5 * (u_cell ** 2 + v_cell ** 2)
        else:
            raise ValueError(
                f"Unknown ke_gradient_scheme: {config.ke_gradient_scheme!r}. "
                f"Must be 'centered' or 'hollingsworth'."
            )
        _Kp_stack = jnp.stack([KE, p_prime_filled], axis=-1)
        _Kp_flat = _Kp_stack.reshape(n_lat_g, n_lon_g, nlev_g * 2)
        _dKp_dx_flat = gradient_x_cgrid(_Kp_flat, grid)  # (n_lat, n_lon+1, nlev*2)
        _dKp_dy_flat = gradient_y_cgrid(_Kp_flat, grid)  # (n_lat+1, n_lon, nlev*2)
        _dKp_dx = _dKp_dx_flat.reshape(
            _dKp_dx_flat.shape[0], _dKp_dx_flat.shape[1], nlev_g, 2,
        )
        _dKp_dy = _dKp_dy_flat.reshape(
            _dKp_dy_flat.shape[0], _dKp_dy_flat.shape[1], nlev_g, 2,
        )
        dKE_dx = _dKp_dx[..., 0]
        dp_dx = _dKp_dx[..., 1]
        dKE_dy = _dKp_dy[..., 0]
        dp_dy = _dKp_dy[..., 1]

    # --- 6b. Adcroft-Campin partial-cell PGF face correction ---
    # When using OceanPartialCellCoordinate, the partial bottom cells
    # at one column are at a shallower geometric depth than the same
    # level k at a deeper-bathymetry neighbour.  The standard
    # gradient_*_cgrid compares pressures at different depths,
    # producing a residual PGF error that drives spurious flow.
    #
    # Adcroft & Campin (2004) shift each cell's pressure to a common
    # face-reference depth (the shallower of the two centroids) before
    # differencing.  Implemented here as an additive correction to
    # dp_dx, dp_dy.  For pure z\\* coord (legacy), all centroids align
    # within a column so the correction is identically zero — bit-exact
    # backwards-compat preserved.
    if isinstance(z_coord, OceanPartialCellCoordinate):
        pgf_scheme = getattr(config, "pgf_scheme", "adcroft")
        if pgf_scheme == "smc03":
            # Replace (centered-diff p_prime gradient) + (Adcroft face
            # correction) with the density-Jacobian PGF evaluated at a
            # smooth-in-k face-reference depth.  Same ``rho_prime`` and
            # eta=0 reference as the Adcroft path, so AD pytree shape
            # is unchanged.
            dp_dx_smc = density_jacobian_pgf_smc03_x(
                rho_prime, z_coord.h_partial, z_coord.is_active,
                grid, g_val,
            )
            dp_dy_smc = density_jacobian_pgf_smc03_y(
                rho_prime, z_coord.h_partial, z_coord.is_active,
                grid, g_val,
            )
            # Match dtype to the existing dp_dx/dp_dy (which inherit
            # from p_prime — float32 in the standard config).
            dp_dx = dp_dx_smc.astype(dp_dx.dtype)
            dp_dy = dp_dy_smc.astype(dp_dy.dtype)
        else:
            # Use eta=0 reference for centroid: rho_prime / p_prime above
            # are computed at the J=1, eta=0 reference (line 802 comment).
            # Using live eta here would make the Adcroft correction time-
            # dependent through eta — small effect at rest (eta=0) but
            # breaks the "rest-state machine-zero" claim once eta evolves.
            centroid_depth = compute_centroid_depth(
                jnp.zeros_like(eta_safe), H_bathy, z_coord,
            )
            dp_dx = dp_dx + partial_cell_pgf_correction_x(
                centroid_depth, rho_prime, grid, g_val,
            )
            dp_dy = dp_dy + partial_cell_pgf_correction_y(
                centroid_depth, rho_prime, grid, g_val,
            )

    return dKE_dx, dp_dx, dKE_dy, dp_dy


def _bc_tracer_tendencies(T, S, config, grid, mask, J, z_coord):
    """Stage 9: horizontal (Laplacian / biharmonic) + explicit vertical tracer
    diffusion for T and S. Pure verbatim extraction (Q8). Returns ``(dT_dt,
    dS_dt)`` (advection + physics are added by the caller / step function)."""
    # --- 9. Tracer tendencies (diffusion + physics only) ---
    # Horizontal AND vertical tracer advection are handled in the step()
    # function using barotropic-averaged transport (Hallberg 1997, #102).
    # Vertical velocity w is diagnosed from the barotropic-averaged
    # per-layer divergence, ensuring 3D transport consistency.
    #
    # The tendency here includes only: diffusion and physics.
    # Stack T, S along a trailing tracer axis and fold it into the level
    # axis so ``laplacian_cgrid`` (and ``bilaplacian_cgrid`` which is two
    # laplacian calls) runs ONCE on the thicker
    # ``(n_lat, n_lon, nlev*2)`` field — the prior vmap-over-(T,S)
    # pattern issued separate halo pads + 5-point stencils per tracer.
    # Vertical diffusion stays per-tracer because it hard-codes the
    # vertical axis at -1.
    tracer_stack = jnp.stack([T, S], axis=-1)  # (n_lat, n_lon, nlev, 2)
    n_lat_t, n_lon_t, nlev_t, n_tracers = tracer_stack.shape
    tracer_flat = tracer_stack.reshape(n_lat_t, n_lon_t, nlev_t * n_tracers)

    horiz_flat = jnp.zeros_like(tracer_flat)
    if config.K_h > 0 and config.K_bih > 0:
        # Both Laplacian and biharmonic active: bilaplacian's *inner*
        # ∇² is identical to the K_h Laplacian, so compute ∇²(tracer_flat)
        # ONCE and feed it to both branches.  Saves one full
        # laplacian_cgrid call (2 gradients + 1 divergence + masking)
        # per RHS evaluation.
        _lap_tr = laplacian_cgrid(tracer_flat, grid, mask=mask)
        horiz_flat = horiz_flat + config.K_h * _lap_tr
        horiz_flat = horiz_flat - config.K_bih * laplacian_cgrid(
            _lap_tr, grid, mask=mask,
        )
    elif config.K_h > 0:
        horiz_flat = horiz_flat + config.K_h * laplacian_cgrid(
            tracer_flat, grid, mask=mask,
        )
    elif config.K_bih > 0:
        horiz_flat = horiz_flat - config.K_bih * bilaplacian_cgrid(
            tracer_flat, grid, mask=mask,
        )
    horiz_stack = horiz_flat.reshape(n_lat_t, n_lon_t, nlev_t, n_tracers)

    # Vertical tracer diffusion (per-tracer; axis -1 of ``tr`` is nlev).
    # Always applied regardless of physics pipeline state — the physics
    # pipeline's vertical_mixing module is a separate concept (e.g.,
    # KPP).  Baseline K_v diffusion should always be active when K_v > 0.
    # (Fixes #150.)
    #
    # Skipped when ``implicit_vertical_mixing`` is enabled — the
    # K_v floor is folded into the implicit K profile in the model step.
    if (config.K_v > 0 and nlev_t >= 2
            and not getattr(config, "implicit_vertical_mixing", False)):
        jac_v = jnp.maximum(J[..., jnp.newaxis], 1e-10)  # (n_lat, n_lon, 1)
        dz_actual_loc = z_coord.dz_ref * jac_v           # (n_lat, n_lon, nlev)

        def _vdiff(tr):
            dtr_dz_half = (tr[..., :-1] - tr[..., 1:]) / (
                z_coord.dz_half_ref * jac_v
            )
            flux = config.K_v * dtr_dz_half
            _pad_axes_tr = ((0, 0),) * (flux.ndim - 1)
            flux_full = jnp.pad(flux, (*_pad_axes_tr, (1, 1)))
            return (flux_full[..., :-1] - flux_full[..., 1:]) / dz_actual_loc

        vdiff_stack = jax.vmap(_vdiff, in_axes=-1, out_axes=-1)(tracer_stack)
        tracer_tend_stack = horiz_stack + vdiff_stack
    else:
        tracer_tend_stack = horiz_stack

    dT_dt = tracer_tend_stack[..., 0]
    dS_dt = tracer_tend_stack[..., 1]
    return dT_dt, dS_dt


def _bc_pv_flux(
    du_dt, dv_dt, u, v, h_u, h_v, h_k, u_mask_3d, v_mask_3d, mask, grid, _mom_adv,
):
    """Stage 7b: vector-invariant potential-vorticity (vorticity) flux
    (Sadourny EC / Arakawa-Lamb-81 triad, or WENO-Z when momentum_advection is
    weno5/weno7). Pure verbatim extraction (Q8). Threads the momentum
    accumulators; returns ``(du_dt, dv_dt, diag_vortcor_u, diag_vortcor_v)``."""
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

    # Layer thickness at vertices.  Use the min-rule (Adcroft-Hill-
    # Marshall 1997 / Pacanowski-Gnanadesikan 1998 / MITgcm convention)
    # rather than a 4-cell arithmetic mean, so the PV thickness is
    # consistent with the face-flux thickness ``h_u = min(h_W, h_E)``
    # used in the same momentum equation.  At a step vertex where the
    # 4 surrounding cells have different ``bot_level``, the
    # circulation integral that defines ζ at that vertex has an
    # effective wet thickness bounded by ``min(h_4)`` (you can only
    # circulate through wet faces).  Dividing ζ by a 4-cell *mean*
    # overstates the wet thickness by 2-17× at step vertices on real
    # ETOPO bathymetry, undervaluing the potential vorticity
    # ``q = ζ / h_vtx`` and the Coriolis advection of momentum it
    # drives — a known root cause of slow rest-state drift on
    # stratified seamounts (Pacanowski & Gnanadesikan 1998).
    # ``min`` must be taken over ACTIVE cells only — inactive cells
    # (below the partial seafloor) have ``h = 0`` and including them
    # in the raw min sends ``h_vtx → 0`` at every step vertex with
    # any inactive neighbour, then ``q = ζ / h_vtx`` blows up.  We
    # mask inactive cells with ``+∞`` (large but not so large as to
    # overflow when later combined with eps protection); then min
    # picks the smallest *active* thickness.  If all 4 surrounding
    # cells are inactive (deep vertex below all seafloors), the
    # final result is ``+∞`` and the downstream ``maximum(h_vtx,
    # 1e-10)`` floor doesn't matter because the surrounding face
    # masks already zero ``ζ`` there.  Mirrors MITgcm's
    # ``hFacZ = min over active hFacC`` convention.
    BIG_H = 1.0e30
    h_sw = jnp.roll(h_k, 1, axis=1)
    h_k_active = jnp.where(h_k > 0.0, h_k, BIG_H)
    h_sw_active = jnp.where(h_sw > 0.0, h_sw, BIG_H)
    # Cell-pad-first (PR357 Bug-2 pattern): pad the cell active-thickness
    # over latitude so the vertex min at a partition cut includes the
    # neighbour rank's adjacent T row (MPI halo exchange).  The pole pad
    # value is BIG_H so a physical-pole vertex reduces to the local two-cell
    # min (bit-identical to the previous boundary rows); interior partition
    # cuts sendrecv the neighbour's real row instead.
    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat
    h_k_pad = pad_with_pole_bc_lat(
        h_k_active, halo=1, south_value=BIG_H, north_value=BIG_H,
    )
    h_sw_pad = pad_with_pole_bc_lat(
        h_sw_active, halo=1, south_value=BIG_H, north_value=BIG_H,
    )
    h_vtx = jnp.minimum(
        jnp.minimum(h_k_pad[:-1], h_k_pad[1:]),
        jnp.minimum(h_sw_pad[:-1], h_sw_pad[1:]),
    )                                                  # (n_lat+1, n_lon, nlev)
    # At the fold, the vertex connects 4 cells: two local (fold row) and two
    # fold-partner cells.  Overwrite the north row only on the owning rank.
    if _fold_is_local(grid):
        fold = grid.fold
        h_k_partner = h_k_active[-1:, fold.perm_T, :]
        h_sw_partner = h_sw_active[-1:, fold.perm_T, :]
        h_vtx_north = jnp.minimum(
            jnp.minimum(h_k_active[-1:], h_sw_active[-1:]),
            jnp.minimum(h_k_partner, h_sw_partner),
        )
        h_vtx = jnp.concatenate([h_vtx[:-1], h_vtx_north], axis=0)
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

    # Average Fu to v-points (4-point; halo-aware over latitude, fold-
    # reflected with u-sign at the north seam on the owning rank).
    # Cell-pad-first / consistent MPI call count (PR357 review): every rank
    # calls pad_ns_zero (halo at partition cuts, zero at the physical pole);
    # only the rank owning the fold overwrites the north row with the
    # u-sign-flipped fold partner.  The previous fold branch called
    # pad_ns_vector_u (an MPI pad) while the else branch did a plain concat,
    # which deadlocked across ranks under the fold-active-everywhere invariant.
    from legoesm.ocean.dynamics.latlon_cgrid_operators import _fold_row
    fold_local = _fold_is_local(grid)
    Fu_ext = pad_ns_zero(Fu)   # (n_lat+2, n_lon+1, nlev)
    if fold_local:
        _f = grid.fold
        Fu_fold_row = _fold_row(
            Fu[-1:], _f.perm_T, _f.vector_sign_u, _f.perm_T.shape[0])
        Fu_ext = jnp.concatenate([Fu_ext[:-1], Fu_fold_row], axis=0)
    Fu_at_v = 0.25 * (Fu_ext[:-1, :-1, :] + Fu_ext[:-1, 1:, :]
                       + Fu_ext[1:, :-1, :] + Fu_ext[1:, 1:, :])  # (n_lat+1, n_lon, nlev)

    # Total velocity at u/v faces (for WENO upwinding direction)
    v_west_total = jnp.roll(v, 1, axis=1)
    v_at_u_core = 0.25 * (v[:-1] + v[1:]
                           + v_west_total[:-1] + v_west_total[1:])
    v_at_u = jnp.concatenate(
        [v_at_u_core, v_at_u_core[:, 0:1, :]], axis=1,
    )  # (n_lat, n_lon+1, nlev)

    # Average total u to v-points (4-point average; fold-reflected at north
    # on tripolar).  Uses total velocity for consistency with total-velocity
    # Sadourny EC PV flux and WENO upwinding (#160).
    u_ext = pad_ns_zero(u)   # halo at cuts, zero at the physical pole
    if fold_local:
        _f = grid.fold
        u_fold_row = _fold_row(
            u[-1:], _f.perm_T, _f.vector_sign_u, _f.perm_T.shape[0])
        u_ext = jnp.concatenate([u_ext[:-1], u_fold_row], axis=0)
    u_at_v = 0.25 * (u_ext[:-1, :-1, :] + u_ext[:-1, 1:, :]
                      + u_ext[1:, :-1, :] + u_ext[1:, 1:, :])  # (n_lat+1, n_lon, nlev)

    # Reconstruct q-flux at velocity points.
    #
    # WENO5/7: WENO-Z reconstruction of q at u/v-faces, multiplied by
    # the centred mass flux average.  Used when the caller explicitly
    # opts into ``momentum_advection="weno5"|"weno7"``.
    #
    # Default ("vector_invariant"): production-grade Arakawa-Lamb 1981
    # 12-point triad PV flux (``pv_flux_al81_partial_cell``).  Uses a
    # 9-vertex stencil around each face with weights chosen so that
    # discrete energy AND discrete potential enstrophy are both
    # conserved on partial-cell topography (Le Sommer et al. 2009;
    # Stewart & Dellar 2016).  Replaces the simple 2-point Sadourny
    # enstrophy form ``q_at_u = ½(q_S + q_N)`` which generates a grid-
    # scale q-noise mode at step vertices that drives a day-19 NaN on
    # real ETOPO under live-T integration.  Same Neumann fill of q at
    # land-adjacent vertices as WENO5 (for the centred-q part of the
    # triad).
    if _mom_adv in ("weno5", "weno7"):
        _weno_order = {"weno5": 5, "weno7": 7}[_mom_adv]
        # Fill PV at land-adjacent vertices so WENO stencils see smooth
        # Neumann extrapolation instead of masked-zero discontinuities.
        vtx_mask = _compute_vertex_mask(mask, grid=grid)
        q_filled = _neumann_fill_vertex(q, vtx_mask)
        q_at_u = _weno_zeta_at_u(
            q_filled, v, v_at_u, order=_weno_order, u_smooth=u)
        q_at_v = _weno_zeta_at_v(
            q_filled, u, u_at_v, order=_weno_order, v_smooth=v)
        diag_vortcor_u = q_at_u * Fv_at_u
        diag_vortcor_v = -(q_at_v * Fu_at_v)
    else:
        vtx_mask_va = _compute_vertex_mask(mask, grid=grid)
        diag_vortcor_u, diag_vortcor_v = pv_flux_al81_partial_cell(
            zeta, h_vtx, h_v, v, h_u, u,
            u_mask_3d, v_mask_3d, vtx_mask_va,
        )

    # Capture PV-flux advection contribution as the `vortcor` diagnostic
    # (per-step closure: Σ components == total to machine precision; see
    # `tests/ocean/unit/test_momentum_diagnostics_closure.py`).
    du_dt = du_dt + diag_vortcor_u
    dv_dt = dv_dt + diag_vortcor_v
    return du_dt, dv_dt, diag_vortcor_u, diag_vortcor_v


def _bc_dterm(du_dt, dv_dt, u, v, u_mask_3d, v_mask_3d, mask, grid, config, _mom_adv):
    """Stage 7c: WENO divergence (D-term) momentum dissipation (Silvestri et al.
    2024 Eqs. 31-32). Active only for weno5/weno7 momentum advection with
    config.weno_d_term; otherwise the diagnostics are zero. Pure verbatim
    extraction (Q8). Returns ``(du_dt, dv_dt, diag_Dterm_u, diag_Dterm_v)``."""
    diag_Dterm_u = jnp.zeros_like(du_dt)
    diag_Dterm_v = jnp.zeros_like(dv_dt)
    # --- 7c. Divergence flux (D term, Silvestri et al. 2024 Eqs. 31-32) ---
    # The two components of ∇·u are treated asymmetrically:
    #
    #   {D}_at_u = {δ_i U; δ_i U}_i  +  ⟨δ_j V⟩_i
    #              \________WENO____/    \__centered__/
    #              matching direction    cross direction
    #
    # The cross-direction component must stay centered: WENO-upwinding
    # the FULL divergence (which embeds δ_j V at u-faces) produces a
    # non-energy-dissipative term `u|u|·δ_i δ_j V` that injects energy
    # at the grid scale (Silvestri Appendix C, Eq. C7).  Symmetric for
    # D_at_v.  Sign per Silvestri Eq. 25 evolution form: ``du/dt -= D·u``
    # (D=∇·u, contribution from flux-form decomposition).
    #
    # Gated by config.weno_d_term so the effect can be isolated; default
    # True (paper-faithful) once the split is in place.
    if _mom_adv in ("weno5", "weno7") and config.weno_d_term:
        dU_di_cell, dV_dj_cell = _split_velocity_divergence(
            u * u_mask_3d, v * v_mask_3d, grid)
        # Fill land cells before WENO stencils (Neumann extrapolation).
        dU_di_filled = _neumann_fill_cgrid(dU_di_cell, mask, grid=grid)
        dV_dj_filled = _neumann_fill_cgrid(dV_dj_cell, mask, grid=grid)
        # Matching direction (WENO upwind), cross direction (centered).
        D_at_u = (_weno_cell_to_uface(dU_di_filled, dU_di_filled, u, order=5)
                  + _centered_cell_to_uface(dV_dj_cell))
        D_at_v = (_centered_cell_to_vface(dU_di_cell)
                  + _weno_cell_to_vface(dV_dj_filled, dV_dj_filled, v, order=5))
        diag_Dterm_u = -(D_at_u * u * u_mask_3d)
        diag_Dterm_v = -(D_at_v * v * v_mask_3d)
        du_dt = du_dt + diag_Dterm_u
        dv_dt = dv_dt + diag_Dterm_v
    return du_dt, dv_dt, diag_Dterm_u, diag_Dterm_v


def _bc_vertical_momentum_advection(
    du_dt, dv_dt, u_prime, v_prime, w, h_u, h_v, u_mask_3d, v_mask_3d,
    grid, _mom_adv, _weno_order, config, diagnose_momentum=False,
):
    """Stage 8: flux-form vertical advection of the perturbation momentum
    (1st-order upwind, or WENO when momentum_advection is weno5/weno7). Pure
    verbatim extraction (Q8). Returns ``(du_dt, dv_dt, diag_vertadv_u,
    diag_vertadv_v)``.

    The ``adaptive_implicit_vertadv`` gate (Shchepetkin 2015 / NEMO
    ln_zad_Aimp) is applied here: when ``config.adaptive_implicit_vertadv``
    is True the explicit vertadv tendency is NOT added to ``du_dt`` (it is
    applied as a separate operator-split stage at the step level on the
    barotropic-consistent ``w``); it is still returned as the start-of-step
    diagnostic estimate when ``diagnose_momentum`` is True."""
    # --- 8. Vertical advection of u, v (perturbation velocity) ---
    # Issue #171 Level-1 fix: use interface-upwind flux-form momentum
    # advection instead of the cell-centered upwind gradient form.
    # The flux-form helper returns -(F_top - F_bot) / h_u with
    # F = w_half * u_upwind_at_interface and F = 0 at top/bottom by
    # construction, eliminating the hard-zero gradient pathology at
    # k=0 / k=nlev-1 and matching the tracer-path interface upwind.
    # Full flux-form momentum update (Level 2) still requires step-
    # function restructuring; tracked on #171.
    # Use the partial-cell-aware face thicknesses already computed for
    # the mass-flux divergence (lines 832-833).  For pure z* these
    # equal dz_ref * J_u / J_v; for partial cells, h_k is zero below
    # the seafloor so divisions inside the flux-form vertical advection
    # do not pull thickness from inactive levels.
    #
    # Adaptive-implicit gate (Shchepetkin 2015 / NEMO ln_zad_Aimp): when
    # ``config.adaptive_implicit_vertadv`` is set, the explicit vertical
    # momentum advection here is NOT added to ``du_dt`` (the slow
    # baroclinic forcing fed to the barotropic solver must stay
    # vertical-advection-free, because vertadv is applied as a separate
    # operator-split stage at the step level
    # ``ocean_model_latlon_cgrid._step_impl`` on the barotropic-consistent
    # ``w``).  The explicit flux-form tendency is STILL computed and
    # stored in ``diag_vertadv_{u,v}`` as the start-of-step estimate of
    # the (otherwise implicit) vertical-advection term, so the momentum
    # budget reports the term being stabilised rather than a silent zero
    # (it matches the O(dt) start-of-step semantics already documented on
    # ``tendencies_with_diagnostics``).
    #
    # Closure contract:
    #   - flag OFF: ``vertadv`` is in ``du_dt``     -> Σ(terms) == du_dt.
    #   - flag ON : ``vertadv`` is a diagnostic only -> Σ(terms) ==
    #     du_dt + diag_vertadv (the vertadv slice is the start-of-step
    #     estimate of the step-level implicit operator, excluded from the
    #     slow forcing on purpose).  ``test_momentum_diagnostics_closure``
    #     runs with the default (flag off) config, so the strict
    #     ``Σ == du_dt`` identity it enforces is unaffected.
    _aimp_vertadv = getattr(config, "adaptive_implicit_vertadv", False)
    # Default the diagnostics to zero so the (Q8-decomposed) helper's
    # ``(du_dt, dv_dt, diag_vertadv_u, diag_vertadv_v)`` return contract
    # holds even on the flag-on / not-diagnosing fast path where the
    # explicit tendency is never built.
    diag_vertadv_u = jnp.zeros_like(du_dt)
    diag_vertadv_v = jnp.zeros_like(dv_dt)
    # Compute the explicit flux-form vertadv tendency when it is either
    # (a) part of the slow forcing (flag off), or (b) needed for the
    # momentum budget as the start-of-step estimate (flag on AND
    # diagnosing).  When the flag is on and we are not diagnosing, skip it
    # entirely so the hot step path adds no extra graph nodes (the actual
    # vertical advection is applied at the step level instead).  These are
    # Python ``if`` on static (compile-time) flags -- the feature-gating
    # exception, not ``jnp.where``.
    if (not _aimp_vertadv) or diagnose_momentum:
        h_u_old = h_u
        h_v_old = h_v
        w_u = interp_cell_to_uface(w)
        w_v = _interp_to_v_points(w, grid=grid)
        if _mom_adv in ("weno5", "weno7"):
            # WENO vertical momentum advection removes the implicit
            # viscosity (~|w|*dz/2) that first-order upwind provides.
            # Requires compensating vertical viscosity (KPP / Richardson-
            # A_v, #204).
            diag_vertadv_u = _flux_form_vertical_momentum_advection_weno(
                u_prime, w_u, h_u_old, order=_weno_order)
            diag_vertadv_v = _flux_form_vertical_momentum_advection_weno(
                v_prime, w_v, h_v_old, order=_weno_order)
        else:
            # Default: 1st-order upwind.  The implicit viscosity
            # (~|w|*dz/2) damps baroclinic shear that explicit A_v=1e-5
            # cannot.  Pass u/v face-activity masks so vertical momentum
            # flux is exactly zero at faces below the seafloor —
            # otherwise float-precision noise in w_u/w_v drives spurious
            # tendencies inside the rock (and poorly-conditions
            # adjoints).  ``u_mask_3d`` may be shape ``(..., 1)`` for
            # pure z* (2D-broadcast) or ``(..., nlev)`` for partial;
            # broadcast to the velocity shape so the helper's per-level
            # slicing along the last axis works.
            u_face_active = jnp.broadcast_to(u_mask_3d, u_prime.shape)
            v_face_active = jnp.broadcast_to(v_mask_3d, v_prime.shape)
            diag_vertadv_u = _flux_form_vertical_momentum_advection(
                u_prime, w_u, h_u_old, face_active=u_face_active)
            diag_vertadv_v = _flux_form_vertical_momentum_advection(
                v_prime, w_v, h_v_old, face_active=v_face_active)
        if not _aimp_vertadv:
            # Explicit path: vertadv is part of the slow baroclinic
            # forcing.  (Flag on: it stays a diagnostic only.)
            du_dt = du_dt + diag_vertadv_u
            dv_dt = dv_dt + diag_vertadv_v
    return du_dt, dv_dt, diag_vertadv_u, diag_vertadv_v


def _bc_horizontal_viscosity(
    du_dt, dv_dt, u, v, grid, mask, u_mask, v_mask, config, z_coord, H_bathy, dt,
):
    """Stages 10 + 10b: horizontal viscosity (A_h Laplacian + B_h biharmonic +
    Smagorinsky + Leith, with cos(lat) / equatorial / polar-cap scaling and the
    slope-foot enhancement) plus the meridional-only Laplacian viscosity. Kept
    together because the meridional block reuses the slope-foot helper/fields
    defined in stage 10. Pure verbatim extraction (Q8). Returns the momentum
    accumulators plus the per-term viscosity diagnostics."""
    diag_Ah_lap_u = jnp.zeros_like(du_dt)
    diag_Ah_lap_v = jnp.zeros_like(dv_dt)
    # Cell-centre A_h coefficient field [m²/s] used to build the FAITHFUL
    # positive-definite K_diss_h EKE source (Veros analogue) — captured ONLY when
    # the prognostic-EKE ``source_kdiss_h`` + ``kdiss_h_flux_form`` options are on
    # (the ACC recipe).  It is the SAME ``A_h × scale`` (cos-power / eq-boost /
    # cap-boost / slope-foot) field applied to the u vector-Laplacian tendency, so
    # the dissipation it weights is exactly the energy that tendency removes.
    # ``None`` otherwise (default), keeping every existing path bit-identical.
    _eke_for_kdiss = (config.gm_redi.eke if config.gm_redi is not None
                      and getattr(config.gm_redi, "eke", None) is not None else None)
    _want_kdiss_flux = bool(
        _eke_for_kdiss is not None
        and getattr(_eke_for_kdiss, "source_kdiss_h", False)
        and getattr(_eke_for_kdiss, "kdiss_h_flux_form", False))
    _ah_scale_center = None   # (n_lat,) latitudinal A_h×scale at cell centres
    _slope_E_center = None    # 3-D slope-foot enhancement at cell centres (or None)
    diag_Bh_bilap_u = jnp.zeros_like(du_dt)
    diag_Bh_bilap_v = jnp.zeros_like(dv_dt)
    diag_Cs_smag_u = jnp.zeros_like(du_dt)
    diag_Cs_smag_v = jnp.zeros_like(dv_dt)
    diag_Cl_leith_u = jnp.zeros_like(du_dt)
    diag_Cl_leith_v = jnp.zeros_like(dv_dt)
    # --- 10. Mixing (viscosity on perturbation velocity) ---
    # Uses the proper vector Laplacian grad(div) - k×grad(curl) directly
    # on face velocities, avoiding the lossy cell-center detour.
    # See issue #105 for details.

    # Slope-foot enhancement (MOM6 OM4 KH_BG_2D analog): multiplicative
    # 3D factor (≥1) that boosts viscosity in the bottom-N levels over
    # steep slopes. Targets f≈0 + steep-bathymetry instabilities
    # (African shelf, ITF). When alpha=0, factor is identically 1
    # (no-op, bit-exact backward compat).
    _slope_foot_alpha = getattr(config, "slope_foot_alpha", 0.0)
    if _slope_foot_alpha > 0.0:
        from legoesm.ocean.dynamics.latlon_cgrid_operators import slope_foot_enhancement_3d
        _is_active = z_coord.is_active if isinstance(z_coord, OceanPartialCellCoordinate) else None
        _slope_E = slope_foot_enhancement_3d(
            H_bathy, mask, grid,
            n_levels_from_bottom=getattr(config, "slope_foot_n_levels", 5),
            alpha=_slope_foot_alpha,
            threshold=getattr(config, "slope_foot_threshold", 0.1),
            is_active=_is_active,
            nlev=u.shape[-1],
        )
        if _want_kdiss_flux:
            # Cell-centre slope-foot factor for the K_diss_h coefficient field.
            _slope_E_center = _slope_E
        # Interpolate cell-centred enhancement to u-faces, v-faces (min-rule
        # for safety: the more conservative neighbour wins, so the boost
        # acts on the steeper of the two adjacent columns).
        _slope_E_u = jnp.minimum(_slope_E, jnp.roll(_slope_E, 1, axis=1))
        _slope_E_u = jnp.concatenate([_slope_E_u, _slope_E_u[:, 0:1, :]], axis=1)
        _slope_E_v_int = jnp.minimum(_slope_E[:-1], _slope_E[1:])
        _slope_E_v = jnp.pad(_slope_E_v_int, ((1, 1), (0, 0), (0, 0)),
                             constant_values=1.0)
    else:
        _slope_E_u = 1.0
        _slope_E_v = 1.0

    def _apply_slope_foot(t_u, t_v):
        # No-op when slope_foot_alpha = 0 (factors are 1.0 scalars).
        return t_u * _slope_E_u, t_v * _slope_E_v

    # Lateral A_h-viscosity OPERATOR dispatch (dispatch discipline: unknown ->
    # ValueError). "flux_divergence" = Veros's component-wise harmonic friction
    # ∇·(A_h∇u); "vector_laplacian" (default) = the grad(div)−k×grad(curl) form
    # below (bit-identical to the historical path).
    _visc_op = getattr(config, "lateral_viscosity_operator", "vector_laplacian")
    if _visc_op not in ("vector_laplacian", "flux_divergence"):
        raise ValueError(
            "lateral_viscosity_operator must be 'vector_laplacian' or "
            f"'flux_divergence', got {_visc_op!r}"
        )
    _use_flux_div = _visc_op == "flux_divergence"
    _kdiss_fluxdiv_cell = None  # set by the flux-div A_h branch when _want_kdiss_flux

    if _use_flux_div and config.A_h > 0:
        # Veros component-wise harmonic friction (``flux_divergence_viscosity_cgrid``)
        # applies the cos(lat) A_h scaling INSIDE the flux (Veros
        # ``enable_hor_friction_cos_scaling`` / ``hor_friction_cosPower``).  The
        # legoESM-specific A_h boosts (eq / polar-cap) and the A_h_floor are NOT part
        # of Veros's harmonic friction, so reject those combinations rather than
        # silently ignoring them (they would change answers without effect here).
        if config.A_h_eq_boost > 1.0 or config.A_h_cap_boost > 1.0 or config.A_h_floor > 0.0:
            raise ValueError(
                "lateral_viscosity_operator='flux_divergence' (Veros harmonic "
                "friction) does not support A_h_eq_boost / A_h_cap_boost / A_h_floor "
                "(those are legoESM vector-Laplacian extensions, not part of Veros's "
                "operator). Set them to their defaults, or use "
                "lateral_viscosity_operator='vector_laplacian'."
            )
        _cos_p = config.A_h_cos_power if config.A_h_lat_scaling else 0
        diag_Ah_lap_u, diag_Ah_lap_v, _kdiss_fluxdiv_cell = (
            flux_divergence_viscosity_cgrid(
                u, v, grid, config.A_h, cos_power=_cos_p,
                mask=mask, u_mask=u_mask, v_mask=v_mask,
                want_dissipation=_want_kdiss_flux,
            )
        )
        diag_Ah_lap_u, diag_Ah_lap_v = _apply_slope_foot(diag_Ah_lap_u, diag_Ah_lap_v)
        du_dt = du_dt + diag_Ah_lap_u
        dv_dt = dv_dt + diag_Ah_lap_v
        if config.B_h > 0:
            # Biharmonic is a separate (vector-Laplacian) operator, unaffected by the
            # A_h operator choice: ∇⁴ = ∇²_vec(∇²_vec).
            bilap_u, bilap_v = vector_bilaplacian_cgrid(
                u, v, grid, mask=mask, u_mask=u_mask, v_mask=v_mask)
            if config.B_h_lat_scaling:
                scale_u, scale_v = biharmonic_scaling_factor(grid)
                diag_Bh_bilap_u = -config.B_h * scale_u[:, None, None] * bilap_u
                diag_Bh_bilap_v = -config.B_h * scale_v[:, None, None] * bilap_v
            else:
                diag_Bh_bilap_u = -config.B_h * bilap_u
                diag_Bh_bilap_v = -config.B_h * bilap_v
            diag_Bh_bilap_u, diag_Bh_bilap_v = _apply_slope_foot(
                diag_Bh_bilap_u, diag_Bh_bilap_v)
            du_dt = du_dt + diag_Bh_bilap_u
            dv_dt = dv_dt + diag_Bh_bilap_v
    elif config.A_h > 0 and config.B_h > 0:
        # Both A_h Laplacian and B_h biharmonic active: the biharmonic's
        # *inner* vector Laplacian is identical to the explicit A_h
        # vector Laplacian, so compute ∇²(u, v) ONCE and feed it to
        # both branches.  Saves one full vector_laplacian_cgrid call
        # (1 div + 1 curl + 2 gradients + 2 gradient_curl_to_*) per
        # RHS evaluation.
        # NOTE: uses total velocity u, v (not u_prime) so the depth-
        # averaged viscous tendency damps the barotropic mode via F_slow.
        _vlap_u, _vlap_v = vector_laplacian_cgrid(
            u, v, grid,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        if config.A_h_lat_scaling:
            _floor = config.A_h_floor / config.A_h if config.A_h_floor > 0 else 0.0
            lap_scale_u, lap_scale_v = laplacian_scaling_factor(
                grid, power=config.A_h_cos_power, floor=_floor)
            if config.A_h_eq_boost > 1.0:
                eb_u, eb_v = equatorial_boost_factor(
                    grid, config.A_h_eq_sigma_deg, config.A_h_eq_boost)
                lap_scale_u = lap_scale_u * eb_u
                lap_scale_v = lap_scale_v * eb_v
            if config.A_h_cap_boost > 1.0:
                cap_u, cap_v = polar_cap_boost_factor(
                    grid, config.A_h_cap_lat_deg,
                    config.A_h_cap_boost, config.A_h_cap_width_deg)
                lap_scale_u = lap_scale_u * cap_u
                lap_scale_v = lap_scale_v * cap_v
            diag_Ah_lap_u = config.A_h * lap_scale_u[:, None, None] * _vlap_u
            diag_Ah_lap_v = config.A_h * lap_scale_v[:, None, None] * _vlap_v
            if _want_kdiss_flux:
                _ah_scale_center = config.A_h * lap_scale_u
        elif config.A_h_eq_boost > 1.0 or config.A_h_cap_boost > 1.0:
            scale_u = jnp.ones((grid.lat.shape[0],), dtype=grid.lat.dtype)
            scale_v = jnp.ones((grid.lat.shape[0] + 1,), dtype=grid.lat.dtype)
            if config.A_h_eq_boost > 1.0:
                eb_u, eb_v = equatorial_boost_factor(
                    grid, config.A_h_eq_sigma_deg, config.A_h_eq_boost)
                scale_u = scale_u * eb_u
                scale_v = scale_v * eb_v
            if config.A_h_cap_boost > 1.0:
                cap_u, cap_v = polar_cap_boost_factor(
                    grid, config.A_h_cap_lat_deg,
                    config.A_h_cap_boost, config.A_h_cap_width_deg)
                scale_u = scale_u * cap_u
                scale_v = scale_v * cap_v
            diag_Ah_lap_u = config.A_h * scale_u[:, None, None] * _vlap_u
            diag_Ah_lap_v = config.A_h * scale_v[:, None, None] * _vlap_v
            if _want_kdiss_flux:
                _ah_scale_center = config.A_h * scale_u
        else:
            diag_Ah_lap_u = config.A_h * _vlap_u
            diag_Ah_lap_v = config.A_h * _vlap_v
            if _want_kdiss_flux:
                _ah_scale_center = config.A_h * jnp.ones(
                    (grid.lat.shape[0],), dtype=grid.lat.dtype)
        diag_Ah_lap_u, diag_Ah_lap_v = _apply_slope_foot(diag_Ah_lap_u, diag_Ah_lap_v)
        du_dt = du_dt + diag_Ah_lap_u
        dv_dt = dv_dt + diag_Ah_lap_v
        bilap_u, bilap_v = vector_laplacian_cgrid(
            _vlap_u, _vlap_v, grid,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        if config.B_h_lat_scaling:
            scale_u, scale_v = biharmonic_scaling_factor(grid)
            diag_Bh_bilap_u = -config.B_h * scale_u[:, None, None] * bilap_u
            diag_Bh_bilap_v = -config.B_h * scale_v[:, None, None] * bilap_v
        else:
            diag_Bh_bilap_u = -config.B_h * bilap_u
            diag_Bh_bilap_v = -config.B_h * bilap_v
        diag_Bh_bilap_u, diag_Bh_bilap_v = _apply_slope_foot(diag_Bh_bilap_u, diag_Bh_bilap_v)
        du_dt = du_dt + diag_Bh_bilap_u
        dv_dt = dv_dt + diag_Bh_bilap_v
    elif config.A_h > 0:
        vlap_u, vlap_v = vector_laplacian_cgrid(
            u, v, grid,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        if config.A_h_lat_scaling:
            _floor = config.A_h_floor / config.A_h if config.A_h_floor > 0 else 0.0
            lap_scale_u, lap_scale_v = laplacian_scaling_factor(
                grid, power=config.A_h_cos_power, floor=_floor)
            if config.A_h_eq_boost > 1.0:
                eb_u, eb_v = equatorial_boost_factor(
                    grid, config.A_h_eq_sigma_deg, config.A_h_eq_boost)
                lap_scale_u = lap_scale_u * eb_u
                lap_scale_v = lap_scale_v * eb_v
            if config.A_h_cap_boost > 1.0:
                cap_u, cap_v = polar_cap_boost_factor(
                    grid, config.A_h_cap_lat_deg,
                    config.A_h_cap_boost, config.A_h_cap_width_deg)
                lap_scale_u = lap_scale_u * cap_u
                lap_scale_v = lap_scale_v * cap_v
            diag_Ah_lap_u = config.A_h * lap_scale_u[:, None, None] * vlap_u
            diag_Ah_lap_v = config.A_h * lap_scale_v[:, None, None] * vlap_v
            if _want_kdiss_flux:
                _ah_scale_center = config.A_h * lap_scale_u
        elif config.A_h_eq_boost > 1.0 or config.A_h_cap_boost > 1.0:
            scale_u = jnp.ones((grid.lat.shape[0],), dtype=grid.lat.dtype)
            scale_v = jnp.ones((grid.lat.shape[0] + 1,), dtype=grid.lat.dtype)
            if config.A_h_eq_boost > 1.0:
                eb_u, eb_v = equatorial_boost_factor(
                    grid, config.A_h_eq_sigma_deg, config.A_h_eq_boost)
                scale_u = scale_u * eb_u
                scale_v = scale_v * eb_v
            if config.A_h_cap_boost > 1.0:
                cap_u, cap_v = polar_cap_boost_factor(
                    grid, config.A_h_cap_lat_deg,
                    config.A_h_cap_boost, config.A_h_cap_width_deg)
                scale_u = scale_u * cap_u
                scale_v = scale_v * cap_v
            diag_Ah_lap_u = config.A_h * scale_u[:, None, None] * vlap_u
            diag_Ah_lap_v = config.A_h * scale_v[:, None, None] * vlap_v
            if _want_kdiss_flux:
                _ah_scale_center = config.A_h * scale_u
        else:
            diag_Ah_lap_u = config.A_h * vlap_u
            diag_Ah_lap_v = config.A_h * vlap_v
            if _want_kdiss_flux:
                _ah_scale_center = config.A_h * jnp.ones(
                    (grid.lat.shape[0],), dtype=grid.lat.dtype)
        diag_Ah_lap_u, diag_Ah_lap_v = _apply_slope_foot(diag_Ah_lap_u, diag_Ah_lap_v)
        du_dt = du_dt + diag_Ah_lap_u
        dv_dt = dv_dt + diag_Ah_lap_v
    elif config.B_h > 0:
        bilap_u, bilap_v = vector_bilaplacian_cgrid(
            u, v, grid,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        if config.B_h_lat_scaling:
            # Scale biharmonic coefficient with (cos(lat)/cos_max)^4 to prevent
            # CFL violation near poles where dx shrinks (MOM6 convention).
            scale_u, scale_v = biharmonic_scaling_factor(grid)
            diag_Bh_bilap_u = -config.B_h * scale_u[:, None, None] * bilap_u
            diag_Bh_bilap_v = -config.B_h * scale_v[:, None, None] * bilap_v
        else:
            diag_Bh_bilap_u = -config.B_h * bilap_u
            diag_Bh_bilap_v = -config.B_h * bilap_v
        diag_Bh_bilap_u, diag_Bh_bilap_v = _apply_slope_foot(diag_Bh_bilap_u, diag_Bh_bilap_v)
        du_dt = du_dt + diag_Bh_bilap_u
        dv_dt = dv_dt + diag_Bh_bilap_v

    if config.C_smag > 0:
        smag_u, smag_v = smagorinsky_biharmonic_tendency_cgrid(
            u, v, grid, config.C_smag,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        diag_Cs_smag_u = -smag_u
        diag_Cs_smag_v = -smag_v
        diag_Cs_smag_u, diag_Cs_smag_v = _apply_slope_foot(diag_Cs_smag_u, diag_Cs_smag_v)
        du_dt = du_dt + diag_Cs_smag_u
        dv_dt = dv_dt + diag_Cs_smag_v

    if getattr(config, "C_smag_lap", 0.0) > 0:
        # Laplacian Smagorinsky: flow-adaptive viscosity via the
        # energy-stable stress-tensor operator.  A_smag = (C·dx)²·|D|
        # at both h-points (cell centers) and q-points (vertices).
        # Uses viscous_tendency_cgrid which is the exact discrete
        # adjoint of the strain rate — guarantees energy dissipation
        # for any non-negative spatially varying coefficient.
        D_T, D_S = strain_rate_cgrid(u, v, grid,
                                      mask=mask, u_mask=u_mask, v_mask=v_mask)
        A_smag_h = smagorinsky_viscosity_cgrid(
            u, v, grid, config.C_smag_lap,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        A_smag_q = smagorinsky_viscosity_q_cgrid(
            D_T, D_S, grid, config.C_smag_lap, mask=mask)
        # Per-cell viscosity ceiling on the Laplacian-Smagorinsky coefficient.
        # A_smag = (C*dx)^2 * |D| grows without bound at sharp jets (|D| large)
        # and self-CFL-violates there -- the western-boundary-current cold-start
        # blowup. Cap A_smag at ``safety * area * cos^2(lat) / dt`` (see
        # ``laplacian_smag_cfl_cap``) -- a TUNED ceiling that supplies the
        # MAXIMUM stabilising viscosity to the under-resolved WBC jets while
        # leaving the quiescent interior untouched (the resolution-appropriate
        # analogue of NEMO's WBC-enhanced eddy_viscosity_3D). NOTE: the stricter
        # rectangular CFL bound safety/(dt*(1/dx^2+1/dy^2)) STARVES the WBC and
        # blew up the eORCA025 cold-start (job 8126978) -- the larger area*cos^2
        # ceiling is what the cold start needs (empirically validated).
        if config.smag_cfl_safety > 0.0:
            _cap_h, _cap_q = laplacian_smag_cfl_cap(
                grid, dt, config.smag_cfl_safety)
            A_smag_h = jnp.minimum(
                A_smag_h, _cap_h[..., None] if A_smag_h.ndim == 3 else _cap_h)
            A_smag_q = jnp.minimum(
                A_smag_q, _cap_q[..., None] if A_smag_q.ndim == 3 else _cap_q)
        _smag_lap_u, _smag_lap_v = viscous_tendency_cgrid(
            u, v, grid, A_smag_h, A_smag_q,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        _smag_lap_u, _smag_lap_v = _apply_slope_foot(_smag_lap_u, _smag_lap_v)
        du_dt = du_dt + _smag_lap_u
        dv_dt = dv_dt + _smag_lap_v
        # Accumulate into the Smagorinsky diagnostic bucket
        diag_Cs_smag_u = diag_Cs_smag_u + _smag_lap_u
        diag_Cs_smag_v = diag_Cs_smag_v + _smag_lap_v

    if getattr(config, "C_leith", 0.0) > 0:
        leith_u, leith_v = leith_biharmonic_tendency_cgrid(
            u, v, grid, config.C_leith,
            modified=getattr(config, "C_leith_modified", False),
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        diag_Cl_leith_u = -leith_u
        diag_Cl_leith_v = -leith_v
        diag_Cl_leith_u, diag_Cl_leith_v = _apply_slope_foot(diag_Cl_leith_u, diag_Cl_leith_v)
        du_dt = du_dt + diag_Cl_leith_u
        dv_dt = dv_dt + diag_Cl_leith_v

    # --- 10b. Meridional-only Laplacian viscosity ---
    # Scalar d²/dy² applied directly at faces, targeting the 2Δy mode
    # without damping zonal flow.  Useful on lat-lon grids with large
    # dx/dy anisotropy where isotropic A_h over-damps zonal structure.
    # Acts on total velocity (like the main viscosity block above).
    _A_h_merid = config.A_h_merid
    if _A_h_merid > 0:
        _dy = grid.radius * (grid.lat[1] - grid.lat[0])  # constant
        _inv_dy2 = 1.0 / (_dy * _dy)
        # u: d²u/dy² at u-faces (u has shape n_lat, n_lon+1, nlev)
        # Mask u at land faces before differencing to avoid reading
        # land zeros as no-slip boundary (free-slip: land neighbors
        # should not contribute to the stencil).
        _u_masked = u * u_mask[:, :, jnp.newaxis]
        _u_pad = jnp.pad(_u_masked, ((1, 1), (0, 0), (0, 0)))
        _d2u_dy2 = (_u_pad[2:, :, :] - 2.0 * _u_pad[1:-1, :, :] +
                    _u_pad[:-2, :, :]) * _inv_dy2
        _merid_u, _ = _apply_slope_foot(_A_h_merid * _d2u_dy2,
                                         jnp.zeros_like(_d2u_dy2))
        du_dt = du_dt + _merid_u * u_mask[:, :, jnp.newaxis]
        # v: d²v/dy² at v-faces (v has shape n_lat+1, n_lon, nlev)
        _v_masked = v * v_mask[:, :, jnp.newaxis]
        _v_pad = jnp.pad(_v_masked, ((1, 1), (0, 0), (0, 0)))
        _d2v_dy2 = (_v_pad[2:, :, :] - 2.0 * _v_pad[1:-1, :, :] +
                    _v_pad[:-2, :, :]) * _inv_dy2
        _, _merid_v = _apply_slope_foot(jnp.zeros_like(_d2v_dy2),
                                         _A_h_merid * _d2v_dy2)
        dv_dt = dv_dt + _merid_v * v_mask[:, :, jnp.newaxis]

    # FAITHFUL positive-definite K_diss_h dissipation density [m²/s³] at cell
    # centres (Veros analogue), built ONLY when source_kdiss_h + kdiss_h_flux_form
    # are on AND the A_h Laplacian is active (the EKE source covers the A_h lateral
    # friction, mirroring Veros's K_diss_h from harmonic_friction; B_h/Smag/Leith
    # are not part of Veros's ACC K_diss_h).  The dissipation form is PAIRED to the
    # operator (consistency with #41):
    #   - "flux_divergence": the component-wise A_h·|∇u|² = 0.5·Σ(Δu·flux)
    #     (Veros ``calc_diss_u``/``calc_diss_v``), built from the SAME face fluxes the
    #     flux-div operator forms (``_kdiss_fluxdiv_cell`` above) — so selecting the
    #     operator gives Veros's EXACT friction AND its EXACT EKE source.
    #   - "vector_laplacian" (default): the Helmholtz A_h·(div²+ζ²)
    #     (``vector_laplacian_dissipation_cgrid``), using the SAME cell-centre A_h×scale
    #     field (cos-power / eq-boost / cap-boost / slope-foot) applied to the vector
    #     tendency.  ``None`` otherwise → the dynamical (clamped) source path is used.
    kdiss_h_cell = None
    if _want_kdiss_flux and _use_flux_div:
        # Component-wise dissipation already built by the flux-div operator from its
        # own face fluxes (None if the A_h Laplacian was inactive).  Slope-foot: the
        # ACC recipe (the only flux-form-K_diss_h path) has slope_foot_alpha=0, so no
        # slope-foot rescaling is needed; with slope-foot on the boosted dissipation
        # would be a localized approximation (same caveat as the vector path).
        kdiss_h_cell = _kdiss_fluxdiv_cell
    elif _want_kdiss_flux and _ah_scale_center is not None:
        _A_h_center = _ah_scale_center
        if _slope_E_center is not None:
            # Cell-centre slope-foot enhancement (3-D); broadcast the lat profile.
            # NB: the applied viscous tendency uses the face-interpolated slope-foot
            # factor (_slope_E_u/_slope_E_v), so this cell-centre form is exact for
            # the energy only where slope-foot is inactive (factor==1) — the ACC
            # recipe (the only oracle path using the flux form) has slope_foot_alpha
            # =0, so it is exact there; with slope-foot on it is a localized
            # bottom-cell approximation of the boosted dissipation.
            _A_h_center = _A_h_center[:, jnp.newaxis, jnp.newaxis] * _slope_E_center
        kdiss_h_cell = vector_laplacian_dissipation_cgrid(
            u, v, grid, _A_h_center,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
    return (du_dt, dv_dt, diag_Ah_lap_u, diag_Ah_lap_v, diag_Bh_bilap_u,
            diag_Bh_bilap_v, diag_Cs_smag_u, diag_Cs_smag_v, diag_Cl_leith_u,
            diag_Cl_leith_v, kdiss_h_cell)


def _bc_bottom_drag(du_dt, dv_dt, u, v, h_u, h_v, J, z_coord, config, grid):
    """Bottom drag: linear or quadratic-with-floor (DRAG_BG_VEL), distributed
    over a BBL thickness or applied at the partial-cell seafloor / deepest
    level. Pure verbatim extraction (Q8). Returns ``(du_dt, dv_dt,
    diag_botdrag_u, diag_botdrag_v)``."""
    diag_botdrag_u = jnp.zeros_like(du_dt)
    diag_botdrag_v = jnp.zeros_like(dv_dt)
    if config.bottom_drag_r > 0:
        # Drag acts on the full velocity (not perturbation) — the ocean
        # floor sees the total flow.  Consistent with MPAS and MOM6.
        # r is in [m/s]: du/dt = -r * u / dz_bottom  (resolution-independent stress).
        H_BBL = getattr(config, "bottom_drag_bbl_thickness", 0.0)
        # MOM6-style background-velocity floor (DRAG_BG_VEL).  When >0,
        # the linear-in-u drag is upgraded to quadratic-with-floor:
        #   r_eff = (bottom_drag_r / u_bg) · √(u² + u_bg²)
        # which (a) recovers linear ``bottom_drag_r`` at |u| → 0 (so
        # legacy weak-flow behaviour is preserved) and (b) scales as
        # quadratic Cd · |u| at |u| ≫ u_bg (production-equivalent
        # to MOM6 OM4's `BOTTOMDRAGLAW="quadratic"` with `DRAG_BG_VEL`).
        # u_bg=0 → exactly the legacy linear formula (bit-exact path).
        u_bg = float(getattr(config, "bottom_drag_bg_velocity", 0.0))
        if u_bg > 0.0:
            Cd_eq = config.bottom_drag_r / u_bg
            r_eff_u = Cd_eq * jnp.sqrt(u * u + u_bg * u_bg)
            r_eff_v = Cd_eq * jnp.sqrt(v * v + u_bg * u_bg)
        else:
            r_eff_u = config.bottom_drag_r
            r_eff_v = config.bottom_drag_r
        if H_BBL > 0:
            # Distributed BBL drag (Killworth & Edwards 1999, MOM6 BBL_thick_min):
            # spread drag over a fixed Ekman thickness ``H_BBL`` near the
            # seafloor instead of applying ``r*u/h_partial_bot`` to a single
            # (possibly very thin) partial cell.  Drag tendency at level k:
            #   ∂u/∂t |_drag = -r * u(k) * (overlap_k / h_u_k) / H_BBL
            # where overlap_k is the cell's geometric overlap with the
            # band [z_seafloor, z_seafloor + H_BBL].
            #
            # Limits:
            #   - If h_u_bot >= H_BBL: BBL fits in the bottom cell.
            #     Bottom-cell overlap = H_BBL → drag = -r*u/h_u_bot
            #     (recovers legacy single-cell drag).  No effect on cells
            #     above (overlap = 0).
            #   - If h_u_bot < H_BBL: BBL spans multiple cells.  Bottom
            #     cell drag = -r*u/H_BBL (much weaker than legacy
            #     -r*u/h_u_bot, fixing the audit-flagged "drag in 5m
            #     partial cell is 100x stronger than deep ocean" issue).
            #     Cells above bottom_level get partial-overlap drag.
            #
            # Build z interfaces at u/v faces from the cumulative thickness
            # along the level axis.  For the seafloor, ``z_seafloor =
            # -sum(h_u, axis=-1)`` (face's wet depth = sum of per-level
            # face thickness, partial-aware via min h).
            def _bbl_drag_for_face(u_field, h_face, r_eff):
                z_half = jnp.concatenate([
                    jnp.zeros(h_face.shape[:-1] + (1,), dtype=h_face.dtype),
                    -jnp.cumsum(h_face, axis=-1),
                ], axis=-1)
                z_top = z_half[..., :-1]
                z_bot = z_half[..., 1:]
                z_seafloor = z_half[..., -1:]
                bbl_top = z_seafloor + H_BBL
                overlap = jnp.maximum(
                    0.0,
                    jnp.minimum(z_top, bbl_top)
                    - jnp.maximum(z_bot, z_seafloor),
                )
                h_safe = jnp.maximum(h_face, 1e-10)
                # Effective BBL thickness: on shelves where the
                # total wet depth is shallower than ``H_BBL`` the
                # boundary-layer band cannot extend to its full
                # nominal thickness.  Divide by the actual total
                # overlap to keep the rate correct (matches
                # ``ocean_tendency_common.bbl_drag_distributed``).
                # Codex iter-39 #2.
                total_overlap = jnp.sum(overlap, axis=-1, keepdims=True)
                h_bbl_eff = jnp.minimum(
                    jnp.maximum(total_overlap, 1e-10), H_BBL,
                )
                return -r_eff * u_field * overlap / (h_safe * h_bbl_eff)
            diag_botdrag_u = _bbl_drag_for_face(u, h_u, r_eff_u)
            diag_botdrag_v = _bbl_drag_for_face(v, h_v, r_eff_v)
        elif isinstance(z_coord, OceanPartialCellCoordinate):
            # Partial cells: apply drag at each column's actual seafloor
            # (the lowest active level, ``bottom_level[i,j]``), using the
            # partial-cell thickness h_partial there.  Without this, drag
            # would only act at the deepest reference level (``nlev-1``)
            # in deep columns and not damp the bottom-trapped spurious
            # flow on shallower seamount slopes.
            n_lev = u.shape[-1]
            level_idx = jnp.arange(n_lev)
            # is_bottom_3d at u-faces / v-faces: 1.0 at the partial
            # bottom for that face's COLUMN.  We use the cell-center
            # bottom_level interpolated to faces (face's bottom level
            # is the SHALLOWER of the two adjacent columns — already
            # the only active level there since the deeper column's
            # cell at that level may be active too).
            bot_lev_cell = z_coord.bottom_level
            # u-face bottom_level: min of west/east cell (shallower wins).
            bot_lev_u_inner = jnp.minimum(
                jnp.roll(bot_lev_cell, 1, axis=1), bot_lev_cell,
            )
            bot_lev_u = jnp.concatenate(
                [bot_lev_u_inner, bot_lev_u_inner[:, 0:1]], axis=1,
            )
            # v-face bottom_level: min of south/north cell.
            bot_lev_v_int = jnp.minimum(bot_lev_cell[:-1], bot_lev_cell[1:])
            bot_lev_v = jnp.pad(bot_lev_v_int, ((1, 1), (0, 0)),
                                 constant_values=0)
            is_bot_u_3d = (level_idx[jnp.newaxis, jnp.newaxis, :]
                            == bot_lev_u[..., jnp.newaxis]).astype(u.dtype)
            is_bot_v_3d = (level_idx[jnp.newaxis, jnp.newaxis, :]
                            == bot_lev_v[..., jnp.newaxis]).astype(v.dtype)
            # Partial-cell h at faces (already partial-aware via h_k dispatch).
            h_u_drag = jnp.maximum(h_u, 1e-10)
            h_v_drag = jnp.maximum(h_v, 1e-10)
            diag_botdrag_u = (
                -r_eff_u * u / h_u_drag * is_bot_u_3d
            )
            diag_botdrag_v = (
                -r_eff_v * v / h_v_drag * is_bot_v_3d
            )
        else:
            dz_bot_u = z_coord.dz_ref[-1] * jnp.maximum(interp_cell_to_uface(J), 1e-10)
            dz_bot_v = z_coord.dz_ref[-1] * jnp.maximum(_interp_to_v_points(J, grid=grid), 1e-10)
            # Capture only at the bottom level; zeros elsewhere.
            r_eff_u_bot = r_eff_u[..., -1] if u_bg > 0.0 else r_eff_u
            r_eff_v_bot = r_eff_v[..., -1] if u_bg > 0.0 else r_eff_v
            diag_botdrag_u = diag_botdrag_u.at[..., -1].set(
                -r_eff_u_bot * u[..., -1] / dz_bot_u)
            diag_botdrag_v = diag_botdrag_v.at[..., -1].set(
                -r_eff_v_bot * v[..., -1] / dz_bot_v)
        du_dt = du_dt + diag_botdrag_u
        dv_dt = dv_dt + diag_botdrag_v
    return du_dt, dv_dt, diag_botdrag_u, diag_botdrag_v


def _bc_explicit_vertical_viscosity(du_dt, dv_dt, u_prime, v_prime, u, J, z_coord, config, grid):
    """Explicit background vertical viscosity A_v on the perturbation velocity
    (skipped when implicit_vertical_mixing is enabled). Pure verbatim extraction
    (Q8). Returns ``(du_dt, dv_dt, diag_Av_vert_u, diag_Av_vert_v)``."""
    diag_Av_vert_u = jnp.zeros_like(du_dt)
    diag_Av_vert_v = jnp.zeros_like(dv_dt)
    # Skip the explicit background vertical viscosity when the host
    # dynamics requested an implicit (backward-Euler) vertical solve —
    # the LatLonCGridOceanConfig.A_v floor is folded into the implicit
    # K profile downstream and applied unconditionally-stable.  The
    # KPP / Richardson / Constant scheme branches above are already
    # ``apply_diffusion=False`` in that mode.
    if (config.A_v > 0
            and u.shape[-1] >= 2
            and not getattr(config, "implicit_vertical_mixing", False)):
        jac_v_u = jnp.maximum(interp_cell_to_uface(J)[..., jnp.newaxis], 1e-10)
        jac_v_v = jnp.maximum(_interp_to_v_points(J, grid=grid)[..., jnp.newaxis], 1e-10)
        for vel, jac, is_u in [(u_prime, jac_v_u, True), (v_prime, jac_v_v, False)]:
            dv_dz_half = (vel[..., :-1] - vel[..., 1:]) / (
                z_coord.dz_half_ref * jac
            )
            flux = config.A_v * dv_dz_half
            # Pad along trailing axis instead of allocating a fresh
            # ``(..., 1)`` zero buffer + 3-array concatenate.
            _pad_axes = ((0, 0),) * (flux.ndim - 1)
            flux_full = jnp.pad(flux, (*_pad_axes, (1, 1)))
            vdiff = (flux_full[..., :-1] - flux_full[..., 1:]) / (
                z_coord.dz_ref * jac
            )
            if is_u:
                diag_Av_vert_u = vdiff
                du_dt = du_dt + vdiff
            else:
                diag_Av_vert_v = vdiff
                dv_dt = dv_dt + vdiff
    return du_dt, dv_dt, diag_Av_vert_u, diag_Av_vert_v


def _bc_physics_tendencies(du_dt, dv_dt, dT_dt, dS_dt, physics_fn, state, grid, z_coord, surface_forcing, u, v):
    """Stage 10b: the physics-pipeline tendencies (KPP/TKE vertical mixing,
    convection, etc.) applied via a cell-centre proxy state, interpolated to
    faces. Returns the captured K_v/A_v profiles for the implicit solve. Pure
    verbatim extraction (Q8)."""
    diag_phys_u = jnp.zeros_like(du_dt)
    diag_phys_v = jnp.zeros_like(dv_dt)
    # --- 10b. Physics tendencies (surface forcing, bottom drag, etc.) ---
    # The physics pipeline expects cell-center u/v shapes (shared with
    # A-grid and cubed-sphere).  Create a cell-center proxy state so
    # the physics functions produce (n_lat, n_lon, nlev) output, then
    # interpolate momentum tendencies to C-grid face points.
    phys_K_v = None
    phys_A_v = None
    if physics_fn is not None:
        u_cell = 0.5 * (u[:, :-1, :] + u[:, 1:, :])  # (n_lat, n_lon, nlev)
        v_cell = 0.5 * (v[:-1, :, :] + v[1:, :, :])   # (n_lat, n_lon, nlev)
        cc_state = state._replace(
            u=state.u.replace(data=u_cell),
            v=state.v.replace(data=v_cell),
        )
        phys = physics_fn(cc_state, grid, z_coord, surface_forcing)
        diag_phys_u = interp_cell_to_uface(phys.du_dt.data)
        diag_phys_v = _interp_to_v_points(phys.dv_dt.data, grid=grid)
        du_dt = du_dt + diag_phys_u
        dv_dt = dv_dt + diag_phys_v
        dT_dt = dT_dt + phys.dT_dt.data
        dS_dt = dS_dt + phys.dS_dt.data
        # Capture K profiles for implicit vertical mixing (avoids
        # recomputing KPP in the model step).
        phys_K_v = getattr(phys, "K_v", None)
        phys_A_v = getattr(phys, "A_v", None)
    return du_dt, dv_dt, dT_dt, dS_dt, phys_K_v, phys_A_v, diag_phys_u, diag_phys_v


def _bc_external_surface_forcing(du_dt, dv_dt, dT_dt, dS_dt, surface_forcing, u, v, T, S, h_k, z_coord, J, grid, rho_0, mask, mask_3d, *, route_heat_to_implicit=False):
    """Stage 10b': external surface forcing (wind stress tau_x/tau_y, net heat
    q_net, penetrating shortwave) from a coupled / OMIP OceanSurfaceForcing,
    with tripolar east-north -> grid-aligned rotation. Pure verbatim extraction
    (Q8).

    WIND STRESS (tau_x/tau_y → du_dt/dv_dt[...,0]) is ALWAYS applied explicitly
    (matching Veros's AB2'd ``tend_tauxyf``).  The HEAT forcing (non-solar q_net
    surface term + penetrating-shortwave column) is normally added to ``dT_dt``
    (explicit).  When ``route_heat_to_implicit=True`` (the
    ``surface_forcing_implicit`` config path) it is instead accumulated into a
    SEPARATE ``dT_surf`` array and WITHHELD from ``dT_dt`` so the model step can
    add it (× dt_tracer) inside the backward-Euler vertical-mixing solve at
    weight 1.0 (Veros's implicit surface-forcing placement).

    Returns ``(du_dt, dv_dt, dT_dt, dS_dt, dT_surf)`` where ``dT_surf`` is a
    full-column zero array unless ``route_heat_to_implicit`` AND a q_net forcing
    were both present."""
    dT_surf = jnp.zeros_like(dT_dt)
    # --- 10b'. External surface forcing (e.g. from JRA55 bulk fluxes) ---
    # When the caller passes an OceanSurfaceForcing carrying tau_x /
    # tau_y / q_net / sw_down, apply them here.  Mirrors
    # mpas_physics.py:183-245 so MPAS and the lat-lon C-grid behave
    # identically under coupled / OMIP forcing.  The bulk-flux solver
    # returns tau in atmospheric (eastward / northward) convention; the
    # ocean needs the reaction force, hence the sign flip.  On tripolar
    # geometries, geographic east/north is rotated to grid-aligned
    # (i, j) components via geom.cos_alpha_u / sin_alpha_u (identity
    # outside the bipolar cap, so regular lat-lon C-grids are
    # bit-exact unchanged).
    if surface_forcing is not None:
        _sf_tau_x = getattr(surface_forcing, "tau_x", None)
        _sf_tau_y = getattr(surface_forcing, "tau_y", None)
        _sf_q_net = getattr(surface_forcing, "q_net", None)
        _sf_sw = getattr(surface_forcing, "sw_down", None)
        _sf_salt = getattr(surface_forcing, "salt_flux", None)

        if _sf_tau_x is not None and _sf_tau_y is not None:
            # Atmosphere convention (opposes wind) -> ocean reaction.
            tau_e_T = -jnp.asarray(_sf_tau_x, dtype=u.dtype)
            tau_n_T = -jnp.asarray(_sf_tau_y, dtype=u.dtype)
            tau_e_u_face = interp_cell_to_uface(tau_e_T)       # (n_lat, n_lon+1)
            tau_n_u_face = interp_cell_to_uface(tau_n_T)
            tau_e_v_face = _interp_to_v_points(tau_e_T, grid=grid)  # (n_lat+1, n_lon)
            tau_n_v_face = _interp_to_v_points(tau_n_T, grid=grid)

            cos_a_u = getattr(grid, "cos_alpha_u", None)
            sin_a_u = getattr(grid, "sin_alpha_u", None)
            cos_a_v = getattr(grid, "cos_alpha_v", None)
            sin_a_v = getattr(grid, "sin_alpha_v", None)
            if cos_a_u is not None and sin_a_u is not None:
                ca_u = jnp.asarray(cos_a_u, dtype=u.dtype)
                sa_u = jnp.asarray(sin_a_u, dtype=u.dtype)
                tau_i_u = tau_e_u_face * ca_u + tau_n_u_face * sa_u
            else:
                tau_i_u = tau_e_u_face
            if cos_a_v is not None and sin_a_v is not None:
                ca_v = jnp.asarray(cos_a_v, dtype=u.dtype)
                sa_v = jnp.asarray(sin_a_v, dtype=u.dtype)
                tau_j_v = -tau_e_v_face * sa_v + tau_n_v_face * ca_v
            else:
                tau_j_v = tau_n_v_face

            dz_0_T = jnp.asarray(z_coord.dz_ref[0], dtype=u.dtype) * J
            dz_0_u = interp_cell_to_uface(dz_0_T)
            dz_0_v = _interp_to_v_points(dz_0_T, grid=grid)
            rho_0_dt = jnp.asarray(rho_0, dtype=u.dtype)
            inv_rho_dz_u = 1.0 / (rho_0_dt * jnp.maximum(dz_0_u, 1e-10))
            inv_rho_dz_v = 1.0 / (rho_0_dt * jnp.maximum(dz_0_v, 1e-10))

            du_dt = du_dt.at[..., 0].add(tau_i_u * inv_rho_dz_u)
            dv_dt = dv_dt.at[..., 0].add(tau_j_v * inv_rho_dz_v)

        if _sf_q_net is not None:
            from legoesm.ocean.eos import c_sw as _c_sw
            dz_0_T_q = jnp.asarray(z_coord.dz_ref[0], dtype=T.dtype) * J
            inv_rho_csw_dz = 1.0 / (
                jnp.asarray(rho_0, dtype=T.dtype)
                * jnp.asarray(_c_sw, dtype=T.dtype)
                * jnp.maximum(dz_0_T_q, 1e-10)
            )
            q_net_T = jnp.asarray(_sf_q_net, dtype=T.dtype)
            # The heat forcing lands on the EXPLICIT ``dT_dt`` (default) or, when
            # ``route_heat_to_implicit``, the SEPARATE ``dT_surf`` accumulator.
            # Bind ``dT_target`` to the recipient array so the explicit branch
            # uses the SAME scatter-add/add sequence as before (bit-identical
            # default-off path); the implicit branch starts from zeros.
            dT_target = dT_surf if route_heat_to_implicit else dT_dt
            if _sf_sw is not None:
                # Split: non-solar at surface, solar penetrating column.
                sw_T = jnp.asarray(_sf_sw, dtype=T.dtype)
                sw_absorbed = sw_T * jnp.asarray(0.94, dtype=T.dtype)
                q_nonsolar = q_net_T - sw_absorbed
                dT_target = dT_target.at[..., 0].add(
                    q_nonsolar * inv_rho_csw_dz * mask
                )
                from legoesm.ocean.physics.shortwave_penetration import (
                    shortwave_penetration_tendency,
                )
                sw_tend = shortwave_penetration_tendency(
                    sw_absorbed,
                    z_coord.dz_ref,
                    z_coord.z_half_ref,
                    J,
                    rho_0=float(rho_0),
                )
                dT_target = dT_target + sw_tend * mask_3d
            else:
                dT_target = dT_target.at[..., 0].add(
                    q_net_T * inv_rho_csw_dz * mask
                )
            if route_heat_to_implicit:
                dT_surf = dT_target
            else:
                dT_dt = dT_target

        # Real salt-mass flux (e.g. sea-ice brine rejection) -> top-layer S.
        # Distinct from the freshwater virtual-salt path (which the freshwater=
        # argument applies); this is the explicit salt-mass channel (#F11).
        if _sf_salt is not None:
            from legoesm.ocean.freshwater import salt_flux_salinity_tendency
            # Use the ACTUAL partial-cell-aware top-layer thickness h_k[...,0]
            # (the tracer cell that carries the salinity mass), NOT
            # dz_ref[0]*J, so the real salt source is mass-conservative on
            # shallow top-partial columns (codex).
            dz_0_T_s = jnp.asarray(h_k[..., 0], dtype=S.dtype)
            dS_salt = salt_flux_salinity_tendency(
                jnp.asarray(_sf_salt, dtype=S.dtype), dz_0_T_s, float(rho_0))
            dS_dt = dS_dt.at[..., 0].add(dS_salt * mask)
    return du_dt, dv_dt, dT_dt, dS_dt, dT_surf


def _bc_sponge_relaxation(du_dt, dv_dt, dT_dt, dS_dt, T, S, u, v, sponge, grid):
    """Stage 10c: sponge-layer relaxation of T/S (and optionally u/v) toward
    reference fields. Pure verbatim extraction (Q8). Returns ``(du_dt, dv_dt,
    dT_dt, dS_dt, diag_sponge_u, diag_sponge_v)``."""
    diag_sponge_u = jnp.zeros_like(du_dt)
    diag_sponge_v = jnp.zeros_like(dv_dt)
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
            diag_sponge_u = gamma_u * (sponge.u_ref.astype(_dt) - u)
            du_dt = du_dt + diag_sponge_u
        if sponge.v_ref is not None:
            gamma_v = _interp_to_v_points(sponge.gamma.astype(_dt), grid=grid)[..., jnp.newaxis]
            diag_sponge_v = gamma_v * (sponge.v_ref.astype(_dt) - v)
            dv_dt = dv_dt + diag_sponge_v
    return du_dt, dv_dt, dT_dt, dS_dt, diag_sponge_u, diag_sponge_v


def _bc_horizontal_momentum_advection_flux_form(
    du_dt, dv_dt, u, v, h_u, h_v, u_mask_3d, v_mask_3d, mask, grid, config,
):
    """Flux-form horizontal momentum advection (alternative to the
    vector-invariant PV flux, `_bc_pv_flux`) — the Veros/MOM6/MITgcm form
    -div(transport (x) velocity), selected by ``momentum_advection="flux_form"``.

    Conservative FV discretization mirroring ``divergence_cgrid``'s metric
    (u-face meridional length ``dy*0.5``, v-face zonal length ``R·cos(lat_v)·dlon``,
    cell ``area``). Momentum is conserved by flux telescoping: the cross-face
    fluxes cancel in the domain sum (periodic in lon, v=0 wall at the poles).
    The advected velocity is reconstructed by ``config.momentum_flux_scheme``
    ("upwind" 1st-order / "centered" 2nd-order).

    Returns ``(du_dt, dv_dt, diag_hadv_u, diag_hadv_v)`` — the horizontal-
    advection contribution fills the same diagnostic slot the PV flux would
    (so the orchestrator + momentum-diagnostics closure are unchanged).
    """
    if is_tripolar(grid):
        raise ValueError(
            "momentum_advection='flux_form' is not yet implemented on tripolar "
            "grids (the vertex-metric handling needs the 2D dx_v/dy_u fields). "
            "Use 'vector_invariant' on tripolar, or extend this substage."
        )
    scheme = getattr(config, "momentum_flux_scheme", "upwind")
    u.shape[1] - 1

    # --- FV metrics (mirror divergence_cgrid) ---
    dy_u = (grid.dy * 0.5)[:, jnp.newaxis, jnp.newaxis]            # (n_lat,1,1)
    lat = grid.lat
    cos_lat_v = jnp.pad(jnp.cos(0.5 * (lat[:-1] + lat[1:])), (1, 1))
    face_dx_v = (grid.radius * cos_lat_v * grid.dlon)[:, jnp.newaxis, jnp.newaxis]  # (n_lat+1,1,1)
    area = grid.area[..., jnp.newaxis]                            # (n_lat,n_lon,1)

    # Volume transports through faces [m^3/s] (h-weighted velocity x face length).
    Q_u = h_u * u * u_mask_3d * dy_u            # (n_lat, n_lon+1, nlev)
    Q_v = h_v * v * v_mask_3d * face_dx_v       # (n_lat+1, n_lon, nlev)

    def _upwind(adv_pos, adv_neg, transport):
        # Upstream value: take adv_pos where transport > 0 (flow from that side).
        if scheme == "centered":
            return 0.5 * (adv_pos + adv_neg)
        return jnp.where(transport > 0.0, adv_pos, adv_neg)

    # ============ u-momentum at u-points (n_lat, n_lon+1) ============
    # x-flux at cell centres: transport_x_centre * u_advected_centre.
    Qx_c = 0.5 * (Q_u[:, :-1, :] + Q_u[:, 1:, :])                 # (n_lat,n_lon,nlev)
    u_c = _upwind(u[:, :-1, :], u[:, 1:, :], Qx_c)               # west when Qx>0
    Fx_uu = Qx_c * u_c                                           # (n_lat,n_lon,nlev)
    # divergence to u-points (periodic in lon): flux[centre J] - flux[centre J-1].
    _dx = Fx_uu - jnp.roll(Fx_uu, 1, axis=1)
    net_zonal_u = jnp.concatenate([_dx, _dx[:, 0:1, :]], axis=1)  # (n_lat,n_lon+1,nlev)

    # y-flux at vertices: transport_y_vertex * u_advected_vertex.
    Qv_west = jnp.roll(Q_v, 1, axis=1)
    Qy_vtx_core = 0.5 * (Q_v + Qv_west)                          # (n_lat+1,n_lon,nlev) at lon-centres? -> lon-faces
    Qy_vtx = jnp.concatenate([Qy_vtx_core, Qy_vtx_core[:, 0:1, :]], axis=1)  # (n_lat+1,n_lon+1,nlev)
    # u to lat-faces (vertices): interior avg of adjacent u rows; poles unused (Qy=0 there).
    u_south = u[:-1, :, :]
    u_north = u[1:, :, :]
    u_vtx_int = _upwind(u_south, u_north, Qy_vtx[1:-1, :, :])    # (n_lat-1,n_lon+1,nlev)
    zero_row = jnp.zeros_like(u[:1, :, :])
    u_vtx = jnp.concatenate([zero_row, u_vtx_int, zero_row], axis=0)  # (n_lat+1,n_lon+1,nlev)
    Fy_vu = Qy_vtx * u_vtx                                       # (n_lat+1,n_lon+1,nlev)
    net_merid_u = Fy_vu[1:, :, :] - Fy_vu[:-1, :, :]            # (n_lat,n_lon+1,nlev)

    # u-cell area at u-faces (avg of adjacent cell areas, periodic).
    a_uc = 0.5 * (area[:, :, 0] + jnp.roll(area[:, :, 0], 1, axis=1))   # (n_lat,n_lon)
    A_u = jnp.concatenate([a_uc, a_uc[:, 0:1]], axis=1)[..., jnp.newaxis]  # (n_lat,n_lon+1,1)
    inv_Ah_u = u_mask_3d / jnp.maximum(A_u * h_u, 1.0e-12)
    diag_hadv_u = -(net_zonal_u + net_merid_u) * inv_Ah_u

    # ============ v-momentum at v-points (n_lat+1, n_lon) ============
    # y-flux at cell centres: transport_y_centre * v_advected_centre.
    Qy_c = 0.5 * (Q_v[:-1, :, :] + Q_v[1:, :, :])               # (n_lat,n_lon,nlev)
    v_c = _upwind(v[:-1, :, :], v[1:, :, :], Qy_c)              # south when Qy>0
    Fy_vv = Qy_c * v_c                                          # (n_lat,n_lon,nlev)
    # divergence to v-points (interior lat-faces; poles are walls -> 0).
    net_merid_v_int = Fy_vv[1:, :, :] - Fy_vv[:-1, :, :]       # (n_lat-1,n_lon,nlev)
    zero_lon = jnp.zeros_like(v[:1, :, :])
    net_merid_v = jnp.concatenate([zero_lon, net_merid_v_int, zero_lon], axis=0)

    # x-flux at vertices (lon-faces). Use roll-based periodicity in lon (drop
    # the u wrap column) so the divergence telescopes EXACTLY regardless of
    # whether the input enforces u[:, n_lon] == u[:, 0] — mirrors the
    # u-momentum x-part above.
    Q_u_core = Q_u[:, :-1, :]                                   # (n_lat,n_lon,nlev) distinct lon-faces
    Qx_vtx_int = 0.5 * (Q_u_core[:-1, :, :] + Q_u_core[1:, :, :])  # (n_lat-1,n_lon,nlev)
    zero_vtx = jnp.zeros_like(Q_u_core[:1, :, :])
    Qx_vtx = jnp.concatenate([zero_vtx, Qx_vtx_int, zero_vtx], axis=0)  # (n_lat+1,n_lon,nlev) at lon-faces
    # v to lon-faces (vertices), periodic: west/east centres are v[:, j-1], v[:, j].
    v_west = jnp.roll(v, 1, axis=1)
    v_vtx = _upwind(v_west, v, Qx_vtx)                         # (n_lat+1,n_lon,nlev) west when Qx>0
    Fx_uv = Qx_vtx * v_vtx                                     # at lon-faces 0..n_lon-1
    # v-cell (centre j) E face = vertex j+1, W face = vertex j (periodic).
    net_zonal_v = jnp.roll(Fx_uv, -1, axis=1) - Fx_uv          # (n_lat+1,n_lon,nlev)

    # v-cell area at v-faces (avg of adjacent cell areas; poles padded edge).
    a_vc = 0.5 * (area[:-1, :, 0] + area[1:, :, 0])           # (n_lat-1,n_lon)
    A_v = jnp.concatenate([a_vc[:1], a_vc, a_vc[-1:]], axis=0)[..., jnp.newaxis]  # (n_lat+1,n_lon,1)
    inv_Ah_v = v_mask_3d / jnp.maximum(A_v * h_v, 1.0e-12)
    diag_hadv_v = -(net_zonal_v + net_merid_v) * inv_Ah_v

    du_dt = du_dt + diag_hadv_u
    dv_dt = dv_dt + diag_hadv_v
    return du_dt, dv_dt, diag_hadv_u, diag_hadv_v


def latlon_cgrid_ocean_baroclinic_tendencies(
    state: LatLonCGridOceanState,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonCGridOceanConfig = LatLonCGridOceanConfig(),
    physics_fn=None,
    surface_forcing=None,
    sponge=None,
    dt: float = 300.0,
    diagnose_momentum: bool = False,
    surface_tracer_forcing_fn=None,
):
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
    diagnose_momentum : bool
        If ``True``, capture each momentum-tendency component at its
        point of computation and return a
        ``(tendencies, MomentumTendencyDiagnostics)`` tuple instead of
        just ``tendencies``.  Used by the budget-closure infrastructure;
        adds memory but no recompilation.  Default ``False`` (existing
        behavior).
    surface_tracer_forcing_fn : callable, optional
        Surface-restoring physics function (``make_surface_forcing_physics``)
        used ONLY when ``config.surface_forcing_implicit`` is True.  In that
        mode ``physics_fn`` is built with ``surface_forcing.scheme="none"`` (so
        the restoring is NOT summed into ``dT_dt``); this function instead
        supplies the restoring RATE which — together with the prescribed
        q_net / shortwave heat (also withheld from ``dT_dt``) — is returned on
        ``tendencies.surface_tracer_forcing`` for the implicit (backward-Euler)
        application in the model step.  ``None`` (default) ⇒ surface forcing is
        applied explicitly (bit-identical legacy path).

    Returns
    -------
    LatLonCGridOceanTendencies
        When ``diagnose_momentum`` is ``False``.
    (LatLonCGridOceanTendencies, MomentumTendencyDiagnostics)
        When ``diagnose_momentum`` is ``True``.  The diagnostics satisfy
        ``Σ components == du_dt`` (and v) to machine precision.
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
    # 3D face masks: when partial coord is active, faces are wet only
    # where BOTH adjacent cells are wet AT THAT LEVEL — handles columns
    # with different ``bottom_level`` correctly (the active-vs-inactive
    # face case from Phase 3b).  For pure z\\* coord (legacy) and for
    # partial cells with all columns having the same bottom_level,
    # this produces identical results to broadcasting the 2D mask.
    if isinstance(z_coord, OceanPartialCellCoordinate):
        u_mask_3d, v_mask_3d = compute_face_masks_3d(z_coord.is_active, grid)
    else:
        u_mask_3d = u_mask[..., jnp.newaxis]
        v_mask_3d = v_mask[..., jnp.newaxis]

    g_val = config.g
    rho_0 = config.rho_0
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta.dtype)
    eta_floor = min_water_col - H_bathy
    eta_safe = jnp.maximum(eta, eta_floor) * mask


    # --- Stages 1-3: geometry (J, h_k) + density (rho_prime) + baroclinic
    # pressure anomaly (p_prime_filled). ---
    J, h_k, rho_prime, p_prime_filled = _bc_geometry_and_density(
        eta_safe, H_bathy, z_coord, config, T, S, mask, grid, rho_0, g_val,
    )

    # --- Stages 4-4b: vertical velocity (w), face thicknesses (h_u, h_v),
    # per-layer flux divergence, and perturbation velocities. ---
    h_u, h_v, flux_div_k, w, u_prime, v_prime = (
        _bc_vertical_and_depthmean_velocity(
            h_k, u, v, u_mask_3d, v_mask_3d, grid, z_coord, u_mask, v_mask,
        )
    )

    # --- 5. Coriolis ---
    # Coriolis is NOT included in the returned momentum tendencies.
    # It is applied as a forward-backward (Matsuno) step in the step
    # function (ocean_model_latlon_cgrid.py), which is unconditionally
    # stable for inertial oscillations.  Forward Euler Coriolis amplifies
    # by sqrt(1 + (f*dt)^2) per step and blows up within ~1 day at
    # high latitudes.

    # --- Stages 6 / 6-7 / 6b: KE gradient + pressure gradient + Adcroft
    # partial-cell PGF correction. ---
    _mom_adv = config.momentum_advection
    # WENO order, used only inside the WENO branches of the vertical-advection
    # stage (None otherwise -> never dereferenced). The PV-flux stage computes
    # its own copy internally.
    _weno_order = {"weno5": 5, "weno7": 7}.get(_mom_adv)
    dKE_dx, dp_dx, dKE_dy, dp_dy = _bc_ke_and_pressure_gradients(
        u, v, p_prime_filled, rho_prime, grid, config, z_coord,
        eta_safe, H_bathy, g_val, mask,
    )
    # Flux-form momentum advection (stage 7b below) provides the FULL horizontal
    # advection -div(transport(x)u), which already includes the kinetic-energy
    # gradient. The vector-invariant form instead splits advection into the KE
    # gradient (here) + the PV flux (stage 7b). So under flux_form, drop the KE
    # gradient here to avoid double-counting; KE_PGF then carries only the
    # pressure gradient.
    if _mom_adv == "flux_form":
        dKE_dx = jnp.zeros_like(dKE_dx)
        dKE_dy = jnp.zeros_like(dKE_dy)

    # --- 7. Momentum tendencies (non-Coriolis only) ---
    # Capture each term as a named local so the same expression feeds
    # both the integration and the optional diagnostics path.
    KE_PGF_u = -dKE_dx - dp_dx / rho_0
    KE_PGF_v = -dKE_dy - dp_dy / rho_0
    du_dt = KE_PGF_u
    dv_dt = KE_PGF_v
    # Diagnostics scaffolding: zero arrays for terms that may be
    # inactive in this config; overwritten below where active.
    _diag_zero_u = jnp.zeros_like(du_dt)
    _diag_zero_v = jnp.zeros_like(dv_dt)
    diag_vortcor_u = _diag_zero_u
    diag_vortcor_v = _diag_zero_v
    diag_Dterm_u = _diag_zero_u    # WENO momentum-advection D-term;
    diag_Dterm_v = _diag_zero_v    # zero unless WENO + weno_d_term active.
    diag_vertadv_u = _diag_zero_u
    diag_vertadv_v = _diag_zero_v
    diag_Ah_lap_u = _diag_zero_u
    diag_Ah_lap_v = _diag_zero_v
    diag_Bh_bilap_u = _diag_zero_u
    diag_Bh_bilap_v = _diag_zero_v
    diag_Cs_smag_u = _diag_zero_u
    diag_Cs_smag_v = _diag_zero_v
    diag_Cl_leith_u = _diag_zero_u
    diag_Cl_leith_v = _diag_zero_v
    diag_botdrag_u = _diag_zero_u
    diag_botdrag_v = _diag_zero_v
    diag_Av_vert_u = _diag_zero_u
    diag_Av_vert_v = _diag_zero_v
    diag_phys_u = _diag_zero_u
    diag_phys_v = _diag_zero_v
    diag_sponge_u = _diag_zero_u
    diag_sponge_v = _diag_zero_v

    # --- Stage 7b: horizontal momentum advection. Flux-form (-div(transport(x)u),
    # Veros/MOM6 style) or the default vector-invariant PV flux. Both fill the
    # same diagnostic slot. ---
    if _mom_adv == "flux_form":
        du_dt, dv_dt, diag_vortcor_u, diag_vortcor_v = (
            _bc_horizontal_momentum_advection_flux_form(
                du_dt, dv_dt, u, v, h_u, h_v, u_mask_3d, v_mask_3d, mask, grid, config,
            )
        )
    else:
        du_dt, dv_dt, diag_vortcor_u, diag_vortcor_v = _bc_pv_flux(
            du_dt, dv_dt, u, v, h_u, h_v, h_k, u_mask_3d, v_mask_3d, mask, grid, _mom_adv,
        )

    # --- Stage 7c: WENO divergence (D-term) dissipation. ---
    du_dt, dv_dt, diag_Dterm_u, diag_Dterm_v = _bc_dterm(
        du_dt, dv_dt, u, v, u_mask_3d, v_mask_3d, mask, grid, config, _mom_adv,
    )

    # --- Stage 8: vertical momentum advection. ---
    du_dt, dv_dt, diag_vertadv_u, diag_vertadv_v = _bc_vertical_momentum_advection(
        du_dt, dv_dt, u_prime, v_prime, w, h_u, h_v, u_mask_3d, v_mask_3d,
        grid, _mom_adv, _weno_order, config, diagnose_momentum,
    )

    # --- Stage 9: tracer diffusion tendencies (dT_dt, dS_dt). ---
    dT_dt, dS_dt = _bc_tracer_tendencies(T, S, config, grid, mask, J, z_coord)

    # --- Stages 10 + 10b: horizontal + meridional viscosity. ---
    (du_dt, dv_dt, diag_Ah_lap_u, diag_Ah_lap_v, diag_Bh_bilap_u,
     diag_Bh_bilap_v, diag_Cs_smag_u, diag_Cs_smag_v, diag_Cl_leith_u,
     diag_Cl_leith_v, kdiss_h_cell) = _bc_horizontal_viscosity(
        du_dt, dv_dt, u, v, grid, mask, u_mask, v_mask, config, z_coord, H_bathy, dt,
    )

    # --- Bottom drag. ---
    du_dt, dv_dt, diag_botdrag_u, diag_botdrag_v = _bc_bottom_drag(
        du_dt, dv_dt, u, v, h_u, h_v, J, z_coord, config, grid,
    )

    # --- Explicit background vertical viscosity (A_v). ---
    du_dt, dv_dt, diag_Av_vert_u, diag_Av_vert_v = _bc_explicit_vertical_viscosity(
        du_dt, dv_dt, u_prime, v_prime, u, J, z_coord, config, grid,
    )

    # --- Stage 10b: physics-pipeline tendencies. ---
    (du_dt, dv_dt, dT_dt, dS_dt, phys_K_v, phys_A_v, diag_phys_u,
     diag_phys_v) = _bc_physics_tendencies(
        du_dt, dv_dt, dT_dt, dS_dt, physics_fn, state, grid, z_coord,
        surface_forcing, u, v,
    )

    # --- Stage 10b': external surface forcing (wind stress / heat / shortwave). ---
    # ``surface_forcing_implicit`` (the ACC recipe): WITHHOLD the surface TRACER
    # heat (q_net + penetrating shortwave) from the explicit ``dT_dt`` and route
    # it into ``dT_surf`` for the implicit (backward-Euler) application; WIND
    # STRESS stays explicit either way (it is AB2'd in both legoESM and Veros).
    _sf_implicit = bool(getattr(config, "surface_forcing_implicit", False))
    du_dt, dv_dt, dT_dt, dS_dt, dT_surf_heat = _bc_external_surface_forcing(
        du_dt, dv_dt, dT_dt, dS_dt, surface_forcing, u, v, T, S, h_k, z_coord, J, grid,
        rho_0, mask, mask_3d, route_heat_to_implicit=_sf_implicit,
    )

    # --- Stage 10b'': surface TRACER restoring (T*/S*) routed for IMPLICIT
    # application when ``surface_forcing_implicit`` is on.  In that mode
    # ``physics_fn`` carries no restoring (its surface_forcing.scheme is "none"),
    # so the restoring RATE is computed here via ``surface_tracer_forcing_fn``
    # and combined with the withheld q_net/shortwave heat into
    # ``surface_tracer_forcing``.  When OFF, this block is skipped entirely and
    # ``surface_tracer_forcing`` stays ``None`` (bit-identical legacy path). ---
    surface_tracer_forcing = None
    if _sf_implicit:
        dT_surf = dT_surf_heat
        dS_surf = jnp.zeros_like(dS_dt)
        if surface_tracer_forcing_fn is not None:
            _sf_phys = surface_tracer_forcing_fn(state, grid, z_coord, surface_forcing)
            dT_surf = dT_surf + _sf_phys.dT_dt.data
            dS_surf = dS_surf + _sf_phys.dS_dt.data
        # Mask consistently with the explicit tracer tendencies (applied below).
        dT_surf = dT_surf * mask_3d
        dS_surf = dS_surf * mask_3d
        surface_tracer_forcing = SurfaceTracerForcing(
            dT_dt=Field(data=dT_surf, name="dT_surf_rate",
                        dims=("lat", "lon", "level"), units="degC/s"),
            dS_dt=Field(data=dS_surf, name="dS_surf_rate",
                        dims=("lat", "lon", "level"), units="PSU/s"),
        )

    # --- Stage 10c: sponge-layer relaxation. ---
    du_dt, dv_dt, dT_dt, dS_dt, diag_sponge_u, diag_sponge_v = _bc_sponge_relaxation(
        du_dt, dv_dt, dT_dt, dS_dt, T, S, u, v, sponge, grid,
    )

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

    # Harmonic LATERAL-viscosity tendencies for the EKE K_diss_h source — exposed
    # ONLY when the prognostic-EKE ``source_kdiss_h`` option is on (the ACC recipe;
    # otherwise None, so the tendency pytree + every existing path is bit-identical).
    # diag_Ah_lap_u/v are the A_h∇²(u,v) tendencies computed ONCE in
    # _bc_horizontal_viscosity (no recompute / no duplicate numerics); face-mask them
    # consistently with the applied du_dt (du_dt = Σ components × u_mask_3d).
    _eke_cfg = getattr(config.gm_redi, "eke", None) if config.gm_redi is not None else None
    if _eke_cfg is not None and getattr(_eke_cfg, "source_kdiss_h", False):
        Ah_visc_u = Field(data=diag_Ah_lap_u * u_mask_3d, name="Ah_visc_u",
                          dims=dims_u, units="m/s^2")
        Ah_visc_v = Field(data=diag_Ah_lap_v * v_mask_3d, name="Ah_visc_v",
                          dims=dims_v, units="m/s^2")
    else:
        Ah_visc_u = None
        Ah_visc_v = None
    # FAITHFUL positive-definite K_diss_h dissipation density [m²/s³] at cell
    # centres — populated ONLY when source_kdiss_h + kdiss_h_flux_form are on (the
    # ACC recipe); ``None`` otherwise so the dynamical (clamped) source path runs.
    if kdiss_h_cell is not None:
        Ah_kediss_cell = Field(data=kdiss_h_cell * mask_3d, name="Ah_kediss_cell",
                               dims=dims_3d, units="m^2/s^3")
    else:
        Ah_kediss_cell = None

    tendencies = LatLonCGridOceanTendencies(
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
        K_v=phys_K_v,
        A_v=phys_A_v,
        Ah_visc_u=Ah_visc_u,
        Ah_visc_v=Ah_visc_v,
        Ah_kediss_cell=Ah_kediss_cell,
        surface_tracer_forcing=surface_tracer_forcing,
    )

    if not diagnose_momentum:
        return tendencies

    # Apply the same land mask to every diagnostic component.  Because
    # the final masking is `du_dt = du_dt * u_mask_3d` and × distributes
    # over +, applying the mask uniformly to all components preserves
    # ``Σ components == total`` exactly (flag off).  With
    # ``adaptive_implicit_vertadv`` on, ``vertadv_{u,v}`` is a
    # diagnostic-only start-of-step estimate excluded from ``total`` /
    # ``du_dt`` -> closure is ``Σ (components except vertadv) == total``.
    def _mu(x):
        return Field(data=x * u_mask_3d, name="diag_u", dims=dims_u, units="m/s^2")
    def _mv(x):
        return Field(data=x * v_mask_3d, name="diag_v", dims=dims_v, units="m/s^2")

    diagnostics = MomentumTendencyDiagnostics(
        KE_PGF_u=_mu(KE_PGF_u),         KE_PGF_v=_mv(KE_PGF_v),
        vortcor_u=_mu(diag_vortcor_u),  vortcor_v=_mv(diag_vortcor_v),
        Dterm_u=_mu(diag_Dterm_u),      Dterm_v=_mv(diag_Dterm_v),
        vertadv_u=_mu(diag_vertadv_u),  vertadv_v=_mv(diag_vertadv_v),
        Ah_lap_u=_mu(diag_Ah_lap_u),    Ah_lap_v=_mv(diag_Ah_lap_v),
        Bh_bilap_u=_mu(diag_Bh_bilap_u),Bh_bilap_v=_mv(diag_Bh_bilap_v),
        Cs_smag_u=_mu(diag_Cs_smag_u),  Cs_smag_v=_mv(diag_Cs_smag_v),
        Cl_leith_u=_mu(diag_Cl_leith_u),Cl_leith_v=_mv(diag_Cl_leith_v),
        botdrag_u=_mu(diag_botdrag_u),  botdrag_v=_mv(diag_botdrag_v),
        Av_vert_u=_mu(diag_Av_vert_u),  Av_vert_v=_mv(diag_Av_vert_v),
        phys_u=_mu(diag_phys_u),        phys_v=_mv(diag_phys_v),
        sponge_u=_mu(diag_sponge_u),    sponge_v=_mv(diag_sponge_v),
        # total_u/v are the actually-applied masked tendencies — must
        # equal Σ of the components above to machine precision.
        total_u=Field(data=du_dt, name="total_u", dims=dims_u, units="m/s^2"),
        total_v=Field(data=dv_dt, name="total_v", dims=dims_v, units="m/s^2"),
    )
    return tendencies, diagnostics


