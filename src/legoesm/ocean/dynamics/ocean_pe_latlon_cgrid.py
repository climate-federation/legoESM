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

Vector-invariant status (see issue #160)
----------------------------------------
The momentum equation is split into baroclinic (this file) and
barotropic (barotropic_latlon_cgrid.py) parts. The baroclinic step
computes the vorticity flux ζ×u using the *perturbation* velocity u'
(see section 7b below) and the barotropic solver is purely linear in
U_bar. The cross terms ζ(u')·V_bar and ζ(U_bar)·v' are therefore
missing from the total momentum budget. This is not equivalent to
integrating (f+ζ_total)·u_total and has no known conservation
property — despite the historical "Sadourny" label, the 2-point ζ /
4-point raw-v' stencil at section 7b is not Sadourny EC, Sadourny EN,
Arakawa-Hsu, or Arakawa-Lamb. Fixing this requires a MOM6-style
slow-forcing coupling; tracked in issue #160.

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
    # Pole rows zero (wall BC); single Pad HLO op.
    pad_axes = ((0, 0),) * (f_interior.ndim - 1)
    return jnp.pad(f_interior, ((1, 1), *pad_axes))


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
    # Pole rows zero (wall BC); single Pad HLO op.
    pad_axes = ((0, 0),) * (f_tvd.ndim - 1)
    return jnp.pad(f_tvd, ((1, 1), *pad_axes))


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

    # Pole rows zero (wall BC); single Pad HLO op.
    pad_axes = ((0, 0),) * (f_upwind.ndim - 1)
    return jnp.pad(f_upwind, ((1, 1), *pad_axes))


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

    # C-grid gradient: compact stencil at face points.  ``gradient_*_cgrid``
    # natively handles 3D input (it broadcasts the lat-only metric over
    # the trailing level axis), so the previous ``moveaxis + vmap +
    # moveaxis`` round-trip was redundant — call directly on 3D.
    dp_dx = gradient_x_cgrid(p_prime_filled, grid)  # (n_lat, n_lon+1, nlev)
    dp_dy = gradient_y_cgrid(p_prime_filled, grid)  # (n_lat+1, n_lon, nlev)

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

    # --- 6. Kinetic energy gradient (from perturbation velocity) ---
    up_cell = 0.5 * (u_prime[:, :-1, :] + u_prime[:, 1:, :])
    vp_cell = 0.5 * (v_prime[:-1, :, :] + v_prime[1:, :, :])
    KE = 0.5 * (up_cell**2 + vp_cell**2)

    # ``gradient_*_cgrid`` natively handles 3D input — call directly
    # on KE instead of the moveaxis + vmap round-trip.
    dKE_dx = gradient_x_cgrid(KE, grid)
    dKE_dy = gradient_y_cgrid(KE, grid)

    # --- 7. Momentum tendencies (non-Coriolis only) ---
    du_dt = -dKE_dx - dp_dx / rho_0
    dv_dt = -dKE_dy - dp_dy / rho_0

    # --- 7b. Relative vorticity flux (issue #153, revisited in #160) ---
    # Vector-invariant advection: (u·∇)u = ∇(KE) + ζ × u.
    # Coriolis (f × u) is handled in the step function.
    #
    # NOTE (#160): This stencil is NOT a standard Sadourny/Arakawa-Hsu/
    # Arakawa-Lamb vorticity-flux scheme, despite the original commit
    # message. Specifically:
    #   (1) ζ is computed from the *perturbation* velocity u', not the
    #       total velocity, so the cross terms ζ(u')·V_bar and
    #       ζ(U_bar)·v' are missing.
    #   (2) zeta_at_u below is a 2-point meridional average (Sadourny
    #       uses a 4-point PV stencil on corner-centred q = (f+ζ)/h).
    #   (3) v_at_u uses raw v' rather than the thickness-weighted mass
    #       flux h·v' required for discrete energy consistency with
    #       the continuity equation.
    # Net effect: the term is O(Δx²)-consistent but conserves neither
    # energy nor enstrophy on the perturbation subsystem. Replacing it
    # cleanly requires the MOM6-style slow-forcing refactor tracked
    # in #160.
    zeta = curl_vertex_cgrid(u_prime, v_prime, grid)  # (n_lat+1, n_lon+1, nlev)

    # Average ζ from vertices to velocity points
    zeta_at_u = 0.5 * (zeta[:-1, :, :] + zeta[1:, :, :])  # (n_lat, n_lon+1, nlev)
    zeta_at_v = 0.5 * (zeta[:, :-1, :] + zeta[:, 1:, :])  # (n_lat+1, n_lon, nlev)

    # Average v' to u-points (4-point arithmetic mean, periodic in lon).
    # NOT thickness-weighted — see NOTE above for the consequences.
    v_west = jnp.roll(v_prime, 1, axis=1)  # v'[:, (j-1)%n_lon, :]
    v_at_u_core = 0.25 * (v_prime[:-1] + v_prime[1:]
                          + v_west[:-1] + v_west[1:])  # (n_lat, n_lon, nlev)
    v_at_u = jnp.concatenate(
        [v_at_u_core, v_at_u_core[:, 0:1, :]], axis=1)  # (n_lat, n_lon+1, nlev)

    # Average u' to v-points (4-point average, zero-padded at poles).
    # Single Pad HLO op replaces alloc-zeros + concatenate-of-three.
    u_ext = jnp.pad(u_prime, ((1, 1), (0, 0), (0, 0)))  # (n_lat+2, n_lon+1, nlev)
    u_at_v = 0.25 * (u_ext[:-1, :-1, :] + u_ext[:-1, 1:, :]
                      + u_ext[1:, :-1, :] + u_ext[1:, 1:, :])  # (n_lat+1, n_lon, nlev)

    du_dt = du_dt + zeta_at_u * v_at_u
    dv_dt = dv_dt - zeta_at_v * u_at_v

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
    # Vertical momentum advection: keep 1st-order upwind.
    # The implicit viscosity (~|w|*dz/2) provides essential damping of
    # baroclinic shear that the explicit A_v=1e-5 cannot.  Upgrading to
    # TVD removes this and causes blowup.  Proper fix: Richardson-number-
    # dependent mixing or KPP (issue #204), not higher-order advection.
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
    if config.K_v > 0 and nlev_t >= 2:
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

    # --- 10. Mixing (viscosity on perturbation velocity) ---
    # Uses the proper vector Laplacian grad(div) - k×grad(curl) directly
    # on face velocities, avoiding the lossy cell-center detour.
    # See issue #105 for details.
    if config.A_h > 0 and config.B_h > 0:
        # Both A_h Laplacian and B_h biharmonic active: the biharmonic's
        # *inner* vector Laplacian is identical to the explicit A_h
        # vector Laplacian, so compute ∇²(u', v') ONCE and feed it to
        # both branches.  Saves one full vector_laplacian_cgrid call
        # (1 div + 1 curl + 2 gradients + 2 gradient_curl_to_*) per
        # RHS evaluation.
        _vlap_u, _vlap_v = vector_laplacian_cgrid(
            u_prime, v_prime, grid,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        du_dt = du_dt + config.A_h * _vlap_u
        dv_dt = dv_dt + config.A_h * _vlap_v
        bilap_u, bilap_v = vector_laplacian_cgrid(
            _vlap_u, _vlap_v, grid,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        scale_u, scale_v = biharmonic_scaling_factor(grid)
        du_dt = du_dt - config.B_h * scale_u[:, None, None] * bilap_u
        dv_dt = dv_dt - config.B_h * scale_v[:, None, None] * bilap_v
    elif config.A_h > 0:
        vlap_u, vlap_v = vector_laplacian_cgrid(
            u_prime, v_prime, grid,
            mask=mask, u_mask=u_mask, v_mask=v_mask)
        du_dt = du_dt + config.A_h * vlap_u
        dv_dt = dv_dt + config.A_h * vlap_v
    elif config.B_h > 0:
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
            # Pad along trailing axis instead of allocating a fresh
            # ``(..., 1)`` zero buffer + 3-array concatenate.
            _pad_axes = ((0, 0),) * (flux.ndim - 1)
            flux_full = jnp.pad(flux, (*_pad_axes, (1, 1)))
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


