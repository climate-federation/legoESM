"""Boussinesq Hydrostatic Primitive Equations on the C-D grid cubed-sphere.

FV3-style C-D grid discretisation for the ocean:

* D-grid winds ``u_d, v_d`` at cell corners (shape ``(6, n+1, n+1, nlev)``)
  are the prognostic velocity variables.
* C-grid velocities at cell edges are diagnosed for mass and tracer transport.
* Vorticity is computed from the integral circulation, exact on the D-grid.
* Bernoulli gradient uses the Arakawa-Lamb 4-point formula.
* Scalars (T, S, h, eta) live at cell centres.

The ocean state containers (``OceanState``, ``OceanTendencies``) use cell-centre
velocity storage ``(6, n, n, nlev)`` for compatibility with the rest of the
ocean infrastructure (barotropic solver, conservation fixers, etc.).
Conversion between cell-centre and D-grid is done at the tendency interface.

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Griffies (2004): Fundamentals of Ocean Climate Models
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.operators_3d import hyperdiffusion_3d
from legoesm.core.operators_cdgrid import (
    center_to_dgrid_vector,
    dgrid_to_center_vector,
    dgrid_to_cgrid,
    dgrid_vorticity,
    cgrid_divergence,
    cgrid_mass_flux_divergence,
    cgrid_tracer_advection_fct,
    cgrid_wet_face_masks,
    cgrid_corner_min,
    arakawa_lamb_gradient,
    interp_center_to_corner,
    extrapolate_boundary_corners,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.ocean.eos import make_eos_fn
from legoesm.ocean.dynamics.pgf_smc03 import reconstruct_harmonic_slopes
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    OceanPartialCellCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
    compute_centroid_depth,
    extrapolate_below_seafloor,
)
from legoesm.ocean.state import OceanState, OceanTendencies, OceanConfig
from legoesm.ocean.dynamics.ocean_tendency_common import (
    bbl_distributed_drag_face_column,
    iterate_eos_and_pressure_anomaly,
    nemo_effective_bottom_drag_r,
    validate_bottom_drag_scheme,
)
from legoesm import constants


# ==============================================================================
# Vertical velocity diagnosis
# ==============================================================================

from legoesm.ocean.vertical import (
    diagnose_w_from_flux_div as _diagnose_w_from_flux_div,
    vertical_advection_ocean as _vertical_advection_ocean,
)
from legoesm.ocean.dynamics.barotropic import fill_land_cells
from legoesm.ocean.physics.mixing import laplacian_viscosity_3d, vertical_diffusion
from legoesm.grids.halo import pad_halo_4d


# ==============================================================================
# Helpers
# ==============================================================================

def _fill_inactive_per_level(
    field: jnp.ndarray, wet_3d: jnp.ndarray, grid: CubedSphereGrid,
) -> jnp.ndarray:
    """Per-level wet/rock Neumann fill of a ``(6, n, n, nlev)`` field.

    ``wet_3d`` is the 3-D wet mask (1 wet, 0 land-or-below-seafloor).  For each
    vertical level this fills every inactive cell (coastline land AND
    below-seafloor rock) from its ACTIVE same-level neighbours by ``vmap``-ing
    the 2-D :func:`fill_land_cells` over the level axis.  Unlike a single
    ``fill_land_cells(field, mask2d)`` call (which only sees the 2-D coastline
    mask, broadcast across all levels), this closes the wet/rock SEAFLOOR-STEP
    jump so the AL corner gradient never differences an active cell against a
    raw below-seafloor value — the cd-grid analogue of the lat-lon wet/rock
    FACE mask.  AD-safe (``fill_land_cells`` safe-divides via
    ``maximum(count, 1)``).  Per-level halos are issued ``nlev`` times rather
    than as one 4-D exchange; fine on a single device, an MPI follow-up.
    """
    fld = jnp.moveaxis(field, -1, 0)      # (nlev, 6, n, n)
    wet = jnp.moveaxis(wet_3d, -1, 0)
    out = jax.vmap(fill_land_cells, in_axes=(0, 0, None))(fld, wet, grid)
    return jnp.moveaxis(out, 0, -1)


def _bc_bottom_drag_cdgrid(du_dt, dv_dt, u_a, v_a, h_k, z_coord, config):
    """Bottom drag on the cd-grid cell-centre velocity.

    Cell-centre analogue of the proven lat-lon ``_bc_bottom_drag``
    (``ocean_pe_latlon_cgrid.py``): linear, MOM6 quadratic-with-floor
    (``bottom_drag_bg_velocity`` = DRAG_BG_VEL), or BBL-distributed
    (``bottom_drag_bbl_thickness``) drag at the partial-cell seafloor /
    deepest level.  No u/v-face split — cd-grid velocities live at cell
    centres, so the lat-lon face machinery collapses to a single
    cell-centre apply.  The cube cold-start instability seeds at the
    bottom/mid-depth over steep sub-grid topography, so bottom drag damps
    it exactly where it grows; it is the proven dissipation-stack piece the
    cube external-physics path was missing.

    Gated by ``config.bottom_drag_r > 0`` (default 0.0) so the existing
    cd-grid path is bit-exact when drag is off.  AD-safe: safe-divide
    ``max(h, 1e-10)``; the bottom-level selector uses the static
    ``bottom_level`` (constant w.r.t. the differentiated u,v).
    """
    r = config.bottom_drag_r
    u_bg = config.bottom_drag_bg_velocity
    H_BBL = config.bottom_drag_bbl_thickness
    _scheme = validate_bottom_drag_scheme(
        str(getattr(config, "bottom_drag_scheme", "legacy")))
    # MOM6 background-velocity floor (DRAG_BG_VEL): r_eff recovers the linear
    # ``r`` at |u| → 0 and scales as quadratic Cd·|u| at |u| ≫ u_bg.
    # u_bg = 0 → exact linear (bit-identical to the legacy single-cell form).
    if _scheme != "legacy":
        # NEMO zdfdrg drag law (np_non_lin / np_loglayer): r = Cd·|U| from
        # the BOTTOM-cell speed with the background KE ke0 in quadrature.
        # On the co-located cd grid the tracer point IS the velocity point,
        # so the lat-lon t-point construction collapses to a direct
        # cell-centre evaluation (no face averaging).
        if isinstance(z_coord, OceanPartialCellCoordinate):
            _bl = jnp.maximum(z_coord.bottom_level, 0)[..., jnp.newaxis]
            u_bot = jnp.take_along_axis(u_a, _bl, axis=-1)[..., 0]
            v_bot = jnp.take_along_axis(v_a, _bl, axis=-1)[..., 0]
            h_bot = jnp.take_along_axis(h_k, _bl, axis=-1)[..., 0]
        else:
            u_bot = u_a[..., -1]
            v_bot = v_a[..., -1]
            h_bot = h_k[..., -1]
        r_t = nemo_effective_bottom_drag_r(
            u_bot, v_bot, h_bot,
            scheme=_scheme,
            cd0=float(config.bottom_drag_cd0),
            cd_max=float(config.bottom_drag_cdmax),
            z0=float(config.bottom_drag_z0),
            ke0=float(config.bottom_drag_ke0),
            uc0=float(config.bottom_drag_uc0),
            von_karman=constants.kappa_von_karman,
        )
        r_eff_u = r_t[..., jnp.newaxis]
        r_eff_v = r_t[..., jnp.newaxis]
    elif u_bg > 0.0:
        # Co-located cell-centre velocities → the physical quadratic bottom
        # stress is the VECTOR form τ = -Cd·|u|·u (MOM6 BOTTOMDRAGLAW), one
        # coefficient from the SPEED magnitude shared by both components.  The
        # lat-lon per-component √(u²+u_bg²) form is a C-grid face-stagger
        # artifact (u,v live on different faces there); on the co-located cd
        # grid the vector speed is correct and damps the diagonal bottom flow a
        # per-component coefficient would under-damp.
        Cd_eq = r / u_bg
        speed = jnp.sqrt(u_a * u_a + v_a * v_a + u_bg * u_bg)
        r_eff_u = Cd_eq * speed
        r_eff_v = Cd_eq * speed
    else:
        r_eff_u = r
        r_eff_v = r
    if H_BBL > 0.0:
        # Distributed BBL drag (Killworth & Edwards 1999 / MOM6): spread the
        # stress over a fixed near-seafloor thickness ``H_BBL`` instead of
        # dumping r·u/h into a single (possibly <1 m) partial cell — the
        # cold-start thin-bottom-cell blowup fix.  #517: route through the
        # shared canonical helper instead of re-deriving the cell-centre
        # column form (was bit-identical to ``bbl_distributed_drag_face_column``).
        drag_u = bbl_distributed_drag_face_column(u_a, h_k, r_eff_u, H_BBL)
        drag_v = bbl_distributed_drag_face_column(v_a, h_k, r_eff_v, H_BBL)
    else:
        n_lev = u_a.shape[-1]
        level_idx = jnp.arange(n_lev)
        if isinstance(z_coord, OceanPartialCellCoordinate):
            # Apply at each column's actual seafloor (lowest active level), so
            # drag damps the bottom-trapped flow on shallow slopes, not only at
            # the deepest reference level in full columns.
            bot_lev = z_coord.bottom_level
        else:
            # z*: every wet column is full depth → deepest reference level.
            bot_lev = jnp.full(u_a.shape[:-1], n_lev - 1, dtype=jnp.int32)
        is_bot_3d = (level_idx == bot_lev[..., jnp.newaxis]).astype(u_a.dtype)
        h_drag = jnp.maximum(h_k, 1e-10)
        drag_u = -r_eff_u * u_a / h_drag * is_bot_3d
        drag_v = -r_eff_v * v_a / h_drag * is_bot_3d
    return du_dt + drag_u, dv_dt + drag_v


# ==============================================================================
# Main tendency function
# ==============================================================================

def ocean_baroclinic_tendencies_cdgrid(
    state: OceanState,
    grid: CubedSphereGrid,
    z_coord: OceanZStarCoordinate,
    cdgrid: CubedSphereCDGrid,
    config: OceanConfig = OceanConfig(),
    physics_fn=None,
    surface_forcing=None,
    dt: float | None = None,
) -> OceanTendencies:
    """Compute 3D baroclinic tendencies using C-D grid operators.

    The state uses cell-centre storage. Velocities are converted to D-grid
    for the momentum computation, then converted back.

    Parameters
    ----------
    state : OceanState
    grid : CubedSphereGrid
    z_coord : OceanZStarCoordinate
    cdgrid : CubedSphereCDGrid
    config : OceanConfig
    physics_fn : callable, optional

    Returns
    -------
    OceanTendencies
    """
    u_a = state.u.data       # (6, n, n, nlev)
    v_a = state.v.data
    T = state.T.data
    S = state.S.data
    eta = state.eta.data      # (6, n, n)
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data
    mask_3d = mask[..., jnp.newaxis]

    g = config.g
    rho_0 = config.rho_0
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta.dtype)
    eta_floor = min_water_col - H_bathy
    eta_safe = jnp.maximum(eta, eta_floor) * mask

    # --- 1. Layer thickness and Jacobian ---
    J = compute_ocean_jacobian(
        eta_safe, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )
    h_k = compute_layer_thickness(
        eta_safe, H_bathy, z_coord,
        min_water_column_m=config.min_water_column_m,
    )

    # --- 1b. Partial-cell below-seafloor handling (C-D grid) ---
    # With an ``OceanPartialCellCoordinate`` the deepest active level per
    # column is ``bottom_level``; cells beneath it carry ``h_partial = 0`` and
    # must be inert.  Mirror the FC backend: hold below-seafloor velocities at
    # zero at the source (so the D-grid conversion, vorticity, KE, divergence
    # and tracer advection all see a dead zone rather than stale IC values),
    # and extrapolate T,S into the rock so the EOS density and the Arakawa-Lamb
    # horizontal stencils cannot import a below-seafloor value into an active
    # cell next to a shallower column.  For a pure ``OceanZStarCoordinate``
    # every wet column is full-depth, so this branch is skipped and the legacy
    # cd-grid path stays BIT-EXACT (the 2D land mask still removes dry columns).
    is_partial = isinstance(z_coord, OceanPartialCellCoordinate)
    if is_partial:
        active_3d = z_coord.is_active.astype(u_a.dtype)
        u_a = u_a * active_3d
        v_a = v_a * active_3d
        T = extrapolate_below_seafloor(T, z_coord)
        S = extrapolate_below_seafloor(S, z_coord)
        # KNOWN LIMITATION (conservation): zeroing the A-cell velocity does not
        # *strictly* close the C-grid face between an active cell and a
        # below-seafloor cell — the dgrid→cgrid averaging can leave a small
        # nonzero face velocity, so ``cgrid_mass_flux_divergence`` /
        # ``cgrid_tracer_advection_fct`` carry a small spurious flux across the
        # seafloor step (the final ``active_3d`` tendency gate keeps the
        # below-seafloor cell inert, so the leak only perturbs the active
        # neighbour's η/w/tracer at O(coastline-error)).  This is the SAME
        # fidelity as the backend's existing horizontal coastline treatment
        # (land u_a is likewise only zeroed, not face-closed).  A strict C-face
        # wet/rock mask (face active iff BOTH adjacent A-cells active, with a
        # cross-seam ``is_active`` halo) is the next conservation upgrade and
        # closes coastline + seafloor faces together; tracked in
        # docs/dev-notes/ocean_faithfulness_nemo.md.

    # --- 2. Density from EOS + 3. Baroclinic pressure anomaly ---
    # Reference Jacobian (J=1, eta=0): the barotropic solver handles
    # -g*grad(eta) and using the actual J here would double-count the
    # free-surface contribution (see #109).
    #
    # The cubed-sphere path runs the cumsum in float64: at depth p has
    # ULP = 0.0625 Pa in float32, so the halo-exchange interpolation
    # of float32 values at face boundaries leaks O(ULP/dx) ≈ 2e-7 Pa/m
    # — a spurious PGF that drives rest-state instability.  Keeping
    # p_prime in float64 reduces the leak by 9 orders of magnitude.
    # ``fill_land_cells`` is ndim-aware: it uses ``pad_halo_4d`` for 4D
    # input so all vertical levels share one MPI halo exchange per pass
    # (instead of nlev separate exchanges under the prior vmap).
    fill_TS = lambda field: fill_land_cells(field, mask, grid)
    eos_fn = make_eos_fn(config.eos, getattr(config, 'eos_linear', None))
    # Partial cells: pass the eta-INDEPENDENT reference thickness
    # ``z_coord.h_partial`` (Σ_k = H_bathy per column) as ``h_actual`` so the
    # hydrostatic EOS pressure and baroclinic-anomaly cumsum integrate to each
    # cell's TRUE *reference* centroid depth, accounting for the partial bottom
    # cell.  Must NOT pass the eta-stretched ``h_k`` here: the J=1/eta=0
    # reference-pressure contract above (#109) reserves the free-surface
    # ``-g·grad(eta)`` for the barotropic solver, so an eta-dependent thickness
    # would reintroduce SSH into the baroclinic pressure and double-count the
    # barotropic forcing.  Matches the proven latlon C-grid backend
    # (ocean_pe_latlon_cgrid.py).  Below-seafloor cells carry ``h_partial = 0``
    # and contribute nothing.  ``h_actual=None`` keeps the legacy z* path
    # bit-exact.
    rho, rho_prime, p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask, fill_TS, eos_fn, z_coord.dz_ref, rho_0, g,
        n_iter=2, hi_precision_pressure=True,
        h_actual=(z_coord.h_partial if is_partial else None),
        is_active_3d=(active_3d if is_partial else None),
        allow_baroclinic_f32=True,   # opt-in f32-EOS lever (LEGOESM_BAROCLINIC_F32)
    )

    # --- 4. Convert to D-grid ---
    u_d, v_d = center_to_dgrid_vector(u_a * mask_3d, v_a * mask_3d, cdgrid)

    # --- 5. C-grid velocities for mass transport ---
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)
    if is_partial:
        # Strict wet/rock C-face closure (codex HIGH): zeroing the A-cell
        # velocity does not by itself close the C-face between an active cell
        # and a below-seafloor (or coastline) cell — the d->c average can leave
        # a small nonzero face velocity, so the mass-flux divergence (w), the
        # velocity divergence, and the tracer advection (all consume u_c/v_c)
        # carry a spurious flux across the seafloor step.  Mask each C-face to
        # wet iff BOTH adjacent A-cells are wet (land_mask AND above seafloor),
        # halo-correctly across cube seams.  This is the conservation upgrade
        # tracked in docs/dev-notes/ocean_faithfulness_nemo.md and closes the
        # coastline + seafloor faces together.  z* path (is_partial False)
        # stays bit-exact (no masking).
        wet_cc_3d = mask_3d * active_3d
        mask_uc, mask_vc = cgrid_wet_face_masks(wet_cc_3d, cdgrid)
        u_c = u_c * mask_uc
        v_c = v_c * mask_vc

    # --- 6. Flux divergence for vertical velocity ---
    # Use cell-centre for flux divergence (cell-centre h_k and velocities).
    # ``cgrid_mass_flux_divergence`` reconstructs h at the faces and returns the
    # THICKNESS-WEIGHTED divergence ∇·(h_k u_k) [m/s] (it is passed ``h_k``).
    # ``diagnose_w_from_flux_div`` must therefore be told ``thickness_weighted=
    # True`` so it cumsums ∇·(h u) DIRECTLY; the default (False) would multiply
    # by ``dz_ref`` a SECOND time, inflating w by a factor ~layer-thickness
    # (tens–hundreds of m) → a huge spurious vertical velocity → vertical
    # advection blowup (seeds at the deepest level where the bottom-up cumsum is
    # largest).  Matches the proven latlon C-grid (ocean_pe_latlon_cgrid.py:911)
    # and mpas (ocean_pe_mpas.py:266) backends, which both pass the flag.
    flux_div_k = cgrid_mass_flux_divergence(
        h_k, u_c, v_c, cdgrid,
    )
    w = _diagnose_w_from_flux_div(flux_div_k, z_coord, thickness_weighted=True)

    # --- 7. Velocity divergence for skew-symmetric correction ---
    div_v = cgrid_divergence(u_c, v_c, cdgrid)

    # --- 8. Vorticity ---
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)

    # --- 9. KE at cell centres from D-grid (orthogonal basis) ---
    u_cc_ke, v_cc_ke = dgrid_to_center_vector(u_d, v_d)
    KE = 0.5 * (u_cc_ke ** 2 + v_cc_ke ** 2)

    # --- 10. Bernoulli and pressure gradients at D-grid corners ---
    # Fill land cells in p_prime before gradient so the 4-point stencil
    # sees smooth values at coastlines instead of the ocean-to-zero jump.
    # ``fill_land_cells`` natively handles 4D input (single halo exchange
    # across all levels), so call it directly.
    p_prime_filled = fill_land_cells(p_prime, mask, grid)
    # Batch the two Arakawa-Lamb gradients (KE, p_prime_filled) into a
    # single call — both are 3D scalar fields on (face, n, n, nlev) and
    # the operator treats the trailing axis as a passive batch.  Same
    # exploit as Loop 119 (CD-grid CE) for K and pi_prime.  2 gradients
    # → 1 (one halo exchange + one 4-point finite-difference + one 2x2
    # metric-matrix multiply on the thicker tensor).
    n_face_kp, n_i_kp, n_j_kp, nlev_kp = KE.shape
    _kp_stack = jnp.stack([KE, p_prime_filled], axis=-1)
    _kp_flat = _kp_stack.reshape(n_face_kp, n_i_kp, n_j_kp, nlev_kp * 2)
    _dkp_dx_flat, _dkp_dy_perp_flat = arakawa_lamb_gradient(_kp_flat, cdgrid)
    _dkp_dx = _dkp_dx_flat.reshape(
        _dkp_dx_flat.shape[0], _dkp_dx_flat.shape[1],
        _dkp_dx_flat.shape[2], nlev_kp, 2,
    )
    _dkp_dy_perp = _dkp_dy_perp_flat.reshape(
        _dkp_dy_perp_flat.shape[0], _dkp_dy_perp_flat.shape[1],
        _dkp_dy_perp_flat.shape[2], nlev_kp, 2,
    )
    dKE_dx = _dkp_dx[..., 0]
    dp_dx = _dkp_dx[..., 1]
    dKE_dy_perp = _dkp_dy_perp[..., 0]
    dp_dy_perp = _dkp_dy_perp[..., 1]
    # Downcast PGF results back to working precision
    dp_dx = dp_dx.astype(T.dtype)
    dp_dy_perp = dp_dy_perp.astype(T.dtype)

    # --- 10b. Partial-cell horizontal PGF correction (cd-grid AL corners) ---
    # On partial-cell topography the 4 cells around a D-grid corner sit at
    # DIFFERENT geometric centroid depths, so the plain AL gradient of p_prime
    # (section 10, differenced at the same level index k) carries a residual
    # partial-cell PGF error → spurious bottom-trapped flow.  VERIFIED to seed
    # the cd-grid cube cold-start blowup: a vertically-STRATIFIED rest column
    # over the real bathy with ZERO forcing (exact PGF must be 0, ∇_h ρ=0 at
    # constant z) blows up at the bottom level ~step 180 — the canonical
    # Adcroft-Campin partial-cell rest test.  Two schemes (config.pgf_scheme),
    # BOTH gated on ``is_partial`` so the pure-z* path is bit-exact (z* never
    # enters this branch).  Both decompose the corner gradient EXACTLY into
    # existing-operator calls because ``arakawa_lamb_gradient`` is LINEAR and
    # the corner reference depth z_ref is constant across a corner's 4 cells.
    # z_ref uses the eta=0 / J=1 reference centroid (matching p_prime → the
    # correction is eta-independent) measured as an ANOMALY from the full-column
    # reference centroid ``cref`` (cumsum(dz_ref) − 0.5·dz_ref) so it is
    # IDENTICALLY zero on full/flat columns (cent_anom ≡ 0 → z_ref ≡ 0): the
    # flat-bottom partial path stays bit-exact vs z* and the smc03 quadratic
    # ``z_ref²`` term cannot amplify FP noise there.
    #   "adcroft": Adcroft & Campin 2004 LINEAR depth shift
    #              p_eff = p' − g·rho'·(centroid − z_ref) ADDED to dp:
    #              corr = −AL_grad(g·rho'·centroid) + z_ref·AL_grad(g·rho').
    #              Leaves a 2nd-order residual for a stratified column (the
    #              spurious bottom PGF above) → not faithful on the cube.
    #   "smc03"  : full Shchepetkin & McWilliams 2003 density-Jacobian PGF
    #              REPLACING dp — see below.  Matches the proven latlon/tripole.
    pgf_scheme = getattr(config, "pgf_scheme", "adcroft")
    if pgf_scheme not in ("adcroft", "smc03", "zero"):
        # Static config value -> validate at fn entry UNCONDITIONALLY. A pure
        # z* run has is_partial=False, so gating this guard inside `if
        # is_partial` let a typo (e.g. 'smc3') silently fall back to adcroft and
        # disable the faithful scheme on z*; a typo must fail loudly on every
        # vertical coordinate (matches the latlon/mpas siblings).
        raise ValueError(
            f"Unknown cd-grid pgf_scheme {pgf_scheme!r}; "
            "expected 'adcroft', 'smc03', or 'zero' (diagnostic).")
    if is_partial:
        cref = jnp.cumsum(z_coord.dz_ref) - 0.5 * z_coord.dz_ref
        if pgf_scheme == "smc03":
            # S&M03 density-Jacobian PGF on the AL corners.  Per-cell geometry
            # from the partial thicknesses (kernel convention z_c = z_top + h/2,
            # positive down, eta=0).  rho_prime is the baroclinic anomaly already
            # extrapolated into the rock (section 1b) and sigma=0 below seafloor,
            # so the AL stencil never imports a raw below-seafloor density.
            h_part = z_coord.h_partial
            z_top = jnp.cumsum(h_part, axis=-1) - h_part
            z_c = z_top + 0.5 * h_part
            sigma = reconstruct_harmonic_slopes(
                rho_prime, z_c, active_3d,
                bottom_slope_2nd_order=config.smc03_bottom_2nd_order,
            )
            cell_dP = g * h_part * rho_prime
            P_top = jnp.cumsum(cell_dP, axis=-1) - cell_dP
            cent_anom = z_c - cref
            # Wet-aware corner reference (codex HIGH): exclude below-seafloor
            # (inactive) cells from the corner-min so z_ref is set by an ACTIVE
            # cell only.  Active cells at level k share the z* cell-top and have
            # cent_anom <= 0 (full cells 0, partial-bottom cells negative), so a
            # BOUNDED positive sentinel (max|cent_anom|+1, ~H_bathy scale) makes
            # inactive cells never the min while staying finite — the z_ref^2
            # term in the decomposition then cannot overflow (a large sentinel
            # leaks huge values through the AL halo / corner->centre average).
            # All-rock corners take the sentinel but carry no active neighbour
            # (sigma=0 -> c=0 there) and are gated to zero at section 18.
            _sent = jnp.max(jnp.abs(cent_anom)) + 1.0
            _ca_active = jnp.where(active_3d > 0.5, cent_anom, _sent)
            z_ref_corner = cgrid_corner_min(_ca_active, cdgrid)
            # The in-cell-k pressure reconstruction
            #   P_k(z) = P_top_k + g·(z − z_top)·[rho' + 0.5·σ·(z + z_top − 2 z_c)]
            # is QUADRATIC in z; rewrite in the centroid anomaly ẑ = z − cref as
            #   P_k = a + b·ẑ + c·ẑ²
            # so the corner gradient at the corner-common z_ref decomposes into
            #   dp = AL_grad(a) + z_ref·AL_grad(b) + z_ref²·AL_grad(c).
            # For a horizontally-uniform stratification a,b become horizontally
            # uniform among the active cells (shifted-centroid columns reconstruct
            # ρ identically — the S&M03 harmonic-slope property), so AL_grad → 0
            # and the rest-state PGF vanishes (the Adcroft linear shift does not
            # achieve this for stratified columns).  Evaluating P_k in-cell (not
            # via the argmax-enclosing-cell of compute_pressure_at_target_smc03)
            # is what makes the polynomial decomposition exact; z_ref = corner-min
            # of the 4 cent_anom keeps the target inside every cell's range
            # (the shallower-centroid convention; avoids the asymmetric seafloor
            # clamp documented for the latlon midpoint target).
            a_p = (P_top + g * (cref - z_top)
                   * (rho_prime + 0.5 * sigma * (cref + z_top - 2.0 * z_c)))
            b_p = g * (rho_prime + sigma * (cref - z_c))
            c_p = 0.5 * g * sigma
            # Wet/rock HORIZONTAL closure of the polynomial coefficients before
            # the AL gradient: fill EVERY inactive cell (coastline land AND
            # below-seafloor rock) at each level from its active same-level
            # neighbours, using the 3-D wet mask ``wet_cc_3d`` (= mask·active_3d,
            # built for the C-face closure in §5).  This is what removes the
            # seafloor-STEP PGF: a corner straddling an active/rock step would
            # otherwise difference a's reconstruction against a raw rock value,
            # leaving a spurious bottom-trapped PGF (the cd-grid analogue of the
            # lat-lon wet/rock FACE mask).  The coefficients are filled (NOT a
            # large sentinel — that overflowed via the z_ref^2 term), so on a
            # horizontally-uniform stratification the filled inactive cells match
            # their active neighbours and AL_grad(a,b,c) → 0 across the step.
            a_f = _fill_inactive_per_level(a_p, wet_cc_3d, grid)
            b_f = _fill_inactive_per_level(b_p, wet_cc_3d, grid)
            c_f = _fill_inactive_per_level(c_p, wet_cc_3d, grid)
            n_f_s, n_i_s, n_j_s, nlev_s = a_f.shape
            _abc_flat = jnp.stack([a_f, b_f, c_f], axis=-1).reshape(
                n_f_s, n_i_s, n_j_s, nlev_s * 3)
            _dabc_dx_flat, _dabc_dy_flat = arakawa_lamb_gradient(_abc_flat, cdgrid)
            _dabc_dx = _dabc_dx_flat.reshape(
                _dabc_dx_flat.shape[0], _dabc_dx_flat.shape[1],
                _dabc_dx_flat.shape[2], nlev_s, 3)
            _dabc_dy = _dabc_dy_flat.reshape(
                _dabc_dy_flat.shape[0], _dabc_dy_flat.shape[1],
                _dabc_dy_flat.shape[2], nlev_s, 3)
            zr = z_ref_corner
            dp_dx = (_dabc_dx[..., 0] + zr * _dabc_dx[..., 1]
                     + zr * zr * _dabc_dx[..., 2]).astype(T.dtype)
            dp_dy_perp = (_dabc_dy[..., 0] + zr * _dabc_dy[..., 1]
                          + zr * zr * _dabc_dy[..., 2]).astype(T.dtype)
        elif pgf_scheme == "adcroft":
            # Adcroft & Campin 2004 linear depth-shift correction ADDED to the
            # section-10 plain AL gradient.  Mirrors the latlon adcroft path
            # (ocean_pe_latlon_cgrid.py:1139).
            centroid0 = compute_centroid_depth(
                jnp.zeros_like(eta), H_bathy, z_coord,
                min_water_column_m=config.min_water_column_m,
            )
            cent_anom = centroid0 - cref
            g_rho = fill_land_cells(g * rho_prime, mask, grid)
            g_rho_cent = fill_land_cells(g * rho_prime * cent_anom, mask, grid)
            z_ref_corner = cgrid_corner_min(cent_anom, cdgrid)
            n_f_c, n_i_c, n_j_c, nlev_c = g_rho.shape
            _gr_flat = jnp.stack([g_rho, g_rho_cent], axis=-1).reshape(
                n_f_c, n_i_c, n_j_c, nlev_c * 2)
            _dgr_dx_flat, _dgr_dy_flat = arakawa_lamb_gradient(_gr_flat, cdgrid)
            _dgr_dx = _dgr_dx_flat.reshape(
                _dgr_dx_flat.shape[0], _dgr_dx_flat.shape[1],
                _dgr_dx_flat.shape[2], nlev_c, 2)
            _dgr_dy = _dgr_dy_flat.reshape(
                _dgr_dy_flat.shape[0], _dgr_dy_flat.shape[1],
                _dgr_dy_flat.shape[2], nlev_c, 2)
            corr_dx = (-_dgr_dx[..., 1] + z_ref_corner * _dgr_dx[..., 0]).astype(T.dtype)
            corr_dy = (-_dgr_dy[..., 1] + z_ref_corner * _dgr_dy[..., 0]).astype(T.dtype)
            dp_dx = dp_dx + corr_dx
            dp_dy_perp = dp_dy_perp + corr_dy

    # --- 10c. PGF-ZERO falsification diagnostic (config.pgf_scheme == "zero") ---
    # Remove the horizontal pressure force ENTIRELY.  A rest state then has NO
    # horizontal force at all, so if the cube cold-start stays stable with
    # pgf_scheme="zero" but blows with adcroft/smc03, the partial-cell PGF
    # residual is the SOLE cause (vs any barotropic / advective / metric term).
    # Works for both partial and z* (zeroes the base AL gradient + any
    # correction).  Diagnostic only — never a faithful run.
    if pgf_scheme == "zero":
        dp_dx = jnp.zeros_like(dp_dx)
        dp_dy_perp = jnp.zeros_like(dp_dy_perp)

    # --- 11. Vorticity + divergence at corners (batched) ---
    # Batch the (zeta, div_v) center-to-corner interpolation: both are
    # cell-centre (face, n, n, nlev) fields and the operator treats
    # the trailing axis as a passive batch.  Stack and fold so a
    # single halo + 4-point average serves both interps.  Same
    # passive-trailing-axis pattern as the CD-grid PE corner interps.
    n_face_zd, n_i_zd, n_j_zd, nlev_zd = zeta.shape
    _zd_stack = jnp.stack([zeta, div_v], axis=-1)
    _zd_corner_flat = interp_center_to_corner(
        _zd_stack.reshape(n_face_zd, n_i_zd, n_j_zd, nlev_zd * 2), cdgrid,
    )
    _zd_corner = _zd_corner_flat.reshape(
        _zd_corner_flat.shape[0], _zd_corner_flat.shape[1],
        _zd_corner_flat.shape[2], nlev_zd, 2,
    )
    zeta_corner = _zd_corner[..., 0]
    div_corner = _zd_corner[..., 1]
    f_corner_3d = cdgrid.f_corner[:, :, :, None]   # (6, n+1, n+1, 1)

    # --- 12. Baroclinic Coriolis split ---
    # Planetary Coriolis: barotropic part (f*v_bar) handled by barotropic
    # substeps; here only the baroclinic deviation is included.
    # H_total + U_bar + V_bar all reduce ``... * h_k`` over the level
    # axis — fuse into one stacked column reduction.
    _bar_triple = jnp.sum(
        jnp.stack([h_k, u_a * h_k, v_a * h_k], axis=-1), axis=-2,
    )
    H_total = jnp.maximum(_bar_triple[..., 0], min_water_col)
    U_bar_a = _bar_triple[..., 1] / H_total * mask
    V_bar_a = _bar_triple[..., 2] / H_total * mask
    u_prime_a = (u_a - U_bar_a[..., jnp.newaxis]) * mask_3d
    v_prime_a = (v_a - V_bar_a[..., jnp.newaxis]) * mask_3d
    u_prime_d, v_prime_d = center_to_dgrid_vector(u_prime_a, v_prime_a, cdgrid)

    # --- 13. D-grid momentum tendencies ---
    # ζ*v + f*v' (relative vorticity × full velocity, Coriolis × deviation)
    du_d_dt = (zeta_corner * v_d + f_corner_3d * v_prime_d
               - dKE_dx - dp_dx / rho_0)
    dv_d_dt = (-zeta_corner * u_d - f_corner_3d * u_prime_d
               - dKE_dy_perp - dp_dy_perp / rho_0)

    # Skew-symmetric correction (``div_corner`` was already computed
    # alongside ``zeta_corner`` via the batched corner interpolation).
    du_d_dt = du_d_dt - 0.5 * u_d * div_corner
    dv_d_dt = dv_d_dt - 0.5 * v_d * div_corner

    # Boundary-corner fix: replace face-boundary corner tendencies with
    # nearest-interior values to eliminate O(dx) halo interpolation error
    # (mirrors atmosphere fix from commit f3f9a86).
    du_d_dt, dv_d_dt = extrapolate_boundary_corners(du_d_dt, dv_d_dt, cdgrid.n)

    # --- 14. Convert D-grid tendencies back to cell-centre ---
    du_dt, dv_dt = dgrid_to_center_vector(du_d_dt, dv_d_dt)

    # --- 15. Vertical advection of u, v (cell-centre) ---
    # Batch the two ``_vertical_advection_ocean`` calls by stacking
    # (u_a, v_a) along a new leading axis.  ``w_full`` / ``jac_safe`` /
    # ``dz_half`` depend only on (w, z_coord, J), so they are
    # computed once and the trailing-axis ``[..., :-1] - [..., 1:]``
    # upwind gradient broadcasts across the new axis.  Same
    # leading-axis batching as Loop 142 in CD-grid CE / PE.
    _uv_a_va = jnp.stack([u_a, v_a], axis=0)
    _uv_a_va_adv = _vertical_advection_ocean(_uv_a_va, w, z_coord, J)
    du_dt = du_dt + _uv_a_va_adv[0]
    dv_dt = dv_dt + _uv_a_va_adv[1]

    # --- 16. Tracer tendencies ---
    # Use C-grid velocities for upwind advection of tracers at cell centres.
    # Stack T, S along a trailing tracer axis and fold it into the level
    # axis so the halo-issuing operators (cgrid_tracer_advection_fct,
    # laplacian_viscosity_3d) run ONCE for both tracers instead of being
    # called twice under vmap-over-(T,S) — each vmap'd call would emit
    # its own pad_halo_4d MPI exchange.  Vertical operators stay
    # per-tracer because they hard-code the vertical axis at -1.
    tracer_stack = jnp.stack([T, S], axis=-1)  # (6, n, n, nlev, 2)
    n_face, n_i, n_j, nlev_t, n_tracers = tracer_stack.shape
    tracer_flat = tracer_stack.reshape(n_face, n_i, n_j, nlev_t * n_tracers)
    # Broadcast C-grid velocities across the combined (level × tracer)
    # axis.  ``tracer_flat`` reshape interleaves levels and tracers as
    # ``[lev0/trc0, lev0/trc1, ..., lev1/trc0, ...]`` — each level's
    # velocity must be duplicated ``n_tracers`` times to align, which
    # ``jnp.repeat`` does directly.  ``jnp.tile`` would instead
    # concatenate the entire array and mis-align tracer ↔ level.
    if n_tracers == 1:
        u_c_b, v_c_b = u_c, v_c
    else:
        u_c_b = jnp.repeat(u_c, n_tracers, axis=-1)
        v_c_b = jnp.repeat(v_c, n_tracers, axis=-1)

    horiz_flat = cgrid_tracer_advection_fct(tracer_flat, u_c_b, v_c_b, cdgrid)
    if config.K_h > 0:
        horiz_flat = horiz_flat + laplacian_viscosity_3d(
            tracer_flat, grid, config.K_h,
        )
    horiz_stack = horiz_flat.reshape(n_face, n_i, n_j, nlev_t, n_tracers)

    # Vertical advection per-tracer (vmap over the trailing tracer axis
    # so JAX produces one batched kernel rather than n_tracers unrolled
    # stencils).
    def _vert_adv(q):
        return _vertical_advection_ocean(q, w, z_coord, J)

    vert_adv_stack = jax.vmap(_vert_adv, in_axes=-1, out_axes=-1)(tracer_stack)

    if config.K_v > 0:

        def _vdiff(q):
            return vertical_diffusion(q, z_coord, J, config.K_v)

        vdiff_stack = jax.vmap(_vdiff, in_axes=-1, out_axes=-1)(tracer_stack)
        tracer_tend_stack = horiz_stack + vert_adv_stack + vdiff_stack
    else:
        tracer_tend_stack = horiz_stack + vert_adv_stack

    dT_dt = tracer_tend_stack[..., 0]
    dS_dt = tracer_tend_stack[..., 1]

    # --- 17. Mixing (always applied from config, grid-native operators) ---
    # Stack u, v along a trailing axis and fold it into the level dim so
    # the halo-issuing horizontal viscosity operators
    # (``laplacian_viscosity_3d``, ``hyperdiffusion_3d``) run ONCE on the
    # thicker (6, n, n, nlev*2) field instead of issuing two separate
    # pad_halo_4d MPI exchanges per call.  Vertical diffusion stays
    # per-component (axis -1 = nlev hard-coded, no halo).
    n_face_v, n_i_v, n_j_v, nlev_v = u_a.shape
    if config.A_h > 0 or config.hyperdiff_coeff > 0:
        vel_masked_stack = jnp.stack(
            [u_a * mask_3d, v_a * mask_3d], axis=-1,
        )  # (6, n, n, nlev, 2)
        vel_masked_flat = vel_masked_stack.reshape(
            n_face_v, n_i_v, n_j_v, nlev_v * 2,
        )
        # Pre-pad the (u, v)-stack ONCE so the explicit Laplacian
        # (``laplacian_viscosity_3d``) and the inner Laplacian of the
        # biharmonic hyperdiffusion (``hyperdiffusion_3d``) share the
        # same halo on ``vel_masked_flat`` instead of issuing two
        # independent ``pad_halo_4d`` collectives on the same input.
        # Saves 1 MPI message per RHS evaluation when both A_h and
        # hyperdiff_coeff are non-zero — the dominant ocean test config.
        _dg_oc = getattr(grid, 'duogrid', None)
        _offsets_oc = None if _dg_oc is not None else grid.halo_interp_offsets
        vel_masked_pad = pad_halo_4d(
            vel_masked_flat, interp_offsets=_offsets_oc, duogrid=_dg_oc,
        )
    if config.A_h > 0:
        vel_lap_flat = laplacian_viscosity_3d(
            vel_masked_flat, grid, config.A_h, padded=vel_masked_pad,
        )
        vel_lap = vel_lap_flat.reshape(n_face_v, n_i_v, n_j_v, nlev_v, 2)
        du_dt = du_dt + vel_lap[..., 0]
        dv_dt = dv_dt + vel_lap[..., 1]
    if config.A_v > 0:

        def _vdiff_uv(q):
            return vertical_diffusion(q, z_coord, J, config.A_v)

        vel_uv = jnp.stack([u_a, v_a], axis=-1)  # (6, n, n, nlev, 2)
        vel_vdiff = jax.vmap(_vdiff_uv, in_axes=-1, out_axes=-1)(vel_uv)
        du_dt = du_dt + vel_vdiff[..., 0]
        dv_dt = dv_dt + vel_vdiff[..., 1]
    if config.hyperdiff_coeff > 0:
        vel_hyper_flat = hyperdiffusion_3d(
            vel_masked_flat, grid, config.hyperdiff_coeff,
            padded=vel_masked_pad,
        )
        vel_hyper = vel_hyper_flat.reshape(n_face_v, n_i_v, n_j_v, nlev_v, 2)
        du_dt = du_dt + vel_hyper[..., 0]
        dv_dt = dv_dt + vel_hyper[..., 1]

    # --- 17a. Bottom drag (cd-grid cell-centre) ---
    # The proven dissipation-stack piece the cube external-physics path was
    # missing.  Runs UNCONDITIONALLY here (not behind ``physics_fn is None``) so
    # the OMIP cube run — whose physics_fn carries no drag (BottomDragConfig
    # scheme="none") — still gets it.  Gated by ``bottom_drag_r`` → off =
    # bit-exact.  ``u_a``/``v_a`` are already zeroed below seafloor (partial),
    # and the final active_3d gate keeps the rock inert.
    if (config.bottom_drag_r > 0
            or validate_bottom_drag_scheme(
                str(getattr(config, "bottom_drag_scheme", "legacy")))
            != "legacy"):
        du_dt, dv_dt = _bc_bottom_drag_cdgrid(
            du_dt, dv_dt, u_a, v_a, h_k, z_coord, config,
        )

    # --- 17b. Physics tendencies (surface forcing, bottom drag, etc.) ---
    if physics_fn is not None:
        # dt (the run's timestep) is forwarded only when known, so custom
        # physics_fns without a dt argument keep working on dt-free calls.
        if dt is None:
            phys = physics_fn(state, grid, z_coord, surface_forcing)
        else:
            phys = physics_fn(state, grid, z_coord, surface_forcing, dt=dt)
        du_dt = du_dt + phys.du_dt.data
        dv_dt = dv_dt + phys.dv_dt.data
        dT_dt = dT_dt + phys.dT_dt.data
        dS_dt = dS_dt + phys.dS_dt.data

    # --- 18. Land masking (+ partial-cell below-seafloor gating) ---
    du_dt = du_dt * mask_3d
    dv_dt = dv_dt * mask_3d
    dT_dt = dT_dt * mask_3d
    dS_dt = dS_dt * mask_3d
    if is_partial:
        # Zero below-seafloor tendencies so the rock stays inert (the
        # extrapolation-fill kept active cells uncontaminated; this keeps the
        # dead cells from accumulating spurious tendencies through the
        # vertical operators / corner interpolation).
        du_dt = du_dt * active_3d
        dv_dt = dv_dt * active_3d
        dT_dt = dT_dt * active_3d
        dS_dt = dS_dt * active_3d

    # --- 19. Free-surface tendency ---
    deta_dt = -jnp.sum(flux_div_k, axis=-1) * mask

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    return OceanTendencies(
        du_dt=Field(data=du_dt, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims_3d, units="m/s^2"),
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
