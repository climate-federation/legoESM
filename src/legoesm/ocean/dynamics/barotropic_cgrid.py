"""C-grid barotropic solver for the cubed-sphere ocean model.

Replaces the A-grid barotropic solver when ``OceanConfig.barotropic_staggering
== "c_grid"``.  Uses the C-grid infrastructure from ``operators_cdgrid.py``
and ``cubed_sphere_cdgrid.py`` to eliminate the 2*dx checkerboard null space
inherent in the A-grid formulation.

Forward-backward (Matsuno) substeps:

    d(eta)/dt = -div(H * u_c, H * v_c)             [cell centres]
    d(u_c)/dt = f * v_at_u - g * d(eta)/dx          [u-edge midpoints]
    d(v_c)/dt = -f * u_at_v - g * d(eta)/dy         [v-edge midpoints]

Coriolis is computed at cell centres (where halo exchange is well-tested)
and projected to C-grid edges via ``fv3_cc2c`` (vector halo exchange with
non-orthogonality correction), avoiding cross-face staggered interpolation.

External API is cell-centre: A-grid velocities are projected to C-grid
on entry and back to cell centres on exit.  No changes to OceanState,
physics, coupler, or ML interfaces.

See issue #182 for motivation and design rationale.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.operators_cdgrid import (
    cgrid_divergence, cgrid_gradient_2d, fv3_cc2c,
    center_to_dgrid_vector, dgrid_to_center_vector,
)
from legoesm.core.precision import cast
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import CubedSphereCDGrid
from legoesm.grids.halo import pad_halo
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.state import OceanState, OceanConfig
from legoesm.ocean.dynamics.eta_floor import clamp_and_redistribute as _clamp_redistribute


# ==============================================================================
# Edge mask derivation
# ==============================================================================

def _derive_edge_masks(mask, grid):
    """Derive C-grid edge masks from cell-centre land mask.

    A u-edge (between cells i-1 and i) is ocean if both flanking cells
    are ocean.  Same logic for v-edges.

    Parameters
    ----------
    mask : jax.Array, shape (6, n, n)
        Cell-centre land mask (1 = ocean, 0 = land).
    grid : CubedSphereGrid

    Returns
    -------
    u_mask : jax.Array, shape (6, n+1, n)
    v_mask : jax.Array, shape (6, n, n+1)
    """
    mask_pad = pad_halo(mask, interp_offsets=None)  # (6, n+2, n+2)
    # u-edge at (i, j): between cell (i-1, j) and cell (i, j)
    u_mask = mask_pad[:, :-1, 1:-1] * mask_pad[:, 1:, 1:-1]   # (6, n+1, n)
    # v-edge at (i, j): between cell (i, j-1) and cell (i, j)
    v_mask = mask_pad[:, 1:-1, :-1] * mask_pad[:, 1:-1, 1:]   # (6, n, n+1)
    return u_mask, v_mask


# ==============================================================================
# C-grid ↔ cell-centre velocity helpers
# ==============================================================================

def _cgrid_to_cell_centre(u_c, v_c):
    """Average C-grid edge-normal velocities back to cell centres.

    Simple 2nd-order average of flanking edge values.  Used for
    reconstructing cell-centre velocity from C-grid for Coriolis
    computation and for the final 3D velocity correction.

    Parameters
    ----------
    u_c : jax.Array, shape (6, n+1, n)
    v_c : jax.Array, shape (6, n, n+1)

    Returns
    -------
    u_cc, v_cc : jax.Array, shape (6, n, n)
    """
    u_cc = 0.5 * (u_c[:, :-1] + u_c[:, 1:])
    v_cc = 0.5 * (v_c[:, :, :-1] + v_c[:, :, 1:])
    return u_cc, v_cc


# ==============================================================================
# Flux-form diffusion on eta (conservative)
# ==============================================================================

def _diffuse_eta_flux_form(eta, u_mask, v_mask, nu_dt_u, nu_dt_v, cdgrid):
    """Conservative flux-form Laplacian diffusion on eta.

    div(nu * grad(eta)) computed with C-grid compact operators so that
    the divergence theorem is exactly satisfied (sum of div*area = 0).

    Parameters
    ----------
    eta : shape (6, n, n)
    u_mask, v_mask : edge masks
    nu_dt_u : shape (6, n+1, n)
    nu_dt_v : shape (6, n, n+1)
    cdgrid : CubedSphereCDGrid

    Returns
    -------
    eta_diffused : shape (6, n, n)
    """
    deta_dx, deta_dy = cgrid_gradient_2d(eta, cdgrid)
    flux_x = nu_dt_u * deta_dx * u_mask
    flux_y = nu_dt_v * deta_dy * v_mask
    return eta + cgrid_divergence(flux_x, flux_y, cdgrid)


# ==============================================================================
# C-grid barotropic substeps
# ==============================================================================

def barotropic_substeps_cgrid(
    state: OceanState,
    dt_s: float,
    n_substeps: int,
    grid: CubedSphereGrid,
    cdgrid: CubedSphereCDGrid,
    z_coord: OceanZStarCoordinate,
    config: OceanConfig,
) -> OceanState:
    """Run barotropic substeps on the cubed-sphere C-grid.

    Forward-backward (Matsuno) time stepping with C-grid staggering:
      1. Forward: update eta from C-grid flux divergence
      2. Backward: update u_c, v_c with compact PGF + Coriolis

    Coriolis is evaluated at cell centres and projected to edges via
    ``fv3_cc2c`` (vector halo exchange + non-orthogonality correction),
    avoiding cross-face staggered velocity interpolation.

    Cell-centre API: converts A-grid -> C-grid on entry, C-grid -> A-grid
    on exit.  No changes to OceanState or external interfaces.

    Parameters
    ----------
    state : OceanState
    dt_s : float
        Substep size [seconds].
    n_substeps : int
    grid : CubedSphereGrid
    cdgrid : CubedSphereCDGrid
    z_coord : OceanZStarCoordinate
    config : OceanConfig

    Returns
    -------
    OceanState with updated eta and velocity.
    """
    # --- Precision management ---
    _M = "barotropic_solver"
    g = cast(jnp.asarray(config.g), _M, "compute")
    H_bathy = cast(state.H_bathy.data, _M, "compute")
    mask = cast(state.land_mask.data, _M, "compute")
    u = cast(state.u.data, _M, "compute")
    v = cast(state.v.data, _M, "compute")
    eta_raw = cast(state.eta.data, _M, "compute")
    min_water_col = cast(jnp.asarray(config.min_water_column_m), _M, "compute")
    dt_s = cast(jnp.asarray(dt_s), _M, "compute")
    eta_floor = min_water_col - H_bathy
    eta = jnp.maximum(eta_raw, eta_floor) * mask
    _area = grid.area

    # --- Derive edge masks ---
    u_mask, v_mask = _derive_edge_masks(mask, grid)

    # --- Compute depth-averaged velocity at cell centres ---
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )
    # ``H_total``, ``U_bar`` and ``V_bar`` numerators all reduce
    # ``... * h_k`` over the level axis — fuse into one stacked sum.
    _bar_triple = jnp.sum(
        jnp.stack([h_k, u * h_k, v * h_k], axis=-1), axis=-2,
    )
    H_total = jnp.maximum(_bar_triple[..., 0], min_water_col)
    U_bar_cc = _bar_triple[..., 1] / H_total * mask
    V_bar_cc = _bar_triple[..., 2] / H_total * mask

    # --- Project to C-grid via vector halo exchange ---
    U_bar, V_bar = fv3_cc2c(U_bar_cc, V_bar_cc, cdgrid)
    U_bar = U_bar * u_mask
    V_bar = V_bar * v_mask

    # --- Coriolis parameter at cell centres ---
    f_cc = grid.f.astype(eta.dtype)  # (6, n, n)

    # --- Barotropic diffusion coefficients ---
    baro_alpha = jnp.asarray(
        config.barotropic_diffusion_alpha, dtype=eta.dtype,
    ) * (dt_s / jnp.asarray(config.barotropic_diffusion_dt_ref, dtype=eta.dtype))

    if config.barotropic_diffusion_alpha > 0.0:
        # Face-centred diffusion coefficients from adjacent cell areas
        area = grid.area.astype(eta.dtype)
        area_pad = pad_halo(area, interp_offsets=None)
        nu_dt_u = baro_alpha * 0.5 * (
            area_pad[:, :-1, 1:-1] + area_pad[:, 1:, 1:-1]
        )  # (6, n+1, n)
        nu_dt_v = baro_alpha * 0.5 * (
            area_pad[:, 1:-1, :-1] + area_pad[:, 1:-1, 1:]
        )  # (6, n, n+1)

    # --- Forward-backward substep body ---
    def substep_body(i, carry):
        eta_c, U_bar_c, V_bar_c = carry

        H_total_c = jnp.maximum(eta_c + H_bathy, min_water_col) * mask

        # -- Forward: update eta from C-grid continuity --
        H_pad = pad_halo(H_total_c, interp_offsets=grid.halo_interp_offsets)
        H_u = 0.5 * (H_pad[:, :-1, 1:-1] + H_pad[:, 1:, 1:-1])  # (6, n+1, n)
        H_v = 0.5 * (H_pad[:, 1:-1, :-1] + H_pad[:, 1:-1, 1:])  # (6, n, n+1)

        flux_u = H_u * U_bar_c * u_mask
        flux_v = H_v * V_bar_c * v_mask

        div_flux = cgrid_divergence(flux_u, flux_v, cdgrid).astype(eta.dtype)
        eta_unfloored = (eta_c - dt_s * div_flux) * mask
        eta_new = _clamp_redistribute(eta_unfloored, eta_floor, mask, _area)

        # -- Backward: update velocity with compact PGF + Coriolis --
        deta_dx, deta_dy = cgrid_gradient_2d(eta_new, cdgrid)
        deta_dx = deta_dx.astype(eta.dtype)
        deta_dy = deta_dy.astype(eta.dtype)

        # Coriolis at cell centres from C-grid velocities, then project
        # to edges via fv3_cc2c (vector halo + non-orthogonality).
        # f*v goes into the u-equation, -f*u into the v-equation.
        u_cc, v_cc = _cgrid_to_cell_centre(U_bar_c, V_bar_c)
        cor_u_cc = f_cc * v_cc * mask    # (6, n, n)
        cor_v_cc = -f_cc * u_cc * mask   # (6, n, n)
        cor_u, cor_v = fv3_cc2c(cor_u_cc, cor_v_cc, cdgrid)

        # Matsuno step 1: update u with old v (forward)
        U_bar_new = (U_bar_c + dt_s * (cor_u - g * deta_dx)) * u_mask

        # Matsuno step 2: update v with NEW u (backward)
        # Recompute full Coriolis vector with updated U but old V, then
        # project via fv3_cc2c.  Both components are needed for correct
        # vector rotation at face boundaries.
        u_cc_new, _ = _cgrid_to_cell_centre(U_bar_new, V_bar_c)
        cor_u_step2_cc = f_cc * v_cc * mask       # u-tendency unchanged (old V)
        cor_v_step2_cc = -f_cc * u_cc_new * mask   # v-tendency from new U
        _, cor_v_new = fv3_cc2c(cor_u_step2_cc, cor_v_step2_cc, cdgrid)
        V_bar_new = (V_bar_c + dt_s * (cor_v_new - g * deta_dy)) * v_mask

        # -- Diffusion on eta only (flux-form, conservative) --
        if config.barotropic_diffusion_alpha > 0.0:
            eta_new = _diffuse_eta_flux_form(
                eta_new, u_mask, v_mask, nu_dt_u, nu_dt_v, cdgrid,
            ) * mask
            eta_new = _clamp_redistribute(eta_new, eta_floor, mask, _area)

        return (eta_new, U_bar_new, V_bar_new)

    # --- Time integration ---
    if config.differentiable_barotropic:
        def scan_body(carry, _):
            new_carry = substep_body(0, carry)
            return new_carry, None

        (eta_f, U_bar_f, V_bar_f), _ = jax.lax.scan(
            scan_body, (eta, U_bar, V_bar), xs=None, length=n_substeps,
        )
    else:
        eta_f, U_bar_f, V_bar_f = jax.lax.fori_loop(
            0, n_substeps, substep_body, (eta, U_bar, V_bar),
        )

    # --- Convert C-grid velocity back to cell centres ---
    U_bar_cc_new, V_bar_cc_new = _cgrid_to_cell_centre(U_bar_f, V_bar_f)
    U_bar_cc_new = U_bar_cc_new * mask
    V_bar_cc_new = V_bar_cc_new * mask

    # --- Correct 3D velocities: preserve baroclinic structure ---
    u_baro_prime = u - U_bar_cc[..., jnp.newaxis]
    v_baro_prime = v - V_bar_cc[..., jnp.newaxis]
    u_new = (u_baro_prime + U_bar_cc_new[..., jnp.newaxis]) * mask[..., jnp.newaxis]
    v_new = (v_baro_prime + V_bar_cc_new[..., jnp.newaxis]) * mask[..., jnp.newaxis]

    # Downcast to storage precision
    eta_f = cast(eta_f, _M, "storage")
    u_new = cast(u_new, _M, "storage")
    v_new = cast(v_new, _M, "storage")

    return state._replace(
        eta=state.eta.replace(data=eta_f),
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )


# ==============================================================================
# FV3-faithful barotropic via the validated cube SW dynamical core
# ==============================================================================

def barotropic_substeps_fv3sw(
    state: OceanState,
    dt_s: float,
    n_substeps: int,
    grid: CubedSphereGrid,
    cdgrid: CubedSphereCDGrid,
    z_coord: OceanZStarCoordinate,
    config: OceanConfig,
    sw_model,
) -> OceanState:
    """Barotropic substeps via the validated FV3 cube shallow-water core.

    The barotropic free-surface mode is a 2-D shallow-water system.  Rather than
    re-deriving a (provably unstable) hand-rolled C/D-grid Coriolis, this routes
    the substeps through ``CDGridShallowWaterModel.step`` — the FV3-faithful
    C-D-grid SW solver (vector-invariant absolute-vorticity flux Coriolis with
    ``cdgrid.f_corner``, SSP-RK3, divergence damping + biharmonic
    hyperdiffusion; W2/W5-validated, edge-artifact-free).  An isolated test on a
    zonal eta initial state stays bit-stably zonal (non-zonal variance 0.0000)
    where the A-grid solver grows a 40% wavenumber-1 imprint and the explicit-f*v
    / bare-tendency C/D-grid solvers blow up (see ``fv3_faithful.md``).

    SW height mapping: ``h = H_bathy + eta``, ``h_s = -H_bathy`` so the SW
    pressure gradient ``g*grad(h + h_s) = g*grad(eta)`` is the free-surface PGF;
    the full column ``H_bathy + eta`` carries the gravity-wave transport.
    Depth-averaged velocity is lifted cc->corner D-grid via the rotation-aware
    ``center_to_dgrid_vector`` and projected back on exit.  Land is enforced by
    the cell mask on the returned eta/velocity (the SW core itself is global).
    """
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterState,
    )

    _f64 = jnp.float64
    H_bathy = jnp.asarray(state.H_bathy.data, _f64)
    mask = jnp.asarray(state.land_mask.data, _f64)
    u = jnp.asarray(state.u.data, _f64)
    v = jnp.asarray(state.v.data, _f64)
    eta_raw = jnp.asarray(state.eta.data, _f64)
    min_water_col = jnp.asarray(config.min_water_column_m, _f64)
    dt_s = jnp.asarray(dt_s, _f64)
    eta_floor = min_water_col - H_bathy
    eta = jnp.maximum(eta_raw, eta_floor) * mask

    # Depth-averaged barotropic velocity at cell centres.
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )
    _bar_triple = jnp.sum(
        jnp.stack([h_k, u * h_k, v * h_k], axis=-1), axis=-2,
    )
    H_total = jnp.maximum(_bar_triple[..., 0], min_water_col)
    U_bar_cc = _bar_triple[..., 1] / H_total * mask
    V_bar_cc = _bar_triple[..., 2] / H_total * mask

    # Lift cc -> corner D-grid (rotation-aware) for the SW state.
    u_d, v_d = center_to_dgrid_vector(U_bar_cc, V_bar_cc, cdgrid)
    u_d = jnp.asarray(u_d, _f64)
    v_d = jnp.asarray(v_d, _f64)

    h_sw = (H_bathy + eta).astype(_f64)
    h_s = (-H_bathy).astype(_f64)

    # Corner-ocean mask for coast impermeability: a D-grid corner wind is kept
    # only when ALL four surrounding cells are ocean, so the normal velocity at
    # any land-adjacent corner is zero and no mass/momentum crosses a coast
    # DURING the substeps (the global SW core is land-free; masking only after
    # the loop would let water leak through coasts mid-loop — codex review).
    mask_pad = pad_halo(mask, interp_offsets=None)  # (6, n+2, n+2)
    corner_ocean = (
        mask_pad[:, :-1, :-1] * mask_pad[:, 1:, :-1]
        * mask_pad[:, :-1, 1:] * mask_pad[:, 1:, 1:]
    ).astype(_f64)  # (6, n+1, n+1)
    u_d = u_d * corner_ocean
    v_d = v_d * corner_ocean
    sw_state = CDGridShallowWaterState(h=h_sw, u_d=u_d, v_d=v_d, h_s=h_s)

    def body(i, s):
        s = sw_model.step(s, dt_s)
        # Re-impose coast impermeability each substep (the SW core has no land).
        return s._replace(
            u_d=s.u_d * corner_ocean,
            v_d=s.v_d * corner_ocean,
        )

    sw_state = jax.lax.fori_loop(0, n_substeps, body, sw_state)

    eta_new = (sw_state.h - H_bathy) * mask
    eta_new = jnp.maximum(eta_new, eta_floor) * mask
    U_bar_cc_new, V_bar_cc_new = dgrid_to_center_vector(sw_state.u_d, sw_state.v_d)
    U_bar_cc_new = U_bar_cc_new * mask
    V_bar_cc_new = V_bar_cc_new * mask

    # Update 3D velocity: preserve the baroclinic (deviation) structure.
    u_baro_prime = u - U_bar_cc[..., jnp.newaxis]
    v_baro_prime = v - V_bar_cc[..., jnp.newaxis]
    u_new = (u_baro_prime + U_bar_cc_new[..., jnp.newaxis]) * mask[..., jnp.newaxis]
    v_new = (v_baro_prime + V_bar_cc_new[..., jnp.newaxis]) * mask[..., jnp.newaxis]

    _M = "barotropic_solver"
    eta_f = cast(eta_new, _M, "storage")
    u_new = cast(u_new, _M, "storage")
    v_new = cast(v_new, _M, "storage")
    return state._replace(
        eta=state.eta.replace(data=eta_f),
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )


# ==============================================================================
# FV3-faithful barotropic via the TRUE FV3 edge-staggered SW core (FV3Edge):
# upwind absolute-vorticity flux + _d2a2c_vect (sin_sg upwind, corner 2x2 solve).
# Gated by barotropic_staggering="fv3edge"; the default "fv3sw" (corner-staggered
# CDGridShallowWaterModel) is unchanged.  cc<->edge lift round-trips to 0.1% on a
# smooth field (validated).
# ==============================================================================

def _derive_fv3edge_masks(mask):
    """Edge masks for FV3Edge winds: u_d (6,n,n+1) x-edge ocean iff both j-flanking
    cells ocean; v_d (6,n+1,n) y-edge ocean iff both i-flanking cells ocean."""
    mp = pad_halo(mask, interp_offsets=None)  # (6, n+2, n+2)
    ux_mask = mp[:, 1:-1, :-1] * mp[:, 1:-1, 1:]   # (6, n, n+1)
    vy_mask = mp[:, :-1, 1:-1] * mp[:, 1:, 1:-1]   # (6, n+1, n)
    return ux_mask, vy_mask


def _cc_to_edge_vector(u_fl, v_fl, cdgrid):
    """Face-local cell-centre (u,v) -> FV3 edge-midpoint D-grid winds
    (u_d (6,n,n+1), v_d (6,n+1,n)).  cc->geographic (cc 4-edge angles) -> scalar
    halo-interp to edges (geographic is seam-continuous) -> rotate into the
    edge-local basis (cos/sin_angle_edge_x/_y).  Inverse is edge->cc averaging
    (round-trips to ~0.1% on smooth fields)."""
    from legoesm.grids.cubed_sphere_cdgrid import cell_centre_angles_from_4edge
    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    ue = ca * u_fl - sa * v_fl   # geographic east at cc
    vn = sa * u_fl + ca * v_fl   # geographic north at cc
    uep = pad_halo(ue, halo=1, interp_offsets=cdgrid.base.halo_interp_offsets)
    vnp = pad_halo(vn, halo=1, interp_offsets=cdgrid.base.halo_interp_offsets)
    ue_x = 0.5 * (uep[:, 1:-1, :-1] + uep[:, 1:-1, 1:])   # (6, n, n+1)
    vn_x = 0.5 * (vnp[:, 1:-1, :-1] + vnp[:, 1:-1, 1:])
    ue_y = 0.5 * (uep[:, :-1, 1:-1] + uep[:, 1:, 1:-1])   # (6, n+1, n)
    vn_y = 0.5 * (vnp[:, :-1, 1:-1] + vnp[:, 1:, 1:-1])
    u_d = cdgrid.cos_angle_edge_x * ue_x + cdgrid.sin_angle_edge_x * vn_x
    v_d = -cdgrid.sin_angle_edge_y * ue_y + cdgrid.cos_angle_edge_y * vn_y
    return u_d, v_d


def _edge_to_cc_vector(u_d, v_d):
    """FV3 edge-midpoint D-grid winds -> face-local cell-centre (u,v) by averaging
    the two flanking edges (inverse of :func:`_cc_to_edge_vector`)."""
    u_cc = 0.5 * (u_d[:, :, :-1] + u_d[:, :, 1:])
    v_cc = 0.5 * (v_d[:, :-1, :] + v_d[:, 1:, :])
    return u_cc, v_cc


def barotropic_substeps_fv3edge(
    state: OceanState,
    dt_s: float,
    n_substeps: int,
    grid: CubedSphereGrid,
    cdgrid: CubedSphereCDGrid,
    z_coord: OceanZStarCoordinate,
    config: OceanConfig,
    sw_edge_model,
) -> OceanState:
    """Barotropic substeps via the TRUE FV3 edge-staggered SW core
    (``FV3EdgeShallowWaterModel``: upwind absolute-vorticity flux + ``_d2a2c_vect``
    + RK3 + div-damp) — the algorithmically-faithful counterpart to
    :func:`barotropic_substeps_fv3sw` (corner-staggered, centered).  Gated by
    ``barotropic_staggering="fv3edge"``.

    cc(face-local) -> FV3 edge-midpoint D-grid via :func:`_cc_to_edge_vector`;
    ``h=H_bathy+eta``, ``h_s=-H_bathy`` (PGF=g*grad eta); coast edge masks re-applied
    after each substep (cross-coast leak bounded to one internal SSP-RK3 step — the
    SW core's RK stages still evaluate unmasked intermediates, same caveat as the
    ``fv3sw`` wrapper; codex 19de9080 review); edge->cc on exit preserves the baroclinic
    deviation.
    """
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterState,
    )
    _f64 = jnp.float64
    H_bathy = jnp.asarray(state.H_bathy.data, _f64)
    mask = jnp.asarray(state.land_mask.data, _f64)
    u = jnp.asarray(state.u.data, _f64)
    v = jnp.asarray(state.v.data, _f64)
    eta_raw = jnp.asarray(state.eta.data, _f64)
    min_water_col = jnp.asarray(config.min_water_column_m, _f64)
    dt_s = jnp.asarray(dt_s, _f64)
    eta_floor = min_water_col - H_bathy
    eta = jnp.maximum(eta_raw, eta_floor) * mask

    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )
    _bar_triple = jnp.sum(
        jnp.stack([h_k, u * h_k, v * h_k], axis=-1), axis=-2,
    )
    H_total = jnp.maximum(_bar_triple[..., 0], min_water_col)
    U_bar_cc = _bar_triple[..., 1] / H_total * mask
    V_bar_cc = _bar_triple[..., 2] / H_total * mask

    u_d, v_d = _cc_to_edge_vector(U_bar_cc, V_bar_cc, cdgrid)
    u_d = jnp.asarray(u_d, _f64)
    v_d = jnp.asarray(v_d, _f64)
    ux_mask, vy_mask = _derive_fv3edge_masks(mask)
    u_d = u_d * ux_mask
    v_d = v_d * vy_mask

    h_sw = (H_bathy + eta).astype(_f64)
    h_s = (-H_bathy).astype(_f64)
    sw_state = FV3EdgeShallowWaterState(h=h_sw, u_d=u_d, v_d=v_d, h_s=h_s)

    def body(i, s):
        s = sw_edge_model.step(s, dt_s)
        return s._replace(u_d=s.u_d * ux_mask, v_d=s.v_d * vy_mask)

    sw_state = jax.lax.fori_loop(0, n_substeps, body, sw_state)

    eta_new = (sw_state.h - H_bathy) * mask
    eta_new = jnp.maximum(eta_new, eta_floor) * mask
    U_bar_cc_new, V_bar_cc_new = _edge_to_cc_vector(sw_state.u_d, sw_state.v_d)
    U_bar_cc_new = U_bar_cc_new * mask
    V_bar_cc_new = V_bar_cc_new * mask

    u_baro_prime = u - U_bar_cc[..., jnp.newaxis]
    v_baro_prime = v - V_bar_cc[..., jnp.newaxis]
    u_new = (u_baro_prime + U_bar_cc_new[..., jnp.newaxis]) * mask[..., jnp.newaxis]
    v_new = (v_baro_prime + V_bar_cc_new[..., jnp.newaxis]) * mask[..., jnp.newaxis]

    _M = "barotropic_solver"
    eta_f = cast(eta_new, _M, "storage")
    u_new = cast(u_new, _M, "storage")
    v_new = cast(v_new, _M, "storage")
    return state._replace(
        eta=state.eta.replace(data=eta_f),
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )
