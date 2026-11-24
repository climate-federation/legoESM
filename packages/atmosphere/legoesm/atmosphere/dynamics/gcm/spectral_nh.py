"""Spectral Non-Hydrostatic Compressible Euler Model.

Solves the fully compressible Euler equations on the sphere using
pseudospectral (SH transform) horizontal operators and height-based
terrain-following (z*) coordinates with reference-state subtraction.

Prognostic variables (spectral):
    vor_hat         : Spectral relative vorticity, (n_sh, nlev)
    div_hat         : Spectral divergence, (n_sh, nlev)
    w_hat           : Spectral vertical velocity, (n_sh, nlev+1)
    theta_prime_hat : Spectral pot. temp. perturbation, (n_sh, nlev)
    rho_prime_hat   : Spectral JACOBIAN-WEIGHTED density perturbation
                      J*rho' [kg/m^3], (n_sh, nlev)
    tracers_hat     : Spectral JACOBIAN-WEIGHTED tracer density
                      J*rho*q [kg/m^3], (n_sh, nlev, n_tracers)

The two mass variables are carried J-weighted (J = terrain Jacobian
dz/dz*) so that their global integrals, ``sum (J rho_ref + J rho') dz dA``
and ``sum (J rho q) dz dA``, are the UNWEIGHTED integrals of spectral
fields: the spectral projection preserves those exactly and the n=0 mode
of a spectral flux divergence vanishes, so dry and tracer mass close to
round-off over terrain too (Rule 0).  Carrying rho' and rho*q unweighted
needed a grid-space 1/J before the projection, which does not preserve the
J-weighted integral (TC2a T21: +4.5e-11 of the dry mass per 6 s step).
Grid-space rho' and rho*q are recovered by dividing by J; for flat terrain
(J = 1) everything is bit-identical to the unweighted form.

Known representation limit (user decision: kept, documented): J is the
untruncated grid terrain, so J*rho_ref*q is not band-limited and a UNIFORM
mixing ratio q0 over terrain is represented only to the spectral projection
of J*rho_ref*q0 -- measured max|q/q0 - 1| = 3.8e-3 at t = 0 on TC2a T21
(transport then changes that q by 7.8e-7 in 5 steps).  Any terrain-tracer
diagnostic must therefore report errors against the t = 0 (projected) q
baseline, not against the analytic q0, or it will book this initialization
error as transport error (see test_tc2a_terrain_dry_and_tracer_mass_close_
without_fixer).

Time integration: split-explicit RK3.
    Slow mode  : spectral horizontal operators (SSP-RK3 outer)
    Fast mode  : forward-backward acoustic substeps in grid space
                 (purely vertical -- no horizontal derivatives)

References
----------
- Skamarock & Klemp (2008): Time-Split Nonhydrostatic Model.
- Bourke (1972): Spectral transform method.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.parallel.metal import place_spectral_grid
from legoesm.core.operators_3d import (
    vertical_advection_height,
)
from legoesm.grids.gaussian import (
    GaussianGrid,
    create_gaussian_grid,
    sh_analysis,
    sh_synthesis,
    sh_analysis_3d,
    sh_synthesis_3d,
    sh_analysis_oc2_3d,
    sh_analysis_dmu_3d,
    uv_from_vordiv_3d,
    spectral_hyperdiffusion_3d,
    sh_synthesis_H_3d,
    dealiasing_mask,
    spectral_gradient_3d,
)
from legoesm.grids.vertical import (
    HeightCoordinate,
    TerrainMetric,
    create_height_coordinate,
    compute_terrain_metric,
)
from legoesm.thermo import saturation_mixing_ratio
from legoesm.timestepping.tridiagonal import thomas_solve_batched
from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    compute_exner_perturbation,
    precompute_si_tridiag_bands,
    sponge_profile,
)
from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
from legoesm.timestepping.integration import refuse_unthreaded_stateful_physics
from legoesm.timestepping.split_explicit import (
    split_explicit_step,
    SplitExplicitConfig,
)
from legoesm import constants

_COS_LAT_MIN = 1.0e-6


# =============================================================================
# State and config
# =============================================================================

class SpectralNHState(NamedTuple):
    """State for the spectral non-hydrostatic compressible Euler equations.

    3D spectral full-level: (n_sh, nlev) complex128
    3D spectral half-level: (n_sh, nlev+1) complex128  (for w)
    2D spectral:            (n_sh,) complex128
    Tracers:                (n_sh, nlev, n_tracers) complex128,
                            J-weighted tracer density J*rho*q [kg/m^3]
    ``rho_prime_hat`` is J*rho' (J-weighted); see the module docstring.
    """
    vor_hat: Field
    div_hat: Field
    w_hat: Field
    theta_prime_hat: Field
    rho_prime_hat: Field
    phis_hat: Field
    tracers_hat: Field


class SpectralNHConfig(NamedTuple):
    """Configuration for spectral non-hydrostatic model."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    hyperdiff_order: int = 2
    sponge_width: float = 10000.0
    sponge_coeff: float = 0.05
    n_acoustic_substeps: int = 6
    small_earth_factor: float = 1.0
    semi_implicit_acoustic: bool = False  # Use tridiagonal solve for acoustics
    # Orszag 2/3-rule de-aliasing of the quadratic/cubic nonlinear slow
    # tendencies (vorticity/divergence advection, theta & rho flux form,
    # tracer transport).  These products spread aliased power across the
    # whole SH spectrum; without truncation the upper band folds back and
    # drives grid-scale instability.  This is the PRIMARY stabiliser for
    # the spectral NH core, which is why ``hyperdiff_coeff`` defaults to
    # 0.0 (a fixed nonzero hyperdiffusion would be resolution-dependent
    # and belongs in the experiment config; the matrix runner sets a
    # resolution-correct nu explicitly).  Default 2/3 (Orszag 1971);
    # 0.0 disables (exact rest-state tendency tests).  Only the upper
    # 1 - fraction of wavenumbers is removed, so resolved fields are
    # unchanged.  See ``grids.gaussian.dealiasing_mask``.
    dealiasing_fraction: float = 0.667
    # iter-9: opt-in anchored dry-mass fixer.  Mirrors the spectral PE
    # iter-3 mechanism (rescale the (n=0,m=0) coefficient of the
    # prognostic variable so the global integral returns to the
    # initial snapshot).  Disabled by default to preserve bit-for-bit
    # baseline for drift-measurement tests.
    fix_mass: bool = False
    anchor_mass_to_initial: bool = False
    rotating: bool = True
        # False => Coriolis identically zero on the small planet.  The
        # small-earth branch below scales ``grid.f`` by the reduction factor
        # (Rossby-preserving, the DCMIP-2025 convention used by TC2); DCMIP
        # TC3 (squall line) specifies NO Coriolis, which the cube and MPAS
        # arms already honour.  Without this the spectral TC3 arm ran at
        # f*60 while its cube/MPAS siblings ran at f=0 (codex r3 P1).



# =============================================================================
# Spectral gradient helper
# =============================================================================

# The geographic spectral gradient is the shared
# ``legoesm.grids.gaussian.spectral_gradient_3d`` (promoted iter 90 so the
# column-forcing geostrophic diagnostic reuses the SAME operator — no parallel copy).


# =============================================================================
# Slow tendency computation
# =============================================================================

def spectral_nh_slow_tendencies(
    state: SpectralNHState,
    grid: GaussianGrid,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    config: SpectralNHConfig,
    physics_tendency: SpectralNHState | None = None,
) -> SpectralNHState:
    """Compute slow tendencies for the spectral NH model.

    Horizontal operations use spectral transforms; vertical operations
    are done in grid space. Returns tendencies in the same pytree
    structure as state.  ``state.rho_prime_hat`` / ``state.tracers_hat``
    are the J-weighted J*rho' / J*rho*q (units ``kg/m^3``); so are the
    returned tendencies.
    """
    _require_tracer_density(state)
    a = grid.radius
    c_p = constants.c_pd

    dz = height_coord.dz
    dz_half = height_coord.dz_half
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    H = height_coord.z_half[0]
    J = terrain_metric.jacobian   # (n_lat, n_lon)

    # --- 1. Transform to grid ---
    # Batch the four (n_sh, nlev) syntheses into one — vor, div,
    # theta_p, rho_p all share shape and the inverse SH transform
    # treats the trailing axis as a passive batch (segment_sum runs
    # along n_sh, IRFFT runs along longitude).  ``w_hat`` has
    # (n_sh, nlev+1) — concatenate it onto the trailing axis of the
    # (vor, div, theta_p, rho_p) batch so that a *single* SH synthesis
    # serves all five fields.  Total trailing axis = ``4*nlev + (nlev+1)``.
    # 5 SH-syntheses → 1 (Loop 180 — same exploit as Loop 179 for the
    # acoustic update path).
    n_sh, nlev_t = state.vor_hat.data.shape
    _vdtr_stack = jnp.stack(
        [
            state.vor_hat.data,
            state.div_hat.data,
            state.theta_prime_hat.data,
            state.rho_prime_hat.data,
        ],
        axis=-1,
    )  # (n_sh, nlev, 4)
    _vdtr_flat = _vdtr_stack.reshape(n_sh, nlev_t * 4)
    _vdtrw_flat = jnp.concatenate(
        [_vdtr_flat, state.w_hat.data], axis=-1,
    )  # (n_sh, 4*nlev + (nlev+1))
    _vdtrw_grid_flat = sh_synthesis_3d(grid, _vdtrw_flat)
    _vdtr_grid_flat = _vdtrw_grid_flat[..., : nlev_t * 4]
    _vdtr_grid = _vdtr_grid_flat.reshape(grid.n_lat, grid.n_lon, nlev_t, 4)
    vor = _vdtr_grid[..., 0]
    div = _vdtr_grid[..., 1]
    theta_p = _vdtr_grid[..., 2]
    rho_p = _vdtr_grid[..., 3] / J[..., None]     # prognostic is J*rho'
    w = _vdtrw_grid_flat[..., nlev_t * 4:]              # (n_lat, n_lon, nlev+1)

    n_tracers = state.tracers_hat.data.shape[-1] if state.tracers_hat.data.ndim >= 3 else 0

    # --- 2. Velocities ---
    u_cos, v_cos = uv_from_vordiv_3d(
        grid, state.vor_hat.data, state.div_hat.data,
    )
    cos_lat_3d = jnp.clip(grid.cos_lat[:, None, None], _COS_LAT_MIN, None)
    u = u_cos / cos_lat_3d
    v = v_cos / cos_lat_3d

    # --- 3. Derived fields ---
    theta_total, rho_total = sanitize_theta_rho(
        theta_0 + theta_p,
        rho_0 + rho_p,
    )

    # --- 4. Exner perturbation ---
    pi_p = compute_exner_perturbation(rho_p, theta_p, height_coord)

    # --- 5. Spectral gradient of Exner perturbation ---
    pi_p_hat = sh_analysis_3d(grid, pi_p)
    dpi_dx, dpi_dy = spectral_gradient_3d(grid, pi_p_hat)

    # PGF vectors
    pgf_x = c_p * theta_total * dpi_dx
    pgf_y = c_p * theta_total * dpi_dy

    # --- 6. Kinetic energy (pole-safe via oc2 transform) ---
    # KE·cos²φ avoids the 1/cos² singularity at the poles; the factor
    # is absorbed by sh_analysis_oc2_3d below.
    KE_cos2 = 0.5 * (u_cos * u_cos + v_cos * v_cos)  # KE·cos²φ

    # --- 7. Absolute vorticity ---
    abs_vor = vor + grid.f[..., None]

    # --- 8. Spectral operators ---
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a

    # Vorticity fluxes
    A_vor = abs_vor * u_cos
    B_vor = abs_vor * v_cos

    # PGF cos-weighted for spectral div/curl
    pgf_u_cos = pgf_x * grid.cos_lat[:, None, None]
    pgf_v_cos = pgf_y * grid.cos_lat[:, None, None]

    # --- 9-10. Vorticity + divergence tendencies (batched SH analyses) ---
    # The vor/div tendencies need oc2 of {A_vor, B_vor, pgf_u_cos,
    # pgf_v_cos, KE_cos2} (5 calls) and dmu of {A_vor, B_vor, pgf_u_cos,
    # pgf_v_cos} (4 calls).  Stack each variant's inputs along the
    # trailing axis and fold into the level dim — 9 sequential SH
    # analyses collapse to 2.  Same trailing-axis-as-passive-batch
    # property as Loops 95 and 96 for spectral PE.
    n_lat_v, n_lon_v, nlev_v = A_vor.shape
    _oc2_stack = jnp.stack(
        [A_vor, B_vor, pgf_u_cos, pgf_v_cos, KE_cos2], axis=-1,
    )  # (..., nlev, 5)
    _dmu_stack = jnp.stack(
        [A_vor, B_vor, pgf_u_cos, pgf_v_cos], axis=-1,
    )  # (..., nlev, 4)
    _oc2_flat = sh_analysis_oc2_3d(
        grid, _oc2_stack.reshape(n_lat_v, n_lon_v, nlev_v * 5),
    ).reshape(-1, nlev_v, 5)
    _dmu_flat = sh_analysis_dmu_3d(
        grid, _dmu_stack.reshape(n_lat_v, n_lon_v, nlev_v * 4),
    ).reshape(-1, nlev_v, 4)
    A_oc2, B_oc2, pgf_u_oc2, pgf_v_oc2, K_hat = (
        _oc2_flat[..., 0], _oc2_flat[..., 1], _oc2_flat[..., 2],
        _oc2_flat[..., 3], _oc2_flat[..., 4],
    )
    A_dmu, B_dmu, pgf_u_dmu, pgf_v_dmu = (
        _dmu_flat[..., 0], _dmu_flat[..., 1], _dmu_flat[..., 2],
        _dmu_flat[..., 3],
    )

    # dvor/dt = -div(abs_vor * v) - curl(PGF)
    flux_vor_div = im_over_a[:, None] * A_oc2 - one_over_a * B_dmu
    pgf_curl = im_over_a[:, None] * pgf_v_oc2 + one_over_a * pgf_u_dmu
    dvor_hat = -flux_vor_div - pgf_curl

    # ddiv/dt = curl(abs_vor * v) - lap(K) - div(PGF)
    flux_vor_curl = im_over_a[:, None] * B_oc2 + one_over_a * A_dmu
    pgf_div = im_over_a[:, None] * pgf_u_oc2 - one_over_a * pgf_v_dmu
    ddiv_hat = flux_vor_curl - grid.lap[:, None] * K_hat - pgf_div

    # --- 11-13. Vertical advection + theta + rho horizontal fluxes (batched) ---
    # All four flux contributions (vert_adv_u/v on momentum, theta on
    # heat, rho on continuity) end up needing oc2(F_u_cos) and
    # dmu(F_v_cos) for the spectral div/curl operators.  Compute the
    # eight grid-space inputs first, then run a single batched oc2
    # call and a single batched dmu call instead of 4+4 = 8 sequential
    # SH analyses.
    vert_adv_u = vertical_advection_height(u, w, dz, dz_half, J)
    vert_adv_v = vertical_advection_height(v, w, dz, dz_half, J)
    vu_cos = vert_adv_u * grid.cos_lat[:, None, None]
    vv_cos = vert_adv_v * grid.cos_lat[:, None, None]
    theta_u_cos = theta_total * u_cos
    theta_v_cos = theta_total * v_cos
    rho_u_cos = rho_total * u_cos * J[..., None]
    rho_v_cos = rho_total * v_cos * J[..., None]

    n_lat_f, n_lon_f, nlev_f = vu_cos.shape
    _flux_oc2_stack = jnp.stack(
        [vu_cos, vv_cos, theta_u_cos, rho_u_cos], axis=-1,
    )  # (..., nlev, 4)
    _flux_dmu_stack = jnp.stack(
        [vu_cos, vv_cos, theta_v_cos, rho_v_cos], axis=-1,
    )
    _flux_oc2 = sh_analysis_oc2_3d(
        grid, _flux_oc2_stack.reshape(n_lat_f, n_lon_f, nlev_f * 4),
    ).reshape(-1, nlev_f, 4)
    _flux_dmu = sh_analysis_dmu_3d(
        grid, _flux_dmu_stack.reshape(n_lat_f, n_lon_f, nlev_f * 4),
    ).reshape(-1, nlev_f, 4)
    vu_oc2, vv_oc2, theta_u_oc2, rho_u_oc2 = (
        _flux_oc2[..., 0], _flux_oc2[..., 1], _flux_oc2[..., 2], _flux_oc2[..., 3],
    )
    vu_dmu, vv_dmu, theta_v_dmu, rho_v_dmu = (
        _flux_dmu[..., 0], _flux_dmu[..., 1], _flux_dmu[..., 2], _flux_dmu[..., 3],
    )

    # Vorticity / divergence: vertical-advection contributions
    vert_vor = im_over_a[:, None] * vv_oc2 + one_over_a * vu_dmu
    vert_div = im_over_a[:, None] * vu_oc2 - one_over_a * vv_dmu
    dvor_hat = dvor_hat + vert_vor
    ddiv_hat = ddiv_hat + vert_div

    # Theta equation: horizontal advection via spectral flux form.
    # NOTE: Vertical advection of theta by w is handled ONLY by the
    # acoustic substeps (forward-backward scheme) to avoid double counting
    # in the split-explicit time integration (Skamarock & Klemp 2008).
    flux_theta_div = im_over_a[:, None] * theta_u_oc2 - one_over_a * theta_v_dmu
    # Continuity equation (rho'): horizontal only — vertical mass flux
    # divergence handled by acoustic step.
    flux_rho_div = im_over_a[:, None] * rho_u_oc2 - one_over_a * rho_v_dmu

    # Sum theta_total*div in grid first, transform once: ``theta_div_hat``
    # and ``rho_horiz_tend_grid`` both feed into a single sh_analysis_3d
    # but rho_horiz needs a sh_synthesis_3d on flux_rho_div first; batch
    # the two analyses into one stacked call.
    #
    # Loop 187 — when ``n_tracers > 0`` the tracer block (section 15)
    # also runs an ``sh_analysis_3d(dq_grid_sum_flat)`` whose trailing
    # axis is ``nlev*n_tracers``.  Defer the (theta_div, rho_horiz)
    # analysis past the tracer block and ``jnp.concatenate`` all three
    # grid inputs along the trailing axis, then run a single
    # ``sh_analysis_3d`` — different trailing-axis sizes
    # (``nlev*2 + nlev*n_tracers``) all flow through the same
    # ``segment_sum + projection`` machinery.  2 SH analyses → 1 when
    # tracers are active.  Same exploit as Loops 184/185.
    # Tendency of the J-weighted J*rho': -div_h(rho v J), NO 1/J before
    # the projection (that 1/J broke the J-weighted mass integral).
    rho_horiz_tend_grid = -sh_synthesis_3d(grid, flux_rho_div)
    _theta_div_grid = theta_total * div
    n_lat_p2, n_lon_p2, nlev_p2 = _theta_div_grid.shape

    # --- 14. w tendency (slow part) ---
    # Slow w tendency is zero: vertical PGF, buoyancy, and w-divergence are
    # all handled by the acoustic substeps (forward-backward scheme).
    dw_hat = jnp.zeros_like(state.w_hat.data)

    # --- 15. Tracer transport (horizontal, flux form) ---
    # ``tracers_hat`` carries the J-weighted tracer density J*rho*q
    # [kg/m^3] (Rule 0): its horizontal tendency is -div_h(rho q v J),
    # built with the SAME spectral flux-divergence, synthesis and analysis
    # path as the J*rho' continuity tendency above, so the global integral of the
    # tendency is zero (the n=0 mode of a spectral divergence vanishes)
    # and a uniform mixing ratio q gives exactly q times the rho'
    # tendency.  The VERTICAL tracer flux is applied in the acoustic
    # update with the substep-integrated mass flux that moves rho'
    # (``_tracer_vertical_transport``), as in WRF/MPAS.  The former
    # advective form (q prognostic, -v.grad q, upwind -w dq/dz) is not
    # conservative: it lost 3.4e-4 of the TC3 total water in 4 min.
    if n_tracers > 0:
        # Fold the tracer dimension into the trailing level axis so all
        # SH transforms in the tracer block run ONCE on a thicker
        # ``(..., nlev*n_tracers)`` tensor (passive-trailing-axis batch).
        tracers_hat_data = state.tracers_hat.data  # (n_sh, nlev, n_tracers)
        n_sh_t = tracers_hat_data.shape[0]
        nlev_tr = tracers_hat_data.shape[-2]
        tracers_hat_flat = tracers_hat_data.reshape(
            n_sh_t, nlev_tr * n_tracers,
        )
        rq_grid_flat = (sh_synthesis_3d(grid, tracers_hat_flat)
                        / J[..., None])                  # rho*q

        if n_tracers == 1:
            u_cos_b = u_cos
            v_cos_b = v_cos
        else:
            u_cos_b = jnp.repeat(u_cos, n_tracers, axis=-1)
            v_cos_b = jnp.repeat(v_cos, n_tracers, axis=-1)

        rq_u_cos_flat = rq_grid_flat * u_cos_b * J[..., None]
        rq_v_cos_flat = rq_grid_flat * v_cos_b * J[..., None]
        flux_q_div_flat = (
            im_over_a[:, None] * sh_analysis_oc2_3d(grid, rq_u_cos_flat)
            - one_over_a * sh_analysis_dmu_3d(grid, rq_v_cos_flat)
        )
        dq_grid_sum_flat = -sh_synthesis_3d(grid, flux_q_div_flat)
        # One analysis serves (theta_div, rho_horiz, tracers): concatenate
        # along the trailing axis (Loop 187).
        _all_an_input = jnp.concatenate(
            [_theta_div_grid, rho_horiz_tend_grid, dq_grid_sum_flat],
            axis=-1,
        )  # (n_lat, n_lon, nlev*(2 + n_tracers))
        _all_an_hat = sh_analysis_3d(grid, _all_an_input)
        theta_div_hat = _all_an_hat[:, :nlev_p2]
        drho_p_hat = _all_an_hat[:, nlev_p2:nlev_p2 * 2]
        dq_hat_flat = _all_an_hat[:, nlev_p2 * 2:]

        # Restore the (n_sh, nlev, n_tracers) layout that matches
        # ``state.tracers_hat.data``.
        dtracers_hat = dq_hat_flat.reshape(n_sh_t, nlev_tr, n_tracers)
    else:
        # No tracers — keep the (theta_div, rho_horiz) 2-batch analysis.
        _theta_rho_pair = jnp.stack(
            [_theta_div_grid, rho_horiz_tend_grid], axis=-1,
        )
        _theta_rho_hat = sh_analysis_3d(
            grid, _theta_rho_pair.reshape(n_lat_p2, n_lon_p2, nlev_p2 * 2),
        ).reshape(-1, nlev_p2, 2)
        theta_div_hat = _theta_rho_hat[..., 0]
        drho_p_hat = _theta_rho_hat[..., 1]
        dtracers_hat = jnp.zeros_like(state.tracers_hat.data)

    dtheta_p_hat = -flux_theta_div + theta_div_hat

    # --- 16. Sponge layer damping ---
    if config.sponge_coeff > 0:
        sponge = sponge_profile(
            height_coord.z_full, H, config.sponge_width, config.sponge_coeff,
        )  # (nlev,)
        # Damp vorticity, divergence, theta toward reference
        dvor_hat = dvor_hat - sponge * state.vor_hat.data
        ddiv_hat = ddiv_hat - sponge * state.div_hat.data
        dtheta_p_hat = dtheta_p_hat - sponge * state.theta_prime_hat.data
        # Damp w at half-levels (matching cubed-sphere and MPAS dycores)
        sponge_half = sponge_profile(
            height_coord.z_half, H, config.sponge_width, config.sponge_coeff,
        )  # (nlev+1,)
        dw_hat = dw_hat - sponge_half * state.w_hat.data

    # --- 17. Spectral hyperdiffusion ---
    if config.hyperdiff_coeff > 0:
        # Batch the three pointwise spectral hyperdiffusions (vor, div,
        # theta_prime) into one call by stacking spectral coefficients
        # along a trailing axis.  ``spectral_hyperdiffusion_3d`` is
        # purely element-wise (``damping * coeffs``), so the trailing
        # axis is a passive batch — same exploit as Loop 120 in
        # spectral PE.  3 kernel launches → 1.
        n_sh_h, nlev_h = state.vor_hat.data.shape
        _vdt_hat = jnp.stack(
            [
                state.vor_hat.data,
                state.div_hat.data,
                state.theta_prime_hat.data,
            ],
            axis=-1,
        )  # (n_sh, nlev, 3)
        _hd_stack = spectral_hyperdiffusion_3d(
            grid, _vdt_hat.reshape(n_sh_h, nlev_h * 3),
            config.hyperdiff_coeff, config.hyperdiff_order,
        ).reshape(n_sh_h, nlev_h, 3)
        dvor_hat = dvor_hat + _hd_stack[..., 0]
        ddiv_hat = ddiv_hat + _hd_stack[..., 1]
        dtheta_p_hat = dtheta_p_hat + _hd_stack[..., 2]

    # --- 18. Physics tendencies ---
    if physics_tendency is not None:
        dvor_hat = dvor_hat + physics_tendency.vor_hat.data
        ddiv_hat = ddiv_hat + physics_tendency.div_hat.data
        dtheta_p_hat = dtheta_p_hat + physics_tendency.theta_prime_hat.data
        drho_p_hat = drho_p_hat + physics_tendency.rho_prime_hat.data
        dtracers_hat = dtracers_hat + physics_tendency.tracers_hat.data

    # --- 19. Orszag 2/3-rule de-aliasing of the nonlinear slow tendencies ---
    # The horizontal slow tendencies above are quadratic/cubic products
    # ((zeta+f)*v, |v|^2, theta*v, rho*v*J, q*v) transformed back to SH
    # space, which folds aliased power across the whole spectrum.  Zero
    # every coefficient with total wavenumber n above
    # floor(dealiasing_fraction * n_max) so the spurious upper band cannot
    # destabilise the run.  The mask is (n_sh,); broadcast over the
    # trailing level / tracer axes.  ``dw_hat`` (slow part = sponge only;
    # acoustic dynamics handled in grid space) is masked for uniformity.
    # n=0 modes (n << n_cut) are always retained, so global integrals are
    # untouched.  Applied LAST so masked modes are held identically at
    # zero — matching spectral_pe.
    _dealias = dealiasing_mask(grid, config.dealiasing_fraction)
    _dealias_2d = _dealias[:, None]            # (n_sh, 1) for (n_sh, nlev[+1])
    dvor_hat = dvor_hat * _dealias_2d
    ddiv_hat = ddiv_hat * _dealias_2d
    dw_hat = dw_hat * _dealias_2d
    dtheta_p_hat = dtheta_p_hat * _dealias_2d
    drho_p_hat = drho_p_hat * _dealias_2d
    # Tracers carry an extra trailing axis (n_sh, nlev, n_tracers); match
    # the mask rank to whatever the tracer tendency rank is (3D in every
    # constructor since tracers_hat = zeros((n_sh, nlev, max(n_tr, 1))),
    # but stay robust to a 2D tracer tendency).
    _dealias_tr = _dealias.reshape(
        (-1,) + (1,) * (dtracers_hat.ndim - 1)
    )
    dtracers_hat = dtracers_hat * _dealias_tr

    return SpectralNHState(
        vor_hat=state.vor_hat.replace(data=dvor_hat),
        div_hat=state.div_hat.replace(data=ddiv_hat),
        w_hat=state.w_hat.replace(data=dw_hat),
        theta_prime_hat=state.theta_prime_hat.replace(data=dtheta_p_hat),
        rho_prime_hat=state.rho_prime_hat.replace(data=drho_p_hat),
        phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
        tracers_hat=state.tracers_hat.replace(data=dtracers_hat),
    )


# =============================================================================
# Acoustic substeps (grid space, purely vertical)
# =============================================================================

def _acoustic_substeps_grid(
    w_grid, theta_p_grid, rho_p_grid,
    dt_s, n_substeps,
    height_coord, terrain_metric, config,
):
    """Run acoustic substeps in grid space.

    Same vertical-only operations as compressible_euler.acoustic_substeps,
    but working on Gaussian grid arrays (n_lat, n_lon, ...) shapes.

    Returns ``(w, theta', rho', mass_flux)``; ``mass_flux`` (half levels,
    zero at top and bottom) is the substep-integrated vertical mass flux
    ``sum_s dt_s * rho_half * w`` [kg/m^2] that moved rho', reused by
    :func:`_tracer_vertical_transport` so tracer mass rides the same flux.
    """
    g = config.g
    c_p = constants.c_pd
    dz = height_coord.dz
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    J = terrain_metric.jacobian  # (n_lat, n_lon)

    def substep_body(i, carry):
        w_c, theta_p_c, rho_p_c, mflux_c = carry

        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p_c,
            rho_0 + rho_p_c,
        )

        # --- Forward: update w ---
        pi_p = compute_exner_perturbation(rho_p_c, theta_p_c, height_coord)
        dpi_dz_inner = (pi_p[..., :-1] - pi_p[..., 1:]) / (
            0.5 * (dz[:-1] + dz[1:])
        )
        theta_half_inner = 0.5 * (theta_total[..., :-1] + theta_total[..., 1:])
        theta_p_half = 0.5 * (theta_p_c[..., :-1] + theta_p_c[..., 1:])
        theta_0_half = 0.5 * (theta_0[:-1] + theta_0[1:])
        buoyancy = g * theta_p_half / theta_0_half

        dw_dt_inner = (
            -c_p * theta_half_inner * dpi_dz_inner / J[..., None]
            + buoyancy
        )
        w_new = w_c.at[..., 1:-1].set(
            w_c[..., 1:-1] + dt_s * dw_dt_inner
        )

        # --- Backward: update rho' ---
        rho_half = 0.5 * (rho_total[..., :-1] + rho_total[..., 1:])
        # Use ``jnp.pad`` to attach the zero top/bottom boundaries
        # instead of allocating ``zeros_like(w_new)`` and scattering
        # the interior; one Pad HLO op vs alloc + scatter inside
        # the per-substep ``fori_loop`` body.
        pad_axes = ((0, 0),) * (w_new.ndim - 1)
        rho_w = jnp.pad(rho_half * w_new[..., 1:-1], (*pad_axes, (1, 1)))
        vert_div = (rho_w[..., :-1] - rho_w[..., 1:]) / dz
        vert_div = vert_div / J[..., None]
        rho_p_new = rho_p_c - dt_s * vert_div
        mflux_new = mflux_c + dt_s * rho_w

        # --- Backward: update theta' ---
        w_full = 0.5 * (w_new[..., :-1] + w_new[..., 1:])
        nlev_local = theta_total.shape[-1]
        if nlev_local > 2:
            dz_half_val = height_coord.dz_half
            dz_centered = dz_half_val[:-1] + dz_half_val[1:]
            inner_grad = (
                theta_total[..., :-2] - theta_total[..., 2:]
            ) / dz_centered
            theta_pad_axes = ((0, 0),) * (theta_total.ndim - 1)
            dtheta_dz = jnp.pad(inner_grad, (*theta_pad_axes, (1, 1)))
        else:
            dtheta_dz = jnp.zeros_like(theta_total)
        theta_p_new = theta_p_c - dt_s * w_full / J[..., None] * dtheta_dz

        return (w_new, theta_p_new, rho_p_new, mflux_new)

    return jax.lax.fori_loop(
        0, n_substeps, substep_body,
        (w_grid, theta_p_grid, rho_p_grid, jnp.zeros_like(w_grid)),
    )


def _acoustic_substeps_grid_semi_implicit(
    w_grid, theta_p_grid, rho_p_grid,
    dt_s, n_substeps,
    height_coord, terrain_metric, config,
):
    """Semi-implicit acoustic substeps in grid space (tridiagonal w solve).

    Same structure as the explicit version but the vertical pressure
    gradient in the w equation is treated implicitly via a tridiagonal
    solve, removing the vertical acoustic CFL constraint.  The tridiagonal
    bands come from the shared :func:`precompute_si_tridiag_bands`
    (stretched-grid metric, both couplings on the boundary diagonal).  The
    substep body is not delegated to ``semi_implicit_acoustic_column_kernel``
    because this model's θ' update zeroes ``∂θ/∂z`` on the top and bottom
    levels where the shared kernel uses one-sided differences.  Returns the
    same 4-tuple as :func:`_acoustic_substeps_grid`.
    """
    g = config.g
    c_p = constants.c_pd
    dz = height_coord.dz
    dz_half = height_coord.dz_half
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    J = terrain_metric.jacobian

    dz_inner = 0.5 * (dz[:-1] + dz[1:])
    nlev = theta_p_grid.shape[-1]
    a_tri, b_tri, c_tri = precompute_si_tridiag_bands(
        height_coord, J, dt_s, g, implicit_buoyancy=False, nlev=nlev,
    )

    def substep_body(i, carry):
        w_c, theta_p_c, rho_p_c, mflux_c = carry

        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p_c,
            rho_0 + rho_p_c,
        )

        # --- Explicit RHS for w ---
        pi_p = compute_exner_perturbation(rho_p_c, theta_p_c, height_coord)
        dpi_dz_inner = (pi_p[..., :-1] - pi_p[..., 1:]) / dz_inner
        theta_half_inner = 0.5 * (theta_total[..., :-1] + theta_total[..., 1:])
        theta_p_half = 0.5 * (theta_p_c[..., :-1] + theta_p_c[..., 1:])
        theta_0_half = 0.5 * (theta_0[:-1] + theta_0[1:])
        buoyancy = g * theta_p_half / theta_0_half

        dw_dt_inner = (
            -c_p * theta_half_inner * dpi_dz_inner / J[..., None]
            + buoyancy
        )

        rhs = w_c[..., 1:-1] + dt_s * dw_dt_inner

        w_inner_new = thomas_solve_batched(a_tri, b_tri, c_tri, rhs)
        w_new = w_c.at[..., 1:-1].set(w_inner_new)

        # --- Backward: update rho' ---
        rho_half = 0.5 * (rho_total[..., :-1] + rho_total[..., 1:])
        pad_axes = ((0, 0),) * (w_new.ndim - 1)
        rho_w = jnp.pad(rho_half * w_new[..., 1:-1], (*pad_axes, (1, 1)))
        vert_div = (rho_w[..., :-1] - rho_w[..., 1:]) / dz
        vert_div = vert_div / J[..., None]
        rho_p_new = rho_p_c - dt_s * vert_div
        mflux_new = mflux_c + dt_s * rho_w

        # --- Backward: update theta' ---
        w_full = 0.5 * (w_new[..., :-1] + w_new[..., 1:])
        if nlev > 2:
            dz_centered = dz_half[:-1] + dz_half[1:]
            inner_grad = (theta_total[..., :-2] - theta_total[..., 2:]) / dz_centered
            theta_pad_axes = ((0, 0),) * (theta_total.ndim - 1)
            dtheta_dz = jnp.pad(inner_grad, (*theta_pad_axes, (1, 1)))
        else:
            dtheta_dz = jnp.zeros_like(theta_total)
        theta_p_new = theta_p_c - dt_s * w_full / J[..., None] * dtheta_dz

        return (w_new, theta_p_new, rho_p_new, mflux_new)

    return jax.lax.fori_loop(
        0, n_substeps, substep_body,
        (w_grid, theta_p_grid, rho_p_grid, jnp.zeros_like(w_grid)),
    )


# Encoding markers (Field.name) of the J-weighted mass prognostics.  States
# from the former encodings -- mixing ratio q (units kg/kg), or unweighted
# rho' / rho*q (Field names "rho_prime_hat" / "tracers_hat") -- carry the
# same shapes and units, so the name is what tells them apart.
_JRHO_NAME = "J_rho_prime_hat"
_JRQ_NAME = "J_rho_q_hat"


def _require_tracer_density(state) -> None:
    """Refuse a state not in the current mass encoding: ``rho_prime_hat`` =
    J*rho' and ``tracers_hat`` = J*rho*q (units ``kg/m^3``), marked by their
    Field names.  A state from an older encoding is refused rather than
    silently mis-read (over terrain an unweighted rho' would be divided by J
    again); migrate it by synthesizing rho' and rho*q to the grid,
    multiplying by J and analysing back."""
    if state.rho_prime_hat.units != "kg/m^3":
        raise ValueError(
            "spectral NH rho_prime_hat must be J*rho' in 'kg/m^3', got "
            f"units {state.rho_prime_hat.units!r}")
    if state.tracers_hat.units != "kg/m^3":
        raise ValueError(
            "spectral NH tracers_hat must be the J-weighted tracer density "
            f"J*rho*q (units 'kg/m^3'), got units {state.tracers_hat.units!r}")
    if (state.rho_prime_hat.name != _JRHO_NAME
            or state.tracers_hat.name != _JRQ_NAME):
        raise ValueError(
            "spectral NH state is not in the J-weighted mass encoding "
            f"(Field names {_JRHO_NAME!r}, {_JRQ_NAME!r}); got "
            f"{state.rho_prime_hat.name!r}, {state.tracers_hat.name!r}.  "
            "Migrate: grid rho' and rho*q times J, analysed back.")


def _tracer_vertical_transport(rq, rho_total, mass_flux, dz, jacobian):
    """Vertical flux-form transport of tracer density ``rq = rho*q``.

    ``mass_flux`` is the substep-integrated interface mass flux returned by
    the acoustic kernels (the exact flux that changed rho' this stage).
    The tracer flux is ``mass_flux * q_up`` with ``q = rq / rho`` taken from
    the upwind (donor) cell -- the flux-form counterpart of the first-order
    upwind scheme the dycore used before.  Boundary fluxes are zero, so the
    column integral of ``J * rq * dz`` is unchanged to round-off, and a
    uniform q stays uniform (the tracer change is q times the rho' change).
    Positivity holds when each cell's TOTAL stage-integrated outflow, the
    sum of the outward mass fluxes through BOTH its faces, does not exceed
    its starting air mass ``J * dz * rho_stage`` (per unit area) -- for a
    one-sided outflow, velocity CFL times rho_face / rho_cell <= 1, within
    a few percent of the former upwind form's bound for a hydrostatically
    stratified column (adjacent-layer density ratios ~1.0-1.1), but not
    identical to it; a diverging cell (outflow through both faces) needs
    the sum of the two face Courant numbers <= 1.  The donor q is frozen at
    its stage-start value while the substep fluxes accumulate (WRF/MPAS
    practice: scalars ride the time-averaged acoustic mass flux), so the
    bound is on the stage-integrated OUTflow vs the cell's starting air
    mass, not on the per-substep flux.

    ``rq``: (..., nlev, n_tr); ``rho_total``: (..., nlev);
    ``mass_flux``: (..., nlev+1); returns the updated ``rq``.
    """
    q = rq / rho_total[..., None]
    f = mass_flux[..., 1:-1, None]                     # interior interfaces
    # Interface i (1..nlev-1) sits between level i-1 (above) and i (below).
    q_up = jnp.where(f > 0, q[..., 1:, :], q[..., :-1, :])
    pad = ((0, 0),) * (rq.ndim - 2)
    fq = jnp.pad(f * q_up, (*pad, (1, 1), (0, 0)))
    return rq - (fq[..., :-1, :] - fq[..., 1:, :]) / (
        dz[:, None] * jacobian[..., None, None])


#: Tracer-tendency encoding a spectral-NH ``physics_fn`` must declare: its
#: ``tracers_hat`` tendency is d(J rho q)/dt (and its ``rho_prime_hat``
#: tendency d(J rho')/dt).  A mixing-ratio tendency has the same shape, so
#: the encoding cannot be inferred -- it is an explicit attribute, checked
#: by :meth:`SpectralCompressibleEulerModel.step`.
TRACER_TENDENCY_ENCODING = "J_rho_q"
#: The legacy encoding :func:`mixing_ratio_physics_adapter` converts from
#: (q and unweighted rho' in, dq/dt and drho'/dt out).  An untagged legacy
#: function is accepted by the (explicit, opt-in) adapter too.
MIXING_RATIO_ENCODING = "mixing_ratio"


def tag_tracer_tendency_encoding(physics_fn):
    """Declare that ``physics_fn`` returns J-weighted tendencies
    (:data:`TRACER_TENDENCY_ENCODING`) and return the DECLARED callable.

    A plain Python function is tagged in place and returned.  Anything else
    (partial, bound method, callable instance, proxy) is wrapped in a plain
    forwarding function (``functools.wraps``, so the stateful-physics marker
    stays reachable through ``__wrapped__``) that carries the declaration --
    ``step()`` reads declarations only from plain functions, the one kind
    of callable whose own attribute storage cannot be redirected.  Use the
    RETURN value.  Physics written for mixing ratios must go through
    :func:`mixing_ratio_physics_adapter` instead."""
    import functools
    import types

    if type(physics_fn) is types.FunctionType:
        physics_fn.tracer_tendency_encoding = TRACER_TENDENCY_ENCODING
        return physics_fn

    @functools.wraps(physics_fn)
    def tagged(*args, **kwargs):
        return physics_fn(*args, **kwargs)
    tagged.tracer_tendency_encoding = TRACER_TENDENCY_ENCODING
    return tagged


def _physics_encoding_tag(physics_fn):
    """The encoding DECLARED on ``physics_fn`` itself, or None.

    Only a plain Python function (``types.FunctionType``, exact type) can
    carry a declaration, read from its own attribute storage through the
    built-in function ``__dict__`` descriptor -- no ``__getattr__`` /
    ``__getattribute__`` / ``__dict__``-property delegation and no inference
    through ``functools.partial`` / ``__wrapped__`` links: what a wrapper
    runs cannot be verified by introspection, so anything else must be
    (re-)declared with :func:`tag_tracer_tendency_encoding`, which wraps it
    in a plain function.  ``functools.wraps`` copies the attribute, i.e. a
    pass-through wrapper inherits its target's declaration by construction."""
    import types

    if type(physics_fn) is not types.FunctionType:
        return None
    local = types.FunctionType.__dict__["__dict__"].__get__(physics_fn)
    return local.get("tracer_tendency_encoding")


def _chain_encoding_tags(physics_fn):
    """Every encoding tag reachable from ``physics_fn`` through
    ``__wrapped__`` and ``functools.partial.func`` links (cycle-guarded) --
    used CONSERVATIVELY by :func:`mixing_ratio_physics_adapter` to refuse
    anything that may already return J-weighted tendencies.  Also follows
    bound methods' ``__func__`` and callable instances' type-resolved
    ``__call__``."""
    import functools
    import types

    queue, seen, tags, keep = [physics_fn], set(), set(), []
    while queue and len(seen) < 256:        # bounded: bound-method / method-
        fn = queue.pop(0)                    # wrapper __call__ chains mint new
        if fn is None or id(fn) in seen:     # objects at every lookup
            continue
        seen.add(id(fn))
        keep.append(fn)                      # keep ids alive while searching
        for tag in (getattr(fn, "tracer_tendency_encoding", None),
                    _physics_encoding_tag(fn)):
            if tag is not None:
                tags.add(tag)
        queue.append(getattr(fn, "__wrapped__", None))
        if issubclass(type(fn), functools.partial):
            # The STORED target (what partial.__call__ runs), immune to a
            # subclass shadowing ``.func``; plus the visible attribute.
            queue.append(functools.partial.func.__get__(fn))
            queue.append(getattr(fn, "func", None))
        # What actually runs for bound methods and callable instances.
        queue.append(getattr(fn, "__func__", None))
        if type(fn) is not types.FunctionType and not isinstance(
                fn, (types.MethodWrapperType, types.WrapperDescriptorType,
                     types.BuiltinFunctionType)):
            queue.append(getattr(type(fn), "__call__", None))
            try:                           # instance-bound call target
                queue.append(getattr(fn, "__call__", None))
            except Exception:              # noqa: BLE001 -- best effort
                pass
    return tags


def _require_physics_encoding(physics_fn) -> None:
    """Reject a ``physics_fn`` without, or with an incompatible, explicit
    tracer-tendency encoding tag."""
    if physics_fn is None:
        return
    tag = _physics_encoding_tag(physics_fn)
    if tag is None:
        raise ValueError(
            "spectral NH physics_fn has no tracer_tendency_encoding tag: "
            "tendencies must be d(J rho q)/dt, d(J rho')/dt -- declare it "
            "with tag_tracer_tendency_encoding(fn), or wrap physics written "
            "for mixing ratios (dq/dt) with mixing_ratio_physics_adapter(fn)")
    if tag != TRACER_TENDENCY_ENCODING:
        raise ValueError(
            f"spectral NH physics_fn tracer_tendency_encoding={tag!r} is "
            f"incompatible with {TRACER_TENDENCY_ENCODING!r}; wrap mixing-"
            "ratio physics with mixing_ratio_physics_adapter(fn)")


def mixing_ratio_physics_adapter(legacy_physics_fn):
    """Explicit opt-in wrapper for a ``physics_fn`` written for the former
    spectral-NH convention: it receives a state whose ``rho_prime_hat`` is
    the UNWEIGHTED rho' and whose ``tracers_hat`` are mixing ratios q
    (units ``kg/kg``), and returns ``drho'/dt`` and ``dq/dt``.

    The wrapper builds that view from the J-weighted prognostics
    (rho' = (J rho')/J, q = (J rho q)/(J rho) on the Gaussian grid, analysed
    back to spectral; the view's q is therefore the spectral projection of
    the grid q), calls the legacy function, and converts its tendencies to
    the prognostic ones using the CURRENT rho and q:

        d(J rho')/dt  = J * drho'/dt
        d(J rho q)/dt = J * (rho * dq/dt + q * drho'/dt)

    Vorticity, divergence and theta' tendencies pass through unchanged; a
    ``(tendencies, diagnostics)`` tuple return is preserved.  NOTE: with
    ``fix_mass`` + ``anchor_mass_to_initial`` the anchored fixer removes any
    net dry-mass source (it anchors dry mass by design), while the
    corresponding ``J q drho'/dt`` tracer source stays: physics that
    changes the DRY mass must run with the anchor off.  The
    ``_requires_phys_state`` tag of a stateful legacy function is still
    found through ``__wrapped__`` (``functools.wraps``).  The wrapper is
    tagged :data:`TRACER_TENDENCY_ENCODING`; a function that already
    declares that encoding is refused (it would be converted twice).
    """
    import functools

    _tags = _chain_encoding_tags(legacy_physics_fn)
    _own = _physics_encoding_tag(legacy_physics_fn)
    if _own is not None:
        _tags.add(_own)
    if TRACER_TENDENCY_ENCODING in _tags:
        raise ValueError(
            "mixing_ratio_physics_adapter: the function (or something it "
            "wraps) declares J-weighted tendencies (tracer_tendency_encoding="
            f"{TRACER_TENDENCY_ENCODING!r}); do not adapt it")
    _other = _tags - {MIXING_RATIO_ENCODING}
    if _other:
        raise ValueError(
            f"mixing_ratio_physics_adapter: tracer_tendency_encoding "
            f"{sorted(_other)} (on the function or its wrapper chain) is not "
            f"the mixing-ratio convention ({MIXING_RATIO_ENCODING!r} or "
            "untagged) this adapter converts from")

    @functools.wraps(legacy_physics_fn)
    def physics_fn(state, grid, height_coord, terrain_metric):
        _require_tracer_density(state)      # decode only the current encoding
        J = terrain_metric.jacobian
        n_sh, nlev, n_tr = state.tracers_hat.data.shape
        theta_p = sh_synthesis_3d(grid, state.theta_prime_hat.data)
        rho_p = sh_synthesis_3d(grid, state.rho_prime_hat.data) / J[..., None]
        _, rho = sanitize_theta_rho(height_coord.theta_ref + theta_p,
                                    height_coord.rho_ref + rho_p)
        rq = sh_synthesis_3d(
            grid, state.tracers_hat.data.reshape(n_sh, nlev * n_tr),
        ).reshape(grid.n_lat, grid.n_lon, nlev, n_tr) / J[..., None, None]
        q = rq / rho[..., None]
        view = state._replace(           # legacy encoding + metadata
            rho_prime_hat=state.rho_prime_hat.replace(
                data=sh_analysis_3d(grid, rho_p), name="rho_prime_hat"),
            tracers_hat=state.tracers_hat.replace(
                data=sh_analysis_3d(
                    grid, q.reshape(grid.n_lat, grid.n_lon, nlev * n_tr),
                ).reshape(n_sh, nlev, n_tr),
                name="tracers_hat", units="kg/kg"),
        )
        result = legacy_physics_fn(view, grid, height_coord, terrain_metric)
        is_tuple = (not isinstance(result, SpectralNHState)
                    and isinstance(result, tuple))
        tend = result[0] if is_tuple else result
        drho = sh_synthesis_3d(grid, tend.rho_prime_hat.data)
        dq = sh_synthesis_3d(
            grid, tend.tracers_hat.data.reshape(n_sh, nlev * n_tr),
        ).reshape(grid.n_lat, grid.n_lon, nlev, n_tr)
        # Conservative conversion with the PROGNOSTIC density (J rho =
        # J rho_ref + J rho', exactly what compute_dry_mass integrates) in
        # BOTH product-rule terms; the sanitized rho only forms the q the
        # legacy code sees.  Where the raw density is not positive (a
        # column the sanitizer floors) q falls back to the sanitized ratio.
        rho_raw = height_coord.rho_ref + rho_p
        q_raw = jnp.where(rho_raw[..., None] > 0.0,
                          rq / jnp.where(rho_raw > 0.0, rho_raw, 1.0)[..., None],
                          q)
        d_jrq = J[..., None, None] * (rho_raw[..., None] * dq
                                      + q_raw * drho[..., None])
        tend = tend._replace(            # back to the J-weighted encoding
            rho_prime_hat=tend.rho_prime_hat.replace(
                data=sh_analysis_3d(grid, J[..., None] * drho),
                name=_JRHO_NAME, units="kg/m^3/s"),
            tracers_hat=tend.tracers_hat.replace(
                data=sh_analysis_3d(
                    grid, d_jrq.reshape(grid.n_lat, grid.n_lon, nlev * n_tr),
                ).reshape(n_sh, nlev, n_tr),
                name=_JRQ_NAME, units="kg/m^3/s"),
        )
        return (tend,) + tuple(result[1:]) if is_tuple else tend

    physics_fn.tracer_tendency_encoding = TRACER_TENDENCY_ENCODING
    return physics_fn


# =============================================================================
# Model class
# =============================================================================

class SpectralCompressibleEulerModel:
    """Spectral non-hydrostatic compressible Euler model.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid with SH transform matrices.
    height_coord : HeightCoordinate
        Height-based vertical coordinate with reference state.
    terrain_metric : TerrainMetric
        Terrain metric (Jacobian, physical heights).
    config : SpectralNHConfig, optional
        Model configuration.
    """

    def __init__(
        self,
        grid: GaussianGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
        config: SpectralNHConfig | None = None,
        *,
        allow_unsupported_backend: bool = False,
        legoesm_config=None,
    ):
        self.height_coord = height_coord
        self.terrain_metric = terrain_metric
        self.config = config or SpectralNHConfig()
        if self.config.small_earth_factor != 1.0:
            factor = self.config.small_earth_factor
            grid = grid._replace(
                radius=constants.R_earth / factor,
                f=(grid.f * factor if self.config.rotating
                   else jnp.zeros_like(grid.f)),
            )
        elif not self.config.rotating:
            grid = grid._replace(f=jnp.zeros_like(grid.f))
        self._use_cpu_for_spectral = False
        self._cpu_device = None
        self._default_device = None

        if legoesm_config is not None:
            allow_unsupported_backend = bool(
                legoesm_config.get(
                    "atmosphere.spectral.allow_unsupported", False
                )
            )

        placement = place_spectral_grid(
            grid, allow_unsupported=allow_unsupported_backend
        )
        self.grid = placement.grid
        self._use_cpu_for_spectral = placement.use_cpu_for_spectral
        self._cpu_device = placement.cpu_device
        self._default_device = placement.default_device
        # iter-9: lazy fp64 dry-mass snapshot for anchor-to-initial.
        self._target_mass = None

        # State-truncation mask for the Orszag 2/3 rule (the band-limit
        # contract of the spectral SW / PE / tracer twins).  The slow
        # TENDENCIES are masked in ``spectral_nh_slow_tendencies``, which
        # FREEZES (does not remove) upper-third modes of the state: initial
        # content (e.g. the under-resolved TC3 warm bubble) and the
        # upper-band power the grid-space acoustic update and vertical
        # tracer transport put back every stage.  Those frozen modes are
        # neither advected, diffused nor given a horizontal pressure
        # gradient, yet the vertical acoustic columns keep acting on them.
        # Truncating the incoming state and every acoustic-update output
        # enforces the band-limit at every stage;
        # exact no-op for band-limited states, and the n=0 row (global
        # mass / tracer integrals) is never touched.
        self._dealias_state = None
        if self.config.dealiasing_fraction > 0.0:
            self._dealias_state = dealiasing_mask(
                self.grid, self.config.dealiasing_fraction,
            )
            # The terrain must lie in the same band: J = (H - z_s)/H weights
            # every mass prognostic, so off-band orography would let the
            # truncation imprint the terrain on a uniform tracer (1.2e-2 at
            # TC2a T21).  Grid-space check, so terrain the transform cannot
            # represent at all (n > n_max) is refused too.  Build orography
            # with :func:`band_limited_orography`.  Checked on the Jacobian
            # the solver consumes: max over points of |J - P(J)| / J (the
            # uniform-tracer error it would cause), tolerance max(1e-10,
            # 10 * dtype eps); the fp64 projection's own round-off in z_s is
            # <= 6.5e-13 relative (TC1/TC2a, T21-T85).
            J = terrain_metric.jacobian
            if self._use_cpu_for_spectral:     # same device as self.grid
                J = jax.device_put(J, self._cpu_device)
            if J.dtype not in (jnp.float32, jnp.float64):
                raise ValueError(
                    f"spectral NH terrain Jacobian must be float32/float64, "
                    f"got {J.dtype}")
            if not bool(jnp.all(jnp.isfinite(J))):
                raise ValueError("spectral NH terrain Jacobian is not finite")
            _jmin = float(jnp.min(J))
            if not _jmin > 0.0:
                raise ValueError(
                    f"spectral NH terrain gives min J = {_jmin} (needs "
                    "z_s < H everywhere)")
            _off = float(jnp.max(jnp.abs(J - band_limited_orography(
                self.grid, J, self.config.dealiasing_fraction)) / J))
            _tol = max(1e-10, 10.0 * float(jnp.finfo(J.dtype).eps))
            if not _off <= _tol:
                raise ValueError(
                    "spectral NH terrain is not band-limited to the active "
                    f"band (dealiasing_fraction="
                    f"{self.config.dealiasing_fraction}): relative Jacobian "
                    f"error {_off:.3e} > {_tol:.1e}.  Build "
                    "z_s with band_limited_orography(grid, z_s, fraction).")

        # Warn if de-aliasing is off — the NH slow tendencies are
        # quadratic/cubic and alias without the Orszag 2/3 truncation.
        # This is the PRIMARY stabiliser (hyperdiff_coeff defaults to 0).
        if self.config.dealiasing_fraction == 0.0:
            import warnings
            warnings.warn(
                "dealiasing_fraction=0.0: spectral aliasing from the "
                "nonlinear NH tendencies is not suppressed (and "
                "hyperdiff_coeff defaults to 0.0). Set "
                "dealiasing_fraction=0.667 for production runs.",
                stacklevel=2,
            )

    def reset_target_mass(self) -> None:
        """Clear the anchored mass target (iter-18; see iter-4 SW twin)."""
        self._target_mass = None

    def set_target_mass(self, target_mass) -> None:
        """Explicitly set the anchored mass target (iter-19)."""
        self._target_mass = target_mass

    def compute_dry_mass(self, state) -> jax.Array:
        """Global dry mass ``∫ (J · rho_ref + J rho') · dz · dA`` (fp64);
        ``rho_prime_hat`` is the J-weighted J*rho'."""
        _require_tracer_density(state)
        jrho_p_grid = sh_synthesis_3d(self.grid, state.rho_prime_hat.data)
        J = self.terrain_metric.jacobian                    # (n_lat, n_lon)
        dz = self.height_coord.dz                           # (nlev,)
        col_mass = jnp.sum(
            (J[..., None] * self.height_coord.rho_ref + jrho_p_grid)
            * dz[None, None, :], axis=-1,
        )                                                   # (n_lat, n_lon)
        acc = jnp.float64
        return jnp.sum(
            col_mass.astype(acc) * self.grid.grid_area.astype(acc),
        )

    def compute_tracer_mass(self, state) -> jax.Array:
        """Global tracer mass ``∫ (J rho q) · dz · dA`` per tracer (fp64).

        ``tracers_hat`` is the J-weighted tracer density J*rho*q, so this
        is the conserved budget quantity (e.g. total water = sum over q_v,
        q_c, q_r).
        """
        _require_tracer_density(state)
        n_sh, nlev, n_tr = state.tracers_hat.data.shape
        jrq = sh_synthesis_3d(
            self.grid, state.tracers_hat.data.reshape(n_sh, nlev * n_tr),
        ).reshape(self.grid.n_lat, self.grid.n_lon, nlev, n_tr)
        w = jnp.broadcast_to(self.height_coord.dz,
                             (self.grid.n_lat, self.grid.n_lon, nlev))
        acc = jnp.float64
        return jnp.einsum(
            "ijkt,ijk,ij->t", jrq.astype(acc), w.astype(acc),
            self.grid.grid_area.astype(acc),
        )

    def _apply_state_truncation(self, state):
        """Truncate the prognostic state to the de-aliased 2/3 band.

        Multiplies vor/div/w/theta'/J*rho' and the tracer densities by the
        0/1 dealiasing mask -- the SAME fields whose slow tendencies are
        masked in :func:`spectral_nh_slow_tendencies`.  ``phis_hat`` is
        static forcing and is never truncated.  No-op when
        ``dealiasing_fraction == 0``.
        """
        if self._dealias_state is None:
            return state
        m = self._dealias_state

        def _cut(f):
            return f.replace(data=f.data * m.reshape(
                (-1,) + (1,) * (f.data.ndim - 1)).astype(f.data.dtype))

        return state._replace(
            vor_hat=_cut(state.vor_hat),
            div_hat=_cut(state.div_hat),
            w_hat=_cut(state.w_hat),
            theta_prime_hat=_cut(state.theta_prime_hat),
            rho_prime_hat=_cut(state.rho_prime_hat),
            tracers_hat=_cut(state.tracers_hat),
        )

    def _apply_mass_fixer(self, state, target_mass=None):
        """Anchor ``∫ (J · rho_ref + J rho') · dz · dA`` to ``target_mass``
        (falls back to ``self._target_mass`` for the eager call sites).

        Uniform additive correction of the J-weighted prognostic J*rho':
        ``Δ = (target − current) / (∫ dz · dA)``.  Adding ``Δ`` to J*rho'
        in physical space is equivalent to adding ``Δ · sqrt(4π)`` to
        ``rho_prime_hat[0, :]`` (the (n=0,m=0) row), broadcast across all
        vertical levels (rho' itself changes by Δ/J; flat terrain: the
        former cubed-sphere / MPAS ``fix_mass_nonhydrostatic`` convention,
        bit-identically).

        Only rho' is corrected: the tracer DENSITY (tracer mass) is left
        untouched, so tracer mass stays conserved and the recovered mixing
        ratio q = (rho q) / rho is rescaled by rho / (rho + delta_rho) wherever the
        fixer applies a finite correction (rescaling rho*q with it would
        create or destroy tracer mass).
        """
        jrho_p_grid = sh_synthesis_3d(self.grid, state.rho_prime_hat.data)
        J = self.terrain_metric.jacobian
        dz = self.height_coord.dz
        col_mass = jnp.sum(
            (J[..., None] * self.height_coord.rho_ref + jrho_p_grid)
            * dz[None, None, :], axis=-1,
        )
        if target_mass is None:
            target_mass = self._target_mass
        acc = jnp.float64
        area_acc = self.grid.grid_area.astype(acc)
        current_mass = jnp.sum(col_mass.astype(acc) * area_acc)
        total_vol = jnp.sum(area_acc) * jnp.sum(dz.astype(acc))
        delta_rho = (target_mass - current_mass) / total_vol
        sqrt_4pi = jnp.sqrt(jnp.asarray(4.0 * jnp.pi, dtype=acc))
        rho_hat = state.rho_prime_hat.data
        rho_hat_new = rho_hat.at[0, :].add(
            (delta_rho * sqrt_4pi).astype(rho_hat.dtype),
        )
        return state._replace(
            rho_prime_hat=state.rho_prime_hat.replace(data=rho_hat_new),
        )

    def _build_se_functions(self, physics_fn=None):
        """Build slow tendency and acoustic update functions for split-explicit.

        ``physics_fn(state, grid, height_coord, terrain_metric)`` (optional)
        returns a :class:`SpectralNHState` of SPECTRAL tendencies; it is
        evaluated at every RK stage and passed to
        :func:`spectral_nh_slow_tendencies` as ``physics_tendency`` (so the
        Orszag de-aliasing mask applies to it too).  Same contract as the
        MPAS NH ``step(physics_fn=...)``, including the plain
        ``(tendencies, diagnostics)`` tuple return.  Only the ``vor_hat``,
        ``div_hat``, ``theta_prime_hat``, ``rho_prime_hat`` and
        ``tracers_hat`` tendencies are applied; ``w_hat`` and ``phis_hat``
        of the returned state are not read.  Tendencies are for the
        PROGNOSTIC variables: ``rho_prime_hat`` -> ``d(J rho')/dt`` and
        ``tracers_hat`` -> ``d(J rho q)/dt`` [kg/m^3/s] (for physics that
        leaves rho unchanged, ``J * rho * dq/dt``), not ``dq/dt``; the
        state it receives is J-weighted too.  The callable must declare
        this with ``tracer_tendency_encoding == TRACER_TENDENCY_ENCODING``
        (:func:`tag_tracer_tendency_encoding`; checked in :meth:`step`).
        Physics written for the former convention (q and unweighted rho'
        in, ``dq/dt`` out) is run through :func:`mixing_ratio_physics_adapter`.
        """
        def slow_tendency_fn(s):
            phys = None
            if physics_fn is not None:
                _phys_result = physics_fn(
                    s, self.grid, self.height_coord, self.terrain_metric,
                )
                phys = (_phys_result[0] if type(_phys_result) is tuple
                        else _phys_result)
            return spectral_nh_slow_tendencies(
                s, self.grid, self.height_coord,
                self.terrain_metric, self.config, phys,
            )

        def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
            # Convert acoustic variables from spectral to grid.  Theta_p
            # and rho_p share (n_sh, nlev); w_hat has (n_sh, nlev+1).
            # Concatenate all three along the trailing axis and run a
            # single ``sh_synthesis_3d`` — the 3D transform treats the
            # trailing axis as a passive batch, so different "level"
            # axes in different fields combine cleanly into one
            # ``segment_sum`` + IRFFT.  3 SH syntheses → 1.  Loop 179
            # extends Loop 97.
            n_sh_a, nlev_a = s.theta_prime_hat.data.shape
            s.w_hat.data.shape[-1]  # nlev + 1
            theta_rho_w_hat = jnp.concatenate(
                [s.theta_prime_hat.data, s.rho_prime_hat.data, s.w_hat.data],
                axis=-1,
            )  # (n_sh, 2*nlev + (nlev+1))
            theta_rho_w_grid = sh_synthesis_3d(self.grid, theta_rho_w_hat)
            J = self.terrain_metric.jacobian
            theta_p_grid = theta_rho_w_grid[..., :nlev_a]
            # Prognostic is J*rho'; the acoustic solver works on rho'.
            rho_p_grid = theta_rho_w_grid[..., nlev_a:2 * nlev_a] / J[..., None]
            w_grid = theta_rho_w_grid[..., 2 * nlev_a:]

            # Run acoustic substeps in grid space
            acoustic_fn = (
                _acoustic_substeps_grid_semi_implicit
                if self.config.semi_implicit_acoustic
                else _acoustic_substeps_grid
            )
            w_new, theta_p_new, rho_p_new, mass_flux = acoustic_fn(
                w_grid, theta_p_grid, rho_p_grid,
                dt_s, n_sub,
                self.height_coord, self.terrain_metric, self.config,
            )

            # Vertical tracer-mass transport with the SAME substep-integrated
            # mass flux that just moved rho' (Rule 0; WRF/MPAS practice).
            # q_up uses the stage's pre-acoustic rho (the denominator of the
            # tracer it was advected with), floored by the same
            # sanitize_theta_rho policy the acoustic solver applies.
            tr_hat = s.tracers_hat.data
            n_tr = tr_hat.shape[-1]
            rq = sh_synthesis_3d(
                self.grid, tr_hat.reshape(n_sh_a, nlev_a * n_tr),
            ).reshape(self.grid.n_lat, self.grid.n_lon, nlev_a, n_tr) / J[
                ..., None, None]                               # rho*q
            _, rho_stage = sanitize_theta_rho(
                self.height_coord.theta_ref + theta_p_grid,
                self.height_coord.rho_ref + rho_p_grid,
            )
            rq_new = _tracer_vertical_transport(
                rq, rho_stage, mass_flux,
                self.height_coord.dz, self.terrain_metric.jacobian,
            )
            tr_hat_new = sh_analysis_3d(
                self.grid,
                (rq_new * J[..., None, None]).reshape(
                    self.grid.n_lat, self.grid.n_lon, nlev_a * n_tr),
            ).reshape(n_sh_a, nlev_a, n_tr)

            # Convert back to spectral via the same concat trick — 3 SH
            # analyses → 1.  Slot order matches the synthesis so we can
            # slice the result back into (theta_hat, rho_hat, w_hat).
            theta_rho_w_new = jnp.concatenate(
                [theta_p_new, rho_p_new * J[..., None], w_new], axis=-1,
            )  # (n_lat, n_lon, 2*nlev + (nlev+1))
            theta_rho_w_new_hat = sh_analysis_3d(self.grid, theta_rho_w_new)
            theta_p_new_hat = theta_rho_w_new_hat[..., :nlev_a]
            rho_p_new_hat = theta_rho_w_new_hat[..., nlev_a:2 * nlev_a]
            w_new_hat = theta_rho_w_new_hat[..., 2 * nlev_a:]
            # The grid-space vertical update re-creates upper-band power in
            # w/theta'/J rho'/tracers; project back to the active band so
            # every stage (not only the step's end) stays band-limited.
            return self._apply_state_truncation(SpectralNHState(
                vor_hat=s.vor_hat,
                div_hat=s.div_hat,
                w_hat=s.w_hat.replace(data=w_new_hat),
                theta_prime_hat=s.theta_prime_hat.replace(
                    data=theta_p_new_hat,
                ),
                rho_prime_hat=s.rho_prime_hat.replace(
                    data=rho_p_new_hat,
                ),
                phis_hat=s.phis_hat,
                tracers_hat=s.tracers_hat.replace(data=tr_hat_new),
            ))

        return slow_tendency_fn, acoustic_update_fn

    def step(self, state: SpectralNHState, dt: float,
             physics_fn=None) -> SpectralNHState:
        """Outer wrapper: snapshots dry mass on first call when
        ``anchor_mass_to_initial`` is on (fp64, outside JIT).

        ``physics_fn`` (optional, static under JIT): see
        :meth:`_build_se_functions`.  ``None`` = dynamics only.
        """
        # No PhysicsState carry is threaded here — refuse a stateful
        # physics_fn rather than silently reseeding it every step.
        refuse_unthreaded_stateful_physics(
            physics_fn, None, where="spectral NH step()")
        _require_physics_encoding(physics_fn)
        _require_tracer_density(state)
        if (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and self._target_mass is None):
            self._target_mass = self.compute_dry_mass(state)
        return self._step_jit(state, dt, physics_fn)

    @partial(jax.jit, static_argnums=(0, 3))
    def _step_jit(self, state: SpectralNHState, dt: float,
                  physics_fn=None) -> SpectralNHState:
        """Advance one time step using split-explicit RK3.

        Slow tendencies use spectral horizontal operators.
        Acoustic substeps run in grid space (purely vertical).
        """
        se_config = SplitExplicitConfig(
            n_substeps=self.config.n_acoustic_substeps,
        )
        slow_tendency_fn, acoustic_update_fn = self._build_se_functions(
            physics_fn)
        # 2/3-rule STATE truncation (see __init__) of the incoming state
        # (IC / restart upper band); the stages then stay in the band
        # (masked slow tendencies, truncated acoustic update), so the
        # returned state is band-limited.  n=0 is kept: masses unchanged.
        # Applied on the execution device (the mask lives with self.grid).
        if self._use_cpu_for_spectral:
            state_cpu = self._apply_state_truncation(
                jax.device_put(state, self._cpu_device))
            result_cpu = split_explicit_step(
                state_cpu, slow_tendency_fn, acoustic_update_fn,
                dt, se_config,
            )
            state_new = jax.device_put(result_cpu, self._default_device)
        else:
            state_new = split_explicit_step(
                self._apply_state_truncation(state), slow_tendency_fn,
                acoustic_update_fn, dt, se_config,
            )

        # iter-9: anchored dry-mass fixer.  ``_target_mass`` is None
        # when disabled OR before the first ``step()`` call (snapshot
        # happens in the Python wrapper).  When set, it's a fp64 scalar
        # that JIT captures as a closure constant.
        if (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and self._target_mass is not None):
            state_new = self._apply_mass_fixer(state_new)

        return state_new

    @partial(jax.jit, static_argnums=(0,))
    def _step_on_cpu(self, state: SpectralNHState, dt: float,
                     target_mass=None) -> SpectralNHState:
        """Step without device transfers (for batched CPU integration on Metal).

        Applies the same anchored-mass fixer as :meth:`_step_jit` — the
        batched path previously skipped it entirely, so long batched-Metal
        integrations silently ran unanchored (codex 2026-07-12).
        ``target_mass`` is a TRACED argument (codex round 2): reading
        ``self._target_mass`` here would freeze the first target into the
        compiled closure, ignoring a later ``set_target_mass``.
        ``_integrate_on_cpu`` takes the snapshot before the loop.
        """
        se_config = SplitExplicitConfig(
            n_substeps=self.config.n_acoustic_substeps,
        )
        slow_tendency_fn, acoustic_update_fn = self._build_se_functions()
        state_new = split_explicit_step(
            self._apply_state_truncation(state), slow_tendency_fn,
            acoustic_update_fn, dt, se_config,
        )
        if (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and target_mass is not None):
            state_new = self._apply_mass_fixer(state_new, target_mass)
        return state_new

    def integrate(
        self,
        state: SpectralNHState,
        duration: float,
        dt: float,
        save_every: int = 1,
    ) -> tuple[SpectralNHState, list]:
        """Integrate forward for a given duration (Python loop).

        On Metal, batches CPU transfers: transfer state to CPU once,
        run all steps on CPU, then transfer results back to Metal.
        This avoids per-step CPU↔Metal round-trips.
        """
        n_steps = int(duration / dt)

        if self._use_cpu_for_spectral:
            return self._integrate_on_cpu(state, n_steps, dt, save_every)

        trajectory = [state]
        for i in range(n_steps):
            state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)
        return state, trajectory

    def _integrate_on_cpu(self, state, n_steps, dt, save_every):
        """Batch integration on CPU: transfer once, not per step."""
        state_cpu = jax.device_put(state, self._cpu_device)
        trajectory_cpu = [state_cpu]

        # Anchored-mass snapshot (mirrors step(); the batched path calls
        # _step_on_cpu directly, so the snapshot must happen here or the
        # fixer never engages).  Taken BEFORE tracing so the jitted step
        # sees a non-None target on its first trace.
        if (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and self._target_mass is None):
            self._target_mass = self.compute_dry_mass(state_cpu)

        for i in range(n_steps):
            state_cpu = self._step_on_cpu(state_cpu, dt, self._target_mass)
            if (i + 1) % save_every == 0:
                trajectory_cpu.append(state_cpu)

        # Transfer back to Metal
        state_out = jax.device_put(state_cpu, self._default_device)
        trajectory_out = [
            jax.device_put(s, self._default_device) for s in trajectory_cpu
        ]
        return state_out, trajectory_out


# =============================================================================
# Initialization helpers
# =============================================================================

def band_limited_orography(grid: GaussianGrid, z_s: jax.Array,
                           dealiasing_fraction: float) -> jax.Array:
    """Project grid orography onto the model's ACTIVE spectral band.

    The model truncates its prognostic state to the Orszag band
    ``n <= floor(dealiasing_fraction * n_max)`` after every step
    (:meth:`SpectralCompressibleEulerModel._apply_state_truncation`).  The
    terrain enters through ``J = (H - z_s)/H`` (linear in ``z_s``), which
    weights every mass prognostic (``J rho'``, ``J rho q``); an orography
    with power above the band would make ``J * rho_ref * q0`` (a uniform
    tracer) not band-limited, so the state truncation would imprint the
    terrain on the recovered mixing ratio.  Truncating ``z_s`` to the same
    band keeps ``J``, ``phis`` and the J-weighted prognostics consistent
    (standard spectral-model practice: orography at model resolution).
    ``dealiasing_fraction <= 0`` (no state truncation) returns ``z_s``
    unchanged (the former terrain of that configuration).
    """
    if dealiasing_fraction <= 0.0:
        return z_s
    mask = dealiasing_mask(grid, dealiasing_fraction)
    z_hat = sh_analysis(grid, z_s.astype(jnp.float64)) * mask
    return sh_synthesis(grid, z_hat)


def dcmip25_tc1_init_spectral(
    grid: GaussianGrid,
    n_levels: int = 88,
    params: dict | None = None,
    dealiasing_fraction: float = SpectralNHConfig._field_defaults["dealiasing_fraction"],
) -> tuple['SpectralNHState', HeightCoordinate, 'TerrainMetric']:
    """Initialize DCMIP-2025 TC1 (gravity waves) on Gaussian grid.

    Evaluates the TC1 initial condition (piecewise lapse rate, uniform
    horizontal wind u=u0*cos(lat), Schaer mountain topography) on the
    Gaussian grid, then transforms to spectral space.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid with SH transform matrices.
    n_levels : int
        Number of vertical levels.
    params : dict, optional
        Override default parameters (see dcmip2025.test_case_1.TC1_PARAMS).
    dealiasing_fraction : float
        Active spectral band of the model the state is built for; the
        orography is truncated to it (:func:`band_limited_orography`).
        Must equal the model's ``SpectralNHConfig.dealiasing_fraction``.

    Returns
    -------
    state : SpectralNHState
        Initial state in spectral space.
    height_coord : HeightCoordinate
        Vertical coordinate with reference state.
    terrain_metric : TerrainMetric
        Terrain metric on Gaussian grid.
    """
    from legoesm.atmosphere.dynamics.gcm.dcmip2025_ic import (
        piecewise_lapse_theta_ref,
        TC1_PARAMS,
    )

    p = {**TC1_PARAMS, **(params or {})}

    # Reference state with piecewise lapse rate
    theta_fn = piecewise_lapse_theta_ref(
        T_s=p["T_s"],
        lapse_tropo=p["lapse_tropo"],
        lapse_strato=p["lapse_strato"],
        z_tropopause=p["z_tropopause"],
    )

    # Vertical coordinate
    height_coord = create_height_coordinate(n_levels, p["H"], theta_fn)

    # --- Topography on Gaussian grid ---
    # Schaer mountain: z_s = h0 * exp(-(d/halfwidth)^2)
    lat_2d = grid.lat[:, None] * jnp.ones(grid.n_lon)[None, :]  # (n_lat, n_lon)
    lon_2d = grid.lon2d
    lat0 = p["mountain_lat"]
    lon0 = p["mountain_lon"]
    h0 = p["mountain_height"]
    halfwidth = p["mountain_halfwidth"]

    dlat = lat_2d - lat0
    dlon = lon_2d - lon0
    a_hav = (
        jnp.sin(dlat / 2) ** 2
        + jnp.cos(lat_2d) * jnp.cos(lat0) * jnp.sin(dlon / 2) ** 2
    )
    angular_dist = 2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(a_hav, 0.0, 1.0)))
    dist = angular_dist * grid.radius
    z_s = h0 * jnp.exp(-(dist / halfwidth) ** 2)
    # Orography at the model's active band (see band_limited_orography);
    # ``dealiasing_fraction`` must equal the model config's.
    z_s = band_limited_orography(grid, z_s, dealiasing_fraction)

    # Terrain metric
    terrain_metric = compute_terrain_metric(z_s, height_coord)

    # --- Initial conditions on Gaussian grid ---
    n_lat = grid.n_lat
    n_lon = grid.n_lon
    nlev = n_levels

    # Horizontal wind: u = u0 * cos(lat), v = 0
    u0 = p["u0"]
    u_grid = jnp.ones((n_lat, n_lon, nlev), dtype=jnp.float64) * (
        u0 * jnp.cos(grid.lat)
    )[:, None, None]
    v_grid = jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64)

    # Transform u,v to spectral vorticity/divergence
    a_rad = grid.radius
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a_rad
    one_over_a = 1.0 / a_rad

    cos_lat_3d = grid.cos_lat[:, None, None]
    u_cos = u_grid * cos_lat_3d
    v_cos = v_grid * cos_lat_3d

    vor_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, v_cos)
        + one_over_a * sh_analysis_dmu_3d(grid, u_cos)
    )
    div_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, v_cos)
    )

    # w = 0, perturbations = 0
    n_sh = grid.n_sh
    w_hat = jnp.zeros((n_sh, nlev + 1), dtype=jnp.complex128)
    theta_p_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    rho_p_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)

    # Surface geopotential
    phis_grid = constants.g * z_s
    phis_hat = sh_analysis(grid, phis_grid.astype(jnp.float64))

    # No tracers for dry dynamics
    tracers_hat = jnp.zeros((n_sh, nlev, 1), dtype=jnp.complex128)

    dims_3d = ("spectral", "level")
    dims_w = ("spectral", "level_half")
    dims_2d = ("spectral",)
    dims_tr = ("spectral", "level", "tracer")

    state = SpectralNHState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        w_hat=Field(data=w_hat, name="w_hat", dims=dims_w, units="m/s"),
        theta_prime_hat=Field(
            data=theta_p_hat, name="theta_prime_hat", dims=dims_3d, units="K",
        ),
        rho_prime_hat=Field(
            data=rho_p_hat, name=_JRHO_NAME, dims=dims_3d, units="kg/m^3",
        ),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
        tracers_hat=Field(
            data=tracers_hat, name=_JRQ_NAME, dims=dims_tr, units="kg/m^3",
        ),
    )

    return state, height_coord, terrain_metric


def nh_rest_state_spectral(
    grid: GaussianGrid,
    height_coord: HeightCoordinate,
    n_tracers: int = 0,
) -> SpectralNHState:
    """Create a rest-state initial condition in spectral space.

    All perturbations are zero, wind is zero.
    """
    nlev = len(height_coord.z_full)
    n_sh = grid.n_sh

    dims_3d = ("spectral", "level")
    dims_w = ("spectral", "level_half")
    dims_2d = ("spectral",)
    dims_tr = ("spectral", "level", "tracer")

    vor_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    div_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    w_hat = jnp.zeros((n_sh, nlev + 1), dtype=jnp.complex128)
    theta_p_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    rho_p_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    phis_hat = jnp.zeros(n_sh, dtype=jnp.complex128)
    tracers_hat = jnp.zeros((n_sh, nlev, max(n_tracers, 1)), dtype=jnp.complex128)

    return SpectralNHState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        w_hat=Field(data=w_hat, name="w_hat", dims=dims_w, units="m/s"),
        theta_prime_hat=Field(
            data=theta_p_hat, name="theta_prime_hat", dims=dims_3d, units="K",
        ),
        rho_prime_hat=Field(
            data=rho_p_hat, name=_JRHO_NAME, dims=dims_3d, units="kg/m^3",
        ),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
        tracers_hat=Field(
            data=tracers_hat, name=_JRQ_NAME, dims=dims_tr, units="kg/m^3",
        ),
    )


def dcmip25_tc2_init_spectral(
    grid: GaussianGrid,
    n_levels: int = 48,
    subcase: str = "a",
    params: dict | None = None,
    dealiasing_fraction: float = SpectralNHConfig._field_defaults["dealiasing_fraction"],
) -> tuple['SpectralNHState', HeightCoordinate, 'TerrainMetric']:
    """Initialize DCMIP-2025 TC2 (mountain flow) on Gaussian grid.

    Isothermal atmosphere with solid-body rotation and mountain topography
    on a small Earth (radius/20). Transforms to spectral space.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid (used for n_max only; a small-Earth grid is created).
    n_levels : int
        Number of vertical levels.
    subcase : str
        "a" for gap flow, "b" for vortex shedding.
    params : dict, optional
        Override default parameters.
    dealiasing_fraction : float
        Active spectral band of the model the state is built for; the
        orography is truncated to it (:func:`band_limited_orography`).
        Must equal the model's ``SpectralNHConfig.dealiasing_fraction``.
    """
    from legoesm.atmosphere.dynamics.gcm.dcmip2025_ic import (
        isothermal_theta_ref,
        TC2_PARAMS,
    )

    p = {**TC2_PARAMS, **(params or {})}
    factor = p["small_earth_factor"]
    # The rotation rate MUST be scaled the same way as on the cube and MPAS
    # (Omega*X preserves the Rossby number on a radius/X planet, which is the
    # DCMIP-2025 small-planet convention).  This call previously omitted
    # ``omega`` entirely, so the spectral arm silently ran at Omega while the
    # cube arm ran at Omega*20 for the SAME case — a 20x cross-grid confound
    # that invalidated any spectral-vs-cube comparison (codex r2 P1).
    small_grid = create_gaussian_grid(
        grid.n_max, radius=constants.R_earth / factor,
        omega=(constants.Omega * factor
               if p.get("rotating", True) else 0.0))

    theta_fn = isothermal_theta_ref(T0=p["T0"])
    height_coord = create_height_coordinate(n_levels, p["H"], theta_fn)

    n_lat, n_lon = small_grid.n_lat, small_grid.n_lon
    lat_2d = small_grid.lat[:, None] * jnp.ones(n_lon)[None, :]
    lon_2d = small_grid.lon2d

    if subcase == "a":
        dlon = jnp.mod(lon_2d - p["chain_lon"] + jnp.pi, 2 * jnp.pi) - jnp.pi
        x_dist = dlon * small_grid.radius * jnp.cos(lat_2d)
        y_dist = (lat_2d - p["gap_lat"]) * small_grid.radius
        z_s = (p["chain_h0"]
               * jnp.exp(-(x_dist / p["chain_halfwidth_lon"]) ** 2)
               * jnp.exp(-(y_dist / p["chain_halfwidth_lat"]) ** 4)
               * (1.0 - jnp.exp(-(y_dist / p["gap_halfwidth"]) ** 2)))
    elif subcase == "b":
        dlat = lat_2d - p["mountain_lat"]
        dlon = lon_2d - p["mountain_lon"]
        a_hav = (jnp.sin(dlat / 2) ** 2
                 + jnp.cos(lat_2d) * jnp.cos(p["mountain_lat"])
                 * jnp.sin(dlon / 2) ** 2)
        dist = 2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(a_hav, 0.0, 1.0))) * small_grid.radius
        z_s = p["mountain_h0"] * jnp.exp(-(dist / p["mountain_d"]) ** 2)
    else:
        raise ValueError(f"Unknown subcase: {subcase!r}")
    # Orography at the model's active band (see band_limited_orography);
    # ``dealiasing_fraction`` must equal the model config's.
    z_s = band_limited_orography(small_grid, z_s, dealiasing_fraction)

    terrain_metric = compute_terrain_metric(z_s, height_coord)

    nlev = n_levels
    u0 = p["u0"]
    u_grid = (jnp.ones((n_lat, n_lon, nlev), dtype=jnp.float64)
              * (u0 * jnp.cos(small_grid.lat))[:, None, None])
    v_grid = jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64)

    a_rad = small_grid.radius
    im_over_a = 1j * small_grid.ms.astype(jnp.float64) / a_rad
    one_over_a = 1.0 / a_rad
    cos_lat_3d = small_grid.cos_lat[:, None, None]
    u_cos = u_grid * cos_lat_3d
    v_cos = v_grid * cos_lat_3d

    vor_hat = (im_over_a[:, None] * sh_analysis_oc2_3d(small_grid, v_cos)
               + one_over_a * sh_analysis_dmu_3d(small_grid, u_cos))
    div_hat = (im_over_a[:, None] * sh_analysis_oc2_3d(small_grid, u_cos)
               - one_over_a * sh_analysis_dmu_3d(small_grid, v_cos))

    n_sh = small_grid.n_sh
    w_hat = jnp.zeros((n_sh, nlev + 1), dtype=jnp.complex128)
    theta_p_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    rho_p_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    phis_hat = sh_analysis(small_grid, (constants.g * z_s).astype(jnp.float64))
    tracers_hat = jnp.zeros((n_sh, nlev, 1), dtype=jnp.complex128)

    dims_3d = ("spectral", "level")
    dims_w = ("spectral", "level_half")
    dims_2d = ("spectral",)
    dims_tr = ("spectral", "level", "tracer")

    return SpectralNHState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        w_hat=Field(data=w_hat, name="w_hat", dims=dims_w, units="m/s"),
        theta_prime_hat=Field(data=theta_p_hat, name="theta_prime_hat", dims=dims_3d, units="K"),
        rho_prime_hat=Field(data=rho_p_hat, name=_JRHO_NAME, dims=dims_3d, units="kg/m^3"),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
        tracers_hat=Field(data=tracers_hat, name=_JRQ_NAME, dims=dims_tr, units="kg/m^3"),
    ), height_coord, terrain_metric


def dcmip25_tc3_init_spectral(
    grid: GaussianGrid,
    n_levels: int = 40,
    params: dict | None = None,
    balanced: bool = True,
) -> tuple['SpectralNHState', HeightCoordinate, 'TerrainMetric']:
    """Initialize DCMIP-2025 TC3 (squall line) on Gaussian grid.

    Wind shear, moisture, and warm bubbles on a small Earth (radius/60).
    Transforms to spectral space.

    ``balanced=True`` (default): the Klemp et al. (2015) / DCMIP2016
    reduced-radius construction -- zonal wind ``u = u_eq(z) cos(lat)`` and
    theta', rho' in hydrostatic + cyclostrophic balance
    (:func:`~legoesm.atmosphere.dynamics.gcm.dcmip2025_ic.cyclostrophic_balance_theta_exner`),
    with the sounding as the equatorial profile.  ``balanced=False``: the
    former IC shared with the cube / MPAS arms -- latitude-independent
    ``u = u_eq(z)`` with rho' = theta' = 0, which no pressure field can
    balance near the poles (u^2 tan(lat)/a diverges) and on the radius/60
    sphere drove polar |w| to 3.3 m/s in 4 min at T21 (vs <0.1 m/s at the
    bubbles).  The warm bubbles are added to theta' at fixed rho' in both
    cases (as in the other arms).

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid (used for n_max only; a small-Earth grid is created).
    n_levels : int
        Number of vertical levels.
    params : dict, optional
        Override default parameters.
    """
    from legoesm.atmosphere.dynamics.gcm.dcmip2025_ic import (
        TC3_PARAMS,
        squall_line_sounding as _squall_line_sounding,
        squall_line_theta_fn as _squall_line_theta_fn,
    )

    p = {**TC3_PARAMS, **(params or {})}
    factor = p["small_earth_factor"]
    # Same-rotation requirement as TC2 above; TC3 is non-rotating by spec, so
    # this resolves to omega=0 and now MATCHES the cube and MPAS arms.
    small_grid = create_gaussian_grid(
        grid.n_max, radius=constants.R_earth / factor,
        omega=(constants.Omega * factor
               if p.get("rotating", True) else 0.0))

    theta_fn = _squall_line_theta_fn(p)
    height_coord = create_height_coordinate(n_levels, p["H"], theta_fn)

    n_lat, n_lon = small_grid.n_lat, small_grid.n_lon
    z_s = jnp.zeros((n_lat, n_lon))
    terrain_metric = compute_terrain_metric(z_s, height_coord)

    nlev = n_levels
    z_full = height_coord.z_full
    T_sounding, _, p_sounding = _squall_line_sounding(z_full, p)

    u_profile = p["U_c"] + p["U_s"] * jnp.minimum(z_full / p["z_s"], 1.0)
    u_grid = jnp.ones((n_lat, n_lon, nlev), dtype=jnp.float64) * u_profile[None, None, :]
    if balanced:
        u_grid = u_grid * small_grid.cos_lat[:, None, None]
    v_grid = jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64)

    a_rad = small_grid.radius
    im_over_a = 1j * small_grid.ms.astype(jnp.float64) / a_rad
    one_over_a = 1.0 / a_rad
    cos_lat_3d = small_grid.cos_lat[:, None, None]
    u_cos = u_grid * cos_lat_3d
    v_cos = v_grid * cos_lat_3d

    vor_hat = (im_over_a[:, None] * sh_analysis_oc2_3d(small_grid, v_cos)
               + one_over_a * sh_analysis_dmu_3d(small_grid, u_cos))
    div_hat = (im_over_a[:, None] * sh_analysis_oc2_3d(small_grid, u_cos)
               - one_over_a * sh_analysis_dmu_3d(small_grid, v_cos))

    n_sh = small_grid.n_sh
    w_hat = jnp.zeros((n_sh, nlev + 1), dtype=jnp.complex128)
    if balanced:
        from legoesm.atmosphere.dynamics.gcm.dcmip2025_ic import (
            tc3_balanced_perturbations,
        )
        theta_bal_p, rho_bal_p = tc3_balanced_perturbations(   # (n_lat, nlev)
            small_grid.lat, height_coord, u_profile)
        ones = jnp.ones((1, n_lon, 1))
        theta_bal_p = theta_bal_p[:, None, :] * ones     # (n_lat, n_lon, nlev)
        rho_bal_p = rho_bal_p[:, None, :] * ones
    else:
        theta_bal_p = jnp.zeros((n_lat, n_lon, nlev))
        rho_bal_p = jnp.zeros((n_lat, n_lon, nlev))
    # Prognostic is J*rho' (flat here, J = 1).
    rho_p_hat = sh_analysis_3d(
        small_grid, terrain_metric.jacobian[..., None] * rho_bal_p)

    RH_profile = jnp.clip(
        p["RH_low"] * jnp.exp(-z_full / p["RH_transition_z"]),
        p["RH_high"], p["RH_low"])
    q_v_profile = RH_profile * saturation_mixing_ratio(T_sounding, p_sounding)

    q_v_grid = jnp.ones((n_lat, n_lon, nlev), dtype=jnp.float64) * q_v_profile[None, None, :]
    q_c_grid = jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64)
    q_r_grid = jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64)

    # Warm bubbles
    lat_2d = small_grid.lat[:, None] * jnp.ones(n_lon)[None, :]
    lon_2d = small_grid.lon2d
    theta_pert = jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64)
    for i in range(p["n_bubbles"]):
        lat_c = (i - p["n_bubbles"] // 2) * p["bubble_spacing"] / small_grid.radius
        dlat = lat_2d - lat_c
        dlon = jnp.mod(lon_2d - p["bubble_lon"] + jnp.pi, 2 * jnp.pi) - jnp.pi
        x_dist = dlon * small_grid.radius * jnp.cos(lat_2d)
        y_dist = dlat * small_grid.radius
        r_horiz = jnp.sqrt(x_dist**2 + y_dist**2)
        z_dist = z_full[None, None, :] - p["bubble_zc"]
        r_norm = jnp.sqrt((r_horiz[..., None] / p["bubble_rh"]) ** 2
                          + (z_dist / p["bubble_rz"]) ** 2)
        theta_pert = theta_pert + p["bubble_dtheta"] * jnp.where(
            r_norm <= 1.0, jnp.cos(0.5 * jnp.pi * r_norm) ** 2, 0.0)

    theta_p_hat = sh_analysis_3d(small_grid, theta_bal_p + theta_pert)
    # Vapour mixing ratio q_v(z) is latitude-independent ON PURPOSE in the
    # balanced state too: DCMIP2016 Eq. 13, q(z, lat) = q_eq(z) (the RH
    # profile is prescribed on the equatorial column only).
    # Prognostic tracer = J * rho * q with the (balanced) total rho; J = 1.
    jrho_3d = terrain_metric.jacobian[..., None] * (
        height_coord.rho_ref[None, None, :] + rho_bal_p)
    tracers_hat = jnp.stack([
        sh_analysis_3d(small_grid, jrho_3d * q_v_grid),
        sh_analysis_3d(small_grid, jrho_3d * q_c_grid),
        sh_analysis_3d(small_grid, jrho_3d * q_r_grid),
    ], axis=-1)
    phis_hat = jnp.zeros(n_sh, dtype=jnp.complex128)

    dims_3d = ("spectral", "level")
    dims_w = ("spectral", "level_half")
    dims_2d = ("spectral",)
    dims_tr = ("spectral", "level", "tracer")

    return SpectralNHState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        w_hat=Field(data=w_hat, name="w_hat", dims=dims_w, units="m/s"),
        theta_prime_hat=Field(data=theta_p_hat, name="theta_prime_hat", dims=dims_3d, units="K"),
        rho_prime_hat=Field(data=rho_p_hat, name=_JRHO_NAME, dims=dims_3d, units="kg/m^3"),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
        tracers_hat=Field(data=tracers_hat, name=_JRQ_NAME, dims=dims_tr, units="kg/m^3"),
    ), height_coord, terrain_metric
