"""Spectral Non-Hydrostatic Compressible Euler Model.

Solves the fully compressible Euler equations on the sphere using
pseudospectral (SH transform) horizontal operators and height-based
terrain-following (z*) coordinates with reference-state subtraction.

Prognostic variables (spectral):
    vor_hat         : Spectral relative vorticity, (n_sh, nlev)
    div_hat         : Spectral divergence, (n_sh, nlev)
    w_hat           : Spectral vertical velocity, (n_sh, nlev+1)
    theta_prime_hat : Spectral pot. temp. perturbation, (n_sh, nlev)
    jrho_prime_hat  : Spectral J*rho' (Jacobian-weighted density
                      perturbation), (n_sh, nlev)
    tracers_hat     : Spectral tracers, (n_sh, nlev, n_tracers)

J*rho' (not rho') is the prognostic density (#2682): continuity and the
paired tracer mass P(J*rho*q) are advanced by the same operators on the
same variable, so dry mass and tracer inventory close to round-off and a
uniform tracer stays uniform over terrain.

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
import numpy as np

from legoesm.core.field import Field
from legoesm.parallel.metal import place_spectral_grid
from legoesm.core.operators_3d import (
    vertical_advection_height,
)
from legoesm.grids.gaussian import (
    GaussianGrid,
    create_gaussian_grid,
    sh_analysis,
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
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    tracer_vertical_face_value,
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
from legoesm.timestepping.split_explicit import (
    split_explicit_step,
    SplitExplicitConfig,
    TracerMassHooks,
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
    Tracers:                (n_sh, nlev, n_tracers) complex128

    ``jrho_prime_hat`` holds J*rho' (#2682): grid rho' = synth / J.
    """
    vor_hat: Field
    div_hat: Field
    w_hat: Field
    theta_prime_hat: Field
    jrho_prime_hat: Field
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
    *,
    include_tracer_advection: bool = True,
    return_mass_flux: bool = False,
) -> SpectralNHState:
    """Compute slow tendencies for the spectral NH model.

    Horizontal operations use spectral transforms; vertical operations
    are done in grid space. Returns tendencies in the same pytree
    structure as state. The paired model disables legacy tracer advection
    and optionally obtains the exact pre-mask horizontal continuity fluxes.
    Independent physics mixing-ratio sources retain their existing mask.
    """
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
            state.jrho_prime_hat.data,
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
    rho_p = _vdtr_grid[..., 3] / J[..., None]   # prognostic is J*rho'
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
    # Continuity for the prognostic J*rho': minus the spectral divergence
    # of the J*rho*v flux, kept in spectral space (no /J round trip, #2682).
    # Vertical mass-flux divergence is handled by the acoustic step.
    flux_rho_div = im_over_a[:, None] * rho_u_oc2 - one_over_a * rho_v_dmu
    drho_p_hat = -flux_rho_div

    # Loop 187 — when ``n_tracers > 0`` the theta_div analysis is merged
    # with the tracer block's ``sh_analysis_3d(dq_grid_sum_flat)`` along
    # the trailing axis (one SH analysis instead of two).
    _theta_div_grid = theta_total * div
    n_lat_p2, n_lon_p2, nlev_p2 = _theta_div_grid.shape

    # --- 14. w tendency (slow part) ---
    # Slow w tendency is zero: vertical PGF, buoyancy, and w-divergence are
    # all handled by the acoustic substeps (forward-backward scheme).
    dw_hat = jnp.zeros_like(state.w_hat.data)

    # --- 15. Tracer advection ---
    if n_tracers > 0:
        if include_tracer_advection:
            # Fold the tracer dimension into the trailing level axis so all
            # SH transforms in the tracer block run ONCE on a thicker
            # ``(..., nlev*n_tracers)`` tensor instead of being
            # ``vmap``'d over the leading tracer axis (which materialises a
            # separate FFT/sum kernel per tracer).  Same passive-trailing-axis
            # exploit as Loop 137 for the spectral ocean tracer block.
            tracers_hat_data = state.tracers_hat.data  # (n_sh, nlev, n_tracers)
            n_sh_t = tracers_hat_data.shape[0]
            nlev_tr = tracers_hat_data.shape[-2]
            tracers_hat_flat = tracers_hat_data.reshape(
                n_sh_t, nlev_tr * n_tracers,
            )

            # 1) Single SH synthesis over (level × tracer) — one IRFFT instead
            # of n_tracers separate ones.
            q_grid_flat = sh_synthesis_3d(grid, tracers_hat_flat)  # (n_lat, n_lon, nlev*n_tr)

            # 2) Build the four flux fields (q*u_cos, q*v_cos, q*div,
            # vertical_advection(q)) directly on the folded tensor.  The
            # ``u_cos``/``v_cos``/``div`` factors broadcast across the
            # combined trailing axis via ``jnp.repeat`` (same as the existing
            # tracer-fold convention used elsewhere in the dycore).
            if n_tracers == 1:
                u_cos_b = u_cos
                v_cos_b = v_cos
                div_b = div
            else:
                u_cos_b = jnp.repeat(u_cos, n_tracers, axis=-1)
                v_cos_b = jnp.repeat(v_cos, n_tracers, axis=-1)
                div_b = jnp.repeat(div, n_tracers, axis=-1)

            q_u_cos_flat = q_grid_flat * u_cos_b
            q_v_cos_flat = q_grid_flat * v_cos_b

            # 3) Two batched SH analyses (oc2, dmu) — 2*n_tracers calls → 2.
            flux_q_div_flat = (
                im_over_a[:, None] * sh_analysis_oc2_3d(grid, q_u_cos_flat)
                - one_over_a * sh_analysis_dmu_3d(grid, q_v_cos_flat)
            )

            # 4) Vertical advection still hard-codes axis -1 as nlev, so
            # keep a vmap, but apply it to the folded tensor (one batched
            # kernel — JAX traces ``vertical_advection_height`` once).
            # We carry the (n_lat, n_lon, nlev, n_tracers) shape for the
            # vertical pass so the level axis remains at -1 during the
            # stencil.
            q_grid = q_grid_flat.reshape(
                q_grid_flat.shape[0], q_grid_flat.shape[1], nlev_tr, n_tracers,
            )
            vert_adv_q = jax.vmap(
                lambda qi: vertical_advection_height(qi, w, dz, dz_half, J),
                in_axes=-1, out_axes=-1,
            )(q_grid)  # (n_lat, n_lon, nlev, n_tracers)
            # Combine the grid-space contributions (q*div + vert_adv) BEFORE
            # the SH analysis, so a single sh_analysis_3d serves all tracers
            # and combinations.  Loops 94/137 linearity exploit.
            dq_grid_sum_flat = (
                q_grid_flat * div_b
                + vert_adv_q.reshape(q_grid_flat.shape)
            )
        else:
            n_sh_t, nlev_tr, _ = state.tracers_hat.data.shape
            flux_q_div_flat = jnp.zeros(
                (n_sh_t, nlev_tr * n_tracers), dtype=state.tracers_hat.data.dtype,
            )
            dq_grid_sum_flat = jnp.zeros(
                (grid.n_lat, grid.n_lon, nlev_tr * n_tracers),
                dtype=_theta_div_grid.dtype,
            )
        # Loop 187 — merge the theta_div analysis (section 13) into the
        # tracer analysis.  Trailing axis = ``nlev + nlev*n_tracers``.
        _all_an_input = jnp.concatenate(
            [_theta_div_grid, dq_grid_sum_flat], axis=-1,
        )  # (n_lat, n_lon, nlev*(1 + n_tracers))
        _all_an_hat = sh_analysis_3d(grid, _all_an_input)
        theta_div_hat = _all_an_hat[:, :nlev_p2]
        dq_hat_flat = -flux_q_div_flat + _all_an_hat[:, nlev_p2:]

        # Restore the (n_sh, nlev, n_tracers) layout that matches
        # ``state.tracers_hat.data``.
        dtracers_hat = dq_hat_flat.reshape(n_sh_t, nlev_tr, n_tracers)
    else:
        theta_div_hat = sh_analysis_3d(grid, _theta_div_grid)
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
        # Physics density tendency is in the prognostic's units, d(J*rho')/dt.
        drho_p_hat = drho_p_hat + physics_tendency.jrho_prime_hat.data
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

    result = SpectralNHState(
        vor_hat=state.vor_hat.replace(data=dvor_hat),
        div_hat=state.div_hat.replace(data=ddiv_hat),
        w_hat=state.w_hat.replace(data=dw_hat),
        theta_prime_hat=state.theta_prime_hat.replace(data=dtheta_p_hat),
        jrho_prime_hat=state.jrho_prime_hat.replace(data=drho_p_hat),
        phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
        tracers_hat=state.tracers_hat.replace(data=dtracers_hat),
    )
    if return_mass_flux:
        return result, (rho_u_cos, rho_v_cos)
    return result


# =============================================================================
# Acoustic substeps (grid space, purely vertical)
# =============================================================================

def _acoustic_substeps_grid(
    w_grid, theta_p_grid, rho_p_grid,
    dt_s, n_substeps,
    height_coord, terrain_metric, config,
    *, return_mass_flux=False,
):
    """Run acoustic substeps in grid space.

    Same vertical-only operations as compressible_euler.acoustic_substeps,
    but working on Gaussian grid arrays (n_lat, n_lon, ...) shapes.
    """
    g = config.g
    c_p = constants.c_pd
    dz = height_coord.dz
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    J = terrain_metric.jacobian  # (n_lat, n_lon)

    def substep_body(i, carry):
        w_c, theta_p_c, rho_p_c = carry[:3]

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

        if return_mass_flux:
            # Positive upward, indexed top to bottom: density uses
            # -(F_top - F_bottom)/(J*dz) at this exact backward update.
            return (w_new, theta_p_new, rho_p_new, carry[3] + dt_s * rho_w)
        return (w_new, theta_p_new, rho_p_new)

    initial = (w_grid, theta_p_grid, rho_p_grid)
    if return_mass_flux:
        initial = (*initial, jnp.zeros_like(w_grid))
    result = jax.lax.fori_loop(0, n_substeps, substep_body, initial)
    if return_mass_flux:
        return result[:3], result[3]
    return result


def _acoustic_substeps_grid_semi_implicit(
    w_grid, theta_p_grid, rho_p_grid,
    dt_s, n_substeps,
    height_coord, terrain_metric, config,
    *, return_mass_flux=False,
):
    """Semi-implicit acoustic substeps in grid space (tridiagonal w solve).

    Same structure as the explicit version but the vertical pressure
    gradient in the w equation is treated implicitly via a tridiagonal
    solve, removing the vertical acoustic CFL constraint.  The tridiagonal
    bands come from the shared :func:`precompute_si_tridiag_bands`
    (stretched-grid metric, both couplings on the boundary diagonal).  The
    substep body is not delegated to ``semi_implicit_acoustic_column_kernel``
    because this model's θ' update zeroes ``∂θ/∂z`` on the top and bottom
    levels where the shared kernel uses one-sided differences.
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
        w_c, theta_p_c, rho_p_c = carry[:3]

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

        if return_mass_flux:
            # Positive upward, indexed top to bottom: density uses
            # -(F_top - F_bottom)/(J*dz) at this exact backward update.
            return (w_new, theta_p_new, rho_p_new, carry[3] + dt_s * rho_w)
        return (w_new, theta_p_new, rho_p_new)

    initial = (w_grid, theta_p_grid, rho_p_grid)
    if return_mass_flux:
        initial = (*initial, jnp.zeros_like(w_grid))
    result = jax.lax.fori_loop(0, n_substeps, substep_body, initial)
    if return_mass_flux:
        return result[:3], result[3]
    return result


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
        # Geometry is fixed at construction; never inspect J inside a trace.
        self._flat_tracer_metric = bool(np.all(np.asarray(terrain_metric.jacobian) == 1.0))
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
        """Global dry mass ``∫ (J · rho_ref + J·rho') · dz · dA`` (fp64)."""
        jrho_p_grid = sh_synthesis_3d(self.grid, state.jrho_prime_hat.data)
        J = self.terrain_metric.jacobian                    # (n_lat, n_lon)
        jrho_total = J[..., None] * self.height_coord.rho_ref + jrho_p_grid
        dz = self.height_coord.dz                           # (nlev,)
        col_mass = jnp.sum(
            jrho_total * dz[None, None, :], axis=-1,
        )                                                   # (n_lat, n_lon)
        acc = jnp.float64
        return jnp.sum(
            col_mass.astype(acc) * self.grid.grid_area.astype(acc),
        )

    def _apply_mass_fixer(self, state, target_mass=None):
        """Anchor ``∫ J · (rho_ref + rho') · dz · dA`` to ``target_mass``
        (falls back to ``self._target_mass`` for the eager call sites).

        Uniform additive correction in physical space (matches the
        cubed-sphere / MPAS ``fix_mass_nonhydrostatic`` convention):
        ``Δρ = (target − current) / (∫ J · dz · dA)``.  Adding ``Δρ`` to
        ``rho'`` adds ``Δρ · J`` to the prognostic ``J·rho'``, i.e.
        ``Δρ · analysis(J)`` to every level of ``jrho_prime_hat``.  With
        J·rho' prognostic the dry mass already closes to round-off, so
        this opt-in fixer only removes round-off.
        """
        current_mass = self.compute_dry_mass(state)
        if target_mass is None:
            target_mass = self._target_mass
        J = self.terrain_metric.jacobian
        dz = self.height_coord.dz
        acc = jnp.float64
        area_acc = self.grid.grid_area.astype(acc)
        total_vol = jnp.sum(J.astype(acc) * area_acc) * jnp.sum(dz.astype(acc))
        delta_rho = (target_mass - current_mass) / total_vol
        jrho_hat = state.jrho_prime_hat.data
        j_hat = sh_analysis(self.grid, J.astype(acc))      # (n_sh,)
        jrho_hat_new = jrho_hat + (delta_rho * j_hat)[:, None].astype(
            jrho_hat.dtype)
        return state._replace(
            jrho_prime_hat=state.jrho_prime_hat.replace(data=jrho_hat_new),
        )

    def _build_se_functions(self, *, paired=False):
        """Build slow tendency and acoustic update functions for split-explicit."""
        def slow_tendency_fn(s):
            return spectral_nh_slow_tendencies(
                s, self.grid, self.height_coord,
                self.terrain_metric, self.config,
                include_tracer_advection=not paired,
                return_mass_flux=paired,
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
                [s.theta_prime_hat.data, s.jrho_prime_hat.data, s.w_hat.data],
                axis=-1,
            )  # (n_sh, 2*nlev + (nlev+1))
            theta_rho_w_grid = sh_synthesis_3d(self.grid, theta_rho_w_hat)
            jac = self.terrain_metric.jacobian[..., None]
            theta_p_grid = theta_rho_w_grid[..., :nlev_a]
            rho_p_grid = theta_rho_w_grid[..., nlev_a:2 * nlev_a] / jac
            w_grid = theta_rho_w_grid[..., 2 * nlev_a:]

            # Run acoustic substeps in grid space
            acoustic_fn = (
                _acoustic_substeps_grid_semi_implicit
                if self.config.semi_implicit_acoustic
                else _acoustic_substeps_grid
            )
            (w_new, theta_p_new, _), integrated_flux = acoustic_fn(
                w_grid, theta_p_grid, rho_p_grid,
                dt_s, n_sub,
                self.height_coord, self.terrain_metric, self.config,
                return_mass_flux=True,
            )
            # J*rho' increment from the stage-integrated vertical mass flux
            # (positive upward, top-to-bottom index): bottom minus top over
            # dz, no /J.  Same formula as the paired tracer-mass update.
            jrho_incr = (
                integrated_flux[..., 1:] - integrated_flux[..., :-1]
            ) / self.height_coord.dz

            # Convert back to spectral via the same concat trick — 3 SH
            # analyses → 1.  Slot order matches the synthesis so we can
            # slice the result back into (theta_hat, jrho_incr_hat, w_hat).
            theta_rho_w_new = jnp.concatenate(
                [theta_p_new, jrho_incr, w_new], axis=-1,
            )  # (n_lat, n_lon, 2*nlev + (nlev+1))
            theta_rho_w_new_hat = sh_analysis_3d(self.grid, theta_rho_w_new)
            theta_p_new_hat = theta_rho_w_new_hat[..., :nlev_a]
            rho_p_new_hat = (s.jrho_prime_hat.data
                             + theta_rho_w_new_hat[..., nlev_a:2 * nlev_a])
            w_new_hat = theta_rho_w_new_hat[..., 2 * nlev_a:]
            result = SpectralNHState(
                vor_hat=s.vor_hat,
                div_hat=s.div_hat,
                w_hat=s.w_hat.replace(data=w_new_hat),
                theta_prime_hat=s.theta_prime_hat.replace(
                    data=theta_p_new_hat,
                ),
                jrho_prime_hat=s.jrho_prime_hat.replace(
                    data=rho_p_new_hat,
                ),
                phis_hat=s.phis_hat,
                tracers_hat=s.tracers_hat,
            )
            if paired:
                return result, integrated_flux
            return result

        return slow_tendency_fn, acoustic_update_fn

    def _split_step(self, state, dt, se_config):
        """Pair spectral tracer mass with RK density on flat and nonflat terrain.

        Nonempty tracers use paired transport; empty axes keep the legacy
        integrator. The construction-time exact J==1 check selects recovery.
        Paired mass coefficients represent P(J*rho*q), with the SAME full native
        projection P as acoustic density. Flat recovery remains
        q_hat=A(PM/(J*rho)); terrain uses q_hat=A(PM/P(J*rho)).
        KNOWN DIFFERENCE: the final projection of q generally loses local
        and pointwise mass equality. Flat-J global inventory instead relies
        on native discrete orthogonality and represented rho. With J*rho'
        prognostic (#2682) terrain recovery preserves a uniform tracer.
        Source-free/fixer-free tests require the global invariant;
        anchored fixer ordering is unchanged.
        Physics sources are re-projected, with the mask applied to the mass
        source: source-only q+dt*S holds only up to that projection. The
        (n=0,m=0) mode and global source inventory are unaffected.
        """
        paired = (state.tracers_hat.data.ndim >= 3
                  and state.tracers_hat.data.shape[-1] > 0)
        slow_fn, acoustic_fn = self._build_se_functions(paired=paired)
        if not paired:
            return split_explicit_step(state, slow_fn, acoustic_fn, dt, se_config)

        grid = self.grid
        hc = self.height_coord
        jac = self.terrain_metric.jacobian[..., None]
        mask = dealiasing_mask(grid, self.config.dealiasing_fraction)[:, None, None]

        def analysis(q):
            nlat, nlon, nz, nt = q.shape
            return sh_analysis_3d(grid, q.reshape(nlat, nlon, nz * nt)).reshape(
                grid.n_sh, nz, nt)

        def synthesis(q_hat):
            nsh, nz, nt = q_hat.shape
            return sh_synthesis_3d(grid, q_hat.reshape(nsh, nz * nt)).reshape(
                grid.n_lat, grid.n_lon, nz, nt)

        def jrho(s):
            # Grid J*rho (total); the prognostic already carries J*rho'.
            return jac * hc.rho_ref + sh_synthesis_3d(grid, s.jrho_prime_hat.data)

        def tracer_mass(q_hat, jr):
            return analysis(jr[..., None] * synthesis(q_hat))

        def recover(mass_hat, jr):
            """Recover q using the geometry's statically selected denominator.

            Terrain: q = A(S(M) / S(A(J*rho))).  Mass M and J*rho' receive
            identical spectral increments (slow flux divergence, acoustic
            integrated flux, RK blends), so M = c * P(J*rho) for a uniform
            tracer c and the ratio stays c to round-off (#2682).
            """
            if self._flat_tracer_metric:
                return analysis(synthesis(mass_hat) / jr[..., None])
            return analysis(synthesis(mass_hat) / synthesis(analysis(jr[..., None])))

        def reconstruct(q_hat):
            # No spectral vertical selector exists: keep native centred faces.
            return tracer_vertical_face_value(synthesis(q_hat), None, "centered")

        def slow(s):
            tend, (fu_cos, fv_cos) = slow_fn(s)
            q = synthesis(s.tracers_hat.data)
            nlat, nlon, nz, nt = q.shape
            # Exported fluxes contain J, density and cos(latitude). Reuse
            # density continuity's exact operator; transport is minus divergence.
            q_u_cos = (fu_cos[..., None] * q).reshape(nlat, nlon, nz * nt)
            q_v_cos = (fv_cos[..., None] * q).reshape(nlat, nlon, nz * nt)
            im_over_a = 1j * grid.ms.astype(jnp.float64) / grid.radius
            one_over_a = 1.0 / grid.radius
            flux_q_div = (
                im_over_a[:, None] * sh_analysis_oc2_3d(grid, q_u_cos)
                - one_over_a * sh_analysis_dmu_3d(grid, q_v_cos)
            )
            horizontal = -flux_q_div.reshape(grid.n_sh, nz, nt) * mask
            # Physics sources are q/s; convert with ACTUAL evaluation density.
            source = tracer_mass(tend.tracers_hat.data, jrho(s))
            return tend, horizontal + source * mask

        def flux_tendency(integrated_flux, faces):
            # T29's centered positive/negative reconstructions are identical,
            # so selecting between them by flux sign reduces to this product.
            flux = integrated_flux[..., None] * faces
            # Top-to-bottom indexing, flux positive upward: bottom minus top.
            # No /J: the carried mass already includes J.
            return analysis((flux[..., 1:, :] - flux[..., :-1, :]) / hc.dz[:, None])

        def set_q(s, q_hat):
            return s._replace(tracers_hat=s.tracers_hat.replace(data=q_hat))

        def zero_q(s):
            return set_q(s, jnp.zeros_like(s.tracers_hat.data))

        def physical(carry):
            s, mass_hat = carry
            return set_q(s, recover(mass_hat, jrho(s)))

        def slow_carry(carry):
            s = physical(carry)
            tend, mass_rhs = slow(s)
            # As in T29, the RHS occupies the transient tracer slot; the
            # independent mass leaf is advanced only by finish_stage below.
            return set_q(tend, mass_rhs), jnp.zeros_like(carry[1])

        def acoustic_carry(carry, slow_tend, dt_s, n_sub, cfg):
            s, mass_hat = carry
            tend = slow_tend[0]
            stage_dt = dt_s * n_sub
            # Preserve T29's base-density recovery and operation order.
            jrho_base = jrho(s) - stage_dt * sh_synthesis_3d(
                grid, tend.jrho_prime_hat.data)
            s = set_q(s, recover(mass_hat, jrho_base))
            updated, integrated_flux = acoustic_fn(s, tend, dt_s, n_sub, cfg)
            return (zero_q(updated), mass_hat), integrated_flux

        def finish_stage(base, eval_state, state_slow, acoustic_out, slow_tend, dt_eff):
            (updated, _), integrated_flux = acoustic_out
            # Recompute the same evaluation-state faces instead of carrying
            # T29's transient reconstruction/norm leaves through the RK tree.
            faces = reconstruct(physical(eval_state).tracers_hat.data)
            mass_hat = (base[1] + dt_eff * slow_tend[0].tracers_hat.data
                        + flux_tendency(integrated_flux, faces))
            return zero_q(updated), mass_hat

        hooks = TracerMassHooks(
            density=lambda c: jrho(c[0]) / jac,
            tracers=lambda c: c[0].tracers_hat.data,
            with_tracers=lambda c, q: (set_q(c[0], q), c[1]),
            finish_stage=finish_stage,
            mass=lambda c: c[1],
            with_mass=lambda c, m: (zero_q(c[0]), m),
        )
        mass_hat = tracer_mass(state.tracers_hat.data, jrho(state))
        carry = split_explicit_step(
            (zero_q(state), mass_hat), slow_carry, acoustic_carry, dt, se_config,
            tracer_mass=hooks,
        )
        return physical(carry)

    def step(self, state: SpectralNHState, dt: float) -> SpectralNHState:
        """Outer wrapper: snapshots dry mass on first call when
        ``anchor_mass_to_initial`` is on (fp64, outside JIT)."""
        if (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and self._target_mass is None):
            self._target_mass = self.compute_dry_mass(state)
        return self._step_jit(state, dt)

    @partial(jax.jit, static_argnums=(0,))
    def _step_jit(self, state: SpectralNHState, dt: float) -> SpectralNHState:
        """Advance one time step using split-explicit RK3.

        Slow tendencies use spectral horizontal operators.
        Acoustic substeps run in grid space (purely vertical).
        """
        se_config = SplitExplicitConfig(
            n_substeps=self.config.n_acoustic_substeps,
        )
        if self._use_cpu_for_spectral:
            state_cpu = jax.device_put(state, self._cpu_device)
            result_cpu = self._split_step(state_cpu, dt, se_config)
            state_new = jax.device_put(result_cpu, self._default_device)
        else:
            state_new = self._split_step(state, dt, se_config)

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
        state_new = self._split_step(state, dt, se_config)
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

def dcmip25_tc1_init_spectral(
    grid: GaussianGrid,
    n_levels: int = 88,
    params: dict | None = None,
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
        jrho_prime_hat=Field(
            data=rho_p_hat, name="jrho_prime_hat", dims=dims_3d, units="kg/m^3",
        ),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
        tracers_hat=Field(
            data=tracers_hat, name="tracers_hat", dims=dims_tr, units="kg/kg",
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
        jrho_prime_hat=Field(
            data=rho_p_hat, name="jrho_prime_hat", dims=dims_3d, units="kg/m^3",
        ),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
        tracers_hat=Field(
            data=tracers_hat, name="tracers_hat", dims=dims_tr, units="kg/kg",
        ),
    )


def dcmip25_tc2_init_spectral(
    grid: GaussianGrid,
    n_levels: int = 48,
    subcase: str = "a",
    params: dict | None = None,
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
        jrho_prime_hat=Field(data=rho_p_hat, name="jrho_prime_hat", dims=dims_3d, units="kg/m^3"),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
        tracers_hat=Field(data=tracers_hat, name="tracers_hat", dims=dims_tr, units="kg/kg"),
    ), height_coord, terrain_metric


def dcmip25_tc3_init_spectral(
    grid: GaussianGrid,
    n_levels: int = 40,
    params: dict | None = None,
) -> tuple['SpectralNHState', HeightCoordinate, 'TerrainMetric']:
    """Initialize DCMIP-2025 TC3 (squall line) on Gaussian grid.

    Wind shear, moisture, and warm bubbles on a small Earth (radius/60).
    Transforms to spectral space.

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
    rho_p_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)

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

    theta_p_hat = sh_analysis_3d(small_grid, theta_pert)
    tracers_hat = jnp.stack([
        sh_analysis_3d(small_grid, q_v_grid),
        sh_analysis_3d(small_grid, q_c_grid),
        sh_analysis_3d(small_grid, q_r_grid),
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
        jrho_prime_hat=Field(data=rho_p_hat, name="jrho_prime_hat", dims=dims_3d, units="kg/m^3"),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
        tracers_hat=Field(data=tracers_hat, name="tracers_hat", dims=dims_tr, units="kg/kg"),
    ), height_coord, terrain_metric
