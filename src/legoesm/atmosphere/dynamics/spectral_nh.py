"""Spectral Non-Hydrostatic Compressible Euler Model.

Solves the fully compressible Euler equations on the sphere using
pseudospectral (SH transform) horizontal operators and height-based
terrain-following (z*) coordinates with reference-state subtraction.

Prognostic variables (spectral):
    vor_hat         : Spectral relative vorticity, (n_sh, nlev)
    div_hat         : Spectral divergence, (n_sh, nlev)
    w_hat           : Spectral vertical velocity, (n_sh, nlev+1)
    theta_prime_hat : Spectral pot. temp. perturbation, (n_sh, nlev)
    rho_prime_hat   : Spectral density perturbation, (n_sh, nlev)
    tracers_hat     : Spectral tracers, (n_sh, nlev, n_tracers)

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
from legoesm.core.operators_3d import (
    vertical_advection_height,
)
from legoesm.grids.gaussian import (
    GaussianGrid,
    sh_analysis,
    sh_synthesis,
    sh_analysis_3d,
    sh_synthesis_3d,
    sh_analysis_oc2_3d,
    sh_analysis_dmu_3d,
    uv_from_vordiv_3d,
    spectral_hyperdiffusion_3d,
    _sh_synthesis_H,
)
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.atmosphere.dynamics.compressible_euler import (
    compute_exner_perturbation,
    _sponge_profile,
)
from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
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
    Tracers:                (n_sh, nlev, n_tracers) complex128
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


# =============================================================================
# Spectral gradient helper
# =============================================================================

def _spectral_gradient_3d(grid, coeffs_3d):
    """Compute horizontal gradient of a 3D scalar from spectral coefficients.

    Returns (dfdx, dfdy) on the Gaussian grid where:
    dfdx = (1/(a*cos(lat))) * d(f)/d(lambda)
    dfdy = (1/a) * d(f)/d(lat)

    Parameters
    ----------
    coeffs_3d : (n_sh, nlev) complex

    Returns
    -------
    dfdx, dfdy : (n_lat, n_lon, nlev)
    """
    a = grid.radius

    # Zonal derivative: im * coeffs -> synthesis (per level)
    # sh_synthesis of (im * coeffs) gives d(f)/d(lambda) on the grid
    c_t = jnp.moveaxis(coeffs_3d, -1, 0)  # (nlev, n_sh)
    ims = grid.ms.astype(jnp.float64)

    cos_lat_2d = jnp.clip(grid.cos_lat[:, None], _COS_LAT_MIN, None)

    def zonal_deriv(c):
        return sh_synthesis(grid, 1j * ims * c) / (a * cos_lat_2d)

    dfdx_t = jax.vmap(zonal_deriv)(c_t)  # (nlev, n_lat, n_lon)
    dfdx = jnp.moveaxis(dfdx_t, 0, -1)

    # Meridional derivative: Hnm synthesis (per level)
    # _sh_synthesis_H gives cos(lat) * d(f)/d(colatitude)
    # df/dy = -(1/a) * d(f)/d(colatitude) = -Hnm_synth / (a * cos(lat))
    def merid_deriv(c):
        return -_sh_synthesis_H(grid, c) / (a * cos_lat_2d)

    dfdy_t = jax.vmap(merid_deriv)(c_t)
    dfdy = jnp.moveaxis(dfdy_t, 0, -1)

    return dfdx, dfdy


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
    structure as state.
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
    vor = sh_synthesis_3d(grid, state.vor_hat.data)   # (n_lat, n_lon, nlev)
    div = sh_synthesis_3d(grid, state.div_hat.data)
    w = sh_synthesis_3d(grid, state.w_hat.data)        # (n_lat, n_lon, nlev+1)
    theta_p = sh_synthesis_3d(grid, state.theta_prime_hat.data)
    rho_p = sh_synthesis_3d(grid, state.rho_prime_hat.data)

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
    dpi_dx, dpi_dy = _spectral_gradient_3d(grid, pi_p_hat)

    # PGF vectors
    pgf_x = c_p * theta_total * dpi_dx
    pgf_y = c_p * theta_total * dpi_dy

    # --- 6. Kinetic energy ---
    K = 0.5 * (u**2 + v**2)

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

    # --- 9. Vorticity tendency ---
    # dvor/dt = -div(abs_vor * v) - curl(PGF)
    flux_vor_div = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, A_vor)
        - one_over_a * sh_analysis_dmu_3d(grid, B_vor)
    )
    pgf_curl = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, pgf_v_cos)
        + one_over_a * sh_analysis_dmu_3d(grid, pgf_u_cos)
    )
    dvor_hat = -flux_vor_div - pgf_curl

    # --- 10. Divergence tendency ---
    # ddiv/dt = curl(abs_vor * v) - lap(K) - div(PGF)
    flux_vor_curl = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, B_vor)
        + one_over_a * sh_analysis_dmu_3d(grid, A_vor)
    )
    K_hat = sh_analysis_3d(grid, K)
    pgf_div = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, pgf_u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, pgf_v_cos)
    )
    ddiv_hat = flux_vor_curl - grid.lap[:, None] * K_hat - pgf_div

    # --- 11. Vertical advection (grid space) ---
    # Vertical advection of momentum
    vert_adv_u = vertical_advection_height(u, w, dz, dz_half, J)
    vert_adv_v = vertical_advection_height(v, w, dz, dz_half, J)

    # Convert to spectral vor/div contributions
    vu_cos = vert_adv_u * grid.cos_lat[:, None, None]
    vv_cos = vert_adv_v * grid.cos_lat[:, None, None]

    vert_vor = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, vv_cos)
        + one_over_a * sh_analysis_dmu_3d(grid, vu_cos)
    )
    vert_div = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, vu_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, vv_cos)
    )
    dvor_hat = dvor_hat + vert_vor
    ddiv_hat = ddiv_hat + vert_div

    # --- 12. Theta equation ---
    # Horizontal advection via spectral flux form: -div(theta*v) + theta*div
    theta_u_cos = theta_total * u_cos
    theta_v_cos = theta_total * v_cos

    flux_theta_div = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, theta_u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, theta_v_cos)
    )
    theta_div_hat = sh_analysis_3d(grid, theta_total * div)

    dtheta_p_hat = -flux_theta_div + theta_div_hat

    # NOTE: Vertical advection of theta by w is handled ONLY by the
    # acoustic substeps (forward-backward scheme) to avoid double counting
    # in the split-explicit time integration (Skamarock & Klemp 2008).

    # --- 13. Continuity equation (rho') ---
    # Horizontal only: vertical mass flux divergence handled by acoustic step.
    rho_u_cos = rho_total * u_cos * J[..., None]
    rho_v_cos = rho_total * v_cos * J[..., None]

    flux_rho_div = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, rho_u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, rho_v_cos)
    )
    # d(rho')/dt_horiz = -(1/J)*div_h(J*rho*v)
    rho_horiz_tend_grid = -sh_synthesis_3d(grid, flux_rho_div) / J[..., None]
    drho_p_hat = sh_analysis_3d(grid, rho_horiz_tend_grid)

    # --- 14. w tendency (slow part) ---
    # Slow w tendency is zero: vertical PGF, buoyancy, and w-divergence are
    # all handled by the acoustic substeps (forward-backward scheme).
    dw_hat = jnp.zeros_like(state.w_hat.data)

    # --- 15. Tracer advection ---
    if n_tracers > 0:
        tracers_hat_t = jnp.moveaxis(state.tracers_hat.data, -1, 0)  # (n_tracers, n_sh, nlev)

        def _single_tracer_tendency(q_hat):
            q = sh_synthesis_3d(grid, q_hat)  # (n_lat, n_lon, nlev)

            # Horizontal advection: -div(q*v) + q*div
            q_u_cos = q * u_cos
            q_v_cos = q * v_cos
            flux_q_div = (
                im_over_a[:, None] * sh_analysis_oc2_3d(grid, q_u_cos)
                - one_over_a * sh_analysis_dmu_3d(grid, q_v_cos)
            )
            q_div_hat = sh_analysis_3d(grid, q * div)
            dq_hat = -flux_q_div + q_div_hat

            # Vertical advection
            vert_adv_q = vertical_advection_height(q, w, dz, dz_half, J)
            dq_hat = dq_hat + sh_analysis_3d(grid, vert_adv_q)
            return dq_hat

        dtracers_hat_t = jax.vmap(_single_tracer_tendency)(tracers_hat_t)
        dtracers_hat = jnp.moveaxis(dtracers_hat_t, 0, -1)
    else:
        dtracers_hat = jnp.zeros_like(state.tracers_hat.data)

    # --- 16. Sponge layer damping ---
    if config.sponge_coeff > 0:
        sponge = _sponge_profile(
            height_coord.z_full, H, config.sponge_width, config.sponge_coeff,
        )  # (nlev,)
        # Damp vorticity, divergence, theta toward reference
        dvor_hat = dvor_hat - sponge * state.vor_hat.data
        ddiv_hat = ddiv_hat - sponge * state.div_hat.data
        dtheta_p_hat = dtheta_p_hat - sponge * state.theta_prime_hat.data

    # --- 17. Spectral hyperdiffusion ---
    if config.hyperdiff_coeff > 0:
        dvor_hat = dvor_hat + spectral_hyperdiffusion_3d(
            grid, state.vor_hat.data, config.hyperdiff_coeff, config.hyperdiff_order,
        )
        ddiv_hat = ddiv_hat + spectral_hyperdiffusion_3d(
            grid, state.div_hat.data, config.hyperdiff_coeff, config.hyperdiff_order,
        )
        dtheta_p_hat = dtheta_p_hat + spectral_hyperdiffusion_3d(
            grid, state.theta_prime_hat.data,
            config.hyperdiff_coeff, config.hyperdiff_order,
        )

    # --- 18. Physics tendencies ---
    if physics_tendency is not None:
        dvor_hat = dvor_hat + physics_tendency.vor_hat.data
        ddiv_hat = ddiv_hat + physics_tendency.div_hat.data
        dtheta_p_hat = dtheta_p_hat + physics_tendency.theta_prime_hat.data
        drho_p_hat = drho_p_hat + physics_tendency.rho_prime_hat.data
        dtracers_hat = dtracers_hat + physics_tendency.tracers_hat.data

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
    """
    g = config.g
    c_p = constants.c_pd
    dz = height_coord.dz
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    J = terrain_metric.jacobian  # (n_lat, n_lon)

    def substep_body(i, carry):
        w_c, theta_p_c, rho_p_c = carry

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
        rho_w = jnp.zeros_like(w_new)
        rho_w = rho_w.at[..., 1:-1].set(rho_half * w_new[..., 1:-1])
        vert_div = (rho_w[..., :-1] - rho_w[..., 1:]) / dz
        vert_div = vert_div / J[..., None]
        rho_p_new = rho_p_c - dt_s * vert_div

        # --- Backward: update theta' ---
        w_full = 0.5 * (w_new[..., :-1] + w_new[..., 1:])
        dtheta_dz = jnp.zeros_like(theta_total)
        nlev_local = theta_total.shape[-1]
        if nlev_local > 2:
            dz_half_val = height_coord.dz_half
            dz_centered = dz_half_val[:-1] + dz_half_val[1:]
            inner_grad = (
                theta_total[..., :-2] - theta_total[..., 2:]
            ) / dz_centered
            dtheta_dz = dtheta_dz.at[..., 1:-1].set(inner_grad)
        theta_p_new = theta_p_c - dt_s * w_full / J[..., None] * dtheta_dz

        return (w_new, theta_p_new, rho_p_new)

    return jax.lax.fori_loop(
        0, n_substeps, substep_body, (w_grid, theta_p_grid, rho_p_grid),
    )


def _acoustic_substeps_grid_semi_implicit(
    w_grid, theta_p_grid, rho_p_grid,
    dt_s, n_substeps,
    height_coord, terrain_metric, config,
):
    """Semi-implicit acoustic substeps in grid space (tridiagonal w solve).

    Same structure as the explicit version but the vertical pressure
    gradient in the w equation is treated implicitly via a tridiagonal
    solve, removing the vertical acoustic CFL constraint.
    """
    from legoesm.timestepping.tridiagonal import thomas_solve_batched

    g = config.g
    c_p = constants.c_pd
    R_d = constants.R_d
    c_v = constants.c_vd
    dz = height_coord.dz
    dz_half = height_coord.dz_half
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    J = terrain_metric.jacobian

    # Linearized sound speed squared
    gamma = c_p / c_v
    T_ref = theta_0 * height_coord.exner_ref
    cs2 = gamma * R_d * T_ref
    cs2_half = 0.5 * (cs2[:-1] + cs2[1:])
    dz_inner = 0.5 * (dz[:-1] + dz[1:])
    nlev = theta_p_grid.shape[-1]

    def substep_body(i, carry):
        w_c, theta_p_c, rho_p_c = carry

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

        # --- Tridiagonal coefficients ---
        alpha = dt_s**2 * cs2_half / (dz_inner * J[..., None])**2

        a_tri = jnp.zeros_like(alpha)
        a_tri = a_tri.at[..., 1:].set(-alpha[..., 1:])

        b_tri = 1.0 + 2.0 * alpha
        b_tri = b_tri.at[..., 0].set(1.0 + alpha[..., 0])
        b_tri = b_tri.at[..., -1].set(1.0 + alpha[..., -1])

        c_tri = jnp.zeros_like(alpha)
        c_tri = c_tri.at[..., :-1].set(-alpha[..., :-1])

        w_inner_new = thomas_solve_batched(a_tri, b_tri, c_tri, rhs)
        w_new = w_c.at[..., 1:-1].set(w_inner_new)

        # --- Backward: update rho' ---
        rho_half = 0.5 * (rho_total[..., :-1] + rho_total[..., 1:])
        rho_w = jnp.zeros_like(w_new)
        rho_w = rho_w.at[..., 1:-1].set(rho_half * w_new[..., 1:-1])
        vert_div = (rho_w[..., :-1] - rho_w[..., 1:]) / dz
        vert_div = vert_div / J[..., None]
        rho_p_new = rho_p_c - dt_s * vert_div

        # --- Backward: update theta' ---
        w_full = 0.5 * (w_new[..., :-1] + w_new[..., 1:])
        dtheta_dz = jnp.zeros_like(theta_total)
        if nlev > 2:
            dz_centered = dz_half[:-1] + dz_half[1:]
            inner_grad = (theta_total[..., :-2] - theta_total[..., 2:]) / dz_centered
            dtheta_dz = dtheta_dz.at[..., 1:-1].set(inner_grad)
        theta_p_new = theta_p_c - dt_s * w_full / J[..., None] * dtheta_dz

        return (w_new, theta_p_new, rho_p_new)

    return jax.lax.fori_loop(
        0, n_substeps, substep_body, (w_grid, theta_p_grid, rho_p_grid),
    )


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
            from legoesm import constants
            factor = self.config.small_earth_factor
            grid = grid._replace(
                radius=constants.R_earth / factor,
                f=grid.f * factor,
            )
        self._use_cpu_for_spectral = False
        self._cpu_device = None
        self._default_device = None

        if legoesm_config is not None:
            allow_unsupported_backend = bool(
                legoesm_config.get(
                    "atmosphere.spectral.allow_unsupported", False
                )
            )

        from legoesm.core.hardware import get_backend
        backend = get_backend()
        if backend == "METAL":
            self._use_cpu_for_spectral = True
            self._cpu_device = jax.devices("cpu")[0]
            self._default_device = jax.devices()[0]
            self.grid = jax.device_put(grid, self._cpu_device)
        else:
            self.grid = grid
            from legoesm.core.hardware import check_spectral_backend
            check_spectral_backend(
                allow_unsupported=allow_unsupported_backend,
            )

    def _build_se_functions(self):
        """Build slow tendency and acoustic update functions for split-explicit."""
        def slow_tendency_fn(s):
            return spectral_nh_slow_tendencies(
                s, self.grid, self.height_coord,
                self.terrain_metric, self.config,
            )

        def acoustic_update_fn(s, slow_tend, dt_s, n_sub, cfg):
            # Convert acoustic variables from spectral to grid
            w_grid = sh_synthesis_3d(self.grid, s.w_hat.data)
            theta_p_grid = sh_synthesis_3d(self.grid, s.theta_prime_hat.data)
            rho_p_grid = sh_synthesis_3d(self.grid, s.rho_prime_hat.data)

            # Run acoustic substeps in grid space
            acoustic_fn = (
                _acoustic_substeps_grid_semi_implicit
                if self.config.semi_implicit_acoustic
                else _acoustic_substeps_grid
            )
            w_new, theta_p_new, rho_p_new = acoustic_fn(
                w_grid, theta_p_grid, rho_p_grid,
                dt_s, n_sub,
                self.height_coord, self.terrain_metric, self.config,
            )

            # Convert back to spectral
            return SpectralNHState(
                vor_hat=s.vor_hat,
                div_hat=s.div_hat,
                w_hat=s.w_hat.replace(data=sh_analysis_3d(self.grid, w_new)),
                theta_prime_hat=s.theta_prime_hat.replace(
                    data=sh_analysis_3d(self.grid, theta_p_new),
                ),
                rho_prime_hat=s.rho_prime_hat.replace(
                    data=sh_analysis_3d(self.grid, rho_p_new),
                ),
                phis_hat=s.phis_hat,
                tracers_hat=s.tracers_hat,
            )

        return slow_tendency_fn, acoustic_update_fn

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: SpectralNHState, dt: float) -> SpectralNHState:
        """Advance one time step using split-explicit RK3.

        Slow tendencies use spectral horizontal operators.
        Acoustic substeps run in grid space (purely vertical).
        """
        se_config = SplitExplicitConfig(
            n_substeps=self.config.n_acoustic_substeps,
        )
        slow_tendency_fn, acoustic_update_fn = self._build_se_functions()

        if self._use_cpu_for_spectral:
            state_cpu = jax.device_put(state, self._cpu_device)
            result_cpu = split_explicit_step(
                state_cpu, slow_tendency_fn, acoustic_update_fn,
                dt, se_config,
            )
            return jax.device_put(result_cpu, self._default_device)

        return split_explicit_step(
            state, slow_tendency_fn, acoustic_update_fn, dt, se_config,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _step_on_cpu(self, state: SpectralNHState, dt: float) -> SpectralNHState:
        """Step without device transfers (for batched CPU integration on Metal)."""
        se_config = SplitExplicitConfig(
            n_substeps=self.config.n_acoustic_substeps,
        )
        slow_tendency_fn, acoustic_update_fn = self._build_se_functions()
        return split_explicit_step(
            state, slow_tendency_fn, acoustic_update_fn, dt, se_config,
        )

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

        for i in range(n_steps):
            state_cpu = self._step_on_cpu(state_cpu, dt)
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
    n_levels: int = 40,
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
    from tests.test_cases.dcmip2025.common import (
        piecewise_lapse_theta_ref,
    )
    from tests.test_cases.dcmip2025.test_case_1 import TC1_PARAMS
    from legoesm.grids.vertical import (
        create_height_coordinate,
        compute_terrain_metric,
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
        rho_prime_hat=Field(
            data=rho_p_hat, name="rho_prime_hat", dims=dims_3d, units="kg/m^3",
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
        rho_prime_hat=Field(
            data=rho_p_hat, name="rho_prime_hat", dims=dims_3d, units="kg/m^3",
        ),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
        tracers_hat=Field(
            data=tracers_hat, name="tracers_hat", dims=dims_tr, units="kg/kg",
        ),
    )
