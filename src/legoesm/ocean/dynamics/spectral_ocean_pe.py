"""Spectral ocean primitive equations on the Gaussian grid.

Uses the pseudospectral (SH transform) method in the
vorticity-divergence formulation, following spectral_pe.py.

Prognostic variables (spectral space):
    vor_hat  : Vorticity SH coefficients, (n_sh, nlev)
    div_hat  : Divergence SH coefficients, (n_sh, nlev)
    T_hat    : Temperature SH coefficients, (n_sh, nlev)
    S_hat    : Salinity SH coefficients, (n_sh, nlev)
    eta_hat  : Sea surface height SH coefficients, (n_sh,)

Land masking: applied in grid space before every SH analysis.
Gibbs oscillations near coastlines controlled by spectral hyperdiffusion.

Equations (Boussinesq hydrostatic, vorticity-divergence form):
    d(vor)/dt  = -div((vor+f)*v) + curl(vert_adv + mixing)
    d(div)/dt  = curl((vor+f)*v) - lap(K + p'/rho_0) + div(vert_adv + mixing)
    d(T)/dt    = -div(T*v) + T*div(v) - w*dT/dz + K_h*lap(T) + vert_diff(T)
    d(S)/dt    = -div(S*v) + S*div(v) - w*dS/dz + K_h*lap(S) + vert_diff(S)
    d(eta)/dt  = -sum_k(div * h_k)
"""

from __future__ import annotations

from functools import partial
import warnings

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
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
)
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm.ocean.eos import wright_eos, compute_hydrostatic_pressure
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
    upwind_vertical_gradient,
)
from legoesm.ocean.state import SpectralOceanState, SpectralOceanConfig
from legoesm.ocean.physics.mixing import vertical_diffusion


def spectral_ocean_tendencies(
    state: SpectralOceanState,
    grid: GaussianGrid,
    z_coord: OceanZStarCoordinate,
    config: SpectralOceanConfig,
) -> SpectralOceanState:
    """Compute spectral tendencies for the ocean PE.

    Pseudospectral workflow:
    1. Transform prognostic fields to grid space
    2. Apply land mask
    3. Compute nonlinear products (EOS, advection, pressure)
    4. Mask products in grid space
    5. SH analysis -> spectral tendencies
    6. Spectral hyperdiffusion

    Returns tendencies as same pytree structure (for SSP-RK3).
    """
    a = grid.radius
    g = config.g
    rho_0 = config.rho_0
    mask = state.land_mask_grid.data  # (n_lat, n_lon)
    mask_3d = mask[..., jnp.newaxis]  # (n_lat, n_lon, 1)

    # --- 1. Transform to grid space ---
    vor = sh_synthesis_3d(grid, state.vor_hat.data) * mask_3d   # (n_lat, n_lon, nlev)
    div = sh_synthesis_3d(grid, state.div_hat.data) * mask_3d
    T = sh_synthesis_3d(grid, state.T_hat.data) * mask_3d
    S = sh_synthesis_3d(grid, state.S_hat.data) * mask_3d
    eta = sh_synthesis(grid, state.eta_hat.data) * mask          # (n_lat, n_lon)
    H_bathy = sh_synthesis(grid, state.H_bathy_hat.data).real
    H_bathy = jnp.maximum(H_bathy, 1.0) * mask + 1.0 * (1.0 - mask)

    # --- 2. Velocities ---
    u_cos, v_cos = uv_from_vordiv_3d(
        grid, state.vor_hat.data, state.div_hat.data,
    )
    cos_lat_3d = grid.cos_lat[:, jnp.newaxis, jnp.newaxis]
    u = u_cos / cos_lat_3d * mask_3d
    v = v_cos / cos_lat_3d * mask_3d

    # --- 3. Layer thickness and Jacobian ---
    J = compute_ocean_jacobian(eta.real, H_bathy.real, z_coord)
    h_k = compute_layer_thickness(eta.real, H_bathy.real, z_coord)

    # --- 4. EOS and hydrostatic pressure ---
    T_real = T.real
    S_real = S.real
    p_hydro = compute_hydrostatic_pressure(
        jnp.full_like(T_real, rho_0),
        eta.real,
        z_coord.dz_ref,
        J.real,
        rho_0,
        g,
    )
    rho = wright_eos(T_real, S_real, p_hydro)
    rho_prime = rho - rho_0

    # Baroclinic pressure perturbation (top-down cumsum)
    dp_layer = rho_prime * g * h_k.real
    p_prime = jnp.cumsum(dp_layer, axis=-1) - dp_layer
    p_prime = p_prime + 0.5 * dp_layer  # at cell center

    # --- 5. Kinetic energy ---
    K = 0.5 * (u.real**2 + v.real**2)

    # --- 6. Absolute vorticity ---
    abs_vor = vor + grid.f[..., jnp.newaxis]

    # --- 7. Diagnose w ---
    div_h = div.real * h_k.real
    div_h_rev = div_h[..., ::-1]
    cumsum_rev = jnp.cumsum(div_h_rev, axis=-1)
    w_inner = -cumsum_rev[..., ::-1]
    zeros_bottom = jnp.zeros((*div.shape[:-1], 1))
    w = jnp.concatenate([w_inner, zeros_bottom], axis=-1)  # (n_lat, n_lon, nlev+1)

    # --- 8. Spectral operators ---
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a

    # --- 9. Vorticity fluxes ---
    A_vor = abs_vor * u_cos * mask_3d
    B_vor = abs_vor * v_cos * mask_3d

    flux_vor_div = (
        im_over_a[:, jnp.newaxis] * sh_analysis_oc2_3d(grid, A_vor)
        - one_over_a * sh_analysis_dmu_3d(grid, B_vor)
    )
    flux_vor_curl = (
        im_over_a[:, jnp.newaxis] * sh_analysis_oc2_3d(grid, B_vor)
        + one_over_a * sh_analysis_dmu_3d(grid, A_vor)
    )

    # --- 10. Energy variable: E = K + p'/rho_0 ---
    E = K + p_prime / rho_0
    E_hat = sh_analysis_3d(grid, E * mask_3d)

    # --- 11. Horizontal tendencies ---
    dvor_hat = -flux_vor_div
    ddiv_hat = flux_vor_curl - grid.lap[:, jnp.newaxis] * E_hat

    # --- 12. Vertical advection of momentum ---
    vert_adv_u = _vertical_advection_spectral(u.real, w, z_coord, J.real)
    vert_adv_v = _vertical_advection_spectral(v.real, w, z_coord, J.real)

    vert_u_cos = vert_adv_u * grid.cos_lat[:, jnp.newaxis, jnp.newaxis] * mask_3d
    vert_v_cos = vert_adv_v * grid.cos_lat[:, jnp.newaxis, jnp.newaxis] * mask_3d

    dvor_hat = dvor_hat + (
        im_over_a[:, jnp.newaxis] * sh_analysis_oc2_3d(grid, vert_v_cos)
        + one_over_a * sh_analysis_dmu_3d(grid, vert_u_cos)
    )
    ddiv_hat = ddiv_hat + (
        im_over_a[:, jnp.newaxis] * sh_analysis_oc2_3d(grid, vert_u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, vert_v_cos)
    )

    # --- 13. Temperature equation ---
    T_u_cos = T * u_cos * mask_3d
    T_v_cos = T * v_cos * mask_3d
    flux_T_div = (
        im_over_a[:, jnp.newaxis] * sh_analysis_oc2_3d(grid, T_u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, T_v_cos)
    )
    T_div = T * div * mask_3d
    dT_hat = -flux_T_div + sh_analysis_3d(grid, T_div)

    vert_adv_T = _vertical_advection_spectral(T.real, w, z_coord, J.real) * mask_3d
    dT_hat = dT_hat + sh_analysis_3d(grid, vert_adv_T)

    # Vertical diffusion of T
    if config.K_v > 0:
        vdiff_T = vertical_diffusion(T.real, z_coord, J.real, config.K_v) * mask_3d
        dT_hat = dT_hat + sh_analysis_3d(grid, vdiff_T)

    # --- 14. Salinity equation ---
    S_u_cos = S * u_cos * mask_3d
    S_v_cos = S * v_cos * mask_3d
    flux_S_div = (
        im_over_a[:, jnp.newaxis] * sh_analysis_oc2_3d(grid, S_u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, S_v_cos)
    )
    S_div = S * div * mask_3d
    dS_hat = -flux_S_div + sh_analysis_3d(grid, S_div)

    vert_adv_S = _vertical_advection_spectral(S.real, w, z_coord, J.real) * mask_3d
    dS_hat = dS_hat + sh_analysis_3d(grid, vert_adv_S)

    if config.K_v > 0:
        vdiff_S = vertical_diffusion(S.real, z_coord, J.real, config.K_v) * mask_3d
        dS_hat = dS_hat + sh_analysis_3d(grid, vdiff_S)

    # --- 15. Explicit viscosity/diffusion ---
    if config.A_h > 0 or config.K_h > 0:
        lap = grid.lap[:, jnp.newaxis]  # negative semi-definite eigenvalues
        if config.A_h > 0:
            dvor_hat = dvor_hat + config.A_h * lap * state.vor_hat.data
            ddiv_hat = ddiv_hat + config.A_h * lap * state.div_hat.data
        if config.K_h > 0:
            dT_hat = dT_hat + config.K_h * lap * state.T_hat.data
            dS_hat = dS_hat + config.K_h * lap * state.S_hat.data

    if config.A_v > 0:
        vdiff_u = vertical_diffusion(u.real, z_coord, J.real, config.A_v) * mask_3d
        vdiff_v = vertical_diffusion(v.real, z_coord, J.real, config.A_v) * mask_3d
        vdiff_u_cos = vdiff_u * grid.cos_lat[:, jnp.newaxis, jnp.newaxis]
        vdiff_v_cos = vdiff_v * grid.cos_lat[:, jnp.newaxis, jnp.newaxis]

        dvor_hat = dvor_hat + (
            im_over_a[:, jnp.newaxis] * sh_analysis_oc2_3d(grid, vdiff_v_cos)
            + one_over_a * sh_analysis_dmu_3d(grid, vdiff_u_cos)
        )
        ddiv_hat = ddiv_hat + (
            im_over_a[:, jnp.newaxis] * sh_analysis_oc2_3d(grid, vdiff_u_cos)
            - one_over_a * sh_analysis_dmu_3d(grid, vdiff_v_cos)
        )

    # --- 16. Free-surface tendency ---
    deta_dt_grid = -jnp.sum(div.real * h_k.real, axis=-1) * mask
    deta_hat = sh_analysis(grid, deta_dt_grid)

    # --- 17. Spectral hyperdiffusion ---
    if config.hyperdiff_coeff > 0:
        dvor_hat = dvor_hat + spectral_hyperdiffusion_3d(
            grid, state.vor_hat.data, config.hyperdiff_coeff, config.hyperdiff_order,
        )
        ddiv_hat = ddiv_hat + spectral_hyperdiffusion_3d(
            grid, state.div_hat.data, config.hyperdiff_coeff, config.hyperdiff_order,
        )
        dT_hat = dT_hat + spectral_hyperdiffusion_3d(
            grid, state.T_hat.data, config.hyperdiff_coeff, config.hyperdiff_order,
        )
        dS_hat = dS_hat + spectral_hyperdiffusion_3d(
            grid, state.S_hat.data, config.hyperdiff_coeff, config.hyperdiff_order,
        )

    # Return as same pytree structure
    return SpectralOceanState(
        vor_hat=state.vor_hat.replace(data=dvor_hat),
        div_hat=state.div_hat.replace(data=ddiv_hat),
        T_hat=state.T_hat.replace(data=dT_hat),
        S_hat=state.S_hat.replace(data=dS_hat),
        eta_hat=state.eta_hat.replace(data=deta_hat),
        H_bathy_hat=state.H_bathy_hat.replace(
            data=jnp.zeros_like(state.H_bathy_hat.data),
        ),
        land_mask_grid=state.land_mask_grid.replace(
            data=jnp.zeros_like(state.land_mask_grid.data),
        ),
    )


def _vertical_advection_spectral(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
) -> jnp.ndarray:
    """Vertical advection -w * d(field)/dz with upwind scheme (grid-space)."""
    w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
    jac_safe = jnp.maximum(jacobian[..., jnp.newaxis], 1.0e-10)
    w_star = w_full / jac_safe

    dz_half = z_coord.dz_half_ref * jac_safe
    grad = upwind_vertical_gradient(field, dz_half, w_star)
    return -w_star * grad


# ==============================================================================
# Model class
# ==============================================================================

class SpectralOceanModel:
    """Spectral ocean model on the Gaussian grid.

    Uses SSP-RK3 for time integration. Barotropic subcycling is
    handled in grid space within the tendency function.

    On Metal (Apple Silicon), SH transforms require complex128 which
    is unsupported on Metal GPU. The model auto-routes computation to
    CPU and transfers results back to Metal, matching the atmospheric
    spectral model pattern.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid with SH transform matrices.
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    config : SpectralOceanConfig, optional
        Model configuration.
    allow_unsupported_backend : bool
        If True, skip backend checks (for testing).
    """

    def __init__(
        self,
        grid: GaussianGrid,
        z_coord: OceanZStarCoordinate,
        config: SpectralOceanConfig | None = None,
        *,
        allow_unsupported_backend: bool = False,
    ):
        self.z_coord = z_coord
        self.config = config or SpectralOceanConfig()
        self._use_cpu_for_spectral = False
        self._cpu_device = None
        self._default_device = None

        if not jax.config.jax_enable_x64:
            msg = (
                "SpectralOceanModel requires float64/complex128 arithmetic. "
                "Set JAX_ENABLE_X64=True before importing JAX modules."
            )
            if allow_unsupported_backend:
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            else:
                raise ValueError(msg)

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

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: SpectralOceanState, dt: float,
    ) -> SpectralOceanState:
        """Advance one time step using SSP-RK3."""
        def tendency_fn(s):
            return spectral_ocean_tendencies(
                s, self.grid, self.z_coord, self.config,
            )

        if self._use_cpu_for_spectral:
            state_cpu = jax.device_put(state, self._cpu_device)
            result_cpu = ssp_rk3_step(state_cpu, tendency_fn, dt)
            return jax.device_put(result_cpu, self._default_device)

        return ssp_rk3_step(state, tendency_fn, dt)

    @partial(jax.jit, static_argnums=(0,))
    def _step_on_cpu(
        self, state: SpectralOceanState, dt: float,
    ) -> SpectralOceanState:
        """Step without device transfers (for batched CPU integration on Metal)."""
        def tendency_fn(s):
            return spectral_ocean_tendencies(
                s, self.grid, self.z_coord, self.config,
            )
        return ssp_rk3_step(state, tendency_fn, dt)

    def integrate(
        self,
        state: SpectralOceanState,
        duration: float,
        dt: float,
        save_every: int = 1,
    ) -> tuple[SpectralOceanState, list[SpectralOceanState]]:
        """Integrate forward for a given duration.

        On Metal, batches CPU transfers: transfer state to CPU once,
        run all steps on CPU, then transfer results back to Metal.
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

        state_out = jax.device_put(state_cpu, self._default_device)
        trajectory_out = [
            jax.device_put(s, self._default_device) for s in trajectory_cpu
        ]
        return state_out, trajectory_out


# ==============================================================================
# Initialization
# ==============================================================================

def rest_state_spectral_ocean(
    grid: GaussianGrid,
    z_coord: OceanZStarCoordinate,
    T_surface: float = 20.0,
    T_deep: float = 2.0,
    S_uniform: float = 35.0,
    H_max: float = 5500.0,
    land_lat_threshold: float = 80.0,
) -> SpectralOceanState:
    """Create a rest-state spectral ocean initial condition.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid.
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    T_surface, T_deep : float
        Surface and deep temperature [degC].
    S_uniform : float
        Uniform salinity [PSU].
    H_max : float
        Maximum ocean depth [m].
    land_lat_threshold : float
        Latitude threshold for land mask [degrees].
    """
    nlev = z_coord.n_levels
    n_sh = grid.n_sh

    dims_3d = ("spectral", "level")
    dims_2d = ("spectral",)

    # Land mask in grid space
    lat_deg = jnp.abs(grid.lat2d) * (180.0 / jnp.pi)
    mask = jnp.where(lat_deg < land_lat_threshold, 1.0, 0.0)

    # Bathymetry in grid space
    H_bathy_grid = jnp.where(mask > 0.5, H_max, 1.0)

    # Temperature profile (exponential stratification)
    scale_depth = 1000.0
    T_profile = T_deep + (T_surface - T_deep) * jnp.exp(z_coord.z_full_ref / scale_depth)
    T_grid = jnp.broadcast_to(
        T_profile[jnp.newaxis, jnp.newaxis, :],
        (grid.n_lat, grid.n_lon, nlev),
    ) * mask[..., jnp.newaxis]

    # Salinity
    S_grid = jnp.full(
        (grid.n_lat, grid.n_lon, nlev), S_uniform,
    ) * mask[..., jnp.newaxis]

    # Transform to spectral
    vor_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    div_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    T_hat = sh_analysis_3d(grid, T_grid.astype(jnp.float64))
    S_hat = sh_analysis_3d(grid, S_grid.astype(jnp.float64))
    eta_hat = jnp.zeros(n_sh, dtype=jnp.complex128)
    H_bathy_hat = sh_analysis(grid, H_bathy_grid.astype(jnp.float64))

    return SpectralOceanState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        T_hat=Field(data=T_hat, name="T_hat", dims=dims_3d, units="degC"),
        S_hat=Field(data=S_hat, name="S_hat", dims=dims_3d, units="PSU"),
        eta_hat=Field(data=eta_hat, name="eta_hat", dims=dims_2d, units="m"),
        H_bathy_hat=Field(data=H_bathy_hat, name="H_bathy_hat", dims=dims_2d, units="m"),
        land_mask_grid=Field(data=mask, name="land_mask", dims=("lat", "lon"), units=""),
    )
