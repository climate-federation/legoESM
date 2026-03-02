"""Spectral Primitive Equation Model using spherical harmonic transforms.

Solves the hydrostatic primitive equations on the sphere using the
vorticity-divergence formulation with pseudospectral (transform) method
in sigma-pressure coordinates.

Prognostic variables (in spectral space):
    vor_hat  : Relative vorticity SH coefficients, (n_sh, nlev)
    div_hat  : Divergence SH coefficients, (n_sh, nlev)
    T_hat    : Temperature SH coefficients, (n_sh, nlev)
    lnps_hat : Log(surface pressure) SH coefficients, (n_sh,)

Equations (vorticity-divergence form, Bourke 1972):
    d(vor)/dt  = -div((vor+f)*v) + curl(vert_adv)
    d(div)/dt  = curl((vor+f)*v) - lap(K + Phi + R_d*T*lnps) + div(vert_adv)
    d(T)/dt    = -div(T*v) + T*div(v) - sigma_dot*dT/dsigma + kappa*T*omega/p
    d(lnps)/dt = -integral(div*dsigma)

References
----------
- Bourke, W. (1972). An Efficient, One-Level, Primitive-Equation Spectral
  Model. Monthly Weather Review, 100, 683-689.
- Hack, J. J. & Jakob, R. (1992). Description of a Global Shallow Water
  Model Based on the Spectral Transform Method. NCAR TN-343+STR.
- Hoskins, B. J. & Simmons, A. J. (1975). A multi-layer spectral model
  and the semi-implicit method. Quart. J. R. Met. Soc., 101, 637-655.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

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
    _sh_synthesis_H,
)
from legoesm.grids.vertical import SigmaCoordinate
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm import constants

_LNPS_MIN = float(jnp.log(100.0))
_LNPS_MAX = float(jnp.log(2.0e6))
_COS_LAT_MIN = 1.0e-6


# =============================================================================
# State and config
# =============================================================================

class SpectralHydrostaticState(NamedTuple):
    """State for the spectral hydrostatic primitive equations.

    3D spectral fields: shape (n_sh, nlev) complex128
    2D spectral fields: shape (n_sh,) complex128
    """
    vor_hat: Field    # Spectral relative vorticity [1/s]
    div_hat: Field    # Spectral divergence [1/s]
    T_hat: Field      # Spectral temperature [K]
    lnps_hat: Field   # Spectral log(surface pressure) [-]
    phis_hat: Field   # Spectral surface geopotential [m^2/s^2] (static)


class SpectralPEConfig(NamedTuple):
    """Configuration for spectral primitive equation model."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    hyperdiff_order: int = 2
    semi_implicit: bool = False      # Use Hoskins-Simmons semi-implicit
    si_T_ref: float = 300.0         # Reference temperature for linearization [K]
    si_alpha: float = 0.5           # Implicitness (0.5 = Crank-Nicolson)
    si_substeps: int = 1            # Internal SI substeps per external model step
    si_hyperdiff_boost: float = 1.0  # Multiply hyperdiffusion in SI mode


# =============================================================================
# Internal vertical helpers (generic shapes, no cubed-sphere assumptions)
# =============================================================================

def _compute_geopotential_gaussian(T, p_s, sigma_coord, phis):
    """Simmons-Burridge geopotential on Gaussian grid.

    Same math as vertical.compute_geopotential but with generic
    broadcasting: T is (..., nlev), p_s is (...), phis is (...).
    """
    R_d = constants.R_d
    ln_ratio = sigma_coord.ln_ratio   # (nlev,)
    alpha = sigma_coord.alpha         # (nlev,)

    dPhi = R_d * T * ln_ratio         # broadcast: (..., nlev)

    dPhi_reversed = dPhi[..., ::-1]
    cumsum_reversed = jnp.cumsum(dPhi_reversed, axis=-1)
    cumsum = cumsum_reversed[..., ::-1]

    Phi_above = phis[..., None] + cumsum

    Phi_below = jnp.concatenate(
        [Phi_above[..., 1:], phis[..., None]], axis=-1,
    )

    Phi_full = Phi_below + alpha * R_d * T
    return Phi_full


def _compute_sigma_dot_gaussian(div_3d, sigma_coord):
    """Sigma-dot on arbitrary grid shape. div_3d is (..., nlev)."""
    dsigma = sigma_coord.dsigma
    fractional_sigma = sigma_coord.fractional_sigma

    div_dsigma = div_3d * dsigma
    D_total = jnp.sum(div_dsigma, axis=-1, keepdims=True)
    cumsum_div = jnp.cumsum(div_dsigma, axis=-1)

    sigma_dot_inner = fractional_sigma * D_total - cumsum_div

    shape_2d = div_3d.shape[:-1]
    zero_top = jnp.zeros((*shape_2d, 1))
    sigma_dot = jnp.concatenate([zero_top, sigma_dot_inner], axis=-1)
    sigma_dot = sigma_dot.at[..., -1].set(0.0)
    return sigma_dot


def _vertical_advection_sigma_gaussian(field, sigma_dot, sigma_coord):
    """Vertical advection -sigma_dot * dfield/dsigma (upwind). Generic shapes."""
    sigma_dot_full = 0.5 * (sigma_dot[..., :-1] + sigma_dot[..., 1:])
    dsigma_bwd = sigma_coord.dsigma_full
    df_bwd = jnp.diff(field, axis=-1)

    grad_bwd = jnp.concatenate(
        [jnp.zeros((*field.shape[:-1], 1)),
         df_bwd / dsigma_bwd],
        axis=-1,
    )
    grad_fwd = jnp.concatenate(
        [df_bwd / dsigma_bwd,
         jnp.zeros((*field.shape[:-1], 1))],
        axis=-1,
    )

    grad = jnp.where(sigma_dot_full > 0, grad_bwd, grad_fwd)
    return -sigma_dot_full * grad


def _compute_omega_gaussian(sigma_dot, p_s, dp_s_dt, sigma_coord):
    """Pressure velocity omega = sigma * dp_s/dt + p_s * sigma_dot_full."""
    sigma_full = sigma_coord.sigma_full
    sigma_dot_full = 0.5 * (sigma_dot[..., :-1] + sigma_dot[..., 1:])
    omega = sigma_full * dp_s_dt[..., None] + p_s[..., None] * sigma_dot_full
    return omega


# =============================================================================
# Tendency computation
# =============================================================================

def spectral_pe_tendencies(
    state: SpectralHydrostaticState,
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
    config: SpectralPEConfig,
    physics_tendency: SpectralHydrostaticState | None = None,
) -> SpectralHydrostaticState:
    """Compute spectral tendencies for the hydrostatic PE.

    Uses the pseudospectral transform method:
    1. Transform prognostic fields to grid space
    2. Compute nonlinear products on grid
    3. Transform products to spectral space
    4. Assemble tendencies using spectral operators

    Returns tendencies in the same pytree structure as state (for SSP-RK3).
    """
    a = grid.radius
    R_d = constants.R_d
    kappa = constants.kappa
    dsigma = sigma_coord.dsigma
    sigma_top = sigma_coord.sigma_half[0]
    sigma_range = 1.0 - sigma_top

    # --- 1. Transform to grid space ---
    vor = sh_synthesis_3d(grid, state.vor_hat.data)   # (n_lat, n_lon, nlev)
    div = sh_synthesis_3d(grid, state.div_hat.data)
    T = sh_synthesis_3d(grid, state.T_hat.data)
    lnps = jnp.clip(
        sh_synthesis(grid, state.lnps_hat.data),
        _LNPS_MIN,
        _LNPS_MAX,
    )    # (n_lat, n_lon)
    phis = sh_synthesis(grid, state.phis_hat.data)

    # --- 2. Velocities ---
    u_cos, v_cos = uv_from_vordiv_3d(
        grid, state.vor_hat.data, state.div_hat.data,
    )  # (n_lat, n_lon, nlev)
    cos_lat_3d = jnp.clip(grid.cos_lat[:, None, None], _COS_LAT_MIN, None)
    u = u_cos / cos_lat_3d
    v = v_cos / cos_lat_3d

    # --- 3. Pressure ---
    p_s = jnp.exp(lnps)
    p_full = p_s[..., None] * sigma_coord.sigma_full  # (n_lat, n_lon, nlev)

    # --- 4. Geopotential ---
    Phi = _compute_geopotential_gaussian(T, p_s, sigma_coord, phis)

    # --- 5. Kinetic energy ---
    K = 0.5 * (u * u + v * v)

    # --- 6. Absolute vorticity ---
    abs_vor = vor + grid.f[..., None]

    # --- 7. Sigma-dot ---
    sigma_dot = _compute_sigma_dot_gaussian(div, sigma_coord)

    # --- 8. Surface pressure tendency ---
    D_total = jnp.sum(div * dsigma, axis=-1)
    dlnps_dt_grid = -D_total / sigma_range
    dp_s_dt_grid = p_s * dlnps_dt_grid

    # --- 9. Spectral operators ---
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a  # (n_sh,)
    one_over_a = 1.0 / a

    # --- 10. Vorticity fluxes: (zeta+f)*u*cos, (zeta+f)*v*cos ---
    A_vor = abs_vor * u_cos   # (n_lat, n_lon, nlev)
    B_vor = abs_vor * v_cos

    # Spectral divergence of vorticity flux -> dvor/dt
    flux_vor_div = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, A_vor)
        - one_over_a * sh_analysis_dmu_3d(grid, B_vor)
    )  # (n_sh, nlev)

    # Spectral curl of vorticity flux -> ddiv/dt contribution
    flux_vor_curl = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, B_vor)
        + one_over_a * sh_analysis_dmu_3d(grid, A_vor)
    )

    # --- 11. Energy variable: E = K + Phi + R_d * T * lnps ---
    # This is the Bourke (1972) formulation for sigma-coordinate PGF.
    E = K + Phi + R_d * T * lnps[..., None]
    E_hat = sh_analysis_3d(grid, E)

    # --- 12. Horizontal tendencies ---
    dvor_hat = -flux_vor_div
    ddiv_hat = flux_vor_curl - grid.lap[:, None] * E_hat

    # --- 13. Temperature equation ---
    # Horizontal: dT/dt = -div(T*v) + T*div(v) (advective form from flux form)
    T_u_cos = T * u_cos
    T_v_cos = T * v_cos

    flux_T_div = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, T_u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, T_v_cos)
    )

    T_div = T * div
    T_div_hat = sh_analysis_3d(grid, T_div)

    dT_hat = -flux_T_div + T_div_hat

    # Vertical advection of T
    vert_adv_T = _vertical_advection_sigma_gaussian(T, sigma_dot, sigma_coord)
    dT_hat = dT_hat + sh_analysis_3d(grid, vert_adv_T)

    # Adiabatic heating: kappa * T * omega / p
    omega = _compute_omega_gaussian(sigma_dot, p_s, dp_s_dt_grid, sigma_coord)
    adiabatic = kappa * T * omega / p_full

    # Material derivative correction: kappa * T * v . grad(lnps)
    # Compute grad(lnps) on grid from spectral
    dfdlon = sh_synthesis(grid, 1j * grid.ms * state.lnps_hat.data)
    cos_lat_2d = jnp.clip(grid.cos_lat[:, None], _COS_LAT_MIN, None)
    dfdx = dfdlon / (a * cos_lat_2d)
    dfdtheta_cos = _sh_synthesis_H(grid, state.lnps_hat.data)
    dfdy = -dfdtheta_cos / (a * cos_lat_2d)
    v_dot_grad_lnps = u * dfdx[..., None] + v * dfdy[..., None]
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps

    dT_hat = dT_hat + sh_analysis_3d(grid, adiabatic)

    # --- 14. Vertical advection of momentum ---
    vert_adv_u = _vertical_advection_sigma_gaussian(u, sigma_dot, sigma_coord)
    vert_adv_v = _vertical_advection_sigma_gaussian(v, sigma_dot, sigma_coord)

    # Convert to spectral vor/div contributions
    vert_u_cos = vert_adv_u * grid.cos_lat[:, None, None]
    vert_v_cos = vert_adv_v * grid.cos_lat[:, None, None]

    # curl(vert_adv) -> dvor_hat
    vert_vor_tend = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, vert_v_cos)
        + one_over_a * sh_analysis_dmu_3d(grid, vert_u_cos)
    )
    # div(vert_adv) -> ddiv_hat
    vert_div_tend = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, vert_u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, vert_v_cos)
    )

    dvor_hat = dvor_hat + vert_vor_tend
    ddiv_hat = ddiv_hat + vert_div_tend

    # --- 15. Surface pressure tendency (spectral) ---
    dlnps_hat = sh_analysis(grid, dlnps_dt_grid)

    # --- 16. Spectral hyperdiffusion ---
    hyperdiff_coeff = config.hyperdiff_coeff
    if config.semi_implicit and config.si_hyperdiff_boost != 1.0:
        hyperdiff_coeff = hyperdiff_coeff * config.si_hyperdiff_boost

    if hyperdiff_coeff > 0:
        dvor_hat = dvor_hat + spectral_hyperdiffusion_3d(
            grid, state.vor_hat.data, hyperdiff_coeff, config.hyperdiff_order,
        )
        ddiv_hat = ddiv_hat + spectral_hyperdiffusion_3d(
            grid, state.div_hat.data, hyperdiff_coeff, config.hyperdiff_order,
        )
        dT_hat = dT_hat + spectral_hyperdiffusion_3d(
            grid, state.T_hat.data, hyperdiff_coeff, config.hyperdiff_order,
        )

    # --- 17. Add physics tendencies if provided ---
    if physics_tendency is not None:
        dvor_hat = dvor_hat + physics_tendency.vor_hat.data
        ddiv_hat = ddiv_hat + physics_tendency.div_hat.data
        dT_hat = dT_hat + physics_tendency.T_hat.data
        dlnps_hat = dlnps_hat + physics_tendency.lnps_hat.data

    # Return as same pytree structure (for SSP-RK3)
    return SpectralHydrostaticState(
        vor_hat=state.vor_hat.replace(data=dvor_hat),
        div_hat=state.div_hat.replace(data=ddiv_hat),
        T_hat=state.T_hat.replace(data=dT_hat),
        lnps_hat=state.lnps_hat.replace(data=dlnps_hat),
        phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
    )


# =============================================================================
# Model class
# =============================================================================

class SpectralPrimitiveEquationModel:
    """Spectral primitive equation model on the sphere.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid with precomputed SH transform matrices.
    sigma_coord : SigmaCoordinate
        Vertical sigma coordinate.
    config : SpectralPEConfig, optional
        Model configuration.
    """

    def __init__(
        self,
        grid: GaussianGrid,
        sigma_coord: SigmaCoordinate,
        config: SpectralPEConfig | None = None,
        *,
        allow_unsupported_backend: bool = False,
        legoesm_config=None,
    ):
        self.sigma_coord = sigma_coord
        self.config = config or SpectralPEConfig()
        self._use_cpu_for_spectral = False
        self._cpu_device = None
        self._default_device = None
        self._si_data = None
        self._si_dt = None

        if self.config.si_substeps < 1:
            raise ValueError(
                f"si_substeps must be >= 1, got {self.config.si_substeps!r}",
            )

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

    def _ensure_si_data(self, dt: float):
        """Lazily precompute semi-implicit matrices and refresh when dt changes."""
        if not self.config.semi_implicit:
            return

        from legoesm.timestepping.semi_implicit import precompute_si_matrices

        dt_si = float(dt) / float(self.config.si_substeps)
        if self._si_data is None or self._si_dt != dt_si:
            self._si_data = precompute_si_matrices(
                self.grid, self.sigma_coord,
                T_ref=self.config.si_T_ref,
                alpha=self.config.si_alpha,
                dt=dt_si,
            )
            self._si_dt = dt_si

    def _do_step(self, state, dt, tendency_fn):
        """Core step: explicit RK3 or semi-implicit RK3."""
        if self.config.semi_implicit:
            from legoesm.timestepping.semi_implicit import ssp_rk3_step_si
            n_substeps = int(self.config.si_substeps)
            dt_si = dt / float(n_substeps)

            if n_substeps == 1:
                return ssp_rk3_step_si(
                    state, tendency_fn, dt_si, self._si_data, self.grid,
                )

            def si_substep(_, s):
                return ssp_rk3_step_si(
                    s, tendency_fn, dt_si, self._si_data, self.grid,
                )

            return jax.lax.fori_loop(0, n_substeps, si_substep, state)
        return ssp_rk3_step(state, tendency_fn, dt)

    @partial(jax.jit, static_argnums=(0, 2))
    def step(self, state: SpectralHydrostaticState, dt: float) -> SpectralHydrostaticState:
        """Advance one time step using SSP-RK3 (explicit or semi-implicit)."""
        self._ensure_si_data(dt)

        def tendency_fn(s):
            return spectral_pe_tendencies(s, self.grid, self.sigma_coord, self.config)

        if self._use_cpu_for_spectral:
            state_cpu = jax.device_put(state, self._cpu_device)
            result_cpu = self._do_step(state_cpu, dt, tendency_fn)
            return jax.device_put(result_cpu, self._default_device)

        return self._do_step(state, dt, tendency_fn)

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def step_with_physics(
        self,
        state: SpectralHydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> SpectralHydrostaticState:
        """Advance one time step with physics forcing."""
        self._ensure_si_data(dt)

        def tendency_fn(s):
            phys = None
            if physics_fn is not None:
                phys = physics_fn(s, self.grid, self.sigma_coord)
            return spectral_pe_tendencies(
                s, self.grid, self.sigma_coord, self.config, phys,
            )

        if self._use_cpu_for_spectral:
            state_cpu = jax.device_put(state, self._cpu_device)
            result_cpu = self._do_step(state_cpu, dt, tendency_fn)
            return jax.device_put(result_cpu, self._default_device)

        return self._do_step(state, dt, tendency_fn)

    @partial(jax.jit, static_argnums=(0, 2))
    def _step_on_cpu(self, state: SpectralHydrostaticState, dt: float) -> SpectralHydrostaticState:
        """Step without device transfers (for batched CPU integration on Metal)."""
        def tendency_fn(s):
            return spectral_pe_tendencies(s, self.grid, self.sigma_coord, self.config)
        return self._do_step(state, dt, tendency_fn)

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _step_on_cpu_with_physics(
        self,
        state: SpectralHydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> SpectralHydrostaticState:
        """Step with physics, no device transfers (for batched CPU integration)."""
        def tendency_fn(s):
            phys = None
            if physics_fn is not None:
                phys = physics_fn(s, self.grid, self.sigma_coord)
            return spectral_pe_tendencies(
                s, self.grid, self.sigma_coord, self.config, phys,
            )
        return self._do_step(state, dt, tendency_fn)

    def integrate(
        self,
        state: SpectralHydrostaticState,
        duration: float,
        dt: float,
        save_every: int = 1,
        physics_fn=None,
    ) -> tuple[SpectralHydrostaticState, list]:
        """Integrate forward for a given duration (Python loop).

        On Metal, batches CPU transfers: transfer state to CPU once,
        run all steps on CPU, then transfer results back to Metal.
        This avoids per-step CPU↔Metal round-trips.
        """
        n_steps = int(duration / dt)
        self._ensure_si_data(dt)

        if self._use_cpu_for_spectral:
            return self._integrate_on_cpu(state, n_steps, dt, save_every, physics_fn)

        trajectory = [state]
        for i in range(n_steps):
            if physics_fn is not None:
                state = self.step_with_physics(state, dt, physics_fn)
            else:
                state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)
        return state, trajectory

    def _integrate_on_cpu(self, state, n_steps, dt, save_every, physics_fn=None):
        """Batch integration on CPU: transfer once, not per step."""
        state_cpu = jax.device_put(state, self._cpu_device)
        trajectory_cpu = [state_cpu]

        for i in range(n_steps):
            if physics_fn is not None:
                state_cpu = self._step_on_cpu_with_physics(state_cpu, dt, physics_fn)
            else:
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

def isothermal_rest_state_spectral(
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
    T_init: float = 300.0,
    p_s_init: float = 1e5,
) -> SpectralHydrostaticState:
    """Create an isothermal rest-state initial condition in spectral space.

    All fields are at rest (zero winds) with uniform temperature and
    uniform surface pressure.
    """
    nlev = sigma_coord.n_levels
    n_sh = grid.n_sh

    dims_3d = ("spectral", "level")
    dims_2d = ("spectral",)

    # Zero winds -> zero vorticity and divergence
    vor_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    div_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)

    # Uniform temperature: only the n=0,m=0 mode is nonzero
    T_grid = jnp.full((grid.n_lat, grid.n_lon), T_init, dtype=jnp.float64)
    T_hat_2d = sh_analysis(grid, T_grid)  # (n_sh,)
    T_hat = jnp.broadcast_to(T_hat_2d[:, None], (n_sh, nlev)).copy()

    # Uniform lnps
    lnps_grid = jnp.full(
        (grid.n_lat, grid.n_lon), jnp.log(p_s_init), dtype=jnp.float64,
    )
    lnps_hat = sh_analysis(grid, lnps_grid)

    # No topography
    phis_hat = jnp.zeros(n_sh, dtype=jnp.complex128)

    return SpectralHydrostaticState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        T_hat=Field(data=T_hat, name="T_hat", dims=dims_3d, units="K"),
        lnps_hat=Field(data=lnps_hat, name="lnps_hat", dims=dims_2d, units=""),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
    )


# =============================================================================
# Baroclinic wave initialization (spectral)
# =============================================================================

def baroclinic_wave_init_spectral(
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
    perturbed: bool = True,
) -> SpectralHydrostaticState:
    """Initialize Jablonowski-Williamson baroclinic wave in spectral space.

    Evaluates the analytic JW06 balanced state on Gaussian grid points,
    then transforms u,v → vorticity/divergence via spectral analysis.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid with SH transform matrices.
    sigma_coord : SigmaCoordinate
        Vertical sigma coordinate.
    perturbed : bool
        If True, add the exponential perturbation to trigger instability.

    Returns
    -------
    SpectralHydrostaticState
        Initial state for the baroclinic wave test.
    """
    import numpy as np
    from legoesm.atmosphere.physics.baroclinic_wave import (
        evaluate_pressure_temperature,
        find_z_for_pressure,
        compute_zonal_wind,
        exponential_perturbation,
        P0,
    )

    nlev = sigma_coord.n_levels
    n_sh = grid.n_sh

    # Grid coordinates as numpy for the analytic solution
    lat_np = np.array(grid.lat)        # (n_lat,)
    lon_np = np.array(grid.lon2d[0])   # (n_lon,) — all rows same longitude
    sigma_full = np.array(sigma_coord.sigma_full)  # (nlev,)

    n_lat = grid.n_lat
    n_lon = grid.n_lon

    # Create 2D lat/lon for each level
    lat_2d = np.broadcast_to(lat_np[:, None], (n_lat, n_lon))
    lon_2d = np.array(grid.lon2d)

    # Allocate 3D fields (n_lat, n_lon, nlev)
    u_3d = np.zeros((n_lat, n_lon, nlev))
    v_3d = np.zeros((n_lat, n_lon, nlev))
    T_3d = np.zeros((n_lat, n_lon, nlev))

    # Compute initial conditions level by level
    for k in range(nlev):
        # Target pressure at this sigma level
        p_target = np.full((n_lat, n_lon), sigma_full[k] * P0)

        # Find height where p(z, lat) = p_target
        z_k = find_z_for_pressure(p_target, lat_2d)

        # Compute temperature at this height
        _, T_k = evaluate_pressure_temperature(z_k, lat_2d)

        # Compute zonal wind from gradient-wind balance
        u_k = compute_zonal_wind(z_k, lat_2d, T_k)

        # Add perturbation if requested
        if perturbed:
            u_k = u_k + exponential_perturbation(lat_2d, lon_2d, z_k)

        u_3d[:, :, k] = u_k
        T_3d[:, :, k] = T_k

    # Convert to JAX float64
    u_jax = jnp.array(u_3d, dtype=jnp.float64)
    v_jax = jnp.array(v_3d, dtype=jnp.float64)
    T_jax = jnp.array(T_3d, dtype=jnp.float64)

    # --- Transform u,v to spectral vorticity/divergence ---
    # The spectral PE uses vorticity = curl(v) and divergence = div(v).
    # From (u, v) on the Gaussian grid, we compute:
    #   vor_hat = curl operator applied to (u*cos, v*cos)
    #   div_hat = div operator applied to (u*cos, v*cos)
    a = grid.radius
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a

    cos_lat_3d = grid.cos_lat[:, None, None]
    u_cos = u_jax * cos_lat_3d
    v_cos = v_jax * cos_lat_3d

    # vor_hat = (im/a) * SH{v*cos/cos^2} + (1/a) * SH_dmu{u*cos}
    vor_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, v_cos)
        + one_over_a * sh_analysis_dmu_3d(grid, u_cos)
    )
    # div_hat = (im/a) * SH{u*cos/cos^2} - (1/a) * SH_dmu{v*cos}
    div_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, v_cos)
    )

    # Temperature to spectral
    T_hat = sh_analysis_3d(grid, T_jax)

    # Uniform surface pressure (no topography for JW06)
    lnps_grid = jnp.full(
        (n_lat, n_lon), jnp.log(P0), dtype=jnp.float64,
    )
    lnps_hat = sh_analysis(grid, lnps_grid)

    # No topography
    phis_hat = jnp.zeros(n_sh, dtype=jnp.complex128)

    dims_3d = ("spectral", "level")
    dims_2d = ("spectral",)

    return SpectralHydrostaticState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        T_hat=Field(data=T_hat, name="T_hat", dims=dims_3d, units="K"),
        lnps_hat=Field(data=lnps_hat, name="lnps_hat", dims=dims_2d, units=""),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
    )


# =============================================================================
# Diagnostic utilities
# =============================================================================

def spectral_pe_to_grid(
    state: SpectralHydrostaticState,
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
) -> dict[str, jax.Array]:
    """Convert spectral PE state to grid-point fields for diagnostics.

    Returns
    -------
    dict with keys: 'u', 'v', 'T', 'vor', 'div', 'lnps', 'p_s', 'phis'
    """
    vor = sh_synthesis_3d(grid, state.vor_hat.data)
    div = sh_synthesis_3d(grid, state.div_hat.data)
    T = sh_synthesis_3d(grid, state.T_hat.data)
    lnps = jnp.clip(
        sh_synthesis(grid, state.lnps_hat.data),
        _LNPS_MIN,
        _LNPS_MAX,
    )
    phis = sh_synthesis(grid, state.phis_hat.data)

    u_cos, v_cos = uv_from_vordiv_3d(
        grid, state.vor_hat.data, state.div_hat.data,
    )
    cos_lat_3d = jnp.clip(grid.cos_lat[:, None, None], _COS_LAT_MIN, None)
    u = u_cos / cos_lat_3d
    v = v_cos / cos_lat_3d

    return {
        'u': u, 'v': v, 'T': T,
        'vor': vor, 'div': div,
        'lnps': lnps, 'p_s': jnp.exp(lnps),
        'phis': phis,
    }
