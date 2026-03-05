"""Hydrostatic Primitive Equations on the cubed-sphere.

The hydrostatic primitive equations in σ-coordinates using the
**vector-invariant form** for momentum:

    dp_s/dt = -∫₀¹ div(p_s · v) dσ                      [surface pressure]
    du/dt   =  (ζ+f)·v - ∂B/∂x - (R_d·T/p)·∂p_s/∂x    [x-momentum]
    dv/dt   = -(ζ+f)·u - ∂B/∂y - (R_d·T/p)·∂p_s/∂y    [y-momentum]
    dT/dt   = -v·∇T - σ̇·∂T/∂σ + κ·T·ω/p               [thermodynamic]

where:
    B = K + Φ           Bernoulli function
    K = 0.5(u² + v²)   kinetic energy
    Φ                   geopotential (from hydrostatic balance)
    σ̇                  vertical velocity in σ-coordinates
    ω = dp/dt           pressure velocity
    κ = R_d / c_pd      Poisson constant

The vector-invariant form avoids explicit momentum advection and only
requires scalar gradients of B across face boundaries, consistent with
the shallow water formulation from Milestone 1.

References
----------
- Simmons & Burridge (1981): Energy and Angular-Momentum Conserving
  Vertical Finite-Difference Scheme.
- Held & Suarez (1994): A Proposal for the Intercomparison of the
  Dynamical Cores of Atmospheric General Circulation Models.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, HydrostaticTendencies
from legoesm.core.operators import (
    gradient_x,
    gradient_y,
    hyperdiffusion,
    global_integral,
)
from legoesm.core.operators_3d import (
    vorticity_3d as _vorticity_3d,
    gradient_x_3d as _gradient_x_3d,
    gradient_y_3d as _gradient_y_3d,
    divergence_3d as _divergence_3d,
    hyperdiffusion_3d as _hyperdiffusion_3d,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import (
    SigmaCoordinate,
    pressure_from_sigma,
    compute_geopotential,
    compute_sigma_dot,
    vertical_advection,
    compute_pressure_velocity,
)
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm.timestepping.ssp_rk34 import ssp_rk34_step
from legoesm.timestepping.ssp_rk54 import ssp_rk54_step
from legoesm.atmosphere.dynamics.edge_blending import (
    blend_scalar_cube_edges,
    blend_vector_cube_edges,
)
from legoesm import constants


class PrimitiveEquationConfig(NamedTuple):
    """Configuration for the hydrostatic primitive equation model."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    hyperdiff_ps_coeff: float = 0.0  # Separate coefficient for p_s ∇⁴ diffusion
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    time_integrator: str = "ssp_rk3"  # "ssp_rk3" | "ssp_rk34"/"ssp34" | "ssp_rk54"/"ssp45"
    edge_blend_uv: float = 0.0
    edge_blend_T: float = 0.0
    edge_blend_p_s: float = 0.0
    edge_blend_width: int = 1


def _apply_hydro_edge_blend(
    state: HydrostaticState,
    grid: CubedSphereGrid,
    config: PrimitiveEquationConfig,
) -> HydrostaticState:
    """Apply variable-specific cubed-sphere edge blending."""
    u_data = state.u.data
    v_data = state.v.data
    T_data = state.T.data
    p_s_data = state.p_s.data

    if config.edge_blend_uv > 0.0:
        u_data, v_data = blend_vector_cube_edges(
            u_data, v_data, grid.cos_angle, grid.sin_angle,
            config.edge_blend_uv, width=config.edge_blend_width,
        )
    if config.edge_blend_T > 0.0:
        T_data = blend_scalar_cube_edges(
            T_data, config.edge_blend_T, width=config.edge_blend_width
        )
    if config.edge_blend_p_s > 0.0:
        p_s_data = blend_scalar_cube_edges(
            p_s_data, config.edge_blend_p_s, width=config.edge_blend_width
        )

    return state._replace(
        u=state.u.replace(data=u_data),
        v=state.v.replace(data=v_data),
        T=state.T.replace(data=T_data),
        p_s=state.p_s.replace(data=p_s_data),
    )


# ==============================================================================
# Tendency computation
# ==============================================================================

def hydrostatic_tendencies(
    state: HydrostaticState,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate,
    config: PrimitiveEquationConfig = PrimitiveEquationConfig(),
    physics_tendency: HydrostaticTendencies | None = None,
) -> HydrostaticTendencies:
    """Compute tendencies for the hydrostatic primitive equations.

    Parameters
    ----------
    state : HydrostaticState
        Current model state.
    grid : CubedSphereGrid
        Horizontal grid.
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    config : PrimitiveEquationConfig
        Model configuration.
    physics_tendency : HydrostaticTendencies, optional
        Physics tendencies to add (e.g., Held-Suarez forcing).

    Returns
    -------
    HydrostaticTendencies : Time derivatives.
    """
    u = state.u.data       # (6, n, n, nlev)
    v = state.v.data       # (6, n, n, nlev)
    T = state.T.data       # (6, n, n, nlev)
    p_s = state.p_s.data   # (6, n, n)
    phis = state.phis.data  # (6, n, n)

    g = config.g
    R_d = constants.R_d
    kappa = constants.kappa
    dsigma = sigma_coord.dsigma  # (nlev,)
    p_s = jnp.clip(p_s, 100.0, 2.0e6)

    # --- 1. Pressure at full and half levels ---
    p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)  # (6,n,n,nlev)
    # p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)

    # --- 2. Geopotential via hydrostatic integration ---
    Phi = compute_geopotential(T, p_s, sigma_coord, phis)  # (6,n,n,nlev)

    # --- 3. Kinetic energy and Bernoulli function ---
    K = 0.5 * (u**2 + v**2)  # (6,n,n,nlev)
    B = K + Phi               # Bernoulli function

    # --- 4. Horizontal dynamics (vmap over levels) ---

    # 4a. Vorticity: ζ = dv/dx - du/dy at each level
    zeta = _vorticity_3d(u, v, grid)  # (6,n,n,nlev)
    # Absolute vorticity: ζ + f
    abs_vor = zeta + grid.f[..., None]  # broadcast f (6,n,n) to (6,n,n,nlev)

    # 4b. Bernoulli gradient
    dB_dx = _gradient_x_3d(B, grid)  # (6,n,n,nlev)
    dB_dy = _gradient_y_3d(B, grid)

    # 4c. Pressure gradient force (σ-coordinate correction)
    # In σ-coordinates: -R_d·T/p · ∇p_s = -R_d·T·σ/p · ∇p_s / σ
    #                                     = -R_d·T/(σ·p_s) · ∇p_s · σ·p_s/p
    # Since p = σ·p_s: R_d·T/p · ∇p_s
    # Compute ∇(ln p_s) = (1/p_s) · ∇p_s
    ln_ps = jnp.log(p_s)
    ln_ps_field = Field(data=ln_ps, name="ln_ps", dims=("face", "x", "y"),
                        units="", staggering="cell")
    dln_ps_dx = gradient_x(ln_ps_field, grid).data  # (6,n,n)
    dln_ps_dy = gradient_y(ln_ps_field, grid).data

    # Pressure gradient correction: R_d * T * ∇(ln p_s)
    # = R_d * T * (1/p_s) * ∇p_s
    # This enters as: -(R_d·T/p)·∂p_s/∂x = -R_d·T·(σ·p_s)/(σ·p_s) · ∂ln(p_s)/∂x
    #                                      = -R_d·T · ∂ln(p_s)/∂x
    # Wait - more carefully: in σ-coordinates the pressure gradient term is:
    #   -∂Φ/∂x|_σ - R_d·T_v·∂(ln p_s)/∂x
    # The Φ gradient is already in B, so the correction is just -R_d·T·∇(ln p_s)
    pg_corr_x = R_d * T * dln_ps_dx[..., None]  # (6,n,n,nlev)
    pg_corr_y = R_d * T * dln_ps_dy[..., None]

    # 4d. Vector-invariant momentum equations
    du_dt_data = abs_vor * v - dB_dx - pg_corr_x
    dv_dt_data = -abs_vor * u - dB_dy - pg_corr_y

    # --- 5. Surface pressure tendency and sigma-dot ---
    # Local-form continuity for internal consistency between dp_s/dt
    # and sigma-dot (both derived from the same div(v) field).
    #
    # The sigma-coordinate continuity equation is:
    #   ∂(ln p_s)/∂t + ∫ div(v) dσ = 0
    #
    # Local form:  dp_s/dt = -p_s · D_total / σ_range
    # where D_total = Σ div(v_k) · Δσ_k
    #
    # This ensures exact discrete consistency: the per-layer residual
    # dp_s/dt + p_s·div(v_k) + p_s·(σ̇_{k+1/2} - σ̇_{k-1/2})/Δσ_k = 0
    # holds to machine precision. Global mass is maintained by the
    # conservation fixer (fix_mass_hydrostatic).
    div_v = _divergence_3d(u, v, grid)  # (6,n,n,nlev)

    sigma_top = sigma_coord.sigma_half[0]
    sigma_range = 1.0 - sigma_top  # total sigma range

    D_total = jnp.sum(
        div_v * dsigma[None, None, None, :], axis=-1,
    )  # (6,n,n)
    dp_s_dt_data = -p_s * D_total / sigma_range

    # Sigma-dot (from same div_v, internally consistent)
    sigma_dot = compute_sigma_dot(div_v, sigma_coord)  # (6,n,n,nlev+1)

    # --- 7. Vertical advection of T ---
    vert_adv_T = vertical_advection(T, sigma_dot, sigma_coord)  # (6,n,n,nlev)

    # Also add vertical advection of u, v for consistency
    vert_adv_u = vertical_advection(u, sigma_dot, sigma_coord)
    vert_adv_v = vertical_advection(v, sigma_dot, sigma_coord)

    du_dt_data = du_dt_data + vert_adv_u
    dv_dt_data = dv_dt_data + vert_adv_v

    # --- 8. Thermodynamic equation ---
    # Horizontal advection of T in advective form: -v·∇T
    # Uses centered gradients, consistent with primitive_eq_latlon.py
    dT_dx = _gradient_x_3d(T, grid)
    dT_dy = _gradient_y_3d(T, grid)
    horiz_adv_T = -(u * dT_dx + v * dT_dy)

    # Adiabatic heating: κ·T·ω/p
    # The full pressure velocity is ω = σ·Dp_s/Dt + p_s·σ̇
    # where Dp_s/Dt = ∂p_s/∂t + v·∇p_s (material derivative).
    # compute_pressure_velocity uses the Eulerian ∂p_s/∂t, so we must
    # add the missing advective contribution: κ·T·v·∇ln(p_s).
    omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt_data, sigma_coord)
    adiabatic = kappa * T * omega / p_full

    # Missing term from full material derivative of p_s:
    # κ·T·σ·v·∇p_s / (σ·p_s) = κ·T·v·∇ln(p_s)
    v_dot_grad_lnps = u * dln_ps_dx[..., None] + v * dln_ps_dy[..., None]
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps

    dT_dt_data = horiz_adv_T + vert_adv_T + adiabatic

    # --- 9. Hyperdiffusion ---
    if config.hyperdiff_coeff > 0:
        diff_u = _hyperdiffusion_3d(u, grid, config.hyperdiff_coeff)
        diff_v = _hyperdiffusion_3d(v, grid, config.hyperdiff_coeff)
        diff_T = _hyperdiffusion_3d(T, grid, config.hyperdiff_coeff)

        du_dt_data = du_dt_data + diff_u
        dv_dt_data = dv_dt_data + diff_v
        dT_dt_data = dT_dt_data + diff_T

    # Surface pressure hyperdiffusion (damps 2Δx checkerboard mode)
    if config.hyperdiff_ps_coeff > 0:
        ps_field = Field(data=p_s, name="p_s", dims=("face", "x", "y"),
                         units="Pa", staggering="cell")
        diff_ps = hyperdiffusion(ps_field, grid, config.hyperdiff_ps_coeff)
        dp_s_dt_data = dp_s_dt_data + diff_ps.data

    # --- 10. Add physics tendencies if provided ---
    if physics_tendency is not None:
        du_dt_data = du_dt_data + physics_tendency.du_dt.data
        dv_dt_data = dv_dt_data + physics_tendency.dv_dt.data
        dT_dt_data = dT_dt_data + physics_tendency.dT_dt.data
        dp_s_dt_data = dp_s_dt_data + physics_tendency.dp_s_dt.data

    # --- Build tendency pytree ---
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    return HydrostaticTendencies(
        du_dt=Field(data=du_dt_data, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt_data, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dT_dt=Field(data=dT_dt_data, name="dT_dt", dims=dims_3d, units="K/s"),
        dp_s_dt=Field(data=dp_s_dt_data, name="dp_s_dt", dims=dims_2d, units="Pa/s"),
        dphis_dt=Field(
            data=jnp.zeros_like(phis), name="dphis_dt", dims=dims_2d, units="m^2/s^3"
        ),
    )


# ==============================================================================
# Model class
# ==============================================================================

class PrimitiveEquationModel:
    """Hydrostatic primitive equation model on the cubed-sphere.

    Parameters
    ----------
    grid : CubedSphereGrid
        Horizontal grid.
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    config : PrimitiveEquationConfig, optional
        Model configuration.

    Example
    -------
    >>> grid = create_cubed_sphere(48)
    >>> sigma = create_sigma_coordinate(20)
    >>> model = PrimitiveEquationModel(grid, sigma)
    >>> state = held_suarez_init(grid, sigma)
    >>> state_new = model.step(state, dt=600.0)
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        sigma_coord: SigmaCoordinate,
        config: PrimitiveEquationConfig | None = None,
    ):
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.config = config or PrimitiveEquationConfig()
        for name in ("edge_blend_uv", "edge_blend_T", "edge_blend_p_s"):
            value = float(getattr(self.config, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        if int(self.config.edge_blend_width) < 1:
            raise ValueError(
                "edge_blend_width must be >= 1, "
                f"got {self.config.edge_blend_width!r}"
            )

    def tendencies(
        self,
        state: HydrostaticState,
        physics_tendency: HydrostaticTendencies | None = None,
    ) -> HydrostaticTendencies:
        """Compute tendencies (pure function wrapper)."""
        return hydrostatic_tendencies(
            state, self.grid, self.sigma_coord, self.config, physics_tendency
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: HydrostaticState,
        dt: float,
    ) -> HydrostaticState:
        """Advance one time step using SSP-RK3.

        Parameters
        ----------
        state : HydrostaticState
            Current state.
        dt : float
            Time step [seconds].

        Returns
        -------
        HydrostaticState : State after one time step.
        """
        def tendency_fn(s):
            tend = hydrostatic_tendencies(
                s, self.grid, self.sigma_coord, self.config
            )
            # Return same pytree structure as state for tree_map compatibility.
            return HydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                T=s.T.replace(data=tend.dT_dt.data),
                p_s=s.p_s.replace(data=tend.dp_s_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
            )

        integrator = self.config.time_integrator.lower()
        if integrator in ("ssp_rk54", "ssp54", "ssp45", "rk54"):
            state_new = ssp_rk54_step(state, tendency_fn, dt)
        elif integrator in ("ssp_rk34", "ssp34", "rk34"):
            state_new = ssp_rk34_step(state, tendency_fn, dt)
        elif integrator in ("ssp_rk3", "ssp3", "rk3"):
            state_new = ssp_rk3_step(state, tendency_fn, dt)
        else:
            raise ValueError(f"Unsupported time_integrator={self.config.time_integrator!r}")

        if (
            self.config.edge_blend_uv > 0.0
            or self.config.edge_blend_T > 0.0
            or self.config.edge_blend_p_s > 0.0
        ):
            state_new = _apply_hydro_edge_blend(state_new, self.grid, self.config)

        # Apply conservation fixers
        if self.config.use_conservation_fixer and self.config.fix_mass:
            from legoesm.core.conservation import fix_mass_hydrostatic
            state_new = fix_mass_hydrostatic(state_new, state, self.grid)

        return state_new

    @partial(jax.jit, static_argnums=(0, 3))
    def step_with_physics(
        self,
        state: HydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> HydrostaticState:
        """Advance one time step with physics forcing.

        Parameters
        ----------
        state : HydrostaticState
            Current state.
        dt : float
            Time step [seconds].
        physics_fn : callable, optional
            Function (state, grid, sigma_coord) -> HydrostaticTendencies.

        Returns
        -------
        HydrostaticState : State after one time step.
        """
        def tendency_fn(s):
            # Recompute physics each RK stage for accuracy
            phys = None
            if physics_fn is not None:
                phys = physics_fn(s, self.grid, self.sigma_coord)
            tend = hydrostatic_tendencies(
                s, self.grid, self.sigma_coord, self.config, phys
            )
            return HydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                T=s.T.replace(data=tend.dT_dt.data),
                p_s=s.p_s.replace(data=tend.dp_s_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
            )

        integrator = self.config.time_integrator.lower()
        if integrator in ("ssp_rk54", "ssp54", "ssp45", "rk54"):
            state_new = ssp_rk54_step(state, tendency_fn, dt)
        elif integrator in ("ssp_rk34", "ssp34", "rk34"):
            state_new = ssp_rk34_step(state, tendency_fn, dt)
        elif integrator in ("ssp_rk3", "ssp3", "rk3"):
            state_new = ssp_rk3_step(state, tendency_fn, dt)
        else:
            raise ValueError(f"Unsupported time_integrator={self.config.time_integrator!r}")

        if (
            self.config.edge_blend_uv > 0.0
            or self.config.edge_blend_T > 0.0
            or self.config.edge_blend_p_s > 0.0
        ):
            state_new = _apply_hydro_edge_blend(state_new, self.grid, self.config)

        if self.config.use_conservation_fixer and self.config.fix_mass:
            from legoesm.core.conservation import fix_mass_hydrostatic
            state_new = fix_mass_hydrostatic(state_new, state, self.grid)

        return state_new

    def integrate(
        self,
        state: HydrostaticState,
        duration: float,
        dt: float,
        save_every: int = 1,
        physics_fn=None,
    ) -> tuple[HydrostaticState, list[HydrostaticState]]:
        """Integrate forward for a given duration.

        Parameters
        ----------
        state : HydrostaticState
            Initial state.
        duration : float
            Total integration time [seconds].
        dt : float
            Time step [seconds].
        save_every : int
            Save state every N steps.
        physics_fn : callable, optional
            Physics forcing function.

        Returns
        -------
        final_state : HydrostaticState
        trajectory : list of HydrostaticState
        """
        n_steps = int(duration / dt)
        trajectory = [state]

        for i in range(n_steps):
            if physics_fn is not None:
                state = self.step_with_physics(state, dt, physics_fn)
            else:
                state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)

        return state, trajectory

    def integrate_scan(
        self,
        state: HydrostaticState,
        n_steps: int,
        dt: float,
    ) -> tuple[HydrostaticState, HydrostaticState]:
        """Integrate using jax.lax.scan (differentiable, JIT-friendly).

        Note: does not support physics_fn (use integrate for that).

        Parameters
        ----------
        state : HydrostaticState
            Initial state.
        n_steps : int
            Number of time steps.
        dt : float
            Time step [seconds].

        Returns
        -------
        final_state : HydrostaticState
        trajectory : HydrostaticState
            All states, each leaf shape: (n_steps, ...).
        """
        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(
            scan_fn, state, xs=None, length=n_steps
        )
        return final_state, trajectory
