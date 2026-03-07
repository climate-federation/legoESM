"""Consistent FV Hydrostatic Primitive Equations on the cubed-sphere.

Uses a compatible face-based discretization where surface pressure,
temperature, sigma_dot, and momentum all share the same FV interface
flux layer:

- Surface pressure: FV flux divergence (PPM) — conservative mass transport
- Temperature: centered advection — energy-consistent with momentum operators
- Sigma-dot: computed from FV divergence (same interface velocities as p_s)
- Momentum: vector-invariant form with centered Bernoulli gradient
- Divergence damping: selective damping using FV-consistent divergence

The FV divergence used for sigma_dot is discretely compatible with the
FV transport of p_s, ensuring that vertical velocity is consistent with
horizontal mass transport.

Note: PPM scalar advection of T was found to be energy-inconsistent with
the centered momentum operators on the cubed-sphere, causing exponential
temperature growth at face boundaries. Centered T advection resolves this.

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core (FV3)
- Harris & Lin (2013): A Two-Way Nested Global-Regional Dynamical Core
- Colella & Woodward (1984): The Piecewise Parabolic Method (PPM)
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
    hyperdiffusion_3d as _hyperdiffusion_3d,
    fv_scalar_advection_3d as _fv_scalar_advection_3d,
)
from legoesm.core.operators_fv import fv_flux_divergence
from legoesm.core.operators_fv_cubed import (
    fv_divergence_3d as _fv_divergence_3d,
    fv_divergence_damping_3d as _fv_divergence_damping_3d,
    face_boundary_weight,
    edge_blend_scalar,
    edge_blend_scalar_3d,
    edge_blend_vector_3d,
)
from legoesm.core.conservation import zero_mean_tendency
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
from legoesm import constants


class FVPrimitiveEquationConfig(NamedTuple):
    """Configuration for the consistent FV hydrostatic PE model.

    Divergence damping is the primary mechanism for controlling
    cube-imprinted computational modes. Edge blending provides
    localized Laplacian smoothing near face boundaries.
    """
    g: float = constants.g
    div_damp_2: float = 0.0       # 2nd-order divergence damping [m²/s]
    div_damp_4: float = 0.0       # 4th-order divergence damping [m⁴/s]
    hyperdiff_coeff: float = 0.0
    hyperdiff_ps_coeff: float = 0.0
    edge_blend_strength: float = 0.25  # Face-boundary blend (0=off)
    edge_blend_depth: int = 2          # Rows to blend near each edge
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    anchor_mass_to_initial: bool = False
    time_integrator: str = "ssp_rk3"
    use_limiter: bool = True


def fv_hydrostatic_tendencies(
    state: HydrostaticState,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate,
    config: FVPrimitiveEquationConfig = FVPrimitiveEquationConfig(),
    physics_tendency: HydrostaticTendencies | None = None,
) -> HydrostaticTendencies:
    """Compute tendencies for the consistent FV hydrostatic primitive equations.

    Key consistency: sigma_dot uses the SAME FV divergence operator as
    the p_s transport, so vertical velocity is compatible with horizontal
    mass flux.

    Parameters
    ----------
    state : HydrostaticState
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate
    config : FVPrimitiveEquationConfig
    physics_tendency : HydrostaticTendencies, optional

    Returns
    -------
    HydrostaticTendencies
    """
    u = state.u.data       # (6, n, n, nlev)
    v = state.v.data
    T = state.T.data
    p_s = state.p_s.data   # (6, n, n)
    phis = state.phis.data

    R_d = constants.R_d
    kappa = constants.kappa
    dsigma = sigma_coord.dsigma

    p_s = jnp.clip(p_s, 100.0, 2.0e6)

    # --- 1. Pressure at full levels ---
    p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)

    # --- 2. Geopotential via hydrostatic integration ---
    Phi = compute_geopotential(T, p_s, sigma_coord, phis)

    # --- 3. Kinetic energy and Bernoulli function ---
    K = 0.5 * (u**2 + v**2)
    B = K + Phi

    # --- 4. Horizontal dynamics (vmap over levels) ---
    zeta = _vorticity_3d(u, v, grid)
    abs_vor = zeta + grid.f[..., None]

    dB_dx = _gradient_x_3d(B, grid)
    dB_dy = _gradient_y_3d(B, grid)

    # Pressure gradient correction: -R_d * T * ∇(ln p_s)
    ln_ps = jnp.log(p_s)
    ln_ps_field = Field(data=ln_ps, name="ln_ps", dims=("face", "x", "y"),
                        units="", staggering="cell")
    dln_ps_dx = gradient_x(ln_ps_field, grid).data
    dln_ps_dy = gradient_y(ln_ps_field, grid).data

    pg_corr_x = R_d * T * dln_ps_dx[..., None]
    pg_corr_y = R_d * T * dln_ps_dy[..., None]

    du_dt_data = abs_vor * v - dB_dx - pg_corr_x
    dv_dt_data = -abs_vor * u - dB_dy - pg_corr_y

    # --- 5. Surface pressure tendency via FV transport ---
    # Vertically integrated velocity: u_int = sum(u_k * dsigma_k)
    sigma_top = sigma_coord.sigma_half[0]
    sigma_range = 1.0 - sigma_top

    u_int = jnp.sum(u * dsigma[None, None, None, :], axis=-1)  # (6, n, n)
    v_int = jnp.sum(v * dsigma[None, None, None, :], axis=-1)

    # FV transport of p_s using vertically-integrated velocity
    dp_s_dt_data = fv_flux_divergence(
        p_s, u_int / sigma_range, v_int / sigma_range, grid,
        limiter=config.use_limiter,
    )
    dp_s_dt_data = zero_mean_tendency(dp_s_dt_data, grid)

    # --- 6. Sigma-dot from FV-consistent divergence ---
    # Uses the SAME FV divergence operator as the mass flux,
    # ensuring discrete compatibility.
    div_v = _fv_divergence_3d(u, v, grid)
    sigma_dot = compute_sigma_dot(div_v, sigma_coord)

    # --- 7. Vertical advection of T, u, v ---
    vert_adv_T = vertical_advection(T, sigma_dot, sigma_coord)
    vert_adv_u = vertical_advection(u, sigma_dot, sigma_coord)
    vert_adv_v = vertical_advection(v, sigma_dot, sigma_coord)

    du_dt_data = du_dt_data + vert_adv_u
    dv_dt_data = dv_dt_data + vert_adv_v

    # --- 8. Thermodynamic equation ---
    # Use centered advection for T (not PPM) to maintain energy consistency
    # with the centered momentum operators. PPM is energy-inconsistent with
    # centered Bernoulli gradient, causing exponential T growth at face edges.
    # PPM is still used for p_s (mass transport) where conservation matters.
    dT_dx = _gradient_x_3d(T, grid)
    dT_dy = _gradient_y_3d(T, grid)
    horiz_adv_T = -(u * dT_dx + v * dT_dy)

    # Adiabatic heating: κ·T·ω/p
    omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt_data, sigma_coord)
    adiabatic = kappa * T * omega / p_full

    # Missing term from full material derivative of p_s
    v_dot_grad_lnps = u * dln_ps_dx[..., None] + v * dln_ps_dy[..., None]
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps

    dT_dt_data = horiz_adv_T + vert_adv_T + adiabatic

    # --- 9. Divergence damping (primary stabilization) ---
    if config.div_damp_2 > 0 or config.div_damp_4 > 0:
        du_damp, dv_damp = _fv_divergence_damping_3d(
            u, v, grid, config.div_damp_2, config.div_damp_4,
        )
        du_dt_data = du_dt_data + du_damp
        dv_dt_data = dv_dt_data + dv_damp

    # --- 10. Hyperdiffusion (secondary) ---
    if config.hyperdiff_coeff > 0:
        diff_u = _hyperdiffusion_3d(u, grid, config.hyperdiff_coeff)
        diff_v = _hyperdiffusion_3d(v, grid, config.hyperdiff_coeff)
        diff_T = _hyperdiffusion_3d(T, grid, config.hyperdiff_coeff)
        du_dt_data = du_dt_data + diff_u
        dv_dt_data = dv_dt_data + diff_v
        dT_dt_data = dT_dt_data + diff_T

    if config.hyperdiff_ps_coeff > 0:
        ps_field = Field(data=p_s, name="p_s", dims=("face", "x", "y"),
                         units="Pa", staggering="cell")
        diff_ps = hyperdiffusion(ps_field, grid, config.hyperdiff_ps_coeff)
        dp_s_dt_data = dp_s_dt_data + diff_ps.data

    # --- 11. Physics ---
    if physics_tendency is not None:
        du_dt_data = du_dt_data + physics_tendency.du_dt.data
        dv_dt_data = dv_dt_data + physics_tendency.dv_dt.data
        dT_dt_data = dT_dt_data + physics_tendency.dT_dt.data
        dp_s_dt_data = dp_s_dt_data + physics_tendency.dp_s_dt.data

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


class FVPrimitiveEquationModel:
    """Consistent FV hydrostatic PE model on the cubed-sphere.

    Uses PPM transport for p_s and T, FV-consistent sigma_dot,
    selective divergence damping, and edge blending near face boundaries.

    Parameters
    ----------
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate
    config : FVPrimitiveEquationConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        sigma_coord: SigmaCoordinate,
        config: FVPrimitiveEquationConfig | None = None,
    ):
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.config = config or FVPrimitiveEquationConfig()
        self._target_mass = None

        cfg = self.config
        if cfg.edge_blend_strength > 0 and cfg.edge_blend_depth > 0:
            self._eb_weight = face_boundary_weight(
                grid.n, cfg.edge_blend_depth, cfg.edge_blend_strength,
            )
        else:
            self._eb_weight = None

    def tendencies(
        self,
        state: HydrostaticState,
        physics_tendency: HydrostaticTendencies | None = None,
    ) -> HydrostaticTendencies:
        """Compute tendencies (pure function wrapper)."""
        return fv_hydrostatic_tendencies(
            state, self.grid, self.sigma_coord, self.config,
            physics_tendency,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: HydrostaticState, dt: float) -> HydrostaticState:
        """Advance one time step."""
        def tendency_fn(s):
            tend = fv_hydrostatic_tendencies(
                s, self.grid, self.sigma_coord, self.config,
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
            raise ValueError(
                f"Unsupported time_integrator={self.config.time_integrator!r}"
            )

        # Edge blending: localized smoothing near face boundaries
        if self._eb_weight is not None:
            u_new, v_new = edge_blend_vector_3d(
                state_new.u.data, state_new.v.data,
                self.grid, self._eb_weight,
            )
            T_new = edge_blend_scalar_3d(
                state_new.T.data, self.grid, self._eb_weight,
            )
            ps_new = edge_blend_scalar(
                state_new.p_s.data, self.grid, self._eb_weight,
            )
            state_new = HydrostaticState(
                u=state_new.u.replace(data=u_new),
                v=state_new.v.replace(data=v_new),
                T=state_new.T.replace(data=T_new),
                p_s=state_new.p_s.replace(data=ps_new),
                phis=state_new.phis,
            )

        if self.config.use_conservation_fixer and self.config.fix_mass:
            if self.config.anchor_mass_to_initial:
                from legoesm.core.conservation import fix_mass_hydrostatic_target
                if self._target_mass is None:
                    self._target_mass = global_integral(state.p_s, self.grid)
                state_new = fix_mass_hydrostatic_target(
                    state_new, self._target_mass, self.grid,
                )
            else:
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
        """Advance one time step with physics forcing."""
        def tendency_fn(s):
            phys = None
            if physics_fn is not None:
                phys = physics_fn(s, self.grid, self.sigma_coord)
            tend = fv_hydrostatic_tendencies(
                s, self.grid, self.sigma_coord, self.config,
                phys,
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
            raise ValueError(
                f"Unsupported time_integrator={self.config.time_integrator!r}"
            )

        # Edge blending: localized smoothing near face boundaries
        if self._eb_weight is not None:
            u_new, v_new = edge_blend_vector_3d(
                state_new.u.data, state_new.v.data,
                self.grid, self._eb_weight,
            )
            T_new = edge_blend_scalar_3d(
                state_new.T.data, self.grid, self._eb_weight,
            )
            ps_new = edge_blend_scalar(
                state_new.p_s.data, self.grid, self._eb_weight,
            )
            state_new = HydrostaticState(
                u=state_new.u.replace(data=u_new),
                v=state_new.v.replace(data=v_new),
                T=state_new.T.replace(data=T_new),
                p_s=state_new.p_s.replace(data=ps_new),
                phis=state_new.phis,
            )

        if self.config.use_conservation_fixer and self.config.fix_mass:
            if self.config.anchor_mass_to_initial:
                from legoesm.core.conservation import fix_mass_hydrostatic_target
                if self._target_mass is None:
                    self._target_mass = global_integral(state.p_s, self.grid)
                state_new = fix_mass_hydrostatic_target(
                    state_new, self._target_mass, self.grid,
                )
            else:
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
        """Integrate forward for a given duration."""
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
        """Integrate using jax.lax.scan."""
        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(
            scan_fn, state, xs=None, length=n_steps
        )
        return final_state, trajectory
