"""FV3-style C-D grid Hydrostatic Primitive Equations on the cubed-sphere.

Uses the same C-D grid discretisation as the shallow water solver:

* D-grid winds (cell corners) are prognostic.
* C-grid velocities (cell edges) are diagnosed for mass/tracer transport.
* Vorticity from circulation (exact on D-grid).
* Bernoulli/geopotential gradient via Arakawa-Lamb at D-grid corners.
* Surface pressure transport via upwind C-grid mass flux.
* Vertical coordinate: sigma or hybrid sigma-pressure.

State is stored on the A-grid (cell centres) for compatibility with the
existing physics infrastructure. Velocities are converted to D-grid for
the momentum computation, then converted back.

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Harris & Lin (2013): A Two-Way Nested Global-Regional Dynamical Core
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, HydrostaticTendencies
from legoesm.core.operators import gradient_x, gradient_y
from legoesm.core.operators_3d import hyperdiffusion_3d as _hyperdiffusion_3d
from legoesm.core.operators_cdgrid import (
    dgrid_to_cgrid,
    dgrid_vorticity,
    cgrid_divergence,
    cgrid_mass_flux_divergence,
    _arakawa_lamb_gradient,
    _interp_center_to_corner,
    _interp_corner_to_center,
    _laplacian_dgrid,
)
from legoesm.core.conservation import zero_mean_tendency
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.cubed_sphere_cdgrid import (
    CubedSphereCDGrid,
    create_cubed_sphere_cdgrid,
)
from legoesm.grids.vertical import (
    SigmaCoordinate,
    HybridSigmaPressureCoordinate,
    pressure_from_sigma,
    pressure_from_hybrid,
    dp_from_hybrid,
    compute_geopotential,
    compute_geopotential_hybrid,
    compute_sigma_dot,
    compute_mass_flux_hybrid,
    vertical_advection,
    vertical_advection_hybrid,
    compute_pressure_velocity,
    compute_omega_hybrid,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm import constants


class CDGridPrimitiveEquationConfig(NamedTuple):
    """Configuration for the C-D grid hydrostatic PE model."""
    g: float = constants.g
    A_h: float = 0.0              # Laplacian viscosity [m^2/s]
    hyperdiff_coeff: float = 0.0
    hyperdiff_ps_coeff: float = 0.0
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    anchor_mass_to_initial: bool = False
    time_integrator: str = "ssp_rk3"


def cdgrid_hydrostatic_tendencies(
    state: HydrostaticState,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    cdgrid: CubedSphereCDGrid,
    config: CDGridPrimitiveEquationConfig = CDGridPrimitiveEquationConfig(),
    physics_tendency: HydrostaticTendencies | None = None,
) -> HydrostaticTendencies:
    """Compute tendencies for the C-D grid hydrostatic primitive equations.

    Parameters
    ----------
    state : HydrostaticState
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    cdgrid : CubedSphereCDGrid
    config : CDGridPrimitiveEquationConfig
    physics_tendency : HydrostaticTendencies, optional

    Returns
    -------
    HydrostaticTendencies
    """
    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    u = state.u.data       # (6, n, n, nlev) — A-grid
    v = state.v.data
    T = state.T.data
    p_s = state.p_s.data   # (6, n, n)
    phis = state.phis.data

    R_d = constants.R_d
    kappa = constants.kappa

    p_s = jnp.clip(p_s, 100.0, 2.0e6)

    # --- 1. Pressure at full levels ---
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)
        dp = dp_from_hybrid(sigma_coord, p_s)
    else:
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)

    # --- 2. Geopotential via hydrostatic integration ---
    if _hybrid:
        Phi = compute_geopotential_hybrid(T, p_s, sigma_coord, phis)
    else:
        Phi = compute_geopotential(T, p_s, sigma_coord, phis)

    # --- 3. Convert A-grid velocities to D-grid ---
    u_d = _interp_center_to_corner(u, cdgrid)
    v_d = _interp_center_to_corner(v, cdgrid)

    # --- 4. Kinetic energy at cell centres ---
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)
    u_center = 0.5 * (u_c[:, :-1, :, :] + u_c[:, 1:, :, :])
    v_center = 0.5 * (v_c[:, :, :-1, :] + v_c[:, :, 1:, :])
    K = 0.5 * (u_center ** 2 + v_center ** 2)

    # --- 5. Bernoulli function ---
    B = K + Phi

    # --- 6. Pressure gradient correction: -R_d * T * grad(ln p_s) ---
    ln_ps = jnp.log(p_s)
    ln_ps_field = Field(data=ln_ps, name="ln_ps", dims=("face", "x", "y"),
                        units="", staggering="cell")
    dln_ps_dx = gradient_x(ln_ps_field, grid).data
    dln_ps_dy = gradient_y(ln_ps_field, grid).data

    # Interpolate pressure gradient correction to D-grid corners
    pg_corr_x = R_d * T * dln_ps_dx[..., None]
    pg_corr_y = R_d * T * dln_ps_dy[..., None]
    pg_corr_x_d = _interp_center_to_corner(pg_corr_x, cdgrid)
    pg_corr_y_d = _interp_center_to_corner(pg_corr_y, cdgrid)

    # --- 7. Vorticity ---
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)
    abs_vor = zeta + cdgrid.base.f[..., None]

    # --- 8. Gradients at D-grid corners (Arakawa-Lamb) ---
    dB_dx, dB_dy = _arakawa_lamb_gradient(B, cdgrid)

    # --- 9. Vorticity at corners ---
    abs_vor_corner = _interp_center_to_corner(abs_vor, cdgrid)

    # --- 10. D-grid momentum tendencies ---
    du_d_dt = abs_vor_corner * v_d - dB_dx - pg_corr_x_d
    dv_d_dt = -abs_vor_corner * u_d - dB_dy - pg_corr_y_d

    # Laplacian viscosity on D-grid
    if config.A_h > 0:
        du_d_dt = du_d_dt + config.A_h * _laplacian_dgrid(u_d, cdgrid)
        dv_d_dt = dv_d_dt + config.A_h * _laplacian_dgrid(v_d, cdgrid)

    # --- 11. Convert D-grid tendencies back to A-grid ---
    du_dt_data = _interp_corner_to_center(du_d_dt)
    dv_dt_data = _interp_corner_to_center(dv_d_dt)

    # --- 12. Surface pressure tendency via C-grid mass flux ---
    if _hybrid:
        div_v = cgrid_divergence(u_c, v_c, cdgrid)
        dp_transport = cgrid_mass_flux_divergence(dp, u_c, v_c, cdgrid)
        dp_s_dt_data = jnp.sum(dp_transport, axis=-1) / sigma_coord.B_range
        dp_s_dt_data = zero_mean_tendency(dp_s_dt_data, grid)

        mass_flux = compute_mass_flux_hybrid(div_v, p_s, sigma_coord)
        vert_adv_T = vertical_advection_hybrid(T, mass_flux, p_s, sigma_coord)
        vert_adv_u = vertical_advection_hybrid(u, mass_flux, p_s, sigma_coord)
        vert_adv_v = vertical_advection_hybrid(v, mass_flux, p_s, sigma_coord)
    else:
        dsigma = sigma_coord.dsigma
        sigma_top = sigma_coord.sigma_half[0]
        sigma_range = 1.0 - sigma_top

        u_int = jnp.sum(u * dsigma[None, None, None, :], axis=-1)
        v_int = jnp.sum(v * dsigma[None, None, None, :], axis=-1)

        # Use C-grid mass flux for ps transport
        u_int_d = _interp_center_to_corner(u_int[..., None], cdgrid)[..., 0]
        v_int_d = _interp_center_to_corner(v_int[..., None], cdgrid)[..., 0]
        u_int_c, v_int_c = dgrid_to_cgrid(u_int_d[..., None], v_int_d[..., None], cdgrid)
        dp_s_dt_data = cgrid_mass_flux_divergence(
            p_s, u_int_c[..., 0] / sigma_range,
            v_int_c[..., 0] / sigma_range, cdgrid,
        )
        dp_s_dt_data = zero_mean_tendency(dp_s_dt_data, grid)

        div_v = cgrid_divergence(u_c, v_c, cdgrid)
        sigma_dot = compute_sigma_dot(div_v, sigma_coord)
        vert_adv_T = vertical_advection(T, sigma_dot, sigma_coord)
        vert_adv_u = vertical_advection(u, sigma_dot, sigma_coord)
        vert_adv_v = vertical_advection(v, sigma_dot, sigma_coord)

    du_dt_data = du_dt_data + vert_adv_u
    dv_dt_data = dv_dt_data + vert_adv_v

    # --- 13. Thermodynamic equation ---
    # Horizontal advection via C-grid upwind
    dT_dt_data = cgrid_scalar_advection_centered(T, u, v, grid)

    # Adiabatic heating: kappa * T * omega / p
    if _hybrid:
        omega = compute_omega_hybrid(mass_flux, p_s, dp_s_dt_data, sigma_coord)
    else:
        omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt_data, sigma_coord)
    adiabatic = kappa * T * omega / p_full

    # Material derivative of ln(p_s)
    v_dot_grad_lnps = u * dln_ps_dx[..., None] + v * dln_ps_dy[..., None]
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps

    dT_dt_data = dT_dt_data + vert_adv_T + adiabatic

    # --- 14. Hyperdiffusion ---
    if config.hyperdiff_coeff > 0:
        du_dt_data = du_dt_data + _hyperdiffusion_3d(u, grid, config.hyperdiff_coeff)
        dv_dt_data = dv_dt_data + _hyperdiffusion_3d(v, grid, config.hyperdiff_coeff)
        dT_dt_data = dT_dt_data + _hyperdiffusion_3d(T, grid, config.hyperdiff_coeff)

    if config.hyperdiff_ps_coeff > 0:
        from legoesm.core.operators import hyperdiffusion
        ps_field = Field(data=p_s, name="p_s", dims=("face", "x", "y"),
                         units="Pa", staggering="cell")
        diff_ps = hyperdiffusion(ps_field, grid, config.hyperdiff_ps_coeff)
        dp_s_dt_data = dp_s_dt_data + diff_ps.data

    # --- 15. Physics ---
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


def cgrid_scalar_advection_centered(T, u, v, grid):
    """Centered horizontal advection of scalar T: -u*dT/dx - v*dT/dy.

    Uses centered differences on the A-grid (energy-consistent with
    the momentum operators).
    """
    from legoesm.core.operators_3d import gradient_x_3d, gradient_y_3d
    dT_dx = gradient_x_3d(T, grid)
    dT_dy = gradient_y_3d(T, grid)
    return -(u * dT_dx + v * dT_dy)


class CDGridPrimitiveEquationModel(IntegrationMixin):
    """FV3-style C-D grid hydrostatic PE model on the cubed-sphere.

    Parameters
    ----------
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    config : CDGridPrimitiveEquationConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
        config: CDGridPrimitiveEquationConfig | None = None,
    ):
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.config = config or CDGridPrimitiveEquationConfig()
        self.cdgrid = create_cubed_sphere_cdgrid(grid)
        self._target_mass = None

    def tendencies(
        self,
        state: HydrostaticState,
        physics_tendency: HydrostaticTendencies | None = None,
    ) -> HydrostaticTendencies:
        return cdgrid_hydrostatic_tendencies(
            state, self.grid, self.sigma_coord, self.cdgrid,
            self.config, physics_tendency,
        )

    @partial(jax.jit, static_argnums=(0, 3), donate_argnums=(1,))
    def step(self, state: HydrostaticState, dt: float, physics_fn=None) -> HydrostaticState:
        """Advance one time step."""
        def tendency_fn(s):
            phys = None
            if physics_fn is not None:
                phys = physics_fn(s, self.grid, self.sigma_coord)
            tend = cdgrid_hydrostatic_tendencies(
                s, self.grid, self.sigma_coord, self.cdgrid,
                self.config, phys,
            )
            return HydrostaticState(
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                T=s.T.replace(data=tend.dT_dt.data),
                p_s=s.p_s.replace(data=tend.dp_s_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
            )

        state_new = dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

        # Conservation fixer
        if self.config.use_conservation_fixer and self.config.fix_mass:
            if self.config.anchor_mass_to_initial:
                from legoesm.core.conservation import fix_mass_hydrostatic_target
                from legoesm.core.operators import global_integral
                if self._target_mass is None:
                    self._target_mass = global_integral(state.p_s, self.grid)
                state_new = fix_mass_hydrostatic_target(
                    state_new, self._target_mass, self.grid,
                )
            else:
                from legoesm.core.conservation import fix_mass_hydrostatic
                state_new = fix_mass_hydrostatic(state_new, state, self.grid)

        return state_new

    def step_with_physics(self, state, dt, physics_fn=None):
        return self.step(state, dt, physics_fn=physics_fn)
