"""FV3 Hydrostatic Primitive Equations on the cubed-sphere (D-grid dynamics).

Prognostic winds are stored on the D-grid (cell corners). This eliminates
the A-to-D round-trip that plagued the earlier A-grid implementation and
caused a 62%-per-step Hollingsworth-type energy inconsistency.

Physics coupling converts D->A at the interface boundary only (diagnostic).

Operator staggering
-------------------
- Momentum:  D-grid prognostic, C-grid diagnostic (for KE / mass flux)
- Vorticity: cell centres (from D-grid circulation)
- Bernoulli / pressure gradient: Arakawa-Lamb gradient at D-grid corners
- Divergence: C-grid flux-form (exact mass conservation)
- Scalar diffusion: A-grid (proper inter-face halo exchange)
- Wind diffusion: D-grid Laplacian (A-grid round-trip with halo)

References
----------
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
- Simmons & Burridge (1981): Energy and Angular-Momentum Conserving Scheme
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import (
    HydrostaticState,
    HydrostaticTendencies,
    FV3HydrostaticState,
    FV3HydrostaticTendencies,
)
from legoesm.core.operators_cdgrid import (
    dgrid_to_cgrid,
    cgrid_divergence,
    dgrid_vorticity,
    _arakawa_lamb_gradient,
    _interp_center_to_corner,
    _interp_corner_to_center,
    _laplacian_dgrid,
)
from legoesm.core.operators_3d import (
    gradient_x_3d as _gradient_x_3d,
    gradient_y_3d as _gradient_y_3d,
    hyperdiffusion_3d as _hyperdiffusion_3d,
    laplacian_compact_3d as _laplacian_compact_3d,
)
from legoesm.core.operators import gradient_x, gradient_y
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


# ==============================================================================
# Configuration
# ==============================================================================

class CDGridPrimitiveEquationConfig(NamedTuple):
    """Configuration for the C-D grid hydrostatic PE model.

    For explicit time integration without semi-implicit gravity wave
    treatment, set ``n_barotropic_substeps > 1`` to subcycle the
    barotropic (external gravity wave) mode.  The barotropic substep
    advances surface pressure and the column-mean divergent flow with
    dt_baro = dt / n_barotropic_substeps, while the baroclinic modes
    (temperature, internal wind structure) evolve on the full dt.

    Alternatively, set ``implicit_grav_wave_damping > 0`` to apply a
    linearized implicit correction to p_s after each step, which damps
    the fastest gravity wave mode without substeps.
    """
    g: float = constants.g
    A_h: float = 0.0              # Laplacian viscosity [m^2/s]
    hyperdiff_coeff: float = 0.0
    hyperdiff_ps_coeff: float = 0.0
    div_damp_coeff: float = 0.0   # Divergence damping coefficient [m^2/s]
    implicit_grav_wave_damping: float = 0.0
        # Implicit damping factor for the external gravity wave mode.
        # Applied as an exponential filter: ps_new *= exp(-alpha * dt * lap(ps))
        # where alpha = implicit_grav_wave_damping.
        # Typical value: 0.5 * c_grav^2 * dt / dx^2 where c_grav ~ 300 m/s.
        # This is a simplified semi-implicit treatment that selectively
        # damps divergent modes without a full barotropic solve.
    T_min: float = 50.0            # Temperature floor [K] (positivity protection)
    p_floor: float = 100.0         # Pressure floor [Pa] for adiabatic heating (limits 1/p)
    sponge_sigma: float = 0.15     # Rayleigh sponge activates above this sigma
    sponge_tau_sec: float = 3600.0 # e-folding time at model top [s]
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    anchor_mass_to_initial: bool = False
    time_integrator: str = "ssp_rk3"
    T_diss_coeff: float = 0.0
        # Velocity-dependent Laplacian dissipation for the temperature
        # equation.  Mimics upwind advection's built-in diffusivity:
        #   nu_T = T_diss_coeff * |v| * dx
        # Typically not needed when A_h (Laplacian viscosity) is used,
        # since A_h already damps intermediate-scale T noise.
        # Typical range when used: 0.1-0.5.  0 disables (default).


# ==============================================================================
# Legacy A-grid tendency function (backward compatibility)
# ==============================================================================

def cdgrid_hydrostatic_tendencies(
    state: HydrostaticState,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    cdgrid: CubedSphereCDGrid,
    config: CDGridPrimitiveEquationConfig = CDGridPrimitiveEquationConfig(),
    physics_tendency: HydrostaticTendencies | None = None,
) -> HydrostaticTendencies:
    """Compute tendencies for the hydrostatic PE (A-grid state interface).

    Internally converts A-grid state to FV3 D-grid state, calls the
    D-grid tendency function, then converts the momentum tendencies
    back to A-grid.  This is a backward-compatibility wrapper; for new
    code use ``fv3_hydrostatic_tendencies`` with ``FV3HydrostaticState``.
    """
    from legoesm.core.operators_cdgrid import agrid_to_dgrid_vector

    # Convert physics tendency to D-grid if provided
    phys_tend_dgrid = None
    if physics_tendency is not None:
        pu_d, pv_d = agrid_to_dgrid_vector(
            physics_tendency.du_dt.data, physics_tendency.dv_dt.data, cdgrid,
        )
        phys_tend_dgrid = FV3HydrostaticTendencies(
            du_d_dt=physics_tendency.du_dt.replace(data=pu_d),
            dv_d_dt=physics_tendency.dv_dt.replace(data=pv_d),
            dT_dt=physics_tendency.dT_dt,
            dp_s_dt=physics_tendency.dp_s_dt,
            dphis_dt=physics_tendency.dphis_dt,
        )

    # Convert A-grid winds to D-grid
    u_d, v_d = agrid_to_dgrid_vector(state.u.data, state.v.data, cdgrid)
    fv3_state = FV3HydrostaticState(
        u_d=state.u.replace(data=u_d, name="u_d"),
        v_d=state.v.replace(data=v_d, name="v_d"),
        T=state.T,
        p_s=state.p_s,
        phis=state.phis,
        tracers=state.tracers,
    )

    fv3_tend = fv3_hydrostatic_tendencies(
        fv3_state, grid, sigma_coord, cdgrid, config, phys_tend_dgrid,
    )

    # Convert D-grid tendencies back to A-grid
    du_a = _interp_corner_to_center(fv3_tend.du_d_dt.data)
    dv_a = _interp_corner_to_center(fv3_tend.dv_d_dt.data)

    dims_3d = ("face", "x", "y", "level")
    return HydrostaticTendencies(
        du_dt=Field(data=du_a, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_a, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dT_dt=fv3_tend.dT_dt,
        dp_s_dt=fv3_tend.dp_s_dt,
        dphis_dt=fv3_tend.dphis_dt,
    )


# ==============================================================================
# FV3 D-grid tendency function (core implementation)
# ==============================================================================

def fv3_hydrostatic_tendencies(
    state: FV3HydrostaticState,
    grid: CubedSphereGrid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    cdgrid: CubedSphereCDGrid,
    config: CDGridPrimitiveEquationConfig = CDGridPrimitiveEquationConfig(),
    physics_tendency: FV3HydrostaticTendencies | None = None,
) -> FV3HydrostaticTendencies:
    """Compute tendencies for the FV3 hydrostatic PE with D-grid winds.

    The prognostic momentum is stored at D-grid cell corners, so there
    is NO A-to-D conversion.  The C-grid velocities are diagnosed from
    the D-grid winds for mass flux and KE computation.

    Parameters
    ----------
    state : FV3HydrostaticState
        Prognostic state with D-grid winds.
    grid : CubedSphereGrid
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    cdgrid : CubedSphereCDGrid
    config : CDGridPrimitiveEquationConfig
    physics_tendency : FV3HydrostaticTendencies, optional

    Returns
    -------
    FV3HydrostaticTendencies
    """
    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    u_d = state.u_d.data   # (6, n+1, n+1, nlev) — D-grid
    v_d = state.v_d.data
    T = state.T.data       # (6, n, n, nlev)
    p_s = state.p_s.data   # (6, n, n)
    phis = state.phis.data

    R_d = constants.R_d
    kappa = constants.kappa

    # Positivity protections
    T = jnp.maximum(T, config.T_min)
    p_s = jnp.clip(p_s, 100.0, 2.0e6)

    # --- 1. D-grid to C-grid ---
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)
    # u_c: (6, n+1, n, nlev),  v_c: (6, n, n+1, nlev)

    # Cell-centre velocities from C-grid (for scalar advection / KE)
    u_cell = 0.5 * (u_c[:, :-1, :, :] + u_c[:, 1:, :, :])   # (6, n, n, nlev)
    v_cell = 0.5 * (v_c[:, :, :-1, :] + v_c[:, :, 1:, :])   # (6, n, n, nlev)

    # --- 2. Pressure at full levels ---
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)
        dp = dp_from_hybrid(sigma_coord, p_s)
    else:
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)

    # --- 3. Geopotential via hydrostatic integration ---
    if _hybrid:
        Phi = compute_geopotential_hybrid(T, p_s, sigma_coord, phis)
    else:
        Phi = compute_geopotential(T, p_s, sigma_coord, phis)

    # --- 4. KE at cell centres from C-grid velocities ---
    KE = 0.5 * (u_cell ** 2 + v_cell ** 2)

    # --- 5. Bernoulli function B = KE + Phi (cell centres) ---
    B = KE + Phi

    # --- 6. D-grid vorticity at cell centres via circulation ---
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)  # (6, n, n, nlev)
    zeta_abs = zeta + grid.f[..., None]

    # Vorticity interpolated to D-grid corners
    zeta_corner = _interp_center_to_corner(zeta_abs, cdgrid)  # (6, n+1, n+1, nlev)

    # --- 7. Bernoulli gradient at D-grid corners (Arakawa-Lamb) ---
    dB_dx, dB_dy = _arakawa_lamb_gradient(B, cdgrid)  # (6, n+1, n+1, nlev)

    # --- 8. Pressure gradient correction at D-grid corners ---
    ln_ps = jnp.log(p_s)
    dln_dx, dln_dy = _arakawa_lamb_gradient(ln_ps, cdgrid)  # (6, n+1, n+1)
    # Harmonic mean for T at corners suppresses spurious PGF from high-n T.
    T_corner = 1.0 / _interp_center_to_corner(1.0 / T, cdgrid)  # (6, n+1, n+1, nlev)
    pg_corr_x = R_d * T_corner * dln_dx[..., None]
    pg_corr_y = R_d * T_corner * dln_dy[..., None]

    # --- 9. D-grid momentum tendencies ---
    du_d_dt = zeta_corner * v_d - dB_dx - pg_corr_x
    dv_d_dt = -zeta_corner * u_d - dB_dy - pg_corr_y

    # Divergence damping at D-grid
    if config.div_damp_coeff > 0:
        div_v_damp = cgrid_divergence(u_c, v_c, cdgrid)  # (6, n, n, nlev)
        ddiv_dx, ddiv_dy = _arakawa_lamb_gradient(div_v_damp, cdgrid)
        du_d_dt = du_d_dt - config.div_damp_coeff * ddiv_dx
        dv_d_dt = dv_d_dt - config.div_damp_coeff * ddiv_dy

    # --- 10. Surface pressure tendency and vertical motion ---
    # C-grid divergence for continuity
    div_v = cgrid_divergence(u_c, v_c, cdgrid)  # (6, n, n, nlev)

    if _hybrid:
        D_total_p = jnp.sum(div_v * dp, axis=-1)
        dp_s_dt_data = -D_total_p / sigma_coord.B_range
        dp_s_dt_data = zero_mean_tendency(dp_s_dt_data, grid)

        mass_flux = compute_mass_flux_hybrid(div_v, p_s, sigma_coord)
        vert_adv_T = vertical_advection_hybrid(T, mass_flux, p_s, sigma_coord)

        # Vertical advection of D-grid winds: interpolate to A-grid,
        # compute vertical advection, interpolate back to D-grid.
        u_a = _interp_corner_to_center(u_d)  # (6, n, n, nlev)
        v_a = _interp_corner_to_center(v_d)
        vert_adv_u_a = vertical_advection_hybrid(u_a, mass_flux, p_s, sigma_coord)
        vert_adv_v_a = vertical_advection_hybrid(v_a, mass_flux, p_s, sigma_coord)
        vert_adv_u_d = _interp_center_to_corner(vert_adv_u_a, cdgrid)
        vert_adv_v_d = _interp_center_to_corner(vert_adv_v_a, cdgrid)

        omega = compute_omega_hybrid(mass_flux, p_s, dp_s_dt_data, sigma_coord)
        p_adiab = jnp.maximum(p_full, config.p_floor)
    else:
        dsigma = sigma_coord.dsigma
        sigma_top = sigma_coord.sigma_half[0]
        sigma_range = 1.0 - sigma_top

        D_total = jnp.sum(div_v * dsigma, axis=-1)
        dp_s_dt_data = -p_s * D_total / sigma_range
        dp_s_dt_data = zero_mean_tendency(dp_s_dt_data, grid)

        sigma_dot = compute_sigma_dot(div_v, sigma_coord)
        vert_adv_T = vertical_advection(T, sigma_dot, sigma_coord)

        # Vertical advection of D-grid winds via A-grid round-trip
        u_a = _interp_corner_to_center(u_d)
        v_a = _interp_corner_to_center(v_d)
        vert_adv_u_a = vertical_advection(u_a, sigma_dot, sigma_coord)
        vert_adv_v_a = vertical_advection(v_a, sigma_dot, sigma_coord)
        vert_adv_u_d = _interp_center_to_corner(vert_adv_u_a, cdgrid)
        vert_adv_v_d = _interp_center_to_corner(vert_adv_v_a, cdgrid)

        omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt_data, sigma_coord)
        p_adiab = jnp.maximum(p_full, config.p_floor)

    du_d_dt = du_d_dt + vert_adv_u_d
    dv_d_dt = dv_d_dt + vert_adv_v_d

    # --- 11. Thermodynamic equation ---
    # Horizontal advection: centred advection using cell-centre velocities
    dT_dx = _gradient_x_3d(T, grid)
    dT_dy = _gradient_y_3d(T, grid)
    horiz_adv_T = -(u_cell * dT_dx + v_cell * dT_dy)

    # Adiabatic heating: kappa * T * omega / p
    # ln_ps gradient at A-grid for the v.grad(ln ps) correction
    ln_ps_field = Field(data=ln_ps, name="ln_ps", dims=("face", "x", "y"),
                        units="", staggering="cell")
    dln_ps_dx = gradient_x(ln_ps_field, grid).data  # (6, n, n)
    dln_ps_dy = gradient_y(ln_ps_field, grid).data
    adiabatic = kappa * T * omega / p_adiab
    v_dot_grad_lnps = u_cell * dln_ps_dx[..., None] + v_cell * dln_ps_dy[..., None]
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps

    dT_dt_data = horiz_adv_T + vert_adv_T + adiabatic

    # --- 11b. Velocity-dependent temperature dissipation ---
    if config.T_diss_coeff > 0:
        wind_speed = jnp.sqrt(u_cell**2 + v_cell**2)
        dx_local = grid.dx[..., None]
        nu_T = config.T_diss_coeff * wind_speed * dx_local
        lap_T = _laplacian_compact_3d(T, grid)
        dT_dt_data = dT_dt_data + nu_T * lap_T

    # --- 12. Diffusion ---
    # 12a. Laplacian viscosity on D-grid winds
    if config.A_h > 0:
        du_d_dt = du_d_dt + config.A_h * _laplacian_dgrid(u_d, cdgrid)
        dv_d_dt = dv_d_dt + config.A_h * _laplacian_dgrid(v_d, cdgrid)
        # Temperature: A-grid Laplacian (proper halo exchange)
        lap_T = _laplacian_compact_3d(T, grid)
        dT_dt_data = dT_dt_data + config.A_h * lap_T

    # 12b. Hyperdiffusion on D-grid winds (biharmonic)
    if config.hyperdiff_coeff > 0:
        du_d_dt = du_d_dt - config.hyperdiff_coeff * _laplacian_dgrid(
            _laplacian_dgrid(u_d, cdgrid), cdgrid)
        dv_d_dt = dv_d_dt - config.hyperdiff_coeff * _laplacian_dgrid(
            _laplacian_dgrid(v_d, cdgrid), cdgrid)
        # Temperature: A-grid hyperdiffusion (proper halo exchange)
        dT_dt_data = dT_dt_data + _hyperdiffusion_3d(T, grid, config.hyperdiff_coeff)

    # Surface pressure hyperdiffusion (A-grid)
    if config.hyperdiff_ps_coeff > 0:
        from legoesm.core.operators import hyperdiffusion
        ps_field = Field(data=p_s, name="p_s", dims=("face", "x", "y"),
                         units="Pa", staggering="cell")
        diff_ps = hyperdiffusion(ps_field, grid, config.hyperdiff_ps_coeff)
        dp_s_dt_data = dp_s_dt_data + diff_ps.data

    # --- 13. Upper-atmosphere Rayleigh sponge (D-grid) ---
    if config.sponge_tau_sec > 0 and config.sponge_sigma > 0:
        sigma_full = sigma_coord.sigma_full
        sponge_frac = jnp.clip(
            (config.sponge_sigma - sigma_full) / config.sponge_sigma, 0.0, 1.0
        )
        sponge_rate = sponge_frac**2 / config.sponge_tau_sec
        du_d_dt = du_d_dt - sponge_rate * u_d
        dv_d_dt = dv_d_dt - sponge_rate * v_d

    # --- 14. Physics ---
    if physics_tendency is not None:
        du_d_dt = du_d_dt + physics_tendency.du_d_dt.data
        dv_d_dt = dv_d_dt + physics_tendency.dv_d_dt.data
        dT_dt_data = dT_dt_data + physics_tendency.dT_dt.data
        dp_s_dt_data = dp_s_dt_data + physics_tendency.dp_s_dt.data

    dims_3d_corner = ("face", "x", "y", "level")
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    return FV3HydrostaticTendencies(
        du_d_dt=Field(data=du_d_dt, name="du_d_dt", dims=dims_3d_corner, units="m/s^2"),
        dv_d_dt=Field(data=dv_d_dt, name="dv_d_dt", dims=dims_3d_corner, units="m/s^2"),
        dT_dt=Field(data=dT_dt_data, name="dT_dt", dims=dims_3d, units="K/s"),
        dp_s_dt=Field(data=dp_s_dt_data, name="dp_s_dt", dims=dims_2d, units="Pa/s"),
        dphis_dt=Field(
            data=jnp.zeros_like(phis), name="dphis_dt", dims=dims_2d, units="m^2/s^3"
        ),
    )


# ==============================================================================
# Adapter functions: A-grid <-> D-grid conversion
# ==============================================================================

def hydrostatic_to_fv3(
    state: HydrostaticState,
    cdgrid: CubedSphereCDGrid,
) -> FV3HydrostaticState:
    """Convert A-grid HydrostaticState to FV3 D-grid state."""
    from legoesm.core.operators_cdgrid import agrid_to_dgrid_vector
    u_d, v_d = agrid_to_dgrid_vector(state.u.data, state.v.data, cdgrid)
    return FV3HydrostaticState(
        u_d=state.u.replace(data=u_d, name="u_d"),
        v_d=state.v.replace(data=v_d, name="v_d"),
        T=state.T,
        p_s=state.p_s,
        phis=state.phis,
        tracers=state.tracers,
    )


def fv3_to_hydrostatic(
    state: FV3HydrostaticState,
    cdgrid: CubedSphereCDGrid,
) -> HydrostaticState:
    """Convert FV3 D-grid state to A-grid HydrostaticState."""
    u_a = _interp_corner_to_center(state.u_d.data)
    v_a = _interp_corner_to_center(state.v_d.data)
    return HydrostaticState(
        u=state.u_d.replace(data=u_a, name="u"),
        v=state.v_d.replace(data=v_a, name="v"),
        T=state.T,
        p_s=state.p_s,
        phis=state.phis,
        tracers=state.tracers,
    )


# ==============================================================================
# Model class
# ==============================================================================

class CDGridPrimitiveEquationModel(IntegrationMixin):
    """Hydrostatic PE model on the cubed-sphere with FV3 C-D grid dynamics.

    Prognostic winds live on the D-grid (cell corners).  The model
    accepts and returns ``FV3HydrostaticState`` from ``step()``.

    For backward compatibility with code that passes ``HydrostaticState``,
    use the ``step_agrid()`` method or the ``hydrostatic_to_fv3`` /
    ``fv3_to_hydrostatic`` adapters.

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
        state,
        physics_tendency=None,
    ):
        """Compute tendencies. Accepts FV3HydrostaticState or HydrostaticState."""
        if isinstance(state, FV3HydrostaticState):
            return fv3_hydrostatic_tendencies(
                state, self.grid, self.sigma_coord, self.cdgrid,
                self.config, physics_tendency,
            )
        # Legacy A-grid path: convert → compute → convert back
        return cdgrid_hydrostatic_tendencies(
            state, self.grid, self.sigma_coord, self.cdgrid,
            self.config, physics_tendency,
        )

    def step(self, state, dt, physics_fn=None):
        """Advance one time step.

        Accepts both ``FV3HydrostaticState`` (D-grid) and
        ``HydrostaticState`` (A-grid, legacy).  When given A-grid
        state, converts to D-grid at the boundary, runs the D-grid
        dycore, and converts back.

        Parameters
        ----------
        state : FV3HydrostaticState or HydrostaticState
        dt : float
        physics_fn : callable, optional

        Returns
        -------
        Same type as input state.
        """
        # Precompute target mass outside JIT boundary (host-side only).
        # This avoids writing traced values into persistent object attributes
        # inside a jit-compiled method.
        if (self.config.use_conservation_fixer and self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and self._target_mass is None):
            from legoesm.core.operators import global_integral
            self._target_mass = global_integral(state.p_s, self.grid)

        if isinstance(state, FV3HydrostaticState):
            return self._step_fv3(state, dt, physics_fn=physics_fn)
        # Legacy A-grid path
        return self._step_agrid(state, dt, physics_fn=physics_fn)

    @partial(jax.jit, static_argnums=(0, 3))
    def _step_fv3(
        self,
        state: FV3HydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> FV3HydrostaticState:
        """Advance one time step with D-grid prognostic winds."""
        from legoesm.core.operators_cdgrid import agrid_to_dgrid_vector

        cdgrid = self.cdgrid

        def tendency_fn(s):
            phys_tend_dgrid = None
            if physics_fn is not None:
                # Convert D-grid state to A-grid for physics
                s_agrid = fv3_to_hydrostatic(s, cdgrid)
                _phys_result = physics_fn(s_agrid, self.grid, self.sigma_coord)
                phys_agrid = _phys_result[0] if type(_phys_result) is tuple else _phys_result
                # Convert A-grid physics tendencies to D-grid
                pu_d, pv_d = agrid_to_dgrid_vector(
                    phys_agrid.du_dt.data, phys_agrid.dv_dt.data, cdgrid,
                )
                phys_tend_dgrid = FV3HydrostaticTendencies(
                    du_d_dt=phys_agrid.du_dt.replace(data=pu_d, name="du_d_dt"),
                    dv_d_dt=phys_agrid.dv_dt.replace(data=pv_d, name="dv_d_dt"),
                    dT_dt=phys_agrid.dT_dt,
                    dp_s_dt=phys_agrid.dp_s_dt,
                    dphis_dt=phys_agrid.dphis_dt,
                )

            tend = fv3_hydrostatic_tendencies(
                s, self.grid, self.sigma_coord, cdgrid,
                self.config, phys_tend_dgrid,
            )
            # Return an FV3HydrostaticState-shaped pytree with tendency data
            # so that the time integrator's tree_map works correctly.
            return FV3HydrostaticState(
                u_d=s.u_d.replace(data=tend.du_d_dt.data),
                v_d=s.v_d.replace(data=tend.dv_d_dt.data),
                T=s.T.replace(data=tend.dT_dt.data),
                p_s=s.p_s.replace(data=tend.dp_s_dt.data),
                phis=s.phis.replace(data=jnp.zeros_like(s.phis.data)),
            )

        state_new = dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

        # Implicit gravity wave damping — post-step Laplacian diffusion on p_s.
        if self.config.implicit_grav_wave_damping > 0:
            from legoesm.core.operators import laplacian_compact
            alpha = self.config.implicit_grav_wave_damping
            lap_ps = laplacian_compact(state_new.p_s.data, self.grid)
            p_s_damped = state_new.p_s.data + alpha * dt * lap_ps
            p_s_damped = jnp.maximum(p_s_damped, 100.0)
            state_new = state_new._replace(
                p_s=state_new.p_s.replace(data=p_s_damped),
            )

        # Conservation fixer (operates on p_s which is at cell centres)
        if self.config.use_conservation_fixer and self.config.fix_mass:
            if self.config.anchor_mass_to_initial:
                from legoesm.core.conservation import fix_mass_hydrostatic_target
                # _target_mass is precomputed in step() outside the JIT boundary.
                state_h = fv3_to_hydrostatic(state_new, cdgrid)
                state_h_fixed = fix_mass_hydrostatic_target(
                    state_h, self._target_mass, self.grid,
                )
                state_new = state_new._replace(p_s=state_h_fixed.p_s)
            else:
                from legoesm.core.conservation import fix_mass_hydrostatic
                state_h_new = fv3_to_hydrostatic(state_new, cdgrid)
                state_h_old = fv3_to_hydrostatic(state, cdgrid)
                state_h_fixed = fix_mass_hydrostatic(
                    state_h_new, state_h_old, self.grid,
                )
                state_new = state_new._replace(p_s=state_h_fixed.p_s)

        return state_new

    @partial(jax.jit, static_argnums=(0, 3))
    def _step_agrid(
        self,
        state: HydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> HydrostaticState:
        """Advance one step, accepting and returning A-grid state.

        This is a convenience wrapper for code that still works with
        ``HydrostaticState``.  The A->D->A conversion happens only at
        the outer boundary of the step; internally the dycore operates
        entirely on D-grid winds.
        """
        fv3_state = hydrostatic_to_fv3(state, self.cdgrid)
        fv3_new = self._step_fv3(fv3_state, dt, physics_fn=physics_fn)
        return fv3_to_hydrostatic(fv3_new, self.cdgrid)

    # Backward-compatible aliases
    step_agrid = _step_agrid

    def step_with_physics(self, state, dt, physics_fn=None):
        return self.step(state, dt, physics_fn=physics_fn)
