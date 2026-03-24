"""FV3 Hydrostatic Primitive Equations on the cubed-sphere (D-grid dynamics).

Prognostic winds are stored on the D-grid (cell corners). The C-grid
velocities are diagnosed for mass flux and kinetic energy computation.

Physics coupling converts D-grid to cell-centre at the interface boundary
only (diagnostic).

Operator staggering
-------------------
- Momentum:  D-grid prognostic, C-grid diagnostic (for KE / mass flux)
- Vorticity: cell centres (from D-grid circulation)
- Bernoulli / pressure gradient: Arakawa-Lamb gradient at D-grid corners
- Divergence: C-grid flux-form (exact mass conservation)
- Scalar diffusion: cell-centre (proper inter-face halo exchange)
- Wind diffusion: D-grid Laplacian with halo

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

    The prognostic momentum is stored at D-grid cell corners.  The C-grid
    velocities are diagnosed from the D-grid winds for mass flux and KE
    computation.

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

        # Vertical advection of D-grid winds: interpolate to cell centres,
        # compute vertical advection, interpolate back to D-grid corners.
        u_cc = _interp_corner_to_center(u_d)  # (6, n, n, nlev)
        v_cc = _interp_corner_to_center(v_d)
        vert_adv_u_cc = vertical_advection_hybrid(u_cc, mass_flux, p_s, sigma_coord)
        vert_adv_v_cc = vertical_advection_hybrid(v_cc, mass_flux, p_s, sigma_coord)
        vert_adv_u_d = _interp_center_to_corner(vert_adv_u_cc, cdgrid)
        vert_adv_v_d = _interp_center_to_corner(vert_adv_v_cc, cdgrid)

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

        # Vertical advection of D-grid winds via cell-centre interpolation
        u_cc = _interp_corner_to_center(u_d)
        v_cc = _interp_corner_to_center(v_d)
        vert_adv_u_cc = vertical_advection(u_cc, sigma_dot, sigma_coord)
        vert_adv_v_cc = vertical_advection(v_cc, sigma_dot, sigma_coord)
        vert_adv_u_d = _interp_center_to_corner(vert_adv_u_cc, cdgrid)
        vert_adv_v_d = _interp_center_to_corner(vert_adv_v_cc, cdgrid)

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
    # ln_ps gradient at cell centres for the v.grad(ln ps) correction
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
        # Temperature: cell-centre Laplacian (proper halo exchange)
        lap_T = _laplacian_compact_3d(T, grid)
        dT_dt_data = dT_dt_data + config.A_h * lap_T

    # 12b. Hyperdiffusion on D-grid winds (biharmonic)
    if config.hyperdiff_coeff > 0:
        du_d_dt = du_d_dt - config.hyperdiff_coeff * _laplacian_dgrid(
            _laplacian_dgrid(u_d, cdgrid), cdgrid)
        dv_d_dt = dv_d_dt - config.hyperdiff_coeff * _laplacian_dgrid(
            _laplacian_dgrid(v_d, cdgrid), cdgrid)
        # Temperature: cell-centre hyperdiffusion (proper halo exchange)
        dT_dt_data = dT_dt_data + _hyperdiffusion_3d(T, grid, config.hyperdiff_coeff)

    # Surface pressure hyperdiffusion (cell-centre)
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
# Adapter: D-grid to cell-centre conversion
# ==============================================================================

def fv3_to_hydrostatic(
    state: FV3HydrostaticState,
    cdgrid: CubedSphereCDGrid,
) -> HydrostaticState:
    """Convert FV3 D-grid state to cell-centre HydrostaticState.

    Uses corner-to-centre interpolation for the wind components.
    """
    u_cc = _interp_corner_to_center(state.u_d.data)
    v_cc = _interp_corner_to_center(state.v_d.data)
    return HydrostaticState(
        u=state.u_d.replace(data=u_cc, name="u"),
        v=state.v_d.replace(data=v_cc, name="v"),
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

    For backward compatibility with code that passes ``HydrostaticState``
    (cell-centre winds), use ``step_cell_centre()`` or ``fv3_to_hydrostatic``.

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

    def _sync_dgrid_boundary(self, state: FV3HydrostaticState):
        """Synchronize D-grid boundary corners across cubed-sphere faces.

        FV3-style direct corner-to-corner sync: convert D-grid corner
        velocities to geographic (east/north) at each shared face edge,
        average with the neighbouring face, then convert back to face-local.

        Edge corners (shared by 2 faces) get a pairwise average.
        Vertex corners (shared by 3 faces) get a 3-way average.
        A 3-point smoothing filter blends the synced boundary into the
        interior (first interior row, weight w=0.45).

        For 3D fields (6, n+1, n+1, nlev), the sync is vmapped over
        vertical levels.
        """
        from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH

        def _sync_2d(u_d_2d, v_d_2d):
            """Sync a single 2D level slice (6, n+1, n+1)."""
            n = self.grid.n
            ca_c = self.cdgrid.cos_angle_corner
            sa_c = self.cdgrid.sin_angle_corner

            # 1. Convert all corners to geographic (read-only reference)
            ue = ca_c * u_d_2d - sa_c * v_d_2d
            vn = sa_c * u_d_2d + ca_c * v_d_2d

            # 2. Edge sync: pairwise average with neighbour faces
            ue_out = ue
            vn_out = vn

            def _get_strip(arr, face, edge):
                if edge == WEST:    return arr[face, 0, :]
                elif edge == EAST:  return arr[face, n, :]
                elif edge == SOUTH: return arr[face, :, 0]
                else:               return arr[face, :, n]

            for face in range(6):
                for edge in [WEST, EAST, SOUTH, NORTH]:
                    nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]
                    nbr_ue = _get_strip(ue, nbr_face, nbr_edge)
                    nbr_vn = _get_strip(vn, nbr_face, nbr_edge)
                    if is_reversed:
                        nbr_ue = nbr_ue[::-1]
                        nbr_vn = nbr_vn[::-1]
                    local_ue = _get_strip(ue, face, edge)
                    local_vn = _get_strip(vn, face, edge)
                    avg_ue = 0.5 * (local_ue + nbr_ue)
                    avg_vn = 0.5 * (local_vn + nbr_vn)
                    if edge == WEST:
                        ue_out = ue_out.at[face, 0, :].set(avg_ue)
                        vn_out = vn_out.at[face, 0, :].set(avg_vn)
                    elif edge == EAST:
                        ue_out = ue_out.at[face, n, :].set(avg_ue)
                        vn_out = vn_out.at[face, n, :].set(avg_vn)
                    elif edge == SOUTH:
                        ue_out = ue_out.at[face, :, 0].set(avg_ue)
                        vn_out = vn_out.at[face, :, 0].set(avg_vn)
                    else:
                        ue_out = ue_out.at[face, :, n].set(avg_ue)
                        vn_out = vn_out.at[face, :, n].set(avg_vn)

            # 3. Vertex sync: 3-way average at cube vertices (8 vertices)
            _vtx = [
                [(0, 0, 0), (3, n, 0), (5, 0, n)],
                [(0, n, 0), (1, 0, 0), (5, n, n)],
                [(0, 0, n), (3, n, n), (4, 0, 0)],
                [(0, n, n), (1, 0, n), (4, n, 0)],
                [(1, n, 0), (2, 0, 0), (5, n, 0)],
                [(1, n, n), (2, 0, n), (4, n, n)],
                [(2, n, 0), (3, 0, 0), (5, 0, 0)],
                [(2, n, n), (3, 0, n), (4, 0, n)],
            ]
            for vtx in _vtx:
                ue_avg = sum(ue[f, i, j] for f, i, j in vtx) / 3.0
                vn_avg = sum(vn[f, i, j] for f, i, j in vtx) / 3.0
                for f, i, j in vtx:
                    ue_out = ue_out.at[f, i, j].set(ue_avg)
                    vn_out = vn_out.at[f, i, j].set(vn_avg)

            # 4. Convert back to face-local (no boundary smoothing needed
            #    with FV3-faithful d2a2c non-orthogonality correction)
            u_d_new = ca_c * ue_out + sa_c * vn_out
            v_d_new = -sa_c * ue_out + ca_c * vn_out
            return u_d_new, v_d_new

        u_d = state.u_d.data  # (6, n+1, n+1, nlev)
        v_d = state.v_d.data

        # vmap the 2D sync over vertical levels
        # Reshape: (6, n+1, n+1, nlev) → (nlev, 6, n+1, n+1)
        u_transposed = jnp.moveaxis(u_d, -1, 0)
        v_transposed = jnp.moveaxis(v_d, -1, 0)
        u_synced, v_synced = jax.vmap(_sync_2d)(u_transposed, v_transposed)
        # Reshape back: (nlev, 6, n+1, n+1) → (6, n+1, n+1, nlev)
        u_d_new = jnp.moveaxis(u_synced, 0, -1)
        v_d_new = jnp.moveaxis(v_synced, 0, -1)

        return state._replace(
            u_d=state.u_d.replace(data=u_d_new),
            v_d=state.v_d.replace(data=v_d_new),
        )

    def tendencies(
        self,
        state,
        physics_tendency=None,
    ):
        """Compute tendencies for FV3HydrostaticState."""
        return fv3_hydrostatic_tendencies(
            state, self.grid, self.sigma_coord, self.cdgrid,
            self.config, physics_tendency,
        )

    def step(self, state, dt, physics_fn=None):
        """Advance one time step.

        Accepts ``FV3HydrostaticState`` (D-grid prognostic winds).
        For legacy ``HydrostaticState`` input, uses the cell-centre
        adapter path.

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
        # Legacy cell-centre state path
        return self._step_cell_centre(state, dt, physics_fn=physics_fn)

    @partial(jax.jit, static_argnums=(0, 3))
    def _step_fv3(
        self,
        state: FV3HydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> FV3HydrostaticState:
        """Advance one time step with D-grid prognostic winds."""
        cdgrid = self.cdgrid

        def tendency_fn(s):
            phys_tend_dgrid = None
            if physics_fn is not None:
                # Convert D-grid state to cell-centre for physics
                s_cc = fv3_to_hydrostatic(s, cdgrid)
                _phys_result = physics_fn(s_cc, self.grid, self.sigma_coord)
                phys_cc = _phys_result[0] if type(_phys_result) is tuple else _phys_result
                # Convert cell-centre physics tendencies to D-grid corners
                pu_d = _interp_center_to_corner(phys_cc.du_dt.data, cdgrid)
                pv_d = _interp_center_to_corner(phys_cc.dv_dt.data, cdgrid)
                phys_tend_dgrid = FV3HydrostaticTendencies(
                    du_d_dt=phys_cc.du_dt.replace(data=pu_d, name="du_d_dt"),
                    dv_d_dt=phys_cc.dv_dt.replace(data=pv_d, name="dv_d_dt"),
                    dT_dt=phys_cc.dT_dt,
                    dp_s_dt=phys_cc.dp_s_dt,
                    dphis_dt=phys_cc.dphis_dt,
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

        # Synchronize D-grid boundary corners across cubed-sphere faces
        state_new = self._sync_dgrid_boundary(state_new)

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
    def _step_cell_centre(
        self,
        state: HydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> HydrostaticState:
        """Advance one step, accepting and returning cell-centre state.

        This is a convenience wrapper for code that still works with
        ``HydrostaticState``.  Cell-centre winds are interpolated to
        D-grid corners at entry and back to cell centres at exit;
        internally the dycore operates entirely on D-grid winds.
        """
        u_d = _interp_center_to_corner(state.u.data, self.cdgrid)
        v_d = _interp_center_to_corner(state.v.data, self.cdgrid)
        fv3_state = FV3HydrostaticState(
            u_d=state.u.replace(data=u_d, name="u_d"),
            v_d=state.v.replace(data=v_d, name="v_d"),
            T=state.T,
            p_s=state.p_s,
            phis=state.phis,
            tracers=state.tracers,
        )
        fv3_new = self._step_fv3(fv3_state, dt, physics_fn=physics_fn)
        return fv3_to_hydrostatic(fv3_new, self.cdgrid)

    # Backward-compatible aliases
    step_cell_centre = _step_cell_centre

    def step_with_physics(self, state, dt, physics_fn=None):
        return self.step(state, dt, physics_fn=physics_fn)


# ==============================================================================
# Backward-compatible aliases referenced by __init__.py
# ==============================================================================

def cdgrid_hydrostatic_tendencies(
    state,
    grid: CubedSphereGrid,
    sigma_coord,
    cdgrid: CubedSphereCDGrid,
    config: CDGridPrimitiveEquationConfig = CDGridPrimitiveEquationConfig(),
    physics_tendency=None,
):
    """Compute hydrostatic tendencies, accepting either HydrostaticState or FV3HydrostaticState.

    If given a HydrostaticState (cell-centre winds), converts to D-grid internally,
    calls fv3_hydrostatic_tendencies, and returns HydrostaticTendencies (cell-centre).

    If given a FV3HydrostaticState, delegates directly to fv3_hydrostatic_tendencies.
    """
    from legoesm.core.state import HydrostaticTendencies

    if isinstance(state, FV3HydrostaticState):
        return fv3_hydrostatic_tendencies(state, grid, sigma_coord, cdgrid, config, physics_tendency)

    # HydrostaticState path: convert cell-centre -> D-grid
    u_d = _interp_center_to_corner(state.u.data, cdgrid)
    v_d = _interp_center_to_corner(state.v.data, cdgrid)
    fv3_state = FV3HydrostaticState(
        u_d=state.u.replace(data=u_d, name="u_d"),
        v_d=state.v.replace(data=v_d, name="v_d"),
        T=state.T,
        p_s=state.p_s,
        phis=state.phis,
        tracers=getattr(state, 'tracers', None),
    )
    fv3_tend = fv3_hydrostatic_tendencies(fv3_state, grid, sigma_coord, cdgrid, config, physics_tendency)

    # Convert D-grid tendencies back to cell-centre
    du_cc = _interp_corner_to_center(fv3_tend.du_d_dt.data)
    dv_cc = _interp_corner_to_center(fv3_tend.dv_d_dt.data)

    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")
    return HydrostaticTendencies(
        du_dt=Field(data=du_cc, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_cc, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dT_dt=fv3_tend.dT_dt,
        dp_s_dt=fv3_tend.dp_s_dt,
        dphis_dt=fv3_tend.dphis_dt,
    )


def hydrostatic_to_fv3(
    state: HydrostaticState,
    cdgrid: CubedSphereCDGrid,
) -> FV3HydrostaticState:
    """Convert cell-centre HydrostaticState to FV3 D-grid state.

    Uses centre-to-corner interpolation for the wind components.
    """
    u_d = _interp_center_to_corner(state.u.data, cdgrid)
    v_d = _interp_center_to_corner(state.v.data, cdgrid)
    return FV3HydrostaticState(
        u_d=state.u.replace(data=u_d, name="u_d"),
        v_d=state.v.replace(data=v_d, name="v_d"),
        T=state.T,
        p_s=state.p_s,
        phis=state.phis,
        tracers=getattr(state, 'tracers', None),
    )
