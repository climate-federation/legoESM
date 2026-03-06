"""FV Shallow Water Equations on the cubed-sphere with divergence damping.

Extends the A-grid FV solver (PPM mass transport + vector-invariant
momentum) with C-grid-style divergence damping. Uses the same proven
unsplit PPM mass transport and PPM-compatible Bernoulli gradient as
shallow_water_fv.py.

The key addition is divergence damping (2nd + 4th order), which
selectively dissipates divergent modes while preserving rotational
flow. This is the damping strategy used by FV3 on its C-D grid.

Note: true C-grid staggering on the cubed-sphere requires staggered
halo exchange at face boundaries — a major implementation effort.
This solver instead uses A-grid storage with divergence damping,
providing the damping benefits of FV3 without staggered storage.

References
----------
- Lin & Rood (1997): An explicit flux-form semi-Lagrangian SWE model
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core (FV3)
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState, ShallowWaterTendencies
from legoesm.core.conservation import apply_conservation_fixer, zero_mean_tendency
from legoesm.core.operators import curl_z
from legoesm.core.operators_fv import fv_flux_divergence, fv_gradient_x, fv_gradient_y
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import pad_halo, pad_halo_vector
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm.timestepping.ssp_rk34 import ssp_rk34_step
from legoesm.timestepping.ssp_rk54 import ssp_rk54_step
from legoesm import constants


class CGShallowWaterCubedConfig(NamedTuple):
    """Configuration for the C-grid shallow water model on cubed-sphere."""
    g: float = constants.g
    div_damp_2: float = 0.0       # 2nd-order divergence damping
    div_damp_4: float = 0.0       # 4th-order divergence damping
    hyperdiff_coeff: float = 0.0  # Velocity hyperdiffusion (backup)
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    fix_energy: bool = True
    time_integrator: str = "ssp_rk3"


# ==============================================================================
# C-grid operators on cubed-sphere
# ==============================================================================

def _cgrid_divergence(u_pad, v_pad, grid):
    """Compute divergence at cell centers using C-grid finite differences.

    Uses the padded velocity field with half-cell metrics.
    """
    hx = grid.hx_ext  # (6, n+2, n+2)
    hy = grid.hy_ext

    # Flux at x-edges: u * hy
    flux_x = u_pad * hy
    # Flux at y-edges: v * hx
    flux_y = v_pad * hx

    # Net flux: right edge - left edge for x, top - bottom for y
    # For cell at padded index (i, j) → interior index (1..n, 1..n)
    # Right x-edge is between (i, j) and (i+1, j): flux_x at (i+0.5, j)
    # We approximate as average: 0.5*(flux_x[i,j] + flux_x[i+1,j])
    # Left x-edge: 0.5*(flux_x[i-1,j] + flux_x[i,j])
    # Net = right - left = 0.5*(flux_x[i+1,j] - flux_x[i-1,j])
    d_flux_x = flux_x[:, 2:, 1:-1] - flux_x[:, :-2, 1:-1]
    d_flux_y = flux_y[:, 1:-1, 2:] - flux_y[:, 1:-1, :-2]

    return (d_flux_x + d_flux_y) / (2.0 * grid.area)


def _cgrid_divergence_damping(u_data, v_data, grid, config):
    """Apply divergence damping to momentum tendencies.

    2nd order: +nu2 * grad(div)
    4th order: -nu4 * grad(lap(div))
    """
    du_damp = jnp.zeros_like(u_data)
    dv_damp = jnp.zeros_like(v_data)

    if config.div_damp_2 <= 0 and config.div_damp_4 <= 0:
        return du_damp, dv_damp

    # Compute divergence
    u_pad, v_pad = pad_halo_vector(
        u_data, v_data,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded, grid.sin_angle_padded,
        interp_offsets=grid.halo_interp_offsets,
    )
    div = _cgrid_divergence(u_pad, v_pad, grid)  # (6, n, n)

    if config.div_damp_2 > 0:
        # grad(div) using centered differences
        div_pad = pad_halo(div, interp_offsets=grid.halo_interp_offsets)
        grad_div_x = (div_pad[:, 2:, 1:-1] - div_pad[:, :-2, 1:-1]) / grid.dx
        grad_div_y = (div_pad[:, 1:-1, 2:] - div_pad[:, 1:-1, :-2]) / grid.dy
        du_damp = du_damp + config.div_damp_2 * grad_div_x
        dv_damp = dv_damp + config.div_damp_2 * grad_div_y

    if config.div_damp_4 > 0:
        # Laplacian of divergence
        div_pad = pad_halo(div, interp_offsets=grid.halo_interp_offsets)
        lap_div = (
            (div_pad[:, 2:, 1:-1] - 2*div_pad[:, 1:-1, 1:-1] + div_pad[:, :-2, 1:-1])
            / (grid.dx / 2)**2
            + (div_pad[:, 1:-1, 2:] - 2*div_pad[:, 1:-1, 1:-1] + div_pad[:, 1:-1, :-2])
            / (grid.dy / 2)**2
        )
        # grad(lap(div))
        lap_div_pad = pad_halo(lap_div, interp_offsets=grid.halo_interp_offsets)
        grad_lap_x = (lap_div_pad[:, 2:, 1:-1] - lap_div_pad[:, :-2, 1:-1]) / grid.dx
        grad_lap_y = (lap_div_pad[:, 1:-1, 2:] - lap_div_pad[:, 1:-1, :-2]) / grid.dy
        du_damp = du_damp - config.div_damp_4 * grad_lap_x
        dv_damp = dv_damp - config.div_damp_4 * grad_lap_y

    return du_damp, dv_damp


# ==============================================================================
# Tendencies
# ==============================================================================

def cgrid_shallow_water_tendencies_cubed(
    state: ShallowWaterState,
    grid: CubedSphereGrid,
    config: CGShallowWaterCubedConfig = CGShallowWaterCubedConfig(),
) -> ShallowWaterTendencies:
    """Compute C-grid shallow water tendencies on cubed-sphere.

    Parameters
    ----------
    state : ShallowWaterState
    grid : CubedSphereGrid
    config : CGShallowWaterCubedConfig

    Returns
    -------
    ShallowWaterTendencies
    """
    h = state.h
    u = state.u
    v = state.v
    h_s = state.h_s
    g = config.g

    # --- 1. Mass continuity: proven unsplit PPM transport ---
    dh_dt_data = fv_flux_divergence(h.data, u.data, v.data, grid, limiter=True)
    dh_dt_data = zero_mean_tendency(dh_dt_data, grid)

    # --- 2. Vorticity ---
    zeta = curl_z(u, v, grid).data
    abs_vor = zeta + grid.f

    # --- 3. Bernoulli gradient (PPM-compatible, same as FV solver) ---
    KE = 0.5 * (u.data**2 + v.data**2)
    B = KE + g * (h.data + h_s.data)
    dBdx = fv_gradient_x(B, grid)
    dBdy = fv_gradient_y(B, grid)

    # --- 4. Momentum tendencies ---
    du_dt_data = abs_vor * v.data - dBdx
    dv_dt_data = -abs_vor * u.data - dBdy

    # --- 5. Divergence damping ---
    du_damp, dv_damp = _cgrid_divergence_damping(u.data, v.data, grid, config)
    du_dt_data = du_dt_data + du_damp
    dv_dt_data = dv_dt_data + dv_damp

    # --- 6. Hyperdiffusion (optional) ---
    if config.hyperdiff_coeff > 0:
        from legoesm.core.operators import hyperdiffusion
        diff_u = hyperdiffusion(u, grid, config.hyperdiff_coeff)
        diff_v = hyperdiffusion(v, grid, config.hyperdiff_coeff)
        du_dt_data = du_dt_data + diff_u.data
        dv_dt_data = dv_dt_data + diff_v.data

    dims = state.h.dims
    return ShallowWaterTendencies(
        dh_dt=Field(data=dh_dt_data, name="dh_dt", dims=dims, units="m/s"),
        du_dt=Field(data=du_dt_data, name="du_dt", dims=dims, units="m/s^2"),
        dv_dt=Field(data=dv_dt_data, name="dv_dt", dims=dims, units="m/s^2"),
    )


# ==============================================================================
# Model class
# ==============================================================================

class CGShallowWaterCubedModel:
    """C-grid shallow water model on the cubed-sphere.

    Uses C-grid-style operators (exact pressure gradient, divergence
    damping, PPM mass transport) with A-grid state storage for
    compatibility with the existing cubed-sphere halo exchange.

    Parameters
    ----------
    grid : CubedSphereGrid
    config : CGShallowWaterCubedConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        config: CGShallowWaterCubedConfig | None = None,
    ):
        self.grid = grid
        self.config = config or CGShallowWaterCubedConfig()

    def tendencies(self, state: ShallowWaterState) -> ShallowWaterTendencies:
        """Compute tendencies (pure function wrapper)."""
        return cgrid_shallow_water_tendencies_cubed(
            state, self.grid, self.config,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: ShallowWaterState, dt: float) -> ShallowWaterState:
        """Advance one time step."""
        def tendency_fn(s):
            tend = cgrid_shallow_water_tendencies_cubed(
                s, self.grid, self.config,
            )
            return ShallowWaterState(
                h=s.h.replace(data=tend.dh_dt.data),
                u=s.u.replace(data=tend.du_dt.data),
                v=s.v.replace(data=tend.dv_dt.data),
                h_s=s.h_s.replace(data=jnp.zeros_like(s.h_s.data)),
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

        if self.config.use_conservation_fixer:
            state_new = apply_conservation_fixer(
                state_new, state, self.grid,
                fix_mass=self.config.fix_mass,
                fix_energy=self.config.fix_energy,
                g=self.config.g,
            )

        return state_new

    def integrate(
        self,
        state: ShallowWaterState,
        duration: float,
        dt: float,
        save_every: int = 1,
    ) -> tuple[ShallowWaterState, list[ShallowWaterState]]:
        """Integrate forward for a given duration."""
        n_steps = int(duration / dt)
        trajectory = [state]
        for i in range(n_steps):
            state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)
        return state, trajectory
