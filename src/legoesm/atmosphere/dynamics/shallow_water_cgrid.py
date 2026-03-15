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
from legoesm.core.operators_fv_cubed import (
    fv_divergence_damping,
    face_boundary_weight,
    edge_blend_scalar,
    edge_blend_vector,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm import constants


class CGShallowWaterCubedConfig(NamedTuple):
    """Configuration for the C-grid shallow water model on cubed-sphere."""
    g: float = constants.g
    div_damp_2: float = 0.0       # 2nd-order divergence damping [m²/s]
    div_damp_4: float = 0.0       # 4th-order divergence damping [m⁴/s]
    hyperdiff_coeff: float = 0.0  # Velocity hyperdiffusion (backup)
    edge_blend_strength: float = 0.0   # Face-boundary blend (0=off)
    edge_blend_depth: int = 0          # Rows to blend near each edge
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    fix_energy: bool = True
    time_integrator: str = "ssp_rk3"


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

    # --- 5. Divergence damping (FV-consistent operator) ---
    if config.div_damp_2 > 0 or config.div_damp_4 > 0:
        du_damp, dv_damp = fv_divergence_damping(
            u.data, v.data, grid,
            config.div_damp_2, config.div_damp_4,
        )
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

class CGShallowWaterCubedModel(IntegrationMixin):
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

        # Precompute edge-blend weight (static, not JIT-traced)
        cfg = self.config
        if cfg.edge_blend_strength > 0 and cfg.edge_blend_depth > 0:
            self._eb_weight = face_boundary_weight(
                grid.n, cfg.edge_blend_depth, cfg.edge_blend_strength,
            )
        else:
            self._eb_weight = None

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

        state_new = dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

        # Edge blending: localized smoothing near face boundaries
        if self._eb_weight is not None:
            h_new = edge_blend_scalar(
                state_new.h.data, self.grid, self._eb_weight,
            )
            u_new, v_new = edge_blend_vector(
                state_new.u.data, state_new.v.data,
                self.grid, self._eb_weight,
            )
            state_new = ShallowWaterState(
                h=state_new.h.replace(data=h_new),
                u=state_new.u.replace(data=u_new),
                v=state_new.v.replace(data=v_new),
                h_s=state_new.h_s,
            )

        if self.config.use_conservation_fixer:
            state_new = apply_conservation_fixer(
                state_new, state, self.grid,
                fix_mass=self.config.fix_mass,
                fix_energy=self.config.fix_energy,
                g=self.config.g,
            )

        return state_new

