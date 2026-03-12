"""Consistent FV Shallow Water Equations on the cubed-sphere.

Uses a compatible face-based discretization where mass, momentum, and
pressure gradient share the same interface-flux layer:

    dh/dt = fv_flux_divergence(h, u, v)      [FV PPM mass continuity]
    du/dt =  (zeta + f) * v - dB/dx + D_u    [vector-invariant momentum]
    dv/dt = -(zeta + f) * u - dB/dy + D_v

Key design choices for consistency:
- Mass transport: unsplit PPM (both x and y on same field)
- Bernoulli gradient: PPM-compatible 4th-order edge values (fv_gradient_x/y)
- Vorticity: centered curl (appropriate for vector-invariant form)
- Divergence damping: selective damping of divergent modes using the
  SAME FV divergence operator as the mass flux, preventing cube-imprinted
  computational modes from growing
- Hyperdiffusion: on velocity only (PPM handles scalar dissipation)
- Edge blending: localized Laplacian smoothing near face boundaries,
  applied as a post-step filter to damp accumulated halo-interpolation
  errors that are spatially coherent at face edges

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
from legoesm.core.state import ShallowWaterState, ShallowWaterTendencies
from legoesm.core.operators import (
    curl_z,
    hyperdiffusion,
)
from legoesm.core.operators_fv import (
    fv_flux_divergence,
    fv_gradient_x,
    fv_gradient_y,
)
from legoesm.core.operators_fv_cubed import (
    fv_divergence_damping,
    face_boundary_weight,
    edge_blend_scalar,
    edge_blend_vector,
)
from legoesm.core.conservation import (
    apply_conservation_fixer,
    zero_mean_tendency,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm.timestepping.ssp_rk34 import ssp_rk34_step
from legoesm.timestepping.ssp_rk54 import ssp_rk54_step
from legoesm import constants


class FVShallowWaterConfig(NamedTuple):
    """Configuration for the FV shallow-water model.

    Divergence damping (div_damp_2, div_damp_4) is the primary mechanism
    for controlling cube-imprinted computational divergent modes. Set both
    to 0 only for diagnostic/debugging runs.

    Edge blending (edge_blend_strength > 0) applies localized Laplacian
    smoothing near face boundaries after each time step.  This damps the
    spatially-coherent errors from halo interpolation that accumulate at
    face edges over many steps.
    """
    g: float = constants.g
    div_damp_2: float = 0.0      # 2nd-order divergence damping [m²/s]
    div_damp_4: float = 0.0      # 4th-order divergence damping [m⁴/s]
    hyperdiff_coeff: float = 0.0 # Only on u,v (PPM handles h dissipation)
    edge_blend_strength: float = 0.25  # Face-boundary blend (0=off)
    edge_blend_depth: int = 3          # Rows to blend near each edge
    use_conservation_fixer: bool = True
    fix_mass: bool = True
    fix_energy: bool = False     # OFF by default for benchmark runs
    time_integrator: str = "ssp_rk3"
    use_limiter: bool = True


def fv_shallow_water_tendencies(
    state: ShallowWaterState,
    grid: CubedSphereGrid,
    config: FVShallowWaterConfig = FVShallowWaterConfig(),
) -> ShallowWaterTendencies:
    """Compute tendencies for the consistent FV shallow water equations.

    Mass uses unsplit PPM, Bernoulli gradient uses PPM-compatible
    reconstruction, and divergence damping uses the same FV divergence
    as the mass flux.

    Parameters
    ----------
    state : ShallowWaterState
    grid : CubedSphereGrid
    config : FVShallowWaterConfig

    Returns
    -------
    ShallowWaterTendencies
    """
    h = state.h
    u = state.u
    v = state.v
    h_s = state.h_s
    g = config.g

    # --- Mass continuity: dh/dt via FV PPM transport ---
    dh_dt_data = fv_flux_divergence(
        h.data, u.data, v.data, grid,
        limiter=config.use_limiter,
    )
    # Conservation cleanup
    dh_dt_data = zero_mean_tendency(dh_dt_data, grid)

    dh_dt = Field(
        data=dh_dt_data,
        name="dh_dt", dims=h.dims, units="m/s",
    )

    # --- Relative vorticity: zeta = dv/dx - du/dy ---
    zeta = curl_z(u, v, grid).data

    # --- Absolute vorticity: zeta + f ---
    abs_vor = zeta + grid.f

    # --- Bernoulli function: B = K + g*(h + h_s) ---
    kinetic_energy = 0.5 * (u.data**2 + v.data**2)
    bernoulli_data = kinetic_energy + g * (h.data + h_s.data)

    # PPM-compatible gradients (same reconstruction as mass flux)
    dB_dx_data = fv_gradient_x(bernoulli_data, grid)
    dB_dy_data = fv_gradient_y(bernoulli_data, grid)

    # --- Vector-invariant momentum equations ---
    du_dt_data = abs_vor * v.data - dB_dx_data
    dv_dt_data = -abs_vor * u.data - dB_dy_data

    # --- Divergence damping (primary stabilization) ---
    if config.div_damp_2 > 0 or config.div_damp_4 > 0:
        du_damp, dv_damp = fv_divergence_damping(
            u.data, v.data, grid,
            config.div_damp_2, config.div_damp_4,
        )
        du_dt_data = du_dt_data + du_damp
        dv_dt_data = dv_dt_data + dv_damp

    # --- Hyperdiffusion on velocity only (secondary) ---
    if config.hyperdiff_coeff > 0:
        diff_u = hyperdiffusion(u, grid, config.hyperdiff_coeff)
        diff_v = hyperdiffusion(v, grid, config.hyperdiff_coeff)
        du_dt_data = du_dt_data + diff_u.data
        dv_dt_data = dv_dt_data + diff_v.data

    du_dt = Field(data=du_dt_data, name="du_dt", dims=u.dims, units="m/s^2")
    dv_dt = Field(data=dv_dt_data, name="dv_dt", dims=v.dims, units="m/s^2")

    return ShallowWaterTendencies(dh_dt=dh_dt, du_dt=du_dt, dv_dt=dv_dt)


class FVShallowWaterModel:
    """Consistent FV shallow water model on the cubed-sphere.

    Uses PPM mass transport, PPM-compatible Bernoulli gradient,
    and FV-consistent divergence damping. Does NOT rely on edge
    blending or excessive hyperdiffusion for stabilization.

    Parameters
    ----------
    grid : CubedSphereGrid
    config : FVShallowWaterConfig, optional
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        config: FVShallowWaterConfig | None = None,
    ):
        self.grid = grid
        self.config = config or FVShallowWaterConfig()

        # Precompute edge-blend weight (static, not JIT-traced)
        cfg = self.config
        if cfg.edge_blend_strength > 0 and cfg.edge_blend_depth > 0:
            self._eb_weight = face_boundary_weight(
                grid.n, cfg.edge_blend_depth, cfg.edge_blend_strength,
            )
        else:
            self._eb_weight = None

    def tendencies(
        self,
        state: ShallowWaterState,
    ) -> ShallowWaterTendencies:
        """Compute tendencies (pure function wrapper)."""
        return fv_shallow_water_tendencies(
            state, self.grid, self.config,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: ShallowWaterState, dt: float) -> ShallowWaterState:
        """Advance one time step using SSP-RK3 with PPM transport.

        Parameters
        ----------
        state : ShallowWaterState
        dt : float
            Time step [seconds].

        Returns
        -------
        ShallowWaterState
        """
        def tendency_fn(s):
            tend = fv_shallow_water_tendencies(
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

        # Apply conservation fixers
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
        """Integrate forward for a given duration.

        Parameters
        ----------
        state : ShallowWaterState
        duration : float
            Total integration time [seconds].
        dt : float
            Time step [seconds].
        save_every : int
            Save state every N steps.

        Returns
        -------
        final_state : ShallowWaterState
        trajectory : list of ShallowWaterState
        """
        n_steps = int(duration / dt)
        trajectory = [state]

        for i in range(n_steps):
            state = self.step(state, dt)
            if (i + 1) % save_every == 0:
                trajectory.append(state)

        return state, trajectory

    def integrate_scan(
        self,
        state: ShallowWaterState,
        n_steps: int,
        dt: float,
    ) -> tuple[ShallowWaterState, ShallowWaterState]:
        """Integrate using jax.lax.scan (differentiable, JIT-friendly).

        Parameters
        ----------
        state : ShallowWaterState
        n_steps : int
        dt : float

        Returns
        -------
        final_state : ShallowWaterState
        trajectory : ShallowWaterState
            All intermediate states (each leaf: (n_steps, 6, n, n)).
        """
        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(
            scan_fn, state, jnp.arange(n_steps)
        )
        return final_state, trajectory
