"""Shallow Water Equations on the cubed-sphere.

The rotating shallow water equations in **vector-invariant form**:

    dh/dt = -div(h * v)                          [mass continuity]
    du/dt =  (zeta + f) * v - dB/dx + D_u        [x-momentum]
    dv/dt = -(zeta + f) * u - dB/dy + D_v        [y-momentum]

where:
    h      = fluid depth
    u, v   = velocity components (grid-aligned)
    zeta   = relative vorticity (dv/dx - du/dy)
    f      = Coriolis parameter (2*Omega*sin(lat))
    B      = Bernoulli function = K + g*(h + h_s)
    K      = kinetic energy per unit mass = 0.5*(u^2 + v^2)
    g      = gravitational acceleration
    h_s    = surface topography
    D_u, D_v = diffusion/hyperdiffusion terms

The vector-invariant form avoids explicit momentum advection, which:
- Eliminates the need for vector halo exchange in advection operators
- Conserves energy and potential enstrophy in the continuous limit
- Only requires scalar gradients (of B) across face boundaries

References
----------
- Williamson et al. (1992): A standard test set for numerical approximations
  to the shallow water equations in spherical geometry.
- Sadourny (1975): The dynamics of finite-difference models of the
  shallow water equations.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState, ShallowWaterTendencies
from legoesm.core.operators import (
    gradient_x,
    gradient_y,
    divergence,
    curl_z,
    hyperdiffusion,
)
from legoesm.core.conservation import apply_conservation_fixer
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import CONNECTIVITY, WEST, EAST, SOUTH, NORTH
from legoesm.timestepping.ssp_rk3 import ssp_rk3_step
from legoesm.timestepping.ssp_rk34 import ssp_rk34_step
from legoesm.timestepping.ssp_rk54 import ssp_rk54_step
from legoesm import constants


def _build_unique_edge_pairs() -> tuple[tuple[int, int, int, int, bool], ...]:
    pairs: list[tuple[int, int, int, int, bool]] = []
    seen = set()
    for face in range(6):
        for edge, (nbr_face, nbr_edge, reversed_idx) in CONNECTIVITY[face].items():
            key = tuple(sorted(((face, edge), (nbr_face, nbr_edge))))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((face, edge, nbr_face, nbr_edge, reversed_idx))
    return tuple(pairs)


_UNIQUE_EDGE_PAIRS = _build_unique_edge_pairs()


def _edge_strip(arr: jax.Array, face: int, edge: int) -> jax.Array:
    if edge == WEST:
        return arr[face, 0, :]
    if edge == EAST:
        return arr[face, -1, :]
    if edge == SOUTH:
        return arr[face, :, 0]
    if edge == NORTH:
        return arr[face, :, -1]
    raise ValueError(f"Unknown edge: {edge}")


def _set_edge_strip(arr: jax.Array, face: int, edge: int, strip: jax.Array) -> jax.Array:
    if edge == WEST:
        return arr.at[face, 0, :].set(strip)
    if edge == EAST:
        return arr.at[face, -1, :].set(strip)
    if edge == SOUTH:
        return arr.at[face, :, 0].set(strip)
    if edge == NORTH:
        return arr.at[face, :, -1].set(strip)
    raise ValueError(f"Unknown edge: {edge}")


def _blend_scalar_cube_edges(arr: jax.Array, strength: float) -> jax.Array:
    """Relax opposite face-edge values toward their shared mean."""
    out = arr
    w = jnp.asarray(strength, dtype=arr.dtype)
    one_minus_w = 1.0 - w

    for face, edge, nbr_face, nbr_edge, reversed_idx in _UNIQUE_EDGE_PAIRS:
        a = _edge_strip(out, face, edge)
        b = _edge_strip(out, nbr_face, nbr_edge)
        if reversed_idx:
            b = b[::-1]

        avg = 0.5 * (a + b)
        a_new = one_minus_w * a + w * avg
        b_new = one_minus_w * b + w * avg

        if reversed_idx:
            b_new = b_new[::-1]

        out = _set_edge_strip(out, face, edge, a_new)
        out = _set_edge_strip(out, nbr_face, nbr_edge, b_new)

    return out


def _apply_edge_continuity_blend(
    state: ShallowWaterState,
    grid: CubedSphereGrid,
    strength: float,
) -> ShallowWaterState:
    """Blend cube-edge values for scalar and geographic vector continuity."""
    h_blend = _blend_scalar_cube_edges(state.h.data, strength)

    # Blend winds in geographic components to avoid face-local orientation bias.
    u = state.u.data
    v = state.v.data
    u_east = grid.cos_angle * u - grid.sin_angle * v
    v_north = grid.sin_angle * u + grid.cos_angle * v

    u_east_blend = _blend_scalar_cube_edges(u_east, strength)
    v_north_blend = _blend_scalar_cube_edges(v_north, strength)

    u_grid = grid.cos_angle * u_east_blend + grid.sin_angle * v_north_blend
    v_grid = -grid.sin_angle * u_east_blend + grid.cos_angle * v_north_blend

    return state._replace(
        h=state.h.replace(data=h_blend),
        u=state.u.replace(data=u_grid),
        v=state.v.replace(data=v_grid),
    )


class ShallowWaterConfig(NamedTuple):
    """Configuration for the shallow-water model."""
    g: float = constants.g                  # Gravitational acceleration [m/s^2]
    hyperdiff_coeff: float = 0.0            # Hyperdiffusion coefficient [m^4/s]
    use_conservation_fixer: bool = True      # Apply mass/energy fixers
    fix_mass: bool = True
    fix_energy: bool = True
    time_integrator: str = "ssp_rk3"        # "ssp_rk3" | "ssp_rk34"/"ssp34" | "ssp_rk54"/"ssp45"
    edge_blend_strength: float = 0.0        # 0..1 cube-edge continuity relaxation


def shallow_water_tendencies(
    state: ShallowWaterState,
    grid: CubedSphereGrid,
    config: ShallowWaterConfig = ShallowWaterConfig(),
) -> ShallowWaterTendencies:
    """Compute tendencies for the shallow water equations.

    Uses the vector-invariant form:
        du/dt =  (zeta + f) * v - dB/dx
        dv/dt = -(zeta + f) * u - dB/dy
    where B = K + g*(h + h_s) is the Bernoulli function and
    K = 0.5*(u^2 + v^2) is the kinetic energy per unit mass.

    This form avoids explicit momentum advection, which:
    - Eliminates vector halo issues (only scalar gradients of B)
    - Conserves energy and enstrophy in the continuous limit

    The vorticity zeta = dv/dx - du/dy is computed via curl_z(), which
    uses pad_halo_vector for correct velocity rotation at face boundaries.

    This is a pure function: state in, tendencies out.
    Fully compatible with jax.grad, jax.jit, jax.vmap.

    Parameters
    ----------
    state : ShallowWaterState
        Current state (h, u, v, h_s).
    grid : CubedSphereGrid
        The cubed-sphere grid.
    config : ShallowWaterConfig
        Model configuration.

    Returns
    -------
    ShallowWaterTendencies : Time derivatives (dh/dt, du/dt, dv/dt).
    """
    h = state.h
    u = state.u
    v = state.v
    h_s = state.h_s
    g = config.g

    # --- Mass continuity: dh/dt = -div(h*v) ---
    hu = Field(data=h.data * u.data, name="hu", dims=h.dims, units="m^2/s")
    hv = Field(data=h.data * v.data, name="hv", dims=h.dims, units="m^2/s")
    dh_dt = Field(
        data=-divergence(hu, hv, grid).data,
        name="dh_dt", dims=h.dims, units="m/s",
    )

    # --- Relative vorticity: zeta = dv/dx - du/dy ---
    # curl_z uses pad_halo_vector for correct velocity rotation at faces
    zeta = curl_z(u, v, grid).data

    # --- Absolute vorticity: zeta + f ---
    abs_vor = zeta + grid.f

    # --- Bernoulli function: B = K + g*(h + h_s) ---
    kinetic_energy = 0.5 * (u.data**2 + v.data**2)
    bernoulli_data = kinetic_energy + g * (h.data + h_s.data)
    bernoulli = Field(data=bernoulli_data, name="bernoulli",
                      dims=h.dims, units="m^2/s^2", staggering="cell")
    dB_dx = gradient_x(bernoulli, grid)
    dB_dy = gradient_y(bernoulli, grid)

    # --- Vector-invariant momentum equations ---
    # du/dt =  (zeta + f) * v - dB/dx
    # dv/dt = -(zeta + f) * u - dB/dy
    du_dt_data = abs_vor * v.data - dB_dx.data
    dv_dt_data = -abs_vor * u.data - dB_dy.data

    # --- Hyperdiffusion (scale-selective damping) ---
    if config.hyperdiff_coeff > 0:
        diff_u = hyperdiffusion(u, grid, config.hyperdiff_coeff)
        diff_v = hyperdiffusion(v, grid, config.hyperdiff_coeff)
        du_dt_data = du_dt_data + diff_u.data
        dv_dt_data = dv_dt_data + diff_v.data

    du_dt = Field(data=du_dt_data, name="du_dt", dims=u.dims, units="m/s^2")
    dv_dt = Field(data=dv_dt_data, name="dv_dt", dims=v.dims, units="m/s^2")

    return ShallowWaterTendencies(dh_dt=dh_dt, du_dt=du_dt, dv_dt=dv_dt)


class ShallowWaterModel:
    """Shallow water model on the cubed-sphere.

    This is the main user-facing class for Milestone 1.

    Parameters
    ----------
    grid : CubedSphereGrid
        The computational grid.
    config : ShallowWaterConfig, optional
        Model configuration.

    Example
    -------
    >>> grid = create_cubed_sphere(48)
    >>> model = ShallowWaterModel(grid)
    >>> state = williamson_test2(grid)
    >>> state_24h = model.integrate(state, duration=86400, dt=600)
    """

    def __init__(
        self,
        grid: CubedSphereGrid,
        config: ShallowWaterConfig | None = None,
    ):
        self.grid = grid
        self.config = config or ShallowWaterConfig()
        if not (0.0 <= self.config.edge_blend_strength <= 1.0):
            raise ValueError(
                "edge_blend_strength must be in [0, 1], "
                f"got {self.config.edge_blend_strength!r}"
            )

    def tendencies(self, state: ShallowWaterState) -> ShallowWaterTendencies:
        """Compute tendencies (pure function wrapper)."""
        return shallow_water_tendencies(state, self.grid, self.config)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: ShallowWaterState, dt: float) -> ShallowWaterState:
        """Advance one time step using SSP-RK3.

        Parameters
        ----------
        state : ShallowWaterState
            Current state.
        dt : float
            Time step [seconds].

        Returns
        -------
        ShallowWaterState : State after one time step.
        """
        def tendency_fn(s):
            tend = shallow_water_tendencies(s, self.grid, self.config)
            # Return same pytree structure as state for tree_map compatibility.
            # Field metadata (name, dims, units) must match the state's fields
            # so that jax.tree.map(f, state, tendencies) works correctly.
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
            raise ValueError(f"Unsupported time_integrator={self.config.time_integrator!r}")

        # Optional cubed-sphere edge continuity relaxation.
        if self.config.edge_blend_strength > 0.0:
            state_new = _apply_edge_continuity_blend(
                state_new,
                self.grid,
                self.config.edge_blend_strength,
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
            Initial state.
        duration : float
            Total integration time [seconds].
        dt : float
            Time step [seconds].
        save_every : int
            Save state every N steps.

        Returns
        -------
        final_state : ShallowWaterState
            Final state.
        trajectory : list of ShallowWaterState
            Saved states at intervals.
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

        This is the preferred method for gradient computation.
        Returns the full trajectory as stacked pytree leaves.

        Parameters
        ----------
        state : ShallowWaterState
            Initial state.
        n_steps : int
            Number of time steps.
        dt : float
            Time step [seconds].

        Returns
        -------
        final_state : ShallowWaterState
            State after n_steps.
        trajectory : ShallowWaterState
            All intermediate states (each leaf shape: (n_steps, 6, n, n)).
        """
        def scan_fn(state, _):
            new_state = self.step(state, dt)
            return new_state, new_state

        final_state, trajectory = jax.lax.scan(
            scan_fn, state, jnp.arange(n_steps)
        )
        return final_state, trajectory
