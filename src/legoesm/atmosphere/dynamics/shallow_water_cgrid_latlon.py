"""C-grid Shallow Water Equations on the lat-lon grid (FV3-inspired).

Implements a proper C-grid staggering where:
- h (height) lives at cell centers: (n_lat, n_lon)
- uc (zonal velocity) lives at longitude interfaces: (n_lat, n_lon)
  uc[j, i] is at the western edge of cell (j, i), i.e., between cells (j, i-1) and (j, i)
  Periodic: uc[j, 0] is between cell (j, n_lon-1) and cell (j, 0)
- vc (meridional velocity) lives at latitude interfaces: (n_lat+1, n_lon)
  vc[j, i] is at the southern edge of cell (j, i), between cells (j-1, i) and (j, i)
  vc[0, :] = 0 at south pole, vc[n_lat, :] = 0 at north pole

Key advantages over A-grid:
- Pressure gradient is EXACT (just difference of neighboring cell values)
- Mass flux uses velocity naturally at the cell interface (no interpolation)
- Better gravity wave dispersion
- Divergence damping is natural

Key trade-off:
- Coriolis term requires 4-point interpolation of the other velocity component

References
----------
- Lin & Rood (1997): An explicit flux-form semi-Lagrangian SWE model
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core (FV3)
- Arakawa & Lamb (1977): Computational design of the basic dynamical processes
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import ShallowWaterState
from legoesm.core.conservation import _global_area_sum, _total_area
from legoesm.core.operators_cgrid_latlon import (
    cgrid_mass_flux,
    cgrid_momentum,
)
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.polar_filter import compute_polar_filter_mask, fourier_filter
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import IntegrationMixin
from legoesm import constants


# ==============================================================================
# C-grid state
# ==============================================================================

class CGShallowWaterState(NamedTuple):
    """Shallow water state on a C-grid lat-lon grid.

    h : (n_lat, n_lon) — height at cell centers
    uc : (n_lat, n_lon) — zonal velocity at western cell edges (periodic in lon)
    vc : (n_lat+1, n_lon) — meridional velocity at southern cell edges
    h_s : (n_lat, n_lon) — surface topography at cell centers
    """
    h: jax.Array
    uc: jax.Array
    vc: jax.Array
    h_s: jax.Array


class CGShallowWaterConfig(NamedTuple):
    """Configuration for C-grid shallow water model."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    div_damp_2: float = 0.0       # 2nd-order divergence damping coefficient
    div_damp_4: float = 0.0       # 4th-order divergence damping coefficient
    use_polar_filter: bool = True
    polar_filter_cutoff_deg: float = 60.0
    polar_filter_max_wave_speed: float = 300.0
    time_integrator: str = "ssp_rk3"


# ==============================================================================
# A-grid <-> C-grid conversions
# ==============================================================================

def a_to_cgrid(state: ShallowWaterState, grid: LatLonGrid) -> CGShallowWaterState:
    """Convert A-grid shallow water state to C-grid.

    u at cell centers -> uc at longitude interfaces (average neighbors).
    v at cell centers -> vc at latitude interfaces (average neighbors).
    """
    u = state.u.data  # (n_lat, n_lon)
    v = state.v.data

    # uc at western edge of cell (j, i): average of cells (j, i-1) and (j, i)
    uc = 0.5 * (u + jnp.roll(u, 1, axis=-1))  # (n_lat, n_lon)

    # vc at southern edge of cell (j, i): average of cells (j-1, i) and (j, i)
    vc_interior = 0.5 * (v[:-1, :] + v[1:, :])  # (n_lat-1, n_lon)
    vc_south = v[0:1, :] * 0.5  # half weight at pole
    vc_north = v[-1:, :] * 0.5
    vc = jnp.concatenate([vc_south, vc_interior, vc_north], axis=0)  # (n_lat+1, n_lon)

    return CGShallowWaterState(
        h=state.h.data,
        uc=uc,
        vc=vc,
        h_s=state.h_s.data,
    )


def cgrid_to_a(state: CGShallowWaterState, grid: LatLonGrid) -> ShallowWaterState:
    """Convert C-grid state back to A-grid for diagnostics/comparison.

    uc at edges -> u at centers (average flanking edges).
    vc at edges -> v at centers (average flanking edges).
    """
    u = 0.5 * (state.uc + jnp.roll(state.uc, -1, axis=-1))
    v = 0.5 * (state.vc[:-1, :] + state.vc[1:, :])

    dims = ("lat", "lon")
    return ShallowWaterState(
        h=Field(data=state.h, name="h", dims=dims, units="m"),
        u=Field(data=u, name="u", dims=dims, units="m/s"),
        v=Field(data=v, name="v", dims=dims, units="m/s"),
        h_s=Field(data=state.h_s, name="h_s", dims=dims, units="m"),
    )


# ==============================================================================
# C-grid shallow water tendencies
# ==============================================================================

def cgrid_shallow_water_tendencies(
    state: CGShallowWaterState,
    grid: LatLonGrid,
    config: CGShallowWaterConfig,
    polar_filter_mask=None,
):
    """Compute C-grid shallow water tendencies.

    Returns
    -------
    (dh_dt, duc_dt, dvc_dt)
    """
    h, uc, vc, h_s = state

    # Mass
    dh_dt = cgrid_mass_flux(h, uc, vc, grid)

    # Zero-mean correction
    dh_dt = dh_dt - jnp.sum(dh_dt * grid.area) / grid.total_area

    # Momentum
    duc_dt, dvc_dt = cgrid_momentum(h, uc, vc, h_s, grid, config)

    # Polar filter
    if config.use_polar_filter and polar_filter_mask is not None:
        dh_dt = fourier_filter(dh_dt, grid, polar_filter_mask)
        duc_dt = fourier_filter(duc_dt, grid, polar_filter_mask)
        # vc has n_lat+1 rows; filter expects (n_lat, n_lon).
        dvc_interior = dvc_dt[1:-1, :]
        dvc_padded = jnp.concatenate([dvc_interior, jnp.zeros((1, dvc_dt.shape[-1]))], axis=0)
        dvc_filtered = fourier_filter(dvc_padded, grid, polar_filter_mask)
        dvc_dt = jnp.concatenate([
            dvc_dt[0:1, :], dvc_filtered[:-1, :], dvc_dt[-1:, :]
        ], axis=0)

    return dh_dt, duc_dt, dvc_dt


# ==============================================================================
# Model class
# ==============================================================================

class CGShallowWaterLatLonModel(IntegrationMixin):
    """C-grid shallow water model on the lat-lon grid (FV3-inspired).

    Uses proper C-grid staggering with:
    - Exact pressure gradients (cell differences)
    - PPM mass transport with naturally colocated edge velocity
    - Divergence damping
    - 4-point Coriolis interpolation

    Parameters
    ----------
    grid : LatLonGrid
    config : CGShallowWaterConfig, optional
    dt : float, optional
        Reference time step for polar filter.
    """

    def __init__(
        self,
        grid: LatLonGrid,
        config: CGShallowWaterConfig | None = None,
        dt: float = 600.0,
    ):
        self.grid = grid
        self.config = config or CGShallowWaterConfig()

        if self.config.use_polar_filter:
            self.polar_filter_mask = compute_polar_filter_mask(
                grid, dt=dt,
                max_wave_speed=self.config.polar_filter_max_wave_speed,
                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
            )
        else:
            self.polar_filter_mask = None

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CGShallowWaterState, dt: float) -> CGShallowWaterState:
        """Advance one time step."""
        def tendency_fn(s):
            dh, duc, dvc = cgrid_shallow_water_tendencies(
                s, self.grid, self.config, self.polar_filter_mask,
            )
            return CGShallowWaterState(h=dh, uc=duc, vc=dvc, h_s=jnp.zeros_like(s.h_s))

        state_new = dispatch_integrator(
            state, tendency_fn, dt, self.config.time_integrator,
        )

        # Mass conservation fixer
        mass_old = _global_area_sum(state.h, self.grid)
        mass_new = _global_area_sum(state_new.h, self.grid)
        h_fixed = state_new.h + (mass_old - mass_new) / _total_area(self.grid)
        state_new = state_new._replace(h=h_fixed)

        return state_new
