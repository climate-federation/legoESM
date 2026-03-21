"""Shared spatial filters for prognostic state variables."""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.state import HydrostaticState
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.polar_filter import fourier_filter, fourier_filter_3d


def filter_state(
    state: HydrostaticState,
    grid: LatLonGrid,
    mask: jnp.ndarray,
) -> HydrostaticState:
    """Apply polar filter to prognostic state variables."""
    return HydrostaticState(
        u=state.u.replace(data=fourier_filter_3d(state.u.data, grid, mask)),
        v=state.v.replace(data=fourier_filter_3d(state.v.data, grid, mask)),
        T=state.T.replace(data=fourier_filter_3d(state.T.data, grid, mask)),
        p_s=state.p_s.replace(data=fourier_filter(state.p_s.data, grid, mask)),
        phis=state.phis,
    )
