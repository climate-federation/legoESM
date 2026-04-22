"""Simplified ocean models on MPAS Voronoi meshes.

Provides fixed-SST, slab, and two-layer mixed-layer ocean modes
operating on (nCells,) shaped arrays. Mirrors :mod:`simple_ocean`
but for the Voronoi mesh topology.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.ocean.mpas_config import MPASSimpleOceanConfig
from legoesm import constants
from legoesm.thermo import saturation_specific_humidity


class MPASSlabOceanState(NamedTuple):
    """State for slab/two-layer ocean on Voronoi mesh.

    Fields
    ------
    T_sfc : Field
        Sea surface temperature [K], shape (nCells,).
    T_deep : Field
        Deep layer temperature [K], shape (nCells,).
    """
    T_sfc: Field
    T_deep: Field


def init_mpas_slab_state(
    nCells: int,
    T_sfc_init: float = 300.0,
    T_deep_init: float = 278.0,
) -> MPASSlabOceanState:
    """Initialize MPAS slab ocean state.

    Parameters
    ----------
    nCells : int
        Number of Voronoi cells.
    T_sfc_init : float
        Initial SST [K].
    T_deep_init : float
        Initial deep temperature [K].

    Returns
    -------
    MPASSlabOceanState
    """
    return MPASSlabOceanState(
        T_sfc=Field(
            data=jnp.full(nCells, T_sfc_init),
            name="T_sfc", dims=("nCells",), units="K",
        ),
        T_deep=Field(
            data=jnp.full(nCells, T_deep_init),
            name="T_deep", dims=("nCells",), units="K",
        ),
    )


def _slab_step(state, forcing, config, dt):
    """Single slab ocean step on Voronoi mesh."""
    T_sfc = state.T_sfc.data
    C_mix = config.rho_ocean * config.c_ocean * config.h_mix

    # Wind speed with smooth floor
    wind = jnp.sqrt(
        forcing.u_lowest**2 + forcing.v_lowest**2 + config.U_min**2
    )

    # Surface humidity: saturated
    q_sfc = saturation_specific_humidity(T_sfc, forcing.p_surface)

    # Bulk fluxes (positive upward)
    rho = forcing.rho_lowest
    shflx = rho * constants.c_pd * config.Ch_ocean * wind * (
        T_sfc - forcing.T_lowest
    )
    lhflx = rho * constants.L_v * config.Ch_ocean * wind * (
        q_sfc - forcing.q_lowest
    )

    # Radiation
    sw_net = (1.0 - config.albedo_ocean) * forcing.sw_down
    lw_net = (config.emissivity_ocean * forcing.lw_down
              - config.emissivity_ocean * constants.sigma_sb * T_sfc**4)

    # Energy balance
    dT_dt = (sw_net + lw_net - shflx - lhflx + config.Q_flux) / C_mix
    T_new = T_sfc + dt * dT_dt
    T_new = jnp.maximum(T_new, config.T_freeze)

    new_state = MPASSlabOceanState(
        T_sfc=state.T_sfc.replace(data=T_new),
        T_deep=state.T_deep,
    )
    return new_state, T_new, jnp.zeros_like(T_new), jnp.zeros_like(T_new)


def _two_layer_step(state, forcing, config, dt):
    """Two-layer ocean step on Voronoi mesh."""
    T_sfc = state.T_sfc.data
    T_deep = state.T_deep.data
    C_mix = config.rho_ocean * config.c_ocean * config.h_mix

    # Wind speed with smooth floor
    wind = jnp.sqrt(
        forcing.u_lowest**2 + forcing.v_lowest**2 + config.U_min**2
    )

    # Fluxes (same as slab)
    q_sfc = saturation_specific_humidity(T_sfc, forcing.p_surface)
    rho = forcing.rho_lowest
    shflx = rho * constants.c_pd * config.Ch_ocean * wind * (
        T_sfc - forcing.T_lowest
    )
    lhflx = rho * constants.L_v * config.Ch_ocean * wind * (
        q_sfc - forcing.q_lowest
    )
    sw_net = (1.0 - config.albedo_ocean) * forcing.sw_down
    lw_net = (config.emissivity_ocean * forcing.lw_down
              - config.emissivity_ocean * constants.sigma_sb * T_sfc**4)

    # Vertical mixing between layers
    mix_flux = config.k_mix * (T_sfc - T_deep) / (
        0.5 * (config.h_mix + config.h_deep)
    )

    # Mixed layer
    dT_sfc_dt = (sw_net + lw_net - shflx - lhflx + config.Q_flux) / C_mix - mix_flux / config.h_mix
    T_sfc_new = T_sfc + dt * dT_sfc_dt
    T_sfc_new = jnp.maximum(T_sfc_new, config.T_freeze)

    # Deep layer
    dT_deep_dt = mix_flux / config.h_deep
    if config.restore_deep:
        dT_deep_dt = dT_deep_dt - (T_deep - config.T_deep_ref) / config.tau_deep
    T_deep_new = T_deep + dt * dT_deep_dt

    new_state = MPASSlabOceanState(
        T_sfc=state.T_sfc.replace(data=T_sfc_new),
        T_deep=state.T_deep.replace(data=T_deep_new),
    )
    return new_state, T_sfc_new, jnp.zeros_like(T_sfc_new), jnp.zeros_like(T_sfc_new)


def make_mpas_ocean(config: MPASSimpleOceanConfig, sst_map=None):
    """Factory for simple MPAS ocean step function.

    Parameters
    ----------
    config : MPASSimpleOceanConfig
    sst_map : jax.Array or None
        Prescribed SST field [K], shape (nCells,). Used for "fixed" mode.

    Returns
    -------
    step_ocean : callable
        ``(state, forcing, dt) -> (state, sst, u_sfc, v_sfc)``
    """
    if config.mode == "fixed":
        _sst = sst_map if sst_map is not None else None
        _const = config.sst_constant

        def step_fixed(state, forcing, dt):
            sst = _sst if _sst is not None else jnp.full_like(
                state.T_sfc.data, _const,
            )
            u_sfc = jnp.zeros_like(sst)
            v_sfc = jnp.zeros_like(sst)
            return state, sst, u_sfc, v_sfc

        return step_fixed

    elif config.mode == "slab":
        def step_slab(state, forcing, dt):
            return _slab_step(state, forcing, config, dt)
        return step_slab

    elif config.mode == "two_layer":
        def step_two_layer(state, forcing, dt):
            return _two_layer_step(state, forcing, config, dt)
        return step_two_layer

    else:
        raise ValueError(f"Unknown MPAS ocean mode: {config.mode!r}")
