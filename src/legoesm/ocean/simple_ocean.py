"""Simple ocean modes for idealized experiments.

Three modes:
- Fixed SST: prescribed constant or spatial map, no state evolution.
- Slab ocean: single mixed-layer with energy balance and optional Q-flux.
- Two-layer slab: mixed layer + deep layer with vertical mixing and
  optional deep-layer restoring.

Factory ``make_ocean`` dispatches on ``SimpleOceanConfig.mode`` and returns
a uniform step function compatible with the coupler.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.core.field import Field
from legoesm.coupler.coupling_fields import AtmToSurface


# ============================================================================
# Configuration
# ============================================================================

class SimpleOceanConfig(NamedTuple):
    """Configuration for simplified ocean modes."""
    mode: str = "fixed"              # "fixed" | "slab" | "two_layer"
    # Fixed SST
    sst_constant: float = 300.0      # Global constant SST [K]
    # Slab ocean
    h_mix: float = 50.0              # Mixed-layer depth [m]
    rho_ocean: float = 1025.0        # Ocean density [kg/m3] (= eos.rho_0)
    c_ocean: float = 3994.0          # Seawater specific heat [J/kg/K] (= eos.c_sw)
    Q_flux: float = 0.0              # Prescribed OHT convergence [W/m2]
    albedo_ocean: float = 0.06
    emissivity_ocean: float = 0.97
    Cd_ocean: float = 1.5e-3         # Drag coefficient
    Ch_ocean: float = 1.5e-3         # Heat transfer coefficient
    U_min: float = 1.0               # Smooth wind floor [m/s]
    T_freeze: float = 271.35         # Freezing clamp [K] (= constants.T_freeze_ocean)
    # Two-layer additions
    h_deep: float = 200.0            # Deep layer depth [m]
    k_mix: float = 1.0e-4            # Vertical mixing coefficient [m2/s]
    restore_deep: bool = False       # Restore deep layer toward T_deep_ref?
    T_deep_ref: float = 278.0        # Deep restoring target [K]
    tau_deep: float = 365.25 * 86400.0  # Restoring timescale [s] (1 year)


# ============================================================================
# State
# ============================================================================

class SlabOceanState(NamedTuple):
    """State for slab and two-layer ocean modes."""
    T_sfc: Field    # SST [K], shape (6, n, n)
    T_deep: Field   # Deep layer temperature [K], shape (6, n, n)


# ============================================================================
# Initialization
# ============================================================================

DIMS_2D = ("face", "x", "y")


def init_slab_state(
    shape: tuple[int, ...],
    T_sfc_init: float = 300.0,
    T_deep_init: float = 278.0,
) -> SlabOceanState:
    """Initialize a SlabOceanState with uniform temperatures."""
    return SlabOceanState(
        T_sfc=Field(
            jnp.full(shape, T_sfc_init),
            name="T_sfc", dims=DIMS_2D, units="K",
        ),
        T_deep=Field(
            jnp.full(shape, T_deep_init),
            name="T_deep", dims=DIMS_2D, units="K",
        ),
    )


# ============================================================================
# Slab ocean physics
# ============================================================================

def _slab_step(
    state: SlabOceanState,
    forcing: AtmToSurface,
    config: SimpleOceanConfig,
    dt: float,
) -> tuple[SlabOceanState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single mixed-layer energy balance step."""
    T_sfc = state.T_sfc.data

    # Smooth wind floor
    wind = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + config.U_min ** 2
    )

    # Surface humidity: saturated
    q_sfc = saturation_mixing_ratio(T_sfc, forcing.p_surface)

    # Bulk fluxes (positive upward)
    rho = forcing.rho_lowest
    shflx = rho * constants.c_pd * config.Ch_ocean * wind * (T_sfc - forcing.T_lowest)
    lhflx = rho * constants.L_v * config.Ch_ocean * wind * (q_sfc - forcing.q_lowest)

    # Radiation
    sw_net = (1.0 - config.albedo_ocean) * forcing.sw_down
    lw_net = (config.emissivity_ocean * forcing.lw_down
              - config.emissivity_ocean * constants.sigma_sb * T_sfc ** 4)

    # Energy balance
    C_mix = config.rho_ocean * config.c_ocean * config.h_mix
    dT_dt = (sw_net + lw_net - shflx - lhflx + config.Q_flux) / C_mix
    T_sfc_new = T_sfc + dt * dT_dt

    # Freezing clamp
    T_sfc_new = jnp.maximum(T_sfc_new, config.T_freeze)

    new_state = SlabOceanState(
        T_sfc=state.T_sfc.replace(data=T_sfc_new),
        T_deep=state.T_deep,  # unchanged in slab mode
    )
    u_sfc = jnp.zeros_like(T_sfc)
    v_sfc = jnp.zeros_like(T_sfc)
    return new_state, T_sfc_new, u_sfc, v_sfc


# ============================================================================
# Two-layer ocean physics
# ============================================================================

def _two_layer_step(
    state: SlabOceanState,
    forcing: AtmToSurface,
    config: SimpleOceanConfig,
    dt: float,
) -> tuple[SlabOceanState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Two-layer slab ocean: mixed layer + deep layer."""
    T_sfc = state.T_sfc.data
    T_deep = state.T_deep.data

    # Smooth wind floor
    wind = jnp.sqrt(
        forcing.u_lowest ** 2 + forcing.v_lowest ** 2 + config.U_min ** 2
    )

    # Surface humidity: saturated
    q_sfc = saturation_mixing_ratio(T_sfc, forcing.p_surface)

    # Bulk fluxes (positive upward)
    rho = forcing.rho_lowest
    shflx = rho * constants.c_pd * config.Ch_ocean * wind * (T_sfc - forcing.T_lowest)
    lhflx = rho * constants.L_v * config.Ch_ocean * wind * (q_sfc - forcing.q_lowest)

    # Radiation
    sw_net = (1.0 - config.albedo_ocean) * forcing.sw_down
    lw_net = (config.emissivity_ocean * forcing.lw_down
              - config.emissivity_ocean * constants.sigma_sb * T_sfc ** 4)

    # Vertical mixing flux (positive downward = heat from surface to deep)
    d_mid = 0.5 * (config.h_mix + config.h_deep)
    F_mix = (config.rho_ocean * config.c_ocean * config.k_mix
             * (T_sfc - T_deep) / d_mid)

    # Mixed layer energy balance
    C_mix = config.rho_ocean * config.c_ocean * config.h_mix
    dT_sfc_dt = (sw_net + lw_net - shflx - lhflx + config.Q_flux - F_mix) / C_mix
    T_sfc_new = T_sfc + dt * dT_sfc_dt

    # Deep layer
    C_deep = config.rho_ocean * config.c_ocean * config.h_deep
    restore = jnp.where(
        config.restore_deep,
        C_deep * (T_deep - config.T_deep_ref) / config.tau_deep,
        0.0,
    )
    dT_deep_dt = (F_mix - restore) / C_deep
    T_deep_new = T_deep + dt * dT_deep_dt

    # Freezing clamp on surface
    T_sfc_new = jnp.maximum(T_sfc_new, config.T_freeze)

    new_state = SlabOceanState(
        T_sfc=state.T_sfc.replace(data=T_sfc_new),
        T_deep=state.T_deep.replace(data=T_deep_new),
    )
    u_sfc = jnp.zeros_like(T_sfc)
    v_sfc = jnp.zeros_like(T_sfc)
    return new_state, T_sfc_new, u_sfc, v_sfc


# ============================================================================
# Factory
# ============================================================================

def make_ocean(config: SimpleOceanConfig, sst_map=None):
    """Factory: returns a step function for the configured ocean mode.

    Parameters
    ----------
    config : SimpleOceanConfig
        Ocean configuration with mode selection.
    sst_map : jax.Array or None
        Prescribed SST map of shape (6, n, n) for fixed mode.
        If None, ``config.sst_constant`` is broadcast.

    Returns
    -------
    step_ocean : callable
        ``(state, forcing, dt) -> (state, sst, u_sfc, v_sfc)``
    """
    mode = config.mode

    if mode == "fixed":
        def step_fixed(state, forcing, dt):
            if sst_map is not None:
                sst = sst_map
            else:
                sst = jnp.broadcast_to(
                    jnp.array(config.sst_constant), forcing.sw_down.shape,
                )
            u_sfc = jnp.zeros_like(sst)
            v_sfc = jnp.zeros_like(sst)
            return state, sst, u_sfc, v_sfc
        return step_fixed

    elif mode == "slab":
        def step_slab(state, forcing, dt):
            return _slab_step(state, forcing, config, dt)
        return step_slab

    elif mode == "two_layer":
        def step_two_layer(state, forcing, dt):
            return _two_layer_step(state, forcing, config, dt)
        return step_two_layer

    else:
        raise ValueError(f"Unknown ocean mode: {mode!r}")
