"""Ocean experiment modules.

This package contains well-organized, self-contained modules for each ocean
test case. Each experiment module provides:

- Comprehensive documentation of scientific purpose and setup
- Standardized configuration classes with meaningful defaults
- Grid-agnostic initial condition creation
- Experiment-specific validation criteria
- Clear separation of concerns (IC, forcing, domain, validation)

Example Usage:
    from legoesm.ocean.experiments import rest_state

    # Create initial conditions
    config = rest_state.RestStateConfig(T_water_init_C=18.0)
    initial_state = rest_state.create_initial_conditions(
        "cubed_sphere", grid, z_coord, config)

    # Validate results
    success, notes = rest_state.validate_results(final_state, diagnostics)

Available Experiments:
- rest_state: Fundamental stability and conservation test
- barotropic_wave: Barotropic gravity wave propagation test
- regional_gyre: Regional wind-driven gyre circulation (base for gyre experiments)
- baroclinic: Baroclinic adjustment from meridional temperature front
- baroclinic_gyre: Regional wind-driven baroclinic gyre with surface restoring
- phillips_two_layer: Two-layer baroclinic instability with relaxation
- inertia_gravity_wave: Inertia-gravity wave propagation with analytical validation
- lock_exchange: Density-driven gravity current with RPE mixing diagnostics
- overflow: Dense water overflow over bathymetric slope
- stommel_gyre_tracer: Passive tracer transport in wind-driven gyre
- geostrophic_adjustment: Geostrophic adjustment from temperature front
- global_barotropic_wind: Global barotropic wind-driven circulation
- eady_instability: Eady baroclinic instability from meridional temperature front
- acc_channel: ACC-like channel with Gaussian ridge and wind forcing
- global_overturning: Global baroclinic overturning with stratification and SST restoring
- dino: DINO (Diabatic Neverworld Ocean) — Kamm et al. 2025 GMD, pole-to-pole sector with re-entrant channel

Design Principles:
1. Each experiment is self-documenting with scientific context
2. Configurations are explicit and well-documented
3. Validation criteria are experiment-specific and meaningful
4. Grid-specific implementations are abstracted away
5. Consistent interfaces enable easy test matrix integration
"""

from . import rest_state
from . import barotropic_wave
from . import regional_gyre
from . import baroclinic
from . import baroclinic_gyre
from . import phillips_two_layer
from . import inertia_gravity_wave
from . import lock_exchange
from . import overflow
from . import stommel_gyre_tracer
from . import geostrophic_adjustment
from . import global_barotropic_wind
from . import eady_instability
from . import acc_channel
from . import global_overturning
from . import dino

# Registry of all available experiments
AVAILABLE_EXPERIMENTS = {
    "rest_state": rest_state.EXPERIMENT_CONFIG,
    "barotropic_wave": barotropic_wave.EXPERIMENT_CONFIG,
    "regional_gyre": regional_gyre.EXPERIMENT_CONFIG,
    "baroclinic": baroclinic.EXPERIMENT_CONFIG,
    "baroclinic_gyre": baroclinic_gyre.EXPERIMENT_CONFIG,
    "phillips_two_layer": phillips_two_layer.EXPERIMENT_CONFIG,
    "inertia_gravity_wave": inertia_gravity_wave.EXPERIMENT_CONFIG,
    "lock_exchange": lock_exchange.EXPERIMENT_CONFIG,
    "overflow": overflow.EXPERIMENT_CONFIG,
    "stommel_gyre_tracer": stommel_gyre_tracer.EXPERIMENT_CONFIG,
    "geostrophic_adjustment": geostrophic_adjustment.EXPERIMENT_CONFIG,
    "global_barotropic_wind": global_barotropic_wind.EXPERIMENT_CONFIG,
    "eady_instability": eady_instability.EXPERIMENT_CONFIG,
    "acc_channel": acc_channel.EXPERIMENT_CONFIG,
    "global_overturning": global_overturning.EXPERIMENT_CONFIG,
    "dino": dino.EXPERIMENT_CONFIG,
}

__all__ = [
    "rest_state",
    "barotropic_wave",
    "regional_gyre",
    "baroclinic",
    "baroclinic_gyre",
    "phillips_two_layer",
    "inertia_gravity_wave",
    "lock_exchange",
    "overflow",
    "stommel_gyre_tracer",
    "geostrophic_adjustment",
    "global_barotropic_wind",
    "eady_instability",
    "acc_channel",
    "global_overturning",
    "dino",
    "AVAILABLE_EXPERIMENTS",
]