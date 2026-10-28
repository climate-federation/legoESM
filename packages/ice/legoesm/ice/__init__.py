"""Cryosphere component for legoESM.

Sea ice model with switchable dynamics and multi-category support:
- ``dynamics="none"``: Slab thermodynamics (default, backward compatible).
- ``dynamics="free_drift"``: Heuristic linear-combination velocity.
- ``dynamics="evp"``: EVP rheology (Hunke & Dukowicz 1997).
- ``dynamics="mevp"``: Modified-EVP pseudo-time relaxation
  (Bouillon 2013 / Kimmritz 2015).

Multi-category ice (``n_categories > 1``) uses a simplified category
transfer scheme for ice thickness redistribution.  Snow depth is not
tracked.
"""

from legoesm.ice.brine import (
    SaltBudgetResult,
    aggregate_salt_flux,
    update_salinity_and_salt_flux,
)
from legoesm.ice.c1d_omip_l3 import C1DOMIPL3Card, build_c1d_omip_l3_card
from legoesm.ice.config import (
    BrineConfig,
    MeltPondConfig,
    RidgingConfig,
    SI3ThermoConfig,
    SeaIceConfig,
    SnowConfig,
    build_si3_orca2_o1_config,
    validate_si3_bulk_config,
    validate_si3_thermo_config,
)
from legoesm.ice.constants_config import (
    IceConstantsConfig,
    NEMO_SI3_CONSTANTS_CONFIG,
)
from legoesm.ice.dynamics import (
    air_ice_stress,
    evp_solver,
    free_drift_velocity,
    mevp_solver,
    ocean_ice_stress,
    si3_cgrid_deformation,
    stress_divergence,
)
from legoesm.ice.itd import (
    aggregate_state,
    category_bounds,
    distribute_to_categories,
    linear_remap,
    lipscomb_2001_remap,
    si3_ice_itd_rem,
    upper_bounds,
)
from legoesm.ice.ponds import step_ponds
from legoesm.ice.rheology import (
    delta_deformation,
    evp_stress_update,
    ice_strength,
    mevp_stress_update,
    strain_rates,
    vp_stress,
)
from legoesm.ice.ridging import (
    SI3JPL1RidgingConfig,
    SI3JPL1RidgingLosses,
    SI3JPL1RidgingState,
    apply_ridging,
    apply_si3_jpl1_ridging,
    participation_weights,
)
from legoesm.ice.sea_ice import (
    grid_supports_ice_dynamics,
    grid_supports_ice_transport,
    step_sea_ice,
    uses_new_physics,
)
from legoesm.ice.shortwave import (
    IceSWResult,
    compute_ice_sw,
    delta_eddington_albedo,
    maykut_untersteiner_albedo,
)
from legoesm.ice.snow import (
    accumulate_snowfall,
    combined_conductive_flux,
    consume_from_snow_then_ice,
    consume_sublimation_from_snow_then_ice,
    snow_ice_flooding,
)
from legoesm.ice.state import (
    DynamicSeaIceState,
    SI3ColumnState,
    SeaIceState,
    distribute_dynamic_state_to_categories,
    dynamic_to_slab,
    init_dynamic_ice_state,
    slab_to_dynamic,
)
from legoesm.ice.transport import advect_ice_tracers

__all__ = [
    # Config and state
    "SeaIceConfig",
    "SeaIceState",
    "DynamicSeaIceState",
    "SI3ColumnState",
    "init_dynamic_ice_state",
    "distribute_dynamic_state_to_categories",
    "dynamic_to_slab",
    "slab_to_dynamic",
    # Main step function
    "step_sea_ice",
    "grid_supports_ice_dynamics",
    "grid_supports_ice_transport",
    "uses_new_physics",
    # Rheology
    "ice_strength",
    "strain_rates",
    "delta_deformation",
    "vp_stress",
    "evp_stress_update",
    "mevp_stress_update",
    # Dynamics
    "stress_divergence",
    "evp_solver",
    "mevp_solver",
    "free_drift_velocity",
    "air_ice_stress",
    "ocean_ice_stress",
    "si3_cgrid_deformation",
    # ITD
    "category_bounds",
    "upper_bounds",
    "aggregate_state",
    "distribute_to_categories",
    "linear_remap",
    "lipscomb_2001_remap",
    "si3_ice_itd_rem",
    # Transport
    "advect_ice_tracers",
    # Configs
    "SnowConfig",
    "BrineConfig",
    "RidgingConfig",
    "MeltPondConfig",
    "SI3ThermoConfig",
    "validate_si3_bulk_config",
    "validate_si3_thermo_config",
    "IceConstantsConfig",
    "NEMO_SI3_CONSTANTS_CONFIG",
    # Snow physics
    "accumulate_snowfall",
    "combined_conductive_flux",
    "consume_from_snow_then_ice",
    "consume_sublimation_from_snow_then_ice",
    "snow_ice_flooding",
    # Brine physics
    "update_salinity_and_salt_flux",
    "aggregate_salt_flux",
    "SaltBudgetResult",
    # Ridging
    "apply_ridging",
    "apply_si3_jpl1_ridging",
    "participation_weights",
    "SI3JPL1RidgingConfig",
    "SI3JPL1RidgingLosses",
    "SI3JPL1RidgingState",
    # Shortwave
    "compute_ice_sw",
    "delta_eddington_albedo",
    "maykut_untersteiner_albedo",
    "IceSWResult",
    # Ponds
    "step_ponds",
    "build_c1d_omip_l3_card",
    "C1DOMIPL3Card",
    # Selectable full ORCA2 SI3 state, aggregation, halos, and restart I/O (O1)
    "build_si3_orca2_o1_config",
    "SI3ORCA2StateConfig",
    "SI3ORCA2Forcing",
    "SI3CategoryBounds",
    "SI3PratherMoments",
    "SI3AllPratherMoments",
    "SI3EquivalentVars",
    "SI3AggregatedVars",
    "SI3State",
    "fortran_sum_categories",
    "si3_hfn_category_bounds",
    "si3_var_salprof",
    "si3_var_snwfra_3d",
    "si3_var_glo2eqv",
    "si3_var_agg",
    "si3_update_snwice_mass",
    "si3_lbc_lnk_orca2",
    "si3_apply_state_halos_orca2",
    "si3_o1_step",
    "si3_rst_to_netcdf_dict",
    "si3_rst_from_netcdf_dict",
    "write_si3_netcdf_restart",
    "read_si3_netcdf_restart",
]


from legoesm.ice.si3_orca2_state import (  # noqa: E402
    SI3AggregatedVars,
    SI3AllPratherMoments,
    SI3CategoryBounds,
    SI3EquivalentVars,
    SI3ORCA2Forcing,
    SI3ORCA2StateConfig,
    SI3PratherMoments,
    SI3State,
    fortran_sum_categories,
    read_si3_netcdf_restart,
    si3_apply_state_halos_orca2,
    si3_hfn_category_bounds,
    si3_lbc_lnk_orca2,
    si3_o1_step,
    si3_rst_from_netcdf_dict,
    si3_rst_to_netcdf_dict,
    si3_update_snwice_mass,
    si3_var_agg,
    si3_var_glo2eqv,
    si3_var_salprof,
    si3_var_snwfra_3d,
    write_si3_netcdf_restart,
)
from legoesm.ice.config import build_si3_orca2_o4_config  # noqa: E402
from legoesm.ice.si3_orca2_state import (  # noqa: E402
    SI3AdvectionDiagnostics,
    SI3ORCA2AdvectionGrid,
    si3_dyn_adv_pra_orca2,
    si3_dyn_orca2,
    si3_o4_step,
)
from legoesm.ice.transport import (  # noqa: E402
    si3_adv_x,
    si3_adv_y,
    si3_hbig_pra,
    si3_hsnow_pra,
    si3_icemax2d_pra,
    si3_icemax3d_pra,
    si3_var_zapneg,
    si3_var_zapsmall,
)

__all__.extend(
    (
        "build_si3_orca2_o4_config",
        "SI3ORCA2AdvectionGrid",
        "SI3AdvectionDiagnostics",
        "si3_adv_x",
        "si3_adv_y",
        "si3_icemax2d_pra",
        "si3_icemax3d_pra",
        "si3_hbig_pra",
        "si3_hsnow_pra",
        "si3_var_zapneg",
        "si3_var_zapsmall",
        "si3_dyn_adv_pra_orca2",
        "si3_dyn_orca2",
        "si3_o4_step",
    )
)

from legoesm.ice.config import build_si3_orca2_o2_config  # noqa: E402
from legoesm.ice.si3_orca2_state import (  # noqa: E402
    SI3O2InitConfig,
    SI3O2ScalarInitialState,
    SI3ORCA2O2Forcing,
    si3_dom_uniq_t_orca2,
    si3_eos_fzp,
    si3_glob_2dsum_ddpdd,
    si3_o2_scalar_initial_state,
    si3_o2_step,
)
from legoesm.ice.si3_finalize import (  # noqa: E402
    SI3FinalizeForcing,
    SI3FinalizeResponse,
)

__all__.extend(
    (
        "build_si3_orca2_o2_config",
        "SI3O2InitConfig",
        "SI3O2ScalarInitialState",
        "SI3ORCA2O2Forcing",
        "SI3FinalizeForcing",
        "SI3FinalizeResponse",
        "si3_dom_uniq_t_orca2",
        "si3_eos_fzp",
        "si3_glob_2dsum_ddpdd",
        "si3_o2_scalar_initial_state",
        "si3_o2_step",
    )
)

from legoesm.ice.config import SI3PondConfig  # noqa: E402
from legoesm.ice.si3_thd_pnd import (  # noqa: E402
    SI3O12CategoryOutput,
    SI3O12Stages,
    SI3PndLevOutputs,
    SI3PondOutputs,
    si3_ice_perm_eff,
    si3_ice_thd_o12_stages,
    si3_ice_thd_pnd,
    si3_pnd_lev,
)

__all__.extend(
    (
        "SI3PondConfig",
        "SI3PndLevOutputs",
        "SI3PondOutputs",
        "SI3O12CategoryOutput",
        "SI3O12Stages",
        "si3_ice_perm_eff",
        "si3_pnd_lev",
        "si3_ice_thd_pnd",
        "si3_ice_thd_o12_stages",
    )
)
