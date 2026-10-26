"""Fast spectral-bin microphysics (FSBM-2) — WRF ``module_mp_fast_sbm.F`` port.

Eulerian bin microphysics on a mass-doubling grid (33 liquid bins,
43 aerosol bins), built incrementally against the WRF/HUCM oracle.
See ``docs/science/specs/bin_microphysics.md`` for the port plan and status.

Oracle-faithfulness status (WRF v4.8.0 06d4240a, campaign 2026-10):
FAITHFUL (reproduce the running oracle to <= 1e-10 rel in fp64,
``tests/atmosphere/microphysics/unit/test_fast_sbm_wrf_oracle.py``):
``precompute_collision_tables`` (courant_bott_KS), ``bott_coalescence``
(coll_xxx_lwf; one sweep NOT certified where the oracle's own 1-ulp spread
exceeds the tolerance), ``bott_riming`` (coll_xyx_lwf, mass only),
``warm_condensation_step`` (one ONECOND1 call), ``freeze_step_routed``
(FREEZ), ``melt_step`` (J_W_MELT, every category ladder) and
``sedimentation.falfluxhucm_z`` (FALFLUXHUCM_Z).
NOT YET FAITHFUL: the column adapter (``column.py``), ``sediment_bins``,
nucleation, ice condensation (ONECOND2/3), breakup, SIP, ECOALMASSM and the
WRF ``run/`` data tables.
"""

from legoesm.atmosphere.physics.microphysics.fast_sbm.config import (
    FastSBMConfig,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.diffusional_growth import (
    drop_growth_coefficient,
    supersat_relaxation_integral,
    vapor_diffusivity,
    ventilation_factor,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.condensation_driver import (
    WarmCondensationResult,
    warm_condensation_step,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.freezing import (
    FreezeResult,
    FreezeRoutedResult,
    bigg_freezing_rate,
    freeze_step,
    freeze_step_routed,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.ice_fall_speed import (
    ice_fall_speed,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.melting import (
    MeltResult,
    melt_fraction,
    melt_step,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.nucleation import (
    NucleationResult,
    activate_ccn,
    critical_dry_radius,
    kelvin_coefficient,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.sedimentation import (
    sediment_bins,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.remap import (
    RemapResult,
    condensation_new_masses,
    remap_spectrum,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.supersaturation import (
    SupersatStep,
    integrate_supersaturation,
    supersat_relaxation_rate,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.collision import (
    GMIN_DEFAULT,
    CollisionTables,
    bott_coalescence,
    bott_riming,
    collision_ck_matrix,
    f_from_g,
    g_from_f,
    mass_density_from_g,
    number_density_from_g,
    precompute_collision_tables,
    precompute_riming_tables,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.grid import (
    COL,
    KRDROP,
    NKR_AEROSOL,
    NKR_LIQUID,
    R_MIN_LIQUID,
    bin_mass_widths,
    bin_mixing_ratios_from_f,
    discretize_exponential,
    discretize_lognormal,
    f_from_bin_mixing_ratios,
    mass_density,
    mass_doubling_grid,
    number_density,
    radius_from_mass,
)

__all__ = [
    "COL",
    "GMIN_DEFAULT",
    "KRDROP",
    "NKR_AEROSOL",
    "NKR_LIQUID",
    "R_MIN_LIQUID",
    "CollisionTables",
    "FastSBMConfig",
    "RemapResult",
    "WarmCondensationResult",
    "condensation_new_masses",
    "FreezeResult",
    "FreezeRoutedResult",
    "MeltResult",
    "NucleationResult",
    "activate_ccn",
    "bigg_freezing_rate",
    "critical_dry_radius",
    "freeze_step",
    "freeze_step_routed",
    "ice_fall_speed",
    "kelvin_coefficient",
    "melt_fraction",
    "melt_step",
    "remap_spectrum",
    "sediment_bins",
    "warm_condensation_step",
    "SupersatStep",
    "integrate_supersaturation",
    "supersat_relaxation_rate",
    "drop_growth_coefficient",
    "supersat_relaxation_integral",
    "vapor_diffusivity",
    "ventilation_factor",
    "bin_mass_widths",
    "bin_mixing_ratios_from_f",
    "bott_coalescence",
    "bott_riming",
    "collision_ck_matrix",
    "discretize_exponential",
    "discretize_lognormal",
    "f_from_bin_mixing_ratios",
    "f_from_g",
    "g_from_f",
    "mass_density",
    "mass_density_from_g",
    "mass_doubling_grid",
    "number_density",
    "number_density_from_g",
    "precompute_collision_tables",
    "precompute_riming_tables",
    "radius_from_mass",
]
