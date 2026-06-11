"""Fast spectral-bin microphysics (FSBM-2) — WRF ``module_mp_fast_sbm.F`` port.

Eulerian bin microphysics on a mass-doubling grid (33 liquid bins,
43 aerosol bins), built incrementally against the WRF/HUCM oracle.
See ``docs/specs/bin_microphysics.md`` for the port plan and status.
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
    collision_ck_matrix,
    f_from_g,
    g_from_f,
    mass_density_from_g,
    number_density_from_g,
    precompute_collision_tables,
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
    "remap_spectrum",
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
    "radius_from_mass",
]
