"""Fast spectral-bin microphysics (FSBM-2) — WRF ``module_mp_fast_sbm.F`` port.

Eulerian bin microphysics on a mass-doubling grid (33 liquid bins,
43 aerosol bins), built incrementally against the WRF/HUCM oracle.
See ``docs/specs/bin_microphysics.md`` for the port plan and status.
"""

from legoesm.atmosphere.physics.microphysics.fast_sbm.grid import (
    COL,
    KRDROP,
    NKR_AEROSOL,
    NKR_LIQUID,
    R_MIN_LIQUID,
    bin_mass_widths,
    bin_mixing_ratios_from_f,
    discretize_lognormal,
    f_from_bin_mixing_ratios,
    mass_density,
    mass_doubling_grid,
    number_density,
    radius_from_mass,
)

__all__ = [
    "COL",
    "KRDROP",
    "NKR_AEROSOL",
    "NKR_LIQUID",
    "R_MIN_LIQUID",
    "bin_mass_widths",
    "bin_mixing_ratios_from_f",
    "discretize_lognormal",
    "f_from_bin_mixing_ratios",
    "mass_density",
    "mass_doubling_grid",
    "number_density",
    "radius_from_mass",
]
