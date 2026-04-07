"""Baroclinic wave test case — backward-compatibility re-exports.

The canonical implementation now lives in ``legoesm.atmosphere.baroclinic_wave``
so that it is available at runtime without the ``tests`` package on
``sys.path``.  This module re-exports every public name for existing tests.
"""

from legoesm.atmosphere.baroclinic_wave import (  # noqa: F401
    # Constants
    P0,
    # Core functions
    evaluate_pressure_temperature,
    find_z_for_pressure,
    compute_zonal_wind,
    exponential_perturbation,
    # Grid-specific initialization
    baroclinic_wave_init,
    baroclinic_wave_init_latlon,
    baroclinic_wave_init_mpas,
    baroclinic_wave_init_spectral,
)
