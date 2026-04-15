#!/usr/bin/env python
"""Ocean test matrix: organized test runner for legoESM ocean dynamical cores.

Runs a comprehensive suite of ocean test cases across cubed-sphere, lat-lon,
and MPAS grids at ~5 degree resolution.

Test cases:
  Existing:
    - rest_state          Rest-state adjustment (stability check)
    - barotropic_wave     Gaussian SSH perturbation propagation
    - barotropic_gyre     Wind-driven single barotropic gyre (Stommel/Munk)
    - barotropic_double_gyre  Wind-driven barotropic double gyre (Holland & Lin)
    - baroclinic          Meridional temperature front relaxation
    - phillips_two_layer  Phillips 2-layer baroclinic instability

  Barotropic dynamics (Bishnu et al. 2024):
    - inertia_gravity_wave  Inertia-gravity (Poincare) wave propagation

  Numerical mixing & dianeutral transport (NEMO; Petersen et al. 2015):
    - lock_exchange       Density-driven gravity current (RPE diagnostic)
    - overflow            Dense water descending a bathymetric slope

  Tracer transport (Hecht et al. 2000):
    - stommel_gyre_tracer Passive tracer in wind-driven Stommel gyre

Output structure:
    results/ocean/<case>/<grid_type>/<resolution>/

Each case folder contains:
    - mean_timeseries.csv / .png      (domain-averaged scalar time series)
    - conservation_timeseries.csv/.png (volume, heat & salt drift)
    - field_snapshots.png              (2D field maps at selected times)
    - snapshots_<field>.png            (per-field snapshot evolution)
    - snapshots_native.npz             (snapshot arrays in native grid coords)
    - snapshots_latlon.npz             (snapshot arrays regridded to 181x360 lat-lon)
    - vertical_profiles.png            (vertical profile evolution)
    - latitude_vertical_cross_sections.png
    - longitude_vertical_cross_sections.png
    - results.txt                      (run metadata)

References:
    Bishnu et al. (2024), JAMES. DOI: 10.1029/2022MS003545
    Petersen et al. (2015), Ocean Modelling 86, 93-113.
        DOI: 10.1016/j.ocemod.2014.12.004
    Hecht et al. (2000), Ocean Modelling 2, 1-15.
        DOI: 10.1016/S1463-5003(00)00004-4

Usage:
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py --quick
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py --only lock_exchange
    JAX_ENABLE_X64=1 .venv/bin/python scripts/run_ocean_test_matrix.py --grid cubed_sphere
"""

from __future__ import annotations

import sys
from pathlib import Path

# Line-buffered stdout for CI/log visibility.
sys.stdout.reconfigure(line_buffering=True)

# Ensure project root is on sys.path (for legoesm imports).
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Ensure scripts/ is on sys.path (for ocean_test_matrix package).
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# JAX configuration must happen before any jax.numpy import.
import jax
jax.config.update("jax_enable_x64", True)

from legoesm.runtime.backend import ensure_metal_or_fallback
ensure_metal_or_fallback()

# Re-export everything from the package for backward compatibility.
# (e.g., run_geoadj_fluxform.py does `import run_ocean_test_matrix as otm;
#  otm._create_ocean_setup`)
from ocean_test_matrix import *  # noqa: F401,F403
from ocean_test_matrix.cli import main  # noqa: F811

if __name__ == "__main__":
    main()
