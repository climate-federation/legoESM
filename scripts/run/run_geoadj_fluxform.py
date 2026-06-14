#!/usr/bin/env python
"""Run geostrophic adjustment with flux_form_tracers=True to reproduce blowup.

Usage:
    PYTHONPATH=src JAX_ENABLE_X64=1 python scripts/run_geoadj_fluxform.py
"""
import sys
import os

# Add scripts/matrix to path so we can import the test matrix runner
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "matrix"))

# Override CLI args for the test matrix
sys.argv = [
    "run_ocean_test_matrix.py",
    "--only", "geostrophic_adjustment",
    "--grid", "mpas",
    "--output", "results/ocean_geoadj_mpas_fluxform",
    "--days", "10",
]

import run_ocean_test_matrix as otm

_orig_setup = otm._create_ocean_setup

def _patched_setup(tc, *args, **kwargs):
    result = _orig_setup(tc, *args, **kwargs)
    if tc.grid_type == "mpas":
        mesh, z_coord, config, model, coord_kind, lon, lat = result
        config_ff = config._replace(flux_form_tracers=True)
        from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
        model_ff = MPASOceanModel(mesh, z_coord, config_ff)
        print(f"\n*** FLUX-FORM TRACERS ENABLED: {config_ff.flux_form_tracers} ***\n")
        return mesh, z_coord, config_ff, model_ff, coord_kind, lon, lat
    return result

otm._create_ocean_setup = _patched_setup

otm.main()
