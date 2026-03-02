"""Kessler warm-rain microphysics for legoESM — backward-compatibility wrapper.

This module delegates to the new microphysics package while preserving
the original API used by test_compressible_euler.py and dcmip2025/test_case_3.py.

The actual implementation now lives in
``legoesm.atmosphere.physics.microphysics.kessler``.
"""

from __future__ import annotations

from legoesm.atmosphere.physics.microphysics.config import (
    KesslerConfig,
    MicrophysicsConfig,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)

# Re-export saturation_mixing_ratio (used by dcmip2025/test_case_3.py)
from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio  # noqa: F401


def kessler_tendencies(state, grid, height_coord, terrain_metric,
                       config=KesslerConfig()):
    """Compute Kessler microphysics tendencies (backward-compat wrapper).

    Delegates to the new microphysics integration bridge.

    Parameters
    ----------
    state : NonHydrostaticState
    grid : CubedSphereGrid
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    config : KesslerConfig

    Returns
    -------
    NonHydrostaticTendencies
    """
    micro_config = MicrophysicsConfig(scheme="kessler", kessler=config)
    physics_fn = make_microphysics_physics(
        micro_config, model_type="nonhydrostatic", dt=1.0,
    )
    return physics_fn(state, grid, height_coord, terrain_metric)
