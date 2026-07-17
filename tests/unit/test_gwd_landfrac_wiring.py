"""E3SM GWD landfrac wiring: the coupled pipeline threads its ``f_land``
(set by the model driver next to ``subgrid_topo_stddev``) into the GWD call
for exactly the ``e3sm_cam`` scheme (the only kernel accepting
``land_frac_col``; E3SM gw_drag.F90:904-906 oro landfrac scaling).

Guards the codex-flagged phantom: an extractor reading an attribute nothing
sets.  The builder flag below is what makes the production coupled path
actually deliver the land fraction.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
)
from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate


def _config(gwd="e3sm_cam"):
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray",
        gravity_wave_drag=gwd,
    )


def _pipe(gwd):
    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    return build_physics_pipeline(grid, sigma, _config(gwd))


def test_e3sm_cam_pipeline_marks_land_frac_capable():
    """gravity_wave_drag='e3sm_cam' -> the pipeline will thread f_land as
    ``land_frac_col`` (once the driver sets ``pipe.f_land``)."""
    assert _pipe("e3sm_cam")._gwd_takes_land_frac is True


def test_other_gwd_schemes_never_send_land_frac():
    """mcfarlane/lindzen kernels do NOT accept ``land_frac_col`` — sending it
    would TypeError in production.  The builder flag must be False for every
    non-e3sm_cam scheme (including 'none')."""
    for scheme in ("mcfarlane", "lindzen", "none"):
        assert _pipe(scheme)._gwd_takes_land_frac is False, scheme


def test_f_land_default_none_no_kwarg():
    """Fresh pipeline has ``f_land=None`` (the driver populates it only when
    land topography is active) -> the GWD branch sends no ``land_frac_col``
    and the kernel default (no scaling) applies — legacy bit-identity."""
    assert _pipe("e3sm_cam").f_land is None
