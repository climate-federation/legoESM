"""convection_config_for: tuned ExperimentConfig fields reach combined lanes.

The MPAS/spectral lanes previously built ``ConvectionConfig(scheme=...)``
with bare defaults, so ``bechtold_*`` / ``sbm_*`` /
``convective_precip_efficiency`` tuning silently never reached them (the
2026-07-23 stack-switch discovery). These tests pin the shared resolver
wrapper.
"""

from __future__ import annotations

from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
)
from legoesm.driver.physics_pipeline import convection_config_for


def _cfg(**kw):
    return ExperimentConfig(
        grid=GridConfig(resolution=8, nlev=8),
        dycore=DycoreConfig(dt=600.0),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical", radiation="gray",
        **kw,
    )


def test_bechtold_tuning_reaches_leaf():
    cc = convection_config_for(_cfg(
        convection="bechtold",
        bechtold_cape_threshold=65.0,
        convective_precip_efficiency=0.8,
    ))
    assert cc.scheme == "bechtold"
    assert cc.bechtold.cape_threshold == 65.0
    assert cc.bechtold.precip_efficiency == 0.8


def test_sbm_tuning_reaches_leaf():
    cc = convection_config_for(_cfg(convection="sbm", sbm_tau_c=3600.0,
                                    sbm_RH_ref=0.75))
    assert cc.scheme == "sbm"
    assert cc.sbm.tau_c == 3600.0
    assert cc.sbm.rh_ref == 0.75


def test_none_passthrough():
    cc = convection_config_for(_cfg(convection="none"))
    assert cc.scheme == "none"


def test_bechtold_grid_dx_autofill():
    # Sentinel dx_m=0.0 + caller grid spacing -> ZTAURES dx filled in.
    cc = convection_config_for(_cfg(convection="bechtold"), grid_dx_m=223e3)
    assert cc.bechtold.dx_m == 223e3


def test_bechtold_explicit_dx_wins_over_grid():
    cc = convection_config_for(
        _cfg(convection="bechtold", bechtold_dx_m=50e3), grid_dx_m=223e3)
    assert cc.bechtold.dx_m == 50e3


def test_grid_dx_ignored_for_non_bechtold():
    cc = convection_config_for(_cfg(convection="sbm"), grid_dx_m=223e3)
    assert cc.scheme == "sbm"
    assert not hasattr(cc.sbm, "dx_m")


def test_pipeline_fills_bechtold_dx_from_grid():
    """F27 (review 2026-10-10): the PhysicsPipeline lanes (lat-lon / cube) fill
    Bechtold's dx_m from the grid like the column lanes, not the 0 sentinel."""
    import numpy as np
    from legoesm.driver.physics_pipeline import (
        build_physics_pipeline, mean_grid_spacing_m)
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_cubed_sphere(8)
    sigma = create_sigma_coordinate(8)
    pipe = build_physics_pipeline(grid, sigma, _cfg(convection="bechtold"))
    dx = float(np.sqrt(np.mean(np.asarray(grid.area))))
    assert dx > 1.0e5
    assert pipe.convection_config.dx_m == mean_grid_spacing_m(grid) == dx
    explicit = build_physics_pipeline(
        grid, sigma, _cfg(convection="bechtold", bechtold_dx_m=50e3))
    assert explicit.convection_config.dx_m == 50e3
