"""Cloud-tuning CLI overrides (``--rh-crit`` / ``--q-c-diagnostic`` /
``--conv-cloud-max``) thread from ExperimentConfig through
``build_physics_pipeline`` into the hot-loop ``CloudConfig``.

These are the SW/LW knob for the coare3 moisture-driven planetary-albedo
overshoot (raise rh_crit / lower q_c_diagnostic => optically thinner / less
stratiform cloud).  Default ``None`` must leave the resolved pipeline at the
CloudConfig defaults (byte-identical to the validated path).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import make_hybrid_levels

NLEV = 10


def _config(**over):
    return ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=NLEV),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"),
        radiation="none", convection="none", turbulence="none",
        cloud_scheme="sundqvist", microphysics="none",
        **over,
    )


def test_cloudconfig_accepts_all_overrides() -> None:
    # The override target fields must exist on CloudConfig and round-trip.
    cc = CloudConfig(scheme="sundqvist", rh_crit=0.82,
                     q_c_diagnostic=3.0e-4, conv_cloud_max=0.18)
    assert (cc.rh_crit, cc.q_c_diagnostic, cc.conv_cloud_max) == (0.82, 3.0e-4, 0.18)


def test_pipeline_threads_cloud_overrides() -> None:
    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(NLEV)
    pipe = build_physics_pipeline(
        grid, sigma,
        _config(cloud_rh_crit=0.82, cloud_q_c_diagnostic=3.0e-4,
                cloud_conv_cloud_max=0.18),
    )
    assert pipe._cloud_rh_crit == 0.82
    assert pipe._cloud_q_c_diagnostic == 3.0e-4
    assert pipe._cloud_conv_cloud_max == 0.18


def test_pipeline_default_overrides_none() -> None:
    # Default config => None => pipeline leaves CloudConfig at its defaults.
    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(NLEV)
    pipe = build_physics_pipeline(grid, sigma, _config())
    assert pipe._cloud_rh_crit is None
    assert pipe._cloud_q_c_diagnostic is None
    assert pipe._cloud_conv_cloud_max is None
