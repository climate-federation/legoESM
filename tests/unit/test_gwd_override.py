"""``gravity_wave_drag_override``: the coupled-path route to nested GWD
scheme options, mirroring ``turbulence_override``.

Codex wave-4 P2: ``_resolve_gwd`` rebuilt ``GravityWaveDragConfig`` from the
scheme STRING alone, silently discarding every nested option
(``mcfarlane.use_e3sm_hdsp``, ``e3sm_cam.use_discrete_ke_heating``, tuned
``fcrit2``) — YAML/CLI users could never enable the faithfulness flags the
GWD program ships default-OFF.
"""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    E3SMCAMConfig,
    GravityWaveDragConfig,
    McFarlaneConfig,
)
from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
)
from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate


def _config(gwd="mcfarlane", override=None):
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray",
        gravity_wave_drag=gwd,
        gravity_wave_drag_override=override,
    )


def _pipe(gwd, override=None):
    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    return build_physics_pipeline(grid, sigma, _config(gwd, override))


def test_override_reaches_built_pipeline_mcfarlane_flag():
    """use_e3sm_hdsp=True survives into the pipeline's resolved GWD config."""
    over = GravityWaveDragConfig(
        scheme="mcfarlane",
        mcfarlane=McFarlaneConfig(use_e3sm_hdsp=True),
    )
    pipe = _pipe("mcfarlane", over)
    assert pipe.gwd_config.use_e3sm_hdsp is True


def test_override_reaches_built_pipeline_e3sm_flag():
    """e3sm_cam.use_discrete_ke_heating=True survives too (wave-3 flag)."""
    over = GravityWaveDragConfig(
        scheme="e3sm_cam",
        e3sm_cam=E3SMCAMConfig(use_discrete_ke_heating=True),
    )
    pipe = _pipe("e3sm_cam", over)
    assert pipe.gwd_config.use_discrete_ke_heating is True


def test_override_none_is_default_byte_identical():
    """No override ⇒ exactly the scheme-string default config (legacy)."""
    pipe = _pipe("mcfarlane", None)
    assert pipe.gwd_config == GravityWaveDragConfig(scheme="mcfarlane").mcfarlane


def test_validate_strict_rejects_scheme_mismatch():
    over = GravityWaveDragConfig(scheme="lindzen")
    cfg = _config("mcfarlane", over)
    with pytest.raises(ValueError, match="gravity_wave_drag_override.scheme"):
        cfg.validate_strict()


def test_validate_strict_rejects_wrong_type():
    cfg = _config("mcfarlane", McFarlaneConfig())
    with pytest.raises(
        ValueError, match="must be a GravityWaveDragConfig"
    ):
        cfg.validate_strict()


def test_validate_strict_accepts_matching_override():
    over = GravityWaveDragConfig(
        scheme="mcfarlane", mcfarlane=McFarlaneConfig(use_e3sm_hdsp=True),
    )
    _config("mcfarlane", over).validate_strict()
