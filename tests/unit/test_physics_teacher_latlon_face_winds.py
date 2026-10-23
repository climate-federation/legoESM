"""The physics teacher snapshot reads cell-centre winds from a lat-lon C-grid
run, which carries face winds between steps."""
from __future__ import annotations

import jax
import numpy as np

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    LatLonFaceWindState)
from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.model_driver import ModelDriver
from legoesm.ml.physics.data import capture_physics_teacher_snapshot

jax.config.update("jax_enable_x64", True)


def test_teacher_snapshot_on_face_wind_state(tmp_path):
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(model_type="hydrostatic",
                            discretization="latlon_cgrid", dt=600.0),
        output=OutputConfig(output_dir=str(tmp_path), diag_days=0),
        radiation="gray", days=1, dataset="analytical")
    d = ModelDriver(cfg)
    d.setup()
    assert d.run(compiled=True) == "COMPLETED"
    assert isinstance(d.state, LatLonFaceWindState)
    snap = capture_physics_teacher_snapshot(d, 1.0, 600.0)
    u = d.physics.adapter.flatten_3d(d.cell_centre_state().u.data)
    np.testing.assert_array_equal(np.asarray(snap.columns.u), np.asarray(u))
