"""Lat-lon 'finite_volume' and 'centered' build the same C-grid solver as
'latlon_cgrid', so they carry the face winds between steps too (and are
refused where that lane is refused); before, they silently re-projected the
winds every step."""
from __future__ import annotations

import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    LatLonFaceWindState)
from legoesm.driver.component_factory import driver_solver_name
from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.model_driver import ModelDriver


def _cfg(out, disc):
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=4),
        dycore=DycoreConfig(model_type="hydrostatic", discretization=disc,
                            dt=600.0, fix_mass=True),
        output=OutputConfig(output_dir=str(out), diag_days=0),
        days=1, dataset="analytical", held_suarez_forcing=True,
        turbulence="none", radiation="none", convection="none",
        microphysics="none", gravity_wave_drag="none", cloud_scheme="none",
        fix_moisture=True, distributed=False, n_devices=1,
        surface_bulk_scheme="none")


@pytest.mark.parametrize("disc", ["finite_volume", "centered", "latlon_cgrid"])
def test_every_spelling_of_the_cgrid_solver_carries_face_winds(tmp_path, disc):
    cfg = _cfg(tmp_path / disc, disc)
    assert driver_solver_name(cfg) == "latlon_cgrid_primitive_equations"
    d = ModelDriver(cfg)
    d.setup()
    assert d.run(compiled=True) == "COMPLETED"
    assert isinstance(d.state, LatLonFaceWindState)
    assert d._persistent_cgrid_active()


@pytest.mark.parametrize("disc", ["finite_volume", "centered"])
def test_training_segment_refuses_the_aliases_too(tmp_path, disc):
    d = ModelDriver(_cfg(tmp_path / disc, disc))
    d.setup()
    with pytest.raises(NotImplementedError, match="face winds"):
        d.build_training_segment(n_steps=1)


def test_alias_restart_resumes_the_saved_face_winds(tmp_path):
    import glob
    import numpy as np
    c1 = _cfg(tmp_path / "a", "finite_volume")
    c1 = c1._replace(output=c1.output._replace(checkpoint_days=1))
    a = ModelDriver(c1)
    a.setup()
    assert a.run(compiled=True) == "COMPLETED"
    ckpt = sorted(glob.glob(str(tmp_path / "a" / "checkpoint_day_*.npz")))[-1]
    c2 = _cfg(tmp_path / "b", "finite_volume")
    b = ModelDriver(c2._replace(days=2))
    b.setup()
    b.load_checkpoint(ckpt)
    b._enter_persistent_cgrid()
    assert isinstance(b.state, LatLonFaceWindState)
    for f in ("u_f", "v_f"):
        np.testing.assert_array_equal(np.asarray(getattr(a.state, f).data),
                                      np.asarray(getattr(b.state, f).data))
