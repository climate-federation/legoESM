"""Lat-lon C-grid face winds: checkpoint round trip and the lanes that refuse.

Checkpoints save the native face winds next to the cell-centre state, so a
restarted run continues exactly; a checkpoint written before they were saved
is lifted once (with a warning) and is close to, not equal to, the straight
run.  Lanes that cannot carry the face winds refuse to run.
"""
from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    LatLonFaceWindState)
from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.model_driver import ModelDriver

_DT = 600.0


def _cfg(out, days, **kw):
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=4),
        dycore=DycoreConfig(model_type="hydrostatic",
                            discretization="latlon_cgrid", dt=_DT,
                            fix_mass=True),
        output=OutputConfig(output_dir=str(out), diag_days=0,
                            checkpoint_days=1),
        days=days, dataset="analytical", held_suarez_forcing=True,
        turbulence="none", radiation="none", convection="none",
        microphysics="none", gravity_wave_drag="none", cloud_scheme="none",
        fix_moisture=True, distributed=False, n_devices=1,
        surface_bulk_scheme="none", **kw)


def _winds(d):
    return (np.asarray(d.state.u_f.data, np.float64),
            np.asarray(d.state.v_f.data, np.float64))


def _beside(ckpt, name):
    """An edited copy of ``ckpt`` in its own run directory: a restart reads the
    writing run's ``run_manifest.json`` beside the checkpoint, and refuses
    (version drift) when there is none."""
    return Path(ckpt).parent / name


def _resume(tmp_path, ckpt, tag):
    """Load ``ckpt`` and enter the face-wind lane exactly as ``run`` does."""
    d = ModelDriver(_cfg(tmp_path / tag, 2))
    d.setup()
    d.load_checkpoint(ckpt)
    d._enter_persistent_cgrid()
    return d


@pytest.fixture(scope="module")
def leg1(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("facewind")
    b1 = ModelDriver(_cfg(tmp / "leg1", 1))
    b1.setup()
    assert b1.run(compiled=True) == "COMPLETED"
    ckpt = sorted(glob.glob(str(tmp / "leg1" / "checkpoint_day_*.npz")))[-1]
    return tmp, b1, ckpt


def test_checkpoint_saves_face_winds_and_restart_resumes_them(leg1):
    """The resumed run starts from the SAVED face winds, not from a lift of
    the cell-centre winds.  (Whole-run restart equality is not asserted: the
    lat-lon compiled lane is not restart-exact even without face winds.)"""
    tmp, b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        assert any("latlon_u_face" in k for k in z.keys())
    d = _resume(tmp, ckpt, "leg2")
    assert isinstance(d.state, LatLonFaceWindState)
    for x, y in zip(_winds(b1), _winds(d)):
        np.testing.assert_array_equal(x, y)


def test_resume_through_run_uses_the_saved_faces(leg1, caplog):
    """run() enters the face-wind lane AFTER load_checkpoint staged the faces:
    no lift warning, and the faces leave the aux carry."""
    tmp, b1, ckpt = leg1
    d = ModelDriver(_cfg(tmp / "leg2_run", 2))
    d.setup()
    step, day = d.load_checkpoint(ckpt)
    with caplog.at_level(logging.WARNING):
        assert d.run(start_step=step, start_day=day,
                     compiled=True) == "COMPLETED"
    assert "no saved face winds" not in caplog.text
    assert isinstance(d.state, LatLonFaceWindState)
    assert not any(k in d._carry_aux for k in d._FACE_WIND_KEYS)


def test_old_checkpoint_is_lifted_once_with_a_warning(leg1, caplog):
    tmp, b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        data = {k: z[k] for k in z.keys() if "latlon_" not in k}
    old = _beside(ckpt, "old_checkpoint_day_0001.npz")
    np.savez(old, **data)
    with caplog.at_level(logging.WARNING):
        d = _resume(tmp, str(old), "leg2_old")
    assert "no saved face winds" in caplog.text
    # A lift of the averaged winds cannot give back the carried faces.
    du = np.abs(_winds(b1)[0] - _winds(d)[0]).max()
    assert du > 0.0, du


def test_zarr_restart_lifts_face_winds_with_a_warning(tmp_path, caplog):
    """zarr does not round-trip carry_aux, so the faces are lifted on restart
    and the warning says so (whether zarr should refuse instead is open)."""
    cfg = _cfg(tmp_path / "z1", 1)
    cfg = cfg._replace(output=cfg.output._replace(checkpoint_format="zarr"))
    z1 = ModelDriver(cfg)
    z1.setup()
    assert z1.run(compiled=True) == "COMPLETED"
    ckpt = sorted(glob.glob(str(tmp_path / "z1" / "checkpoint_day_*")))[-1]
    with caplog.at_level(logging.WARNING):
        d = _resume(tmp_path, ckpt, "z2")
    assert "no saved face winds (a zarr" in caplog.text
    assert np.abs(_winds(z1)[0] - _winds(d)[0]).max() > 0.0


@pytest.mark.parametrize("corrupt", [
    lambda a: a * 1.1,
    # mean-preserving: every face shifted one cell east
    lambda a: np.roll(a, 1, axis=1),
])
def test_faces_from_another_state_are_refused(leg1, corrupt):
    tmp, b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        data = {k: z[k] for k in z.keys()}
    key = next(k for k in data if k.endswith("latlon_u_face"))
    data[key] = corrupt(data[key])
    for k in [k for k in data if k.endswith("_sha256")]:
        data.pop(k)                 # a pre-digest file: the average check
    bad = _beside(ckpt, "bad_checkpoint_day_0001.npz")
    np.savez(bad, **data)
    d = ModelDriver(_cfg(tmp / "leg2_bad", 2))
    d.setup()
    step, day = d.load_checkpoint(str(bad))
    for _ in range(2):          # a retried run must refuse again, not lift
        with pytest.raises(ValueError, match="do not average"):
            d.run(start_step=step, start_day=day, compiled=True)
        assert all(k in d._carry_aux for k in d._FACE_WIND_KEYS)
        assert not isinstance(d.state, LatLonFaceWindState)


def test_unsupported_lanes_refuse(tmp_path):
    ens = ModelDriver(_cfg(tmp_path / "ens", 1, ensemble_size=2))
    ens.setup()
    with pytest.raises(SystemExit, match="ensemble"):
        ens.run(compiled=True)

    trn = ModelDriver(_cfg(tmp_path / "trn", 1))
    trn.setup()
    with pytest.raises(NotImplementedError, match="face"):
        trn.build_training_segment(4)


def test_checkpoint_with_one_face_wind_is_refused(leg1):
    tmp, _b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        data = {k: z[k] for k in z.keys() if "latlon_v_face" not in k}
    half = _beside(ckpt, "half_checkpoint_day_0001.npz")
    np.savez(half, **data)
    with pytest.raises(ValueError, match="only one of the face winds"):
        _resume(tmp, str(half), "leg2_half")


def test_resume_into_sfno_drops_the_faces(tmp_path):
    """A same-shape SFNO Gaussian run (10x20 lat-lon vs T5) loads a lat-lon
    checkpoint but does not carry face winds; it must not re-save them next
    to its newer state."""
    c = _cfg(tmp_path / "ll", 1)
    a = ModelDriver(c._replace(grid=c.grid._replace(resolution=10)))
    a.setup()
    assert a.run(compiled=True) == "COMPLETED"
    ckpt = sorted(glob.glob(str(tmp_path / "ll" / "checkpoint_day_*.npz")))[-1]
    c = _cfg(tmp_path / "sfno", 2)
    d = ModelDriver(c._replace(
        grid=GridConfig(grid_type="gaussian", resolution=5, nlev=4),
        dycore=c.dycore._replace(discretization="sfno", energy_fixer=False)))
    d.setup()
    step, day = d.load_checkpoint(ckpt)
    d._enter_persistent_cgrid()
    d.save_checkpoint(step, day)
    new = sorted(glob.glob(str(tmp_path / "sfno" / "checkpoint_day_*.npz")))[-1]
    with np.load(new, allow_pickle=True) as z:
        assert not any("latlon_" in k for k in z.keys())


def test_load_then_save_keeps_the_saved_winds(leg1):
    """Load, enter the staggered lane, load again and save without running:
    the second load must not leave the lane flagged on a cell-centre state."""
    tmp, _b1, ckpt = leg1
    d = _resume(tmp, ckpt, "leg2_resave")
    assert d._persistent_cgrid_active()
    step, day = d.load_checkpoint(ckpt)
    d.save_checkpoint(step, day)
    new = sorted(glob.glob(str(tmp / "leg2_resave" / "checkpoint_day_*.npz")))[-1]
    with np.load(ckpt, allow_pickle=True) as a, np.load(new, allow_pickle=True) as b:
        for k in ModelDriver._FACE_WIND_KEYS:
            np.testing.assert_array_equal(a[f"carry_{k}"], b[f"carry_{k}"])
