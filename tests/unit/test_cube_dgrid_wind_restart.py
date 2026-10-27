"""Cube D-grid winds: checkpoint round trip.

With persistent D-grid winds the checkpoint saves the corner winds next to
the cell-centre state, so a restarted run continues from exactly the winds
the straight run carried; a checkpoint written before they were saved is
lifted once, with a warning.
"""
from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    FV3HydrostaticState)
from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.model_driver import ModelDriver


def _cfg(out, days):
    return ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=8, nlev=4),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid",
                            dt=1200.0, fix_mass=True, persistent_dgrid=True),
        output=OutputConfig(output_dir=str(out), diag_days=0,
                            checkpoint_days=1),
        days=days, dataset="analytical", held_suarez_forcing=True,
        turbulence="none", radiation="none", convection="none",
        microphysics="none", gravity_wave_drag="none", cloud_scheme="none",
        fix_moisture=True, distributed=False, n_devices=1,
        surface_bulk_scheme="none")


def _winds(d):
    return tuple(np.asarray(getattr(d.state, f).data, np.float64)
                 for f in ("u_d", "v_d"))


def _beside(ckpt, name):
    """An edited copy of ``ckpt`` in its run directory, next to the run
    manifest the restart version check reads."""
    return Path(ckpt).parent / name


def _resume(tmp, ckpt, tag):
    d = ModelDriver(_cfg(tmp / tag, 2))
    d.setup()
    d.load_checkpoint(ckpt)
    d._enter_persistent_dgrid()
    return d


@pytest.fixture(scope="module")
def leg1(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cubewind")
    b1 = ModelDriver(_cfg(tmp / "leg1", 1))
    b1.setup()
    assert b1.run() == "COMPLETED"
    ckpt = sorted(glob.glob(str(tmp / "leg1" / "checkpoint_day_*.npz")))[-1]
    return tmp, b1, ckpt


def test_restart_resumes_the_saved_corner_winds(leg1):
    tmp, b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        assert any("cube_u_d" in k for k in z.keys())
    d = _resume(tmp, ckpt, "leg2")
    assert isinstance(d.state, FV3HydrostaticState)
    for x, y in zip(_winds(b1), _winds(d)):
        np.testing.assert_array_equal(x, y)
    assert not any(k in d._carry_aux for k in d._CORNER_WIND_KEYS)


def test_two_legs_equal_the_straight_run(leg1):
    tmp, _b1, ckpt = leg1
    straight = ModelDriver(_cfg(tmp / "straight", 2))
    straight.setup()
    assert straight.run() == "COMPLETED"
    leg2 = ModelDriver(_cfg(tmp / "leg2_run", 2))
    leg2.setup()
    step, day = leg2.load_checkpoint(ckpt)
    assert leg2.run(start_step=step, start_day=day) == "COMPLETED"
    for x, y in zip(_winds(straight), _winds(leg2)):
        np.testing.assert_array_equal(x, y)
    for f in ("T", "p_s"):
        np.testing.assert_array_equal(np.asarray(getattr(straight.state, f).data),
                                      np.asarray(getattr(leg2.state, f).data))


def test_old_checkpoint_is_lifted_once_with_a_warning(leg1, caplog):
    tmp, b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        drop = {f"carry_{k}{x}" for k in ModelDriver._CORNER_WIND_KEYS
                for x in ("", "_sha256")}
        assert drop <= set(z.keys())
        data = {k: z[k] for k in z.keys() if k not in drop}
    old = _beside(ckpt, "old_checkpoint_day_0001.npz")
    np.savez(old, **data)
    with caplog.at_level(logging.WARNING):
        d = _resume(tmp, str(old), "leg2_old")
    assert "no saved D-grid winds" in caplog.text
    assert np.abs(_winds(b1)[0] - _winds(d)[0]).max() > 0.0


def test_zarr_restart_lifts_with_a_warning(tmp_path, caplog):
    """zarr does not round-trip the aux carry, so the corner winds are lifted
    on restart and the warning says so (whether zarr should refuse is open)."""
    cfg = _cfg(tmp_path / "z1", 1)
    cfg = cfg._replace(output=cfg.output._replace(checkpoint_format="zarr"))
    z1 = ModelDriver(cfg)
    z1.setup()
    assert z1.run() == "COMPLETED"
    ckpt = sorted(glob.glob(str(tmp_path / "z1" / "checkpoint_day_*")))[-1]
    with caplog.at_level(logging.WARNING):
        d = _resume(tmp_path, ckpt, "z2")
    assert "no saved D-grid winds (a zarr" in caplog.text
    assert np.abs(_winds(z1)[0] - _winds(d)[0]).max() > 0.0


@pytest.mark.parametrize("corrupt", [
    lambda a: a * 1.1,
    lambda a: np.roll(a, 1, axis=-2),
])
def test_corner_winds_from_another_state_are_refused(leg1, corrupt):
    tmp, _b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        data = {k: z[k] for k in z.keys()}
    key = next(k for k in data if k.endswith("cube_u_d"))
    data[key] = corrupt(data[key])
    for k in [k for k in data if k.endswith("_sha256")]:
        data.pop(k)                 # a pre-digest file: the average check
    bad = _beside(ckpt, "bad_checkpoint_day_0001.npz")
    np.savez(bad, **data)
    d = ModelDriver(_cfg(tmp / "leg2_bad", 2))
    d.setup()
    d.load_checkpoint(str(bad))
    for _ in range(2):          # a retried entry must refuse again, not lift
        with pytest.raises(ValueError, match="do not average"):
            d._enter_persistent_dgrid()
        assert all(k in d._carry_aux for k in d._CORNER_WIND_KEYS)


def test_corruption_that_averages_out_is_refused(leg1):
    """A checkerboard on the corners cancels in the corner -> centre average,
    so only the saved digest can catch it."""
    tmp, _b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        data = {k: z[k] for k in z.keys()}
    key = f"carry_{ModelDriver._CORNER_WIND_KEYS[0]}"
    i = np.arange(data[key].shape[1])[:, None]
    j = np.arange(data[key].shape[2])[None, :]
    data[key] = data[key] + (np.where((i + j) % 2 == 0, 1.0, -1.0)[None, :, :, None]
                             ).astype(data[key].dtype)
    bad = _beside(ckpt, "checker_checkpoint_day_0001.npz")
    np.savez(bad, **data)
    d = ModelDriver(_cfg(tmp / "leg2_checker", 2))
    d.setup()
    d.load_checkpoint(str(bad))
    with pytest.raises(ValueError, match="SHA-256"):
        d._enter_persistent_dgrid()


def test_resume_with_the_flag_off_runs(leg1):
    tmp, _b1, ckpt = leg1
    c = _cfg(tmp / "leg2_off", 2)
    d = ModelDriver(c._replace(dycore=c.dycore._replace(persistent_dgrid=False)))
    d.setup()
    step, day = d.load_checkpoint(ckpt)
    # per-step: the lane whose checkpoint re-writes the loaded aux carry
    assert d.run(start_step=step, start_day=day, compiled=False) == "COMPLETED"
    # the stale corner winds are not re-saved next to the newer state
    new = sorted(glob.glob(str(tmp / "leg2_off" / "checkpoint_day_*.npz")))[-1]
    with np.load(new, allow_pickle=True) as z:
        assert not any("cube_" in k for k in z.keys())


def test_float64_winds_are_checked_bit_for_bit(leg1):
    """Stored float64 winds keep their dtype on load and their digest covers
    every bit: the untouched file resumes, a one-ulp edit is refused."""
    tmp, b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        data = {k: z[k] for k in z.keys()}
    for k in ModelDriver._CORNER_WIND_KEYS:
        data[f"carry_{k}"] = data[f"carry_{k}"].astype(np.float64)
        data[f"carry_{k}_sha256"] = np.asarray(
            ModelDriver._native_wind_digest(data[f"carry_{k}"]))
    good = _beside(ckpt, "f64_checkpoint_day_0001.npz")
    np.savez(good, **data)
    d = _resume(tmp, str(good), "leg2_f64")
    for x, y in zip(_winds(b1), _winds(d)):
        np.testing.assert_array_equal(x, y)
    key = f"carry_{ModelDriver._CORNER_WIND_KEYS[0]}"
    data[key] = data[key].copy()
    data[key].flat[7] = np.nextafter(data[key].flat[7], np.inf)
    bad = _beside(ckpt, "f64ulp_checkpoint_day_0001.npz")
    np.savez(bad, **data)
    d = ModelDriver(_cfg(tmp / "leg2_f64ulp", 2))
    d.setup()
    d.load_checkpoint(str(bad))
    with pytest.raises(ValueError, match="SHA-256"):
        d._enter_persistent_dgrid()


def test_one_digest_missing_is_refused(leg1):
    tmp, _b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        data = {k: z[k] for k in z.keys()}
    data.pop(f"carry_{ModelDriver._CORNER_WIND_KEYS[0]}_sha256")
    bad = _beside(ckpt, "halfdigest_checkpoint_day_0001.npz")
    np.savez(bad, **data)
    d = ModelDriver(_cfg(tmp / "leg2_half", 2))
    d.setup()
    d.load_checkpoint(str(bad))
    with pytest.raises(ValueError, match="only one"):
        d._enter_persistent_dgrid()


def test_digests_without_winds_are_refused(leg1):
    tmp, _b1, ckpt = leg1
    with np.load(ckpt, allow_pickle=True) as z:
        data = {k: z[k] for k in z.keys()
                if k not in {f"carry_{k}" for k in ModelDriver._CORNER_WIND_KEYS}}
    bad = _beside(ckpt, "orphan_checkpoint_day_0001.npz")
    np.savez(bad, **data)
    d = ModelDriver(_cfg(tmp / "leg2_orphan", 2))
    d.setup()
    d.load_checkpoint(str(bad))
    with pytest.raises(ValueError, match="damaged"):
        d._enter_persistent_dgrid()


def test_load_then_save_keeps_the_saved_winds(leg1):
    """Load, enter the staggered lane, load again and save without running:
    the second load must not leave the lane flagged on a cell-centre state."""
    tmp, _b1, ckpt = leg1
    d = _resume(tmp, ckpt, "leg2_resave")
    assert d._persistent_dgrid_active()
    step, day = d.load_checkpoint(ckpt)
    d.save_checkpoint(step, day)
    new = sorted(glob.glob(str(tmp / "leg2_resave" / "checkpoint_day_*.npz")))[-1]
    with np.load(ckpt, allow_pickle=True) as a, np.load(new, allow_pickle=True) as b:
        for k in ModelDriver._CORNER_WIND_KEYS:
            np.testing.assert_array_equal(a[f"carry_{k}"], b[f"carry_{k}"])


def test_failed_load_then_save_keeps_the_carried_winds(leg1):
    tmp, _b1, ckpt = leg1
    d = _resume(tmp, ckpt, "leg2_failload")
    with pytest.raises(Exception):
        d.load_checkpoint(str(_beside(ckpt, "missing_checkpoint_day_0001.npz")))
    d.save_checkpoint(1, 1.0)
    new = sorted(glob.glob(str(tmp / "leg2_failload" / "checkpoint_day_*.npz")))[-1]
    with np.load(new, allow_pickle=True) as z:
        for k, x in zip(ModelDriver._CORNER_WIND_KEYS, _winds(d)):
            np.testing.assert_array_equal(z[f"carry_{k}"], x)
