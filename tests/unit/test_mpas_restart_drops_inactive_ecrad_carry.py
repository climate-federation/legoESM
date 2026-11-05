"""A run without radiation='ecrad_ifs' that restores an MPAS checkpoint
holding the ecRad carry (``physstate_ecrad_carry``) drops it with a warning
naming the field (vote U2b), and resumes."""
from __future__ import annotations

import glob
import logging
import os
import shutil

import numpy as np
import pytest

pytest.importorskip("jax")

from legoesm.driver.config import (  # noqa: E402
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver  # noqa: E402

DT = 300.0
TWO_STEPS_DAYS = 601.0 / 86400.0


def _driver(tmpdir: str, days: float) -> ModelDriver:
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=8,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="mpas", dt=DT),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=1),
        days=days, dataset="analytical", distributed=False,
        n_devices=1, turbulence="louis", radiation="none",
        gravity_wave_drag="none", cloud_scheme="none", precision="fp64")
    d = ModelDriver(cfg, output_dir=tmpdir)
    d.setup()
    return d


@pytest.mark.slow
def test_ecrad_carry_in_checkpoint_is_dropped_loudly(tmp_path, caplog):
    dA = _driver(str(tmp_path / "a"), TWO_STEPS_DAYS)
    assert dA.run() == "COMPLETED"
    ckpt = sorted(glob.glob(str(tmp_path / "a" / "checkpoint_day_*.npz")))[-1]
    with np.load(ckpt) as d:
        payload = {k: d[k] for k in d.files}
    assert "physstate_rad_heating" in payload            # a carry was saved
    assert "physstate_ecrad_carry" not in payload        # None is never saved
    ncol = payload["physstate_rad_heating"].shape[0]
    planted = dict(payload, physstate_ecrad_carry=np.ones((ncol, 58)))
    old = os.path.join(str(tmp_path), "with_ecrad_carry.npz")
    np.savez(old, **planted)
    shutil.copy(os.path.join(str(tmp_path / "a"), "run_manifest.json"), str(tmp_path))
    dB = _driver(str(tmp_path / "b"), 2.0 * TWO_STEPS_DAYS)
    dB.load_checkpoint(old)
    with caplog.at_level(logging.INFO):
        assert dB.run() == "COMPLETED"
    hits = [r.getMessage() for r in caplog.records
            if "DROPPED physics-state field" in r.getMessage()]
    assert len(hits) == 1 and "'ecrad_carry'" in hits[0], caplog.text
    assert dB._mpas_phys_state.ecrad_carry is None
    # every other saved field still restored ...
    restored = [r.getMessage() for r in caplog.records
                if "Restored physics state from checkpoint" in r.getMessage()]
    assert restored and "rad_heating" in restored[0], caplog.text
    # ... and the resumed run is bitwise the one resumed from the original
    dC = _driver(str(tmp_path / "c"), 2.0 * TWO_STEPS_DAYS)
    dC.load_checkpoint(ckpt)
    assert dC.run() == "COMPLETED"
    assert np.array_equal(np.asarray(dB.state.T.data), np.asarray(dC.state.T.data))
