"""An MPAS checkpoint written before MG2 two-moment snow (no ``N_s`` tracer)
is refused on restart (oracle row 22, user pick 'hard error'), neither
seeded nor silently run on single-moment snow."""
from __future__ import annotations

import glob
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


def _driver(tmpdir: str) -> ModelDriver:
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=20,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=DT),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=1),
        days=TWO_STEPS_DAYS, dataset="analytical", radiation="rrtmgp",
        microphysics="morrison", morrison_flavor="mg",
        morrison_warm_rain_scheme="kk2000",
        # CAM6 MG2 mode (the only mode that carries N_s) needs the CLUBB host
        morrison_warm_rain_incloud=True, turbulence="clubb",
        cloud_scheme="cam6_clubb", use_clubb_cloud_fraction=True,
        cld_macmic_num_steps=2,
        subgrid_autoconversion=False,
        convection="none", precision="fp64",
        distributed=False,
    )
    d = ModelDriver(cfg, output_dir=tmpdir)
    d.setup()
    return d


def test_old_checkpoint_without_snow_number_is_refused(tmp_path):
    dA = _driver(str(tmp_path / "a"))
    assert "N_s" in dA.tracer_registry.names
    assert dA.run() == "COMPLETED"
    ckpt = sorted(glob.glob(str(tmp_path / "a" / "checkpoint_day_*.npz")))[-1]
    with np.load(ckpt) as d:
        payload = {k: d[k] for k in d.files}
    assert "trc_N_s" in payload
    # what a pre-two-moment-snow build wrote: no N_s slot at all
    del payload["trc_N_s"]
    payload["tracer_names"] = np.asarray(
        [n for n in payload["tracer_names"] if str(n) != "N_s"])
    old = os.path.join(str(tmp_path), "old.npz")
    np.savez(old, **payload)
    # the restart version gate reads the writing run's manifest beside it
    shutil.copy(os.path.join(str(tmp_path / "a"), "run_manifest.json"),
                str(tmp_path))

    dB = _driver(str(tmp_path / "b"))
    with pytest.raises(ValueError, match="no snow number N_s"):
        dB.load_checkpoint(old)
    # a checkpoint with no tracer list at all is refused too
    bare = {k: v for k, v in payload.items()
            if k != "tracer_names" and not k.startswith("trc_")}
    old_bare = os.path.join(str(tmp_path), "old_bare.npz")
    np.savez(old_bare, **bare)
    with pytest.raises(ValueError, match="no snow number N_s"):
        dB.load_checkpoint(old_bare)
    # the unmodified checkpoint (with N_s) still restarts
    dC = _driver(str(tmp_path / "c"))
    step, day = dC.load_checkpoint(ckpt)
    assert "N_s" in dC.state.tracers
