"""An MPAS checkpoint written before MG2 two-moment snow (no ``N_s`` tracer)
restarts with the snow number seeded at zero (user decision 2026-10-20),
not silently on single-moment snow."""
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


def test_old_checkpoint_without_snow_number_seeds_it_at_zero(tmp_path):
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
    # carried snow with zero number in sub-freezing air: the state the seed
    # actually creates (the analytical IC is too warm to hold snow)
    cold = payload["T"] <= np.quantile(payload["T"], 0.2)
    payload["T"] = np.where(cold, 255.0, payload["T"])
    payload["trc_q_s"] = np.where(cold, 1.0e-4, payload["trc_q_s"])
    old = os.path.join(str(tmp_path), "old.npz")
    np.savez(old, **payload)
    # the restart version gate reads the writing run's manifest beside it
    shutil.copy(os.path.join(str(tmp_path / "a"), "run_manifest.json"),
                str(tmp_path))

    dB = _driver(str(tmp_path / "b"))
    step, day = dB.load_checkpoint(old)
    assert "N_s" in dB.state.tracers
    n_s = np.asarray(dB.state.tracers["N_s"].data)
    assert n_s.shape == np.asarray(dB.state.tracers["q_v"].data).shape
    assert np.all(n_s == 0.0)
    assert dB.run(start_step=step, start_day=day) == "COMPLETED"
    assert "N_s" in dB.state.tracers
    for name in ("N_s", "q_s", "q_v"):
        assert np.all(np.isfinite(np.asarray(dB.state.tracers[name].data))), name
    # the carried-snow, zero-number state steps without NaN
    assert np.all(np.isfinite(np.asarray(dB.state.T.data)))
