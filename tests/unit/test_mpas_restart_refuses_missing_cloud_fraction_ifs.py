"""An MPAS checkpoint without IFS CLOUDSC's advected cloud fraction
(``cloud_fraction_ifs``) is refused on restart of an ifs_cloudsc run (vote
VC26, the N_s rule), neither seeded nor run without it."""
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
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=8,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="mpas", dt=DT),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=1),
        days=TWO_STEPS_DAYS, dataset="analytical", distributed=False,
        n_devices=1, turbulence="louis", radiation="none",
        convection="bechtold", bechtold_use_ifs_ascent=True,
        microphysics="ifs_cloudsc", gravity_wave_drag="none",
        cloud_scheme="none", cld_macmic_num_steps=1, precision="fp64")
    d = ModelDriver(cfg, output_dir=tmpdir)
    d.setup()
    return d


@pytest.mark.slow
def test_old_checkpoint_without_cloud_fraction_ifs_is_refused(tmp_path):
    dA = _driver(str(tmp_path / "a"))
    assert "cloud_fraction_ifs" in dA.tracer_registry.names
    # no-op callback: switches on the per-step forcing build that hands the
    # IFS convection its surface fluxes (the analytical case has no SST)
    assert dA.run(segment_callback=lambda drv, day, dt: None) == "COMPLETED"
    ckpt = sorted(glob.glob(str(tmp_path / "a" / "checkpoint_day_*.npz")))[-1]
    with np.load(ckpt) as d:
        payload = {k: d[k] for k in d.files}
    assert "trc_cloud_fraction_ifs" in payload
    del payload["trc_cloud_fraction_ifs"]
    payload["tracer_names"] = np.asarray(
        [n for n in payload["tracer_names"] if str(n) != "cloud_fraction_ifs"])
    old = os.path.join(str(tmp_path), "old.npz")
    np.savez(old, **payload)
    shutil.copy(os.path.join(str(tmp_path / "a"), "run_manifest.json"),
                str(tmp_path))
    dB = _driver(str(tmp_path / "b"))
    with pytest.raises(ValueError, match="has no cloud_fraction_ifs"):
        dB.load_checkpoint(old)
    # the unmodified checkpoint still restarts with the tracer
    dC = _driver(str(tmp_path / "c"))
    dC.load_checkpoint(ckpt)
    assert "cloud_fraction_ifs" in dC.state.tracers
