"""An MPAS checkpoint without IFS CLOUDSC's advected cloud fraction
(``cloud_fraction_ifs``) -- not listed, or listed without its array -- is
refused on restart of an ifs_cloudsc run (votes VC26, VC37, the N_s rule);
the intact checkpoint restores it with the registry's unit "1" (VC38).  The
run itself also goes with the IFS diurnal-cycle option OFF, which only
works because the IFS ascent alone switches the surface-flux feed on (VC22,
VC35)."""
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


def _driver(tmpdir: str, capdcycl: bool = True) -> ModelDriver:
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=8,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="mpas", dt=DT),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=1),
        days=TWO_STEPS_DAYS, dataset="analytical", distributed=False,
        n_devices=1, turbulence="louis", radiation="none",
        convection="bechtold", bechtold_use_ifs_ascent=True,
        bechtold_use_ifs_capdcycl=capdcycl,
        microphysics="ifs_cloudsc", gravity_wave_drag="none",
        cloud_scheme="none", cld_macmic_num_steps=1, precision="fp64")
    d = ModelDriver(cfg, output_dir=tmpdir)
    d.setup()
    return d


@pytest.mark.slow
def test_ifs_run_without_the_diurnal_option_still_gets_surface_fluxes(tmp_path):
    d = _driver(str(tmp_path), capdcycl=False)
    # the IFS ascent refuses to run without the surface fluxes; with the
    # diurnal-cycle option off only the ascent itself turns the feed on
    assert d.run(segment_callback=lambda drv, day, dt: None) == "COMPLETED"


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
    shutil.copy(os.path.join(str(tmp_path / "a"), "run_manifest.json"),
                str(tmp_path))
    # (a) not listed and no array; (b) listed but its array missing
    unlisted = dict(payload)
    del unlisted["trc_cloud_fraction_ifs"]
    unlisted["tracer_names"] = np.asarray(
        [n for n in payload["tracer_names"] if str(n) != "cloud_fraction_ifs"])
    no_array = dict(payload)
    del no_array["trc_cloud_fraction_ifs"]
    for name, bad in (("unlisted", unlisted), ("no_array", no_array)):
        old = os.path.join(str(tmp_path), f"{name}.npz")
        np.savez(old, **bad)
        dB = _driver(str(tmp_path / f"b_{name}"))
        with pytest.raises(ValueError, match="has no cloud_fraction_ifs"):
            dB.load_checkpoint(old)
    # the unmodified checkpoint still restarts with the tracer, unit "1"
    dC = _driver(str(tmp_path / "c"))
    dC.load_checkpoint(ckpt)
    assert dC.state.tracers["cloud_fraction_ifs"].units == "1"
