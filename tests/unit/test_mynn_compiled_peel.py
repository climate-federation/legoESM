"""turbulence=mynn25 on the compiled pipeline: the cold-start step is peeled
once, the host flag rides carry_aux beside the packed carry, and mismatched
restores are refused (votes M3, N2, R1a, R2a)."""
from types import SimpleNamespace

import numpy as np
import pytest

from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.model_driver import ModelDriver


def _cfg():
    return ExperimentConfig(
        # pure sigma: the lane diagnoses w by sigma continuity (votes M1/R1)
        grid=GridConfig(resolution=8, nlev=8, vertical_coord="sigma"),
        dycore=DycoreConfig(energy_fixer=False, dt=600.0),
        output=OutputConfig(diag_days=1),
        days=3 * 600.0 / 86400.0, dataset="analytical",
        radiation="gray", turbulence="mynn25")


def _run(tmp_path, monkeypatch=None, peel=True):
    drv = ModelDriver(_cfg(), output_dir=tmp_path)
    drv.setup()
    if not peel:
        monkeypatch.setattr(type(drv), "_mynn_peel_pending",
                            lambda self, **k: False)
    assert drv.run() == "COMPLETED"
    return drv


def test_compiled_lane_peels_the_cold_start_once(tmp_path, monkeypatch):
    a = _run(tmp_path / "a")
    qke = np.asarray(a._carry_aux["qke"])
    assert qke.shape[-2] == 3 and np.all(np.isfinite(qke))
    assert bool(np.asarray(a._carry_aux["mynn_initialized"]))
    # without the peel the zero seed is never initialised: different carry
    b = _run(tmp_path / "b", monkeypatch, peel=False)
    assert not np.array_equal(qke, np.asarray(b._carry_aux["qke"]))


@pytest.mark.parametrize("aux", [{"qke": 1.0}, {"mynn_initialized": True}])
def test_restored_flag_and_carry_come_together(aux):
    fake = SimpleNamespace(_carry_aux=aux)
    with pytest.raises(ValueError, match="both or neither"):
        ModelDriver._mynn_peel_pending(fake)


def test_restored_pair_skips_the_peel_and_fresh_run_owes_it():
    done = SimpleNamespace(_carry_aux={"qke": 1.0,
                                       "mynn_initialized": np.asarray(True)})
    assert ModelDriver._mynn_peel_pending(done) is False
    assert ModelDriver._mynn_peel_pending(SimpleNamespace(_carry_aux={}))


def test_pipeline_step_build_needs_the_flag():
    from legoesm.driver.physics_pipeline import _is_mynn_kernel
    from legoesm.atmosphere.physics.turbulence.mynn_wrf_adapter import (
        mynn_wrf_turbulence)
    assert _is_mynn_kernel(mynn_wrf_turbulence)
    pipe = SimpleNamespace(turbulence_fn=mynn_wrf_turbulence)
    from legoesm.driver.physics_pipeline import PhysicsPipeline
    with pytest.raises(ValueError, match="mynn_initflag"):
        PhysicsPipeline.build_step_unified(pipe, static_need_rad=True)
