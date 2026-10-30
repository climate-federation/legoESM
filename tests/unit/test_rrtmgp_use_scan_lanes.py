"""The MPAS column and spectral lanes build RRTMGP with ``rrtmgp_use_scan``.

Both lanes used to hand-build their ``RRTMGPConfig`` without that field, so
``--rrtmgp-use-scan`` reached only the compiled pipeline and was silently
inert there.  Each lane's first ``make_physics`` call is intercepted and
raises, so nothing compiles.  ``None`` (flag unset) is the default: it defers
to ``rte_utils.recurrent_op_with_halos``, which picks the unrolled loop on CPU.
"""
from __future__ import annotations

import pytest

import legoesm.atmosphere.physics.combined as combined
from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver


class _Seen(Exception):
    pass


_LANES = {
    "mpas": (
        dict(grid=GridConfig(grid_type="mpas", resolution=3, nlev=20,
                             vertical_coord="hybrid"),
             dycore=DycoreConfig(discretization="mpas", dt=300.0)),
        "_run_column"),
    "spectral": (
        dict(grid=GridConfig(grid_type="gaussian", resolution=8, nlev=8),
             dycore=DycoreConfig(energy_fixer=False, model_type="hydrostatic",
                                 discretization="spectral", dt=600.0)),
        "_run_spectral"),
}


@pytest.mark.parametrize("lane", sorted(_LANES))
@pytest.mark.parametrize("flag", [True, False, None])
def test_lane_rrtmgp_config_honours_use_scan(lane, flag, monkeypatch,
                                             tmp_path):
    grid_dycore, run = _LANES[lane]
    cfg = ExperimentConfig(
        **grid_dycore,
        output=OutputConfig(output_dir="", diag_days=0),
        days=1, dataset="analytical", radiation="rrtmgp",
        convection="none", turbulence="none", precision="fp64",
        rrtmgp_use_scan=flag,
    )
    seen = []

    def _spy(phys_cfg, *args, **kwargs):
        seen.append(phys_cfg.radiation.rrtmgp.use_scan)
        raise _Seen

    monkeypatch.setattr(combined, "make_physics", _spy)
    drv = ModelDriver(cfg, output_dir=str(tmp_path))
    drv.setup()
    with pytest.raises(_Seen):
        getattr(drv, run)()
    assert seen == [flag]
