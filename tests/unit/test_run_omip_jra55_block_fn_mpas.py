"""The JRA55 block-scan builders must accept an MPAS model (flat MPASOceanConfig).

They read ``model.config.barotropic.maxvel_barotropic`` (the lat-lon nesting);
``MPASOceanConfig`` carries ``maxvel_barotropic`` flat, so every MPAS + JRA55
run_omip launch died at block-function build time (first MPAS SPMD GPU smoke,
job 27326196: ``AttributeError: 'MPASOceanConfig' object has no attribute
'barotropic'``).  Building the block functions for a minimal MPAS-shaped model
is the gate; it fails on the old attribute access.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _jra55_state():
    from legoesm.coupler.config import CouplerConfig
    return {
        "coupler_cfg": CouplerConfig(bulk_scheme="large_yeager", z_ref=10.0,
                                     z_t_atm=10.0, z_q_atm=10.0),
        "co2_ppmv": 400.0, "T_ramp_seconds": 86400.0,
        "enable_sponge": False, "enable_sss_restoring": False,
        "enable_freeze_cap": False, "enable_sea_ice": False,
    }


def test_block_fn_builds_for_mpas_config():
    """The host-regrid block builder (the lane the MPAS SPMD run uses); the
    GPU-interp builder shares the same fixed attribute access but needs the
    full regrid state to build and is refused under --enable-mpas-spmd."""
    from legoesm.ocean.mpas_config import MPASOceanConfig
    from scripts.run import run_omip
    model = SimpleNamespace(config=MPASOceanConfig(), _step_impl=lambda *a, **k: None)
    fn = run_omip._build_jra55_block_fn(model, _jra55_state(), 300.0)
    assert callable(fn)
