"""build_mode_components must train at the DRIVER's CFL-safe dt (#797).

The old ``_cfl_dt`` re-derived the CFL from the MERIDIONAL spacing only
(pi*R/n_lat) and missed the pole-cell ZONAL limit, handing the training
segment dt=81.8 s (CFL 1.47 vs c~390 m/s) at the C32 smoke size while the
driver itself integrates at 30 s. Physics mode survives on physics damping;
the pure-dycore modes (zero-init neural_gcm / sfno) blow up -> loss=nan.
The contract: the trainer uses exactly the driver's post-setup dycore dt.

Heavier than the import tests (builds a real ModelDriver twice); smoke-sized
lat-lon C32/L8 gray keeps it seconds. Needs JAX_ENABLE_X64 not required.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_ENTRY = _ROOT / "scripts" / "run" / "train_weatherbench_scale.py"
_spec = importlib.util.spec_from_file_location("train_wb_scale_dt", _ENTRY)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_SMOKE_YML = {
    "n_lat": 32, "n_lon": 64, "nlev": 8, "dt": 300.0,
    "radiation": "gray", "era5_zarr": "unused", "train_years": [2015],
    "loss": {"multi_step_hours": [6, 12]},
}


def test_training_dt_equals_driver_cfl_safe_dt():
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.training.scale_build import (
        build_latlon_config,
        build_mode_components,
    )

    cfg = _mod.build_scale_config_from_args(
        ["--mode", "physics", "--smoke"])
    yml = dict(_SMOKE_YML)

    *_rest, dt = build_mode_components(cfg, yml)

    ref = ModelDriver(build_latlon_config(cfg, yml))
    ref.setup()
    safe_dt = float(ref.config.dycore.dt)

    assert dt == safe_dt
    # non-vacuous: the YAML dt (300) and the factory pole-clamp (~82 s at
    # C32) must both have been reduced for this grid
    assert dt < 80.0
