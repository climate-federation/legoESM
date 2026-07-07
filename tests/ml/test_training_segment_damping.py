"""Training segments must carry the driver's Rayleigh friction (#797 bug 11).

``build_training_segment`` hardcoded ``fric_decay=ones`` (no friction) and
the WB trainer never threaded the driver's profile through. The forward
rollout survives 6 h undamped, but the ADJOINT through 720 steps of the
undamped dycore returns NaN gradients (probe job 26082581: loss=0.714
finite, grads_finite=False) — the per-sample optimizer update then poisons
the params and the epoch mean prints loss=nan for the pure-dycore modes
(neural_gcm / sfno). Physics mode only survived because its parameterized
turbulence damps the adjoint too.

``qv_smooth_coeff`` deliberately stays 0 on lat-lon: the smoothing operator
is cube-only (#798 fix 5).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_ENTRY = _ROOT / "scripts" / "run" / "train_weatherbench_scale.py"
_spec = importlib.util.spec_from_file_location("train_wb_scale_damp", _ENTRY)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_SMOKE_YML = {
    "n_lat": 32, "n_lon": 64, "nlev": 8, "dt": 300.0,
    "radiation": "gray", "era5_zarr": "unused", "train_years": [2015],
    "loss": {"multi_step_hours": [6, 12]},
}


def test_build_training_segment_threads_fric_decay(monkeypatch):
    """An explicit fric_decay reaches build_segment_fn verbatim; omitting it
    keeps the legacy no-friction ones."""
    from legoesm.training import training_driver as td

    recorded = {}

    def _recorder(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(raw=None)

    monkeypatch.setattr(td, "build_segment_fn", _recorder)
    sigma = SimpleNamespace(sigma_full=np.linspace(0.05, 0.95, 8),
                            dsigma=np.full(8, 1.0 / 8))
    grid = SimpleNamespace(lat=jnp.zeros(4), lon=jnp.zeros(8))

    custom = jnp.linspace(0.9, 1.0, 8)
    td.build_training_segment(None, None, grid, sigma, 30.0,
                               fric_decay=custom)
    assert bool(jnp.all(recorded["fric_decay"] == custom))

    td.build_training_segment(None, None, grid, sigma, 30.0)
    assert bool(jnp.all(recorded["fric_decay"] == 1.0))


def test_wb_modes_receive_driver_friction(monkeypatch):
    """build_mode_components hands the DRIVER's boundary-layer friction
    profile (non-trivial: < 1 near the surface) to the training segment."""
    from legoesm.training import training_driver as td
    from legoesm.training.scale_build import build_mode_components

    recorded = {}

    def _recorder(model, step, grid, sigma, dt, **kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(raw=None)

    monkeypatch.setattr(td, "build_training_segment", _recorder)

    cfg = _mod.build_scale_config_from_args(["--mode", "physics", "--smoke"])
    yml = dict(_SMOKE_YML)
    _, _, _, params, make_run_seg, _, _ = build_mode_components(cfg, yml)
    make_run_seg(params)

    fric = recorded.get("fric_decay")
    assert fric is not None, "fric_decay not threaded to the training segment"
    fric = jnp.asarray(fric)
    # BL Rayleigh friction: decay < 1 at the lowest level, ~free atmosphere aloft
    assert float(fric[-1]) < 1.0
    assert bool(jnp.all(fric > 0.0)) and bool(jnp.all(fric <= 1.0))
