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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_ENTRY = _ROOT / "scripts" / "run" / "train_weatherbench_scale.py"
_spec = importlib.util.spec_from_file_location("train_wb_scale_damp", _ENTRY)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# The sfno mode builds a Gaussian grid, which hard-requires x64 (spectral
# transforms). neural_gcm has no such dependency and always runs.
_needs_x64 = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="sfno mode builds a Gaussian grid; needs JAX_ENABLE_X64=1",
)

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


def _record_threaded_fric(monkeypatch, mode, *, yml_overrides=None):
    """Return the ``fric_decay`` ``build_mode_components`` threads to the
    training segment for the given WB ``mode``.

    ``build_training_segment`` is monkeypatched to capture kwargs, so the
    segment is built (fric_decay threaded) but never run.
    """
    from legoesm.training import training_driver as td
    from legoesm.training.scale_build import build_mode_components

    recorded = {}

    def _recorder(model, step, grid, sigma, dt, **kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(raw=None)

    monkeypatch.setattr(td, "build_training_segment", _recorder)

    cfg = _mod.build_scale_config_from_args(["--mode", mode, "--smoke"])
    yml = dict(_SMOKE_YML, **(yml_overrides or {}))
    _, _, _, params, make_run_seg, _, _ = build_mode_components(cfg, yml)
    make_run_seg(params)

    fric = recorded.get("fric_decay")
    assert fric is not None, "fric_decay not threaded to the training segment"
    return jnp.asarray(fric)


def test_wb_modes_receive_driver_friction(monkeypatch):
    """build_mode_components hands the DRIVER's friction profile to the
    training segment, correctly GATED by whether a real BL scheme owns
    surface momentum (#931).

    - turbulence="louis": the rollout applies the physical Louis surface
      stress, so the Held-Suarez Rayleigh surrogate is gated to an exact no-op
      (decay == 1 at the surface) -- keeping it would double-count surface drag
      (#931).  Physics-mode adjoint stability comes from Louis, not this term.
    - turbulence="none": no BL scheme owns momentum, so the driver keeps the
      Rayleigh dissipation profile (decay < 1 near the surface) that the
      pure-dycore adjoint needs (#797 bug 11).
    """
    # Louis owns surface momentum -> Rayleigh drag gated off (no-op).
    fric_louis = _record_threaded_fric(monkeypatch, "physics",
                                       yml_overrides={"turbulence": "louis"})
    assert float(fric_louis[-1]) == 1.0
    assert bool(jnp.all(fric_louis > 0.0)) and bool(jnp.all(fric_louis <= 1.0))

    # No BL scheme -> the Rayleigh dissipation profile is retained and threaded.
    fric_none = _record_threaded_fric(monkeypatch, "physics",
                                      yml_overrides={"turbulence": "none"})
    assert float(fric_none[-1]) < 1.0
    assert bool(jnp.all(fric_none > 0.0)) and bool(jnp.all(fric_none <= 1.0))


@pytest.mark.parametrize(
    "mode", ["neural_gcm", pytest.param("sfno", marks=_needs_x64)]
)
def test_pure_dycore_wb_modes_keep_rayleigh_damping(monkeypatch, mode):
    """End-to-end #797 guard: the pure-dycore WB modes (neural_gcm / sfno) run
    NO boundary-layer scheme, so `build_latlon_config` declares
    turbulence="none" for them -> the #931 gate KEEPS the Held-Suarez Rayleigh
    drag, which is their SOLE adjoint dissipation of the undamped dycore.

    Critically the yml here does NOT set turbulence (it defaults to "louis"),
    yet the threaded fric_decay is still the non-trivial Rayleigh profile
    (surface < 1), NOT the louis no-op -- proving the mode-driven override,
    byte-identical to the pre-#931 always-on drag these modes relied on.
    """
    fric = _record_threaded_fric(monkeypatch, mode)  # yml turbulence -> "louis"
    assert float(fric[-1]) < 1.0
    assert bool(jnp.all(fric > 0.0)) and bool(jnp.all(fric <= 1.0))
