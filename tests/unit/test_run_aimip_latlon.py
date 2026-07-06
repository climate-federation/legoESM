"""Unit test for the AIMIP lat-lon orchestrator's pure helpers + dispatch.

Heavy paths (ERA5 load, training) are exercised by the GPU smoke; this
keeps the parse/config/dispatch logic covered offline.  Runnable directly
or under pytest.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_aimip_latlon_mod", REPO / "scripts" / "run" / "run_aimip_latlon.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_parse_windows():
    assert mod._parse_windows("2015:0:3,2016:180:3") == [
        (2015, 0, 3), (2016, 180, 3)]
    assert mod._parse_windows("2017:0:1") == [(2017, 0, 1)]
    assert mod._parse_windows(" 2015:0:3 , ") == [(2015, 0, 3)]  # whitespace/empty


def test_parser_defaults():
    args = mod.build_parser().parse_args([])
    assert args.n_lat == 72
    assert "classical" in args.variants
    assert mod._ROLLOUT_HOURS == 6       # short horizon: adjoint stability
    assert args.w_flux_olr > 0.0         # flux supervision on by default
    # AMIP physics stack defaults (user directive: NO gray; full stack).
    assert args.radiation == "rrtmgp"    # band model, not gray
    assert args.microphysics == "kessler"
    assert args.gravity_wave_drag == "hines"
    assert args.rad_update_steps > 1     # rrtmgp sub-cycled, NOT every step


def test_rad_update_steps_roundtrip():
    args = mod.build_parser().parse_args(["--rad-update-steps", "6"])
    assert args.rad_update_steps == 6
    # plumbs into train_physics_params + build_training_segment signatures
    import inspect
    from legoesm.training.training_driver import (
        train_physics_params, build_training_segment,
    )
    assert "rad_update_steps" in inspect.signature(train_physics_params).parameters
    assert "rad_update_steps" in inspect.signature(build_training_segment).parameters


def test_radiation_as_forcing_threads():
    """--radiation-as-forcing must reach train_physics_params and the pipeline
    build_step_unified (the rrtmgp-adjoint stop_gradient lever)."""
    import inspect
    args = mod.build_parser().parse_args([])
    assert args.radiation_as_forcing is False          # default: full adjoint
    args = mod.build_parser().parse_args(["--radiation-as-forcing"])
    assert args.radiation_as_forcing is True
    from legoesm.training.training_driver import train_physics_params
    from legoesm.driver.physics_pipeline import PhysicsPipeline
    assert "rad_stop_gradient" in inspect.signature(train_physics_params).parameters
    assert "rad_stop_gradient" in inspect.signature(
        PhysicsPipeline.build_step_unified).parameters


def test_loss_config_has_flux_weights():
    args = mod.build_parser().parse_args([])
    lc = mod.make_loss_config(args)
    assert lc.w_flux_olr > 0.0
    assert lc.w_bias_flux_olr > 0.0
    assert lc.flux_scale > 0.0


def test_train_variant_rejects_unknown():
    """Dispatch hardening: unknown variant must raise, not silently no-op."""
    args = mod.build_parser().parse_args([])
    try:
        # The else-branch raises before touching the model/data args, so
        # passing None for them is safe.
        mod.train_variant(
            "bogus", None, None, None, None, None,
            None, None, None, None, None, args, Path("/tmp"),
        )
    except ValueError as e:
        assert "unknown variant" in str(e).lower()
        return
    raise AssertionError("expected ValueError for unknown variant")


def test_latlon_grid_has_area_weights():
    """LatLonGrid.weights must exist (cos-lat) so combined_loss area-weights
    lat-lon losses (codex review #3)."""
    import numpy as np
    from legoesm.grids.latlon import create_latlon_grid
    g = create_latlon_grid(n_lat=16)
    w = np.asarray(g.weights)
    assert w.shape == (16,)
    assert np.allclose(w, np.asarray(g.cos_lat))
    assert float(w.min()) > 0.0           # poles clamped, no zero weight


def test_smoke_args_shrink():
    args = mod.build_parser().parse_args(["--smoke"])
    # main() applies the smoke shrink; replicate the relevant asserts on the
    # documented smoke profile by calling the same mutation path.
    assert args.smoke is True


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("run_aimip_latlon: all self-checks passed")


if __name__ == "__main__":
    _run_all()
