"""CLI round-trip for the Phase-1 M1 ``--sw-core`` matrix flag.

The FB lane (``--sw-core fb``) must parse, default to production, reject
unknown cores loudly, and build a duogrid ``FV3FBShallowWaterModel`` with
the M1 validated preset (nord=1 d4_bg=0.16 dddmp=0.2 damp_v=0.02 nord_v=2).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_matrix_module():
    """Spec-load the matrix-runner script without sys.path pollution."""
    script = (Path(__file__).resolve().parents[2]
              / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")
    name = "_sw_core_cli_unit"
    spec = importlib.util.spec_from_file_location(name, script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def test_sw_core_flag_round_trip():
    M = _load_matrix_module()
    p = M.build_parser()
    assert p.parse_args([]).sw_core == "production"
    assert p.parse_args(["--sw-core", "fb"]).sw_core == "fb"
    assert p.parse_args(["--sw-core", "production"]).sw_core == "production"


def test_sw_core_flag_rejects_unknown():
    M = _load_matrix_module()
    p = M.build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--sw-core", "bogus"])


def test_fb_m1_preset_config_values():
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import fb_m1_preset_config
    cfg = fb_m1_preset_config()
    assert cfg.nord == 1
    assert cfg.d4_bg == 0.16
    assert cfg.dddmp == 0.2
    assert cfg.d2_bg == 0.0
    assert cfg.damp_v == 0.02
    # Explicit 2, NOT the -1 sentinel (which would derive min(2, nord)=1).
    assert cfg.nord_v == 2
    # FB never consumes these production-path fields.
    assert cfg.div_damp == 0.0
    assert cfg.hyperdiff_coeff == 0.0


def test_fb_cube_sw_model_builder():
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import FV3FBShallowWaterModel
    M = _load_matrix_module()
    model = M._fb_cube_sw_model(12, 2)
    assert isinstance(model, FV3FBShallowWaterModel)
    # FB is duogrid-only: the builder must hand it a duogrid grid.
    assert model.cdgrid.base.duogrid is not None
    assert model.config.nord_v == 2
    # Modons (test 8) run non-rotating (f derives from the grid omega).
    import jax.numpy as jnp
    model8 = M._fb_cube_sw_model(12, 8)
    assert float(jnp.max(jnp.abs(model8.grid.f))) == 0.0
    assert float(jnp.max(jnp.abs(model.grid.f))) > 0.0


def test_run_shallow_water_fb_branch(tmp_path):
    """Exercise the ACTUAL run_shallow_water FB branch (codex M1 LOW):
    a tiny C12 W2 run with _SW_CORE='fb' must route through
    _fb_cube_sw_model and complete without stale production-lane state."""
    M = _load_matrix_module()
    calls = []
    orig = M._fb_cube_sw_model

    def _spy(n, test_num):
        calls.append((n, test_num))
        return orig(n, test_num)

    M._fb_cube_sw_model = _spy
    M._SW_CORE = "fb"
    try:
        tc = M.TestCase("shallow_water", "williamson2", "cubed_sphere",
                        "C12", "none", 5, 1, {"test_num": 2})
        status, wall, notes = M.run_shallow_water(
            tc, tmp_path, 0.02)  # 5 steps at dt=300
    finally:
        M._SW_CORE = "production"
        M._fb_cube_sw_model = orig
    assert calls == [(12, 2)]
    assert status in ("PASS", "FAIL")  # ran to completion, no crash
    assert "L2=" in notes


def test_run_shallow_water_unknown_core_raises(tmp_path):
    M = _load_matrix_module()
    M._SW_CORE = "bogus"
    try:
        tc = M.TestCase("shallow_water", "williamson2", "cubed_sphere",
                        "C12", "none", 5, 1, {"test_num": 2})
        with pytest.raises(ValueError, match="unknown --sw-core"):
            M.run_shallow_water(tc, tmp_path, 0.02)
    finally:
        M._SW_CORE = "production"
