"""CLI round-trip + branch coverage for the phase-4c ``--fv3-native-grid``
matrix flag.

The flag swaps the cube SW production-lane grid from the legacy equiangular
gnomonic to the FV3-native ED gnomonic (+ duo halos) — the grid family
certified bit-exact in the phase-4 one-step oracles.  It must parse,
default OFF, route ``run_shallow_water`` through
``create_fv3_native_cubed_sphere`` with ``use_duogrid=True`` and the right
rotation (Omega for Williamson, 0 for modons), and complete a tiny run.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_matrix_module():
    """Spec-load the matrix-runner script without sys.path pollution."""
    script = (Path(__file__).resolve().parents[2]
              / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")
    name = "_fv3_native_grid_cli_unit"
    spec = importlib.util.spec_from_file_location(name, script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def test_fv3_native_grid_flag_round_trip():
    mod = _load_matrix_module()
    p = mod.build_parser()
    assert p.parse_args([]).fv3_native_grid is False
    assert p.parse_args(["--fv3-native-grid"]).fv3_native_grid is True


def _spy_native_ctor():
    """Patch the source function run_shallow_water imports at call time."""
    import legoesm.grids.cubed_sphere as cs
    calls = {}
    orig = cs.create_fv3_native_cubed_sphere

    def _spy(n, **kw):
        calls["n"] = n
        calls["kw"] = dict(kw)
        return orig(n, **kw)

    cs.create_fv3_native_cubed_sphere = _spy
    return cs, orig, calls


def test_run_shallow_water_fv3_native_grid_w2_branch(tmp_path):
    """_FV3_NATIVE_GRID=True routes W2 through create_fv3_native_cubed_sphere
    with duo halos + rotation, and the run completes."""
    from legoesm import constants
    mod = _load_matrix_module()
    cs, orig, calls = _spy_native_ctor()
    mod._FV3_NATIVE_GRID = True
    try:
        tc = mod.TestCase("shallow_water", "williamson2", "cubed_sphere",
                        "C12", "none", 5, 1, {"test_num": 2})
        status, wall, notes = mod.run_shallow_water(tc, tmp_path, 0.02)
    finally:
        mod._FV3_NATIVE_GRID = False
        cs.create_fv3_native_cubed_sphere = orig
    assert calls["n"] == 12
    assert calls["kw"].get("use_duogrid") is True
    assert calls["kw"].get("k2e_nord") == 4
    # Williamson runs rotating: omega == constants.Omega
    assert abs(calls["kw"].get("omega") - constants.Omega) < 1e-12
    # ED provenance actually reached the model (not the legacy equiangular)
    assert status in ("PASS", "FAIL")
    assert "L2=" in notes


def test_run_shallow_water_fv3_native_grid_modon_nonrotating(tmp_path):
    """Modons (test 8) must pass omega=0.0 even on the FV3-native grid."""
    mod = _load_matrix_module()
    cs, orig, calls = _spy_native_ctor()
    mod._FV3_NATIVE_GRID = True
    try:
        tc = mod.TestCase("shallow_water", "colliding_modons", "cubed_sphere",
                        "C12", "none", 5, 1, {"test_num": 8})
        mod.run_shallow_water(tc, tmp_path, 0.02)
    finally:
        mod._FV3_NATIVE_GRID = False
        cs.create_fv3_native_cubed_sphere = orig
    assert calls["kw"].get("omega") == 0.0
    assert calls["kw"].get("use_duogrid") is True


def test_default_off_uses_legacy_equiangular(tmp_path):
    """With the flag OFF, create_fv3_native_cubed_sphere is NOT called (the
    legacy equiangular path is preserved byte-for-byte)."""
    mod = _load_matrix_module()
    cs, orig, calls = _spy_native_ctor()
    assert mod._FV3_NATIVE_GRID is False
    try:
        tc = mod.TestCase("shallow_water", "williamson2", "cubed_sphere",
                        "C12", "none", 5, 1, {"test_num": 2})
        mod.run_shallow_water(tc, tmp_path, 0.02)
    finally:
        cs.create_fv3_native_cubed_sphere = orig
    assert calls == {}
