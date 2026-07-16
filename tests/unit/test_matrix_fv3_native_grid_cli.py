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
    """Patch the source function run_shallow_water imports at call time,
    capturing both the kwargs AND the returned grid so a test can prove ED
    provenance actually reached the model."""
    import legoesm.grids.cubed_sphere as cs
    calls = {}
    orig = cs.create_fv3_native_cubed_sphere

    def _spy(n, **kw):
        calls["n"] = n
        calls["kw"] = dict(kw)
        grid = orig(n, **kw)
        calls["grid"] = grid
        return grid

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
    # ED provenance actually reached the model, with duo halos on (codex
    # flag-review P2): the constructed grid is gnomonic_ed, not equiangular
    grid = calls["grid"]
    assert grid.gnomonic_form == "ed"
    assert grid.duogrid is not None
    assert status in ("PASS", "FAIL")
    assert "L2=" in notes


def test_fv3_native_angles_flag_round_trip():
    mod = _load_matrix_module()
    p = mod.build_parser()
    assert p.parse_args([]).fv3_native_angles is False
    assert p.parse_args(
        ["--fv3-native-grid", "--fv3-native-angles", "--sw-core", "fb"]
    ).fv3_native_angles is True


def test_fb_lane_native_grid_builds_ed():
    """The FB core honours --fv3-native-grid: _fb_cube_sw_model builds the
    ED gnomonic + duo grid (the ED grid's intended consumer)."""
    mod = _load_matrix_module()
    model = mod._fb_cube_sw_model(12, 2, fv3_native_grid=True)
    assert model.grid.gnomonic_form == "ed"
    assert model.grid.duogrid is not None
    # default equiangular baseline is unchanged
    base = mod._fb_cube_sw_model(12, 2)
    assert base.grid.gnomonic_form == "equiangular"


def test_fb_lane_native_angles_actually_change_cdgrid():
    """fv3_native_angles must actually install DIFFERENT seam angles, not be
    silently ignored: on the SAME ED grid, the native-angle cdgrid's cosa_u
    differs from the legacy-angle one at O(1) seam values (codex p4c
    FB-review P2 — non-vacuous)."""
    import numpy as np
    mod = _load_matrix_module()
    m_native = mod._fb_cube_sw_model(12, 2, fv3_native_grid=True,
                                     fv3_native_angles=True)
    m_legacy = mod._fb_cube_sw_model(12, 2, fv3_native_grid=True,
                                     fv3_native_angles=False)
    assert m_native.grid.gnomonic_form == "ed"
    ca_native = np.asarray(m_native.cdgrid.cosa_u)
    ca_legacy = np.asarray(m_legacy.cdgrid.cosa_u)
    assert ca_native.shape == ca_legacy.shape
    # they must genuinely differ (native cross-face seam averaging)
    assert not np.allclose(ca_native, ca_legacy)


def test_fb_native_angles_require_ed_grid_at_model_level():
    """FV3FBShallowWaterModel itself (not just the builder) rejects
    fv3_native_angles on a non-ED grid — no silent fallback to legacy
    angles (codex p4c FB-review P2)."""
    import pytest
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        FV3FBShallowWaterModel,
        fb_m1_preset_config,
    )
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    eq_grid = create_cubed_sphere(12, use_duogrid=True)   # equiangular
    assert eq_grid.gnomonic_form == "equiangular"
    with pytest.raises(ValueError, match="requires an ED gnomonic grid"):
        FV3FBShallowWaterModel(eq_grid, fb_m1_preset_config(),
                               fv3_native_angles=True)
    # builder guard (flag-level) still rejects angles without the ED grid
    mod = _load_matrix_module()
    with pytest.raises(ValueError, match="requires fv3_native_grid"):
        mod._fb_cube_sw_model(12, 2, fv3_native_angles=True)


def test_fv3_native_flag_error_helper():
    """The pure validation helper main() uses rejects exactly the invalid
    combinations and accepts the valid ones (codex p4c FB-review P2 — makes
    the main() cross-flag logic unit-testable without running the matrix)."""
    mod = _load_matrix_module()
    err = mod._fv3_native_flag_error
    # valid: no angles at all
    assert err(False, False, "production") is None
    assert err(True, False, "production") is None
    # valid: angles + grid + fb
    assert err(True, True, "fb") is None
    # invalid: angles without the ED grid
    assert "requires --fv3-native-grid" in err(False, True, "fb")
    # invalid: angles with the production core
    assert "requires --sw-core fb" in err(True, True, "production")


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
