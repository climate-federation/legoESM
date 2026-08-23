"""Unit tests for the Phase C long-term-simulation diagnostics:
RPE drift (``legoesm.ocean.rpe``), energy + tracer budgets
(``legoesm.ocean.budgets``), and the restart harness
(``legoesm.ocean.restart``).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import importlib.util

import numpy as np
import pytest


def _matrix_module():
    """Load ``scripts/matrix/run_ocean_test_matrix.py`` once for fixture reuse."""
    if not hasattr(_matrix_module, "_mod"):
        repo_root = Path(__file__).resolve().parents[3]
        scripts_dir = repo_root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir / "matrix"))
        matrix_path = scripts_dir / "matrix" / "run_ocean_test_matrix.py"
        if not matrix_path.exists():
            matrix_path = scripts_dir / "run_ocean_test_matrix.py"
        spec = importlib.util.spec_from_file_location(
            "_rom_for_diag_tests", matrix_path,
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_rom_for_diag_tests"] = mod
        spec.loader.exec_module(mod)
        _matrix_module._mod = mod
    return _matrix_module._mod


def _rest_state(grid_type, res, H_max=5500.0, nlev=10):
    matrix_mod = _matrix_module()
    from ocean_test_matrix.setup import _create_ocean_setup
    from ocean_test_matrix.testcase import TestCase
    tc = TestCase("diag_test", grid_type, res, 1.0, 0.1)
    grid, z, _, model, _, _, _ = _create_ocean_setup(
        tc, H_max=H_max, nlev=nlev,
    )
    state = matrix_mod._create_rest_state(tc, grid, z, H_max=H_max)
    return state, grid, z, model


# ---------------------------------------------------------------------------
# RPE diagnostic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grid_type,res", [
    ("latlon", "36x72"),
    ("mpas", "ico3"),
    ("cubed_sphere", "C24"),
])
def test_rpe_finite_and_negative_at_rest(grid_type, res):
    from legoesm.ocean.rpe import compute_rpe
    state, grid, z, _ = _rest_state(grid_type, res)
    rpe = compute_rpe(state, z, grid_type=grid_type, grid=grid, eos="wright")
    assert np.isfinite(rpe)
    # PE = g * sum(rho * z * vol) with z < 0 below the surface ->
    # RPE must be negative for a stratified rest state.
    assert rpe < 0.0


def test_rpe_three_grids_agree_at_rest():
    """All three grids on the same rest state must produce RPE values
    within 5 % of each other (cell-area discretisation tolerance)."""
    from legoesm.ocean.rpe import compute_rpe
    values = []
    for grid_type, res in [
        ("latlon", "36x72"),
        ("mpas", "ico3"),
        ("cubed_sphere", "C24"),
    ]:
        state, grid, z, _ = _rest_state(grid_type, res)
        values.append(compute_rpe(state, z, grid_type=grid_type, grid=grid))
    ref = values[0]
    spread = max(abs(v - ref) for v in values) / abs(ref)
    assert spread < 0.05, f"RPE spread across grids = {spread:.3%}"


def test_rpe_drift_rate_units():
    from legoesm.ocean.rpe import rpe_drift_rate_per_m2
    # 1 J across 1 m^2 over 1 s -> 1 W/m^2
    assert rpe_drift_rate_per_m2(0.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)
    # Zero delta -> zero
    assert rpe_drift_rate_per_m2(5.0, 5.0, 1.0, 1.0) == pytest.approx(0.0)
    # Invalid inputs -> NaN
    assert np.isnan(rpe_drift_rate_per_m2(0.0, 1.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Energy + tracer budgets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grid_type,res", [
    ("latlon", "36x72"),
    ("mpas", "ico3"),
    ("cubed_sphere", "C24"),
])
def test_energy_budget_zero_at_rest(grid_type, res):
    from legoesm.ocean.budgets import compute_energy_budget
    state, grid, z, _ = _rest_state(grid_type, res)
    eb = compute_energy_budget(state, z, grid_type=grid_type, grid=grid)
    # Rest state -> KE = 0, eta = 0 -> APE = 0 (linear free-surface).
    assert eb.KE == pytest.approx(0.0, abs=1e-12)
    assert eb.APE == pytest.approx(0.0, abs=1e-12)
    assert eb.total == pytest.approx(0.0, abs=1e-12)
    assert eb.volume > 0.0


@pytest.mark.parametrize("grid_type,res", [
    ("latlon", "36x72"),
    ("mpas", "ico3"),
    ("cubed_sphere", "C24"),
])
def test_tracer_budget_finite_at_rest(grid_type, res):
    from legoesm.ocean.budgets import compute_tracer_budget
    state, grid, z, _ = _rest_state(grid_type, res)
    tb = compute_tracer_budget(state, z, grid_type=grid_type, grid=grid)
    assert tb.volume > 0.0
    assert tb.heat_content > 0.0
    assert tb.salt_mass > 0.0
    # eta = 0 at rest -> the SSH integral is exactly zero.
    assert tb.eta_integral == pytest.approx(0.0, abs=1e-9)


def test_tracer_budget_volume_matches_three_grids():
    from legoesm.ocean.budgets import compute_tracer_budget
    vols = []
    for grid_type, res in [
        ("latlon", "36x72"),
        ("mpas", "ico3"),
        ("cubed_sphere", "C24"),
    ]:
        state, grid, z, _ = _rest_state(grid_type, res)
        vols.append(compute_tracer_budget(
            state, z, grid_type=grid_type, grid=grid,
        ).volume)
    spread = max(abs(v - vols[0]) for v in vols) / vols[0]
    assert spread < 0.02, f"Volume spread across grids = {spread:.3%}"


# ---------------------------------------------------------------------------
# Restart harness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grid_type,res", [
    ("latlon", "36x72"),
    ("mpas", "ico3"),
    ("cubed_sphere", "C24"),
])
def test_restart_round_trip_bit_identical(grid_type, res, tmp_path):
    """The ARCHIVE round-trip is exact: every ``Field``-valued slot comes back
    ``np.array_equal``.

    SCOPE OF THE NAME, so it is not requoted as more than it proves.  This
    asserts save -> load array equality, and only for ``Field`` slots (the
    legacy ``save_restart`` drops non-``Field`` carries silently).  It does NOT
    assert that a model resumed from a restart STEPS to identical results
    against an uninterrupted run — nothing here integrates past the restart.
    It also holds only because ``JAX_ENABLE_X64`` is the same on both sides:
    ``load_restart`` rebuilds via a bare ``jnp.asarray``, which DEMOTES f64 to
    f32 with x64 unset and does not check for it (the RUN pair,
    ``load_run_restart``, does).  See ``ocean/restart.py``'s SCOPE OF THE
    GUARANTEE.
    """
    from legoesm.ocean.restart import save_restart, load_restart, restart_metadata
    state, grid, z, model = _rest_state(grid_type, res)
    # Step once so the state has non-trivial floats.
    state2 = model.step(state, 300.0)
    out_path = tmp_path / "restart.npz"
    save_restart(state2, out_path, time_s=12345.0, step=42, sha="deadbeef")
    state3 = load_restart(out_path, state2)
    for name in state2._fields:
        a = getattr(state2, name)
        b = getattr(state3, name)
        if a is None:
            assert b is None
            continue
        np.testing.assert_array_equal(
            np.asarray(a.data), np.asarray(b.data),
            err_msg=f"field {name!r} differs after restart round-trip",
        )
    meta = restart_metadata(out_path)
    assert meta["time_s"] == pytest.approx(12345.0)
    assert meta["step"] == 42
    assert meta["sha"] == "deadbeef"
