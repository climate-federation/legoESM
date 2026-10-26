"""Tests for the LES-suite matrix wiring (spec + selection + dispatch hardening)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from legoesm.atmosphere.les_suite import registry as reg
from legoesm.atmosphere.les_suite.matrix import (
    MatrixCase,
    _les_suite_matrix_spec,
    build_test_matrix,
    select_cases,
    valid_grid_labels,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _fresh_registry():
    reg.clear_registry()
    yield
    reg.clear_registry()


# --- spec ---------------------------------------------------------------------
def test_matrix_spec_routes_through_shared_selector():
    from legoesm.core.setup_selector import MatrixRunnerSpec

    spec = _les_suite_matrix_spec()
    assert isinstance(spec, MatrixRunnerSpec)
    assert spec.runner_path == "scripts/matrix/run_les_suite_matrix.py"
    assert spec.case_flag == "--only"
    assert spec.exact_prefix == "="
    # LES matrix has no levels/dt/days/resolution flags
    assert spec.levels_flag is None
    assert spec.dt_flag is None


def test_valid_grid_labels_present():
    labels = valid_grid_labels()
    assert "96x96x96" in labels  # the dry production grid + cbl anchor
    assert all(isinstance(lbl, str) for lbl in labels)


def test_spec_valid_grids_matches_labels():
    spec = _les_suite_matrix_spec()
    assert set(spec.valid_grids) == set(valid_grid_labels())


# --- build_test_matrix (every (name, grid) is a real case) --------------------
def test_build_test_matrix_all_pairs_real():
    matrix = build_test_matrix()
    assert matrix, "matrix must be non-empty"
    assert all(isinstance(m, MatrixCase) for m in matrix)
    reg_cases = {(c.name, c.grid.label) for c in reg.list_cases()}
    matrix_pairs = {(m.case, m.grid_type) for m in matrix}
    assert matrix_pairs == reg_cases


def test_build_command_selects_a_real_case():
    # every enumerated (name, grid) must be selectable by the exact filter
    for m in build_test_matrix():
        sel = select_cases(only=f"={m.case}", grid=m.grid_type)
        assert len(sel) == 1
        assert sel[0].name == m.case


# --- selection + dispatch hardening -------------------------------------------
def test_select_all_when_no_filter():
    assert len(select_cases()) == len(reg.list_cases())


def test_select_exact_name():
    sel = select_cases(only="=cbl_nieuwstadt")
    assert len(sel) == 1 and sel[0].name == "cbl_nieuwstadt"


def test_select_substring():
    sel = select_cases(only="dry_grid")
    assert len(sel) >= 1
    assert all("dry_grid" in c.name for c in sel)


def test_select_by_grid():
    sel = select_cases(grid="96x96x96")
    assert sel
    assert all(c.grid.label == "96x96x96" for c in sel)


def test_exact_name_no_match_raises_systemexit():
    with pytest.raises(SystemExit):
        select_cases(only="=does_not_exist")


def test_substring_no_match_raises_systemexit():
    with pytest.raises(SystemExit):
        select_cases(only="zzz_nope")


def test_unknown_grid_raises_systemexit():
    with pytest.raises(SystemExit):
        select_cases(grid="1x1x1")


def test_exact_name_wrong_grid_raises_systemexit():
    # cbl_nieuwstadt exists only at 96x96x96
    with pytest.raises(SystemExit):
        select_cases(only="=cbl_nieuwstadt", grid="64x64x96")


# --- runner script exposes _build_test_matrix for the harness -----------------
def test_runner_exposes_build_test_matrix():
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "matrix"))
    try:
        import run_les_suite_matrix as run_matrix  # noqa: PLC0415
    finally:
        pass
    matrix = run_matrix._build_test_matrix()
    assert matrix
    assert all(hasattr(m, "case") and hasattr(m, "grid_type") for m in matrix)


def test_runner_list_mode_runs():
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "matrix"))
    import run_les_suite_matrix as run_matrix  # noqa: PLC0415

    rc = run_matrix.main(["--list", "--only", "=cbl_nieuwstadt"])
    assert rc == 0


def test_runner_exact_miss_exits_nonzero():
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "matrix"))
    import run_les_suite_matrix as run_matrix  # noqa: PLC0415

    with pytest.raises(SystemExit):
        run_matrix.main(["--only", "=nope_case"])
