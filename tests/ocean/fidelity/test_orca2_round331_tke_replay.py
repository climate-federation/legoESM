"""Direct tests for the round-331 offline TKE statement replay."""

from __future__ import annotations

import numpy as np

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round331_tke_replay as replay,
)


def test_score_by_level_is_bit_exact_and_its_plant_moves_one_row():
    expected = np.zeros((2, 3, 4), dtype=np.float64)
    support = np.ones_like(expected, dtype=bool)
    exact = replay.score_by_level(expected.copy(), expected, support)
    assert exact["bit_identical"]
    assert exact["unequal_by_level"] == [0, 0, 0, 0]

    planted = expected.copy()
    planted[1, 2, 3] = np.nextafter(0.0, np.inf)
    moved = replay.score_by_level(planted, expected, support)
    assert not moved["bit_identical"]
    assert moved["unequal"] == 1
    assert moved["unequal_by_level"] == [0, 0, 0, 1]


def test_boundary_score_does_not_hide_a_dry_solve_operand_plant():
    expected = np.zeros((1, 1, 2), dtype=np.float64)
    actual = expected.copy()
    actual[..., 1] = 1.0
    wet = np.array([[[True, False]]])
    scored = replay.score_boundary(actual, expected, wet)
    assert scored["bit_identical"]
    assert not scored["all_loop_cells"]["bit_identical"]
    assert scored["all_loop_cells"]["unequal"] == 1


def test_assemble_uses_the_recorded_i_rank_decomposition():
    records = {
        0: {"fields": {"x": np.full((2, 3, 1), 7.0)}},
        1: {"fields": {"x": np.full((2, 3, 1), 9.0)}},
    }
    result = replay.assemble(records, "x")
    assert result.shape == (3, 4, 1)
    assert np.all(result[:, :2] == 7.0)
    assert np.all(result[:, 2:] == 9.0)


def test_replay_names_every_admitted_statement_boundary():
    source = replay.Path(replay.__file__).read_text()
    for name in (
        "shear_operand", "entry_avm", "entry_dissl",
        "after_boundary_conditions", "matrix_upper", "matrix_lower",
        "matrix_diagonal", "after_matrix_rhs", "after_solve_and_floor",
        "after_tke_energy", "after_tke_avm", "after_tke_avt",
    ):
        assert f'"{name}"' in source
    assert "expose_live_stage_operands=True" in source
