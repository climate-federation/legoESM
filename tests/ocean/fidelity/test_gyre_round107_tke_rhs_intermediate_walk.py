"""Round-107 controls for the one-output TKE RHS intermediate walk."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
TESTCASES = ROOT / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(TESTCASES))
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_l2_gyre_round46_kt2_stage_gate",
    TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py",
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _literal_solve(**kwargs):
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing import tke as tke_mod

    cfg = tke_mod.TKEConfig(
        tke_matrix_evaluation="nemo_literal",
        tke_buoyancy_sink="nemo_explicit",
        tke_solver_evaluation="nemo_literal",
        dissipation_discretization="nemo_1p5_split",
        surface_bc="nemo_dirichlet",
        tke_surface_bc_level="nemo_z0",
        tke_background=0.0,
        tke_surface_min=0.0,
    )
    values = dict(
        e_old=jnp.asarray([[1.0, 0.7]], dtype=jnp.float64),
        K_M_old=jnp.asarray([[0.2, 0.3]], dtype=jnp.float64),
        K_H_old=jnp.asarray([[0.1, 0.15]], dtype=jnp.float64),
        P_s=jnp.asarray([[0.01, 0.02]], dtype=jnp.float64),
        N2=jnp.asarray([[1.0e-5, -1.0e-5]], dtype=jnp.float64),
        l_eps=jnp.asarray([[1.0, 1.2]], dtype=jnp.float64),
        dz_half=jnp.asarray([[2.0, 3.0]], dtype=jnp.float64),
        surface_flux=jnp.asarray([0.0], dtype=jnp.float64),
        dt=jnp.float64(2.0),
        cfg=cfg,
        dz_surface=jnp.asarray([1.0], dtype=jnp.float64),
        surface_dirichlet=jnp.asarray([0.8], dtype=jnp.float64),
        surface_bc_level="nemo_z0",
        bottom_dirichlet=jnp.asarray([0.2], dtype=jnp.float64),
        K_M_surface=jnp.asarray([0.25], dtype=jnp.float64),
        w_active=jnp.asarray([[1.0, 0.0]], dtype=jnp.float64),
        nemo_e3t=jnp.asarray([[1.0, 2.0, 3.0]], dtype=jnp.float64),
        dissl_old=jnp.asarray([[0.1, 0.2]], dtype=jnp.float64),
        return_statement_trace=True,
    )
    values.update(kwargs)
    return tke_mod._solve_tke_backward_euler(**values)


def test_trace_returns_only_the_selected_source_intermediate():
    out = _literal_solve(rhs_intermediate="after_stratification")
    assert len(out) == 7
    selected = np.asarray(out[-1])
    p_sh2 = np.asarray([[0.01, 0.02]], dtype=np.float64)
    p_avt = np.asarray([[0.1, 0.15]], dtype=np.float64)
    rn2 = np.asarray([[1.0e-5, -1.0e-5]], dtype=np.float64)
    expected = p_sh2 - p_avt * rn2
    np.testing.assert_array_equal(
        selected.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize(
    ("selector", "expected"),
    (
        ("p_avt_operand", np.asarray([[0.1, 0.15]], dtype=np.float64)),
        ("rn2_operand", np.asarray([[1.0e-5, -1.0e-5]], dtype=np.float64)),
    ),
)
def test_posthoc_operand_selector_returns_only_named_input(selector, expected):
    out = _literal_solve(rhs_intermediate=selector)
    np.testing.assert_array_equal(
        np.asarray(out[-1]).view(np.uint64), expected.view(np.uint64))


def test_selected_association_also_runs_in_state_only_reference_closure():
    out = _literal_solve(
        rhs_intermediate="after_stratification",
        return_statement_trace=False,
    )
    assert np.asarray(out).shape == (1, 2)
    assert np.all(np.isfinite(np.asarray(out)))


def test_rhs_measurement_selectors_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _literal_solve(
            rhs_materialization="p_avt_rn2",
            rhs_intermediate="p_avt_rn2",
        )


def test_intermediate_scorer_is_bit_exact_and_ulp_plant_is_nonzero():
    from nemo_testcase_l2_gyre_round103_tke_block_replay import rebuild_block
    from nemo_testcase_l2_gyre_round54_tke_operands import read_record

    if not gate.TKE_OPERAND_RECORD.exists() or not gate.TKE_STATEMENT_ROOT.exists():
        pytest.skip("round-59/101 TKE records are unavailable")
    operands = read_record(gate.TKE_OPERAND_RECORD)
    statement = gate.read_admitted_tke_statement_walk(gate.TKE_STATEMENT_ROOT)
    rebuilt = rebuild_block(
        operands["arrays"], statement["arrays"]["en_after_langmuir"])
    reference = np.asarray(rebuilt["p_avt_rn2"]).swapaxes(0, 1)
    trace = SimpleNamespace(tke_statement_trace=SimpleNamespace(
        rhs_intermediate=reference))

    clean = gate._tke_rhs_intermediate_rows(
        trace, operands, statement, "TEST", "p_avt_rn2")
    assert clean["reference_replay"]["classification"] == "BIT"
    assert clean["row"]["classification"] == "BIT"
    assert clean["row"]["n_unequal"] == 0

    planted = gate._tke_rhs_intermediate_rows(
        trace, operands, statement, "TEST", "p_avt_rn2",
        plant="stage-tke-rhs-intermediate-ulp")
    assert planted["row"]["n_unequal"] == 1
    assert planted["row"]["plant_baseline"] != 0.0
    assert planted["plant_target"] == planted["row"]["name"]


def test_posthoc_rn2_swap_moves_the_downstream_product():
    from nemo_testcase_l2_gyre_round54_tke_operands import read_record

    if not gate.TKE_OPERAND_RECORD.exists() or not gate.TKE_STATEMENT_ROOT.exists():
        pytest.skip("round-59/101 TKE records are unavailable")
    operands = read_record(gate.TKE_OPERAND_RECORD)
    statement = gate.read_admitted_tke_statement_walk(gate.TKE_STATEMENT_ROOT)
    rn2 = np.asarray(operands["arrays"]["rn2"])[..., 1:30].swapaxes(0, 1)
    trace = SimpleNamespace(tke_statement_trace=SimpleNamespace(
        rhs_intermediate=rn2))
    clean = gate._tke_rhs_intermediate_rows(
        trace, operands, statement, "TEST", "rn2_operand")
    assert clean["row"]["classification"] == "BIT"
    assert clean["downstream_product_replay"]["classification"] == "BIT"

    moved_rn2 = rn2.copy()
    recorded_avt = np.asarray(
        operands["arrays"]["avt_entry"]
    )[..., 1:30].swapaxes(0, 1)
    all_moved = np.nextafter(rn2, np.float64(np.inf))
    before_product = recorded_avt * rn2
    after_product = recorded_avt * all_moved
    sensitive = np.argwhere(
        (rn2 != 0.0)
        & (before_product.view(np.uint64) != after_product.view(np.uint64))
    )
    assert sensitive.size
    index = tuple(int(value) for value in sensitive[0])
    moved_rn2[index] = all_moved[index]
    moved_trace = SimpleNamespace(tke_statement_trace=SimpleNamespace(
        rhs_intermediate=moved_rn2))
    moved = gate._tke_rhs_intermediate_rows(
        moved_trace, operands, statement, "TEST", "rn2_operand")
    assert moved["row"]["n_unequal"] == 1
    assert moved["downstream_product_replay"]["n_unequal"] == 1


def test_gate_enumerates_compiled_intermediates_in_source_order():
    assert gate.TKE_RHS_INTERMEDIATES == (
        "p_avt_rn2",
        "zfact3_dissl",
        "dissipation_product",
        "after_stratification",
        "parenthesized_sum",
        "dt_product",
        "masked_increment",
        "final_accumulation",
    )
    assert gate.TKE_RHS_POSTHOC_OPERANDS == (
        "p_avt_operand",
        "rn2_operand",
    )


def test_round107_bit_classification_rejects_signed_zero_change():
    reference = np.asarray([-0.0], dtype=np.float64)
    candidate = np.asarray([0.0], dtype=np.float64)
    row = gate._bitwise_classification(
        gate.score(
            "round107.signed_zero.plant",
            reference,
            candidate,
            np.ones(reference.shape, dtype=bool),
        )
    )
    assert row["absolute_max"] == 0.0
    assert row["n_unequal"] == 1
    assert row["classification"] == "AT-BAR"
    assert not row["exact"]
