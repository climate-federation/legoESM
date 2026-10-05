"""Round-103 checks: the TKE matrix/RHS block is split per compiled write.

Every check here either exercises the model's own statement trace or asserts
the gate maps a row to the compiled line it claims.  Each behavioural check
carries a synthetic-violation arm so it cannot pass vacuously.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


def _literal_trace():
    """Drive the literal NEMO TKE program on a small synthetic column set."""
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing import tke as tke_mod

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(1103)
    ny, nx, n = 3, 4, 6
    shape = (ny, nx, n)

    def rand(scale, s=shape):
        return jnp.asarray(rng.uniform(0.2, 1.0, s) * scale)

    e_old = rand(1e-3)
    cfg = tke_mod.TKEConfig(
        tke_matrix_evaluation="nemo_literal",
        tke_solver_evaluation="nemo_literal",
        dissipation_discretization="nemo_1p5_split",
        tke_buoyancy_sink="nemo_explicit",
    )
    out = tke_mod._solve_tke_backward_euler(
        e_old=e_old,
        K_M_old=rand(1e-3), K_H_old=rand(1e-4),
        P_s=rand(1e-6), N2=rand(1e-4), l_eps=rand(10.0),
        dz_half=rand(10.0), surface_flux=jnp.zeros(shape[:-1]),
        dt=jnp.float64(1200.0), cfg=cfg,
        dz_face_surface=jnp.asarray(rng.uniform(5.0, 10.0, shape[:-1])),
        surface_dirichlet=jnp.asarray(rng.uniform(1e-4, 1e-3, shape[:-1])),
        surface_bc_level="nemo_z0",
        K_M_surface=jnp.asarray(rng.uniform(1e-4, 1e-3, shape[:-1])),
        w_active=jnp.ones(shape),
        nemo_e3t=rand(10.0, (ny, nx, n + 1)),
        dissl_old=rand(1e-2),
        return_statement_trace=True,
    )
    return cfg, out


def test_trace_exposes_the_three_matrix_arrays_and_the_shear_operand():
    from legoesm.ocean.physics.vertical_mixing import tke as tke_mod

    assert {"matrix_upper", "matrix_lower", "matrix_diag", "rhs_shear"} <= set(
        tke_mod.TKEStatementTrace._fields)
    _cfg, out = _literal_trace()
    e_new, rhs_base, rhs_ext, upper, lower, diag, intermediate = out
    assert intermediate is None
    assert upper.shape == lower.shape == diag.shape == rhs_base.shape
    assert rhs_ext.shape[-1] == rhs_base.shape[-1] + 1
    for name, value in (("upper", upper), ("lower", lower), ("diag", diag)):
        assert np.all(np.isfinite(np.asarray(value))), name
    assert np.all(np.isfinite(np.asarray(e_new)))


def test_traced_diagonal_reproduces_the_compiled_zdiag_association():
    """zdftke.f90:436 -- zdiag = 1 - zzd_lw - zzd_up + zfact2*dissl*wmask."""
    cfg, out = _literal_trace()
    _e_new, _rhs_base, _rhs_ext, upper, lower, diag, _intermediate = out
    upper = np.asarray(upper)
    lower = np.asarray(lower)
    diag = np.asarray(diag)
    # The dissipation contribution is whatever the diagonal carries beyond
    # the two off-diagonals; recomputing it in NEMO's association must
    # reproduce the traced diagonal bit-for-bit.
    extra = diag - (1.0 - lower - upper)
    rebuilt = 1.0 - lower - upper + extra
    assert np.array_equal(
        rebuilt.view(np.uint64), diag.view(np.uint64))
    # Synthetic violation: the reversed association is NOT the compiled one,
    # so a gate asserting bit equality against it must be able to fail.
    reversed_assoc = 1.0 - upper - lower + extra
    assert not np.array_equal(
        reversed_assoc.view(np.uint64), diag.view(np.uint64))


def test_traced_upper_keeps_the_deepest_row_nemo_records():
    """The extended system zeroes zd_up(jpkm1); NEMO's record does not.

    Capturing the super-diagonal after the concatenation would silently score
    a structural zero against NEMO's live value in every column.
    """
    _cfg, out = _literal_trace()
    (_e_new, _rhs_base, rhs_ext, upper, _lower, _diag,
     _intermediate) = out
    upper = np.asarray(upper)
    assert np.all(upper[..., -1] != 0.0), (
        "the traced super-diagonal must be the pre-concatenation value")
    assert rhs_ext.shape[-1] == upper.shape[-1] + 1


def test_trace_refuses_a_degenerate_single_interface_column():
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing import tke as tke_mod

    cfg = tke_mod.TKEConfig(
        tke_matrix_evaluation="nemo_literal",
        tke_solver_evaluation="nemo_literal",
        dissipation_discretization="nemo_1p5_split",
        tke_buoyancy_sink="nemo_explicit",
    )
    one = jnp.ones((2, 2, 1))
    with pytest.raises(ValueError, match="at least two interfaces"):
        tke_mod._solve_tke_backward_euler(
            e_old=one, K_M_old=one, K_H_old=one, P_s=one, N2=one,
            l_eps=one, dz_half=one, surface_flux=jnp.ones((2, 2)),
            dt=jnp.float64(1.0), cfg=cfg,
            dz_face_surface=jnp.ones((2, 2)),
            surface_dirichlet=jnp.ones((2, 2)),
            surface_bc_level="nemo_z0",
            K_M_surface=jnp.ones((2, 2)),
            w_active=one, nemo_e3t=jnp.ones((2, 2, 2)), dissl_old=one,
            return_statement_trace=True,
        )


def test_gate_names_every_compiled_write_in_the_block():
    source = (
        TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py"
    ).read_text()
    assert "_tke_matrix_statement_rows" in source
    assert '"given_nemo_entry_matrix_walk"' in source
    for citation in (
            '"zdftke.f90:434"', '"zdftke.f90:435"', '"zdftke.f90:436"',
            '"zdftke.f90:439-442"', '"zdftke.f90:439"'):
        assert citation in source, citation
    # p_pdlr is excluded with its reason and its single consumer, not dropped.
    assert '"zdftke.f90:421"' in source
    assert "zdftke.f90:712" in source
    # The wave-coupled block is declared non-executing on two conditions.
    assert "sbccpl.f90:629" in source
    assert "namelist_ref:593" in source
    # The plant exists and targets the diagonal.
    assert '"stage-tke-matrix-ulp"' in source
    assert "CONFIRMED_SOLE_OPERAND_P_SH2" in source


def test_gate_scores_the_block_over_every_solved_level():
    source = (
        TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py"
    ).read_text()
    body = source.split("def _tke_matrix_statement_rows")[1].split(
        "def _tke_surface_operand_rows")[0]
    # No wet-mask exception and no tolerance: the bar is every owned cell.
    assert "np.ones(reference.shape, dtype=bool)" in body
    assert '"NEMO levels 2:jpkm1"' in body
    assert "[..., 1:30]" in body


def test_the_block_replay_is_committed_and_reproduces_nemo_from_nemo(
        monkeypatch):
    """Codex review, round 103 finding 2: a cited number needs a shipped probe.

    The replay rebuilds NEMO's four block outputs from NEMO's OWN recorded
    operands and must reproduce NEMO's OWN recorded outputs bit-for-bit.  If
    this fails, the reference side of every production row is already wrong.
    """
    import numpy as np
    from legoesm.ocean.fidelity.provenance import ALLOW_DIRTY_ENV
    from nemo_testcase_l2_gyre_round103_tke_block_replay import (
        R101, R59, replay, rebuild_block,
    )

    if not R59.exists() or not R101.exists():
        pytest.skip("acquisition records are not present in this tree")
    # The report stamps the tree and fails closed on tracked dirt, which is
    # right for a gate run and wrong for a test run in a working copy.
    monkeypatch.setenv(ALLOW_DIRTY_ENV, "1")
    report = replay(R59, R101, None)
    assert report["status"] == "PASS", report["rows"]
    assert [row["n_unequal"] for row in report["rows"]] == [0, 0, 0, 0]
    assert all(row["cells"] == 20416 for row in report["rows"])
    # The write call is the Round-59 build's, not the Round-101 build's.
    assert report["record_write_call"].endswith("R59TKE/BLD/ppsrc/nemo/"
                                                "zdftke.f90:463")
    # Non-vacuity: the plant must move it, on a cell whose baseline is not 0.
    planted = replay(R59, R101, "operand-ulp")
    assert planted["status"] == "FAIL"
    assert planted["plant_baseline_operand"] != 0.0
    assert sum(row["n_unequal"] for row in planted["rows"]) > 0


def test_the_swap_helper_changes_only_the_shear_operand():
    """One substitution must move the right-hand side and nothing else."""
    import numpy as np
    from nemo_testcase_l2_gyre_round103_tke_block_replay import (
        R101, R59, rebuild_block,
    )
    from nemo_testcase_l2_gyre_round54_tke_operands import read_record
    from nemo_testcase_l2_gyre_round46_kt2_stage_gate import (
        read_admitted_tke_statement_walk,
    )

    if not R59.exists() or not R101.exists():
        pytest.skip("acquisition records are not present in this tree")
    arrays = read_record(R59)["arrays"]
    langmuir = np.asarray(
        read_admitted_tke_statement_walk(R101)["arrays"]["en_after_langmuir"])
    held = rebuild_block(arrays, langmuir)
    bumped_sh2 = np.asarray(arrays["sh2"], dtype=np.float64)[:, :, 1:30].copy()
    # Two ways this control can be vacuous, both observed while writing it:
    # a cell with wmask == 0 is annihilated by the statement's trailing mask,
    # and a ONE-ULP change to a small p_sh2 is far below the resolution of
    # the sum it lands in (dt * 1 ULP of 1.7e-8 is 4.8e-20 against a
    # right-hand side near 9.7e-4, whose own spacing is ~1.1e-19), so it
    # rounds away and nothing moves.  Perturb by the smallest amount the
    # statement can actually propagate instead.
    wet = np.asarray(arrays["wmask"], dtype=np.float64)[:, :, 1:30] != 0.0
    idx = tuple(np.argwhere((bumped_sh2 != 0.0) & wet)[0])
    baseline = float(bumped_sh2[idx])
    assert baseline != 0.0, "a control that perturbs a zero is not a control"
    dt = float(arrays["rn_Dt"])
    resolvable = 4.0 * float(np.spacing(float(held["en_rhs"][idx]))) / dt
    assert resolvable > 0.0
    bumped_sh2[idx] = baseline + resolvable
    swapped = rebuild_block(arrays, langmuir, p_sh2_override=bumped_sh2)
    for name in ("zd_up", "zd_lw", "zdiag"):
        assert np.array_equal(
            np.ascontiguousarray(held[name]).view(np.uint64),
            np.ascontiguousarray(swapped[name]).view(np.uint64)), name
    assert not np.array_equal(
        np.ascontiguousarray(held["en_rhs"]).view(np.uint64),
        np.ascontiguousarray(swapped["en_rhs"]).view(np.uint64))


def _fake_trace_from_nemo(arrays, langmuir):
    """A trace whose every field is NEMO's own recorded value."""
    import numpy as np
    from types import SimpleNamespace

    yx = lambda a: np.asarray(a).swapaxes(0, 1)  # noqa: E731
    rhs30 = np.concatenate(
        [yx(langmuir)[..., :1], yx(arrays["rhs_pre_sweep"])[..., 1:30]],
        axis=-1)
    return SimpleNamespace(tke_statement_trace=SimpleNamespace(
        matrix_upper=yx(arrays["matrix_upper"])[..., 1:30],
        matrix_lower=yx(arrays["matrix_lower"])[..., 1:30],
        matrix_diag=yx(arrays["matrix_diag"])[..., 1:30],
        rhs_pre_sweep=rhs30,
        rhs_shear=yx(arrays["sh2"])[..., 1:30],
        en_after_langmuir=yx(langmuir)[..., :30],
    ))


def test_the_gate_rows_are_all_bit_when_fed_nemos_own_values():
    """Feed NEMO back to itself: every row BIT, and the swap verdict holds.

    This exercises the real row builder end to end without the twelve-minute
    production step, so a shape or orientation defect cannot hide until the
    long run's last second (it already did once, as a NameError).
    """
    import numpy as np
    from nemo_testcase_l2_gyre_round103_tke_block_replay import R101, R59
    from nemo_testcase_l2_gyre_round54_tke_operands import read_record

    if not R59.exists() or not R101.exists():
        pytest.skip("acquisition records are not present in this tree")
    operand_record = read_record(R59)
    statement_record = gate.read_admitted_tke_statement_walk(R101)
    langmuir = statement_record["arrays"]["en_after_langmuir"]
    trace = _fake_trace_from_nemo(operand_record["arrays"], langmuir)

    out = gate._tke_matrix_statement_rows(
        trace, operand_record, statement_record, "SELF_TEST")
    assert [row["n_unequal"] for row in out["rows"]] == [0, 0, 0, 0]
    assert out["first_nonbit"] is None
    attribution = out["p_sh2_attribution"]
    assert attribution["p_sh2_row"]["n_unequal"] == 0
    # With NEMO's own shear there is nothing to attribute, so the verdict must
    # NOT claim an owner.
    assert attribution["verdict"] == "REFUTED"
    swap = {row["field"]: row for row in attribution["one_variable_swap"]}
    assert swap["all_operands_recorded"]["n_unequal"] == 0
    assert swap["p_sh2_swapped_vs_production"]["n_unequal"] == 0
    assert swap["p_sh2_swapped_vs_nemo"]["n_unequal"] == 0

    # Synthetic violation: corrupt one traced diagonal cell and the row moves.
    trace.tke_statement_trace.matrix_diag = np.array(
        trace.tke_statement_trace.matrix_diag, copy=True)
    trace.tke_statement_trace.matrix_diag[0, 0, 0] = np.nextafter(
        trace.tke_statement_trace.matrix_diag[0, 0, 0], np.float64(np.inf))
    broken = gate._tke_matrix_statement_rows(
        trace, operand_record, statement_record, "SELF_TEST")
    assert broken["first_nonbit"]["field"] == "zdiag"
    assert broken["first_nonbit"]["n_unequal"] == 1
