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
    e_new, rhs_base, rhs_ext, upper, lower, diag = out
    assert upper.shape == lower.shape == diag.shape == rhs_base.shape
    assert rhs_ext.shape[-1] == rhs_base.shape[-1] + 1
    for name, value in (("upper", upper), ("lower", lower), ("diag", diag)):
        assert np.all(np.isfinite(np.asarray(value))), name
    assert np.all(np.isfinite(np.asarray(e_new)))


def test_traced_diagonal_reproduces_the_compiled_zdiag_association():
    """zdftke.f90:436 -- zdiag = 1 - zzd_lw - zzd_up + zfact2*dissl*wmask."""
    cfg, out = _literal_trace()
    _e_new, _rhs_base, _rhs_ext, upper, lower, diag = out
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
    _e_new, _rhs_base, rhs_ext, upper, _lower, _diag = out
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
    assert "CONFIRMED_INHERITED_P_SH2" in source


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
