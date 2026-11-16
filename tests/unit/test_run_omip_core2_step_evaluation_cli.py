"""``--tke-step-evaluation``: which TKE tridiagonal step runs on the tripole card.

``tke.py`` carries two assemblies of the zdftke step: the card's ``factored``
matrix with the ``shared_thomas`` solver, and the DINO-certified
``nemo_literal`` transcription of ``zdftke.F90:403-455`` (matrix AND
recurrences). Only the DINO twin could select the literal pair; this pins the
driver flag that selects it for an OMIP arm.
"""
from __future__ import annotations

import pytest


def _core2():
    import scripts.run.run_omip_core2 as core2
    return core2


def test_card_default_is_unchanged_without_the_flag():
    cfg = _core2().orca1_zdftke_config()
    assert (cfg.tke_matrix_evaluation, cfg.tke_solver_evaluation) == ("factored", "shared_thomas")


def test_literal_sets_the_whole_bundle():
    cfg = _core2().orca1_zdftke_config(step_evaluation="nemo_literal")
    assert (cfg.tke_matrix_evaluation, cfg.tke_solver_evaluation) == ("nemo_literal", "nemo_literal")
    # operands of the transcribed matrix (tke.py:1220, :2786-2800)
    assert cfg.tke_n2_evaluation_stage == "step_entry"
    assert cfg.tke_dry_wmask is True


def test_card_default_bundle_is_not_the_literal_one():
    cfg = _core2().orca1_zdftke_config()
    assert cfg.tke_n2_evaluation_stage == "implicit_solve_state"
    assert cfg.tke_dry_wmask is False


def test_validator_demands_the_two_companion_flags():
    v = _core2()._validate_tke_card_grid
    with pytest.raises(SystemExit, match="nemo_z0"):
        v("tripole", "tke", tke_step_evaluation="nemo_literal")
    with pytest.raises(SystemExit, match="carried_previous_step"):
        v("tripole", "tke", tke_step_evaluation="nemo_literal", tke_surface_bc_level="nemo_z0")
    v("tripole", "tke", tke_step_evaluation="nemo_literal", tke_surface_bc_level="nemo_z0",
      tke_preclosure_coeff_source="carried_previous_step")


def test_explicit_factored_is_the_card_pair():
    cfg = _core2().orca1_zdftke_config(step_evaluation="factored")
    assert (cfg.tke_matrix_evaluation, cfg.tke_solver_evaluation) == ("factored", "shared_thomas")


def test_unknown_value_raises():
    with pytest.raises(ValueError, match="step_evaluation"):
        _core2().orca1_zdftke_config(step_evaluation="literal")


def test_it_rides_the_tripole_vmix_builder():
    vm = _core2().build_tripole_vmix_config("tke", tke_step_evaluation="nemo_literal")
    assert vm.tke.tke_matrix_evaluation == "nemo_literal"
    assert vm.tke.tke_solver_evaluation == "nemo_literal"


def test_builder_rejects_the_flag_off_the_tke_closure():
    with pytest.raises(ValueError, match="--tke-step-evaluation"):
        _core2().build_tripole_vmix_config("none", tke_step_evaluation="nemo_literal")


def test_validator_rejects_the_flag_off_the_tripole_tke_lane():
    with pytest.raises(SystemExit, match="--tke-step-evaluation"):
        _core2()._validate_tke_card_grid("mpas", "tke", tke_step_evaluation="nemo_literal", mpas_vmix="tke")


def test_parser_accepts_the_flag():
    p = _core2()._build_arg_parser()
    assert p.parse_args([]).tke_step_evaluation is None
    assert p.parse_args(["--tke-step-evaluation", "nemo_literal"]).tke_step_evaluation == "nemo_literal"
    with pytest.raises(SystemExit):
        p.parse_args(["--tke-step-evaluation", "literal"])
