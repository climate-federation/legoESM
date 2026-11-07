"""The UP3 momentum-advection statement belongs to the tanks, not to ORCA2.

Round 61: four rounds in a row walked NEMO's UP3 face-flux statement while
reporting its improvement as ORCA2's.  ORCA2 and GYRE take NEMO's
vector-invariant momentum advection and never evaluate that statement, so a
UP3 edit can only move OVERFLOW and LOCK_EXCHANGE.  These tests make that
mechanical instead of a thing somebody has to remember.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PROBE = (Path(__file__).resolve().parents[3] / "scripts" / "validate"
          / "ocean_fidelity" / "testcases"
          / "nemo_testcase_up3_card_scope.py")


def _probe():
    spec = importlib.util.spec_from_file_location(
        "nemo_testcase_up3_card_scope", _PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_the_tank_cards_select_nemo_up3():
    """The two vector-invariant cards cannot reach the UP3 face flux."""
    probe = _probe()
    if not probe.ORCA2_DECK.exists():
        pytest.skip("ORCA2 immutable input deck is not installed")
    scope = probe.card_scope()
    assert probe.check_card_scope(scope) == []
    assert scope["ORCA2-zps"][0] == "vector_invariant"
    assert scope["GYRE-zco"][0] == "vector_invariant"
    assert scope["OVERFLOW-zps"] == ("flux_form", "nemo_up3")
    assert scope["LOCK-zco"] == ("flux_form", "nemo_up3")


def test_card_scope_check_is_not_vacuous():
    """A card that swapped to the tank's advection must be caught."""
    probe = _probe()
    planted = {name: value for name, value
               in probe.EXPECTED_CARD_SCOPE.items()}
    planted["ORCA2-zps"] = ("flux_form", "nemo_up3")
    assert probe.check_card_scope(planted) != []


def test_the_up3_face_flux_has_one_guarded_production_call_site():
    """One caller, in the ``flux_form`` arm; the sibling arm is NEMO's ENE/EEN."""
    source = (Path(__file__).resolve().parents[3] / "packages" / "ocean"
              / "legoesm" / "ocean" / "dynamics"
              / "ocean_pe_latlon_cgrid.py").read_text().splitlines()
    hits = [i for i, line in enumerate(source, start=1)
            if "_bc_horizontal_momentum_advection_flux_form" in line]
    definitions = [i for i in hits if source[i - 1].lstrip().startswith("def ")]
    calls = [i for i in hits if i not in definitions]
    assert len(definitions) == 1 and len(calls) == 1
    guard = max(i for i, line in enumerate(source[:calls[0]], start=1)
                if line.strip().startswith(("if ", "elif ", "else:")))
    assert source[guard - 1].strip() == 'if _mom_adv == "flux_form":'


def test_row_ratio_and_cellwise_ratio_disagree_when_the_maxima_differ():
    """The row statistic is not the cellwise one; the probe reports both."""
    probe = _probe()
    oracle = np.array([0.0, 0.0, 0.0])
    # cell 0 already carries the row's largest error and does not move;
    # cell 1 carries a tiny error and is the one the candidate worsens.
    reference = np.array([10.0, 1.0, 0.0])
    candidate = np.array([10.0, 5.0, 0.0])
    result = probe.row_move_ratios(oracle, reference, candidate)
    assert result["max_reference_residual"] == 10.0
    assert result["max_worsening"] == 4.0
    assert result["row_ratio"] == pytest.approx(0.4)          # looks small
    assert result["cellwise_ratio_at_worst_cell"] == pytest.approx(4.0)
    assert result["n_cells_move_exceeds_own_residual"] == 1


def test_row_move_ratios_reports_no_move_as_zero():
    probe = _probe()
    oracle = np.array([1.0, 2.0])
    reference = np.array([1.5, 2.5])
    result = probe.row_move_ratios(oracle, reference, reference)
    assert result["max_worsening"] == 0.0
    assert result["n_cells_move_exceeds_own_residual"] == 0
