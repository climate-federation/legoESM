import copy
from types import SimpleNamespace

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round206_omt0_atomic_fold_census as gate,
)
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    nemo_literal_external_mode_active,
)


def _row(kt, checkpoint, field, *, rms, maximum, exact=False):
    return {
        "kt": kt,
        "checkpoint": checkpoint,
        "field": field,
        "bit_identical": exact,
        "unequal": 0 if exact else 1,
        "first_unequal_index": None if exact else [0, 0],
        "max_abs": maximum,
        "rms": rms,
        "mean_abs_over_unequal": rms,
    }


def _ladder(candidate=False):
    rows = []
    for kt in range(1, 11):
        for checkpoint in ("entry", "stage1", "stage2", "stage3"):
            for field in ("T", "S", "u", "v", "ssh"):
                exact = checkpoint == "entry"
                value = 0.0 if exact else 2.0
                if candidate and not exact:
                    value = 1.0
                rows.append(_row(
                    kt, checkpoint, field, rms=value, maximum=value,
                    exact=exact))
    return {
        "row_count": 200,
        "first_non_bit_checkpoint": {"kt": 1, "checkpoint": "stage1", "field": "T"},
        "private_arm": gate.PRIVATE_ARM if candidate else gate.BASE_ARM,
        "rows": rows,
    }


def _documents():
    base = {label: _ladder(False) for label in gate.LABELS}
    candidate = {label: _ladder(True) for label in gate.LABELS}
    exact = {"comparison_bit_exact": True}
    state = [{
        "substep": index,
        "ssh": {"value": index},
        "ua_b": {"value": index},
        "va_b": {"value": index},
    } for index in range(1, 66)]
    substep = {
        "slow_v_arm": {"input": exact},
        "association_arm": {"substep_table": copy.deepcopy(state)},
        "slow_v_association_unit": {"substep_table": copy.deepcopy(state)},
        "complete_fold_unit": {"substep_table": copy.deepcopy(state)},
    }
    return base, candidate, substep


def test_clean_net_improvement_is_eligible():
    result = gate.classify(*_documents())
    assert result["status"] == "ELIGIBLE_ATOMIC_OMT0_FOLD_UNIT"
    assert result["decision96_eligible"] is True


@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_every_plant_fires(plant):
    with pytest.raises((gate.GateError, gate.compare_gate.GateError)):
        gate.classify(*_documents(), plant=plant)


def test_candidate_that_moves_majority_away_is_held():
    base, candidate, substep = _documents()
    for label in gate.LABELS:
        for row in candidate[label]["rows"]:
            if row["checkpoint"] != "entry":
                row["rms"] = 3.0
                row["max_abs"] = 3.0
                row["mean_abs_over_unequal"] = 3.0
    result = gate.classify(base, candidate, substep)
    assert result["status"] == "HELD_ATOMIC_OMT0_FOLD_UNIT"
    assert result["decision96_eligible"] is False


def test_literal_external_mode_predicate_is_atomic():
    barotropic = SimpleNamespace(
        barotropic_face_depth="nemo_ssh_avg",
        barotropic_continuity_evaluation="nemo_literal",
        barotropic_transport_accumulation_evaluation="nemo_literal",
    )
    config = SimpleNamespace(
        momentum_time_integrator="rk3_ws",
        momentum_advection="flux_form",
        barotropic=barotropic,
    )
    assert nemo_literal_external_mode_active(config)
    for field in (
        "barotropic_face_depth",
        "barotropic_continuity_evaluation",
        "barotropic_transport_accumulation_evaluation",
    ):
        changed = SimpleNamespace(**vars(barotropic))
        setattr(changed, field, "legacy")
        assert not nemo_literal_external_mode_active(
            SimpleNamespace(**{**vars(config), "barotropic": changed}))


def test_ladder_gate_has_explicit_before_and_after_atomic_control():
    source = gate.omt0.Path(gate.omt0.__file__).read_text(encoding="utf-8")
    assert "barotropic_atomic_fold_unit=atomic_fold_unit" in source


def test_literal_unit_keeps_exact_cartesian_depth_representation():
    source = gate.omt0.Path(
        gate.omt0.__file__).parents[4] / "packages/ocean/legoesm/ocean/dynamics" \
        / "barotropic_latlon_cgrid.py"
    text = source.read_text(encoding="utf-8")
    assert "if (_raw_hu_0 is None) != (_raw_hv_0 is None):" in text
    assert "if _raw_hu_0 is not None:" in text
