from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round219_omt2_ladder_gate as gate,
)


DECK_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0"
)


def test_omt2_card_is_exact_shared_linear_drag_edge() -> None:
    card = gate.build_omt2_card(DECK_ROOT)
    report = gate.validate_omt2_card(DECK_ROOT, card)
    assert report["changed_bottom_drag_fields"] == {
        "bottom_drag_scheme": {"before": "legacy", "after": "nemo_linear"},
    }
    assert set(report["changed_composition_fields"]) == {
        "zdf_drag_in_matrix", "barotropic_drag_substep",
    }
    assert report["resolved_linear_drag"] == {
        "bottom_drag_scheme": "nemo_linear",
        "rn_Cd0": 1.0e-3,
        "rn_Uc0": 0.4,
        "zdf_drag_in_matrix": True,
        "zdf_baroclinic_only": True,
        "barotropic_drag_substep": True,
    }


def test_omt2_card_module_plant_fires() -> None:
    with pytest.raises(gate.GateError, match="bottom-drag edge moved"):
        gate.validate_omt2_card(
            DECK_ROOT, gate.build_omt2_card(DECK_ROOT, plant="card-module"))
