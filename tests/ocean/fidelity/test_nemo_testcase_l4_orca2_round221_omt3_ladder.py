from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round221_omt3_ladder_gate as gate,
)


DECK_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0"
)


def test_omt3_card_is_exact_momentum_ldf_edge() -> None:
    card = gate.build_omt3_card(DECK_ROOT)
    report = gate.validate_omt3_card(DECK_ROOT, card)
    assert report["changed_lateral_viscosity_fields"] == {
        "A_h": {"before": 0.0, "after": 1.0e5},
    }
    assert report["resolved_momentum_ldf"] == {
        "A_h": 1.0e5,
        "operator": "nemo_div_curl",
        "e3_weighting": "nemo_e3",
        "coefficient_source": "nemo_ahm_3d_file",
        "side_bc": "free_slip",
    }


def test_omt3_card_module_plant_fires() -> None:
    with pytest.raises(gate.GateError, match="viscosity edge moved"):
        gate.validate_omt3_card(
            DECK_ROOT, gate.build_omt3_card(DECK_ROOT, plant="card-module"))
