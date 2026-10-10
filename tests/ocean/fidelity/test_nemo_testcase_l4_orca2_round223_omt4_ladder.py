from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round223_omt4_ladder_gate as gate,
)


DECK_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0"
)


def test_omt4_card_is_exact_tracer_advection_edge() -> None:
    card = gate.build_omt4_card(DECK_ROOT)
    report = gate.validate_omt4_card(DECK_ROOT, card)
    assert report["changed_model_config_fields"] == {
        "tracer_advection": {"before": "none", "after": "fct2"},
    }
    assert report["resolved_tracer_advection"] == {
        "scheme": "fct2", "nn_fct_h": 2, "nn_fct_v": 2,
    }


def test_omt4_card_module_plant_fires() -> None:
    with pytest.raises(gate.GateError, match="edge moved"):
        gate.validate_omt4_card(
            DECK_ROOT, gate.build_omt4_card(DECK_ROOT, plant="card-module"))
