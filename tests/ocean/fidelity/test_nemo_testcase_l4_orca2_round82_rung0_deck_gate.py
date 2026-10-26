from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round82_rung0_deck_gate as gate,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (
    namelist_values,
)


SOURCE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round69/"
    "acquisition/orca1ice_surface_only_240step_np2/namelist_cfg"
)
CPP = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_OMIP_L4/"
    "cpp_ORCA2_OMIP_L4.fcm"
)


def _candidate(tmp_path):
    path = tmp_path / "namelist_cfg"
    path.write_text(gate.render_rung0(SOURCE.read_text()))
    return path


def test_rung0_render_has_frozen_switches(tmp_path):
    candidate = _candidate(tmp_path)
    values = namelist_values(candidate)
    for key, wanted in gate.EXPECTED.items():
        assert gate._normalise(values[key]) == wanted
    assert gate.validate(SOURCE, candidate, CPP)["status"] == "PASS_RUNG0_DECK"


@pytest.mark.parametrize(
    "plant", ["extra-delta", "mixing-value", "live-module", "cpp"]
)
def test_rung0_deck_plants_fire(tmp_path, plant):
    with pytest.raises(gate.GateError):
        gate.validate(SOURCE, _candidate(tmp_path), CPP, plant)


def test_rung0_render_is_exactly_reproducible():
    first = gate.render_rung0(SOURCE.read_text())
    second = gate.render_rung0(SOURCE.read_text())
    assert first == second
    assert first.count("&namsbc_flx") == 1
