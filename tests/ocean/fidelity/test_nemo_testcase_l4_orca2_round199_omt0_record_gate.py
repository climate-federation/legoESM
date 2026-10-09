from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round199_omt0_record_gate as gate,
)
from scripts.validate.ocean_fidelity.testcases.nemo_testcase_oracle_gate import (
    namelist_values,
)


SOURCE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round83/"
    "acquisition/rung0_namelist_cfg"
)
CPP = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_OMIP_L4/"
    "cpp_ORCA2_OMIP_L4.fcm"
)


def test_omt0_render_is_exact_decision103_delta(tmp_path: Path) -> None:
    candidate = tmp_path / "namelist_cfg"
    candidate.write_text(gate.render_omt0(SOURCE.read_text()))

    report = gate.validate_deck(SOURCE, candidate, CPP)

    assert report["status"] == "PASS_OMT0_DECK"
    assert report["changed_assignments"] == sorted(gate.CHANGED)
    assert report["added_assignments"] == sorted(gate.ADDED)


@pytest.mark.parametrize("plant", ["extra-delta", "live-module"])
def test_omt0_deck_plants_fire(tmp_path: Path, plant: str) -> None:
    candidate = tmp_path / "namelist_cfg"
    candidate.write_text(gate.render_omt0(SOURCE.read_text()))

    with pytest.raises(gate.GateError):
        gate.validate_deck(SOURCE, candidate, CPP, plant)


@pytest.mark.parametrize(
    ("itend", "stock", "steps"),
    [
        (2, 2, (1,)),
        (gate.TWIN_ITEND, gate.TWIN_ITEND, gate.TWIN_STEPS),
        (gate.MONTH_ITEND, gate.MONTH_ITEND, gate.MONTH_STEPS),
    ],
)
def test_run_deck_protocol_is_only_run_control_delta(
    tmp_path: Path, itend: int, stock: int, steps: tuple[int, ...],
) -> None:
    canonical = gate.render_omt0(SOURCE.read_text())
    rendered = gate.render_run_deck(
        canonical, itend=itend, stock=stock, restart_steps=steps,
    )
    path = tmp_path / "namelist_cfg"
    path.write_text(rendered)
    before = namelist_values(SOURCE)
    after = namelist_values(path)

    assert int(after["namrun.nn_itend"].split()[0]) == itend
    assert int(after["namrun.nn_stock"].split()[0]) == stock
    assert tuple(int(value.strip()) for value in after["namrun.nn_stocklist"].split(",")) == steps
    assert all(key in after for key in gate.ADDED)
    assert before["namsbc.nn_fsbc"] == after["namsbc.nn_fsbc"]


def test_restart_capacity_plant_is_nonvacuous() -> None:
    with pytest.raises(gate.GateError, match="capacity 10"):
        gate.render_run_deck(
            gate.render_omt0(SOURCE.read_text()),
            itend=12,
            stock=12,
            restart_steps=tuple(range(1, 12)),
        )


def test_preflight_census_is_frozen() -> None:
    report = gate.preflight()
    assert report["status"] == "PASS_R199_OMT0_PREFLIGHT"
    assert report["twin_steps"] == list(range(1, 11))
    assert report["month_steps"] == [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    assert report["restart_list_capacity"] == 10
