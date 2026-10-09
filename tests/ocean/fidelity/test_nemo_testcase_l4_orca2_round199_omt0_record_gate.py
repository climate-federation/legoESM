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


def test_run_deck_protocol_is_only_run_control_delta(
    tmp_path: Path,
) -> None:
    itend = gate.MONTH_ITEND
    stock = gate.MONTH_ITEND
    steps = gate.MONTH_STEPS
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


def test_frequency_run_deck_is_retracted_by_surface_cadence() -> None:
    canonical = gate.render_omt0(SOURCE.read_text())
    with pytest.raises(gate.GateError, match="nn_stock=1.*nn_fsbc=2"):
        gate.render_frequency_run_deck(canonical, itend=gate.TWIN_ITEND)


def test_adjacent_restart_list_is_refused() -> None:
    with pytest.raises(gate.GateError, match="first step|leave one step"):
        gate.render_run_deck(
            gate.render_omt0(SOURCE.read_text()),
            itend=gate.TWIN_ITEND,
            stock=gate.TWIN_ITEND,
            restart_steps=gate.TWIN_STEPS,
        )


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
    assert report["status"] == "REFUTED_R202_OMT0_FREQUENCY_RECOVERY"
    assert report["twin_steps"] == list(range(1, 11))
    assert report["twin_itend"] == 10
    assert report["twin_restart_mode"].startswith("RETRACTED")
    assert report["month_steps"] == [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    assert report["month_steps_expected_available"] == [10]
    assert report["expected_oracle_stop_step"] == 11
    assert report["restart_list_capacity"] == 10


ORACLE_STOP = """
 stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s  or  S <= 0
 kt 11 |ssh| max   3.853     at i j     9  90    MPI rank 0
 kt 11 |U|   max   3.041     at i j k  21  84 27 MPI rank 0
 kt 11 |V|   max   10.24     at i j k  22  84 27 MPI rank 0
 kt 11 Sal   min   21.58     at i j k  37 133  1 MPI rank 0
 kt 11 Sal   max   37.25     at i j k  35  19  5 MPI rank 0
"""


def test_oracle_stop_boundary_is_exact_and_plant_fires() -> None:
    report = gate._oracle_stop_report(ORACLE_STOP)

    assert report["step"] == 11
    assert report["fields"]["v"] == {
        "value": 10.24,
        "location": [22, 84, 27],
    }
    with pytest.raises(gate.GateError, match="boundary moved"):
        gate._oracle_stop_report(ORACLE_STOP, "wrong-oracle-stop")
