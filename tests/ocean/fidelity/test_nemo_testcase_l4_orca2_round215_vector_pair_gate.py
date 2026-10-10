from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round215_vector_pair_gate as gate,
)
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    nemo_raw_surface_vmask,
    nemo_vector_form_update_active,
)


REPORT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round215/"
    "vector_pair_replay.json"
)


def test_vector_form_predicate_and_raw_surface_mask() -> None:
    vector = SimpleNamespace(
        momentum_time_integrator="rk3_ws",
        momentum_advection="vector_invariant",
    )
    assert nemo_vector_form_update_active(vector)
    assert not nemo_vector_form_update_active(SimpleNamespace(
        momentum_time_integrator="rk3",
        momentum_advection="vector_invariant",
    ))
    raw = SimpleNamespace(vmask=np.array([
        [[0.0, 1.0], [0.0, 0.0]],
        [[1.0, 1.0], [0.0, 1.0]],
    ]))
    actual = np.asarray(nemo_raw_surface_vmask(
        SimpleNamespace(nemo_een_barotropic=raw), np.float64))
    assert np.array_equal(actual, np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
    ]))


@pytest.mark.skipif(not REPORT.is_file(), reason="round-215 measurement not present")
def test_round215_report_classifies() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert gate.classify(report)["status"] == "PASS_R215_OMT1_VECTOR_PAIR_REPLAY"


@pytest.mark.skipif(not REPORT.is_file(), reason="round-215 measurement not present")
@pytest.mark.parametrize("plant", gate.PLANTS[1:])
def test_round215_plants_fire(plant: str) -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    with pytest.raises(gate.GateError):
        gate.classify(report, plant)
