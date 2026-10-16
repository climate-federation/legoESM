"""Round 301: stage-1 U-transport factor split instrument."""

import copy
import json
import struct
import sys

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round301_u_transport_factors as gate,
)


def _operands(seed=301):
    rng = np.random.default_rng(seed)
    shape = (5, 7, 4)
    umask = (rng.random(shape) > 0.3).astype(np.float64)
    umask[0, 0] = 0.0
    return (rng.normal(0.0, 0.1, shape) * umask,          # uu(Kmm)
            rng.normal(0.0, 50.0, shape[:2]),             # un_adv
            1.0 / rng.uniform(100.0, 4000.0, shape[:2]),  # r1_hu_0/(1+r3u)
            rng.normal(0.0, 0.05, shape[:2]),             # uu_b(Kmm)
            umask,
            rng.uniform(1e4, 2e5, shape[:2]),             # e2u
            rng.uniform(5.0, 500.0, shape) * umask)       # e3u(Kmm)


def test_production_helpers_replay_nemo_association_bit_exact():
    operands = _operands()
    _, expected = gate.nemo_zfu(*operands)
    assert np.array_equal(
        gate.production_zfu(*operands).view(np.uint64), expected.view(np.uint64))


def _bt_file(tmp_path, kt):
    nx, ny = 94, 152
    blocks = [np.arange(nx * ny, dtype=np.float64).reshape((nx, ny), order="F") + n
              for n in range(4)]
    path = tmp_path / f"bt_{kt}.bin"
    path.write_bytes(b"NEMO_L1_BTFRM_1 " + struct.pack("=6i", 1, kt, 3, nx, ny, 64)
                     + b"".join(b.tobytes(order="F") for b in blocks))
    return path, blocks


def test_read_bt_returns_owned_west_half_and_refuses_wrong_step(tmp_path):
    path, blocks = _bt_file(tmp_path, 6)
    values = gate.read_bt(path, 6)
    assert values["un_adv"].shape == (148, 90)
    assert np.array_equal(values["un_adv"], blocks[2][2:-2, 2:-2].T)
    with pytest.raises(gate.GateError, match="header moved"):
        gate.read_bt(path, 7)


def _region(unequal, max_abs, rel):
    score = {"support": 10, "unequal": unequal, "max_abs": max_abs}
    return {"complete": score, "interior": score, "fold_band": score,
            "fold_oracle_max": 1.0, "fold_relative": rel}


def _reports():
    rows = {name: _region(0, 0.0, 0.0) for name in gate.SOURCE_ORDER}
    rows["uu_b_kmm"] = _region(3, 0.5, 0.5)
    rows["uu_kmm"] = _region(3, 9.0, 9.0)
    rows["zfu"] = _region(3, 99.0, 99.0)
    history = [{"kt": 1, "u": _region(0, 0.0, 0.0)},
               {"kt": 2, "u": _region(4, 0.02, 0.2)}]
    exact = {"complete": {"unequal": 0}}
    report = {"label": "independent", "passivity": {"state": True},
              "closure": exact, "statement_replay": exact, "rows": rows,
              "history": history,
              "one_bit_control": {"unequal": 1}}
    other = copy.deepcopy(report)
    other["label"] = "given_nemo_entry"
    return [report, other]


def test_classify_names_source_order_first_and_largest_input_factor():
    result = gate.classify(_reports())
    assert result["first_nonbit_factor"] == "uu_b_kmm"
    assert result["dominant_input_factor"] == "uu_kmm"
    assert result["fold_u_onset_kt"] == 2


@pytest.mark.parametrize("plant", ["one-bit", "label"])
def test_classify_plants_refuse(plant):
    with pytest.raises(gate.GateError):
        gate.classify(_reports(), plant)


@pytest.mark.parametrize("field", ["closure", "statement_replay"])
def test_classify_refuses_inexact_nemo_instrument(field):
    reports = _reports()
    reports[1][field] = {"complete": {"unequal": 1}}
    with pytest.raises(gate.GateError):
        gate.classify(reports)


def test_green_plant_has_its_own_exit_code(tmp_path, monkeypatch, capsys):
    paths = []
    for report in _reports():
        path = tmp_path / f"{report['label']}.json"
        path.write_text(json.dumps(report))
        paths += ["--report-in", str(path)]
    monkeypatch.setattr(gate, "classify", lambda reports, plant: {"status": "X"})
    monkeypatch.setattr(sys, "argv", ["gate", *paths, "--plant", "label"])
    assert gate.main() == gate.EXIT_PLANT_GREEN != gate.EXIT_REFUSE
    assert "PLANT-GREEN" in capsys.readouterr().out
