"""Non-vacuity checks for the ORCA2 round-16 slow-forcing gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[3]
GATE = (REPO / "scripts/validate/ocean_fidelity/orca2_l4"
        / "nemo_testcase_l4_orca2_round16_slow_forcing_gate.py")


def _gate():
    spec = importlib.util.spec_from_file_location("_r16_slow_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fields(gate, slow_u: float = 4.0):
    shape = (2, 2, 3)
    fields = {name: np.zeros(shape, dtype=np.float64) for name in (
        "pgf_u", "trd_u", "slow_u", "u_entry", "u_exit",
        "pgf_v", "trd_v", "slow_v", "v_entry", "v_exit",
    )}
    fields["pgf_u"][:] = 1.0
    fields["trd_u"][:] = 2.0
    fields["slow_u"][:] = slow_u
    fields["u_entry"][:] = 5.0
    fields["pgf_v"][:] = 2.0
    fields["trd_v"][:] = 3.0
    fields["slow_v"][:] = 1.0
    fields["v_entry"][:] = 7.0
    mask = np.ones(shape[1:], dtype=bool)
    for face in ("u", "v"):
        values = gate._arithmetic_values(
            fields, face, 0, np.float64(0.5), mask)
        fields[f"{face}_exit"][0] = values["masked_exit"]
        fields[f"{face}_exit"][1] = values["masked_exit"]
    fields["dt"] = 0.5
    return fields


def test_arithmetic_walk_localizes_a_slow_forcing_perturbation():
    gate = _gate()
    oracle = _fields(gate)
    candidate = _fields(gate, slow_u=np.nextafter(4.0, np.inf))
    masks = {"u": np.ones((2, 3), dtype=bool),
             "v": np.ones((2, 3), dtype=bool)}

    rows, controls, first = gate._arithmetic_walk(candidate, oracle, masks)

    assert first["face"] == "u"
    assert first["boundary"] == "rhs_plus_slow"
    assert rows["u"]["pgf_plus_trend"]["bit_exact"]
    assert not rows["u"]["rhs_plus_slow"]["bit_exact"]
    assert controls["u"]["oracle_replay_matches_record"]["bit_exact"]
    assert controls["u"]["candidate_replay_matches_trace"]["bit_exact"]


def test_slow_forcing_hook_is_private_and_off_by_default():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    assert _NEMOWSRK3TestHooks().barotropic_slow_forcing_override is None


def test_round15_orders_both_slow_operands_before_exits():
    gate = _gate()
    names = [name for name, _, _ in gate.round15.SOURCE_ORDER]
    assert names[-4:] == ["slow_u", "slow_v", "u_exit", "v_exit"]
