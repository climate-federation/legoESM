import copy

import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round27_consumer_gate as gate,
)


def _score(exact):
    return {"cells": 2, "unequal": 0 if exact else 1,
            "bit_identical": exact, "max_abs": 0.0 if exact else 1.0}


def _capture(variant):
    direct = {
        "kbb_vs_kmm_u": _score(False),
        "kbb_vs_kmm_v": _score(True),
        "kbb_grad_div_u_vs_base": _score(False),
        "kbb_grad_div_v_vs_base": _score(False),
        "kbb_curl_u_vs_base": _score(True),
        "kbb_curl_v_vs_base": _score(True),
        "kmm_grad_div_u_vs_base": _score(True),
        "kmm_grad_div_v_vs_base": _score(True),
        "kmm_curl_u_vs_base": _score(False),
        "kmm_curl_v_vs_base": _score(False),
    }
    return {
        "helper_variant": variant,
        "consumer_inventory": [
            "dynldf_lev F-curl thickness",
            "dynvor EEN potential-vorticity thickness",
        ],
        "dtype": "float64", "backend": "cpu", "record_root": "/record",
        "kt1_entry_velocity_max_abs": 0.0,
        "kt1_ldf": {"u_max_abs": 0.0, "v_max_abs": 0.0},
        "kt2_direct_ldf": direct,
    }


def _arrays(vorticity):
    return {
        "kt1_ldf_u": np.zeros(2), "kt1_ldf_v": np.zeros(2),
        "stage2_vorticity_u": np.array([vorticity, 0.0]),
        "stage2_vorticity_v": np.zeros(2),
    }


def test_consumer_gate_retracts_confound_and_plants_bind():
    parent, raw = _capture("parent"), _capture("raw_f")
    result = gate.compare_captures(
        parent, raw, _arrays(0.0), _arrays(1.0), plant=False)
    assert result["status"] == "HELD"
    assert "withdrawn" in result["retraction"]
    with pytest.raises(gate.GateError, match="planted"):
        gate.compare_captures(
            parent, raw, _arrays(0.0), _arrays(1.0), plant=True)

    missing = copy.deepcopy(raw)
    missing["consumer_inventory"].pop()
    with pytest.raises(gate.GateError, match="inventory"):
        gate.compare_captures(
            parent, missing, _arrays(0.0), _arrays(1.0), plant=False)
