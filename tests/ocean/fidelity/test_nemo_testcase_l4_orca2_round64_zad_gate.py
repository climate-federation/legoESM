"""Controls for the round-64 ORCA2 source-order ZAD replay."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_round64_zad_gate.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("round64_zad_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _arrays(seed=64):
    rng = np.random.default_rng(seed)
    shape = (7, 8, 4)
    arrays = {
        "uu_Kmm": rng.normal(scale=1.0e-3, size=shape),
        "vv_Kmm": rng.normal(scale=1.0e-3, size=shape),
        "ww": rng.normal(scale=1.0e-8, size=shape),
        "e1e2t": rng.uniform(0.5, 1.5, size=shape),
        "r1_e1e2u": rng.uniform(0.5, 1.5, size=shape),
        "r1_e1e2v": rng.uniform(0.5, 1.5, size=shape),
        "e3u_Kmm": rng.uniform(1.0, 2.0, size=shape),
        "e3v_Kmm": rng.uniform(1.0, 2.0, size=shape),
        "after_keg_u": rng.normal(scale=1.0e-7, size=shape),
        "after_keg_v": rng.normal(scale=1.0e-7, size=shape),
        "umask": np.ones(shape),
        "vmask": np.ones(shape),
    }
    return arrays


def test_source_replay_and_one_ulp_plant_bind_both_components():
    header = {
        "jpi": 7,
        "jpj": 8,
        "jpk": 4,
        "jpkm1": 3,
        "ntsi": 2,
        "ntei": 5,
        "ntsj": 2,
        "ntej": 6,
    }
    arrays = _arrays()
    arrays["after_zad_u"], arrays["after_zad_v"] = gate.replay_zad(arrays, header)
    exact = gate.score(arrays, header)
    planted = gate.score(arrays, header, plant=True)
    assert exact["U"]["unequal"] == exact["V"]["unequal"] == 0
    assert planted["U"]["unequal"] == 1
    assert planted["V"]["unequal"] == 0


def test_live_thickness_mutation_is_visible():
    header = {
        "jpi": 7,
        "jpj": 8,
        "jpk": 4,
        "jpkm1": 3,
        "ntsi": 2,
        "ntei": 5,
        "ntsj": 2,
        "ntej": 6,
    }
    arrays = _arrays(seed=65)
    arrays["after_zad_u"], arrays["after_zad_v"] = gate.replay_zad(arrays, header)
    arrays["e3u_Kmm"] = np.ones_like(arrays["e3u_Kmm"])
    assert gate.score(arrays, header)["U"]["unequal"] > 0
