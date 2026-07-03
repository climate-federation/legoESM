"""Direct tests for scripts/data/generate_aerobulk_reference.py.

The generator itself needs aerobulk-python (a separate conda env), so only
the pure-Python pieces are exercised here: deterministic input sampling and
schema agreement between the committed baseline and the script's declared
layout (a regenerated baseline must not silently change shape or coverage).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.data.generate_aerobulk_reference import (
    ALGOS,
    HEIGHTS,
    N_RANDOM,
    NITER,
    SEED,
    sample_marine_inputs,
)

_BASE = Path(__file__).parent / "baselines" / "aerobulk_noskin_v1"


def test_sampling_is_deterministic_and_physical():
    a = sample_marine_inputs()
    b = sample_marine_inputs()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    n = a["sst"].shape[0]
    assert n > N_RANDOM  # random + structured edge cases
    assert np.all((a["sst"] >= 270.0) & (a["sst"] <= 320.0))
    assert np.all((a["q_air"] > 0.0) & (a["q_air"] < 0.08))
    assert np.all((a["slp"] > 80000.0) & (a["slp"] < 110000.0))
    assert np.all(np.hypot(a["u"], a["v"]) < 50.0)


def test_committed_baseline_matches_script_schema():
    d = np.load(_BASE.with_suffix(".npz"))
    inp = sample_marine_inputs()
    n = inp["sst"].shape[0]
    # inputs in the committed baseline are exactly the script's sample
    for k, v in inp.items():
        np.testing.assert_allclose(d[f"in_{k}"], v, rtol=0, atol=0)
    # every (algo, height) output block present with matching length
    for algo in ALGOS:
        for zt, zu in HEIGHTS:
            tag = f"{algo}_zt{int(zt)}_zu{int(zu)}"
            for name in ("ql", "qh", "taux", "tauy", "evap"):
                arr = d[f"{tag}_{name}"]
                assert arr.shape == (n,), f"{tag}_{name}"
                assert np.all(np.isfinite(arr)), f"{tag}_{name} non-finite"
    meta = json.loads(_BASE.with_suffix(".json").read_text())
    assert meta["seed"] == SEED
    assert meta["niter"] == NITER
    assert meta["n_total"] == n
