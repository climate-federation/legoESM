"""Direct test of the V1 substitution probe's pure helpers and arm guard."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROBE = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/testcases/"
         "nemo_testcase_v1_orca2tke/v1_substitution_probe.py")
spec = importlib.util.spec_from_file_location("v1_substitution_probe", PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_w_layout_cuts_nemo_w_levels_to_the_closure_layout():
    nk = 4
    full = np.arange(2 * (nk + 1), dtype=float).reshape(2, nk + 1)
    assert np.array_equal(probe.w_layout(full, nk + 1, nk), full)
    assert np.array_equal(probe.w_layout(full, nk, nk), full[:, :nk])
    assert np.array_equal(probe.w_layout(full, nk - 1, nk), full[:, 1:nk])
    with pytest.raises(SystemExit):
        probe.w_layout(full, nk - 2, nk)


def test_unequal_counts_last_bit_cells_and_orders_by_size():
    ref = np.ones((2, 3))
    cand = ref.copy()
    cand[0, 1] = np.nextafter(1.0, 2.0)
    cand[1, 2] = 1.5
    r = probe.unequal(ref, cand)
    assert r["unequal"] == 2 and r["cells"][0] == [1, 2] and r["max_abs"] == 0.5
    assert probe.unequal(ref, ref)["unequal"] == 0
    with pytest.raises(SystemExit):
        probe.unequal(ref, ref[:1])


def test_arms_must_start_plain_and_match_the_kind():
    base = ["--root", "x", "--restart-root", "x", "--kt", "3", "--state-in", "x",
            "--output", "x"]
    for kind, arms in (("entry", "nemo_ts"), ("entry", "plain,nemo_avt"),
                       ("closure", "plain,nemo_ts")):
        with pytest.raises(SystemExit, match="arms must start with plain"):
            probe.main(base + ["--kind", kind, "--arms", arms])
