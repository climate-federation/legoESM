"""Direct test of the closed-card state-digest instrument."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
          "testcases/nemo_testcase_l1_vortex_smt5_closed_card_hashes.py")
_SPEC = importlib.util.spec_from_file_location("smt5_closed_hashes", SCRIPT)
hashes = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hashes)


def _state(delta=0.0):
    rng = np.random.default_rng(0)
    fields = {n: SimpleNamespace(data=rng.standard_normal((3, 4, 2)))
              for n in hashes.STATE_FIELDS}
    fields["T"].data[1, 2, 0] += delta
    return SimpleNamespace(**fields)


def test_digest_is_deterministic_and_moves_on_one_ulp():
    base = hashes.state_digest(_state())
    assert base == hashes.state_digest(_state())
    one_ulp = np.spacing(_state().T.data[1, 2, 0])
    assert hashes.state_digest(_state(one_ulp)) != base


def test_digest_refuses_a_nonfinite_state():
    bad = _state()
    bad.S.data[0, 0, 0] = np.nan
    with pytest.raises(SystemExit):
        hashes.state_digest(bad)
