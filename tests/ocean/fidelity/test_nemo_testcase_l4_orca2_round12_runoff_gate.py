"""Direct controls for round 12's ORCA2 river-runoff gate.

The gate quotes a bit-identity, so its own scoring is untrusted code until it
is shown it CAN report a difference: a scorer that always returns "equal"
would pass every row vacuously.  These controls check the two helpers the
verdict rests on, and that the gate's compiled citations name the statements
the module docstring claims.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_GATE = (REPO_ROOT / "scripts/validate/ocean_fidelity/orca2_l4"
         / "nemo_testcase_l4_orca2_round12_runoff_gate.py")
_NEMO_PPSRC = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs"
    "/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo")


@pytest.fixture(scope="module")
def gate():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "orca2_round12_runoff_gate", _GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_scorer_reports_equal_only_when_it_is_bitwise_equal(gate):
    a = np.array([1.0, -0.0, 3.5e-17, 7.0])
    where = np.ones_like(a, dtype=bool)
    assert gate._bitwise_score(a, a.copy(), where)["bit_identical"]
    moved = a.copy()
    moved[2] = np.nextafter(moved[2], np.inf)
    row = gate._bitwise_score(moved, a, where)
    assert not row["bit_identical"]
    assert row["unequal"] == 1
    # A signed zero is a DIFFERENT bit pattern and the scorer must say so,
    # because the deposit is skipped rather than added as a zero precisely to
    # keep -0.0 from becoming +0.0.
    signed = a.copy()
    signed[1] = 0.0
    assert not gate._bitwise_score(signed, a, where)["bit_identical"]


def test_the_plant_moves_exactly_one_value(gate):
    a = np.array([[1.0, 2.0], [-9.0, 0.5]])
    planted = gate._plant_one_value(a)
    assert int(np.count_nonzero(planted != a)) == 1
    assert float(np.abs(planted - a).max()) > 0.0


@pytest.mark.skipif(not _NEMO_PPSRC.is_dir(),
                    reason="the compiled ORCA2 branch is not on this machine")
def test_every_cited_line_carries_the_statement_it_is_cited_for(gate):
    expected = {
        "runoff_on": "IF( ln_rnf ) THEN",
        "content_tem": "rnf_tsc(:,:,jp_tem) = MAX( sst_m",
        "content_sal": "rnf_tsc(:,:,jp_sal) = zrnf_sal",
        "zero_salinity": "zrnf_sal = 0._wp",
        "reciprocal_first": "zdep = 1._wp / h_rnf(ji,jj)",
        "stage_switch": "SELECT CASE( kstg )",
    }
    for key, fragment in expected.items():
        citation = gate.CITATIONS[key]
        name, span = citation.rsplit(":", 1)
        path = _NEMO_PPSRC / Path(name).name
        lines = path.read_text().splitlines()
        first = int(span.split("-")[0])
        last = int(span.split("-")[-1])
        window = "\n".join(lines[first - 1:last])
        assert fragment in window, (
            f"{citation} does not carry {fragment!r}; it carries {window!r}")
