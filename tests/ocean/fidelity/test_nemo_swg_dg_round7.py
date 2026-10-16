"""Direct tests for the SWG_DG round-7 restart discriminator."""
from __future__ import annotations

import importlib.util
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

_PATH = (Path(__file__).resolve().parents[3]
         / "scripts/validate/ocean_fidelity/testcases/"
         "nemo_testcase_l1_swg_dg_round7.py")
_SPEC = importlib.util.spec_from_file_location("swg_dg_round7", _PATH)
round7 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(round7)


class _Ladder:
    @staticmethod
    def _localized_comparison(reference, candidate, mask):
        return {"all": {"population": int(np.count_nonzero(mask))}}


def test_exact_row_distinguishes_signed_zero_and_localizes_one_ulp():
    reference = np.asarray([[0.0, 1.0]], dtype=np.float64)
    candidate = np.asarray(
        [[-0.0, np.nextafter(1.0, np.inf)]], dtype=np.float64)
    row = round7.exact_row(
        reference, candidate, np.ones_like(reference, dtype=bool),
        name="synthetic", ladder=_Ladder())
    assert row["bit_unequal"] == 2
    assert row["numeric_unequal"] == 1
    assert row["signed_zero_only"] == 1
    assert row["first_unequal_cell"] == [0, 0]
    assert row["status"] == "DEBT"


def test_compare_named_fields_refuses_missing_coverage():
    field = np.zeros((2, 2), dtype=np.float64)
    masks = {"T": np.ones((2, 2), dtype=bool)}
    with pytest.raises(RuntimeError, match="missing candidate field"):
        round7.compare_named_fields(
            {"sshn": field}, {}, (("ssh_k", "sshn", "T"),), masks,
            prefix="missing", ladder=_Ladder())


def test_advance_stage_chain_calls_all_stages_with_fixed_before_boundary():
    Entry = namedtuple("Entry", "value")

    class Model:
        def __init__(self):
            self.calls = []

        def replay_swe_rk3_stage(self, current, dt, stage, before):
            self.calls.append((current.value, dt, stage, before.value))
            return Entry(current.value + stage)

    model = Model()
    s1, s2, s3 = round7.advance_stage_chain(model, Entry(10), 50.0)
    assert (s1.value, s2.value, s3.value) == (11, 13, 16)
    assert model.calls == [
        (10, 50.0, 1, 10),
        (11, 50.0, 2, 10),
        (13, 50.0, 3, 10),
    ]


def test_restart_specs_cover_every_field_once():
    restart = {
        "sshn": None, "un": None, "vn": None, "uu_n": None,
        "vv_n": None, "utau_b": None, "vtau_b": None,
        "emp_b": None, "qns_b": None, "sfx_b": None,
    }
    specs = round7._restart_specs(restart)
    assert [row[0] for row in specs] == list(restart)
    assert dict((name, kind) for name, _, kind in specs) == {
        "sshn": "T", "un": "U", "vn": "V", "uu_n": "U",
        "vv_n": "V", "utau_b": "U", "vtau_b": "V",
        "emp_b": "T", "qns_b": "T", "sfx_b": "T",
    }


def test_json_writer_refuses_overwrite(tmp_path):
    path = tmp_path / "round7.json"
    round7._write_json(path, {"answer": np.int64(7)})
    assert path.read_text() == '{\n  "answer": 7\n}\n'
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        round7._write_json(path, {"answer": 8})


def test_signed_zero_scope_reports_forcing_debt_but_rejects_prognostic_scope():
    forcing = {
        "forcing": [{"kt": 1, "rows": [{
            "name": "production_boundary.kt1.vtau_before",
            "signed_zero_only": 640,
        }]}],
    }
    scope = round7.signed_zero_scope(forcing)
    assert scope["forcing_only_observation"]
    assert scope["policy"] == (
        "maintainer_ruling_2026-10-13: prognostic fields exact; "
        "sign of zero in forcing fields excepted")
    assert "pending" not in scope["policy"]
    assert not scope["prognostic_or_unknown_violations"]

    prognostic = {
        "plant": [{"kt": 1, "rows": [{
            "name": "production_state_vs_restart.kt1.sshn",
            "signed_zero_only": 1,
        }]}],
    }
    planted = round7.signed_zero_scope(prognostic)
    assert not planted["forcing_only_observation"]
    assert planted["prognostic_or_unknown_violations"][0]["field"] == "sshn"
