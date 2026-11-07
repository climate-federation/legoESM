from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_round72_stage1_transport_gate.py"
)
SPEC = importlib.util.spec_from_file_location("round72_stage1_transport", SCRIPT)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def test_transport_replay_and_ulp_plant(monkeypatch) -> None:
    shape2, shape3 = (2, 2), (2, 2, 3)
    mask = np.ones(shape3, dtype=np.float64)
    fields = {"header": (2, 2, 1, 3, 36, 26, 31, 64)}
    for face, metric_name in (("u", "e2u"), ("v", "e1v")):
        adv = np.full(shape2, 3.0)
        inverse_depth = np.full(shape2, 0.25)
        barotropic = np.full(shape2, 0.5)
        correction = adv * inverse_depth - barotropic
        velocity = np.full(shape3, 2.0)
        thickness = np.full(shape3, 4.0)
        metric = np.full(shape2, 5.0)
        fields.update({
            f"{face}n_adv": adv,
            f"r1_h{face}": inverse_depth,
            f"{face}{face}_b": barotropic,
            f"z{face}b": correction,
            f"{face}{face}": velocity,
            f"{face}mask": mask,
            f"e3{face}": thickness,
            metric_name: metric,
            f"zF{face}": (
                metric[..., None] * thickness
            ) * (velocity + correction[..., None] * mask),
        })
    monkeypatch.setattr(GATE, "_read", lambda _path: fields)
    rows = GATE.validate(Path("unused"))
    assert all(row["bit_exact"] for row in rows.values())
    with pytest.raises(SystemExit, match="not bit-exact"):
        GATE.validate(Path("unused"), replay_ulp=True)


def test_bit_comparison_distinguishes_signed_zero() -> None:
    assert not GATE._bits_equal(np.array([0.0]), np.array([-0.0]))
