from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (Path(__file__).parents[3] / "scripts" / "validate" /
          "ocean_fidelity" / "dino_1226" /
          "standalone_year_transfer_receipt.py")
SPEC = importlib.util.spec_from_file_location(
    "standalone_year_transfer_receipt", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RECEIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECEIPT)


def test_frame_adaptive_alignment_and_plants():
    assert RECEIPT.self_test() == {
        "full_frame_alignment": "PASS",
        "core_frame_alignment": "PASS",
        "unsupported_frame_plant": "FIRED",
        "prediction_classifier_plants": "FIRED",
    }


def test_alignment_refuses_mask_or_shape_drift():
    nemo = np.zeros((6, 8, 3))
    mask = np.ones_like(nemo, dtype=bool)
    mask[..., -1] = False
    with pytest.raises(ValueError, match="mask shape"):
        RECEIPT.align_nemo_to_standalone(
            (6, 8, 2), nemo, mask[:, :-1])
    with pytest.raises(ValueError, match="neither full"):
        RECEIPT.align_nemo_to_standalone((5, 8, 2), nemo, mask)


def test_statistics_uses_only_registered_wet_population():
    delta = np.array([[1.0, 100.0], [-1.0, 100.0]])
    wet = np.array([[True, False], [True, False]])
    row = RECEIPT.statistics(delta, wet)
    assert row == {"n": 2, "rms": 1.0, "bias": 0.0, "max_abs": 1.0}
    with pytest.raises(ValueError, match="empty"):
        RECEIPT.statistics(delta, np.zeros_like(wet))
