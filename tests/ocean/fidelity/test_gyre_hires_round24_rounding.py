from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / (
    "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_hires_round24_rounding.py"
)
SPEC = importlib.util.spec_from_file_location("gyre_round24_rounding", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ROUND24 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUND24)


def _verdict_rows(self_scale: float) -> dict[str, dict[str, float]]:
    return {
        field: {"self": self_scale, "legoesm_nemo": 4.0, "nemo_twin": 1.0}
        for field in ROUND24.FIELDS
    }


def test_preregistered_terminal_verdicts_are_exhaustive_and_inventory_closed():
    assert ROUND24.registered_verdict(_verdict_rows(4.0)) == "CONFIRM_FULL"
    assert (ROUND24.registered_verdict(_verdict_rows(1.0))
            == "REFUTE_ROUNDING_EXPLANATION")
    rows = _verdict_rows(1.0)
    rows["ssh"]["self"] = 2.0
    assert ROUND24.registered_verdict(rows) == "PARTIAL_OR_INCONCLUSIVE"
    rows.pop("sst")
    with pytest.raises(Exception, match="inventory"):
        ROUND24.registered_verdict(rows)


def test_seed_statistics_uses_native_face_slices_and_active_words():
    delta_u = np.zeros((2, 4, 1))
    delta_v = np.zeros((3, 3, 1))
    delta_u[0, 1, 0] = -2.0
    delta_u[1, 3, 0] = 4.0
    delta_v[1, 0, 0] = 3.0
    masks = {
        "u": np.ones((2, 3, 1), dtype=bool),
        "v": np.ones((2, 3, 1), dtype=bool),
    }
    assert ROUND24.seed_statistics(delta_u, delta_v, masks) == {
        "u": {"max_abs": 4.0, "unequal_active_words": 2},
        "v": {"max_abs": 3.0, "unequal_active_words": 1},
    }


def test_regional_self_sse_localizes_a_western_only_pattern():
    control = np.zeros((180, 2, 10))
    perturbed = control.copy()
    perturbed[29:, :, :2] = 2.0
    rows = ROUND24._regional_self_sse(
        control, perturbed, np.ones((2, 10), dtype=bool))
    assert rows[0]["region"] == "western_boundary"
    assert rows[0]["share"] == 1.0
    assert all(row["share"] == 0.0 for row in rows[1:])


def test_preregistration_and_comparison_artifact_are_pinned():
    assert ROUND24.sha256(ROUND24.PREREGISTRATION) == \
        ROUND24.PREREGISTRATION_SHA256
    if ROUND24.ROUND11_ARRAYS.is_file():
        assert ROUND24.sha256(ROUND24.ROUND11_ARRAYS) == \
            ROUND24.ROUND11_ARRAYS_SHA256


def test_seed_loader_refuses_a_moved_artifact(tmp_path):
    seed = tmp_path / "seed.npz"
    with seed.open("wb") as stream:
        np.savez_compressed(stream, delta_u=np.zeros((1, 2, 1)),
                            delta_v=np.zeros((2, 1, 1)))
    manifest = {
        "seed_file": seed.name,
        "seed_sha256": ROUND24.sha256(seed),
        "statistics": ROUND24.REGISTERED_SEED,
    }
    (tmp_path / "seed_manifest.json").write_text(json.dumps(manifest))
    ROUND24._load_seed(tmp_path)
    seed.write_bytes(seed.read_bytes() + b"plant")
    with pytest.raises(Exception, match="moved"):
        ROUND24._load_seed(tmp_path)


def test_gpu_wrapper_requires_an_explicit_physical_device():
    wrapper = ROOT / "scripts/run/run_gyre_hires_round24_rounding_gpu.sh"
    result = subprocess.run([str(wrapper)], check=False, capture_output=True,
                            text=True)
    assert result.returncode == 64
    assert "--physical-gpu" in result.stderr
