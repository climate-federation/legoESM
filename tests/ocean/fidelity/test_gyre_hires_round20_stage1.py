from __future__ import annotations

import importlib.util
import inspect
import struct
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "scripts/validate/ocean_fidelity/testcases"
SCRIPT = SCRIPT_DIR / "nemo_testcase_l2_gyre_hires_round20_stage1.py"
ACQUISITION_DIR = (
    SCRIPT_DIR / "nemo_testcase_l2_gyre_hires_round20_stage1")


def _load_module():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("gyre_round20_stage1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_envelope_score_and_one_ulp_plant_fire():
    module = _load_module()
    scalar = np.ones((2, 2), dtype=np.float64)
    vector = scalar.copy()
    candidate = scalar.copy()
    mask = np.ones_like(scalar, dtype=bool)

    exact = module.envelope_score(candidate, scalar, vector, mask)
    assert exact["legoesm_minus_scalar"]["bit_exact"]
    assert not exact["clears_four_times_twin"]

    planted = module._plant_one_ulp(candidate, mask)
    moved = module.envelope_score(planted, scalar, vector, mask)
    assert moved["legoesm_minus_scalar"]["unequal_words"] == 1
    assert moved["clears_four_times_twin"]


def test_envelope_uses_registered_four_times_maximum():
    module = _load_module()
    scalar = np.zeros(2, dtype=np.float64)
    vector = np.array([1.0, 0.0], dtype=np.float64)
    mask = np.ones(2, dtype=bool)
    below = module.envelope_score(
        np.array([3.0, 0.0]), scalar, vector, mask)
    at_bar = module.envelope_score(
        np.array([4.0, 0.0]), scalar, vector, mask)
    assert not below["clears_four_times_twin"]
    assert at_bar["clears_four_times_twin"]
    assert at_bar["candidate_to_twin_max_ratio"] == 4.0


def test_first_clearing_stage_respects_stage_and_face_order():
    module = _load_module()

    def row(clears):
        return {"clears_four_times_twin": clears}

    stages = [
        {"stage": 1, "rows": {"u": row(False), "v": row(True)}},
        {"stage": 2, "rows": {"u": row(True), "v": row(True)}},
    ]
    result = module.first_clearing_stage(stages)
    assert result is not None
    assert (result["stage"], result["field"]) == (1, "v")


def test_first_clearing_source_boundary_respects_executed_order():
    module = _load_module()

    def row(clears):
        return {"clears_four_times_twin": clears}

    boundaries = [
        {
            "boundary": "after_hpg",
            "term": "hydrostatic_pressure_gradient",
            "rows": {"u": row(False), "v": row(False)},
        },
        {
            "boundary": "after_ldf",
            "term": "lateral_diffusion",
            "rows": {"u": row(False), "v": row(True)},
        },
        {
            "boundary": "after_vor",
            "term": "total_vorticity_coriolis",
            "rows": {"u": row(True), "v": row(True)},
        },
    ]
    result = module.first_clearing_source_boundary(boundaries)
    assert result is not None
    assert (result["boundary"], result["term"], result["field"]) == (
        "after_ldf", "lateral_diffusion", "v")


def test_active_level_mapping_drops_only_nemo_closed_jpk(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "NATIVE_DIMS", (6, 5, 4))
    monkeypatch.setattr(module, "ACTIVE_LEVELS", 3)
    field = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    active = module._active_levels(field)
    assert active.shape == (2, 3, 3)
    np.testing.assert_array_equal(active, field[..., :3])
    with pytest.raises(Exception, match="unexpected shape"):
        module._active_levels(field[..., :3])


def test_completed_rhs_reader_rejects_bad_header(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "NATIVE_DIMS", (3, 2, 2))
    count = 3 * 2 * 2
    fields = np.arange(2 * count, dtype=np.float64)
    path = tmp_path / module.RHS_RECORD
    path.write_bytes(
        b"NEMO_L1_RHS___1".ljust(16)
        + struct.pack("=7i", 1, module.STEP, 3, 3, 2, 2, 64)
        + fields.tobytes()
    )
    result = module.read_completed_rhs(path)
    assert result["header"]["kt"] == module.STEP

    changed = bytearray(path.read_bytes())
    changed[16:20] = struct.pack("=i", 2)
    path.write_bytes(changed)
    with pytest.raises(Exception, match="bad header"):
        module.read_completed_rhs(path)


def test_bt_reader_rejects_truncation(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "NATIVE_DIMS", (6, 5, 2))
    count = 6 * 5
    path = tmp_path / module.BT_RECORD
    payload = (
        b"NEMO_L1_BTFRM_1".ljust(16)
        + struct.pack("=6i", 1, module.STEP, 3, 6, 5, 64)
        + np.arange(4 * count, dtype=np.float64).tobytes()
    )
    path.write_bytes(payload)
    result = module.read_bt_handoff(path)
    assert result["uu_b"].shape == (1, 2)
    path.write_bytes(payload[:-8])
    with pytest.raises(Exception, match="malformed"):
        module.read_bt_handoff(path)


def test_rhs_family_reader_enforces_header_layout_and_source_order(
    tmp_path, monkeypatch,
):
    module = _load_module()
    monkeypatch.setattr(module, "NATIVE_DIMS", (6, 5, 2))
    monkeypatch.setattr(module, "ACTIVE_LEVELS", 1)
    count = 6 * 5 * 2
    payload = np.arange(10 * count, dtype=np.float64)
    path = tmp_path / module.RHS_FAMILY_RECORD
    path.write_bytes(
        b"NEMO_L2_R20RHS_1"
        + struct.pack("=8i", 1, module.STEP, 1, 3, 6, 5, 2, 64)
        + struct.pack("=10i", *((count,) * 10))
        + payload.tobytes()
    )
    result = module.read_stage1_rhs_family(path)
    assert result["after_hpg_u"].shape == (1, 2, 2)
    first = payload[:count].reshape((6, 5, 2), order="F")[
        2:-2, 2:-2].transpose(1, 0, 2)
    np.testing.assert_array_equal(result["after_hpg_u"], first)
    last = payload[-count:].reshape((6, 5, 2), order="F")[
        2:-2, 2:-2].transpose(1, 0, 2)
    np.testing.assert_array_equal(result["after_adv_v"], last)

    changed = bytearray(path.read_bytes())
    changed[48:52] = struct.pack("=i", count - 1)
    path.write_bytes(changed)
    with pytest.raises(Exception, match="array-size contract"):
        module.read_stage1_rhs_family(path)


def test_trace_source_boundary_maps_owned_faces():
    module = _load_module()

    class Field:
        def __init__(self, data):
            self.data = data

    operands = {
        "after_hpg_u": Field(np.arange(2 * 4 * 2).reshape(2, 4, 2)),
        "after_hpg_v": Field(np.arange(3 * 3 * 2).reshape(3, 3, 2)),
    }
    mapped = module._trace_source_boundary(operands, "after_hpg")
    assert mapped["u"].shape == (2, 3, 2)
    assert mapped["v"].shape == (2, 3, 2)


def test_run_uses_source_order_arm_only_for_cumulative_family():
    """The acquired NEMO cumulatives must not use ordinary legoESM order."""
    module = _load_module()
    source = inspect.getsource(module.run)
    assert '"stage1-source-order", None' in source
    assert "live_operands = source_order_trace.operator_operands[0]" in source
    assert "live_operands = trace.operator_operands[0]" not in source
    assert "source_order_trace.state_after, ordinary" in source


def test_operator_acquisition_is_single_rank_passive_and_creates_parent():
    run = (ACQUISITION_DIR / "run.sh").read_text(encoding="utf-8")
    assert "RANKS must be 1" in run
    assert "mkdir -p \"$(dirname -- \"$EVIDENCE\")\"" in run
    assert "./makenemo -r \"$SOURCE_NAME\" -n \"$TARGET_NAME\" -d 'OCE'" in run
    assert "export GYRE_R20_WRITE=$enabled" in run
    assert "run_nemo control 0" in run
    assert "run_nemo instrumented 1" in run
    assert "REFUSE_WRITER_MOVED_TRAJECTORY" in run
    assert "mpirun -np 1" in run
    assert "rg " not in run


def test_writer_patch_has_five_source_ordered_cumulative_frames():
    patch = (ACQUISITION_DIR / "stp2d_round20.patch").read_text(
        encoding="utf-8")
    assert patch.count(
        "WRITE(r20_unit) uu(:,:,:,Krhs), vv(:,:,:,Krhs)") == 5
    # Zero-context insertions are safe here because run.sh pins the exact
    # engine-source hash and the preflight separately compiles the result.
    assert "4bfe31a332c3782b2701db0e6cff689876dc5cdd3af91b546ac46369e2e3dc8b" in (
        ACQUISITION_DIR / "run.sh").read_text(encoding="utf-8")
    # The patch inserts immediately after the five pinned source call lines;
    # prove those line-number boundaries remain strictly ordered.
    headers = ["@@ -128,0", "@@ -131,0", "@@ -146,0", "@@ -163,0",
               "@@ -165,0"]
    positions = [patch.find(header) for header in headers]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)
