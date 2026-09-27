import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (Path(__file__).parents[3] / "scripts" / "validate" /
          "ocean_fidelity" / "testcases" /
          "nemo_testcase_l2_gyre_round186_qsr_walk.py")
SPEC = importlib.util.spec_from_file_location("round186_qsr", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def _root(tmp_path):
    root = tmp_path / "record"
    root.mkdir()
    values = np.zeros(MOD.VALUE_COUNT, dtype="=f8")
    raw = MOD.MAGIC + struct.pack(f"={MOD.HEADER_INTS}i", *MOD.HEADER) + values.tobytes()
    record = root / "oracle_qsr_walk_kt00001080.bin"
    record.write_bytes(raw)
    digest = MOD.sha256(record)
    (root / "producer_commit.txt").write_text("abc\n")
    (root / "qsr_record.stamp").write_text(f"{digest} abc {record.name}\n")
    return root


def test_round186_qsr_record_admits_exact_replay(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setattr(MOD.subprocess, "check_output", lambda *a, **k: "tip\n")
    assert MOD.admit(root, "abc")["calibration"]["cells_unequal"] == 0


def test_round186_qsr_record_uses_compiled_no_halo_qsr_extent(tmp_path):
    root = _root(tmp_path)
    record = root / "oracle_qsr_walk_kt00001080.bin"
    parsed = MOD.read_record(record)
    assert parsed["qsr"].shape == (32, 22)
    assert parsed["qsr_bounds_1based"] == [3, 34, 3, 24]

    # A full-domain qsr payload is the original arithmetic defect and must not
    # be silently accepted as a different record layout.
    extra = np.zeros(MOD.JPI * MOD.JPJ - MOD.QSR_NI * MOD.QSR_NJ, dtype="=f8")
    record.write_bytes(record.read_bytes() + extra.tobytes())
    with pytest.raises(MOD.GateError, match="bytes, expected"):
        MOD.read_record(record)


def test_round186_qsr_header_tracks_step_1080_slot_rotation():
    assert MOD.HEADER[:5] == (1, 1080, 3, 2, 1)


def test_round186_qsr_record_ulp_plant_fires(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setattr(MOD.subprocess, "check_output", lambda *a, **k: "tip\n")
    with pytest.raises(MOD.GateError, match="STATUS PLANT-FIRED"):
        MOD.admit(root, "abc", "actual-increment-ulp")


def test_round188_qsr_walk_score_is_bitwise_and_masked():
    reference = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    candidate = reference.copy()
    candidate[0, 0] = np.nextafter(candidate[0, 0], np.inf)
    mask = np.array([[False, True], [True, True]])
    assert MOD._score(reference, candidate, mask)["classification"] == "BIT"
    mask[0, 0] = True
    row = MOD._score(reference, candidate, mask)
    assert row["classification"] == "NON-BIT"
    assert row["cells_unequal"] == 1


def test_round189_removed_fraction_is_directional_and_fail_closed():
    assert MOD._removed_fraction(4.0, 1.0) == pytest.approx(0.75)
    assert MOD._removed_fraction(4.0, 5.0) == pytest.approx(-0.25)
    with pytest.raises(MOD.GateError, match="zero baseline"):
        MOD._removed_fraction(0.0, 0.0)


def test_round189_qsr_override_requires_process_trace():
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card("GYRE-zco")
    hooks = _NEMOWSRK3TestHooks(
        stage3_qsr_stretch_override=np.ones((22, 32), dtype=np.float64))
    with pytest.raises(ValueError, match="requires tracer_process_trace"):
        LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord,
            card.recipe.model_config, _nemo_ws_test_hooks=hooks)


def test_round190_qsr_association_uses_live_qco_weights():
    bsbc = np.array([[[2.0, 3.0]]], dtype=np.float64)
    qmm = np.array([[1.25]], dtype=np.float64)
    qaa = np.array([[0.75]], dtype=np.float64)
    rate = np.array([[[2.0e-8, -3.0e-8]]], dtype=np.float64)
    expected = ((bsbc * qaa[..., None]
                 + 14400.0 * qmm[..., None] * rate)
                / qaa[..., None] - bsbc)
    np.testing.assert_array_equal(
        MOD._associate_qsr(bsbc, qmm, qaa, rate), expected)
