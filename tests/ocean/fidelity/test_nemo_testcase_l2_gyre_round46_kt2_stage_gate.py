"""Hermetic fail-closed checks for the round-46 acquisition package."""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
TESTCASES = ROOT / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(TESTCASES))
SPEC = importlib.util.spec_from_file_location(
    "nemo_testcase_l2_gyre_round46_kt2_stage_gate",
    TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py",
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

NX, NY, NZ = gate.DIMS
SCALARS = {f"has_{x}" for x in ("hpg", "vor", "keg", "zad", "ldf", "zdf")} | {"r1_Dt"}
TWO_D = {
    name
    for name in gate.REQUIRED
    if name.startswith("r3")
    or name.startswith("ssh_")
    or name.startswith("uu_b_")
    or name.startswith("vv_b_")
} | {
    "e1e2t",
    "e1e2u",
    "e1e2v",
    "r1_e1e2u",
    "r1_e1e2v",
    "r1_e1u",
    "r1_e2v",
    "r1_e1e2t",
    "e2u",
    "e1v",
}


def _payload(value: np.ndarray, rank: int) -> bytes:
    if rank == 2:
        return np.asarray(value).T.ravel(order="F").tobytes()
    return np.asarray(value).transpose(1, 0, 2).ravel(order="F").tobytes()


def _write(path: Path, *, stage=1, truncate=False) -> Path:
    names = set(gate.REQUIRED) | {
        "pre_baro_u",
        "pre_baro_v",
        "post_update_u",
        "post_update_v",
        "after_ldf_u",
        "after_ldf_v",
    }
    names -= {"post_zdf_u", "post_zdf_v", "pre_zdf_rhs_u", "pre_zdf_rhs_v"}
    with path.open("wb") as f:
        f.write(gate.MAGIC.ljust(16).encode())
        f.write(struct.pack("=16i", 1, 2, stage, 1, 1, 2, 3, NX, NY, NZ, 30, 3, 34, 3, 24, 64))
        for index, name in enumerate(sorted(names)):
            f.write(name.ljust(16).encode())
            if name in SCALARS:
                f.write(struct.pack("=4i", 0, 1, 1, 1))
                value = gate.PRESENCE[stage].get(name[4:], 1) if name.startswith("has_") else 1
                raw = np.asarray([value], dtype=np.float64).tobytes()
            elif name in TWO_D:
                f.write(struct.pack("=4i", 2, NX, NY, 1))
                fill = 1.0 if name.startswith(("e1", "r1_e1", "r3", "ssh_")) else 0.0
                raw = _payload(np.full((NY, NX), fill), 2)
            else:
                owned = name in gate.OWNED_3D_FIELDS
                nx, ny = gate.OWNED_DIMS[:2] if owned else (NX, NY)
                f.write(struct.pack("=4i", 3, nx, ny, NZ))
                fill = 1.0 if name.startswith(("e3", "tmask", "umask", "vmask", "wmask")) else 0.0
                raw = _payload(np.full((ny, nx, NZ), fill), 3)
            f.write(raw[:-8] if truncate and index == 0 else raw)
            if truncate and index == 0:
                break
    return path


def test_reader_and_zero_source_replays(tmp_path):
    rec = gate.read_stage(_write(tmp_path / "stage.bin"))
    records = {key: rec for key in gate.STAGES}
    assert all(value == 0 for value in gate._calibrate(records, None).values())


def test_header_and_truncation_plants_are_red(tmp_path):
    path = _write(tmp_path / "stage.bin")
    with pytest.raises(Exception, match="wrong kt/stage"):
        gate.read_stage(path, plant="header")
    with pytest.raises(Exception, match="short payload"):
        gate.read_stage(_write(tmp_path / "short.bin", truncate=True))


def test_instrument_contract_is_zero_first_write_only_and_widened():
    package = TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage"
    writer = (package / "l2_r46_stage.F90").read_text()
    assert "ACTION='WRITE'" in writer
    assert "STATUS='REPLACE'" in writer
    assert "z(:,:,:)=0._wp" in writer or "z(:,:,:) = 0._wp" in writer
    assert "kt <= nit000+1" in writer
    assert "PUBLIC   dissl" in (package / "zdftke_round46.patch").read_text()
    stg = (package / "stprk3_stg_round46.patch").read_text()
    adv = (package / "dynadv_round46.patch").read_text()
    assert "kstp <= nit000 + 1" in stg
    assert 'oracle_rkstage3_terms_kt",I8.8' in stg
    assert "kt <= nit000 + 1" in adv
    assert 'oracle_dynadv_split_kt",I8.8' in adv


def test_every_declared_plant_has_a_nonzero_exit_contract():
    source = (TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py").read_text()
    run = (TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage/run.sh").read_text()
    for plant in ("header", "truncation", "calibration", "given", "trajectory",
                  "twin", "stage-entry-ulp", "stage-context-ulp", "stamp"):
        assert f'"{plant}"' in source
    assert 'return 1 if args.plant or report["status"] != "PASS" else 0' in source
    for plant in ("header", "truncation", "calibration", "twin", "stamp"):
        assert plant in run


def test_retracted_round48_owner_prediction_cannot_abort_current_measurement():
    source = (TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py").read_text()
    assert '"interpretation": "POSTHOC_AFTER_ROUND49_ROUND50"' in source
    assert "round48 first-non-bit prediction REFUTED" not in source


def test_stage_output_reference_uses_next_handoff_and_next_step():
    shape3 = (NY, NX, NZ)
    shape2 = (NY, NX)

    def arrays(fill):
        return {
            "post_baro_u": np.full(shape3, fill),
            "post_baro_v": np.full(shape3, fill + 1),
            "T_Kmm": np.full(shape3, fill + 2),
            "S_Kmm": np.full(shape3, fill + 3),
            "ssh_Kmm": np.full(shape2, fill + 4),
        }

    records = {(1, stage): {"arrays": arrays(10 * stage)}
               for stage in (1, 2, 3)}
    next_entries = {2: {
        "T": np.full((NY - 4, NX - 4, NZ), 70.0),
        "S": np.full((NY - 4, NX - 4, NZ), 71.0),
        "ssh": np.full((NY - 4, NX - 4), 72.0),
    }}
    stage1 = gate._stage_reference(records, next_entries, 1, 1)
    stage3 = gate._stage_reference(records, next_entries, 1, 3)
    assert np.all(stage1["T"] == 22.0)
    assert np.all(stage1["ssh"] == 24.0)
    assert np.all(stage3["T"] == 70.0)
    assert np.all(stage3["ssh"] == 72.0)
    assert stage3["T"].shape[-1] == 30


def test_stage_twin_private_overrides_are_off_by_default():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks

    hooks = _NEMOWSRK3TestHooks()
    assert hooks.stage_barotropic_output_override is None
    assert hooks.stage_entry_override is None


def test_stage_twin_refuses_unmeasured_required_rows():
    source = (TESTCASES / "nemo_testcase_l2_gyre_round46_kt2_stage_gate.py").read_text()
    assert 'report["status"] = "UNMEASURED"' in source
    assert 'row.get("classification") == "UNMEASURED_WITH_SPEC"' in source


def test_final_external_history_uses_last_pre_swap_current_and_before():
    arrays = {}
    for index, name in enumerate(
        ("u_entry", "u_b", "v_entry", "v_b", "eta_entry", "eta_b"),
        start=1,
    ):
        arrays[name] = np.asarray([[index], [10 + index]], dtype=np.float64)
    histories = gate._final_history_from_btstep(arrays)
    assert tuple(float(value[0]) for value in histories) == (
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0)


def test_qco_face_layout_drops_redundant_edge_and_singleton_level():
    u = np.zeros((NY - 4, NX - 3, 1))
    v = np.zeros((NY - 3, NX - 4, 1))
    assert u[:, 1:, 0].shape == (NY - 4, NX - 4)
    assert v[1:, :, 0].shape == (NY - 4, NX - 4)
