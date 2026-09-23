"""Direct tests for the Round-156 developed stage-2 admission gate.

Every assertion is paired with a synthetic violation that must make the gate
refuse, so none of them can pass vacuously.  The gate is loaded by path
because it is a script, not an installed module.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

GATE_PATH = (Path(__file__).resolve().parents[3] / "scripts" / "validate"
             / "ocean_fidelity" / "testcases"
             / "nemo_testcase_l2_gyre_round156_developed_stage2_gate.py")


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("_r156_gate", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


JPI, JPJ, JPK = 6, 5, 4
NTSI, NTEI, NTSJ, NTEJ = 2, JPI - 1, 2, JPJ - 1


def _pair3(rng):
    return (rng.normal(size=(JPI, JPJ, JPK)), rng.normal(size=(JPI, JPJ, JPK)))


def _write(path: Path, gate, *, groups=None, commit="c" * 40,
           corrupt_final=False):
    """Write a synthetic record whose corrected velocity is self-consistent."""
    rng = np.random.default_rng(1156)
    umask = np.zeros((JPI, JPJ, JPK))
    umask[NTSI - 1:NTEI, NTSJ - 1:NTEJ, :JPK - 1] = 1.0
    vmask = umask.copy()
    raw_u, raw_v = _pair3(rng)
    zub = rng.normal(size=(NTEI - NTSI + 1, NTEJ - NTSJ + 1)) * 1e-3
    zvb = rng.normal(size=zub.shape) * 1e-3
    window = (slice(NTSI - 1, NTEI), slice(NTSJ - 1, NTEJ))
    final_u, final_v = raw_u.copy(), raw_v.copy()
    final_u[window] = raw_u[window] + zub[..., None] * umask[window]
    final_v[window] = raw_v[window] + zvb[..., None] * vmask[window]
    if corrupt_final:
        final_u[NTSI - 1, NTSJ - 1, 0] += 1.0
    payload = {
        "rhs_entry": _pair3(rng), "uu_vv_Kbb": _pair3(rng),
        "uu_vv_Kmm": _pair3(rng),
        "r3u_r3v_Kbb": (rng.normal(size=(JPI, JPJ)) * 1e-3,
                        rng.normal(size=(JPI, JPJ)) * 1e-3),
        "r3u_r3v_Kmm": (rng.normal(size=(JPI, JPJ)) * 1e-3,
                        rng.normal(size=(JPI, JPJ)) * 1e-3),
        "r3u_r3v_Kaa": (rng.normal(size=(JPI, JPJ)) * 1e-3,
                        rng.normal(size=(JPI, JPJ)) * 1e-3),
        "r3t_Kmm_r3f": (rng.normal(size=(JPI, JPJ)) * 1e-3,
                        rng.normal(size=(JPI, JPJ)) * 1e-3),
        "ssh_Kmm_ssh_Kaa": (rng.normal(size=(JPI, JPJ)),
                            rng.normal(size=(JPI, JPJ))),
        "umask_vmask": (umask, vmask),
        "rDt_r1_Dt": (np.float64(7200.0), np.float64(1.0 / 7200.0)),
        "flags_vec_linssh": (np.float64(1.0), np.float64(0.0)),
        "rhd_ww": _pair3(rng), "after_hpg": _pair3(rng),
        "after_vor": _pair3(rng), "after_adv": _pair3(rng),
        "uu_vv_Kaa_raw": (raw_u, raw_v), "zub_zvb": (zub, zvb),
        "uu_b_vv_b_Kaa": (rng.normal(size=(JPI, JPJ)),
                          rng.normal(size=(JPI, JPJ))),
        "uu_vv_Kaa_final": (final_u, final_v),
    }
    if groups is not None:
        payload = {name: payload[name] for name in groups}
    blob = bytearray(gate.MAGIC)
    blob += struct.pack("<16i", 1, 1081, 2, 1, 2, 3, 3, JPI, JPJ, JPK, 64,
                        len(gate.GROUPS), NTSI, NTEI, NTSJ, NTEJ)
    for name, (left, right) in payload.items():
        blob += name.ljust(16).encode("ascii")
        rank = left.ndim
        dims = tuple(left.shape) + (1,) * (3 - rank)
        blob += struct.pack("<4i", rank, *dims)
        for values in (left, right):
            blob += np.ascontiguousarray(
                np.asarray(values, dtype="<f8").T).tobytes()
    (path / gate.RECORD).write_bytes(bytes(blob))
    digest = gate.sha256(path / gate.RECORD)
    (path / f"{gate.RECORD}.stamp").write_text(
        f"{digest} {commit} {gate.RECORD}\n")
    return payload


def _restarts(root: Path, baseline: Path, *, move_last=False):
    for index, name in enumerate(gate_restarts()):
        (baseline / name).write_bytes(b"restart-%d" % index)
        (root / name).write_bytes(
            b"moved" if (move_last and index == 5) else b"restart-%d" % index)


def gate_restarts():
    return ("GYRE_OMIP_L2_P3_00000180_restart.nc",
            "GYRE_OMIP_L2_P3_00000360_restart.nc",
            "GYRE_OMIP_L2_P3_00000540_restart.nc",
            "GYRE_OMIP_L2_P3_00000720_restart.nc",
            "GYRE_OMIP_L2_P3_00000900_restart.nc",
            "GYRE_OMIP_L2_P3_00001080_restart.nc")


@pytest.fixture()
def record(tmp_path, gate):
    root, baseline = tmp_path / "run", tmp_path / "base"
    root.mkdir()
    baseline.mkdir()
    _write(root, gate)
    _restarts(root, baseline)
    return root, baseline


def test_round156_gate_admits_a_self_consistent_record(record, gate):
    root, baseline = record
    report = gate.audit(root, baseline, "c" * 40)
    assert report["status"] == "PASS"
    assert report["record"]["groups"] == list(gate.GROUPS)
    assert all(row["cells_unequal"] == 0
               for row in report["in_run_calibration"].values())
    assert all(row["identical"] for row in report["restarts"].values())
    # The assignment rebuild is REPORTED, not gated: it must be present and
    # must carry the flag that says so.
    assert report["assignment_rebuild_reported"]["u"]["gated"] is False


def test_round156_gate_refuses_a_broken_corrected_velocity(tmp_path, gate):
    root, baseline = tmp_path / "run", tmp_path / "base"
    root.mkdir()
    baseline.mkdir()
    # The recorded corrected velocity no longer equals the recorded raw
    # velocity plus the recorded correction on one interior wet face.
    _write(root, gate, corrupt_final=True)
    _restarts(root, baseline)
    with pytest.raises(gate.GateError, match="in-run calibration failed"):
        gate.audit(root, baseline, "c" * 40)


def test_round156_gate_refuses_a_moved_restart(tmp_path, gate):
    root, baseline = tmp_path / "run", tmp_path / "base"
    root.mkdir()
    baseline.mkdir()
    _write(root, gate)
    _restarts(root, baseline, move_last=True)
    with pytest.raises(gate.GateError, match="moved NEMO's own restarts"):
        gate.audit(root, baseline, "c" * 40)


def test_round156_gate_refuses_a_missing_group(tmp_path, gate):
    root, baseline = tmp_path / "run", tmp_path / "base"
    root.mkdir()
    baseline.mkdir()
    _write(root, gate, groups=[name for name in gate.GROUPS
                               if name != "after_vor"])
    _restarts(root, baseline)
    with pytest.raises(gate.GateError, match="groups differ from the registry"):
        gate.audit(root, baseline, "c" * 40)


def test_round156_gate_stage3_alignment_is_byte_exact_or_refuses(gate):
    rng = np.random.default_rng(7)
    u = rng.normal(size=(3, 4, 2))
    v = rng.normal(size=(3, 4, 2))
    stage2 = {"uu_vv_Kaa_final": (u, v)}
    transport = {"uu_Kmm": u.copy(), "vv_Kmm": v.copy()}
    rows = gate.compare_stage3_alignment(stage2, transport)
    assert rows["u"]["cells_unequal"] == 0 and rows["v"]["cells_unequal"] == 0

    moved = u.copy()
    moved[0, 0, 0] = np.nextafter(moved[0, 0, 0], np.inf)
    with pytest.raises(gate.GateError, match="not byte-identical"):
        gate.compare_stage3_alignment(
            {"uu_vv_Kaa_final": (moved, v)}, transport)


@pytest.mark.parametrize("plant", ["stamp", "truncation", "restart-byte",
                                   "operand-ulp"])
def test_round156_gate_plants_all_fire(record, gate, plant, capsys):
    root, baseline = record
    assert gate.main(["--root", str(root), "--baseline", str(baseline),
                      "--expect-commit", "c" * 40, "--plant", plant]) == 1
    assert f"STATUS PLANT-FIRED: {plant}" in capsys.readouterr().out
