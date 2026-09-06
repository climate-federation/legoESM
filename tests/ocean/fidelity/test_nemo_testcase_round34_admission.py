"""Non-vacuity tests for the CARD-GENERAL consumed-field admission gate.

Round 34 generalised ``nemo_testcase_l2_gyre_round21_admission.py`` from the
one GYRE acquisition it was written for to any card: dimensions come from each
record's own header, the record kind is its own 16-byte magic, and the
inventory is discovered by globbing the source run.  These arms drive it on
LOCK_EXCHANGE's real geometry (134x7x21) and OVERFLOW's (206x7x101) using
synthetic records in the instrument's exact binary layout, so nothing here
needs the acquisition to have run.

Every arm that asserts a PASS has a paired arm that must FAIL.
"""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest

PATH = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/"
    "nemo_testcase_l2_gyre_round21_admission.py"
)
SPEC = importlib.util.spec_from_file_location("consumed_field_admission", PATH)
assert SPEC and SPEC.loader
admission = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(admission)

LOCK = (134, 7, 21)
OVERFLOW = (206, 7, 101)


def _transport(dims, values: np.ndarray) -> bytes:
    nx, ny, nz = dims
    return b"NEMO_L1_TRANSP_1" + struct.pack(
        "=8i", 1, 1, 1, 1, nx, ny, nz, 64) + values.tobytes()


def _stage(dims, values: np.ndarray) -> bytes:
    nx, ny, nz = dims
    return b"NEMO_L1_STAGE_1 " + struct.pack(
        "=9i", 1, 1, 3, 3, nx, ny, nz, 2, 64) + values.tobytes()


def _btfrm(dims, values: np.ndarray) -> bytes:
    nx, ny, _ = dims
    return b"NEMO_L1_BTFRM_1 " + struct.pack(
        "=6i", 1, 1, 3, nx, ny, 64) + values.tobytes()


def _flat(dims, i, j, k):
    nx, ny, _ = dims
    return i + nx * j + nx * ny * k


@pytest.mark.parametrize("dims", [LOCK, OVERFLOW])
def test_dims_come_from_the_record_header_not_an_argument(tmp_path, dims):
    nx, ny, nz = dims
    values = np.zeros(3 * nx * ny * nz, dtype=np.float64)
    path = tmp_path / "oracle_transport_kt00000001_s1.bin"
    path.write_bytes(_transport(dims, values))
    magic, header, hx, hy, hz, layout = admission.read_header(
        path.read_bytes(), path)
    assert magic == "NEMO_L1_TRANSP_1"
    assert (hx, hy, hz) == dims
    assert [name for name, _, _, _ in layout] == ["zFu", "zFv", "zFw"]


@pytest.mark.parametrize("dims", [LOCK, OVERFLOW])
def test_a_halo_difference_is_admitted_and_an_owned_one_is_not(tmp_path, dims):
    nx, ny, nz = dims
    n3 = nx * ny * nz
    values = np.zeros(3 * n3, dtype=np.float64)
    baseline = tmp_path / "oracle_transport_kt00000001_s1.bin"
    baseline.write_bytes(_transport(dims, values))
    candidate = tmp_path / "candidate.bin"

    # zFu is a DEFINED slot, so only the halo rule can admit it.  (1, 0, 0) is
    # inside the 2-cell halo on both axes; (2, 2, 0) is the first owned cell.
    halo = values.copy()
    halo[_flat(dims, 1, 0, 0)] = 1.0
    candidate.write_bytes(_transport(dims, halo))
    report = admission.compare_record(baseline, candidate, [True])
    assert report["consumed_equal"]
    assert report["changed_fields"][0]["changed_in_owned_cells"] == 0
    assert report["admitted_differences"][0]["reason"] == "halo"
    assert report["admitted_differences"][0]["candidate_value"] == 1.0

    owned = values.copy()
    owned[_flat(dims, 2, 2, 0)] = 1.0
    candidate.write_bytes(_transport(dims, owned))
    report = admission.compare_record(baseline, candidate, [True])
    assert not report["consumed_equal"]
    assert report["changed_fields"][0]["changed_in_owned_cells"] == 1
    # A violation is never listed as admitted.
    assert report["admitted_differences"] == []


def test_the_j_halo_alone_is_enough_to_admit(tmp_path):
    """The tanks are 7 cells wide, so j = 0 and j = 6 are BOTH halo.

    OVERFLOW's real differences sit at j = 6 and j = 1, which the i-only rule
    would have called owned.  This arm fails if the j test is dropped.
    """
    dims = OVERFLOW
    nx, ny, nz = dims
    values = np.zeros(3 * nx * ny * nz, dtype=np.float64)
    baseline = tmp_path / "oracle_transport_kt00000001_s1.bin"
    baseline.write_bytes(_transport(dims, values))
    candidate = tmp_path / "candidate.bin"
    changed = values.copy()
    changed[_flat(dims, 100, 6, 11)] = 4.2e-318
    changed[_flat(dims, 100, 1, 16)] = 4.0e-318
    candidate.write_bytes(_transport(dims, changed))
    report = admission.compare_record(baseline, candidate, [True])
    assert report["consumed_equal"]
    assert report["changed_fields"][0]["changed_in_owned_cells"] == 0
    assert len(report["admitted_differences"]) == 2


def test_a_schema_that_does_not_fit_the_file_raises(tmp_path):
    """A short or long payload is a GATE ERROR, never a comparison."""
    dims = LOCK
    nx, ny, nz = dims
    short = np.zeros(3 * nx * ny * nz - 1, dtype=np.float64)
    path = tmp_path / "oracle_transport_kt00000001_s1.bin"
    path.write_bytes(_transport(dims, short))
    with pytest.raises(admission.AdmissionError, match="does not fit"):
        admission.read_header(path.read_bytes(), path)


def test_an_unregistered_magic_raises(tmp_path):
    path = tmp_path / "oracle_mystery_kt00000001.bin"
    path.write_bytes(b"NEMO_L9_NOSUCH_1" + struct.pack("=8i", *([1] * 8)))
    with pytest.raises(admission.AdmissionError, match="unregistered record"):
        admission.read_header(path.read_bytes(), path)


def test_every_schema_declares_a_reason_for_every_undefined_slot():
    """An undefined slot with no registered reason is a silent admission."""
    for magic, (_, _, fields) in admission.SCHEMAS.items():
        for name, _, _, defined in fields(1, 1, 1):
            if not defined:
                assert (magic, name) in admission.UNDEFINED_REASON, (
                    f"{magic}:{name} is admitted with no reason")


def _tank_run(root, dims, *, restart=b"restart", extra=None,
              mutate=None, plant=False, allowed_new=None):
    """Build a two-directory synthetic acquisition under ``root`` and admit it."""
    nx, ny, nz = dims
    n3, n2 = nx * ny * nz, nx * ny
    base = root / "source"
    cand = root / "instrumented"
    base.mkdir(parents=True)
    cand.mkdir(parents=True)
    rng = np.random.default_rng(0)
    records = {
        "oracle_transport_kt00000001_s1.bin": _transport(
            dims, rng.standard_normal(3 * n3)),
        "oracle_stage_kt00000001_s3.bin": _stage(
            dims, rng.standard_normal(4 * n3 + n2)),
        "oracle_bt_frames_kt00000001.bin": _btfrm(
            dims, rng.standard_normal(4 * n2)),
    }
    for name, blob in records.items():
        (base / name).write_bytes(blob)
        (cand / name).write_bytes(mutate(name, blob) if mutate else blob)
    (base / "restart.nc").write_bytes(restart)
    (cand / "restart.nc").write_bytes(restart)
    if extra:
        (cand / extra).write_bytes(b"new")
    if allowed_new is None:
        allowed_new = {extra} if extra else set()
    return admission.run(base, cand, twin=None, identical=("restart.nc",),
                         allowed_new=allowed_new, plant_consumed=plant)


def test_a_clean_acquisition_passes(tmp_path):
    report = _tank_run(tmp_path, LOCK, extra="oracle_zdf_matrix_kt00000001.bin")
    assert report["verdict"] == "PASS", report["violations"]
    assert report["byte_identical_records"] == 3
    assert report["admitted_difference_count"] == 0


def test_an_undeclared_new_record_is_a_violation(tmp_path):
    report = _tank_run(tmp_path, LOCK, extra="oracle_surprise.bin",
                       allowed_new=set())
    assert report["verdict"] == "FAIL"
    assert any("unexpected candidate records" in v
               for v in report["violations"])


def test_a_changed_restart_fails_even_with_identical_records(tmp_path):
    root = tmp_path / "b"
    report = _tank_run(root, LOCK)
    assert report["verdict"] == "PASS"
    (root / "instrumented" / "restart.nc").write_bytes(b"moved")
    report = admission.run(root / "source", root / "instrumented", twin=None,
                           identical=("restart.nc",), allowed_new=set())
    assert report["verdict"] == "FAIL"
    assert "restart.nc" in report["violations"]


def _halo_only(name, blob):
    """Perturb ONE halo cell of zFu, the way the real tanks differ."""
    if not name.startswith("oracle_transport"):
        return blob
    head, payload = blob[:48], np.frombuffer(blob[48:], np.float64).copy()
    payload[_flat(LOCK, 1, 0, 0)] = 6.9e-310
    return head + payload.tobytes()


def test_the_plant_flips_one_owned_bit_and_turns_the_gate_red(tmp_path):
    """The plant must fail on a run that otherwise passes."""
    clean = _tank_run(tmp_path / "clean", LOCK, mutate=_halo_only)
    assert clean["verdict"] == "PASS"
    assert clean["admitted_difference_count"] == 1

    planted = _tank_run(tmp_path / "planted", LOCK, mutate=_halo_only,
                        plant=True)
    assert planted["plant_applied"] is True
    assert planted["verdict"] == "FAIL"
    assert planted["violations"] == ["oracle_transport_kt00000001_s1.bin"]


def test_a_missing_inherited_record_is_a_violation(tmp_path):
    root = tmp_path / "m"
    _tank_run(root, LOCK)
    (root / "instrumented" / "oracle_bt_frames_kt00000001.bin").unlink()
    report = admission.run(root / "source", root / "instrumented", twin=None,
                           identical=("restart.nc",), allowed_new=set())
    assert report["verdict"] == "FAIL"
    assert any("missing inherited records" in v for v in report["violations"])


def test_allowed_new_that_never_appeared_is_a_violation(tmp_path):
    """A declared addition the run never wrote means the arm did not run."""
    root = tmp_path / "n"
    _tank_run(root, LOCK)
    report = admission.run(root / "source", root / "instrumented", twin=None,
                           identical=("restart.nc",),
                           allowed_new={"oracle_zdf_matrix_kt00000001.bin"})
    assert report["verdict"] == "FAIL"
    assert any("never wrote" in v for v in report["violations"])
