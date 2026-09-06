"""Round 33, item 3: the TANK Rule-12 reader, validated before its record exists.

The acquisition that produces the two tanks' ``dyn_zdf`` records has to be run
by hand (``nemo_testcase_l1_tanks_round33_zdf/run.sh``), so the reader would
otherwise be untested code the first time it is ever asked a question.  Rule:
validate the instrument BEFORE quoting its number.

Three things are under test:

* ``record_geometry``, the layout conversion the discharge divides by, against
  an INDEPENDENT construction of the same arrays from each tank's own mesh and
  ``domain.F90:140-147``/``:159``, plus the two open association rows, which
  must carry NUMBERS -- and the reciprocal one must be NONZERO, or it is
  decorative;
* the round-33 add-on patch itself: WRITE-only, stacking on the round-29 one,
  and every array label exactly sixteen characters, because a short label
  desynchronises the self-describing stream and turns the whole record to
  garbage;
* the whole gate END TO END on a SYNTHETIC record written in the exact binary
  layout the instrument emits, built so the operator must reproduce the tank's
  own stage-3 velocity exactly, with two plants that must turn it red;
* the fail-closed path, which must name the acquisition command rather than
  report a status it could not measure.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
GATES = REPO / "scripts/validate/ocean_fidelity/testcases"
LOCK_TWIN = Path("/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_kt1_10")
OVERFLOW_TWIN = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/phase3/overflow_kt1_10")

pytestmark = pytest.mark.skipif(
    not (LOCK_TWIN / "mesh_mask.nc").is_file(),
    reason="the L1 tank oracle roots are not on this machine")


def _load(name: str):
    if str(GATES) not in sys.path:
        sys.path.insert(0, str(GATES))
    spec = importlib.util.spec_from_file_location(name, GATES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load("nemo_testcase_l1_tanks_round33_zdf_rule12")
# NOT _load again: re-executing a module the gate already imported would make
# a SECOND GateError class, and every ``pytest.raises`` below would miss.
SWEEP = sys.modules["nemo_testcase_phase3_stage_sweep_gate"]
R29 = sys.modules["nemo_testcase_l2_gyre_round29_zdf_matrix"]


# --------------------------------------------------------------------------
# the geometry the discharge divides by
# --------------------------------------------------------------------------

def _nemo_reference_geometry(card: str):
    """NEMO's own e3u_0, umask, hu_0, r1_hu_0 for one tank, in FORTRAN layout.

    This is the fixture, not the thing under test: it stands in for what the
    round-33 add-on patch dumps from inside ``dyn_zdf``.  ``e3u_0`` comes from
    the card's own ``mesh_mask.nc`` where the mesh carries it (OVERFLOW,
    ``key_vco_3d``) and from ``e3t_1d`` where it does not (LOCK,
    ``key_vco_1d``, where ``domzgr_substitute.h90:89`` DEFINES
    ``e3u_0(i,j,k)`` to be ``e3t_1d(k)`` -- so this is NEMO's definition, not
    an approximation).  ``hu_0`` and ``r1_hu_0`` follow
    ``domain.F90:140-147`` and ``:159``.
    """
    import netCDF4

    spec = GATE.CARDS[card]
    nx, ny, nz = SWEEP.DIMS[spec["case"]]
    with netCDF4.Dataset(str(spec["twin"] / "mesh_mask.nc")) as data:
        umask = np.asarray(
            data.variables["umask"][:]).squeeze().transpose(1, 2, 0)
        if "e3u_0" in data.variables:
            e3u_0 = np.asarray(
                data.variables["e3u_0"][:]).squeeze().transpose(1, 2, 0)
        else:
            ladder = np.asarray(data.variables["e3t_1d"][:]).squeeze()
            e3u_0 = np.broadcast_to(ladder, umask.shape).copy()
    e3u_0 = np.asarray(e3u_0, dtype=np.float64)
    umask = np.asarray(umask, dtype=np.float64)
    hu_0 = np.sum(e3u_0 * umask, axis=-1)
    ssumask = (umask > 0).any(axis=-1).astype(np.float64)
    r1_hu_0 = ssumask / (hu_0 + 1.0 - ssumask)     # domain.F90:159, verbatim

    def _to_f(scored, rank):
        if rank == 3:
            out = np.zeros((nx, ny, nz), dtype=np.float64)
            out[2:-2, 2:-2, :] = scored.transpose(1, 0, 2)
        else:
            out = np.zeros((nx, ny), dtype=np.float64)
            out[2:-2, 2:-2] = scored.T
        return out

    return {
        "e3u_0": _to_f(e3u_0, 3), "umask": _to_f(umask, 3),
        "hu_0": _to_f(hu_0, 2), "r1_hu_0": _to_f(r1_hu_0, 2),
        "scored": {"e3u_0": e3u_0, "umask": umask, "hu_0": hu_0,
                   "r1_hu_0": r1_hu_0},
    }


@pytest.mark.parametrize("card", ["LOCK_EXCHANGE", "OVERFLOW"])
def test_record_geometry_round_trips_nemos_own_arrays(tmp_path, card,
                                                      monkeypatch):
    """The layout conversion must return exactly what the record carried."""
    root = _synthetic_root(tmp_path, card)
    monkeypatch.setitem(GATE.CARDS[card], "root", root)
    rec = R29.read_zdf_matrix(root / GATE.ZDF_MATRIX_RECORD)
    geom = GATE.record_geometry(rec, GATE.CARDS[card]["case"])
    want = _nemo_reference_geometry(card)["scored"]
    assert np.array_equal(geom["umask"], want["umask"])
    assert np.array_equal(geom["h_face"], want["e3u_0"] * want["umask"])
    assert np.array_equal(geom["depth"], want["hu_0"])


@pytest.mark.parametrize("card", ["LOCK_EXCHANGE", "OVERFLOW"])
def test_the_two_open_association_rows_are_measured_not_argued(tmp_path, card,
                                                               monkeypatch):
    """Both must be REPORTED with a number, and the right quantity measured.

    RETRACTED while writing this: the first version asserted the reciprocal
    gap is NONZERO, on the argument that ``hu_0 + 1 - 1`` is not ``hu_0`` in
    floating point.  Measured on both tanks it is EXACTLY zero -- their column
    depths are small exactly-representable values, so ``domain.F90:159``
    returns the correctly rounded reciprocal.  The claim was wrong and the
    assertion is corrected rather than deleted.

    The row that survives is the one that matters: NEMO forms
    ``SUM(e3u_0*uu(Kaa)) * r1_hu_0`` and the operator forms the same sum
    DIVIDED by ``hu_0``, and ``x/h`` is not ``x*(1/h)`` even when ``1/h`` is
    correctly rounded.  That is measured on NEMO's own column transport.
    """
    root = _synthetic_root(tmp_path, card)
    monkeypatch.setitem(GATE.CARDS[card], "root", root)
    rec = R29.read_zdf_matrix(root / GATE.ZDF_MATRIX_RECORD)
    rows = GATE.record_geometry(
        rec, GATE.CARDS[card]["case"])["open_association_rows"]
    assert rows["face"] == "u"
    assert rows["rebuilt_sum_minus_nemo_depth_max_abs"] == 0.0
    assert rows["one_over_depth_minus_nemo_reciprocal_max_abs"] == 0.0
    assert "column_mean_divided_minus_multiplied_max_abs" in rows
    assert rows["column_mean_divided_minus_multiplied_max_abs"] >= 0.0


def test_record_geometry_refuses_a_record_without_the_round33_patch(tmp_path,
                                                                    monkeypatch):
    """A round-29-only record must be REFUSED, not silently rebuilt."""
    root = _synthetic_root(tmp_path, "LOCK_EXCHANGE")
    rec = R29.read_zdf_matrix(root / GATE.ZDF_MATRIX_RECORD)
    del rec["arrays"]["hu_0"]
    with pytest.raises(SWEEP.GateError, match="round-33 reference-geometry"):
        GATE.record_geometry(rec, "LOCK_EXCHANGE-zco")


def test_the_round33_patch_only_adds_and_stacks_on_the_round29_one(tmp_path):
    """The instrument is WRITE-only, and the two patches compose."""
    import subprocess

    shipped = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2"
                   "/src/OCE/DYN/dynzdf.F90")
    r29 = GATES / ("nemo_testcase_l2_gyre_round29_zdf/dynzdf_round29.patch")
    r33 = GATES / ("nemo_testcase_l1_tanks_round33_zdf/"
                   "dynzdf_round33_refgeom.patch")
    for patch_file in (r29, r33):
        removed = [line for line in patch_file.read_text().splitlines()
                   if line.startswith("-") and not line.startswith("---")]
        assert removed == [], f"{patch_file.name} deletes a shipped line"
    work = tmp_path / "dynzdf.F90"
    work.write_bytes(shipped.read_bytes())
    for patch_file in (r29, r33):
        done = subprocess.run(["patch", "-s", str(work)],
                              stdin=patch_file.open("rb"),
                              capture_output=True)
        assert done.returncode == 0, done.stderr
    body = work.read_text()
    # every literal the reader looks for must be exactly 16 characters, or the
    # self-describing stream desynchronises and the whole record is garbage
    for name in ("e3u_0", "e3v_0", "hu_0", "r1_hu_0", "hv_0", "r1_hv_0"):
        literal = f"'{name.ljust(16)}'"
        assert literal in body, f"{name} is not written with a 16-char label"


# --------------------------------------------------------------------------
# the whole gate, on a synthetic record in the instrument's own binary layout
# --------------------------------------------------------------------------

def _write_synthetic_record(path: Path, case: str, field_f, target_f,
                            geometry: dict) -> None:
    """The exact stream ``dynzdf_round29.patch`` writes, with our own payload.

    16-byte magic, sixteen default integers, then ``(name[16], rank, n1, n2,
    n3, payload)`` per array.  Every array the reader demands is present;
    only the two the gate consumes carry real values.
    """
    nx, ny, nz = SWEEP.DIMS[case]
    header = (1, 1, 3, 1, 2, 3, 3, nx, ny, nz, nz - 1, 3, nx - 2, 3, ny - 2, 64)
    payloads = {}
    names = list(R29.EXPECTED_ARRAYS) + ["e3u_0", "hu_0", "r1_hu_0"]
    # Anything else the caller supplied travels too, so a fixture can add the
    # v-side reference geometry the real round-33 patch also dumps.
    names += [name for name in geometry if name not in names]
    for name in names:
        if name in geometry:
            values = geometry[name]
            payloads[name] = (3 if values.ndim == 3 else 2, values)
            continue
        if name in ("rDt", "rho0"):
            payloads[name] = (0, np.array([1.0 if name == "rDt" else 1026.0]))
        elif name in ("uu_b_Kaa", "vv_b_Kaa", "rCdU_bot", "mbku", "mbkv",
                      "utauU", "vtauV"):
            values = (target_f if name == "uu_b_Kaa"
                      else np.zeros((nx, ny), dtype=np.float64))
            payloads[name] = (2, values)
        else:
            values = (field_f if name == "uu_Kaa_out"
                      else np.zeros((nx, ny, nz), dtype=np.float64))
            payloads[name] = (3, values)
    with path.open("wb") as handle:
        handle.write(b"NEMO_L2_ZDFMX_1 ")
        handle.write(struct.pack("=16i", *header))
        for name in names:
            rank, values = payloads[name]
            if rank == 0:
                n1, n2, n3 = 1, 1, 1
            elif rank == 2:
                n1, n2, n3 = nx, ny, 1
            else:
                n1, n2, n3 = nx, ny, nz
            handle.write(name.ljust(16).encode("ascii"))
            handle.write(struct.pack("=4i", rank, n1, n2, n3))
            handle.write(np.ascontiguousarray(
                np.asarray(values, dtype="<f8").ravel(order="F")).tobytes())


@pytest.fixture(autouse=True)
def _stamp_does_not_depend_on_the_working_tree(monkeypatch):
    """Let these tests run on a tree that is mid-edit.

    ``run`` stamps the worktree and the stamper REFUSES a dirty tree, which is
    right for a gate producing an artifact and wrong for a unit test: it made
    every arithmetic test below fail for a reason that has nothing to do with
    the arithmetic.  Round 32 lost four gate results to exactly that.  The
    stamp itself is exercised when the gate really runs, on a clean tree; here
    it is stubbed and only its PRESENCE in the report is asserted.
    """
    monkeypatch.setattr(GATE, "worktree_stamp",
                        lambda *a, **k: {"stubbed_by_the_unit_test": True})


def _synthetic_root(tmp_path: Path, card: str, *, perturb_target=False) -> Path:
    """A record whose operator output IS the tank's own stage-3 velocity.

    Built by inverting the operator rather than by guessing: hand it NEMO's own
    stage-3 ``u`` as the pre-correction field and the depth mean that field
    already has as the target, so the shift it installs is exactly zero and the
    answer must come back equal to the input.  That makes the END-TO-END path
    -- reader, layout, trim, mask, score -- testable with no NEMO run.
    """
    spec = GATE.CARDS[card]
    case = spec["case"]
    nx, ny, nz = SWEEP.DIMS[case]
    nemo = _nemo_reference_geometry(card)
    scored = nemo["scored"]
    h_face = scored["e3u_0"] * scored["umask"]
    wet2d = (scored["umask"] > 0).any(axis=-1).astype(np.float64)
    stage = SWEEP.read_stage(spec["twin"] / GATE.STAGE3_RECORD, case, 3)
    field = np.asarray(stage["u"], dtype=np.float64)
    # The OPERATOR'S OWN reduction, imported rather than re-typed: the fixture
    # inverts what the operator computes, so any reduction of its own would
    # test the reduction instead of the plumbing.  ROUND 36: it MULTIPLIES by
    # NEMO's own r1_hu_0 (stprk3_stg.f90:522) rather than dividing by hu_0, so
    # the inverted target does the same -- with the divide here the OVERFLOW
    # arm went red on 5 of its 606 columns.  ROUND 37: and it accumulates in
    # ascending k rather than letting XLA pick a tree, which moved those same
    # 5 columns; a jnp.sum here would now reintroduce exactly that difference.
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.barotropic_common import _ascending_level_sum
    own_mean = np.asarray(
        _ascending_level_sum(jnp.asarray(field * h_face)) * scored["r1_hu_0"])
    if perturb_target:
        own_mean = own_mean + 1.0e-9
    field_f = np.zeros((nx, ny, nz), dtype=np.float64)
    field_f[2:-2, 2:-2, :] = field.transpose(1, 0, 2)
    target_f = np.zeros((nx, ny), dtype=np.float64)
    target_f[2:-2, 2:-2] = own_mean.T
    root = tmp_path / f"round33_{card.lower()}_zdf_matrix"
    root.mkdir()
    _write_synthetic_record(
        root / GATE.ZDF_MATRIX_RECORD, case, field_f, target_f,
        {k: nemo[k] for k in ("e3u_0", "umask", "hu_0", "r1_hu_0")})
    return root


@pytest.mark.parametrize("card", ["LOCK_EXCHANGE", "OVERFLOW"])
def test_the_gate_runs_end_to_end_on_a_synthetic_record(tmp_path, card,
                                                        monkeypatch):
    root = _synthetic_root(tmp_path, card)
    monkeypatch.setitem(GATE.CARDS[card], "root", root)
    report = GATE.run(card)
    assert report["status"] == "AT-BAR"
    assert report["rows"][0]["exact"] is True
    assert report["card"] == card
    assert "worktree" in report
    # ROUND-34 RETRACTION.  This used to assert the report named the v face
    # as UNMEASURED, on the claim that the stage record carries no v.  It
    # does carry v; the reader skipped it.  The v face is NOT APPLICABLE for
    # a MEASURED reason -- NEMO's own vmask on these 2-D x-z tanks is
    # identically zero -- and the report must say that instead.
    assert report["unmeasured"] == []
    assert [row["face"] for row in report["not_applicable_faces"]] == ["v"]
    assert report["not_applicable_faces"][0]["nemo_velocity_max_abs"] == 0.0
    assert "vmask" in report["not_applicable_faces"][0]["reason"]
    assert [row["name"].rsplit(".", 1)[-1] for row in report["rows"]] == ["u"]
    # nothing is reconstructed any more: the geometry comes from the record
    assert report["inputs_reconstructed_not_nemo"] == []
    assert report["open_association_rows"]["u"][
        "rebuilt_sum_minus_nemo_depth_max_abs"] == 0.0


@pytest.mark.parametrize("card", ["LOCK_EXCHANGE", "OVERFLOW"])
def test_the_planted_unit_offset_turns_the_gate_red(tmp_path, card,
                                                    monkeypatch):
    root = _synthetic_root(tmp_path, card)
    monkeypatch.setitem(GATE.CARDS[card], "root", root)
    assert GATE.run(card)["status"] == "AT-BAR"          # the control's control
    planted = GATE.run(card, plant=True)
    assert planted["status"] == "DEBT"
    assert planted["rows"][0]["absolute_max"] >= 1.0


def test_a_wrong_barotropic_target_turns_the_gate_red(tmp_path, monkeypatch):
    """The second plant: NEMO's uu_b(Kaa) moved by 1 nm/s, nothing else."""
    root = _synthetic_root(tmp_path, "LOCK_EXCHANGE", perturb_target=True)
    monkeypatch.setitem(GATE.CARDS["LOCK_EXCHANGE"], "root", root)
    report = GATE.run("LOCK_EXCHANGE")
    assert report["status"] == "DEBT"
    assert report["rows"][0]["absolute_max"] > 1.0e-10


def test_the_gate_fails_closed_before_the_acquisition_has_run(tmp_path,
                                                              monkeypatch):
    monkeypatch.setitem(GATE.CARDS["OVERFLOW"], "root", tmp_path / "absent")
    with pytest.raises(SystemExit) as excinfo:
        GATE.run("OVERFLOW")
    message = str(excinfo.value)
    assert "UNMEASURED" in message
    assert "nemo_testcase_l1_tanks_round33_zdf/run.sh OVERFLOW" in message


def test_the_add_only_check_can_fire_on_a_deleted_blank_line():
    """The DIFF review's finding: ``^-[^-]`` misses a bare ``-``.

    A unified diff emits a deleted BLANK line as a single ``-``, so the old
    pattern classified it as "adds only".  This drives the SAME shell
    expression run.sh uses, on a patch that deletes one blank line, and it
    must count as a deletion.
    """
    import subprocess

    patch_text = ("--- a\n+++ b\n@@ -1,3 +1,3 @@\n line one\n-\n+added\n"
                  " line three\n")
    scratch = tmp = Path(__file__).parent / "_r33_addonly_probe.patch"
    try:
        scratch.write_text(patch_text)
        minus = subprocess.run(["grep", "-c", "^-", str(scratch)],
                               capture_output=True, text=True).stdout.strip()
        header = subprocess.run(["grep", "-c", "^---", str(scratch)],
                                capture_output=True, text=True).stdout.strip()
        assert int(minus) != int(header), (
            "a deleted blank line must be counted as a deletion")
        # and the pattern the review defeated does NOT see it
        old_pattern = subprocess.run(["grep", "-c", "^-[^-]", str(scratch)],
                                     capture_output=True, text=True)
        assert old_pattern.stdout.strip() == "0"
    finally:
        if scratch.exists():
            scratch.unlink()


def test_the_acquisition_script_refuses_without_a_card():
    """The one thing about run.sh testable without running NEMO."""
    import subprocess

    script = GATES / "nemo_testcase_l1_tanks_round33_zdf/run.sh"
    assert script.is_file()
    done = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert done.returncode == 64
    assert "LOCK_EXCHANGE|OVERFLOW" in done.stderr
    # and it must be syntactically valid, so a typo cannot sit here unnoticed
    check = subprocess.run(["bash", "-n", str(script)], capture_output=True)
    assert check.returncode == 0


def test_the_v_face_is_scored_when_nemo_wets_it(tmp_path, monkeypatch):
    """The v arm must not be dead code.

    Round 34 opened the v face after finding the record had carried ``v`` all
    along.  On the two real tanks NEMO's own ``vmask`` is identically zero, so
    the v row is reported NOT APPLICABLE and the arm never executes -- which
    is exactly the shape of a branch that could be broken and never noticed.
    This drives the SAME gate on a record whose ``vmask`` IS wet, built by
    copying the card's own u-face geometry onto the v side, and asserts the v
    row appears and scores.  The paired arm perturbs NEMO's own ``vv_Kaa_out``
    by 1 m/s and requires the same row to go red, so a v branch that silently
    scored nothing could not pass both.
    """
    card = "LOCK_EXCHANGE"
    spec = GATE.CARDS[card]
    case = spec["case"]
    nx, ny, nz = SWEEP.DIMS[case]
    nemo = _nemo_reference_geometry(card)
    scored = nemo["scored"]
    h_face = scored["e3u_0"] * scored["umask"]
    wet2d = (scored["umask"] > 0).any(axis=-1).astype(np.float64)
    stage = SWEEP.read_stage(spec["twin"] / GATE.STAGE3_RECORD, case, 3)
    field = np.asarray(stage["u"], dtype=np.float64)
    import jax.numpy as jnp
    # ROUND 36: invert the operator's MULTIPLY, as the fixture above does.
    own_mean = np.asarray(
        jnp.sum(field * h_face, axis=-1) * scored["r1_hu_0"])
    field_f = np.zeros((nx, ny, nz), dtype=np.float64)
    field_f[2:-2, 2:-2, :] = field.transpose(1, 0, 2)
    target_f = np.zeros((nx, ny), dtype=np.float64)
    target_f[2:-2, 2:-2] = own_mean.T

    # NEMO's own stage-3 v on this tank is identically zero, so the v side of
    # the record is zero too and the correction must return zero.  The u side
    # keeps the inverted-operator pair.
    real_masks = SWEEP.expected_masks
    monkeypatch.setattr(
        GATE, "expected_masks",
        lambda c: {**real_masks(c), "v": real_masks(c)["u"]})

    def _run(v_field):
        geometry = {k: nemo[k] for k in ("e3u_0", "umask", "hu_0", "r1_hu_0")}
        geometry.update({
            "e3v_0": nemo["e3u_0"], "vmask": nemo["umask"],
            "hv_0": nemo["hu_0"], "r1_hv_0": nemo["r1_hu_0"],
            "vv_Kaa_out": v_field,
            "vv_b_Kaa": np.zeros((nx, ny), dtype=np.float64),
        })
        root = tmp_path / f"wet_v_{'planted' if v_field.any() else 'clean'}"
        root.mkdir()
        _write_synthetic_record(root / GATE.ZDF_MATRIX_RECORD, case,
                                field_f, target_f, geometry)
        monkeypatch.setitem(GATE.CARDS[card], "root", root)
        return GATE.run(card)

    report = _run(np.zeros((nx, ny, nz), dtype=np.float64))
    faces = [row["name"].rsplit(".", 1)[-1] for row in report["rows"]]
    assert faces == ["u", "v"], faces
    assert report["not_applicable_faces"] == []
    assert all(row["exact"] for row in report["rows"])
    assert report["status"] == "AT-BAR"
    assert report["open_association_rows"]["v"]["face"] == "v"
    assert report["rows"][1]["nemo_statement"] == "stprk3_stg.F90:441,445"
    assert report["rows"][1]["n"] == report["rows"][0]["n"]

    planted = np.zeros((nx, ny, nz), dtype=np.float64)
    planted[nx // 2, ny // 2, 0] = 1.0
    report = _run(planted)
    assert report["rows"][1]["status"] == "DEBT"
    assert report["status"] == "DEBT"
