"""The round-35 ``tra_zdf`` reader and gate, proven BEFORE the record exists.

The NEMO acquisition is user-executed and has not been run.  So the reader is
exercised against a SYNTHETIC record written by a Python twin of the Fortran
writer -- same magic, same sixteen header integers, same
``(name[16], rank, n1, n2, n3, payload)`` groups, same column-major payload.

THE PAYLOAD IS NOT BUILT BY THE READER'S OWN TRANSCRIPTION.  ``_nemo_columns``
below is a plain per-column triple loop written straight off ``trazdf.F90``;
the reader's ``nemo_rebuild`` is a vectorised one.  If the two agreed because
one called the other, the calibration arm would be tautological on this
record, which is exactly the failure mode a synthetic fixture invites.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
GATE = (REPO / "scripts/validate/ocean_fidelity/testcases"
        / "nemo_testcase_l2_gyre_round35_trazdf_matrix.py")
sys.path.insert(0, str(GATE.parent))

from legoesm.ocean.fidelity.provenance import allow_dirty_stamps  # noqa: E402
import nemo_testcase_l2_gyre_round35_trazdf_matrix as R  # noqa: E402

JPI, JPJ, JPK = 9, 9, 6
# The domain the gate PINS for the real card, captured before any fixture
# substitutes the synthetic one.  Round 36 pinned it because an independent
# attack cropped the record's domain, kept the halo symmetric, and got a
# green verdict out of a run that scored 30 cells instead of 21120.
PINNED_DOMAIN = R.EXPECTED_DOMAIN


@pytest.fixture(autouse=True)
def _synthetic_domain(monkeypatch):
    """Every synthetic record here is 9x9x6, not GYRE's 36x26x31.

    The gate refuses a domain that is not the card's, so the tests that build
    their own geometry say so EXPLICITLY rather than the gate accepting any
    geometry.  Tests that read the REAL record ask for ``real_domain``.
    """
    monkeypatch.setattr(R, "EXPECTED_DOMAIN", (JPI, JPJ, JPK))


@pytest.fixture
def real_domain(monkeypatch):
    monkeypatch.setattr(R, "EXPECTED_DOMAIN", PINNED_DOMAIN)
    return PINNED_DOMAIN
NTSI, NTEI, NTSJ, NTEJ = 3, 7, 3, 7          # a 2-cell halo on every side
JPKM1 = JPK - 1
RDT = 1234.5


def _f(name: str) -> bytes:
    assert len(name) <= 16
    return name.ljust(16).encode("ascii")


def _nemo_columns(avt, a33, e3w, e3t_kaa, e3t_kbb, e3t_kmm, tmask,
                  t_bb, t_rhs, p2dt):
    """``trazdf.F90`` for ONE column, written as the Fortran loops are.

    Independent of the reader on purpose: plain scalar recurrences, 1-based
    NEMO indices translated by hand, so that agreement between this and the
    reader's array form is evidence rather than a tautology.
    """
    nk = len(avt)
    jpkm1 = nk - 1
    zwt = [0.0] * nk
    for jk in range(2, nk + 1):                       # :172-174
        zwt[jk - 1] = avt[jk - 1] + a33[jk - 1]
    zwt[0] = 0.0                                      # :204
    zwi = [0.0] * jpkm1
    zws = [0.0] * jpkm1
    zwd = [0.0] * jpkm1
    for jk in range(1, jpkm1 + 1):                    # :218-222
        zwi[jk - 1] = -p2dt * zwt[jk - 1] / e3w[jk - 1]
        zws[jk - 1] = -p2dt * zwt[jk] / e3w[jk]
        zwd[jk - 1] = e3t_kaa[jk - 1] - (zwi[jk - 1] + zws[jk - 1])
    lu = list(zwt)
    lu[0] = zwd[0]                                    # :257
    for jk in range(2, jpkm1 + 1):                    # :260
        lu[jk - 1] = zwd[jk - 1] - zwi[jk - 1] * zws[jk - 2] / lu[jk - 2]
    rhs = [0.0] * nk
    rhs[0] = e3t_kbb[0] * t_bb[0] + p2dt * e3t_kmm[0] * t_rhs[0]   # :272-273
    for jk in range(2, jpkm1 + 1):                    # :276-277
        rhs[jk - 1] = (e3t_kbb[jk - 1] * t_bb[jk - 1]
                       + p2dt * e3t_kmm[jk - 1] * t_rhs[jk - 1])
    fwd = list(rhs)
    for jk in range(2, jpkm1 + 1):                    # :278
        fwd[jk - 1] = rhs[jk - 1] - zwi[jk - 1] / lu[jk - 2] * fwd[jk - 2]
    sol = list(fwd)
    sol[jpkm1 - 1] = fwd[jpkm1 - 1] / lu[jpkm1 - 1] * tmask[jpkm1 - 1]  # :282
    for jk in range(jpkm1 - 1, 0, -1):                # :285-286
        sol[jk - 1] = ((fwd[jk - 1] - zws[jk - 1] * sol[jk])
                       / lu[jk - 1] * tmask[jk - 1])
    return zwt, zwi, zwd, zws, lu, rhs, sol


def synthetic_record(path: Path, *, dry_bottom: bool = False,
                     a33_live: bool = False, clamp_fires: bool = False,
                     arm: dict | None = None,
                     magic: str = "NEMO_L2_TRAZD_1",
                     version: int = 1, bits: int = 64,
                     duplicate: bool = False, truncate: int = 0,
                     trailing: int = 0, nonfinite: bool = False) -> Path:
    """A Python twin of the Fortran writer, byte layout included."""
    rng = np.random.default_rng(20260906)
    shape = (JPI, JPJ, JPK)
    tmask = np.ones(shape)
    if dry_bottom:
        tmask[:, :, JPK - 2:] = 0.0
    avt = 1e-5 + 1e-4 * rng.random(shape)
    avt[:, :, 0] = 0.0
    avt[:, :, JPK - 1] = 0.0                 # NEMO masks avt at jpk
    a33 = (1e-6 * rng.random(shape)) if a33_live else np.zeros(shape)
    if a33_live:
        a33[:, :, 0] = 0.0
    e3t_0 = np.broadcast_to(
        np.linspace(10.0, 60.0, JPK), shape).copy()
    e3w_0 = np.broadcast_to(
        np.linspace(5.0, 55.0, JPK), shape).copy()
    r3 = {k: 1e-3 * rng.random((JPI, JPJ)) for k in ("Kbb", "Kmm", "Kaa")}
    e3t = {k: e3t_0 * (1.0 + r3[k][:, :, None] * tmask) for k in r3}
    e3w_kmm = e3w_0 * (1.0 + r3["Kmm"][:, :, None])
    fields = {"T": (10.0 + rng.random(shape), 1e-4 * rng.random(shape)),
              "S": (35.0 + rng.random(shape), 1e-5 * rng.random(shape))}
    if clamp_fires:
        fields["S"] = (-40.0 + rng.random(shape), 1e-5 * rng.random(shape))

    out = {n: np.zeros(shape) for n in
           ("zwt_mix", "zwt_lu", "rhs_T", "rhs_S", "fwd_T", "fwd_S",
            "sol_T_pre_clamp", "sol_S_pre_clamp")}
    for n in ("zwi", "zwd", "zws"):
        out[n] = np.zeros((JPI, JPJ, JPKM1))
    for i in range(NTSI - 1, NTEI):
        for j in range(NTSJ - 1, NTEJ):
            for tag in ("T", "S"):
                zwt, zwi, zwd, zws, lu, rhs, sol = _nemo_columns(
                    avt[i, j], a33[i, j], e3w_kmm[i, j], e3t["Kaa"][i, j],
                    e3t["Kbb"][i, j], e3t["Kmm"][i, j], tmask[i, j],
                    fields[tag][0][i, j], fields[tag][1][i, j], RDT)
                fwd = list(rhs)
                for jk in range(2, JPKM1 + 1):
                    fwd[jk - 1] = (rhs[jk - 1]
                                   - zwi[jk - 1] / lu[jk - 2] * fwd[jk - 2])
                out["zwt_mix"][i, j] = zwt
                out["zwt_lu"][i, j] = lu
                out["zwi"][i, j] = zwi
                out["zwd"][i, j] = zwd
                out["zws"][i, j] = zws
                out[f"rhs_{tag}"][i, j] = rhs
                out[f"fwd_{tag}"][i, j] = fwd
                out[f"sol_{tag}_pre_clamp"][i, j] = sol
    post = {t: out[f"sol_{t}_pre_clamp"].copy() for t in ("T", "S")}
    post["S"][post["S"] < 0.0] = 0.1                     # trazdf.F90:89-91

    flags = {"ln_zdfddm": 0.0, "ln_zad_Aimp": 0.0, "ln_zdfmfc": 0.0,
             "ln_traldf_msc": 0.0, "l_ldfslp": 1.0, "ln_SEOS": 0.0,
             "a33_allocated": 1.0, "rDt": RDT, "rn_b0": 0.0,
             "jp_tem": 1.0, "jp_sal": 2.0}
    flags.update(arm or {})
    payload = {
        **flags,
        "T_Kbb_in": fields["T"][0], "S_Kbb_in": fields["S"][0],
        "T_Kmm_in": fields["T"][0], "S_Kmm_in": fields["S"][0],
        "T_Krhs_in": fields["T"][1], "S_Krhs_in": fields["S"][1],
        "zwt_mix": out["zwt_mix"], "zwi": out["zwi"], "zwd": out["zwd"],
        "zws": out["zws"], "zwt_lu": out["zwt_lu"],
        "rhs_T": out["rhs_T"], "rhs_S": out["rhs_S"],
        "fwd_T": out["fwd_T"], "fwd_S": out["fwd_S"],
        "sol_T_pre_clamp": out["sol_T_pre_clamp"],
        "sol_S_pre_clamp": out["sol_S_pre_clamp"],
        "sol_T_post_clamp": post["T"], "sol_S_post_clamp": post["S"],
        "avt": avt, "avs": avt, "ah_wslp2": a33, "akz": np.zeros(shape),
        "tmask": tmask,
        "e3t_Kbb": e3t["Kbb"], "e3t_Kmm": e3t["Kmm"], "e3t_Kaa": e3t["Kaa"],
        "e3w_Kmm": e3w_kmm, "e3t_0": e3t_0, "e3w_0": e3w_0,
        "r3t_Kbb": r3["Kbb"], "r3t_Kmm": r3["Kmm"], "r3t_Kaa": r3["Kaa"],
    }
    if nonfinite:
        payload["avt"] = payload["avt"].copy()
        payload["avt"][NTSI, NTSJ, 2] = np.nan

    blob = bytearray()
    blob += _f(magic)
    blob += struct.pack("=16i", version, 1, 3, 1, 2, 3, 4, JPI, JPJ, JPK,
                        JPKM1, NTSI, NTEI, NTSJ, NTEJ, bits)
    names = list(R.EXPECTED_ARRAYS)
    if duplicate:
        names.append("avt")
    for name in names:
        value = payload[name]
        blob += _f(name)
        if np.isscalar(value):
            blob += struct.pack("=4i", 0, 1, 1, 1)
            blob += struct.pack("<d", float(value))
        elif np.asarray(value).ndim == 2:
            blob += struct.pack("=4i", 2, JPI, JPJ, 1)
            blob += np.asarray(value, "<f8").tobytes(order="F")
        else:
            arr = np.asarray(value, "<f8")
            blob += struct.pack("=4i", 3, JPI, JPJ, arr.shape[2])
            blob += arr.tobytes(order="F")
    if truncate:
        blob = blob[:-truncate]
    if trailing:
        blob += b"\x00" * trailing
    path.write_bytes(bytes(blob))
    return path


@pytest.fixture(scope="module")
def clean_record(tmp_path_factory) -> Path:
    return synthetic_record(
        tmp_path_factory.mktemp("r35") / "oracle_trazdf_matrix_kt00000001.bin")


def _run(record: Path, **kw) -> dict:
    with allow_dirty_stamps(True):
        return R.run(record, with_card=False, **kw)


def test_the_reader_parses_the_twin_and_the_arms_read_what_they_should(
        clean_record):
    """Calibration exact; assembly and sweep at bar; the RHS form is NOT.

    The last one is a RESULT, not a fixture artifact: legoESM's content
    builder groups the update as ``h*(p2dt*T)`` where NEMO writes
    ``p2dt*h*T`` (``trazdf.F90:219-220`` for the matrix, the same style at
    :276-277 for the RHS), and the two are not the same floating-point
    expression.  It matters because round 36 cannot flip ``tracer_combine``
    to the content form and expect bit equality from that alone.
    """
    report = _run(clean_record)
    assert report["calibrated"] is True, [
        r for r in report["calibration_rows"] if r["status"] not in R.PASSING]
    assert set(report["arrays_read"]) == set(R.EXPECTED_ARRAYS)

    by_name = {r["name"].rsplit(".trazdf.", 1)[1]: r
               for r in report["given_inputs_rows"]}
    # ROUND-37.  The two off-diagonals used to be AT-BAR-SIGNED-ZERO here,
    # 704 cells each: NEMO's zwi at the surface row and zws at the bottom are
    # -0.0 (trazdf.f90:419 sets zwt(:,1) = 0; :443-444 divide it) and legoESM
    # wrote +0.0.  It now writes the negative zeros, so both rows are exactly
    # AT-BAR.  The equivalence class itself is NOT deleted -- it is still
    # reachable, and the test below drives it directly on a constructed pair,
    # so a future +0.0 regression would still be classified rather than
    # silently folded into AT-BAR.
    for name in ("assembly.zwd", "sweep.T", "sweep.S",
                 "assembly.zwi", "assembly.zws"):
        assert by_name[name]["status"] == "AT-BAR", by_name[name]
        assert by_name[name]["bit_unequal"] == 0, by_name[name]
    # ROUND-36 RETRACTION.  Round 35 recorded this row as VALUE-AT-BAR with
    # 1/125 cells unequal at 5.2e-17 and read that as a property of
    # ``thickness_weighted_tracer_content``.  It was not: the ARM formed
    # ``e3t_Kmm*(p2dt*T_Krhs)``, an association NEMO never writes.  Under
    # NEMO's own grouping ``(p2dt*e3t_Kmm)*T_Krhs`` (trazdf.f90:528-529) the
    # builder is BIT-EXACT, here and on the real GYRE record (0/21120).
    rhs = by_name["rhs_content.T"]
    assert rhs["status"] == "AT-BAR", rhs
    assert rhs["bit_unequal"] == 0 and rhs["absolute_max"] == 0.0
    # the old grouping is kept as a REPORTED sensitivity, never scored, and
    # it must still carry its residual or the retraction has no evidence
    sensitivity = {r["name"].rsplit(".trazdf.", 1)[1]: r
                   for r in report["rhs_association_sensitivity"]}
    premultiplied = sensitivity["rhs_content_premultiplied.T"]
    assert 0 < premultiplied["bit_unequal"] < premultiplied["n"]
    assert premultiplied["normalized_max_abs"] < R.BAR
    # and a REPORTED row cannot decide the verdict
    assert all(r["status"] in R.PASSING
               for group in ("calibration_rows", "given_inputs_rows",
                             "clamp_rows", "condition_rows")
               for r in report[group]), report["status"]
    assert report["status"] == "AT-BAR"


def test_the_calibration_is_not_tautological(clean_record):
    """The payload came from the scalar loops, the rebuild from array form."""
    rec = R.read_trazdf_matrix(clean_record)
    rebuilt = R.nemo_rebuild(rec)
    dumped = R._box(rec, "zwd")
    assert np.array_equal(dumped.view(np.uint64),
                          rebuilt["zwd"].view(np.uint64))
    # and the two really are different code: perturbing an operand moves the
    # rebuild and not the dump
    rec["arrays"]["avt"] = rec["arrays"]["avt"] * 1.5
    assert not np.array_equal(R.nemo_rebuild(rec)["zwd"], dumped)


@pytest.mark.parametrize("plant", R.PLANT_ARMS)
def test_every_plant_turns_the_gate_red_and_MOVES_a_row(clean_record, plant):
    """A red verdict is not enough when the baseline is already red.

    This gate's unplanted run carries one DEBT row -- the RHS association --
    so "the planted run exited non-zero" would be true with the plant
    deleted.  Each plant must be shown to MOVE a row.
    """
    report = _run(clean_record, plant=plant)
    assert report["status"] == "DEBT", plant
    assert report["plant_landed"] is True, plant
    assert report["plant_moved_rows"], plant
    # and the moved row must belong to the arm the plant names
    arm_rows = {"operand": "calibration", "matrix": "calibration",
                "sweep": "sweep", "assembly": "assembly",
                "rhs": "calibration", "a33": "a33_fold_inert",
                "clamp": "clamp"}[plant]
    assert any(arm_rows in name for name in report["plant_moved_rows"]), (
        plant, report["plant_moved_rows"])


def test_an_inert_plant_would_be_refused(clean_record, monkeypatch):
    """Non-vacuity of the landing check itself: a plant that perturbs
    nothing must be reported as landing nowhere, not as a red gate."""
    import nemo_testcase_l2_gyre_round35_trazdf_matrix as gate
    original = gate.np.nextafter
    monkeypatch.setattr(gate.np, "nextafter", lambda a, b: a)
    try:
        report = _run(clean_record, plant="operand")
    finally:
        monkeypatch.setattr(gate.np, "nextafter", original)
    assert report["plant_landed"] is False
    assert report["plant_moved_rows"] == []


# ONE arm end to end, not seven.  The CLI's exit-code contract does not
# depend on which arm was planted -- ``main`` maps (plant landed, status) to
# an exit code the same way for all of them, and every arm's LANDING is
# already scored in-process above.  Seven subprocesses against the real
# record cost fourteen JAX compilations and took the fidelity suite from 100
# seconds to over twenty minutes: a slower suite, not more coverage.  run.sh
# still drives all seven arms on every acquisition.
@pytest.mark.parametrize("plant", ["assembly"])
def test_every_plant_exits_non_zero_end_to_end(plant, real_domain):
    """End to end through the CLI, on the record that exists.

    ROUND 36: this used to drive a 9x9x6 SYNTHETIC record.  The gate now pins
    the card's domain -- because an attack cropped a record's domain, kept the
    halo symmetric, and got a green verdict out of 30 scored cells -- and a
    subprocess cannot be monkeypatched, so relaxing the pin for this test
    would have meant an env var that relaxes the very guard it protects.
    Driving NEMO's own record instead is strictly better evidence anyway; the
    synthetic-record plant behaviour is covered in-process above.
    """
    record = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                  "round35_oracle_trazdf_matrix/"
                  "oracle_trazdf_matrix_kt00000001.bin")
    if not record.exists():
        pytest.skip("round-35 record absent")
    env = {"PYTHONPATH": ":".join(
        str(REPO / f"packages/{p}") for p in
        ("core", "ocean", "atmosphere", "coupler", "ice", "land", "ml",
         "tools")) + f":{REPO / 'src'}",
        "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1",
        "LEGOESM_GATE_ALLOW_DIRTY": "1", "PATH": "/usr/bin:/bin"}
    base = [sys.executable, str(GATE), "--record", str(record), "--no-card"]
    clean = subprocess.run(base, env=env, capture_output=True, text=True)
    assert "SCORED " in clean.stdout and "BLIND-SPOT " in clean.stdout
    planted = subprocess.run(base + ["--plant", plant], env=env,
                             capture_output=True, text=True)
    # A non-zero exit alone would also be produced by a CRASH, so require the
    # gate's own verdict -- the same rule run.sh applies.
    assert planted.returncode != 0, planted.stdout[-3000:]
    assert "STATUS DEBT" in planted.stdout
    # A non-zero exit and a DEBT verdict are BOTH available with the plant
    # deleted, because this record's baseline is already red.  The plant must
    # also say it moved something.
    assert f"PLANT {plant} landed=True" in planted.stdout


def test_a_dry_bottom_makes_the_named_deviations_LIVE(tmp_path):
    """Non-vacuity for the deviation report itself.

    On GYRE the scored box has no dry cell, so all three deviations are inert
    and the gate says so.  A record that DOES have one must flip them, or the
    report would be a decoration that always reads the same.
    """
    record = synthetic_record(tmp_path / "dry.bin", dry_bottom=True)
    report = _run(record)
    live = [d for d in report["named_deviations"] if not d["inert_here"]]
    assert len(live) == 2, report["named_deviations"]
    assert all("dry" in d["inert_when"] for d in live)
    # and the masking difference is REAL, not merely declared: legoESM's
    # assembly and NEMO's disagree on this record
    assert any(r["status"] != "AT-BAR"
               for r in report["given_inputs_rows"]
               if ".assembly." in r["name"])


def test_a_live_a33_fold_turns_the_condition_row_red(tmp_path):
    record = synthetic_record(tmp_path / "a33.bin", a33_live=True)
    report = _run(record)
    row = [r for r in report["condition_rows"] if "a33_fold_inert" in r["name"]]
    assert row and row[0]["status"] == "DEBT"
    assert row[0]["max_abs"] > 0.0
    # the fold is still INSIDE the matrix, so the calibration must survive it
    assert report["calibrated"] is True


def test_a_firing_clamp_turns_the_clamp_rows_red(tmp_path):
    record = synthetic_record(tmp_path / "clamp.bin", clamp_fires=True)
    report = _run(record)
    assert any(r["status"] == "DEBT" for r in report["clamp_rows"])


@pytest.mark.parametrize("kwargs,fragment", [
    ({"magic": "NEMO_L2_WRONG_9"}, "bad magic"),
    ({"version": 7}, "bad version"),
    ({"bits": 32}, "not 64-bit"),
    ({"duplicate": True}, "written twice"),
    ({"truncate": 8}, "runs past EOF"),
    ({"trailing": 16}, "truncated array header"),
    ({"nonfinite": True}, "non-finite"),
    ({"arm": {"ln_zad_Aimp": 1.0}}, "arm this reader does not transcribe"),
    ({"arm": {"ln_zdfddm": 1.0}}, "arm this reader does not transcribe"),
    ({"arm": {"ln_zdfmfc": 1.0}}, "arm this reader does not transcribe"),
    ({"arm": {"ln_traldf_msc": 1.0}}, "arm this reader does not transcribe"),
    ({"arm": {"l_ldfslp": 0.0}}, "arm this reader does not transcribe"),
    ({"arm": {"rDt": 0.0}}, "rDt is"),
    ({"arm": {"jp_sal": 1.0}}, "same index"),
])
def test_a_wrong_record_gets_a_verdict_not_a_traceback(tmp_path, kwargs,
                                                       fragment):
    record = synthetic_record(tmp_path / "bad.bin", **kwargs)
    with pytest.raises(R.RecordError) as caught:
        R.read_trazdf_matrix(record)
    assert fragment in str(caught.value)


def test_a_short_record_names_what_is_missing(tmp_path):
    good = synthetic_record(tmp_path / "good.bin")
    raw = good.read_bytes()
    cut = raw.index(b"r3t_Kbb")
    (tmp_path / "short.bin").write_bytes(raw[:cut])
    with pytest.raises(R.RecordError) as caught:
        R.read_trazdf_matrix(tmp_path / "short.bin")
    assert "short of" in str(caught.value)
    assert "r3t_Kbb" in str(caught.value)


def test_the_cli_refuses_a_malformed_record_with_a_status(tmp_path):
    bad = synthetic_record(tmp_path / "bad.bin", magic="NEMO_L2_WRONG_9")
    env = {"PYTHONPATH": ":".join(
        str(REPO / f"packages/{p}") for p in
        ("core", "ocean", "atmosphere", "coupler", "ice", "land", "ml",
         "tools")) + f":{REPO / 'src'}",
        "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1",
        "LEGOESM_GATE_ALLOW_DIRTY": "1", "PATH": "/usr/bin:/bin"}
    done = subprocess.run(
        [sys.executable, str(GATE), "--record", str(bad), "--no-card"],
        env=env, capture_output=True, text=True)
    assert done.returncode == 2
    assert done.stdout.startswith("FAIL:")


def test_the_report_carries_a_commit_stamp_and_the_blind_spot(clean_record):
    report = _run(clean_record)
    assert set(report["worktree"]) >= {"commit", "branch", "clean"}
    assert report["rhs_blind_spot"]["status"] == "UNMEASURED"
    assert "tracer_combine" in report["rhs_blind_spot"]["reason"]


def test_the_gate_prints_which_rhs_arm_each_card_resolves():
    """Rule 10: the resolution is measured from the built card, not declared."""
    rows = R.resolved_rhs_arm()
    assert set(rows) == {"GYRE-zco", "LOCK_EXCHANGE-zco", "OVERFLOW-zps"}
    for case, row in rows.items():
        assert row["zdf_implicit_solver_evaluation"] == "nemo_literal", case
        assert row["tracer_combine"] in ("concentration", "thickness_weighted")


def test_the_solve_arm_drives_the_trajectorys_own_function():
    """Rule 10: the same object, not a copy of the statements."""
    from legoesm.ocean.physics.vertical_mixing import (
        implicit_vertical_diffusion_nemo_tracer_pair,
        nemo_ordered_tridiagonal_solve,
        nemo_tracer_tridiagonal,
    )
    assert nemo_ordered_tridiagonal_solve.__name__ == "_nemo_ordered_solve"
    import jax.numpy as jnp
    rng = np.random.default_rng(7)
    shape = (4, 5)
    K = jnp.asarray(1e-4 * rng.random(shape[:-1] + (shape[-1] - 1,)))
    e3t = jnp.asarray(10.0 + rng.random(shape))
    e3w = jnp.asarray(8.0 + rng.random(shape[:-1] + (shape[-1] - 1,)))
    wet = jnp.ones(shape, bool)
    rhs1 = jnp.asarray(rng.random(shape))
    rhs2 = jnp.asarray(rng.random(shape))
    lower, diag, upper = nemo_tracer_tridiagonal(K, e3t, e3w, 900.0, wet)
    composed = (np.asarray(nemo_ordered_tridiagonal_solve(
        lower, diag, upper, rhs1)),
        np.asarray(nemo_ordered_tridiagonal_solve(lower, diag, upper, rhs2)))
    paired = implicit_vertical_diffusion_nemo_tracer_pair(
        rhs1, rhs2, K, e3t, e3w, 900.0, wet)
    for got, want in zip(composed, paired):
        assert np.array_equal(np.asarray(got).view(np.uint64),
                              np.asarray(want).view(np.uint64))


# --------------------------------------------------------------------------
# The shared admission gate learns self-describing records
# --------------------------------------------------------------------------
def _admission():
    import nemo_testcase_l2_gyre_round21_admission as A
    return A


R29_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round29_oracle_v2_zdf_matrix/oracle_zdf_matrix_kt00000001.bin")


def test_the_admission_gate_reads_a_self_describing_record(clean_record):
    """The round-35 record's own magic must be registered, not unknown.

    Before this round the gate RAISED on any magic outside its fixed schema
    table, and both this record and round 29's momentum one are outside it.
    """
    A = _admission()
    assert "NEMO_L2_TRAZD_1" in A.SELF_DESCRIBING
    assert "NEMO_L2_ZDFMX_1" in A.SELF_DESCRIBING
    magic, dims, fields, order = A._read_self_describing(clean_record)
    assert magic == "NEMO_L2_TRAZD_1"
    assert dims == (JPI, JPJ, JPK)
    assert set(order) == set(R.EXPECTED_ARRAYS)


def test_a_grown_record_passes_and_a_changed_shared_field_does_not(tmp_path):
    """The one legitimate reason two records differ in LENGTH, and its limit.

    Round 33's reference-geometry addition APPENDS arrays to an existing
    record, so a candidate written with it is longer than its baseline.  That
    must pass; a state change in a field BOTH sides carry must not.
    """
    A = _admission()
    base = synthetic_record(tmp_path / "base.bin")
    raw = bytearray(base.read_bytes())
    grown = tmp_path / "grown.bin"
    extra = np.arange(JPI * JPJ, dtype="<f8")
    grown.write_bytes(bytes(raw) + b"appended_field  "
                      + struct.pack("=4i", 2, JPI, JPJ, 1)
                      + extra.tobytes(order="F"))
    result = A._compare_self_describing(base, grown, None)
    assert result["consumed_equal"] is True
    assert result["fields_only_on_one_side"] == ["appended_field"]
    assert result["appended_bytes"] > 0

    # now change one OWNED cell of a field both sides carry
    A2 = _admission()
    changed = bytearray(grown.read_bytes())
    magic, dims, fields, order = A2._read_self_describing(grown)
    rank, off, _ = fields["avt"]
    target = A2._first_owned_index(rank, JPI * JPJ * JPK, JPI, JPJ)
    at = off + 8 * target
    changed[at:at + 8] = struct.pack("<d", 12345.0)
    (tmp_path / "changed.bin").write_bytes(bytes(changed))
    bad = A2._compare_self_describing(base, tmp_path / "changed.bin", None)
    assert bad["consumed_equal"] is False
    assert any(row[0] == "avt" for row in bad["owned_field_differences"])


def test_the_admission_plant_lands_in_an_owned_cell(clean_record):
    """Non-vacuity: a plant at flat index 0 would sit in the halo and be
    ADMITTED, which is the defect round 34 removed from the record-level
    plant and which this branch must not reintroduce."""
    A = _admission()
    assert A._compare_self_describing(
        clean_record, clean_record, None)["consumed_equal"] is True
    planted = A._compare_self_describing(clean_record, clean_record, [False])
    assert planted["consumed_equal"] is False
    assert planted["owned_field_differences"], "the plant landed nowhere"
    # and the targeting itself: for the two projections that HAVE a halo, the
    # chosen cell must be outside it.  (A rank-0 scalar has no halo, so every
    # difference in one is a violation by construction.)
    for rank, size in ((3, JPI * JPJ * JPK), (2, JPI * JPJ)):
        target = A._first_owned_index(rank, size, JPI, JPJ)
        assert target is not None
        i, j = target % JPI, (target // JPI) % JPJ
        assert i >= A.HALO and j >= A.HALO, (rank, target, i, j)
        assert i < JPI - A.HALO and j < JPJ - A.HALO, (rank, target, i, j)


@pytest.mark.skipif(not R29_RECORD.exists(), reason="round-29 record absent")
def test_the_branch_reads_the_real_round29_record():
    """Rule 10: run it against the record that exists, not only a fixture."""
    A = _admission()
    magic, dims, fields, order = A._read_self_describing(R29_RECORD)
    assert magic == "NEMO_L2_ZDFMX_1"
    assert dims[0] > 2 * A.HALO and dims[1] > 2 * A.HALO
    assert "r1_hu_0" not in fields, (
        "the round-29 baseline must NOT already carry the reference geometry; "
        "if it does, the round-35 acquisition adds nothing")
    assert A._compare_self_describing(
        R29_RECORD, R29_RECORD, None)["consumed_equal"] is True
    assert A._compare_self_describing(
        R29_RECORD, R29_RECORD, [False])["consumed_equal"] is False


def test_a_record_from_another_step_is_refused(tmp_path, clean_record):
    """The campaign's claim is kt=1 stage 3; anything else is not scorable.

    The row prefix is also built from the record's own header rather than a
    constant, so a label can never name a step the record is not from -- the
    two cannot disagree even if this refusal were ever relaxed.
    """
    raw = bytearray(clean_record.read_bytes())
    raw[16 + 4:16 + 8] = struct.pack("=i", 7)      # kt = 7
    (tmp_path / "kt7.bin").write_bytes(bytes(raw))
    with pytest.raises(R.RecordError) as caught:
        R.read_trazdf_matrix(tmp_path / "kt7.bin")
    assert "kt is 7, not 1" in str(caught.value)
    report = _run(clean_record)
    assert all(".kt1.stage3." in row["name"]
               for row in report["calibration_rows"])


def test_a_shrunken_scored_box_is_refused(tmp_path, clean_record):
    """The vacuous pass: a sub-box scores a handful of cells, same verdict."""
    raw = bytearray(clean_record.read_bytes())
    for slot, value in ((11, 4), (12, 4), (13, 4), (14, 4)):   # ntsi..ntej
        raw[16 + 4 * slot:16 + 4 * slot + 4] = struct.pack("=i", value)
    (tmp_path / "small.bin").write_bytes(bytes(raw))
    with pytest.raises(R.RecordError) as caught:
        R.read_trazdf_matrix(tmp_path / "small.bin")
    assert "not the whole domain" in str(caught.value)
    assert _run(clean_record)["scored_box"]["cells"] == (
        (NTEI - NTSI + 1) * (NTEJ - NTSJ + 1) * (JPK - 1))


def test_a_manufactured_zero_fold_is_refused(tmp_path):
    """The writer emits zeros for an UNALLOCATED ah_wslp2, so a record that
    claims the slopes are on while saying the array is absent would let the
    fold-inert row pass by construction."""
    record = synthetic_record(tmp_path / "fake.bin", arm={"a33_allocated": 0.0})
    with pytest.raises(R.RecordError) as caught:
        R.read_trazdf_matrix(record)
    assert "unallocated" in str(caught.value)


def test_a_clamp_that_never_ran_is_refused(tmp_path):
    record = synthetic_record(tmp_path / "noclamp.bin",
                              arm={"ln_SEOS": 1.0, "rn_b0": 0.0})
    with pytest.raises(R.RecordError) as caught:
        R.read_trazdf_matrix(record)
    assert "clamp did not run" in str(caught.value)


def test_a_wrong_third_extent_gets_a_verdict_not_a_broadcast_error(tmp_path,
                                                                   clean_record):
    """A short rank-3 array used to pass structural validation and die three
    functions later inside the rebuild."""
    raw = clean_record.read_bytes()
    at = raw.index(b"avt             ")
    head = struct.unpack("=4i", raw[at + 16:at + 32])
    assert head == (3, JPI, JPJ, JPK)
    short = (bytearray(raw[:at + 16])
             + struct.pack("=4i", 3, JPI, JPJ, JPK - 1)
             + bytearray(raw[at + 32:at + 32 + 8 * JPI * JPJ * (JPK - 1)])
             + bytearray(raw[at + 32 + 8 * JPI * JPJ * JPK:]))
    (tmp_path / "short3.bin").write_bytes(bytes(short))
    with pytest.raises(R.RecordError) as caught:
        R.read_trazdf_matrix(tmp_path / "short3.bin")
    assert "'avt'" in str(caught.value)


def test_a_comparison_over_zero_cells_is_refused(clean_record):
    """Non-vacuity of the bar itself: an empty row must not read AT-BAR."""
    with pytest.raises(R.RecordError) as caught:
        R.bit_row("empty", np.array([]), np.array([]))
    assert "nothing to compare" in str(caught.value)


def test_signed_zero_status_cannot_swallow_a_real_difference():
    """AT-BAR-SIGNED-ZERO is an equivalence class, not a tolerance."""
    both_zero = R.bit_row("z", np.array([0.0, 1.0]), np.array([-0.0, 1.0]))
    assert both_zero["status"] == "AT-BAR-SIGNED-ZERO"
    assert both_zero["signed_zero_only"] == 1
    tiny = R.bit_row("t", np.array([0.0, 1.0]), np.array([5e-324, 1.0]))
    assert tiny["status"] != "AT-BAR-SIGNED-ZERO"
    big = R.bit_row("b", np.array([0.0, 1.0]), np.array([1.0, 1.0]))
    assert big["status"] == "DEBT"


# --------------------------------------------------------------------------
# ROUND 36 -- the admission gate's own PRINT path
# --------------------------------------------------------------------------
def test_both_comparators_admit_the_same_shape_so_the_report_can_be_printed(
        tmp_path, capsys):
    """The end of an acquisition must not be where the gate learns its shape.

    ``run`` concatenates ``admitted_differences`` across every record and
    ``main`` prints them through ONE format string.  The self-describing
    comparator used to emit a 5-element LIST while ``compare_record`` emitted
    a dict, so round 35's real run decided ``verdict PASS``, wrote its JSON,
    and then died with ``TypeError: list indices must be integers`` -- taking
    the round-35 gate, its seven plants and the outputs manifest down with it
    under ``set -e``.

    This drives the SAME path: two runs whose only difference is a HALO cell
    of a self-describing record, which is admitted and therefore printed.
    """
    A = _admission()
    base_dir, cand_dir = tmp_path / "base", tmp_path / "cand"
    base_dir.mkdir()
    cand_dir.mkdir()
    name = "oracle_trazdf_matrix_kt00000001.bin"
    base = synthetic_record(base_dir / name)
    raw = bytearray(base.read_bytes())
    _, _, fields, _ = A._read_self_describing(base)
    rank, off, _ = fields["avt"]
    assert rank == 3
    halo_flat = 0                      # i = j = 0, inside the nn_hls halo
    assert A._first_owned_index(rank, JPI * JPJ * JPK, JPI, JPJ) != halo_flat
    at = off + 8 * halo_flat
    raw[at:at + 8] = struct.pack("<d", 12345.0)
    (cand_dir / name).write_bytes(bytes(raw))
    for folder in (base_dir, cand_dir):
        (folder / "mesh_mask.nc").write_bytes(b"identical")

    code = A.main(["--baseline", str(base_dir), "--candidate", str(cand_dir),
                   "--twin", "/nonexistent", "--identical", "mesh_mask.nc",
                   "--allowed-new"])
    printed = capsys.readouterr().out
    assert code == 0, printed
    assert "CONSUMED_FIELD_ADMISSION PASS" in printed
    # the crash was in the line that PRINTS an admitted difference, so the
    # difference must have been admitted AND rendered
    assert f"ADMITTED {name} avt" in printed

    # and the shape contract that printer depends on, stated directly
    report = A.run(base_dir, cand_dir, twin=None, identical=("mesh_mask.nc",))
    assert report["admitted_difference_count"] >= 1
    for entry in report["admitted_differences"]:
        assert isinstance(entry, dict), entry
        assert {"record", "field"} <= set(entry)
        assert "note" in entry or {
            "index_0based", "reason", "baseline_value", "candidate_value",
        } <= set(entry), entry


# --------------------------------------------------------------------------
# ROUND 36 -- avt and avs are NOT full-domain arrays in NEMO
# --------------------------------------------------------------------------
def _shorten_to_tile(source: Path, target: Path, names, *,
                     ni=NTEI - NTSI + 1, nj=NTEJ - NTSJ + 1) -> Path:
    """Rewrite ``names``' payloads to the interior box, headers untouched.

    This reproduces the real round-35 defect byte for byte: ``zdf_oce.f90:85-86``
    allocates ``avt``/``avs`` over ``Nis0:Nie0 x Njs0:Nje0 x jpk``, so
    ``WRITE(unit) avt`` after a header declaring ``jpi, jpj, jpk`` emits
    ``(ntei-ntsi+1)*(ntej-ntsj+1)*jpk`` values -- 57536 bytes short per array
    on GYRE -- and every array after ``avs`` decodes as garbage.
    """
    raw = source.read_bytes()
    out = bytearray(raw[:80])
    off = 80
    while off < len(raw):
        name = raw[off:off + 16].decode("ascii").rstrip()
        rank, n1, n2, n3 = struct.unpack("=4i", raw[off + 16:off + 32])
        count = n1 * n2 * n3
        payload = raw[off + 32:off + 32 + 8 * count]
        if name in names:
            block = np.frombuffer(payload, "<f8").reshape((n1, n2, n3),
                                                          order="F")
            payload = block[NTSI - 1:NTSI - 1 + ni,
                            NTSJ - 1:NTSJ - 1 + nj, :].tobytes(order="F")
            assert len(payload) == 8 * ni * nj * n3
        out += raw[off:off + 32] + payload
        off += 32 + 8 * count
    target.write_bytes(bytes(out))
    return target


def test_the_reader_recovers_avt_and_avs_written_over_the_interior(
        tmp_path, clean_record):
    """A short avt/avs must decode to the SAME gate report, not to garbage.

    ``avm`` on ``zdf_oce.f90:85`` is ``(jpi,jpj,jpk)`` and ``avt``/``avs`` on
    the same statement are not, which is what the round-35 instrument missed.
    The recovery is accepted only because the stream PROVES it -- the tile
    extent is taken only when the declared one fails to leave a known array
    header behind and the tile one does -- and because the calibration arm,
    which rebuilds NEMO's own zwt_mix/zwi/zwd/zws from the embedded avt,
    stays bit-exact.  A misaligned embedding cannot do that.
    """
    short = _shorten_to_tile(clean_record, tmp_path / "short.bin",
                             {"avt", "avs"})
    assert short.stat().st_size < clean_record.stat().st_size

    full_rec = R.read_trazdf_matrix(clean_record)
    short_rec = R.read_trazdf_matrix(short)
    assert set(short_rec["tile_shaped_salvage"]) == {"avt", "avs"}
    assert not full_rec["tile_shaped_salvage"], (
        "a full-length record must NOT be salvaged; the salvage would then "
        "be unconditional and could hide a real truncation")
    for name in ("avt", "avs"):
        assert np.array_equal(R._box(short_rec, name).view(np.uint64),
                              R._box(full_rec, name).view(np.uint64))
    for name in ("tmask", "e3t_Kbb", "e3w_Kmm", "r3t_Kaa", "rhs_T", "zwd"):
        assert np.array_equal(
            np.asarray(short_rec["arrays"][name]).view(np.uint64),
            np.asarray(full_rec["arrays"][name]).view(np.uint64)), name

    full, salvaged = _run(clean_record), _run(short)
    for group in ("calibration_rows", "given_inputs_rows", "clamp_rows"):
        assert ([(r["name"], r["status"], r["absolute_max"])
                 for r in salvaged[group]]
                == [(r["name"], r["status"], r["absolute_max"])
                    for r in full[group]]), group
    assert salvaged["calibrated"] is True


def test_a_payload_matching_neither_extent_is_refused(tmp_path, clean_record):
    """Non-vacuity: the salvage must not swallow a genuinely broken record."""
    wrong = _shorten_to_tile(clean_record, tmp_path / "wrong.bin", {"avt"},
                             ni=NTEI - NTSI, nj=NTEJ - NTSJ)
    with pytest.raises(R.RecordError, match="neither a known array header"):
        R.read_trazdf_matrix(wrong)


def test_the_real_round35_record_needed_the_salvage(real_domain):
    """Rule 10: say it about the record that exists, not only a fixture."""
    record = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                  "round35_oracle_trazdf_matrix/"
                  "oracle_trazdf_matrix_kt00000001.bin")
    if not record.exists():
        pytest.skip("round-35 record absent")
    rec = R.read_trazdf_matrix(record)
    h = rec["header"]
    assert set(rec["tile_shaped_salvage"]) == {"avt", "avs"}
    for name, row in rec["tile_shaped_salvage"].items():
        assert row["declared"] == [h["jpi"], h["jpj"], h["jpk"]]
        assert row["written"] == [h["ntei"] - h["ntsi"] + 1,
                                  h["ntej"] - h["ntsj"] + 1, h["jpk"]]


def test_a_cropped_domain_is_refused_before_it_can_score_thirty_cells(
        clean_record, real_domain):
    """The defeat an independent attack on round 36 actually landed.

    Cropping the record to a 5x5 domain with a SYMMETRIC halo of 2 satisfied
    every check the reader had -- the halo guard constrains the halo, not the
    domain, and jpi/jpj/jpk come from the record's own header -- so the gate
    scored ONE column, 30 cells instead of 21120, and printed exit 0 with
    STATUS AT-BAR while the real record is DEBT.  Every plant still reported
    ``landed=True``, so no control could tell the difference.

    The fixture domain is deliberately NOT GYRE's, which is the point: this
    gate scores one card's geometry and must refuse any other.
    """
    with pytest.raises(R.RecordError, match="is not GYRE-zco's"):
        R.read_trazdf_matrix(clean_record)
    assert (JPI, JPJ, JPK) != real_domain


def test_the_real_record_is_the_pinned_domain(real_domain):
    record = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                  "round35_oracle_trazdf_matrix/"
                  "oracle_trazdf_matrix_kt00000001.bin")
    if not record.exists():
        pytest.skip("round-35 record absent")
    h = R.read_trazdf_matrix(record)["header"]
    assert (h["jpi"], h["jpj"], h["jpk"]) == real_domain
    assert (h["ntei"] - h["ntsi"] + 1) * (h["ntej"] - h["ntsj"] + 1) \
        * h["jpkm1"] == 21120
