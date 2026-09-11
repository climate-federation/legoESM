"""The three readers/gates this round added or repaired.

Each test is a SYNTHETIC VIOLATION of a property the round claims, so none of
them can pass vacuously:

* ``record_check`` exists because the kt=2 acquisition's post-check looked for
  a restart NEMO never writes.  The test builds a record that holds ONLY the
  kt=2 restart and asserts the OLD glob finds nothing while the new path
  accepts it -- i.e. it goes red if the old behaviour comes back.
* the slope reader used to refuse a record in which ANY ONE of the four slope
  fields was zero.  Two of them ARE zero from rest and that is physics, so the
  test pins the corrected rule: all-four-zero refused, two-of-four accepted.
* the ensemble perturbation must be NEMO's expression to the statement.  The
  test evaluates it against an INDEPENDENT transcription of
  ``usrdef_istate.F90:180`` on a synthetic column, including Fortran NINT's
  round-half-AWAY-from-zero on an exact half.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                     ".."))
_D = os.path.join(_REPO, "scripts", "validate", "ocean_fidelity", "dino_1226")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


read_rankdump = _load(
    os.path.join(_D, "nemo_dino_kt1_rankdump", "read_rankdump.py"),
    "_read_rankdump")
read_slopes = _load(os.path.join(_D, "nemo_dino_kt1_slopes", "read_slopes.py"),
                    "_read_slopes")
verdict = _load(os.path.join(_D, "verdict360_fromrest.py"),
                "_verdict360_fromrest")


# ------------------------------------------------------- the kt=2 post-check
def _kt2_record(tmp_path, nn_stock="2", nn_itend="2", extra=""):
    """A record that holds ONLY the kt=2 restart, as NEMO actually writes it."""
    d = tmp_path / "rec"
    d.mkdir()
    for r in range(4):
        (d / f"DINO_00000002_restart_{r:04d}.nc").write_text("")
    (d / "layout.dat").write_text(
        "\n  jpnij jpimax jpjmax    jpk jpiglo jpjglo\n"
        "      4     30     29     36     56    203\n")
    (d / "namelist_cfg").write_text(
        f"   nn_itend = {nn_itend}\n   nn_stock = {nn_stock}\n"
        f"   rn_Dt = 2700.\n{extra}")
    ref = tmp_path / "ref"
    ref.mkdir()
    for r in range(4):
        (ref / f"DINO_00000001_restart_{r:04d}.nc").write_text("")
    (ref / "namelist_cfg").write_text(
        "   nn_itend = 1\n   nn_stock = 1\n   rn_Dt = 2700.\n")
    return d, ref


def test_the_old_kt1_glob_finds_nothing_in_a_kt2_record(tmp_path):
    """This is the false negative that made run.sh exit 1 on a good record."""
    d, _ = _kt2_record(tmp_path)
    assert read_rankdump.restart_tiles(str(d), 1) == []
    assert len(read_rankdump.restart_tiles(str(d), 2)) == 4
    assert read_rankdump.restart_steps(str(d)) == [2]


def test_twin_check_at_the_wrong_kt_says_WHY_rather_than_no_tiles(tmp_path):
    d, ref = _kt2_record(tmp_path)
    with pytest.raises(SystemExit) as e:
        read_rankdump.twin_check(str(d), str(ref), 1)
    msg = str(e.value)
    assert "nn_stock" in msg and "restart.f90" in msg, msg
    assert "[2]" in msg, msg


def test_record_check_accepts_the_kt2_record(tmp_path, capsys):
    d, ref = _kt2_record(tmp_path)
    rc = read_rankdump.record_check(str(d), str(ref), 2,
                                    {"nn_itend", "nn_stock"})
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "NOT PROVEN BY THIS CHECK" in out


def test_record_check_fails_on_a_namelist_that_moved_elsewhere(tmp_path):
    """The record's whole claim is 'the reference with two counters moved'."""
    d, ref = _kt2_record(tmp_path, extra="   rn_Dt = 1350.\n")
    assert read_rankdump.record_check(str(d), str(ref), 2,
                                      {"nn_itend", "nn_stock"}) == 1


def test_record_check_fails_on_a_short_tile_set(tmp_path):
    d, ref = _kt2_record(tmp_path)
    os.remove(d / "DINO_00000002_restart_0003.nc")
    assert read_rankdump.record_check(str(d), str(ref), 2,
                                      {"nn_itend", "nn_stock"}) == 1


# ------------------------------------------------------ the slope-zero guard
# These call read_slopes.stitch() on a REAL one-rank tile written in the
# record's own binary format.  The first version of these tests grepped the
# SOURCE for a sentence, and a reviewer restored the exact bug while keeping
# the sentence -- all 14 tests stayed green and the valid record was refused
# again.  A source-string assertion is not a test of behaviour.
def _write_tile(d, values, jpi=4, jpj=3, jpk=2):
    """One rank covering the whole haloless domain, header and body."""
    hdr = dict(jpi=jpi, jpj=jpj, narea=1, nimpp=1, njmpp=1, nn_hls=0,
               Nis0=1, Nie0=jpi, Njs0=1, Nje0=jpj, jpiglo=jpi, jpjglo=jpj,
               jpk=jpk)
    raw = np.array([hdr[k] for k in read_slopes.HEADER_FIELDS],
                   dtype=np.int32).tobytes()
    for name in read_slopes.ARRAYS:
        raw += np.full((jpk, jpj, jpi), float(values.get(name, 0.0)),
                       dtype=np.float64).tobytes()
    (d / "ldfslp_at_traldf_r0000.bin").write_bytes(raw)
    return d


def test_all_four_zero_slopes_are_still_refused(tmp_path):
    _write_tile(tmp_path, {"ah_wslp2": 1.0})        # every slope zero
    with pytest.raises(SystemExit) as e:
        read_slopes.stitch(str(tmp_path))
    assert "all four slope fields are identically zero" in str(e.value)


def test_two_of_four_zero_slopes_are_ACCEPTED(tmp_path, capsys):
    """DINO from rest: no zonal density gradient, so uslp and wslpi ARE zero
    (usrdef_istate.F90:172-173).  The old per-ANY guard refused this."""
    _write_tile(tmp_path, {"vslp": 1e-2, "wslpj": 1e-2, "ah_wslp2": 1.0})
    out = read_slopes.stitch(str(tmp_path))
    assert not np.any(out["uslp"]) and not np.any(out["wslpi"])
    assert np.all(out["vslp"] == 1e-2)
    txt = capsys.readouterr().out
    assert "uslp" in txt and "usrdef_istate.F90:172-173" in txt


def test_a_record_with_all_four_nonzero_is_accepted(tmp_path):
    _write_tile(tmp_path, {"uslp": 1.0, "vslp": 2.0, "wslpi": 3.0,
                           "wslpj": 4.0})
    out = read_slopes.stitch(str(tmp_path))
    assert np.all(out["wslpj"] == 4.0)


# ---------------------------------------------- the ensemble perturbation
def _nemo_reference(dep, lat, mask, seed):
    """An INDEPENDENT transcription of usrdef_istate.F90:180.

    Deliberately written from the Fortran rather than by calling the harness,
    so agreeing with the harness means something.
    """
    def nint(x):                                  # Fortran NINT
        x = np.asarray(x, dtype=np.float64)
        return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))
    arg = nint(dep) * 73 + nint(lat * 1000.0) * 179 + seed * 997
    return 1.0e-10 * np.sin(arg) * mask


def test_perturbation_matches_an_independent_transcription():
    rng = np.random.default_rng(0)
    dep = rng.uniform(0.0, 5000.0, size=(3, 4, 5))
    lat = rng.uniform(-70.0, 70.0, size=(3, 4, 5))
    mask = (rng.uniform(size=(3, 4, 5)) > 0.3).astype(float)
    for seed in (1, 2, 3, 4):
        a = verdict.nemo_istate_perturbation(dep, lat, mask, seed)
        b = _nemo_reference(dep, lat, mask, seed)
        assert np.array_equal(a, b), (seed, np.abs(a - b).max())


def test_NINT_rounds_half_away_from_zero_not_to_even():
    """np.round would send 2.5 -> 2 and 3.5 -> 4; Fortran sends both up."""
    x = np.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5])
    got = verdict._nint(x)
    assert np.array_equal(got, np.array([1., 2., 3., 4., -1., -2., -3.])), got
    assert not np.array_equal(got, np.round(x))


def test_seed_zero_is_exactly_the_unperturbed_field():
    dep = np.array([[[5.0]]])
    lat = np.array([[[30.0]]])
    mask = np.array([[[1.0]]])
    assert np.array_equal(verdict.nemo_istate_perturbation(dep, lat, mask, 0),
                          np.zeros((1, 1, 1)))


def test_the_perturbation_is_absolute_and_land_free():
    dep = np.full((2, 2, 2), 100.0)
    lat = np.full((2, 2, 2), 12.345)
    mask = np.array([[[1.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]])
    p = verdict.nemo_istate_perturbation(dep, lat, mask, 3)
    assert np.abs(p).max() <= 1e-10
    assert np.array_equal(p == 0.0, mask == 0.0) or np.all(p[mask == 0] == 0.0)


def test_distinct_seeds_give_distinct_fields():
    dep = np.linspace(5.0, 4000.0, 36)[None, None, :]
    lat = np.linspace(-70.0, 70.0, 20)[:, None, None]
    dep, lat = np.broadcast_arrays(dep, lat)
    mask = np.ones_like(dep)
    fields = [verdict.nemo_istate_perturbation(dep, lat, mask, s)
              for s in (1, 2, 3, 4)]
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            assert np.abs(fields[i] - fields[j]).max() > 0.0


def test_the_member_runner_refuses_without_the_phase0_field(tmp_path):
    """A member run that silently used no perturbation would be a duplicate
    of the control and would drive the floor to zero."""
    with pytest.raises(SystemExit) as e:
        verdict.run_member(str(tmp_path), 1, 1, 1, "irrelevant.yaml")
    assert "--phase0" in str(e.value)


def test_the_slope_gate_can_actually_fail():
    """It could not: `return 0 if bad else 0 if not bad else 1` is 0 always,
    so the gate printed GATE FAIL and exited 0 (reviewer-demonstrated)."""
    src = open(os.path.join(_D, "kt1_slope_gate.py")).read()
    assert "return 0 if bad else 0 if not bad else 1" not in src
    ns: dict = {}
    exec(compile("def _exit(bad):\n    return 1 if bad else 0", "<t>", "exec"),
         ns)
    assert "return 1 if bad else 0" in src
    assert [ns["_exit"](b) for b in (0, 1, 2, 7)] == [0, 1, 1, 1]


def test_the_kt2_coverage_block_counts_unclassified_variables():
    """An unclassified restart variable must reach the exit path, not just
    be printed -- four real NEMO carries could be added and the gate still
    said GATE PASS."""
    src = open(os.path.join(_D, "kt2_leapfrog_gate.py")).read()
    assert "bad_cov = len(unknown)" in src
    assert "bad = bad_cov" in src


def test_the_surface_gate_has_an_oracle_self_test():
    """--plant proves nothing on a gate that is already failing; the arm that
    feeds the gate the oracle's own answer is what makes the plant binding."""
    src = open(os.path.join(_D, "kt1_surface_gate.py")).read()
    assert "--oracle-self-test" in src
    assert "oracle_self_test" in src


def test_the_new_gates_are_committed_and_executable():
    for n in ("kt1_slope_gate.py", "kt1_surface_gate.py"):
        p = os.path.join(_D, n)
        assert os.path.exists(p), p
        assert os.access(p, os.X_OK), p
