"""The kt<=2 acquisition, the copy rule it exists to obey, and two ports.

Four properties, each tested by a SYNTHETIC VIOLATION rather than by a happy
path, because every one of them can fail silently:

1. NO ACQUISITION SCRIPT MAY COPY ``cfgs/DINO`` WHOLESALE.  That directory is
   60 GB, essentially all of it this campaign's ``RUN_*`` output, and the
   oracle tree sits in a home directory at its quota -- a ``cp -a`` there
   fills the disk and the acquisition never runs.  Every script must build
   its config copy with ``makenemo -r DINO -n <COPY>`` (a ~137 MB skeleton)
   and must refuse a copy that turns out to contain a ``RUN_*`` directory.

2. THE kt<=2 PATCH MUST PUT ``kt`` IN EVERY FILENAME.  Without it the kt=2
   streams overwrite their kt=1 twins and the record is one step wearing the
   name of two.

3. THE PATCH MUST NOT REUSE A FORTRAN UNIT THE SOURCE ALREADY OPENS.  Two
   ``OPEN``s on one unit is not a compile error; the second silently closes
   the first, and the record loses the arrays it exists to capture.

4. THE TWO PORTS OF NEMO ROUTINES MUST REPRODUCE A NUMBER THE RECORD STATES
   OUT LOUD before either is used: ``ts_wgt`` must give ``icycle = 45`` for
   the Euler first step (ocean.output:1265), and the ladder's
   half-step-back-interpolation calibration must be able to FAIL.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import re

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_D1226 = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "scripts", "validate", "ocean_fidelity",
    "dino_1226"))
_ORACLE_SRC = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
               "MY_SRC/dynspg_ts.F90")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- rule 1
def _acquisition_scripts():
    return sorted(glob.glob(os.path.join(_D1226, "nemo_dino_*", "run.sh")))


def test_there_are_acquisition_scripts_to_check():
    # A glob that matches nothing would make every test below vacuous.
    assert len(_acquisition_scripts()) >= 3


@pytest.mark.parametrize("path", _acquisition_scripts(),
                         ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_no_acquisition_script_copies_the_oracle_config_wholesale(path):
    body = "".join(ln.split("#", 1)[0] for ln in open(path))
    bad = re.findall(r"(?:cp|rsync)\s+[^\n|;]*cfgs/DINO(?:\s|\"|$)", body)
    assert not bad, (
        f"{path} copies cfgs/DINO wholesale ({bad}); that is 60 GB of RUN_* "
        "output and it fills the home quota. Build the copy with "
        "`makenemo -r DINO -n <COPY>`.")


@pytest.mark.parametrize("path", _acquisition_scripts(),
                         ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_every_acquisition_script_builds_a_makenemo_skeleton(path):
    body = open(path).read()
    assert "./makenemo -r DINO -n" in body, (
        f"{path} does not build its config copy with makenemo -r DINO")


#: Scripts that ALSO carry the runtime refusal -- a check on the copy that
#: actually exists, so an edit reintroducing a wholesale copy fails there
#: instead of on the disk.  GROW-ONLY: a script may join this list, never
#: leave it.  It is not yet every script because the rest have never had a
#: wholesale copy in them, and widening the diff to five untouched files buys
#: nothing the test above does not already enforce.  Named as debt rather
#: than left implicit.
_RUNTIME_REFUSAL_REQUIRED = ("nemo_dino_zdf_matrix", "nemo_dino_kt2_rankdump")


@pytest.mark.parametrize("name", _RUNTIME_REFUSAL_REQUIRED)
def test_the_fixed_scripts_refuse_a_copy_containing_a_run_directory(name):
    path = os.path.join(_D1226, name, "run.sh")
    body = open(path).read()
    assert "-name 'RUN_*'" in body and "REFUS" in body, (
        f"{path} lost its runtime refusal for a copy that contains a RUN_* "
        "directory; the copy rule would then live only in the comment")


def test_the_wholesale_copy_detector_can_fail(tmp_path):
    # Synthetic violation: the exact line that was in nemo_dino_zdf_matrix.
    p = tmp_path / "run.sh"
    p.write_text('#!/bin/bash\ncp -a "$NEMO/cfgs/DINO" "$COPY"\n')
    body = "".join(ln.split("#", 1)[0] for ln in open(p))
    assert re.findall(r"(?:cp|rsync)\s+[^\n|;]*cfgs/DINO(?:\s|\"|$)", body)


def test_a_comment_mentioning_the_old_copy_is_not_a_violation(tmp_path):
    # The fixed script DESCRIBES the defect it removed; that must not trip.
    p = tmp_path / "run.sh"
    p.write_text('#!/bin/bash\n# used to be cp -a $NEMO/cfgs/DINO $COPY\n'
                 './makenemo -r DINO -n "$CFGNAME" -m "$ARCH" -j 0\n')
    body = "".join(ln.split("#", 1)[0] for ln in open(p))
    assert not re.findall(r"(?:cp|rsync)\s+[^\n|;]*cfgs/DINO(?:\s|\"|$)", body)


# ---------------------------------------------------------------- rules 2, 3
_PATCH = os.path.join(_D1226, "nemo_dino_kt2_rankdump", "kt2_rankdump_patch.py")
_needs_oracle_src = pytest.mark.skipif(
    not os.path.exists(_ORACLE_SRC), reason=f"no oracle source {_ORACLE_SRC}")


@_needs_oracle_src
def test_kt2_patch_tags_every_stream_by_rank_and_step(tmp_path):
    m = _load(_PATCH, "kt2_rankdump_patch")
    dst = tmp_path / "dynspg_ts.F90"
    dst.write_text(open(_ORACLE_SRC).read())
    m.patch(str(dst))
    out = dst.read_text()
    # the per-substep stream carries rank AND step AND substep
    assert "'_kt', kt, '_s', jn, '.bin'" in out
    # the once-per-step streams carry rank AND step
    assert "'_r', narea-1, '_kt', kt, '.bin'" in out
    # and the guard fires at kt=2, not only at kt=1
    assert "ll_spg_dump = ( kt <= nit000 + 1 )" in out
    # both sub-state dumps landed, with the six arrays each
    for tag in ("start", "end"):
        assert f"'substate_{tag}_r'" in out
    for nm in m.SUBSTATE:
        assert out.count("WRITE(93") and f"{nm}(:,:)" in out
    # WRITE-only
    assert "ACTION='READ'" not in out and "ACTION='READWRITE'" not in out


@_needs_oracle_src
def test_kt2_patch_refuses_to_double_apply(tmp_path):
    m = _load(_PATCH, "kt2_rankdump_patch")
    dst = tmp_path / "dynspg_ts.F90"
    dst.write_text(open(_ORACLE_SRC).read())
    m.patch(str(dst))
    with pytest.raises(SystemExit, match="already patched"):
        m.patch(str(dst))


@_needs_oracle_src
def test_kt2_patch_never_reuses_an_open_fortran_unit(tmp_path):
    m = _load(_PATCH, "kt2_rankdump_patch")
    dst = tmp_path / "dynspg_ts.F90"
    dst.write_text(open(_ORACLE_SRC).read())
    m.patch(str(dst))
    units = re.findall(r"UNIT=(\d+)", dst.read_text())
    assert len(units) == len(set(units)), (
        f"a unit is opened twice: {sorted(units)}")


def test_the_unit_collision_detector_can_fail():
    # Synthetic violation: the units the first draft of the patch reached for.
    m = _load(_PATCH, "kt2_rankdump_patch")
    assert m._unit("OPEN( UNIT=9301, ...)", 9302) == 9302
    with pytest.raises(SystemExit, match="already opened"):
        m._unit("OPEN( UNIT=9301, ...)", 9301)


def test_kt2_patch_refuses_the_pristine_oracle_tree(tmp_path):
    m = _load(_PATCH, "kt2_rankdump_patch")
    if not os.path.exists(_ORACLE_SRC):
        pytest.skip("no oracle source")
    with pytest.raises(SystemExit, match="read-only oracle"):
        m.patch(_ORACLE_SRC)


# ---------------------------------------------------------------- rule 4
def test_ts_wgt_port_reproduces_the_records_own_icycle():
    g = _load(os.path.join(_D1226, "kt2_leapfrog_gate.py"), "kt2_gate")
    # kt=1, Euler start: ll_fw_start = .TRUE. -> the record says icycle = 45
    k1, w1, _ = g.nemo_ts_wgt(ll_fw=True)
    assert k1 == 45, "the port disagrees with ocean.output:1265"
    assert all(w == 1.0 for w in w1) and len(w1) == 45
    # kt=2, ln_bt_fw = .FALSE.: ll_fw_start reset -> centre moves to 2*nn_e
    k2, w2, _ = g.nemo_ts_wgt(ll_fw=False)
    assert k2 == 68
    assert [i + 1 for i, w in enumerate(w2) if w != 0.0] == list(range(24, 69))


def test_ts_wgt_port_would_notice_a_changed_nn_e():
    # Non-vacuity: the port must be a function of nn_e, not a constant 45/68.
    g = _load(os.path.join(_D1226, "kt2_leapfrog_gate.py"), "kt2_gate")
    assert g.nemo_ts_wgt(ll_fw=True, nn_e=10)[0] == 19
    assert g.nemo_ts_wgt(ll_fw=False, nn_e=10)[0] == 29
    with pytest.raises(SystemExit):
        g.nemo_ts_wgt(ll_fw=True, nn_bt_flt=3)


def _fake_record(n=4, shape=(3, 5), rng=None):
    """A synthetic NEMO substep record that satisfies NEMO's own statements."""
    rng = rng or np.random.default_rng(0)
    lad = _load(os.path.join(
        _D1226, "spg_kt1_barotropic_ladder.py"), "spg_lad")
    subs = []
    sshn = np.zeros(shape)
    un = np.zeros(shape)
    vn = np.zeros(shape)
    hist = [np.zeros(shape), np.zeros(shape)]     # sshb_e, sshbb_e
    for i in range(n):
        ssha = rng.normal(size=shape)
        ua = rng.normal(size=shape)
        va = rng.normal(size=shape)
        za = lad._BCK.get(i + 1, lad._BCK_AB3AM4)
        zp2 = za[0] * ssha + za[1] * sshn + za[2] * hist[0] + za[3] * hist[1]
        subs.append({"sshn_e": sshn, "ssha_e": ssha, "zsshp2_e": zp2,
                  "un_e": un, "vn_e": vn, "ua_e": ua, "va_e": va})
        hist = [sshn, hist[0]]
        sshn, un, vn = ssha, ua, va
    return lad, subs


def test_ladder_calibration_passes_on_a_record_that_obeys_nemo():
    lad, subs = _fake_record()
    m = np.ones((3, 5), bool)
    tab = lad.Row()
    assert lad.nemo_self_calibration(subs, m, m, m, tab) is True
    assert not tab.fail


@pytest.mark.parametrize("break_it", ["swap", "bck"])
def test_ladder_calibration_can_fail(break_it):
    lad, subs = _fake_record()
    m = np.ones((3, 5), bool)
    if break_it == "swap":
        subs[2] = dict(subs[2], sshn_e=subs[2]["sshn_e"] + 1.0)
    else:
        subs[2] = dict(subs[2], zsshp2_e=subs[2]["zsshp2_e"] + 1.0)
    tab = lad.Row()
    assert lad.nemo_self_calibration(subs, m, m, m, tab) is False
    assert tab.fail
    assert tab.rows[-1]["status"] == "UNMEASURED"


def test_ladder_substep_rows_count_a_nan_as_unequal():
    # A row that filtered non-finite cells OUT would report AT BAR on an
    # operator that returns NaN exactly where it disagrees.
    lad, subs = _fake_record(n=1, shape=(3, 5))
    tr = {k: np.zeros((1, 3, 6)) if k.startswith(("u_", "flux_u", "pgf_u",
                                                  "slow_u", "drag_u"))
          else np.zeros((1, 4, 5)) if k.startswith(("v_", "flux_v", "pgf_v",
                                                   "slow_v", "drag_v"))
          else np.zeros((1, 3, 5)) for k, _ in
          [(n, 0) for n in ("eta_entry", "u_entry", "v_entry", "eta_pgf",
                            "eta_exit", "u_exit", "v_exit")]}
    subs[0] = {k: np.zeros((3, 5)) for k in subs[0]}
    tr["eta_exit"] = tr["eta_exit"].at[0, 0, 0].set(np.nan) if hasattr(
        tr["eta_exit"], "at") else tr["eta_exit"]
    tr["eta_exit"][0, 0, 0] = np.nan
    m = np.ones((3, 5), bool)
    per, first, n = lad.substep_ladder(subs, tr, 1, m, m, m)
    assert per[0]["eta_exit"][0] == 1
    assert first == (1, "eta_exit")
