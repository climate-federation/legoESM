"""``scripts/data/relabel_les_reference.py`` edits archived data, so its
refusals are the feature.

Two production LES drivers stamped a run of their SECOND deck with the FIRST
deck's name. The frames are the right physics, and regenerating them costs
hours of GPU to change one string, so the label is corrected in place -- but
only against evidence that separates the confusable pair, and never silently.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "data" / "relabel_les_reference.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("_relabel", _SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules["_relabel"] = m
    spec.loader.exec_module(m)
    return m


def _make(tmp_path, *, case_stamp, hours, z_top, n_frames, name="ref"):
    d = tmp_path / name
    (d / "profiles").mkdir(parents=True)
    for i in range(n_frames):
        t = hours * (i + 1) / n_frames
        np.savez(
            d / "profiles" / f"prof_{i:03d}.npz",
            case=np.asarray(case_stamp),
            t_hours=np.asarray(t),
            z=np.linspace(5.0, z_top, 12),
            theta=np.linspace(288.0, 300.0, 12),
            qv=np.linspace(0.009, 0.001, 12),
        )
    return d


def test_astex_stamped_dycoms_is_relabelled(tmp_path, mod, capsys):
    """The real case: 6 h to 2000 m in 37 frames IS astex."""
    d = _make(tmp_path, case_stamp="dycoms", hours=6.0, z_top=1995.0,
              n_frames=37)
    assert mod.main([str(d), "astex", "--apply"]) == 0
    got = np.load(d / "profiles" / "prof_000.npz", allow_pickle=True)
    assert str(got["case"]) == "astex"
    record = json.loads((d / "RELABELLED.json").read_text())
    assert record["was"] == "dycoms" and record["now"] == "astex"
    assert record["evidence"]["hours"] == pytest.approx(6.0, abs=1e-6)


def test_a_real_dycoms_directory_is_refused_as_astex(tmp_path, mod):
    """THE failure this tool exists to prevent: 4 h to 1500 m is NOT astex,
    and relabelling it would put the wrong reference under the right name --
    exactly what the loader's guard is for."""
    d = _make(tmp_path, case_stamp="dycoms", hours=4.0, z_top=1496.0,
              n_frames=25)
    assert mod.main([str(d), "astex", "--apply"]) == 2
    assert str(np.load(d / "profiles" / "prof_000.npz",
                       allow_pickle=True)["case"]) == "dycoms"
    assert not (d / "RELABELLED.json").exists()
    assert not (tmp_path / "ref.pre-relabel").exists()


def test_a_cbl_directory_is_refused_as_wangara(tmp_path, mod):
    d = _make(tmp_path, case_stamp="cbl", hours=4.0, z_top=1600.0, n_frames=25)
    assert mod.main([str(d), "wangara", "--apply"]) == 2


def test_the_frame_count_alone_can_refuse(tmp_path, mod):
    """Right duration and depth, wrong number of frames -- a truncated or
    re-run directory, which the loader has its own reasons to distrust."""
    d = _make(tmp_path, case_stamp="dycoms", hours=6.0, z_top=1995.0,
              n_frames=30)
    assert mod.main([str(d), "astex", "--apply"]) == 2


def test_an_unregistered_case_cannot_be_relabelled(tmp_path, mod):
    d = _make(tmp_path, case_stamp="dycoms", hours=6.0, z_top=1995.0,
              n_frames=37)
    assert mod.main([str(d), "bomex", "--apply"]) == 2


def test_dry_run_is_the_default(tmp_path, mod):
    d = _make(tmp_path, case_stamp="dycoms", hours=6.0, z_top=1995.0,
              n_frames=37)
    assert mod.main([str(d), "astex"]) == 0
    assert str(np.load(d / "profiles" / "prof_000.npz",
                       allow_pickle=True)["case"]) == "dycoms"
    assert not (d / "RELABELLED.json").exists()


def test_nothing_but_the_label_changes(tmp_path, mod):
    d = _make(tmp_path, case_stamp="dycoms", hours=6.0, z_top=1995.0,
              n_frames=37)
    before = {k: np.asarray(v) for k, v in
              np.load(d / "profiles" / "prof_005.npz", allow_pickle=True).items()}
    assert mod.main([str(d), "astex", "--apply"]) == 0
    after = np.load(d / "profiles" / "prof_005.npz", allow_pickle=True)
    assert set(after.files) == set(before)
    for k in before:
        if k == "case":
            continue
        assert np.array_equal(before[k], np.asarray(after[k])), k


def test_an_already_correct_directory_is_a_no_op(tmp_path, mod):
    d = _make(tmp_path, case_stamp="astex", hours=6.0, z_top=1995.0,
              n_frames=37)
    assert mod.main([str(d), "astex", "--apply"]) == 0
    assert not (d / "RELABELLED.json").exists()
    assert not (tmp_path / "ref.pre-relabel").exists()


def test_an_existing_backup_is_never_overwritten(tmp_path, mod):
    """A second run must not clobber the good copy from the first."""
    d = _make(tmp_path, case_stamp="dycoms", hours=6.0, z_top=1995.0,
              n_frames=37)
    (tmp_path / "ref.pre-relabel").mkdir()
    with pytest.raises(SystemExit, match="backup"):
        mod.main([str(d), "astex", "--apply"])


def test_the_registered_signatures_separate_the_confusable_pairs(mod):
    """Non-vacuous: each case's own signature must FAIL its lookalike's."""
    for case, other in (("astex", "dycoms"), ("dycoms", "astex"),
                        ("wangara", "cbl"), ("cbl", "wangara")):
        want = mod._EXPECTED[other]
        evidence = {"hours": want["hours"], "z_top_m": want["z_top_m"],
                    "n_frames": want["n_frames"], "stamped": other}
        assert mod._check(evidence, case), (
            f"{other}'s signature is accepted as {case}; the two are not "
            "separated and the tool would relabel the wrong reference")
        assert not mod._check(evidence, other)
