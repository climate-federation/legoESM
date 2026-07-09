"""Blow-up forensics: the failing state must be WRITTEN (blowup_state_day_*.npz),
never left as a resumable checkpoint, and a dump failure must never mask the
BLOWUP status (fail-open).  Before this, every blowup discarded the sick state
("caught at checkpoint; not written") — the #871 hunt had nothing to autopsy.

save_checkpoint branches use TWO filename conventions (codex critical): MPAS
writes the ABSOLUTE rounded day, the generic branch writes the ELAPSED
truncated day — the helper must protect and detect BOTH.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from legoesm.driver.model_driver import ModelDriver


def _stub(tmp_path, save_impl, start_day=0.0):
    s = SimpleNamespace(_output_dir=Path(tmp_path), save_checkpoint=save_impl,
                        config=SimpleNamespace(start_day=start_day))
    s._write_blowup_state = ModelDriver._write_blowup_state.__get__(s)
    return s


def test_blowup_state_written_and_renamed_absolute_convention(tmp_path):
    def fake_save(step, day):  # MPAS convention: absolute rounded day
        (tmp_path / f"checkpoint_day_{int(round(day)):04d}.npz").write_bytes(b"x")
        (tmp_path / f"checkpoint_day_{int(round(day)):04d}.meta.json").write_text("{}")

    _stub(tmp_path, fake_save)._write_blowup_state(100, 20.0)
    assert (tmp_path / "blowup_state_day_0020.npz").exists()
    assert (tmp_path / "blowup_state_day_0020.meta.json").exists()
    # the poisoned state must NOT remain under the restart-chain glob
    assert not (tmp_path / "checkpoint_day_0020.npz").exists()


def test_blowup_state_written_elapsed_convention(tmp_path):
    def fake_save(step, day):  # generic convention: int(day - start_day)
        d = int(day - 5.0)
        (tmp_path / f"checkpoint_day_{d:04d}.npz").write_bytes(b"x")

    # day 15.6, start_day 5 -> generic writes 0010; absolute candidate is 0016
    _stub(tmp_path, fake_save, start_day=5.0)._write_blowup_state(100, 15.6)
    assert (tmp_path / "blowup_state_day_0010.npz").exists()
    assert not (tmp_path / "checkpoint_day_0010.npz").exists()


def test_dump_failure_is_nonfatal(tmp_path):
    def boom(step, day):
        raise RuntimeError("writer exploded")

    # must not raise — forensics never masks the BLOWUP status
    _stub(tmp_path, boom)._write_blowup_state(100, 20.0)


def test_missing_local_file_is_nonfatal(tmp_path):
    # distributed/custom writers may not produce the local filename
    _stub(tmp_path, lambda step, day: None)._write_blowup_state(100, 20.0)
    assert not (tmp_path / "blowup_state_day_0020.npz").exists()


def test_healthy_same_day_checkpoint_survives(tmp_path):
    """A daily-print blowup at day N.x rounds onto day N whose HEALTHY periodic
    checkpoint already exists — the dump must not clobber it."""
    healthy = tmp_path / "checkpoint_day_0020.npz"
    healthy.write_bytes(b"HEALTHY")

    def fake_save(step, day):
        (tmp_path / f"checkpoint_day_{int(round(day)):04d}.npz").write_bytes(b"SICK")

    _stub(tmp_path, fake_save)._write_blowup_state(100, 20.3)
    assert (tmp_path / "blowup_state_day_0020.npz").read_bytes() == b"SICK"
    assert healthy.read_bytes() == b"HEALTHY"  # restored, resumable chain intact


def test_directory_form_checkpoint_renamed(tmp_path):
    """Distributed writers produce a checkpoint DIRECTORY (checkpoint_day_NNNN/)
    — the poisoned dir must also be moved out of the restart glob (codex r3)."""
    def fake_save(step, day):
        (tmp_path / f"checkpoint_day_{int(round(day)):04d}").mkdir()

    _stub(tmp_path, fake_save)._write_blowup_state(100, 20.0)
    assert (tmp_path / "blowup_state_day_0020").is_dir()
    assert not (tmp_path / "checkpoint_day_0020").exists()


def test_healthy_checkpoint_restored_even_when_save_raises(tmp_path):
    """codex HIGH: the healthy backup must be restored on the FAILURE path too
    (restore lives in finally), or a failed dump hides the resumable file."""
    healthy = tmp_path / "checkpoint_day_0020.npz"
    healthy.write_bytes(b"HEALTHY")

    def boom(step, day):
        raise RuntimeError("writer exploded mid-dump")

    _stub(tmp_path, boom)._write_blowup_state(100, 20.0)
    assert healthy.exists() and healthy.read_bytes() == b"HEALTHY"
    assert not (tmp_path / "checkpoint_day_0020.npz.pre_blowup").exists()
