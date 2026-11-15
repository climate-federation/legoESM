"""The production card's exit status is the driver's: a failed model run
must fail the SLURM job, so `afterok` dependents (leg 2, scoring, gates) never
run on a broken leg."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

CARD = (Path(__file__).resolve().parents[2] / "scripts" / "cluster"
        / "omip_nemo" / "orca1_faithful_card.sbatch")


WRITE_FINAL = 'echo leg > "$OUT/snapshot_final.npz"'


def _run_card(tmp_path, driver_status, leg="1", body=""):
    fake = tmp_path / "fakepy"
    fake.write_text(
        "#!/bin/bash\n"
        f'if [ "$1" = "-c" ]; then exec {sys.executable} "$@"; fi\n'
        f"{body}\necho fake driver; exit {driver_status}\n")
    fake.chmod(0o755)
    card = tmp_path / "card.sh"
    src = CARD.read_text().splitlines()
    src = [f"ROOT={tmp_path}; cd $ROOT || exit 1" if ln.startswith("ROOT=")
           else f"PY={fake}" if ln.startswith("PY=") else ln for ln in src]
    card.write_text("\n".join(src) + "\n")
    env = dict(os.environ, OUT=str(tmp_path / "out"), LEG=leg, DAYS="30")
    return subprocess.run(["bash", str(card)], capture_output=True, text=True,
                           env=env, cwd=tmp_path, timeout=60)


@pytest.mark.parametrize("status", [0, 3])
def test_card_exits_with_the_driver_status(tmp_path, status):
    r = _run_card(tmp_path, status, body=WRITE_FINAL)
    assert f"=== card exit: {status} ===" in r.stdout
    assert r.returncode == status


def test_an_unwritten_run_log_fails_the_card_even_when_the_driver_succeeds(tmp_path):
    (tmp_path / "out" / "run.log").mkdir(parents=True)   # tee cannot open it
    r = _run_card(tmp_path, 0)
    assert r.returncode != 0
    assert "=== card exit: 0 ===" not in r.stdout


def test_leg1_keeps_its_final_snapshot_under_its_day(tmp_path):
    r = _run_card(tmp_path, 0, leg="1", body=WRITE_FINAL)
    assert r.returncode == 0
    assert (tmp_path / "out" / "snapshot_day0015.npz").read_text() == "leg\n"


@pytest.mark.parametrize("leg, status", [("2", 0), ("1", 3)])
def test_no_boundary_copy_for_leg2_or_a_failed_leg(tmp_path, leg, status):
    if leg == "2":
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "restart_leg1.npz").write_text("")
    _run_card(tmp_path, status, leg=leg, body=WRITE_FINAL)
    assert not (tmp_path / "out" / "snapshot_day0015.npz").exists()


def test_a_leg1_without_a_final_snapshot_fails(tmp_path):
    assert _run_card(tmp_path, 0, leg="1").returncode != 0
