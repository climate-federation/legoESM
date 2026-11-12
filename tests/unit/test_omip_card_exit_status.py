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


def _run_card(tmp_path, driver_status):
    fake = tmp_path / "fakepy"
    fake.write_text(
        "#!/bin/bash\n"
        f'if [ "$1" = "-c" ]; then exec {sys.executable} "$@"; fi\n'
        f"echo fake driver; exit {driver_status}\n")
    fake.chmod(0o755)
    card = tmp_path / "card.sh"
    src = CARD.read_text().splitlines()
    src = [f"ROOT={tmp_path}; cd $ROOT || exit 1" if ln.startswith("ROOT=")
           else f"PY={fake}" if ln.startswith("PY=") else ln for ln in src]
    card.write_text("\n".join(src) + "\n")
    env = dict(os.environ, OUT=str(tmp_path / "out"), LEG="1", DAYS="30")
    return subprocess.run(["bash", str(card)], capture_output=True, text=True,
                           env=env, cwd=tmp_path, timeout=60)


@pytest.mark.parametrize("status", [0, 3])
def test_card_exits_with_the_driver_status(tmp_path, status):
    r = _run_card(tmp_path, status)
    assert f"=== card exit: {status} ===" in r.stdout
    assert r.returncode == status
