"""The oracle deck builders publish a family transactionally.

codex 2026-09-14 (two rounds): a failed control left decks carrying
PROVENANCE; then `rm -rf old && mv staging` deleted a deck before a failing
mv. The publish/rollback functions are extracted from the sbatch and run
in a sandbox with an injected mv failure."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_BUILDERS = [
    "build_ksplit_oracle.sbatch",
    "build_terminator_oracle.sbatch",
    "build_warm_tracer_oracle.sbatch",
]


def _functions(text: str) -> str:
    out = []
    for name in ("publish_all", "rollback", "unstage_all"):
        m = re.search(rf"^{name} \(\) \{{.*?^\}}$|^{name} \(\) \{{[^\n]*\}}$",
                      text, re.M | re.S)
        assert m, f"{name} not found"
        out.append(m.group(0))
    return "\n".join(out) + "\n"


@pytest.mark.parametrize("builder", _BUILDERS)
def test_publish_rolls_back_on_a_failed_move_and_leaves_nothing_behind(
        tmp_path, builder):
    text = (_ROOT / "scripts" / "cluster" / "fv3_native" / builder).read_text()
    (tmp_path / "funcs.sh").write_text(_functions(text))
    script = r'''
set -u
PIN=$(pwd)
source ./funcs.sh
mkdir -p "$PIN/A" "$PIN/A.staging.$$" "$PIN/B.staging.$$"
echo old > "$PIN/A/PROVENANCE"; echo new > "$PIN/A.staging.$$/PROVENANCE"; echo new > "$PIN/B.staging.$$/PROVENANCE"
STAGED=(A B)
mv () { case "$2" in */B) return 1;; esac; command mv "$@"; }
( publish_all ) >/dev/null 2>&1; rc=$?
echo "rc=$rc A=$(cat "$PIN/A/PROVENANCE") staging=$(ls -d "$PIN"/*.staging.* 2>/dev/null | wc -l) old=$(ls -d "$PIN"/*.old.* 2>/dev/null | wc -l) B=$([ -e "$PIN/B" ] && echo yes || echo no)"
unset -f mv
mkdir -p "$PIN/A.staging.$$" "$PIN/B.staging.$$"; echo new > "$PIN/A.staging.$$/PROVENANCE"; echo new > "$PIN/B.staging.$$/PROVENANCE"
( publish_all ) >/dev/null 2>&1; rc=$?
echo "rc=$rc A=$(cat "$PIN/A/PROVENANCE") B=$(cat "$PIN/B/PROVENANCE") left=$(ls -d "$PIN"/*.staging.* "$PIN"/*.old.* 2>/dev/null | wc -l)"
'''
    res = subprocess.run(["bash", "-c", script], cwd=tmp_path,
                         capture_output=True, text=True, timeout=30)
    lines = res.stdout.strip().splitlines()
    assert lines[0] == "rc=11 A=old staging=0 old=0 B=no", res.stdout + res.stderr
    assert lines[1] == "rc=0 A=new B=new left=0", res.stdout + res.stderr
