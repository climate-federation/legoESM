"""The ocean free-convection case runs through the SCM matrix runner (#2103).

Runs ``run_scm_test_matrix.py ocean_convective_deepening`` in-process: the
case passes its analytic oracle (heat conservation + sqrt(t) deepening).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def test_ocean_convective_deepening_case_passes():
    sys.path.insert(0, str(_SCRIPTS / "matrix"))
    import run_scm_test_matrix as S  # noqa: PLC0415
    assert S.main(["ocean_convective_deepening"]) == 0
