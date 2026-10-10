from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round235_bt_alpha_replay as gate,
)


def test_resolved_filter_ignores_commented_assignments(tmp_path: Path):
    path = tmp_path / "namelist_cfg"
    path.write_text(
        "! nn_bt_flt = 1\n"
        "! rn_bt_alpha = 0.\n"
        "nn_bt_flt = 3\n"
        "rn_bt_alpha = 0.09 ! executing value\n",
        encoding="utf-8",
    )
    assert gate.resolved_filter(path) == (3, 0.09)


def test_resolved_filter_requires_both_fields(tmp_path: Path):
    path = tmp_path / "namelist_cfg"
    path.write_text("nn_bt_flt = 3\n", encoding="utf-8")
    with pytest.raises(gate.GateError):
        gate.resolved_filter(path)


def test_ulp_gap_is_exact():
    assert gate._ulp_gap(1.0, 1.0) == 0
    import numpy as np
    assert gate._ulp_gap(1.0, np.nextafter(1.0, np.inf)) == 1
