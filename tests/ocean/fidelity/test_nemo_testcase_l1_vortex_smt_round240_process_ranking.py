"""Non-vacuity and registry controls for the round-240 ranking probe."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = (Path(__file__).parents[3] / "scripts" / "validate" /
         "ocean_fidelity" / "testcases")
sys.path.insert(0, str(TOOLS))

import nemo_testcase_l1_vortex_smt_round240_process_ranking as probe  # noqa: E402


def _rows(value=2.552708052055443e-4):
    return {name: {"100": {"T_rms": value}} for name in probe.ARM_NAMES}


def test_ranking_refuses_a_missing_family():
    rows = _rows()
    rows.pop("bottom_drag_off")
    with pytest.raises(Exception, match="family registry differs"):
        probe.rank_rows(rows)


def test_effect_plant_crosses_the_hpg_floor(tmp_path: Path):
    path = tmp_path / "ranking.json"
    path.write_text(json.dumps({"arms": _rows()}))
    with pytest.raises(Exception, match="HPG floor control moved"):
        probe._plant(path, "effect")


def test_all_arm_config_diffs_are_exactly_registered():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    card = build_nemo_testcase_card(probe.CASE)
    for arm in probe.ARM_NAMES:
        _, _, rows = probe.build_arm(card, arm)
        expected = len(probe.EXPECTED_CONFIG_DIFFS[arm])
        expected += int(arm in ("hpg_source_order", "barotropic_replacement_off"))
        assert len(rows) == expected


def test_round210_card_runner_is_reused_not_copied():
    source = Path(probe.__file__).read_text()
    assert "from nemo_testcase_l1_vortex_round210_100day_comparison import" in source
    assert "run_lego_card" in source
    assert "for step in range" not in source
