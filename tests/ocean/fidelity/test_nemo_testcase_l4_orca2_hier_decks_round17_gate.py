"""Controls for Decision 83's ORCA2 hierarchy rung-2 replacement."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_hier_decks_round17_gate.py"
)
RUNNER = SCRIPT.parent / "nemo_testcase_l4_orca2_hier_decks_round17_acquisition/run.sh"
SPEC = importlib.util.spec_from_file_location("orca2_hier_decks_round17_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_preflight_is_one_line_replacement_and_preserves_gm_mle_boundary():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS_RUNG2_HAVTB0"
    assert report["replacement_delta"] == {"namzdf.nn_havtb": ["1", "0"]}
    assert report["line_delta"] == [[419, gate.OLD_LINE, gate.NEW_LINE]]
    old_admission = gate.json.loads(gate._old_paths()[2].read_text())
    assert report["rung3_boundary_delta"] == old_admission["deck_delta_from_rung3"]


@pytest.mark.parametrize("plant", gate.PRECHECK_PLANTS)
def test_preflight_plants_refuse(plant):
    with pytest.raises(gate.GateError):
        gate.preflight(plant=plant)


def test_staged_deck_is_exact_and_idempotent(tmp_path):
    gate.stage_deck(tmp_path)
    first = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    gate.stage_deck(tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == first
    assert gate.sha256(tmp_path / "namelist_cfg") == gate.NEW_CFG_SHA


def test_resolved_avoids_superseded_chain_and_routes_plants(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        gate.legacy,
        "validate_gm_mle_resolved",
        lambda root, plant="none": {"status": "PASS_RUNG2_GM_MLE_RESOLVED", "plant": plant},
    )
    monkeypatch.setattr(
        gate.legacy,
        "validate_resolved",
        lambda *args, **kwargs: pytest.fail("superseded inherited chain was called"),
    )

    def upper(root, plant="none"):
        seen.append(plant)
        if plant == "resolved-havtb":
            raise gate.upper.GateError("planted background mismatch")
        return {"status": "PASS_RUNG3_HAVTB0_RESOLVED"}

    monkeypatch.setattr(gate.upper, "validate_resolved", upper)
    assert gate.validate_resolved(tmp_path)["status"] == "PASS_RUNG2_HAVTB0_RESOLVED"
    with pytest.raises(gate.upper.GateError):
        gate.validate_resolved(tmp_path, plant="resolved-havtb")
    assert (
        gate.validate_resolved(tmp_path, plant="gm-mle-consequence")["gm_mle"]["plant"]
        == "gm-mle-consequence"
    )
    assert seen == ["none", "resolved-havtb", "none"]


def test_runner_is_fail_closed_and_preserves_superseded_record():
    runner = RUNNER.read_text()
    assert "makenemo" not in runner
    assert "/usr/bin/time" not in runner
    assert "REFUSE:" in runner
    assert "--admit-existing" in runner
    assert 'mv "$OLD_RECORD" "$SUPERSEDED_RECORD"' in runner
    assert 'mv "$OLD_ADMISSION" "$SUPERSEDED_ADMISSION"' in runner
    assert "sha256sum -c SHA256SUMS" in runner
