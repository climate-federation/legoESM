"""Controls for Decision 83's ORCA2 hierarchy rung-1 replacement."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[3]
    / "scripts/validate/ocean_fidelity/orca2_l4"
    / "nemo_testcase_l4_orca2_hier_decks_round18_gate.py"
)
RUNNER = SCRIPT.parent / "nemo_testcase_l4_orca2_hier_decks_round18_acquisition/run.sh"
SPEC = importlib.util.spec_from_file_location("orca2_hier_decks_round18_gate", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_preflight_is_one_line_replacement_and_preserves_bbl_bbc_boundary():
    report = gate.preflight()
    assert report["status"] == "PREFLIGHT_PASS_RUNG1_HAVTB0"
    assert report["replacement_delta"] == {"namzdf.nn_havtb": ["1", "0"]}
    assert report["line_delta"] == [[419, gate.OLD_LINE, gate.NEW_LINE]]
    assert report["rung2_boundary_delta"] == {
        "nambbc.ln_trabbc": [".true.", ".false."],
        "nambbl.ln_trabbl": [".true.", ".false."],
    }


def test_main_rung0_boundary_is_damping_plus_protocol_only():
    diff = gate.preflight()["main_rung0_semantic_differences"]
    assert set(diff) == {
        "namrun.ln_rst_list",
        "namrun.nn_itend",
        "namrun.nn_stock",
        "namrun.nn_stocklist",
        "namsbc_flx.sn_emp",
        "namsbc_flx.sn_qsr",
        "namsbc_flx.sn_qtot",
        "namsbc_flx.sn_utau",
        "namsbc_flx.sn_vtau",
        "namtra_dmp.ln_tradmp",
        "namtsd.ln_tsd_dmp",
    }


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
        "validate_bbl_bbc_resolved",
        lambda root, plant="none": {"status": "PASS_RUNG1_BBL_BBC_RESOLVED", "plant": plant},
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
        return {"status": "PASS_RUNG2_HAVTB0_RESOLVED"}

    monkeypatch.setattr(gate.upper, "validate_resolved", upper)
    assert gate.validate_resolved(tmp_path)["status"] == "PASS_RUNG1_HAVTB0_RESOLVED"
    with pytest.raises(gate.upper.GateError):
        gate.validate_resolved(tmp_path, plant="resolved-havtb")
    assert (
        gate.validate_resolved(tmp_path, plant="bbl-bbc-consequence")["bbl_bbc"]["plant"]
        == "bbl-bbc-consequence"
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
