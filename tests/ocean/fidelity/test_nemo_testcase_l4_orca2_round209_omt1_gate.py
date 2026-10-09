from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round209_omt1_deck_gate as deck_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round209_omt1_frame_record_gate as record_gate,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round209_omt1_ladder_gate as ladder_gate,
)


SOURCE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
    "round203/acquisition/omt0_namelist_cfg"
)
DECK_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0"
)


def test_omt1_deck_is_exactly_one_module_edge(tmp_path: Path) -> None:
    candidate = tmp_path / "namelist_cfg"
    candidate.write_text(deck_gate.render_omt1(SOURCE.read_text()))

    report = deck_gate.validate_deck(SOURCE, candidate)

    assert report["changed_assignments"] == sorted(deck_gate.CHANGED)
    assert set(report["changed_assignments"]) == {
        "namdyn_adv.ln_dynadv_off", "namdyn_adv.ln_dynadv_vec",
    }


@pytest.mark.parametrize("plant", deck_gate.PLANTS[1:])
def test_deck_plants_fire(tmp_path: Path, plant: str) -> None:
    candidate = tmp_path / "namelist_cfg"
    candidate.write_text(deck_gate.render_omt1(SOURCE.read_text()))
    with pytest.raises(deck_gate.GateError):
        deck_gate.validate_deck(SOURCE, candidate, plant)


def test_record_preflight_reuses_admitted_writer() -> None:
    report = record_gate.preflight()
    assert report["status"] == "PASS_R211_OMT1_FRAME_PREFLIGHT"
    assert report["expected_frames_per_twin"] == 64
    assert report["writer_preflight"]["removed_source_lines"] == 0


def test_missing_frame_plant_is_nonvacuous(tmp_path: Path) -> None:
    for name in record_gate._wanted_frames():
        (tmp_path / name).touch()
    with pytest.raises(record_gate.GateError, match="frame inventory mismatch"):
        record_gate._validate_inventory(tmp_path, "missing-frame")


def test_wrong_boundary_plant_refuses_moved_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "nemo").write_bytes(b"binary")
    (tmp_path / "run.user.stdout.log").write_text(
        "MPI_ABORT was invoked\nErrorcode: 123\n"
    )
    (tmp_path / "ocean.output").write_text(
        "stp_ctl: |ssh| > 20 m  or  |U| > 10 m/s\n kt 9 |V| max 10.13\n"
    )
    (tmp_path / "output.abort_0000.nc").touch()
    monkeypatch.setattr(record_gate, "sha256", lambda path: "binary-sha")
    monkeypatch.setattr(
        record_gate.omt1_gate, "validate_run_deck", lambda *args, **kwargs: {},
    )
    with pytest.raises(record_gate.GateError, match="boundary moved"):
        record_gate._stability_boundary(
            tmp_path, tmp_path / "canonical", "binary-sha", "wrong-boundary",
        )


def test_omt1_card_restores_only_vector_module() -> None:
    omt0 = ladder_gate.omt0.build_omt0_card(DECK_ROOT)
    omt1 = ladder_gate.build_omt1_card(DECK_ROOT)
    selectors = ladder_gate.validate_omt1_card(omt1)
    before = omt0.recipe.model_config
    after = omt1.recipe.model_config
    moved = {
        name for name in before._fields
        if repr(getattr(before, name)) != repr(getattr(after, name))
    }
    assert all(selectors.values())
    assert moved == {
        "momentum_advection", "momentum_flux_scheme",
        "vertical_momentum_scheme", "vorticity_scheme",
    }


def test_card_module_plant_fires() -> None:
    card = ladder_gate.build_omt1_card(DECK_ROOT, plant="card-module")
    with pytest.raises(ladder_gate.GateError):
        ladder_gate.validate_omt1_card(card)
