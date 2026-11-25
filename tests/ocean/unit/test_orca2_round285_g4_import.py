"""Controls for the G4-certified statements imported by ORCA2 round 285."""

from pathlib import Path

import pytest


FIELDS = (
    "barotropic_wind_nemo_stored_depth",
    "surface_stress_nemo_fold_vector_sign",
    "zdf_surface_stress_nemo_face_thickness",
    "rk3_stage_tracer_written_order",
    "rk3_stage3_heat_nemo_written",
    "rk3_stage3_tracer_nemo_stored_thickness",
)


def test_g4_import_defaults_are_fail_closed():
    from legoesm.ocean.state import LatLonCGridOceanConfig

    assert all(LatLonCGridOceanConfig._field_defaults[name] is False
               for name in FIELDS)


def test_orca2_selects_g4_set_and_unforced_rung_scopes_heat_unit_off():
    pytest.importorskip("netCDF4")
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
    )

    deck = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
    if not deck.is_dir():
        pytest.skip("ORCA2 input deck is absent")
    shipped = build_orca2_zps_card(deck).recipe.model_config
    assert all(getattr(shipped, name) for name in FIELDS)
    unforced = rung0.build_rung0_card(deck).recipe.model_config
    assert all(getattr(unforced, name) for name in FIELDS[:4])
    assert not unforced.rk3_stage3_heat_nemo_written
    assert not unforced.rk3_stage3_tracer_nemo_stored_thickness


def test_round285_arm_scope_and_plant():
    pytest.importorskip("netCDF4")
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round285_g4_d122_gate as gate,
    )

    deck = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
    if not deck.is_dir():
        pytest.skip("ORCA2 input deck is absent")
    card = build_orca2_zps_card(deck)
    control = gate.selected_card(card, candidate=False, rung="rung10")
    assert not any(getattr(control.recipe.model_config, name) for name in FIELDS)
    candidate = gate.selected_card(card, candidate=True, rung="rung10")
    assert all(getattr(candidate.recipe.model_config, name) for name in FIELDS)
    with pytest.raises(gate.GateError, match="scope moved"):
        gate.selected_card(card, candidate=False, rung="rung10",
                           plant="retained-selector")
