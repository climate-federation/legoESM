"""Controls for the G5-certified tracer-surface statements imported by ORCA2 round 304."""

from pathlib import Path

import pytest

FIELDS = ("rk3_stage2_tracer_nemo_stored_thickness",
          "rk3_stage3_salt_nemo_written")
DECK = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")


def test_g5_import_defaults_are_fail_closed():
    from legoesm.ocean.state import LatLonCGridOceanConfig, OceanSurfaceForcing

    assert all(LatLonCGridOceanConfig._field_defaults[name] is False
               for name in FIELDS)
    assert OceanSurfaceForcing._field_defaults["salt_flux_sfx"] is None
    assert "implicit_ice_ocean_top_drag" not in LatLonCGridOceanConfig._fields


def test_orca2_selects_g5_set_and_unforced_rung_scopes_salt_off():
    pytest.importorskip("netCDF4")
    if not DECK.is_dir():
        pytest.skip("ORCA2 input deck is absent")
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
        nemo_testcase_l4_orca2_round304_g5_d122_gate as gate,
    )

    shipped = build_orca2_zps_card(DECK)
    assert all(getattr(shipped.recipe.model_config, name) for name in FIELDS)
    unforced = rung0.build_rung0_card(DECK).recipe.model_config
    assert unforced.rk3_stage2_tracer_nemo_stored_thickness
    assert not unforced.rk3_stage3_salt_nemo_written
    control = gate.selected_card(shipped, candidate=False, rung="rung10")
    assert not any(getattr(control.recipe.model_config, n) for n in FIELDS)
    assert gate.selected_card(shipped, candidate=True, rung="rung10") is shipped
    with pytest.raises(gate.GateError, match="control retains"):
        gate.selected_card(shipped, candidate=False, rung="rung10",
                           plant="retained-selector")
    with pytest.raises(gate.GateError, match="scope moved"):
        gate.selected_card(shipped, candidate=True, rung="rung0")
