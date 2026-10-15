"""Red-capable controls for the two G1b statements measured in round 282."""

from pathlib import Path

import numpy as np
import pytest


OMT4 = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
    "round222/acquisition/orca2_omt4_frames_10step_a_np2")


def _card():
    if not (OMT4 / "namelist_cfg").is_file():
        pytest.skip("admitted OMT-4 deck is absent")
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round269_g1_decision96_gate as prior,
    )
    return prior._selected_card("omt4", OMT4)


def test_g1b_orca2_selectors_default_off():
    from legoesm.ocean.state import BarotropicConfig, LatLonCGridOceanConfig

    assert LatLonCGridOceanConfig._field_defaults[
        "een_nemo_relative_vorticity"] is False
    assert BarotropicConfig._field_defaults[
        "barotropic_literal_fold_unit_vector_form"] is False


def test_round282_arms_change_exactly_the_two_registered_fields():
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round282_g1b_orca2_gate as gate,
    )

    base = _card()
    assert base.recipe.model_config.een_nemo_relative_vorticity is True
    assert (base.recipe.model_config.barotropic
            .barotropic_literal_fold_unit_vector_form) is False
    for arm, expected in gate.ARMS.items():
        selected = gate.arm_card(base, arm)
        cfg = selected.recipe.model_config
        actual = {
            name for name, value in (
                (gate.A, cfg.een_nemo_relative_vorticity),
                (gate.B,
                 cfg.barotropic.barotropic_literal_fold_unit_vector_form),
            ) if value
        }
        assert actual == set(expected)
    with pytest.raises(gate.GateError, match="outside A/B"):
        gate.arm_card(base, "npcrv_f_area", plant="extra-selector")


def test_npcrv_area_uses_raw_nemo_f_metrics_and_control_can_fail():
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        _nemo_f_vertex_area,
    )

    card = _card()
    grid = card.recipe.grid
    z_coord = card.recipe.z_coord
    actual = np.asarray(_nemo_f_vertex_area(grid, z_coord))
    raw = z_coord.nemo_een_barotropic
    expected = (np.asarray(raw.e1f, dtype=actual.dtype)
                * np.asarray(raw.e2f, dtype=actual.dtype))
    np.testing.assert_array_equal(actual[1:, 1:], expected)
    assert np.count_nonzero(np.asarray(grid.area_q)[1:, 1:] != expected) > 0


def test_vector_fold_unit_predicate_is_selected_only_by_its_arm():
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        nemo_literal_external_mode_active,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round282_g1b_orca2_gate as gate,
    )

    base = _card()
    assert not nemo_literal_external_mode_active(
        gate.arm_card(base, "control").recipe.model_config)
    assert nemo_literal_external_mode_active(
        gate.arm_card(base, "vector_fold_unit").recipe.model_config)
