"""Red-capable controls for the two G2 statements imported in round 276."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_g2_imported_selectors_default_off():
    from legoesm.ocean.state import LatLonCGridOceanConfig

    assert LatLonCGridOceanConfig._field_defaults[
        "zdf_tracer_nemo_rounded_thickness"] is False
    assert LatLonCGridOceanConfig._field_defaults[
        "barotropic_rhs_ascending_level_sum"] is False


def test_ascending_level_sum_is_written_order_and_control_can_fail():
    from legoesm.ocean.dynamics.barotropic_common import ascending_level_sum

    rng = np.random.default_rng(276)
    values = rng.standard_normal((400, 30))
    values *= np.exp(rng.uniform(-20.0, 0.0, values.shape))

    expected = np.zeros(values.shape[:-1], dtype=np.float64)
    for level in range(values.shape[-1]):
        expected = expected + values[..., level]

    ordered = np.asarray(jax.jit(ascending_level_sum)(jnp.asarray(values)))
    np.testing.assert_array_equal(ordered, expected)
    lanes = [values[:, level::4].sum(axis=1) for level in range(4)]
    four_lane = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
    assert np.count_nonzero(four_lane != expected) > 0


def test_round276_private_arms_select_exactly_one_registered_unit():
    from pathlib import Path

    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round269_g1_decision96_gate as prior,
        nemo_testcase_l4_orca2_round276_omt_source_census_gate as round276,
    )

    deck = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round211/acquisition/orca2_omt1_frames_8step_a_np2")
    if not (deck / "namelist_cfg").is_file():
        pytest.skip("admitted OMT-1 deck is absent")
    base = prior._selected_card("omt1", deck)
    for arm, expected in round276.ARMS.items():
        selected = round276.arm_card(base, arm)
        cfg = selected.recipe.model_config
        actual = {name for name in round276.MODEL if getattr(cfg, name)}
        actual |= {
            name for name in round276.BAROTROPIC
            if getattr(cfg.barotropic, name)
        }
        if selected.recipe.z_coord.nemo_r3t_stored_product:
            actual.add("nemo_r3t_stored_product")
        assert actual == set(expected)
    with pytest.raises(round276.GateError, match="selected fields differ"):
        round276.arm_card(base, "stage_velocity", plant="extra-selector")
