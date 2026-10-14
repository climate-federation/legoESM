"""Controls for the G3 stored-r3u statement imported in round 277."""

from pathlib import Path

import pytest


def test_g3_stored_r3u_selector_defaults_off():
    from legoesm.ocean.state import LatLonCGridOceanConfig

    assert LatLonCGridOceanConfig._field_defaults[
        "zdf_momentum_stage_r3u_nemo_literal"] is False


def test_round277_g3_arm_selects_the_ratio_with_its_divisor_host():
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round269_g1_decision96_gate as prior,
        nemo_testcase_l4_orca2_round277_rung_census_gate as round277,
    )

    deck = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
        "round211/acquisition/orca2_omt1_frames_8step_a_np2")
    if not (deck / "namelist_cfg").is_file():
        pytest.skip("admitted OMT-1 deck is absent")
    base = prior._selected_card("omt1", deck)
    assert not getattr(base.recipe.model_config, round277.STAGE_R3U)
    cfg = round277.r276.arm_card(
        base, "g3_divisors_stage_r3u").recipe.model_config
    on = {name for name in round277.r276.MODEL if getattr(cfg, name)}
    assert on == {"zdf_momentum_nemo_face_thickness", round277.STAGE_R3U}
