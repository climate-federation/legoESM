"""Controls for the round-92 ORCA2 hierarchy rung-0 card."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_orca2_initial_ts,
    build_orca2_zps_card,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as gate,
)


DECK = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round90/"
    "acquisition/orca2_rung0_entry_stage_runoff_guarded_10step_np2"
)


@pytest.fixture(scope="module")
def cards():
    if not DECK.is_dir():
        pytest.skip("ORCA2 input deck is unavailable")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    return build_orca2_zps_card(DECK), gate.build_rung0_card(DECK)


def test_rung0_card_states_every_hierarchy_exclusion(cards) -> None:
    shipped, rung0 = cards
    cfg = rung0.recipe.model_config
    vmix = cfg.physics.vertical_mixing
    assert rung0.case == "ORCA2-rung0-zps"
    assert cfg.gm_redi is None
    assert cfg.physics.mle is None
    assert cfg.physics.shortwave_penetration is None
    assert vmix.scheme == "constant"
    assert (vmix.constant.A_v, vmix.constant.K_v) == (1.2e-4, 1.2e-5)
    assert not vmix.iwm.enabled and not vmix.ddm.enabled
    assert cfg.physics.convection.scheme == "enhanced_diffusion"
    assert (cfg.bbl_adv_option, cfg.bbl_diffusive_option) == (0, 0)
    assert rung0.unmeasured_features == ("linear_implicit_bottom_drag",)
    assert rung0.surface_input_operator == "nemo_fld_read_no_tsd_dmp"
    assert shipped.unmeasured_features[-1] == "si3_jpl5_layered_prather_state"


def test_rung0_initial_ts_obeys_disabled_damping_guard(cards) -> None:
    shipped, rung0 = cards
    tmask = np.asarray(rung0.recipe.z_coord.is_active, dtype=bool)
    expected_t, expected_s = build_orca2_initial_ts(
        DECK / "data_1m_potential_temperature_nomask.nc",
        DECK / "data_1m_salinity_nomask.nc",
        tmask,
        apply_hand_alterations=False,
    )
    assert np.array_equal(np.asarray(rung0.recipe.initial_state.T.data), expected_t)
    assert np.array_equal(np.asarray(rung0.recipe.initial_state.S.data), expected_s)

    shipped_t = np.asarray(shipped.recipe.initial_state.T.data)
    shipped_s = np.asarray(shipped.recipe.initial_state.S.data)
    rung0_t = np.asarray(rung0.recipe.initial_state.T.data)
    rung0_s = np.asarray(rung0.recipe.initial_state.S.data)
    assert np.count_nonzero((shipped_t != rung0_t) & tmask) == 1283
    assert np.count_nonzero((shipped_s != rung0_s) & tmask) == 720


def test_rung0_validator_rejects_live_tsd_damping_metadata(cards) -> None:
    _, rung0 = cards
    changed = rung0._replace(surface_input_operator="nemo_fld_read")
    with pytest.raises(gate.GateError, match="module exclusion"):
        gate.validate_rung0_card(changed)


def test_rung0_execution_refuses_only_declared_drag(cards) -> None:
    _, rung0 = cards
    with pytest.raises(ValueError, match="linear_implicit_bottom_drag$"):
        gate.validate_rung0_card(rung0, execution=True)


def test_rung0_validator_rejects_live_excluded_module(cards) -> None:
    shipped, rung0 = cards
    changed = rung0._replace(recipe=rung0.recipe._replace(
        model_config=rung0.recipe.model_config._replace(
            gm_redi=shipped.recipe.model_config.gm_redi)))
    with pytest.raises(gate.GateError, match="module exclusion"):
        gate.validate_rung0_card(changed)


def test_rank_complete_stage0_bridge_is_bit_exact(cards) -> None:
    if not RECORD.is_dir():
        pytest.skip("admitted round-90 frame record is unavailable")
    _, rung0 = cards
    entry = gate.assemble_frame(RECORD, 1, 0)
    state = gate.bridge_entry(rung0, entry)
    for name, expected in entry.items():
        actual = gate.candidate_fields(state)[name]
        assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    with pytest.raises(gate.GateError, match="cover the domain exactly once"):
        gate.assemble_frame(RECORD, 1, 0, plant_layout=True)
