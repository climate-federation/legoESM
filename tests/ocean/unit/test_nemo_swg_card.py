"""SWG round-1 card transcription and fail-closed execution boundary."""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_DEVIATIONS, SWG_NAMELIST, SWG_UNMEASURED,
    SwgResolvedNamelist, build_swg_card, swg_horizontal_coordinates,
    swg_wet_mask, validate_nemo_testcase_card,
    validate_nemo_testcase_card_for_execution,
)


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    old = get_policy(); set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(old)


@pytest.fixture(scope="module")
def card():
    return build_swg_card()


def test_card_shape_constants_and_initial_state(card):
    state = card.recipe.initial_state
    assert card.case == "SWG-zco"
    assert (card.dt_s, card.n_steps, card.dummy_bottom_records) == (3600.0, 172800, 1)
    assert state.eta.data.shape == (84, 84)
    assert state.u.data.shape == (84, 85, 1)
    assert state.v.data.shape == (85, 84, 1)
    assert int(np.count_nonzero(card.recipe.land_mask)) == 6400
    assert card.recipe.z_coord.n_levels == 1
    assert card.recipe.model_config.constants.g == 0.02
    for field in (state.T, state.S, state.u, state.v, state.eta, state.uu_b, state.vv_b):
        assert not np.any(np.asarray(field.data))


def test_source_coordinate_formula_and_absolute_y_coriolis():
    src = swg_horizontal_coordinates()
    assert (src["ni"], src["nj"], src["spacing_m"]) == (84, 84, 25000.0)
    assert src["glamt"][0, 0] == src["gphit"][0, 0] == -12500.0
    assert src["glamu"][0, 0] == src["gphiv"][0, 0] == 0.0
    want = SWG_NAMELIST.rn_f0_s + SWG_NAMELIST.rn_beta_m_s * np.abs(src["gphit"])
    assert np.array_equal(src["ff_t"], want)
    signed = SWG_NAMELIST.rn_f0_s + SWG_NAMELIST.rn_beta_m_s * src["gphit"]
    assert not np.array_equal(signed, want)


def test_wet_mask_prediction_and_plant():
    wet = swg_wet_mask()
    jj, ii = np.where(wet > 0.5)
    assert (wet.sum(), jj.min(), jj.max(), ii.min(), ii.max()) == (6400, 1, 80, 1, 80)
    planted = wet.copy(); planted[1, 1] = 0.0
    assert planted.sum() == 6399


def test_all_resolved_fields_are_explicit_namedtuple_values():
    assert set(SwgResolvedNamelist._fields) == set(SWG_NAMELIST._asdict())
    assert len(SwgResolvedNamelist._fields) >= 55
    assert SWG_DEVIATIONS == (("key_xios", True, False,
                               "conda-scalarmath reference toolchain"),)


def test_execution_refuses_every_unbuilt_oracle_arm(card):
    assert len(SWG_UNMEASURED) == 3
    with pytest.raises(ValueError, match="ENS total-vorticity"):
        validate_nemo_testcase_card_for_execution(card)
    for item in SWG_UNMEASURED:
        assert item in str(card.unmeasured_features)


def test_validator_refuses_planted_card_drift(card, monkeypatch):
    monkeypatch.setattr(recipe, "SWG_UNMEASURED", ("planted",))
    with pytest.raises(ValueError, match="SWG_UNMEASURED"):
        validate_nemo_testcase_card(card)


def test_unknown_rotation_is_refused(monkeypatch):
    monkeypatch.setattr(recipe, "SWG_NAMELIST", SWG_NAMELIST._replace(rn_theta_deg=1.0))
    with pytest.raises(ValueError, match="rn_theta"):
        recipe.swg_horizontal_coordinates()
