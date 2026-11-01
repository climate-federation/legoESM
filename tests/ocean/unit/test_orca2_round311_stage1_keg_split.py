"""Round 311: G8's stage-1 Krhs order and G1b's dyn_keg guard are separate settings.

NEMO runs both in one stage-1 accumulator (stp2d.f90:141/:144/:147, then
dyn_keg whose subtraction is dynkeg.f90:129-130).  Before round 311 the
stage-1 order replaced G1b's flag in the shared operator-order argument and
re-added the KE gradient as a plain sum.
"""
from __future__ import annotations

import jax
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as m
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe


@pytest.fixture(autouse=True, scope="module")
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def _jaxpr(case, private_arm=None, **flags):
    card = recipe.build_nemo_testcase_card(case)
    r = card.recipe
    hooks = ({} if private_arm is None else {"_nemo_ws_test_hooks": m._NEMOWSRK3TestHooks(
        nemo_stage_rhs_accumulation_order_arm=private_arm)})
    model = m.LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config._replace(**flags), **hooks)
    return str(jax.make_jaxpr(lambda s: model.step(s, card.dt_s))(r.initial_state))


def test_stage1_order_with_vector_form_needs_g1b_order():
    card = recipe.build_nemo_testcase_card("VORTEX_VEC-zco")
    r = card.recipe
    assert r.model_config.momentum_advection == "vector_invariant"
    with pytest.raises(ValueError, match="requires rk3_vector_rhs_nemo_order=True"):
        m.LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config._replace(
            rk3_stage1_ldf_source_order=True, rk3_vector_rhs_nemo_order=False))


def test_stage1_order_beside_private_order_arm_refuses():
    cfg = recipe.build_nemo_testcase_card("VORTEX_VEC-zco").recipe.model_config
    on = cfg._replace(rk3_stage1_ldf_source_order=True)
    with pytest.raises(ValueError, match="both select the stage-1 Krhs order"):
        m._nemo_ws_stage1_ldf_source_order(on, True)
    assert m._nemo_ws_stage1_ldf_source_order(on, None) is True
    assert m._nemo_ws_stage1_ldf_source_order(cfg, True) is False


def test_stage1_order_keeps_g1b_keg_guard_on_vector_form():
    both = _jaxpr("VORTEX_VEC-zco", rk3_vector_rhs_nemo_order=True,
                  rk3_stage1_ldf_source_order=True)
    guard = _jaxpr("VORTEX_VEC-zco", ("stage1-source-order", None, "nemo-keg-guard"),
                   rk3_vector_rhs_nemo_order=True)
    plain = _jaxpr("VORTEX_VEC-zco", ("stage1-source-order", None, "baseline"),
                   rk3_vector_rhs_nemo_order=True)
    assert both == guard
    assert both != plain


def test_stage1_order_on_flux_form_keeps_the_certified_plain_add():
    card_cfg = recipe.build_nemo_testcase_card("VORTEX-zco").recipe.model_config
    assert card_cfg.momentum_advection == "flux_form"
    public = _jaxpr("VORTEX-zco", rk3_stage1_ldf_source_order=True)
    old = _jaxpr("VORTEX-zco", ("stage1-source-order", None, "baseline"))
    assert public == old
