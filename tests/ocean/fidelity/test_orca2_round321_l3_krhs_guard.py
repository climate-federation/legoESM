"""Round 321 (Decision 147): L3's eddy-induced velocity refuses without the
stage-3 Krhs tracer route, at construction and (backstop) at trace time."""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.nemo_testcase_recipe import orca2_l3_eiv_gm_redi

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
            "round5/acquisition/orca1ice_surface_entry_every_step_a_np2")


@pytest.fixture(scope="module")
def rung10():
    if not ROOT.is_dir():
        pytest.skip("rung-10 acquisition is unavailable")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as gate,
    )
    card = gate.card_fields(ROOT)[1]
    fields = gate.assemble_surface_fields(ROOT, 1)
    kw = dict(zip(("freshwater", "surface_forcing"),
                  gate._surface_forcings(card, ROOT, fields, 1)))
    return card, kw


def _model(card, cfg):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    r = card.recipe
    return LatLonCGridOceanModel(r.grid, r.z_coord, cfg,
                                 iwm_forcing=getattr(r, "iwm_forcing", None))


def test_l3_without_krhs_route_refuses_at_construction(rung10) -> None:
    card, _ = rung10
    cfg = card.recipe.model_config._replace(gm_redi=orca2_l3_eiv_gm_redi())
    assert cfg.rk3_stage3_tracer_krhs_nemo_order is False
    with pytest.raises(ValueError, match="requires rk3_stage3_tracer_krhs"):
        _model(card, cfg)
    _model(card, cfg._replace(rk3_stage3_tracer_krhs_nemo_order=True))
    _model(card, card.recipe.model_config)


def test_l3_backstop_refuses_when_route_dropped_after_construction(
        rung10) -> None:
    card, kw = rung10
    cfg = card.recipe.model_config._replace(
        gm_redi=orca2_l3_eiv_gm_redi(), rk3_stage3_tracer_krhs_nemo_order=True)
    model = _model(card, cfg)
    model.config = cfg._replace(rk3_stage3_tracer_krhs_nemo_order=False)
    with pytest.raises(ValueError, match="eddy-induced velocity"):
        jax.make_jaxpr(lambda s: model.step(s, card.dt_s, **kw))(
            card.recipe.initial_state)


def test_l3_refuses_when_route_dropped_after_first_trace(rung10) -> None:
    # ``self`` is a static jit argument: a post-trace config edit hits the
    # cached trace, so the refusal must run on every step call.  BBL off
    # only so the route-on first trace composes (stage3_krhs_terms refuses
    # the diffusive BBL).
    card, kw = rung10
    cfg = card.recipe.model_config._replace(
        gm_redi=orca2_l3_eiv_gm_redi(), rk3_stage3_tracer_krhs_nemo_order=True,
        bbl_diffusive_option=0)
    model = _model(card, cfg)

    def trace():  # fresh callable: make_jaxpr caches on function identity
        return jax.make_jaxpr(lambda s: model.step(s, card.dt_s, **kw))(
            card.recipe.initial_state)

    trace()
    model.config = cfg._replace(rk3_stage3_tracer_krhs_nemo_order=False)
    with pytest.raises(ValueError, match="eddy-induced velocity"):
        trace()
