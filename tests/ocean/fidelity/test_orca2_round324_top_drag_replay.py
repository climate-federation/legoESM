"""ORCA2 round 324 (DECISION 150): the ice-ocean top drag needs NEMO's record.

Selecting ``implicit_ice_ocean_top_drag`` without the recorded rCdU_ice
(iceupdate.F90:391) must refuse on every public run path.  VORTEX carries the
G5 path at small cost; the ORCA2 comparison card is the ladder's helper.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe


@pytest.fixture(scope="module")
def vortex():
    card = recipe.build_nemo_testcase_card("VORTEX-zco")
    cfg = recipe._nemo_linear_bottom_drag(card.recipe.model_config)._replace(
        implicit_ice_ocean_top_drag=True)
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import OceanSurfaceForcing
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    zero = jnp.zeros_like(card.recipe.initial_state.eta.data)
    # A full surface forcing with the one record absent.
    return card, model, OceanSurfaceForcing(tau_x=zero, tau_y=zero)


PATHS = {
    "step": lambda m, s, dt, sf: m.step(s, dt, surface_forcing=sf),
    "step_no_forcing": lambda m, s, dt, sf: m.step(s, dt),
    "step_checked": lambda m, s, dt, sf: m.step_checked(s, dt, surface_forcing=sf),
    "integrate": lambda m, s, dt, sf: m.integrate(
        s, dt, dt, step_inputs=[{"surface_forcing": sf}]),
    "integrate_no_inputs": lambda m, s, dt, sf: m.integrate(s, dt, dt),
    "integrate_scan": lambda m, s, dt, sf: m.integrate_scan(
        s, 1, dt, step_inputs=[{"surface_forcing": sf}]),
    "integrate_scan_no_inputs": lambda m, s, dt, sf: m.integrate_scan(s, 1, dt),
}


@pytest.mark.parametrize("path", sorted(PATHS))
def test_top_drag_without_record_refuses_on_every_run_path(vortex, path):
    card, model, sf = vortex
    with pytest.raises(ValueError, match="requires the recorded"):
        PATHS[path](model, card.recipe.initial_state, card.dt_s, sf)


@pytest.fixture(scope="module")
def gyre():
    # comparison_card also selects TKE nn_mxlice = 2 (round 326), so it needs a
    # card with physics; VORTEX has none.
    return recipe.build_nemo_testcase_card("GYRE-zco")


def test_comparison_card_selects_only_the_top_drag_and_mxlice(gyre):
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )
    card = gyre
    cfg = card.recipe.model_config
    assert not cfg.implicit_ice_ocean_top_drag
    got = ladder.comparison_card(card)
    mixing = cfg.physics.vertical_mixing
    assert got.recipe.model_config == cfg._replace(
        implicit_ice_ocean_top_drag=True,
        physics=cfg.physics._replace(vertical_mixing=mixing._replace(
            tke=mixing.tke._replace(nemo_mxlice=2))))
    assert got._replace(recipe=card.recipe) == card
    with pytest.raises(ladder.GateError, match="DECISION 133"):
        ladder.comparison_card(got)


def test_round324_arms_move_only_rung10_drag(gyre):
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
        nemo_testcase_l4_orca2_round324_top_drag_d122_gate as gate,
    )
    card = gyre
    assert gate.selected_card(card, arm="drag", rung="rung10") == \
        ladder.comparison_card(card)
    for rung in gate.RUNGS:
        assert gate.selected_card(card, arm="control", rung=rung) is card
        if rung != "rung10":
            assert gate.selected_card(card, arm="drag", rung=rung) is card
    with pytest.raises(gate.GateError):
        gate.selected_card(card, arm="D122", rung="rung10")
    with pytest.raises(gate.GateError, match="shipped card selects"):
        gate.selected_card(ladder.comparison_card(card), arm="control",
                           rung="rung10")
