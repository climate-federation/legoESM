"""ORCA2 round 326 (DECISION 154): TKE nn_mxlice = 2 needs NEMO's recorded hm_i.

Selecting it without SI3's mean ice thickness (zdftke.f90:620-623) must refuse
on every public run path; GYRE runs the same TKE ln_mxl0 anchor at small cost.
The comparison card selects it on rung 10 only and feeds round 326's record.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    nemo_testcase_l4_orca2_round326_mxlice_d122_gate as gate,
)

ROUND5 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/"
              "round5/acquisition/orca1ice_surface_entry_every_step_a_np2")


def _with_mxlice(cfg, value):
    mixing = cfg.physics.vertical_mixing
    return cfg._replace(physics=cfg.physics._replace(vertical_mixing=mixing._replace(
        tke=mixing.tke._replace(nemo_mxlice=value))))


@pytest.fixture(scope="module")
def gyre():
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import OceanSurfaceForcing

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = recipe.build_nemo_testcase_card("GYRE-zco")
    cfg = _with_mxlice(card.recipe.model_config, 2)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord, cfg)
    zero = jnp.zeros_like(card.recipe.initial_state.eta.data)
    # A surface forcing with stress and ice fraction but no hm_i.
    return card, model, OceanSurfaceForcing(
        tau_x=zero, tau_y=zero, taum=zero, ice_concentration=zero)


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
def test_mxlice_without_hm_i_refuses_on_every_run_path(gyre, path):
    card, model, sf = gyre
    with pytest.raises(ValueError, match="nemo_mxlice=2 requires"):
        PATHS[path](model, card.recipe.initial_state, card.dt_s, sf)


def test_shipped_cards_keep_mxlice_off_and_arms_move_only_rung10(gyre):
    card = gyre[0]
    assert card.recipe.model_config.physics.vertical_mixing.tke.nemo_mxlice == 0
    for rung in gate.RUNGS:
        if rung != "rung10":
            for arm in gate.ARMS:
                assert gate.selected_card(card, arm=arm, rung=rung) is card
    mx = gate.selected_card(card, arm="mxlice", rung="rung10")
    assert mx == ladder.comparison_card(card)
    control = gate.selected_card(card, arm="control", rung="rung10")
    assert control.recipe.model_config == _with_mxlice(
        mx.recipe.model_config, 0)
    with pytest.raises(gate.GateError, match="shipped card selects"):
        gate.selected_card(mx, arm="control", rung="rung10")
    with pytest.raises(ladder.GateError, match="DECISION 135"):
        ladder.comparison_card(card._replace(recipe=card.recipe._replace(
            model_config=_with_mxlice(card.recipe.model_config, 2))))


@pytest.mark.skipif(not ladder.ROUND326_HMI_RUN.is_dir() or not ROUND5.is_dir(),
                    reason="round-326 hm_i record not on this machine")
def test_recorded_hm_i_is_fed_at_its_own_step_only():
    for kt in (1, 2, 10):
        fr_i = ladder.assemble_surface_fields(ROUND5, kt)["fr_i"]
        hm_i = ladder.recorded_hm_i(kt, fr_i)
        assert hm_i.shape == fr_i.shape == (148, 180)
        assert np.all(hm_i >= 0) and np.all(hm_i[fr_i == 0] == 0)
        assert hm_i.max() > 0
    # kt 3's ice fraction is not kt 2's: a record fed at the wrong step refuses.
    with pytest.raises(ladder.GateError, match="not the surface frame's"):
        ladder.recorded_hm_i(2, ladder.assemble_surface_fields(ROUND5, 3)["fr_i"])
    with pytest.raises(ladder.GateError, match="no admitted hm_i record"):
        ladder.recorded_hm_i(11, fr_i)
