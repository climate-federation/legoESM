"""Guards for round 160's second per-stage continuity solve.

NEMO solves continuity twice per stage on the vector-invariant deck.  The
momentum vertical advection reads a vertical velocity built from the raw stage
velocity; the tracer transport re-solves continuity on the barotropically
corrected transports and overwrites the same array.  Round 160 gives legoESM
both solves at stages 2 and 3.

These guards are structural and run without a GYRE step: which cards resolve
the two-solve program, that the stage transport hands back both fields, that
the pair the oracle partitions twice is refused rather than run once, and that
the private arms are refused unless they are plain booleans.  The measured
numbers are in the round-160 receipt.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
    _nemo_ws_stage_transport,
    nemo_stage_momentum_wzv_executes,
    nemo_stage_momentum_wzv_resolved,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card


@pytest.fixture(scope="module")
def card():
    return build_nemo_testcase_card("GYRE-zco")


def _model(card, **hooks):
    return LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hooks))


def test_the_gyre_card_resolves_the_two_solve_stage_program(card):
    """GYRE takes the RK3-WS stage program, the vector-invariant momentum
    advection and the literal continuity solve together, which is exactly
    NEMO's condition for running the solve twice."""
    assert nemo_stage_momentum_wzv_resolved(card.recipe.model_config) is True


def test_no_card_takes_the_second_solve_while_the_round_is_held(card):
    """The year refused the split, so production keeps the single shared
    solve and nothing executes the second one without the private arm."""
    config = card.recipe.model_config
    assert nemo_stage_momentum_wzv_executes(config) is False
    assert nemo_stage_momentum_wzv_executes(
        config, _NEMOWSRK3TestHooks()) is False
    assert nemo_stage_momentum_wzv_executes(
        config, _NEMOWSRK3TestHooks(
            nemo_stage_momentum_wzv_split=True)) is True


@pytest.mark.parametrize("case", ["LOCK_EXCHANGE-zco", "OVERFLOW-zps"])
def test_the_tanks_do_not_resolve_the_two_solve_stage_program(case):
    """Both tanks take the flux-form momentum advection, which is NEMO's
    ``ELSE`` arm: one solve, on the transports, for both consumers."""
    config = build_nemo_testcase_card(case).recipe.model_config
    assert config.momentum_advection == "flux_form"
    assert nemo_stage_momentum_wzv_resolved(config) is False


def test_the_generic_nemo_gyre_recipe_does_not_resolve_it():
    """The generic recipe takes the generic continuity solve, so it never
    reaches the literal two-solve program."""
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe

    config = build_nemo_gyre_recipe().model_config
    assert config.wzv_call2_evaluation != "nemo_literal"
    assert nemo_stage_momentum_wzv_resolved(config) is False


@pytest.mark.parametrize("recipe", ["nemo_dino_kamm", "nemo_dino_kamm_mlf"])
def test_dino_does_not_resolve_it(recipe):
    """Both DINO cards take the Euler momentum integrator, so NEMO's RK3
    stage program — and therefore its second solve — never runs there."""
    from legoesm.ocean.experiments.dino import (
        dino_config_for_recipe,
        dino_lat_lon_grid,
        dino_lat_lon_model_config,
    )

    dino = dino_config_for_recipe(recipe)
    grid = dino_lat_lon_grid(dino, n_lon=10)
    config, _ = dino_lat_lon_model_config(grid, dino, physics=True)
    assert config.momentum_time_integrator != "rk3_ws"
    assert nemo_stage_momentum_wzv_resolved(config) is False


class _Config:
    """The two fields the stage transport reads off the card's config."""

    adaptive_implicit_vertadv = False
    min_water_column_m = 10.0


def _stage_transport_inputs(card):
    """One stage's operands, straight off the card's own rest state."""
    state = card.recipe.initial_state
    grid = card.recipe.grid
    u_mask = jnp.broadcast_to(
        jnp.asarray(state.u_mask.data)[..., None],
        np.asarray(state.u.data).shape)
    v_mask = jnp.broadcast_to(
        jnp.asarray(state.v_mask.data)[..., None],
        np.asarray(state.v.data).shape)
    from legoesm.ocean.vertical import compute_layer_thickness

    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data,
        card.recipe.z_coord, min_water_column_m=10.0)
    h_stage = compute_layer_thickness(
        state.eta.data, state.H_bathy.data, card.recipe.z_coord,
        min_water_column_m=10.0)
    return dict(
        stage_velocity=(state.u.data, state.v.data),
        h_stage=h_stage,
        eta_stage=state.eta.data,
        h_ref=h_ref,
        Hu_avg=jnp.zeros_like(state.eta.data)[:, :1] * 0.0,
        u_mask_3d=u_mask, v_mask_3d=v_mask, grid=grid,
        z_coord=card.recipe.z_coord, H_bathy=state.H_bathy.data,
        dt=card.dt_s, eta_before=state.eta.data,
        eta_after=state.eta.data,
    )


def test_the_stage_transport_hands_back_both_vertical_velocities(card):
    """Slot 11 is the momentum solve's field and is ``None`` when the card
    does not run the second solve, so the consumers fall back to the one
    field legoESM built before round 160."""
    kwargs = _stage_transport_inputs(card)
    kwargs["Hu_avg"] = jnp.zeros(np.asarray(kwargs["stage_velocity"][0]).shape[:-1])
    kwargs["Hv_avg"] = jnp.zeros(np.asarray(kwargs["stage_velocity"][1]).shape[:-1])
    off = _nemo_ws_stage_transport(
        kwargs.pop("stage_velocity"), kwargs.pop("h_stage"), 1,
        config=_Config(), literal_wzv=True,
        momentum_velocity_form_w=False, **kwargs)
    assert len(off) == 12
    assert off[11] is None


def test_the_adaptive_implicit_pair_is_refused_rather_than_run_once(card):
    """NEMO partitions the momentum pair and the tracer pair separately at
    stage 3.  Only the tracer partition is transcribed, so selecting both
    must raise instead of handing one partition to two consumers."""

    class _Aimp(_Config):
        adaptive_implicit_vertadv = True

    kwargs = _stage_transport_inputs(card)
    kwargs["Hu_avg"] = jnp.zeros(np.asarray(kwargs["stage_velocity"][0]).shape[:-1])
    kwargs["Hv_avg"] = jnp.zeros(np.asarray(kwargs["stage_velocity"][1]).shape[:-1])
    with pytest.raises(ValueError, match="TWO stage-3 partitions"):
        _nemo_ws_stage_transport(
            kwargs.pop("stage_velocity"), kwargs.pop("h_stage"), 2,
            config=_Aimp(), literal_wzv=True,
            momentum_velocity_form_w=True, **kwargs)


@pytest.mark.parametrize("bad", [1, 0, "true", "", (True,), None])
def test_a_split_arm_that_is_not_a_bool_is_refused(card, bad):
    """Anything truthy would silently select the two-solve program and the
    walk would score one compiled program under the other's name."""
    with pytest.raises(ValueError, match="nemo_stage_momentum_wzv_split"):
        _model(card, nemo_stage_momentum_wzv_split=bad)


@pytest.mark.parametrize("bad", [1, 0, "true", (True,), None])
def test_a_clock_pair_arm_that_is_not_a_bool_is_refused(card, bad):
    """The stage clock and the stage after-level move together or not at all;
    a truthy non-bool would select half a control."""
    with pytest.raises(ValueError, match="stage2_momentum_wzv_clock_pair"):
        _model(card, stage2_momentum_wzv_clock_pair=bad)


@pytest.mark.parametrize("bad", [4, -1, "2", 1.5, True])
def test_an_out_of_range_face_ratio_exposure_is_refused(card, bad):
    """The exposure names a stage; anything else would read the wrong stage's
    free-surface ratios and score them against the oracle's stage 2."""
    with pytest.raises(ValueError, match="expose_stage_face_r3"):
        _model(card, expose_stage_face_r3=bad)


def test_the_production_defaults_keep_every_arm_off():
    """Round 160 is HELD, so no card constructs any of the private arms and
    the second continuity solve is off on every one of them."""
    hooks = _NEMOWSRK3TestHooks()
    assert hooks.nemo_stage_momentum_wzv_split is False
    assert hooks.stage2_momentum_wzv_clock_pair is False
    assert hooks.expose_stage_momentum_w is False
    assert hooks.expose_stage_face_r3 == 0


def test_the_admission_gate_census_uses_the_model_s_own_predicate():
    """The Decision-43 card census must not restate the code's condition.

    A gate that re-derives "does this card execute the route" can encode a
    predicate the model does not have, which is how a landing once shipped
    while the gate said the card was not reached.  This asserts agreement on
    every certified card, so a re-derivation that drifts goes red.
    """
    import importlib.util
    from pathlib import Path

    path = (Path(__file__).resolve().parents[3]
            / "scripts/validate/ocean_fidelity/testcases"
            / "nemo_testcase_l2_gyre_decision43_gate.py")
    spec = importlib.util.spec_from_file_location("_d43_round160", path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    rows = gate._card_execution("stage_momentum_wzv")
    # What the candidate would reach if it were selected...
    assert [name for name, row in rows.items() if row["executes_route"]] == [
        "GYRE-zco"]
    # ...and what reaches it today, which is nothing while the round is held.
    assert [name for name, row in rows.items()
            if row["executes_at_this_tip"]] == []

    configs = {
        "GYRE-zco": build_nemo_testcase_card("GYRE-zco").recipe.model_config,
        "LOCK_EXCHANGE-zco": build_nemo_testcase_card(
            "LOCK_EXCHANGE-zco").recipe.model_config,
        "OVERFLOW-zps": build_nemo_testcase_card(
            "OVERFLOW-zps").recipe.model_config,
    }
    for name, config in configs.items():
        assert rows[name]["executes_route"] is (
            nemo_stage_momentum_wzv_resolved(config))
        assert rows[name]["executes_at_this_tip"] is (
            nemo_stage_momentum_wzv_executes(config))
