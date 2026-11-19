"""Self-contained contract and dispatch tests for the O17 full SI3 step (``si3_o17``).

The O17 numerical tests (full step against NEMO records, compiled and eager) live on
the SI3 fidelity lane with the records they need.  These tests need only model code
and the NEMO reference namelists shipped under ``tests/ice/unit/data``: they pin the
frozen ORCA2_SAS_ICE card, the explicit O17 config, and how ``step_sea_ice`` reaches
the step.
"""

from pathlib import Path

import jax.numpy as jnp
import pytest
from legoesm.ice import sea_ice
from legoesm.ice.config import (
    SI3_HARD_SWITCHES,
    SeaIceConfig,
    build_si3_orca2_o2_config,
    build_si3_orca2_o17_config,
)
from legoesm.ice.si3_finalize import SI3FinalizeForcing
from legoesm.ice.si3_o17 import (
    SI3O17Card,
    SI3O17LiveForcing,
    SI3O17StepState,
    build_orca2_sas_ice_o17_card,
    si3_o17_step,
)
from legoesm.ice.si3_orca2_state import SI3ORCA2StateConfig, SI3State
from legoesm.ice.si3_surface import SI3SurfaceConfig

EXPREF = Path(__file__).parent / "data" / "orca2_sas_ice_expref"


def _surface(switches=None) -> SI3SurfaceConfig:
    cfg = SI3SurfaceConfig.from_input_namelists(
        EXPREF / "namelist_ref", EXPREF / "namelist_cfg",
        EXPREF / "namelist_ice_ref", EXPREF / "namelist_ice_cfg",
    )
    return cfg if switches is None else cfg._replace(switches=switches)


def _card(surface=None) -> SI3O17Card:
    return build_orca2_sas_ice_o17_card(
        surface=_surface(SI3_HARD_SWITCHES) if surface is None else surface,
        thermo_namelist={"ln_icedA": True},
    )


def test_typed_boundary_has_no_mid_step_injection_slots() -> None:
    assert SI3O17LiveForcing._fields[:4] == ("kt", "sss_m", "sst_m", "rn_amax_2d")
    assert not (
        {"redistribution_state", "correction_state", "thermo", "fluxes"}
        & set(SI3O17LiveForcing._fields)
    )
    assert SI3O17StepState._fields == ("ice", "aevp")
    assert SI3O17Card._fields[-1] == "core2_precipitation_closure"


def test_card_freezes_the_orca2_sas_ice_deck_and_keeps_the_callers_surface() -> None:
    surface = _surface(SI3_HARD_SWITCHES)
    card = _card(surface)
    state = card.state
    assert (state.jpl, state.nlay_i, state.nlay_s, state.nn_hls) == (5, 10, 5, 2)
    assert state.ln_icedyn and state.ln_icethd and state.ln_dynadv_pra
    assert state.rDt_ice == card.aevp.dt_s == card.correction.rdt_ice == 10800.0
    assert (card.aevp.n_subcycles, card.aevp.category_count) == (100, 5)
    assert card.core2_precipitation_closure is False
    assert card.ln_drgice_imp is False and card.ln_frazil is False
    # The builder takes the surface config as given: the caller selects the switches
    # (exact-NEMO gates pass SI3_HARD_SWITCHES; main's default is the smooth form).
    assert card.surface is surface
    smooth = _surface()
    assert _card(smooth).surface.switches == smooth.switches != SI3_HARD_SWITCHES


def test_card_builder_refuses_untyped_inputs() -> None:
    with pytest.raises(TypeError, match="resolved SI3SurfaceConfig"):
        build_orca2_sas_ice_o17_card(surface=object(), thermo_namelist={})
    with pytest.raises(TypeError, match="thermodynamic namelist"):
        build_orca2_sas_ice_o17_card(surface=_surface(), thermo_namelist=None)


def test_o17_config_is_explicit_and_reports_the_cards_switches() -> None:
    card = _card()
    cfg = build_si3_orca2_o17_config(card)
    assert cfg.thermo_scheme == "si3_orca2_o2"
    assert (cfg.dynamics, cfg.transport) == ("none", "none")
    assert cfg.si3_switches == card.surface.switches == SI3_HARD_SWITCHES
    smooth = _surface()
    assert smooth.switches != SI3_HARD_SWITCHES
    assert build_si3_orca2_o17_config(_card(smooth)).si3_switches == smooth.switches
    assert cfg.core2_precipitation_closure is False
    assert cfg.si3_o17 is card and cfg.si3_orca2 is card.state
    with pytest.raises(TypeError, match="explicit SI3O17Card"):
        build_si3_orca2_o17_config(object())
    with pytest.raises(ValueError, match="core2_precipitation_closure=False"):
        build_si3_orca2_o17_config(card._replace(core2_precipitation_closure=True))


def test_o17_step_refuses_a_card_outside_its_contract_before_computing() -> None:
    card = _card()
    entry = SI3O17StepState(None, None)
    forcing = SI3O17LiveForcing(1, *([None] * (len(SI3O17LiveForcing._fields) - 1)))
    with pytest.raises(ValueError, match="core2_precipitation_closure=False"):
        si3_o17_step(entry, forcing, card._replace(core2_precipitation_closure=True))
    for field in ("ln_icedyn", "ln_icethd", "ln_dynadv_pra"):
        off = card._replace(state=card.state._replace(**{field: False}))
        with pytest.raises(ValueError, match="explicit live dynamics"):
            si3_o17_step(entry, forcing, off)


def test_step_sea_ice_dispatches_live_forcing_to_the_o17_step(monkeypatch) -> None:
    card = _card()
    config = build_si3_orca2_o17_config(card)
    entry = SI3O17StepState(None, None)
    forcing = SI3O17LiveForcing(1, *([None] * (len(SI3O17LiveForcing._fields) - 1)))
    expected = (entry, object())
    calls = []

    def planted(step_entry, live_forcing, actual_card):
        calls.append((step_entry, live_forcing, actual_card))
        return expected

    monkeypatch.setattr(sea_ice, "si3_o17_step", planted)
    actual = sea_ice.step_sea_ice(entry, forcing, None, None, None, config, 0.0, card.state.rDt_ice)
    assert actual is expected
    assert calls == [(entry, forcing, card)]


def test_step_sea_ice_refuses_mixed_or_cardless_o17_requests(monkeypatch) -> None:
    monkeypatch.setattr(
        sea_ice, "si3_o17_step", lambda *_: pytest.fail("the O17 step must not run")
    )
    card = _card()
    live = SI3O17LiveForcing(1, *([None] * (len(SI3O17LiveForcing._fields) - 1)))
    o17_cfg = build_si3_orca2_o17_config(card)
    dt = card.state.rDt_ice
    z = jnp.zeros((2, 2))
    # Live O17 forcing with an O2 state: refused, never run.
    o2_state = SI3State.zeros(6, 6, card.state)
    with pytest.raises(ValueError, match="requires typed SI3O17StepState"):
        sea_ice.step_sea_ice(o2_state, live, z, z, z, o17_cfg, 0.0, dt)
    # Live O17 forcing without the card: refused (the O2 entry config has no card).
    cardless = build_si3_orca2_o2_config(SI3ORCA2StateConfig.for_orca2_sas_ice())
    assert cardless.si3_o17 is None
    with pytest.raises(ValueError, match="explicit config.si3_o17 card"):
        sea_ice.step_sea_ice(
            SI3O17StepState(None, None), live, z, z, z, cardless, 0.0, cardless.si3_orca2.rDt_ice
        )
    # The O17 state cannot enter the O15/O16 finalization arm.
    finalize = SI3FinalizeForcing(*([None] * len(SI3FinalizeForcing._fields)))
    with pytest.raises(ValueError, match="requires SI3State, not SI3O17StepState"):
        sea_ice.step_sea_ice(SI3O17StepState(None, None), finalize, z, z, z, o17_cfg, 0.0, dt)


def test_o17_card_is_off_by_default() -> None:
    assert SeaIceConfig().si3_o17 is None
