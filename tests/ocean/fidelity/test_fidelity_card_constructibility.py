"""Every registered fidelity card must still BUILD its model.

Lane regression this exists to stop (2026-09): a guard added on the NEMO
test-case lane (`eos_depth="geometric"` restricted to `eos="nemo_teos10"`)
made all three NEMO-faithful DINO cards unconstructible, because DINO selects
the same geometric depth ladder with `eos="nemo_seos"`.  Nothing failed until
someone tried to run DINO, because no cheap gate ever asked the DINO cards to
build.  A card that cannot be instantiated is a dead certificate, so this test
instantiates each one on CPU fp64 and asserts only that it does not raise.

Scope: constructibility, deliberately.  Numbers belong to each card's own
fidelity gate; this is the tripwire that runs in seconds and covers the
cross-lane blast radius those gates do not.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    """Oracle cards are fp64 models (skill rule 1c); restore the policy after."""
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(previous)


DINO_CARDS = ("legoesm_default", "nemo_paper", "nemo_dino_kamm",
              "nemo_dino_kamm_mlf")


def _build_dino(recipe_name: str):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.experiments.dino import (
        create_dino_z_star,
        dino_config_for_recipe,
        dino_lat_lon_grid,
        dino_lat_lon_model_config,
    )

    cfg = dino_config_for_recipe(recipe_name)
    # n_lon is a pure setup knob (the card is the scheme selection); the small
    # value keeps the tripwire cheap without touching any selector.
    grid = dino_lat_lon_grid(cfg, n_lon=12)
    z_coord = create_dino_z_star(cfg)
    model_config, _physics = dino_lat_lon_model_config(grid, cfg)
    return cfg, LatLonCGridOceanModel(grid, z_coord, model_config)


@pytest.mark.parametrize("recipe_name", DINO_CARDS)
def test_dino_card_constructs(recipe_name):
    cfg, model = _build_dino(recipe_name)
    assert model.config.eos == cfg.eos
    # fp64 geometry, not just fp64 state (skill rule 1c).  DINO's analytic
    # z-star carries no bridged `t_depth_ref`, so the reference ladder is the
    # thing to check.
    assert np.asarray(model.z_coord.z_full_ref).dtype == np.float64


@pytest.mark.parametrize("case", ("LOCK_EXCHANGE-zco", "OVERFLOW-zps"))
def test_nemo_testcase_card_constructs(case):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
        validate_nemo_testcase_card,
    )

    card = build_nemo_testcase_card(case)
    validate_nemo_testcase_card(card)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    assert model.config.eos == "nemo_teos10"
    assert model.config.eos_depth == "geometric"


def test_l2_gyre_testcase_card_constructs():
    """The lane-2 GYRE card (adcroft-class trapezoid pairing) now builds.

    CLOSED half of the finding this file carried as an xfail: the guard
    `pgf_quadrature="nemo_trapezoid" requires pgf_scheme="nemo_sco"` had too
    narrow a premise -- hpg_zco (dynhpg.F90:270-296) accumulates the SAME
    -g/2 * e3w(Kmm) * (rhd(jk)+rhd(jk-1)) trapezoid as hpg_sco (:343-374).
    The allow-list now admits {"nemo_sco", "adcroft"}; the sibling
    non-vacuity test proves the guard still bites on an uncertified scheme.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_gyre_zco_card,
        validate_nemo_testcase_card,
    )

    card = build_gyre_zco_card()
    validate_nemo_testcase_card(card)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    assert np.asarray(model.z_coord.z_full_ref).dtype == np.float64
    assert model.config.pgf_quadrature == "nemo_trapezoid"


def test_legacy_native_gyre_recipe_pgf_pair_no_longer_the_blocker():
    """OPEN FINDING, NARROWED: the legacy native GYRE recipe still fails.

    The pgf half is fixed (asserted here: the trapezoid pairing is no longer
    what stops it).  What remains is a SEPARATE, pre-existing tightening from
    commit 36d4a2f72 on the lane merge-base, which made `rk3_ws` one coupled
    momentum+tracer stage identity.  `build_nemo_gyre_recipe` sets
    `momentum_time_integrator="rk3_ws"` (nemo_recipe.py) while leaving
    `tracer_time_integrator="euler"`, and it also selects
    `vertical_momentum_scheme="upwind_perturbation"` /
    `momentum_flux_scheme="upwind"`, neither of which is part of either
    complete WS-RK3 momentum program.

    Reconciling that is a SCIENTIFIC CHOICE about what this legacy card should
    simulate (couple the tracer to rk3_ws, or move momentum off rk3_ws) and is
    deliberately NOT made here.  This test pins the remaining cause so the
    finding cannot be mistaken for the pgf one again.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe

    recipe = build_nemo_gyre_recipe()
    with pytest.raises(ValueError) as excinfo:
        LatLonCGridOceanModel(
            recipe.grid, recipe.z_coord, recipe.model_config)
    message = str(excinfo.value)
    assert "pgf_quadrature" not in message, (
        "the pgf allow-list widening did not land: " + message)
    assert "rk3_ws" in message


def test_trapezoid_quadrature_guard_still_bites_on_uncertified_pgf():
    """Non-vacuity: the pgf allow-list is an allow-list, not a removed guard.

    `smc03` is the density-Jacobian partial-cell PGF; NEMO has no dynhpg arm
    pairing it with the e3w trapezoid recurrence, so that pairing must raise.
    Without this the test above would pass just as well against a deleted
    guard.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe

    recipe = build_nemo_gyre_recipe()
    uncertified = recipe.model_config._replace(pgf_scheme="smc03")
    with pytest.raises(ValueError, match="certified only with pgf_scheme"):
        LatLonCGridOceanModel(recipe.grid, recipe.z_coord, uncertified)


def test_geometric_eos_depth_guard_still_bites_on_uncertified_eos():
    """Non-vacuity: the allow-list is an allow-list, not a removed guard.

    `wright` has no NEMO `eos_insitu` arm taking `gdept`, so pairing it with
    the geometric ladder must still raise.  Without this the test above would
    pass just as well against a deleted guard.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.experiments.dino import (
        create_dino_z_star,
        dino_config_for_recipe,
        dino_lat_lon_grid,
        dino_lat_lon_model_config,
    )

    cfg = dataclasses.replace(
        dino_config_for_recipe("legoesm_default"), eos_depth="geometric")
    grid = dino_lat_lon_grid(cfg, n_lon=12)
    z_coord = create_dino_z_star(cfg)
    model_config, _physics = dino_lat_lon_model_config(grid, cfg)
    with pytest.raises(ValueError, match="certified only with"):
        LatLonCGridOceanModel(grid, z_coord, model_config)
