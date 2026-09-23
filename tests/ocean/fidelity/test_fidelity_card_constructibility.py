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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OPEN FINDING (2026-09, unfixed here — out of this change's scope): "
        "commit 36d4a2f72 also added `pgf_quadrature=\"nemo_trapezoid\" "
        "requires pgf_scheme=\"nemo_sco\"`, which rejects the NEMO GYRE card "
        "(pgf_scheme=\"adcroft\" + nemo_trapezoid).  Same class as the "
        "eos_depth allow-list fixed alongside this test: NEMO's e3w-weighted "
        "trapezoid quadrature is used by hpg_zco (dynhpg.F90:235-302) as well "
        "as hpg_sco (:305-393), so the guard's premise that the trapezoid IS "
        "the hpg_sco recurrence is too narrow.  The GYRE card built at the "
        "lane base f81436227b1f and does not build at HEAD.  Flip this to a "
        "plain test once the pgf allow-list is widened with its citation."),
)
def test_nemo_gyre_card_constructs():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe

    recipe = build_nemo_gyre_recipe()
    model = LatLonCGridOceanModel(
        recipe.grid, recipe.z_coord, recipe.model_config)
    assert np.asarray(model.z_coord.z_full_ref).dtype == np.float64


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
