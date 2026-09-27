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


@pytest.mark.parametrize(
    "case", ("LOCK_EXCHANGE-zco", "OVERFLOW-zps", "VORTEX-zco"))
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
    # VORTEX is the first card on this identity with a LIVE rotation
    # operator, and building the model is what discovered that the model
    # REFUSES the scheme pair its namelist selects (EEN vorticity with
    # flux-form momentum).  The card now declares that gap instead of
    # substituting another Coriolis operator, so what this row checks is
    # that the declared-gap card still builds a model at all.
    if case == "VORTEX-zco":
        assert card.unmeasured_features
        assert not model.config.vorticity_scheme.endswith("_total")
        assert not model.config.adaptive_implicit_vertadv


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


def test_legacy_native_gyre_recipe_now_constructs():
    """CLOSED FINDING.  This test used to assert that the card RAISES.

    It pinned two separate blockers in turn.  The pgf half was fixed first,
    and what remained was that ``rk3_ws`` is one coupled momentum+tracer stage
    identity: the card selected a vertical momentum scheme and a momentum flux
    scheme that belong to NEITHER complete WS-RK3 momentum program, so the
    model refused itself at construction.  The old docstring called
    reconciling that "a SCIENTIFIC CHOICE ... deliberately NOT made here".

    User decision 15C made it.  NEMO's GYRE resolves the vector form, whose
    printed program is "keg + zad + vor" (``dynadv.F90:144``), and
    ``ln_zad_Aimp`` is ``.false.``, so the card takes NEMO's own vertical
    momentum advection and drops the adaptive-implicit path.  That was the
    last missing field, and the card constructs.

    The two fields are asserted BY NAME rather than only through a successful
    build, so reverting either turns this red with the reason visible instead
    of a bare construction error.  The pgf guard's own non-vacuity lives in
    the next test, which still requires the uncertified pairing to raise.

    Constructing is not stepping: the card still cannot take a step, because
    its z-star coordinate carries no active-cell mask.  That is a separate,
    newly surfaced finding tracked in the round-29 receipt, and it is why the
    assertion here stops at construction.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe

    recipe = build_nemo_gyre_recipe()
    assert recipe.model_config.vertical_momentum_scheme == "nemo_advective"
    assert recipe.model_config.adaptive_implicit_vertadv is False
    LatLonCGridOceanModel(recipe.grid, recipe.z_coord, recipe.model_config)


def test_the_ws_rk3_momentum_program_guard_still_bites():
    """Non-vacuity for the test above: it passes because the card is complete,
    not because the guard was deleted.  Putting either field back the way it
    was must refuse the card again."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_recipe import build_nemo_gyre_recipe

    recipe = build_nemo_gyre_recipe()
    reverted = recipe.model_config._replace(
        vertical_momentum_scheme="upwind_perturbation")
    with pytest.raises(ValueError, match="rk3_ws"):
        LatLonCGridOceanModel(recipe.grid, recipe.z_coord, reverted)


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


def test_geometric_eos_depth_accepts_nemo_eos80():
    """The ORCA2 EOS-80/QCO depth pair is an admitted NEMO source arm.

    ``eosbn2.F90:260`` supplies live ``gdept`` to both ``np_teos10`` and
    ``np_eos80``.  This focused tripwire stops the allow-list from making the
    source-certified ORCA2 card unconstructible again without needing the
    external ORCA2 deck in the unit-test environment.
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
    model_config = model_config._replace(eos="nemo_eos80")
    model = LatLonCGridOceanModel(grid, z_coord, model_config)
    assert model.config.eos == "nemo_eos80"
    assert model.config.eos_depth == "geometric"


def test_orca2_card_selects_resolved_rk3_sh2():
    """ORCA2 pins NEMO's face-native NOW-squared ``zdf_sh2`` tuple.

    The external deck path is supplied by the campaign environment.  Skip
    only when that immutable input is absent from a generic unit-test host.
    """
    from pathlib import Path

    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_orca2_zps_card,
        validate_nemo_testcase_card,
    )

    deck = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
    if not deck.exists():
        pytest.skip("ORCA2 immutable input deck is not installed")
    card = build_orca2_zps_card(deck)
    tke = card.recipe.model_config.physics.vertical_mixing.tke
    assert (
        tke.tke_shear_production,
        tke.tke_shear_avm_weighting,
        tke.tke_shear_evaluation_stage,
        tke.tke_shear_metric_source,
    ) == (
        "nemo_face_native_nbb2", "nemo_face", "step_entry",
        "nemo_qco_live_face",
    )
    assert tke.bottom_tke_bc is True
    assert tke.eice == 1
    assert tke.tke_langmuir_evaluation == "vectorized"
    validate_nemo_testcase_card(card)

    # Binding selector plant: the former tuple must be rejected by the real
    # card validator rather than compared by hand.
    old_tke = tke._replace(
        tke_shear_production="squared_centered",
        tke_shear_avm_weighting="tpoint",
        tke_shear_metric_source="tpoint_jacobian",
    )
    planted_cfg = card.recipe.model_config._replace(
        physics=card.recipe.model_config.physics._replace(
            vertical_mixing=(
                card.recipe.model_config.physics.vertical_mixing._replace(
                    tke=old_tke))))
    planted = card._replace(
        recipe=card.recipe._replace(
            model_config=planted_cfg, physics_config=planted_cfg.physics))
    with pytest.raises(ValueError, match="Nbb\\*Nbb face-native selector"):
        validate_nemo_testcase_card(planted)

    bottom_off = tke._replace(bottom_tke_bc=False)
    planted_cfg = card.recipe.model_config._replace(
        physics=card.recipe.model_config.physics._replace(
            vertical_mixing=(
                card.recipe.model_config.physics.vertical_mixing._replace(
                    tke=bottom_off))))
    planted = card._replace(
        recipe=card.recipe._replace(
            model_config=planted_cfg, physics_config=planted_cfg.physics))
    with pytest.raises(ValueError, match="bottom-friction Dirichlet"):
        validate_nemo_testcase_card(planted)

    eice_off = tke._replace(eice=0)
    planted_cfg = card.recipe.model_config._replace(
        physics=card.recipe.model_config.physics._replace(
            vertical_mixing=(
                card.recipe.model_config.physics.vertical_mixing._replace(
                    tke=eice_off))))
    planted = card._replace(
        recipe=card.recipe._replace(
            model_config=planted_cfg, physics_config=planted_cfg.physics))
    with pytest.raises(ValueError, match="nn_eice=1"):
        validate_nemo_testcase_card(planted)

    langmuir_literal = tke._replace(tke_langmuir_evaluation="nemo_literal")
    planted_cfg = card.recipe.model_config._replace(
        physics=card.recipe.model_config.physics._replace(
            vertical_mixing=(
                card.recipe.model_config.physics.vertical_mixing._replace(
                    tke=langmuir_literal))))
    planted = card._replace(
        recipe=card.recipe._replace(
            model_config=planted_cfg, physics_config=planted_cfg.physics))
    with pytest.raises(ValueError, match="tke_langmuir_evaluation"):
        validate_nemo_testcase_card(planted)


# ---------------------------------------------------------------------------
# Guards RELAXED by the lane-1/lane-2 merge (c9526e585).  Each was an
# iso-side/merge-base guard that the GYRE lane deleted or narrowed; each is
# justified below from NEMO's own source, and each of these tests FAILS (the
# card stops constructing) if its guard is restored.  The oracle namelist that
# selects all three is
# /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10/output.namelist.dyn.
# ---------------------------------------------------------------------------

def _gyre_card_and_model():
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_gyre_zco_card

    card = build_gyre_zco_card()
    return card, LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)


def test_live_coriolis_split_runs_with_the_ab3am4_filter():
    """DELETED guard: live split "incompatible with" nemo_ab3am4.

    NEMO runs exactly this pair.  `nn_bt_flt=3` selects the Demange AB3-AM4
    filter (dynspg_ts.F90:199-200 `ll_bt_av=.FALSE.`; :1270 "Demange time
    filter"), and in the SAME run the barotropic Coriolis is subcycled live.
    GYRE's cpp keys (provenance/cpp_GYRE_OMIP_L2_P3.fcm) select `key_RK3`, so
    the EXECUTED branch is dynspg_ts.F90's RK3 Phase-1 (:270-302), not its
    MLF twin (:303-452) -- the pre-step subtraction there takes the Kmm
    depth-mean velocity (:296 `CALL dyn_cor_2D(puu_b(:,:,Kmm), ...)`, :299
    `zu_frc = zu_frc - zu_trd*ssumask`) while the in-substep call (shared by
    both branches, unconditional on nn_bt_flt) takes the AB3-EXTRAPOLATED
    velocity (:550,552-553 `ua_e/va_e = za1*un_e + za2*ub_e + za3*ubb_e`,
    then :689 `CALL dyn_cor_2D(ua_e, va_e, ...)`).  The deleted guard
    asserted that substep-0 "would not cancel bit-exactly" and told the
    caller to use the boxcar filter instead; NEMO neither requires that
    cancellation nor offers that choice -- the za1/za2/za3 coefficient block
    (:535-543) is not conditioned on nn_bt_flt at all, so no filter choice
    ever avoids it. What IS true: at kt==nit000 with LN_RSTART=F (confirmed
    in this GYRE run: ocean.output:226 `ln_rstart = F`), NEMO's init block
    sets ll_init=.TRUE. (dynspg_ts.F90:223, the `ELSE ! init bb fields with
    0` arm), so jn=1 gets za1=1,za2=za3=0 (:536-538) and un_e is seeded from
    puu_b(:,:,Kmm) under LN_BT_FW=T (:487, confirmed via output.namelist.dyn
    LN_BT_FW=T) -- so substep 1's operand IS bit-identical to the pre-step
    one and the two calls cancel exactly. Substep>=2 (un_e has already been
    dynamically updated by then) and every later kt (ll_init reverts to
    ll_bt_av=.FALSE. once kt!=nit000) do NOT cancel -- that residual is the
    intended AB3 evolution NEMO always carries in the barotropic mode, not
    an incompatibility the guard could route around. The guard was wrong,
    and GYRE could not be built with it.

    The oracle run selects NN_BT_FLT=3, LN_BT_FW=T (output.namelist.dyn).
    """
    card, model = _gyre_card_and_model()
    assert model.config.barotropic_coriolis_split == "live"
    assert model.config.barotropic.barotropic_time_filter == "nemo_ab3am4"


def test_ene_vorticity_requires_the_ene_barotropic_coriolis_not_een():
    """WIDENED guard: the barotropic stencil was hard-wired to een/een_metric.

    NEMO derives the barotropic Coriolis stencil from the SAME `nvor_scheme`
    as the 3-D vorticity operator: `dyn_cor_2D_init` switches on it at
    dynspg_ts.F90:1326, and its `np_ENE` arm (:1383-1400) builds FOUR 2-point
    Sadourny coefficients (`r1_4 * ... * ff_f(ji,jj)` for the north pair,
    `ff_f(ji,jj-1)` for the south pair), whereas the `np_EEN` arm
    (:1327-1340) builds 3-point triads (`zpvo_nw = ff_f(i-1,j)+ff_f(i,j)+
    ff_f(i,j-1)`).  `dyn_cor_2D` (:1483-1506) then applies whichever set was
    built.  So under `ln_dynvor_ene=T` the subtraction stencil must be ENE;
    demanding EEN triads there would be a stencil NEMO never runs.

    The oracle run selects LN_DYNVOR_ENE=T, LN_DYNVOR_EEN=F.
    """
    card, model = _gyre_card_and_model()
    assert model.config.vorticity_scheme == "ene_total"
    assert model.config.barotropic.barotropic_coriolis == "ene_metric"


def test_rk3_ws_admits_pure_redi_but_still_refuses_a_staged_gm_bolus():
    """NARROWED guard: rk3_ws refused ANY gm_redi; now only a live bolus.

    The guard's subject is the staged GM BOLUS transport, and NEMO gates that
    separately from isoneutral diffusion: the eddy-induced transport is added
    only under `ln_ldfeiv` (traadv.F90:208 `IF( ln_ldfeiv .AND. .NOT.
    ln_traldf_triad ) THEN ! Add the eiv transport`; ldftra.F90:536 `IF( .NOT.
    ln_ldfeiv ) THEN !== Parametrization not used ==!`), while `ln_traldf_iso`
    runs the Redi diffusion regardless.  In this model the bolus is linear in
    kappa_GM (gm_redi.py:11-13, psi = kappa_GM * S), so kappa_GM=0 removes it
    exactly and leaves pure Redi -- which is what the oracle runs.  Refusing
    every gm_redi would refuse NEMO's own GYRE.  The narrowed guard still
    raises on a non-zero bolus; the second half of this test proves that.

    The oracle run selects LN_TRALDF_ISO=T with LN_LDFEIV=F.
    """
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    card, model = _gyre_card_and_model()
    assert model.config.gm_redi.kappa_GM == 0.0
    assert model.config.gm_redi.kappa_Redi > 0.0
    cfg = card.recipe.model_config
    with pytest.raises(ValueError, match="staged GM bolus"):
        LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord,
            cfg._replace(gm_redi=cfg.gm_redi._replace(kappa_GM=1.0)))
