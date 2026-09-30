"""Shared-code changes on the NEMO-testcase branch must be card-selected.

Every test here pins one rule: a NEMO-literal transcription only changes a
run when that run's CONFIG asks for it.  The production cards that do NOT ask
keep the behaviour they had before the branch.  Each test is written so that
reverting its fix (dropping the flag, or re-deriving unconditionally) makes it
fail.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.vertical_mixing.config import TKEConfig
from legoesm.ocean.physics.vertical_mixing.tke import (
    _mixing_length_floor,
    _mxl0_anchor_floor,
    _mxl0_surface_anchor,
    nemo_tke_effective_ice_fraction,
)


# --------------------------------------------------------------------------
# B1 — the NEMO rmxl_min arms (zdftke.F90:841-848)
# --------------------------------------------------------------------------

def test_iwm_card_keeps_its_forced_floor_not_the_derivation():
    """ORCA1/ORCA2 run ln_zdfiwm=.TRUE., which FORCES rmxl_min = 1e-3.

    zdftke.F90:842-843 sets rn_emin=1e-10 and rmxl_min=1e-3 and never
    evaluates the :846 derivation.  With rn_emin=1e-10 that derivation
    returns 1.0 m — a thousand times NEMO's floor — so an unconditional
    derivation (the pre-fix code) fails this test.
    """
    cfg = TKEConfig(tke_mxl_choice=4, c_k=0.1,
                    tke_background=1.0e-10, mxl_min=1.0e-3)
    assert float(_mixing_length_floor(cfg)) == 1.0e-3
    # and the derivation this card must NOT take:
    derived = TKEConfig(tke_mxl_choice=4, c_k=0.1, tke_background=1.0e-10,
                        mxl_min=1.0e-3, nemo_derived_mxl_min=True)
    assert float(_mixing_length_floor(derived)) == pytest.approx(1.0, rel=1e-12)


def test_orca1_card_resolves_to_the_nemo_forced_floor():
    """The shipped ORCA1 TKE card, built by its own driver helper."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "_run_omip_core2_for_test", root / "scripts" / "run" / "run_omip_core2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The ORCA1 deck runs ln_zdfiwm=T; that arm of the card sets rn_emin=1e-10
    # and rmxl_min=1e-3 from zdftke.F90:841-843.
    cfg = module.orca1_zdftke_config(iwm_enabled=True)
    assert cfg.tke_mxl_choice == 4
    assert cfg.tke_background == 1.0e-10
    assert float(_mixing_length_floor(cfg)) == 1.0e-3
    # The card's other arm keeps its own floor too (the unconditional
    # derivation returned 1e-2 there, a million times this value).
    off = module.orca1_zdftke_config(iwm_enabled=False)
    assert float(_mixing_length_floor(off)) == 1.0e-8


def test_non_iwm_nemo_card_takes_the_derived_floor():
    """GYRE/DINO run ln_zdfiwm=.FALSE.: rmxl_min = 1e-6/(c_k*sqrt(rn_emin))."""
    cfg = TKEConfig(tke_mxl_choice=3, c_k=0.1, tke_background=1.0e-6,
                    mxl_min=1.0e-8, nemo_derived_mxl_min=True)
    assert float(_mixing_length_floor(cfg)) == pytest.approx(1.0e-2, rel=1e-12)


def test_gyre_and_dino_cards_select_the_derived_floor():
    from legoesm.ocean.experiments.dino import DINO_RECIPES
    from legoesm.ocean.fidelity.nemo_recipe import _nemo_tke_config

    gyre = _nemo_tke_config()
    assert gyre.nemo_derived_mxl_min is True
    assert float(_mixing_length_floor(gyre)) == pytest.approx(1.0e-2, rel=1e-12)
    assert DINO_RECIPES["nemo_dino_kamm"]["tke_nemo_derived_mxl_min"] is True


def test_veros_and_fesom_choices_keep_their_configured_floor():
    """A non-NEMO card is untouched by the branch, whatever its choice."""
    for choice in (1, 2, 3, 4):
        cfg = TKEConfig(tke_mxl_choice=choice, mxl_min=3.0e-7)
        assert float(_mixing_length_floor(cfg)) == 3.0e-7


# --------------------------------------------------------------------------
# S6 — the ln_mxl0 surface anchor's tmask operand
# --------------------------------------------------------------------------

def test_anchor_without_a_surface_mask_is_accepted_by_default():
    """FESOM has no surface T-mask and must keep working (main's behaviour)."""
    cfg = TKEConfig(tke_mxl_choice=3, mxl_min=1.0e-8)
    anchor = _mxl0_surface_anchor(
        cfg, jnp.asarray([0.07, 0.0]), 1026.0, 9.80665, None)
    assert anchor is not None
    assert np.all(np.isfinite(np.asarray(anchor)))


def test_nemo_card_requires_the_masked_statement():
    cfg = TKEConfig(tke_mxl_choice=3, nemo_mxl0_surface_tmask=True)
    with pytest.raises(ValueError, match="surface_tmask"):
        _mxl0_surface_anchor(cfg, jnp.asarray([0.07]), 1026.0, 9.80665, None)
    masked = _mxl0_surface_anchor(
        cfg, jnp.asarray([0.07, 0.07]), 1026.0, 9.80665,
        jnp.asarray([1.0, 0.0]))
    floor = float(_mxl0_anchor_floor(cfg))
    assert float(np.asarray(masked)[1]) == pytest.approx(floor)
    assert float(np.asarray(masked)[0]) > floor


# --------------------------------------------------------------------------
# D66 — the ORCA1 OMIP card keeps its unmasked ln_mxl0 anchor
# --------------------------------------------------------------------------

def _orca1_tke_config(iwm_enabled):
    """Resolve the ORCA1 OMIP card's TKE config from its own builder."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    path = root / "scripts" / "run" / "run_omip_core2.py"
    spec = importlib.util.spec_from_file_location("_omip_core2_card", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.orca1_zdftke_config(iwm_enabled=iwm_enabled)


@pytest.mark.parametrize("iwm_enabled", [False, True])
def test_orca1_card_keeps_the_unmasked_ln_mxl0_anchor(iwm_enabled):
    """Decision 66: the ORCA1 card must NOT take NEMO's masked anchor.

    NEMO's compiled statement multiplies the surface stress by tmask(:,:,1)
    (zdftke.F90:602), which collapses the anchor to the mixing-length
    floor on LAND columns.  That transcription belongs to the NEMO-literal
    cards; this card keeps the behaviour it had before the branch.  The test
    fails both ways: if the card re-selects the mask, and if the library
    default stops being the unmasked arm.
    """
    cfg = _orca1_tke_config(iwm_enabled)
    assert cfg.nemo_mxl0_surface_tmask is False
    assert TKEConfig().nemo_mxl0_surface_tmask is False
    # ... and the resolved anchor is the unmasked one: a LAND column (mask 0)
    # still sees its own wind stress, exactly as it did before the branch.
    taum = jnp.asarray([0.07, 0.07])
    unmasked = _mxl0_surface_anchor(cfg, taum, 1026.0, 9.80665,
                                    jnp.asarray([1.0, 0.0]))
    assert float(np.asarray(unmasked)[1]) == float(np.asarray(unmasked)[0])
    assert float(np.asarray(unmasked)[0]) > float(_mxl0_anchor_floor(cfg))


# --------------------------------------------------------------------------
# D72 — the ORCA1 OMIP card keeps MAIN's calm-column ln_mxl0 floor (0.04 m)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("iwm_enabled", [False, True])
def test_orca1_card_keeps_mains_rn_mxl0_surface_floor(iwm_enabled):
    """Decision 72: the ORCA1 card's CALM surface anchor stays 0.04 m.

    NEMO floors the ln_mxl0 wind anchor at ``rn_mxl0``
    (``zmxlm(ji,1) = MAX( rn_mxl0, zmxlm(ji,1) )``, GYRE ppsrc
    zdftke.f90:610), and ``zdf_tke_init`` has already OVERWRITTEN that
    namelist ``rn_mxl0`` with the active mixing-length floor because
    ``ln_mxl0`` is true (shipped zdftke.F90:859-862; GYRE ppsrc:828-831) —
    1.0e-3 m on this ORCA1 arm, since ``ln_zdfiwm`` forces rmxl_min = 1.0e-3
    (shipped zdftke.F90:841-843).  The user's decision is that Pierre's card
    keeps the namelist value main used, and NEMO's overwrite stays behind the
    NEMO-literal cards' flag.

    Fails both ways: if the card takes the overwrite, and if the library
    default stops being main's no-overwrite arm.
    """
    cfg = _orca1_tke_config(iwm_enabled)
    assert cfg.nemo_mxl0_rmxl_min_overwrite is False
    assert TKEConfig().nemo_mxl0_rmxl_min_overwrite is False
    assert cfg.mxl0_min_m == 0.04
    # By VALUE, on the three columns the decision is about.  The calm column
    # is the one that moves; the windy and land columns must not.
    anchor = np.asarray(_mxl0_surface_anchor(
        cfg, jnp.asarray([0.10, 0.0, 0.07]), 1026.0, 9.80665,
        jnp.asarray([1.0, 1.0, 0.0])))
    assert float(anchor[1]) == 0.04
    assert float(anchor[0]) == 0.7951003609964353      # wet, 0.10 Pa
    assert float(anchor[2]) == 0.5565702526975047      # land, 0.07 Pa
    # and the overwrite arm — what the card must NOT take — is NEMO's value.
    overwritten = np.asarray(_mxl0_surface_anchor(
        cfg._replace(nemo_mxl0_rmxl_min_overwrite=True),
        jnp.asarray([0.0]), 1026.0, 9.80665, jnp.asarray([1.0])))
    assert float(overwritten[0]) == (1.0e-3 if iwm_enabled else 1.0e-8)


def test_nemo_literal_cards_take_the_rn_mxl0_overwrite():
    """The other half: GYRE and the two NEMO DINO cards DO select it.

    Their certified trajectories are pinned to the overwrite arm, so this is
    what keeps decision 72 from moving them.  ORCA2 is a separate test below,
    because building it needs its external deck.
    """
    from legoesm.ocean.experiments.dino import DINO_RECIPES
    from legoesm.ocean.fidelity.nemo_recipe import _nemo_tke_config

    gyre = _nemo_tke_config()
    assert gyre.nemo_mxl0_rmxl_min_overwrite is True
    # GYRE runs ln_zdfiwm=.FALSE., so the overwrite value is the DERIVED
    # rmxl_min = 1e-6/(c_k*sqrt(rn_emin)) = 1e-2 m, not rn_mxl0.
    assert float(_mxl0_anchor_floor(gyre)) == pytest.approx(1.0e-2, rel=1e-12)
    assert float(_mxl0_anchor_floor(gyre)) != gyre.mxl0_min_m
    for recipe in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
        assert DINO_RECIPES[recipe]["tke_nemo_mxl0_rmxl_min_overwrite"] is True


def test_orca2_card_inherits_the_rn_mxl0_overwrite():
    """ORCA2 takes the overwrite by inheriting the GYRE identity it specialises.

    It is asserted on the RESOLVED card rather than on the source, because the
    inheritance is the whole claim: the ORCA2 builder replaces several TKE
    fields and must not disturb this one.  Its floor is NEMO's forced
    rmxl_min = 1e-3 (ln_zdfiwm true on that deck), not rn_mxl0.  The external
    deck is a campaign input; skip where it is absent, as the neighbouring
    ORCA2 card test does.
    """
    from pathlib import Path

    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_orca2_zps_card,
    )

    deck = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
    if not deck.exists():
        pytest.skip("ORCA2 immutable input deck is not installed")
    tke = (build_orca2_zps_card(deck)
           .recipe.model_config.physics.vertical_mixing.tke)
    assert tke.nemo_mxl0_rmxl_min_overwrite is True
    assert float(_mxl0_anchor_floor(tke)) == 1.0e-3
    assert float(_mxl0_anchor_floor(tke)) != tke.mxl0_min_m


def test_orca2_card_states_the_after_ssh_form_and_the_cfl_cap():
    """Decisions 75 and 76 on the ORCA2 card, measured rather than argued.

    ``nemo_first_wzv_after_ssh`` is fail-closed: a card that reaches NEMO's
    first ``wzv`` call and leaves it unset RAISES.  ORCA2's own build runs the
    RK3 vector-invariant program -- ``stp2d.f90`` takes the "Vector Inv. Form"
    Coriolis arm and the "only KEG + ZAD in Vector Inv. Form" advection -- and
    that program leaves the previous step's linear extrapolation in the after
    slot (``stprk3.f90:241``), so the card must resolve ``rk3_extrapolated``.

    The lateral-mixing CFL cap must not be inherited from a library default
    (decision 75): the card is built twice with every ``enforce_cfl`` default
    the card could inherit flipped in between, and its resolved block must not
    move.  Reverting either statement makes this red.
    """
    import importlib.util
    from pathlib import Path

    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_orca2_zps_card,
    )

    deck = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
    if not deck.exists():
        pytest.skip("ORCA2 immutable input deck is not installed")

    config = build_orca2_zps_card(deck).recipe.model_config
    assert config.momentum_time_integrator == "rk3_ws"
    assert config.momentum_advection == "vector_invariant"
    assert config.nemo_first_wzv_after_ssh == "rk3_extrapolated"

    # Reuse the decision-75 flip harness rather than writing a second one.
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "_vortex_card_tests_for_orca2",
        root / "tests" / "ocean" / "unit" / "test_nemo_vortex_card.py")
    vortex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vortex)
    clean = build_orca2_zps_card(deck).recipe.model_config.physics
    with vortex._lateral_mixing_defaults_flipped():
        flipped = build_orca2_zps_card(deck).recipe.model_config.physics
    assert flipped.lateral_mixing == clean.lateral_mixing, (
        "ORCA2's resolved configuration follows a lateral-mixing library "
        "default; state the field on the card instead of re-pinning")
    assert clean.lateral_mixing.scheme == "none"
    assert clean.lateral_mixing.harmonic.enforce_cfl is False
    assert clean.lateral_mixing.biharmonic.enforce_cfl is True

    # ... and the after-SSH form is STATED on the ORCA2 card, not inherited
    # from the shared GYRE identity it specialises.  The shared builder is
    # made to hand back a sentinel; a card that merely inherited would carry
    # the sentinel through, and deleting the card's own line makes this red.
    import legoesm.ocean.fidelity.nemo_testcase_recipe as recipe_module

    original = recipe_module._model_config

    def _sentinel_base(*args, **kwargs):
        built = original(*args, **kwargs)
        # Poison ONLY the shared GYRE identity the ORCA2 branch builds on,
        # not the ORCA2 identity's own return value.
        if kwargs.get("whole_step_identity") == "gyre_vector_ene_c2":
            return built._replace(
                nemo_first_wzv_after_ssh="leapfrog_continuity")
        return built

    recipe_module._model_config = _sentinel_base
    try:
        stated = build_orca2_zps_card(deck).recipe.model_config
    finally:
        recipe_module._model_config = original
    assert stated.nemo_first_wzv_after_ssh == "rk3_extrapolated"


# --------------------------------------------------------------------------
# S5 — one meaning per nn_eice value, on every integration
# --------------------------------------------------------------------------

def test_fesom_eice_uses_the_shared_nemo_numbering():
    """FESOM must route through the shared helper, not its own mode-1 form."""
    import inspect

    from legoesm.ocean.physics.vertical_mixing import fesom_integration

    frac = jnp.asarray([0.0, 0.25, 0.9])
    mode1 = np.asarray(nemo_tke_effective_ice_fraction(frac, 1))
    mode2 = np.asarray(nemo_tke_effective_ice_fraction(frac, 2))
    np.testing.assert_allclose(mode1, np.tanh(10.0 * np.asarray(frac)),
                               rtol=1e-12)
    np.testing.assert_allclose(mode2, np.asarray(frac), rtol=1e-12)
    # the old FESOM meaning of 1 was the raw fraction; the two must differ
    assert not np.allclose(mode1, mode2)
    text = inspect.getsource(fesom_integration.make_tke_profiles_fesom)
    assert "nemo_tke_effective_ice_fraction" in text
    assert "_eice == 1 else" not in text


def test_fesom_builder_accepts_nn_eice_2_and_still_refuses_an_unknown():
    """Behavioural, not textual: the builder validates eice at build time."""
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.fesom_integration import (
        make_tke_profiles_fesom,
    )

    def build(eice):
        return make_tke_profiles_fesom(VerticalMixingConfig(
            scheme="tke", tke=TKEConfig(prognostic=True, eice=eice)))

    for eice in (0, 1, 2, 3):
        assert callable(build(eice))          # 2 raised before the fix
    with pytest.raises(ValueError, match="nn_eice"):
        build(5)


# --------------------------------------------------------------------------
# S4 — the literal NEMO EEN/ENE Coriolis fails closed on a curvilinear mesh
# --------------------------------------------------------------------------

def test_literal_een_coriolis_refuses_a_curvilinear_grid_without_ff_f():
    from legoesm.grids.latlon import create_latlon_geometry
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        nemo_een_ene_vertex_coriolis,
        vertex_coriolis,
    )

    grid = create_latlon_geometry(6, 8)
    # A rectilinear grid: V and F share a latitude, so the generic value IS
    # the F-point value and the fallback stays.
    np.testing.assert_array_equal(
        np.asarray(nemo_een_ene_vertex_coriolis(grid)),
        np.asarray(vertex_coriolis(grid)))
    folded = grid._replace(fold=grid.fold._replace(is_active=True))
    with pytest.raises(ValueError, match="ff_f"):
        nemo_een_ene_vertex_coriolis(folded)


# --------------------------------------------------------------------------
# B2 — the carried external mode is a CONFIG choice, not a state detail
# --------------------------------------------------------------------------

def test_carried_seed_is_selected_by_config_not_by_state_presence():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _carried_nemo_depth_mean,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=6, n_lon=8)
    z = create_ocean_z_star(n_levels=3, H_max=300.0)
    carried = rest_state_latlon_cgrid_ocean(
        grid, z, H_max=300.0, land_lat_threshold=70.0,
        nemo_prognostic_barotropic_velocity=True)
    plain = LatLonCGridOceanConfig()
    # The pair EXISTS but the config does not select the identity: the window
    # must still reduce.  Pre-fix this returned the carried pair.
    assert _carried_nemo_depth_mean(carried, jnp.float64, plain) is None
    nemo = plain._replace(barotropic=plain.barotropic._replace(
        nemo_prognostic_barotropic_state=True))
    assert _carried_nemo_depth_mean(carried, jnp.float64, nemo) is not None
    # Config selects it, state cannot provide it: fail closed, never reduce.
    bare = rest_state_latlon_cgrid_ocean(
        grid, z, H_max=300.0, land_lat_threshold=70.0)
    with pytest.raises(ValueError, match="carries no uu_b"):
        _carried_nemo_depth_mean(bare, jnp.float64, nemo)


def test_no_dino_recipe_allocates_the_rk3_carried_barotropic_pair():
    """Resolved state and resolved config, not dictionary keys.

    The carried external-mode pair is the RK3 stepper's storage contract: one
    slot, committed at the end of a step and read at the start of the next,
    which is only the window seed's level because ``stprk3.F90:213`` swaps the
    slot.  DINO compiles no ``key_RK3`` and runs ``stp_MLF``, whose rotation
    puts that value one level too new for its ``ln_bt_fw = .false.`` seed.
    Selecting it on the DINO cards cost a measured 3.42x on the certified
    from-rest month; see
    ``docs/ocean/fidelity/testcases/
    nemo_testcases_l2_gyre_dino_month_regression_receipt.md``.
    """
    from legoesm.ocean.experiments.dino import (
        DINOConfig,
        DINO_RECIPES,
        dino_config_for_recipe,
        dino_lat_lon_grid,
        dino_lat_lon_state,
    )
    from legoesm.ocean.vertical import create_ocean_z_star

    assert DINOConfig().nemo_prognostic_barotropic_state is False
    for recipe in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
        # Written down in the card, not merely left at the library default, so
        # that re-selecting it is a visible edit and this row goes red.
        assert DINO_RECIPES[recipe]["nemo_prognostic_barotropic_state"] is False
        assert dino_config_for_recipe(
            recipe).nemo_prognostic_barotropic_state is False

    plain = dino_config_for_recipe("legoesm_default")
    assert plain.nemo_prognostic_barotropic_state is False

    grid = dino_lat_lon_grid(plain, n_lon=6)
    z = create_ocean_z_star(n_levels=3, H_max=float(plain.H_deep))
    for cfg in (plain, dino_config_for_recipe("nemo_dino_kamm_mlf")):
        state = dino_lat_lon_state(grid, z, cfg)
        assert state.uu_b is None and state.vv_b is None


def test_the_nemo_testcase_cards_still_require_the_carried_pair():
    """The other half, so the DINO row above cannot pass by deleting it.

    GYRE and both tanks compile ``key_RK3`` (their ``cpp_*.fcm`` files), so
    for them the one-slot carry IS the window seed's level and the identity
    stays required.
    """
    import pytest

    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_gyre_zco_card,
        validate_nemo_testcase_card,
    )

    card = build_gyre_zco_card()
    validate_nemo_testcase_card(card)
    cfg = card.recipe.model_config
    assert cfg.barotropic.nemo_prognostic_barotropic_state is True

    # Non-vacuity: the validator, not a source string, is what refuses the
    # DINO arrangement on a key_RK3 card.
    planted = card._replace(
        recipe=card.recipe._replace(
            model_config=cfg._replace(
                barotropic=cfg.barotropic._replace(
                    nemo_prognostic_barotropic_state=False))))
    with pytest.raises(ValueError,
                       match="nemo_prognostic_barotropic_state"):
        validate_nemo_testcase_card(planted)


# --------------------------------------------------------------------------
# S8 — a version-3 barotropic history is refused in plain words
# --------------------------------------------------------------------------

def test_v3_bt_hist_archive_is_refused_with_a_readable_message():
    from pathlib import Path
    from legoesm.ocean.restart import _refuse_v3_deviation_bt_hist

    class _S:
        bt_hist = None

    plain = _S()
    assert _refuse_v3_deviation_bt_hist(plain, Path("x.npz")) is plain

    class _T:
        bt_hist = tuple(jnp.zeros((2, 2)) for _ in range(6))

    with pytest.raises(ValueError, match="must be REGENERATED"):
        _refuse_v3_deviation_bt_hist(_T(), Path("old_v3.npz"))
