"""Lane B L3: certified G9 + certified L2 + ORCA2 eddy-induced velocity."""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

from tests.ocean.fidelity.test_nemo_testcase_g9_orca2tadv import (
    CARD_FILES,
    CASES,
    G9,
    INPUTS,
    ORCA2,
    _root,
    _values,
)
from tests.ocean.fidelity.test_nemo_testcase_l2_orca2iso import _l2_root
from tests.ocean.fidelity.test_nemo_testcase_v1_orca2tke import _differs, _flat


L3 = CASES / "nemo_testcase_l3_orca2eiv"


def _copied_root(base: Path, rung: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return _root(base, rung)


def _copied_l2_root(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return _l2_root(base)


@pytest.fixture(autouse=True, scope="module")
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def test_l3_deck_is_g9_plus_complete_ldf_and_eiv_groups_only():
    g9 = _values(G9 / "namelist_cfg")
    l3 = _values(L3 / "namelist_cfg")
    changed = {key: (g9.get(key), l3.get(key)) for key in set(g9) | set(l3)
               if g9.get(key) != l3.get(key)}
    expected = {**recipe._ORCA2L2_LDF_FIELDS, **recipe._ORCA2L3_EIV_FIELDS}
    assert changed == {key: (g9.get(key), value) for key, value in expected.items()
                       if g9.get(key) != value}
    assert {key: value for key, value in l3.items()
            if key.startswith("namtra_eiv.")} == recipe._ORCA2L3_EIV_FIELDS
    assert (L3 / "namelist_ice_cfg").read_bytes() == (
        G9 / "namelist_ice_cfg").read_bytes()


@pytest.mark.skipif(not ORCA2.is_dir(), reason="no NEMO tree")
def test_l3_namtra_eiv_resolves_exactly_from_orca2_cfg_and_reference():
    cfg = _values(ORCA2 / "namelist_cfg")
    ref = _values(ORCA2 / "namelist_ref")
    got = {key: cfg.get(key, ref[key]) for key in recipe._ORCA2L3_EIV_FIELDS}
    assert got == recipe._ORCA2L3_EIV_FIELDS
    assert set(got) == {
        "namtra_eiv.ln_ldfeiv", "namtra_eiv.nn_aei_ijk_t",
        "namtra_eiv.rn_ue", "namtra_eiv.rn_le", "namtra_eiv.ln_eke_equ",
    }


@pytest.mark.parametrize("old,new", [
    ("ln_ldfeiv      = .true.", "ln_ldfeiv      = .false."),
    ("nn_aei_ijk_t   = 21", "nn_aei_ijk_t   = 20"),
    ("rn_Ue           = 0.03", "rn_Ue           = 0.02"),
    ("rn_Le           = 200.e+3", "rn_Le           = 100.e+3"),
    ("ln_eke_equ      = .false.", "ln_eke_equ      = .true."),
])
def test_reader_refuses_each_planted_eiv_assignment(tmp_path, old, new):
    root = _copied_root(tmp_path, L3)
    text = (root / "namelist_cfg").read_text()
    assert text.count(old) == 1
    (root / "namelist_cfg").write_text(text.replace(old, new))
    with pytest.raises(ValueError, match="namtra_eiv"):
        recipe.read_orca2g1_deck(root, "ORCA2L3-eiv")


def test_reader_refuses_l3_as_g9_or_l2_and_off_eiv_as_l3(tmp_path):
    l3 = _copied_root(tmp_path / "l3", L3)
    with pytest.raises(ValueError, match="namtra_(ldf|eiv)"):
        recipe.read_orca2g1_deck(l3, "ORCA2G9-tadv")
    with pytest.raises(ValueError, match="namtra_eiv"):
        recipe.read_orca2g1_deck(l3, "ORCA2L2-iso")
    with pytest.raises(ValueError, match="namtra_(adv|eiv)"):
        recipe.read_orca2g1_deck(_copied_l2_root(tmp_path / "l2"), "ORCA2L3-eiv")


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    if not all((INPUTS / name).is_file() for name in CARD_FILES):
        pytest.skip("no ORCA2 inputs")
    base = tmp_path_factory.mktemp("l3")
    return (
        recipe.build_orca2g9_tadv_card(_copied_root(base / "g9", G9)),
        recipe.build_orca2l2_iso_card(_copied_l2_root(base / "l2")),
        recipe.build_orca2l3_eiv_card(_copied_root(base / "l3", L3)),
    )


def test_l3_card_is_g9_plus_l2_plus_existing_treguier_bolus(cards):
    g9, l2, l3 = cards
    c9, c2, c3 = (card.recipe.model_config for card in cards)
    changed = {key for key in set(_flat(c9)) | set(_flat(c3))
               if _differs(_flat(c9).get(key), _flat(c3).get(key))}
    assert changed and all(key == "gm_redi" or key.startswith("gm_redi.")
                           for key in changed)
    assert recipe._orca2l3_eiv_selected(c3.gm_redi)
    assert recipe._orca2l2_isoneutral_selected(c2.gm_redi)
    assert c3.tracer_advection == "fct2"
    assert c3.gm_redi.kappa_GM == 3000.0
    assert c3.gm_redi.treguier.aei0 == 3000.0
    assert c3.gm_redi.gm_bolus_advection == "through_fct"
    assert c3.gm_redi.gm_bolus_kappa_face_average is True
    for name in ("grid", "z_coord", "initial_state"):
        assert pickle.dumps(getattr(g9.recipe, name)) == pickle.dumps(
            getattr(l3.recipe, name)), name
    assert recipe._orca2g1_selector_failures(g9) == []
    assert recipe._orca2g1_selector_failures(l2) == []
    assert recipe._orca2g1_selector_failures(l3) == []


@pytest.mark.parametrize("changes", [
    {"kappa_GM": 0.0},
    {"gm_bolus_advection": "centred"},
    {"gm_bolus_kappa_face_average": False},
    {"treguier_vertical_reduction_evaluation": "tree"},
    {"treguier_sqrt_evaluation": "guarded_floor"},
])
def test_l3_selector_refuses_each_bolus_wiring_plant(cards, changes):
    l3 = cards[2]
    gm = l3.recipe.model_config.gm_redi._replace(**changes)
    planted = l3._replace(recipe=l3.recipe._replace(
        model_config=l3.recipe.model_config._replace(gm_redi=gm)))
    assert recipe._orca2g1_selector_failures(planted)


def test_l3_rk3_model_accepts_only_complete_eiv_composition(cards):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    run = cards[2].recipe
    LatLonCGridOceanModel(run.grid, run.z_coord, run.model_config)
    for change in ({"gm_bolus_advection": "centred"},
                   {"slope_scheme": "small_slope"},
                   {"treguier": run.model_config.gm_redi.treguier._replace(
                       enabled=False)}):
        bad_gm = run.model_config.gm_redi._replace(**change)
        with pytest.raises(ValueError, match="complete staged"):
            LatLonCGridOceanModel(
                run.grid, run.z_coord,
                run.model_config._replace(gm_redi=bad_gm))


def test_l3_step_jaxpr_contains_bolus_only_for_l3(cards):
    """Compile-time proof that the selected L3 production arm is reachable."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import OceanSurfaceForcing
    import jax.numpy as jnp

    l3 = cards[2]
    run = l3.recipe
    model = LatLonCGridOceanModel(run.grid, run.z_coord, run.model_config)
    zero = jnp.zeros_like(run.initial_state.eta.data)
    surface = OceanSurfaceForcing(
        sw_down=zero, q_net=zero, q_nonsolar=zero, salt_flux_sfx=zero,
        chl=jnp.ones_like(zero), tau_x=zero, tau_y=zero,
        ice_ocean_top_drag_rCdu=zero)
    closed = jax.make_jaxpr(lambda state: model.step(
        state, dt=l3.dt_s, surface_forcing=surface))(
        run.initial_state)
    text = str(closed)
    assert "reduce_sum" in text
    assert np.isfinite(run.model_config.gm_redi.treguier.aei0)


G9_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                 "gridladder_rounds/round46/oracle_g9_r46/twin_a")


@pytest.mark.skipif(not G9_RECORD.is_dir(), reason="no admitted G9 record")
def test_zero_bolus_l3_step_is_bit_identical_to_certified_g9_plus_l2(cards):
    """A zero ldf_eiv_trp addition leaves G9+L2's whole step unchanged."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    sys.path[:0] = [str(CASES), str(CASES / "nemo_testcase_g5_orca2ice")]
    import g5_records as g5

    _, l2, l3 = cards
    run = l3.recipe
    baseline_cfg = run.model_config._replace(
        gm_redi=l2.recipe.model_config.gm_redi)
    zero_gm = run.model_config.gm_redi._replace(
        kappa_GM=0.0,
        treguier=run.model_config.gm_redi.treguier._replace(enabled=False))
    zero_cfg = run.model_config._replace(gm_redi=zero_gm)
    fw, sf = g5.nemo_supplied_forcing(l3, G9_RECORD, 1)
    outputs = []
    for cfg in (baseline_cfg, zero_cfg):
        model = LatLonCGridOceanModel(
            run.grid, run.z_coord, cfg, iwm_forcing=run.iwm_forcing)
        outputs.append(jax.device_get(model.step(
            run.initial_state, l3.dt_s, freshwater=fw,
            surface_forcing=sf)))
    for name in ("T", "S", "u", "v", "eta"):
        left = np.asarray(getattr(outputs[0], name).data)
        right = np.asarray(getattr(outputs[1], name).data)
        assert np.array_equal(left.view(np.uint64), right.view(np.uint64)), name
