"""SWG round-1 card transcription and fail-closed execution boundary."""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    SWG_COMPILED_NAMELIST_READS, SWG_DEVIATIONS, SWG_NAMELIST, SWG_UNMEASURED,
    SwgResolvedNamelist, build_swg_card, swg_horizontal_coordinates,
    swg_wet_mask, validate_nemo_testcase_card,
    validate_nemo_testcase_card_for_execution,
)


# Frozen independently from the object under test by reading the eight compiled
# SWE namelist readers cited on each row of SWG_COMPILED_NAMELIST_READS.
_COMPILED_READS = {
    "namrun": ("ln_1st_euler", "nn_rstctl", "nn_stock", "ln_mskland"),
    "namtile": ("ln_tile",),
    "namsbc": (
        "nn_fsbc", "ln_usr_sbc", "ln_flx", "ln_blk", "ln_abl", "ln_cpl",
        "ln_mixcpl", "nn_components", "nn_ice", "ln_ice_embd", "ln_traqsr",
        "ln_dm2dc", "ln_rnf", "nn_fwb", "ln_ssr", "ln_apr_dyn", "ln_wave",
        "nn_lsm",
    ),
    "namlbc": ("rn_shlat", "ln_vorlat"),
    "nam_vvl": (
        "ln_vvl_zstar", "ln_vvl_ztilde", "ln_vvl_layer",
        "ln_vvl_ztilde_as_zstar", "ln_vvl_zstar_at_eqtor", "rn_ahe3",
        "rn_rst_e3t_days", "rn_lf_cutoff_days", "rn_zdef_max", "ln_vvl_dbg",
        "nn_vvl_interp",
    ),
    "namdyn_adv": (
        "ln_dynadv_OFF", "ln_dynadv_vec", "nn_dynkeg", "ln_dynadv_cen2",
        "ln_dynadv_up3",
    ),
    "namdyn_vor": (
        "ln_dynvor_ens", "ln_dynvor_ene", "ln_dynvor_enT", "ln_dynvor_een",
        "nn_e3f_typ", "ln_dynvor_mix", "ln_dynvor_msk",
    ),
    "namdyn_ldf": (
        "ln_dynldf_OFF", "nn_dynldf_typ", "ln_dynldf_lap", "ln_dynldf_blp",
        "ln_dynldf_lev", "ln_dynldf_hor", "ln_dynldf_iso", "nn_ahm_ijk_t",
        "rn_Uv_m_s", "rn_Lv_m", "rn_ahm_b_m2_s", "rn_csmc", "rn_minfac",
        "rn_maxfac",
    ),
}

_ALL_CARD_FIELDS = (
    "nn_it000", "nn_itend", "nn_leapy", "nn_write", "nn_istate", "ln_rstart",
    "ln_1st_euler", "nn_rstctl", "nn_stock", "ln_mskland", "rn_Dt_s",
    "ln_meshmask", "ln_shuman", "ln_tile", "ln_read_cfg", "nn_SWG", "jpkglo",
    "rn_theta_deg", "nn_gc", "rn_domsiz_m", "rn_dx_m", "rn_tau_pa",
    "rn_modified_grav_m_s2", "rn_rfr_s", "rn_beta_m_s", "rn_f0_s",
    "nn_dynldf_lap_typ", "ln_dynldf_lap_PM", "rn_abp", "nn_cnp", "rn_fsp_s",
    "ln_Iperio", "ln_Jperio", "key_xios", "key_qco", "key_vco_3d", "key_RK3",
    "nn_fsbc", "ln_usr_sbc", "ln_flx", "ln_blk", "ln_abl", "ln_cpl",
    "ln_mixcpl", "nn_components", "nn_ice", "ln_ice_embd", "ln_traqsr",
    "ln_dm2dc", "ln_rnf", "nn_fwb", "ln_ssr", "ln_apr_dyn", "ln_wave",
    "nn_lsm", "rn_shlat", "ln_vorlat", "ln_drg_OFF", "ln_dynadv_OFF",
    "ln_dynadv_vec", "nn_dynkeg", "ln_dynadv_cen2", "ln_dynadv_up3",
    "ln_dynvor_ene", "ln_dynvor_ens", "ln_dynvor_enT", "ln_dynvor_een",
    "ln_dynvor_mix", "nn_e3f_typ", "ln_dynvor_msk", "ln_vvl_zstar",
    "ln_vvl_ztilde", "ln_vvl_layer", "ln_vvl_ztilde_as_zstar",
    "ln_vvl_zstar_at_eqtor", "rn_ahe3", "rn_rst_e3t_days",
    "rn_lf_cutoff_days", "rn_zdef_max", "ln_vvl_dbg", "nn_vvl_interp",
    "ln_dynspg_exp", "nn_bt_flt", "rn_bt_alpha", "ln_dynldf_OFF",
    "nn_dynldf_typ", "ln_dynldf_lap", "ln_dynldf_blp", "ln_dynldf_lev",
    "ln_dynldf_hor", "ln_dynldf_iso", "nn_ahm_ijk_t", "rn_Uv_m_s", "rn_Lv_m",
    "rn_ahm_b_m2_s", "rn_csmc", "rn_minfac", "rn_maxfac", "ln_zdftke",
    "ln_zdfevd", "nn_evdm", "rn_evd_m2_s", "rn_avm0_m2_s", "rn_avt0_m2_s",
    "nn_avb", "nn_havtb", "nn_etau",
)


def _assert_engine_inventory(fields):
    expected = {field for group in _COMPILED_READS.values() for field in group}
    missing = sorted(expected - set(fields))
    assert not missing, f"missing compiled SWE namelist reads: {missing}"


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    old = get_policy(); set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(old)


@pytest.fixture(scope="module")
def card():
    return build_swg_card()


def test_card_shape_constants_and_initial_state(card):
    state = card.recipe.initial_state
    assert card.case == "SWG-zco"
    assert (card.dt_s, card.n_steps, card.dummy_bottom_records) == (3600.0, 172800, 1)
    assert state.eta.data.shape == (84, 84)
    assert state.u.data.shape == (84, 85, 1)
    assert state.v.data.shape == (85, 84, 1)
    assert int(np.count_nonzero(card.recipe.land_mask)) == 6400
    assert card.recipe.z_coord.n_levels == 1
    assert card.recipe.model_config.constants.g == 0.02
    for field in (state.T, state.S, state.u, state.v, state.eta, state.uu_b, state.vv_b):
        assert not np.any(np.asarray(field.data))


def test_source_coordinate_formula_and_absolute_y_coriolis():
    src = swg_horizontal_coordinates()
    assert (src["ni"], src["nj"], src["spacing_m"]) == (84, 84, 25000.0)
    assert src["glamt"][0, 0] == src["gphit"][0, 0] == -12500.0
    assert src["glamu"][0, 0] == src["gphiv"][0, 0] == 0.0
    want = SWG_NAMELIST.rn_f0_s + SWG_NAMELIST.rn_beta_m_s * np.abs(src["gphit"])
    assert np.array_equal(src["ff_t"], want)
    signed = SWG_NAMELIST.rn_f0_s + SWG_NAMELIST.rn_beta_m_s * src["gphit"]
    assert not np.array_equal(signed, want)


def test_wet_mask_prediction_and_plant():
    wet = swg_wet_mask()
    jj, ii = np.where(wet > 0.5)
    assert (wet.sum(), jj.min(), jj.max(), ii.min(), ii.max()) == (6400, 1, 80, 1, 80)
    planted = wet.copy(); planted[1, 1] = 0.0
    assert planted.sum() == 6399


def test_resolved_inventory_is_the_frozen_compiled_and_card_inventory():
    assert SWG_COMPILED_NAMELIST_READS == _COMPILED_READS
    assert SwgResolvedNamelist._fields == _ALL_CARD_FIELDS
    _assert_engine_inventory(SwgResolvedNamelist._fields)
    assert SWG_DEVIATIONS == (("key_xios", True, False,
                               "conda-scalarmath reference toolchain"),)


@pytest.mark.parametrize("field", ["ln_abl", "ln_tile"])
def test_resolved_inventory_plant_reports_a_removed_compiled_field(field):
    deficient = set(SwgResolvedNamelist._fields) - {field}
    with pytest.raises(AssertionError, match=field):
        _assert_engine_inventory(deficient)


def test_engine_read_defaults_match_the_resolved_oracle_echo():
    expected = {
        "ln_1st_euler": False, "nn_rstctl": 0, "nn_stock": 0,
        "ln_mskland": False, "ln_tile": False,
        "ln_abl": False, "ln_mixcpl": False, "nn_components": 0,
        "nn_ice": 0, "ln_ice_embd": False, "ln_dm2dc": False,
        "nn_fwb": 0, "ln_ssr": False, "ln_wave": False, "nn_lsm": 0,
        "ln_vorlat": False, "ln_dynvor_mix": False,
        "ln_vvl_zstar": True, "ln_vvl_ztilde": False,
        "ln_vvl_layer": False, "ln_vvl_ztilde_as_zstar": False,
        "ln_vvl_zstar_at_eqtor": False, "rn_ahe3": 0.0,
        "rn_rst_e3t_days": 30.0, "rn_lf_cutoff_days": 5.0,
        "rn_zdef_max": 0.9, "ln_vvl_dbg": False, "nn_vvl_interp": 2,
        "ln_dynldf_OFF": False, "nn_dynldf_typ": 0,
        "ln_dynldf_blp": False, "ln_dynldf_hor": False,
        "ln_dynldf_iso": False, "rn_ahm_b_m2_s": 0.0,
        "rn_csmc": 3.5, "rn_minfac": 1.0, "rn_maxfac": 1.0,
    }
    assert {name: getattr(SWG_NAMELIST, name) for name in expected} == expected


def test_execution_refuses_every_unbuilt_oracle_arm(card):
    assert len(SWG_UNMEASURED) == 3
    with pytest.raises(ValueError, match="ENS total-vorticity"):
        validate_nemo_testcase_card_for_execution(card)
    for item in SWG_UNMEASURED:
        assert item in str(card.unmeasured_features)


def test_validator_refuses_planted_card_drift(card, monkeypatch):
    monkeypatch.setattr(recipe, "SWG_UNMEASURED", ("planted",))
    with pytest.raises(ValueError, match="SWG_UNMEASURED"):
        validate_nemo_testcase_card(card)


def test_unknown_rotation_is_refused(monkeypatch):
    monkeypatch.setattr(recipe, "SWG_NAMELIST", SWG_NAMELIST._replace(rn_theta_deg=1.0))
    with pytest.raises(ValueError, match="rn_theta"):
        recipe.swg_horizontal_coordinates()
