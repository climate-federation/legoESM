"""SWG round-1 card transcription and fail-closed execution boundary."""
from __future__ import annotations

import importlib.util
from pathlib import Path

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


_INVENTORY_PATH = (
    Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
    "testcases/nemo_testcase_l1_swg_namelist_inventory.py")
_INV_SPEC = importlib.util.spec_from_file_location("swg_namelist_inventory", _INVENTORY_PATH)
inventory = importlib.util.module_from_spec(_INV_SPEC); _INV_SPEC.loader.exec_module(inventory)
_COMPILED_READS = inventory.EXPECTED_CARD_READS

_ALL_CARD_FIELDS = (
    "sn_cfctl", "ln_timing", "ln_diacfl", "nn_isplt", "nn_jsplt", "nn_ictls",
    "nn_npfchk", "nn_ictle", "nn_jctls", "nn_jctle", "ln_read_cfg",
    "cn_domcfg", "ln_closea", "ln_write_cfg", "cn_domcfg_out",
    "cn_ocerst_indir", "cn_ocerst_outdir", "nn_stocklist", "ln_rst_list",
    "nn_no", "cn_exp", "cn_ocerst_in", "cn_ocerst_out", "nn_it000",
    "nn_itend", "nn_date0", "nn_time0", "nn_leapy", "nn_write", "nn_istate",
    "ln_rstart", "ln_1st_euler", "nn_rstctl", "nn_stock", "ln_mskland",
    "ln_clobber", "nn_chunksz", "ln_cfmeta", "ln_xios_read", "nn_wxios",
    "ln_top", "rn_Dt_s", "rn_atfp", "ln_c1d", "ln_meshmask", "ln_shuman",
    "ln_tile", "nn_ltile_i", "nn_ltile_j", "nn_nchunks_i", "nn_nchunks_j",
    "nn_nchunks_k", "ln_nc4zip", "nn_SWG", "jpkglo", "rn_theta_deg", "nn_gc",
    "rn_domsiz_m", "rn_dx_m", "rn_tau_pa", "rn_modified_grav_m_s2", "rn_rfr_s",
    "rn_beta_m_s", "rn_f0_s", "nn_dynldf_lap_typ", "ln_dynldf_lap_PM",
    "rn_abp", "nn_cnp", "rn_fsp_s", "ln_Iperio", "ln_Jperio", "key_xios",
    "key_qco", "key_vco_3d", "key_RK3", "nn_fsbc", "ln_usr_sbc", "ln_flx",
    "ln_blk", "ln_abl", "ln_cpl", "ln_mixcpl", "nn_components", "nn_ice",
    "ln_ice_embd", "ln_traqsr", "ln_dm2dc", "ln_rnf", "nn_fwb", "ln_ssr",
    "ln_apr_dyn", "ln_wave", "nn_lsm", "rn_shlat", "ln_vorlat", "ln_bdy",
    "nb_bdy", "ln_coords_file", "cn_coords_file", "ln_mask_file", "cn_mask_file",
    "cn_dyn2d", "nn_dyn2d_dta", "cn_dyn3d", "nn_dyn3d_dta", "cn_tra",
    "nn_tra_dta", "ln_tra_dmp", "ln_dyn3d_dmp", "rn_time_dmp_days",
    "rn_time_dmp_out_days", "cn_ice", "nn_ice_dta", "ln_vol", "nn_volctl",
    "nn_rimwidth", "ln_drg_OFF", "ln_dynadv_OFF", "ln_dynadv_vec", "nn_dynkeg",
    "ln_dynadv_cen2", "ln_dynadv_up3", "ln_dynvor_ene", "ln_dynvor_ens",
    "ln_dynvor_enT", "ln_dynvor_een", "ln_dynvor_mix", "nn_e3f_typ",
    "ln_dynvor_msk", "ln_vvl_zstar", "ln_vvl_ztilde", "ln_vvl_layer",
    "ln_vvl_ztilde_as_zstar", "ln_vvl_zstar_at_eqtor", "rn_ahe3",
    "rn_rst_e3t_days", "rn_lf_cutoff_days", "rn_zdef_max", "ln_vvl_dbg",
    "nn_vvl_interp", "ln_dynspg_exp", "nn_bt_flt", "rn_bt_alpha",
    "ln_dynldf_OFF", "nn_dynldf_typ", "ln_dynldf_lap", "ln_dynldf_blp",
    "ln_dynldf_lev", "ln_dynldf_hor", "ln_dynldf_iso", "nn_ahm_ijk_t",
    "rn_Uv_m_s", "rn_Lv_m", "rn_ahm_b_m2_s", "rn_csmc", "rn_minfac",
    "rn_maxfac", "ln_zdftke", "ln_zdfevd", "nn_evdm", "rn_evd_m2_s",
    "rn_avm0_m2_s", "rn_avt0_m2_s", "nn_avb", "nn_havtb", "nn_etau",
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
    assert card.recipe.model_config.physics.constants.g == 0.02
    assert card.recipe.model_config.g == 0.02
    assert card.recipe.model_config.outer_integrator == "swe_rk3"
    assert card.recipe.model_config.vorticity_scheme.scheme == "ens_total"
    assert card.recipe.model_config.vorticity_scheme.ln_dynvor_msk is True
    assert card.recipe.model_config.coriolis_scheme == "explicit_ab2"
    assert card.recipe.model_config.een_e3f_scheme == "nemo_avg4"
    assert card.recipe.model_config.een_metric_weighting == "nemo"
    assert card.recipe.model_config.een_q_boundary == "nemo_live"
    for field in (state.T, state.S, state.u, state.v, state.eta, state.uu_b, state.vv_b):
        assert not np.any(np.asarray(field.data))


@pytest.mark.parametrize("value", (None, "true", 1))
def test_swg_card_requires_explicit_boolean_ln_dynvor_msk(card, value):
    bad_vor = card.recipe.model_config.vorticity_scheme._replace(
        ln_dynvor_msk=value)
    bad_config = card.recipe.model_config._replace(vorticity_scheme=bad_vor)
    bad = card._replace(recipe=card.recipe._replace(model_config=bad_config))
    with pytest.raises(ValueError, match="compiled vor_ens total vorticity"):
        validate_nemo_testcase_card(bad)


def test_ens_vorticity_config_has_no_mask_default():
    from legoesm.ocean.state import ENSVorticityConfig

    with pytest.raises(TypeError):
        ENSVorticityConfig("ens_total")


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
    assert len(_COMPILED_READS) == 13
    assert sum(map(len, _COMPILED_READS.values())) == 131
    assert SwgResolvedNamelist._fields == _ALL_CARD_FIELDS
    _assert_engine_inventory(SwgResolvedNamelist._fields)
    assert SWG_DEVIATIONS == (("key_xios", True, False,
                               "conda-scalarmath reference toolchain"),)


@pytest.mark.parametrize(
    "field", [field for fields in _COMPILED_READS.values() for field in fields])
def test_resolved_inventory_plant_reports_a_removed_compiled_field(field):
    deficient = set(SwgResolvedNamelist._fields) - {field}
    with pytest.raises(AssertionError, match=field):
        _assert_engine_inventory(deficient)


def test_inventory_parser_directly_reads_multiline_fortran(tmp_path):
    source = tmp_path / "reader.f90"
    source.write_text(
        "      NAMELIST/namdemo/ first, second, & ! continued\n"
        "         & third\n"
        "      READ(unit,namdemo)\n")
    assert inventory.parse_namelist_declaration(source, "namdemo") == (
        "first", "second", "third")


@pytest.mark.skipif(
    not inventory.DEFAULT_SOURCE_ROOT.is_dir(), reason="compiled SWG source not mounted")
def test_source_enumerator_matches_all_131_compiled_reads():
    parsed = inventory.compiled_card_inventory(inventory.DEFAULT_SOURCE_ROOT)
    inventory.validate_inventory(parsed)
    assert parsed == SWG_COMPILED_NAMELIST_READS


def test_engine_read_defaults_match_the_resolved_oracle_echo():
    expected = {
        "sn_cfctl": (
            ("l_runstat", False), ("l_trcstat", False), ("l_obsstat", False),
            ("l_oceout", False), ("l_layout", False), ("l_prtctl", False),
            ("l_prttrc", False), ("l_oasout", False), ("procmin", 0),
            ("procmax", 1_000_000), ("procincr", 1), ("ptimincr", 1),
        ),
        "ln_timing": False, "ln_diacfl": False, "nn_isplt": 1,
        "nn_jsplt": 1, "nn_ictls": 0, "nn_npfchk": 0, "nn_ictle": 0,
        "nn_jctls": 0, "nn_jctle": 0,
        "ln_read_cfg": False, "cn_domcfg": "domain_cfg", "ln_closea": False,
        "ln_write_cfg": False, "cn_domcfg_out": "domain_cfg_out",
        "cn_ocerst_indir": ".", "cn_ocerst_outdir": ".",
        "nn_stocklist": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        "ln_rst_list": False, "nn_no": 0, "cn_exp": "SWG",
        "cn_ocerst_in": "restart", "cn_ocerst_out": "restart",
        "nn_date0": 10101, "nn_time0": 0, "ln_clobber": True,
        "nn_chunksz": 0, "ln_cfmeta": False, "ln_xios_read": False,
        "nn_wxios": 0, "ln_top": True,
        "ln_1st_euler": False, "nn_rstctl": 0, "nn_stock": 0,
        "ln_mskland": False, "rn_atfp": 0.1, "ln_c1d": False,
        "ln_tile": False, "nn_ltile_i": 99999, "nn_ltile_j": 10,
        "nn_nchunks_i": 4, "nn_nchunks_j": 4, "nn_nchunks_k": 31,
        "ln_nc4zip": False,
        "ln_abl": False, "ln_mixcpl": False, "nn_components": 0,
        "nn_ice": 0, "ln_ice_embd": False, "ln_dm2dc": False,
        "nn_fwb": 0, "ln_ssr": False, "ln_wave": False, "nn_lsm": 0,
        "ln_vorlat": False, "ln_dynvor_mix": False,
        "ln_bdy": False, "nb_bdy": 0, "ln_coords_file": True,
        "cn_coords_file": "coordinates.bdy.nc", "ln_mask_file": False,
        "cn_mask_file": "", "cn_dyn2d": "none", "nn_dyn2d_dta": 0,
        "cn_dyn3d": "none", "nn_dyn3d_dta": 0, "cn_tra": "none",
        "nn_tra_dta": 0, "ln_tra_dmp": False, "ln_dyn3d_dmp": False,
        "rn_time_dmp_days": 1.0, "rn_time_dmp_out_days": 1.0,
        "cn_ice": "none", "nn_ice_dta": 0, "ln_vol": False,
        "nn_volctl": 1, "nn_rimwidth": 10,
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


def test_execution_admits_the_complete_dispatched_program(card):
    assert SWG_UNMEASURED == ()
    validate_nemo_testcase_card_for_execution(card)


def test_validator_refuses_planted_card_drift(card, monkeypatch):
    monkeypatch.setattr(recipe, "SWG_UNMEASURED", ("planted",))
    with pytest.raises(ValueError, match="SWG_UNMEASURED"):
        validate_nemo_testcase_card(card)


def test_unknown_rotation_is_refused(monkeypatch):
    monkeypatch.setattr(recipe, "SWG_NAMELIST", SWG_NAMELIST._replace(rn_theta_deg=1.0))
    with pytest.raises(ValueError, match="rn_theta"):
        recipe.swg_horizontal_coordinates()
