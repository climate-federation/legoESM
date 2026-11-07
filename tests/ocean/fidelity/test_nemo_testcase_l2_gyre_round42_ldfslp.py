"""Round-42 selector contract for NEMO's compiled GYRE ldf_slp path."""

from legoesm.ocean.experiments.dino import dino_config_for_recipe
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card


def test_gyre_selects_the_single_nemo_ldfslp_association_bundle():
    gm = build_nemo_testcase_card("GYRE-zco").recipe.model_config.gm_redi
    assert gm.mld_criterion == "n2_integral"
    assert gm.slope_n2_evaluation == "carried_step_entry"
    assert gm.slope_metric_evaluation == "nemo_reciprocal"
    assert gm.slope_face_thickness_evaluation == "nemo_qco_live"
    assert gm.slope_depth_evaluation == "nemo_qco_live_literal"
    assert gm.redi_a33_evaluation == "nemo_literal"


def test_tanks_do_not_execute_either_round42_changed_operator():
    for case in ("LOCK_EXCHANGE-zco", "OVERFLOW-zps"):
        cfg = build_nemo_testcase_card(case).recipe.model_config
        assert cfg.momentum_advection == "flux_form"
        assert cfg.gm_redi is None


def test_dino_nemo_card_keeps_its_preexisting_n2_integral_selection():
    oracle = dino_config_for_recipe("nemo_dino_kamm")
    non_oracle = dino_config_for_recipe("nemo_paper")
    assert oracle.gm_redi_mld_criterion == "n2_integral"
    assert non_oracle.gm_redi_mld_criterion == "rho_c"
