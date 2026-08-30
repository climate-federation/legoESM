"""Red-capable decision tests for the current-default climate re-battery."""
from scripts.validate.ocean_fidelity.dino_1226 import climate_rebattery_score as R


def test_mld_classifier_branches():
    assert R.mld_verdict(R.MLD_CONFIRM_MAX) == "CONFIRM"
    assert R.mld_verdict(R.MLD_REFUTE_MIN) == "REFUTE"
    assert R.mld_verdict(15.0) == "PARTIAL/INDETERMINATE"


def test_basin_classifier_branches_and_floor_priority():
    assert R.basin_verdict(R.HISTORICAL_BASIN_GAP, True) == "UNRESOLVED/FLOOR"
    assert R.basin_verdict(0.0, True) == "CONFIRMED"
    assert R.basin_verdict(0.0, False) == "INVALID_CURRENT"
    assert R.basin_verdict(
        R.HISTORICAL_BASIN_GAP - 3 * R.BASIN_FLOOR, True) == "REFUTED"


def test_wall_classifier_branches():
    assert R.wall_verdict(1.0, 0.1) == "CONFIRMED"
    assert R.wall_verdict(3.0, 0.5) == "REFUTED"
    assert R.wall_verdict(1.5, 0.2) == "UNRESOLVED"


def test_all_plants_fire():
    assert all(R.classifier_controls().values())


def test_registered_current_defaults_are_card_scoped():
    from legoesm.ocean.experiments.dino import dino_config_for_recipe

    for recipe in ("nemo_dino_kamm", "nemo_dino_kamm_mlf"):
        config = dino_config_for_recipe(recipe)
        for name, expected in R.current_defaults_for_recipe(recipe).items():
            if name == "bridge_tke":
                continue  # harness launch choice, not a DINOConfig field
            assert getattr(config, name) == expected, (recipe, name)


def test_fe_qco_admission_diff_is_exactly_the_mlf_only_pair():
    mlf = R.current_defaults_for_recipe("nemo_dino_kamm_mlf")
    fe = R.current_defaults_for_recipe("nemo_dino_kamm")
    assert {name for name in mlf if mlf[name] != fe[name]} == {
        "barotropic_diffusion_alpha",
        "zad_qco_evaluation", "wzv_call2_evaluation",
    }


def test_unknown_rebattery_recipe_is_red():
    import pytest
    with pytest.raises(RuntimeError, match="unsupported re-battery recipe"):
        R.current_defaults_for_recipe("unknown")
