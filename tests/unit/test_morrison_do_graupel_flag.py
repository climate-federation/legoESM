"""The Morrison graupel switch is an explicit deck setting on every lane.

CAM6's MG2 carries no graupel; our Morrison port does, by a leaf default that
no deck used to select.  ``morrison_do_graupel`` makes the choice visible: the
production deck names it (True, its measured behaviour); the one-variable
graupel-off arm is that deck plus ``--no-morrison-do-graupel``.
"""
from __future__ import annotations

import pathlib

import pytest

_AMIP = pathlib.Path(__file__).resolve().parents[2] / "config" / "amip"


def test_applier_sets_the_leaf_and_refuses_misuse():
    from legoesm.atmosphere.physics.microphysics.config import (
        MorrisonConfig,
        ThompsonConfig,
        apply_microphysics_experiment_flags,
    )
    off = apply_microphysics_experiment_flags(
        MorrisonConfig(), "morrison", morrison_do_graupel=False)
    assert off.do_graupel is False
    assert apply_microphysics_experiment_flags(
        MorrisonConfig(), "morrison").do_graupel is True
    with pytest.raises(ValueError, match="morrison_do_graupel"):
        apply_microphysics_experiment_flags(
            ThompsonConfig(), "thompson", morrison_do_graupel=False)
    with pytest.raises(TypeError, match="bool"):
        apply_microphysics_experiment_flags(
            MorrisonConfig(), "morrison", morrison_do_graupel="false")


def test_flat_default_is_locked_to_the_leaf():
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    from legoesm.forcing.amip_config import AMIPExperimentConfig
    assert (ExperimentConfig._field_defaults["morrison_do_graupel"]
            is MorrisonConfig().do_graupel
            is AMIPExperimentConfig().morrison_do_graupel is True)


def test_validate_strict_and_threading():
    from legoesm.atmosphere.physics.microphysics.config import (
        MorrisonConfig,
        ThompsonConfig,
    )
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import thread_morrison_scalars
    ExperimentConfig(microphysics="thompson").validate_strict()
    ExperimentConfig(microphysics="morrison",
                     morrison_do_graupel=False).validate_strict()
    with pytest.raises(ValueError, match="morrison_do_graupel"):
        ExperimentConfig(microphysics="thompson",
                         morrison_do_graupel=False).validate_strict()
    with pytest.raises(ValueError, match="must be a bool"):
        ExperimentConfig(microphysics="morrison",
                         morrison_do_graupel="false").validate_strict()
    base = MorrisonConfig()
    # untouched config: the leaf object passes through unchanged
    assert thread_morrison_scalars(ExperimentConfig(microphysics="morrison"),
                                   "morrison", base) is base
    off = thread_morrison_scalars(
        ExperimentConfig(microphysics="morrison", morrison_do_graupel=False),
        "morrison", base)
    assert off.do_graupel is False
    with pytest.raises(ValueError, match="only supported by the morrison"):
        thread_morrison_scalars(
            ExperimentConfig(microphysics="thompson", morrison_do_graupel=False),
            "thompson", ThompsonConfig())


def test_cli_and_amip_round_trip():
    from scripts.run.run_amip import (
        _postprocess_args,
        build_arg_parser,
        build_config_from_args,
    )
    from legoesm.driver.config import ExperimentConfig
    parser = build_arg_parser()
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--microphysics", "morrison",
        "--no-morrison-do-graupel"]), parser))
    assert cfg.morrison_do_graupel is False
    assert ExperimentConfig.from_amip_config(
        cfg.to_amip_config()).morrison_do_graupel is False
    d = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert d.morrison_do_graupel is True


def test_production_deck_names_the_switch():
    from scripts.run.run_amip import build_arg_parser
    from legoesm.driver.run_config_yaml import load_yaml_config
    parser = build_arg_parser()
    keys = load_yaml_config(_AMIP / "amip_production.yaml", parser)
    assert keys["morrison_do_graupel"] is True
    assert keys["microphysics"] == "morrison"
    # the graupel-off arm is this deck plus one CLI flag, which must win
    parser.set_defaults(**keys)
    assert parser.parse_args([]).morrison_do_graupel is True
    assert parser.parse_args(
        ["--no-morrison-do-graupel"]).morrison_do_graupel is False
