"""The two opt-in CLUBB/MG2 arms ported from the ENSO fork
(amip_production_clubb_host_omega.yaml, amip_production_boutle_relvar.yaml):
each RESOLVED ExperimentConfig differs from production only in its own rows."""
from __future__ import annotations

import pathlib

import pytest

from legoesm.driver.run_config_yaml import load_yaml_config

_AMIP = pathlib.Path(__file__).resolve().parents[2] / "config" / "amip"


def _cfg(deck):
    from scripts.run import run_amip
    p = run_amip.build_arg_parser()
    p.set_defaults(**load_yaml_config(_AMIP / deck, p))
    return run_amip.build_config_from_args(p.parse_args([]))


def _flat(x, prefix=""):
    if hasattr(x, "_asdict"):
        out = {}
        for k, v in x._asdict().items():
            out.update(_flat(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): x}


@pytest.mark.parametrize("deck, rows", [
    ("amip_production_clubb_host_omega.yaml", {"clubb_host_omega": True}),
    ("amip_production_boutle_relvar.yaml", {"morrison_relvar_scheme": "boutle2014"}),
    ("amip_production_clubb_mf.yaml", {"clubb_mf": True}),
])
def test_arm_differs_from_production_only_in_its_rows(deck, rows):
    prod, arm = _flat(_cfg("amip_production.yaml")), _flat(_cfg(deck))
    assert prod.keys() == arm.keys()
    assert {k: arm[k] for k in prod if prod[k] != arm[k]} == rows


def test_arms_build_their_physics():
    from legoesm.driver.physics_pipeline import turbulence_config_for, thread_morrison_scalars
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    assert turbulence_config_for(_cfg("amip_production_clubb_host_omega.yaml")).host_omega is True
    assert turbulence_config_for(_cfg("amip_production_clubb_mf.yaml")).clubb_mf is True
    m = thread_morrison_scalars(_cfg("amip_production_boutle_relvar.yaml"), "morrison",
                                MorrisonConfig())
    assert m.relvar_scheme == "boutle2014"


@pytest.mark.parametrize("line, cli, want", [
    ("cloud_clubb_mf_cldfrac_fac: 2.5", [], 2.5),
    ("cloud_clubb_mf_cldfrac_fac: null", [], None),
    ("cloud_clubb_mf_cldfrac_fac: 2.5", ["--cloud-clubb-mf-cldfrac-fac", "4"], 4.0),
])
def test_cldfrac_fac_yaml_round_trip(tmp_path, line, cli, want):
    """YAML deck value (number / null) reaches ExperimentConfig; the CLI flag
    overrides the deck."""
    from scripts.run import run_amip
    deck = tmp_path / "deck.yaml"
    deck.write_text(f"include: {_AMIP / 'amip_production_clubb_mf.yaml'}\n{line}\n")
    p = run_amip.build_arg_parser()
    p.set_defaults(**load_yaml_config(deck, p))
    cfg = run_amip.build_config_from_args(p.parse_args(cli))
    assert cfg.cloud_clubb_mf_cldfrac_fac == want and cfg.clubb_mf is True


def test_bsort_arm_differs_from_mf_only_in_one_row():
    mf = _flat(_cfg("amip_production_clubb_mf.yaml"))
    arm = _flat(_cfg("amip_production_clubb_mf_bsort.yaml"))
    assert mf.keys() == arm.keys()
    assert {k: arm[k] for k in mf if mf[k] != arm[k]} == {"clubb_mf_bsort": True}
    from legoesm.driver.physics_pipeline import turbulence_config_for
    tc = turbulence_config_for(_cfg("amip_production_clubb_mf_bsort.yaml"))
    assert tc.clubb_mf and tc.clubb_mf_bsort


@pytest.mark.parametrize("args, want", [([], False), (["--clubb-mf-bsort"], True),
                                       (["--no-clubb-mf-bsort"], False)])
def test_bsort_cli(args, want):
    from scripts.run import run_amip
    parser = run_amip.build_arg_parser()
    parser.set_defaults(**load_yaml_config(_AMIP / "amip_production_clubb_mf.yaml", parser))
    assert run_amip.build_config_from_args(parser.parse_args(args)).clubb_mf_bsort is want


@pytest.mark.parametrize("value", [True, "true", 1])
def test_bsort_strict_refuses_without_mf_or_non_bool(value):
    cfg = _cfg("amip_production_clubb_mf.yaml")._replace(clubb_mf=False, clubb_mf_bsort=value)
    with pytest.raises(ValueError, match="clubb_mf_bsort"):
        cfg.validate_strict()


def test_bsort_builder_refuses_without_mf_and_ignored_override():
    from legoesm.driver.physics_pipeline import turbulence_config_for
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    cfg = _cfg("amip_production_clubb_mf_bsort.yaml")
    with pytest.raises(ValueError, match="clubb_mf_bsort"):
        turbulence_config_for(cfg._replace(clubb_mf=False))
    with pytest.raises(ValueError, match="clubb_mf_bsort"):
        turbulence_config_for(cfg._replace(turbulence_override=TurbulenceConfig(
            scheme="clubb", clubb_mf=True)))
