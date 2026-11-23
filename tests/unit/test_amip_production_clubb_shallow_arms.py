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
