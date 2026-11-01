"""config/amip/amip_production_prognostic_nc.yaml: the production deck with the
CAM6 MG2 prognostic droplet number (the 2026-10-24 production setting), kept as
a non-default arm after production went back to the aerosol-diagnosed droplet
number (user decision 2026-11-01, #2216).

Pins that production resolves to the aerosol-diagnosed number, that the arm
resolves to the prognostic one, and that the two RESOLVED ExperimentConfigs
differ in exactly the two droplet-number fields; and that every deck including
production inherits the production setting.
"""
from __future__ import annotations

import pathlib

import pytest

from legoesm.driver.run_config_yaml import load_yaml_config, read_yaml_with_includes

_AMIP = pathlib.Path(__file__).resolve().parents[2] / "config" / "amip"
_PROD = _AMIP / "amip_production.yaml"
_ARM = _AMIP / "amip_production_prognostic_nc.yaml"
# Decks that include amip_production.yaml (grep "include: amip_production.yaml").
_INHERITORS = ("amip_production_emanuel.yaml", "amip_production_fv3duo_c24.yaml",
               "amip_production_latlon48.yaml",
               "amip_production_seaice_melt_albedo.yaml")


def _cfg(deck):
    from scripts.run import run_amip
    p = run_amip.build_arg_parser()
    p.set_defaults(**load_yaml_config(deck, p))
    return run_amip.build_config_from_args(p.parse_args([]))


def _flat(x, prefix=""):
    if hasattr(x, "_asdict"):
        out = {}
        for k, v in x._asdict().items():
            out.update(_flat(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): x}


def test_production_and_arm_resolve_to_the_two_droplet_paths():
    prod, arm = _cfg(_PROD), _cfg(_ARM)
    assert prod.morrison_prognostic_droplet_number is False
    assert prod.nc_from_aerosol is True
    assert arm.morrison_prognostic_droplet_number is True
    assert arm.nc_from_aerosol is False


def test_resolved_configs_differ_only_in_the_droplet_rows():
    prod, arm = _flat(_cfg(_PROD)), _flat(_cfg(_ARM))
    assert prod.keys() == arm.keys()
    diff = {k for k in prod if prod[k] != arm[k]}
    assert diff == {"morrison_prognostic_droplet_number", "nc_from_aerosol"}


def test_arm_deck_flips_only_the_two_yaml_rows():
    prod, arm = read_yaml_with_includes(_PROD), read_yaml_with_includes(_ARM)
    diff = {k for k in prod.keys() | arm.keys() if prod.get(k) != arm.get(k)}
    # plus the readjust width the arm names at its code default (#2216): it is
    # read only on the prognostic path, so it is not a third physics change
    assert diff == {"aerosol_ccn", "morrison_prognostic_droplet_number",
                    "morrison_mg2_drop_readjust_log_width"}
    from legoesm.driver.config import ExperimentConfig
    assert arm["morrison_mg2_drop_readjust_log_width"] == ExperimentConfig._field_defaults[
        "morrison_mg2_drop_readjust_log_width"]


@pytest.mark.parametrize("deck", _INHERITORS)
def test_inheriting_decks_carry_the_production_droplet_setting(deck):
    keys = read_yaml_with_includes(_AMIP / deck)
    assert keys["aerosol_ccn"] is True
    assert keys["morrison_prognostic_droplet_number"] is False
