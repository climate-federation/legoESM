"""config/amip/amip_production_cam6_{conv_water_rad,snow_rad,rad_specified_nc}.yaml:
the production deck with one CAM6 cloud-radiation option each (ported from the
ENSO fork).  Each RESOLVED ExperimentConfig differs from production in exactly
its own rows, and production keeps every option off.
"""
from __future__ import annotations

import pathlib

import pytest

from legoesm.driver.run_config_yaml import load_yaml_config

_AMIP = pathlib.Path(__file__).resolve().parents[2] / "config" / "amip"
_PROD = _AMIP / "amip_production.yaml"
_ARMS = {
    "amip_production_cam6_conv_water_rad.yaml": {
        "cloud_cam6_conv_water_in_rad": (0, 1),
        "cloud_cam6_conv_water_frac_limit": (None, 0.01),
        "cloud_cam6_rad_gate_log_width": (None, 0.01)},
    "amip_production_cam6_snow_rad.yaml": {
        "cloud_cam6_snow_in_rad": (False, True),
        "cloud_cam6_snow_optics_file": ("", "inputs/legoesm/physprops/iceoptics_c080917.nc"),
        "cloud_cam6_rad_gate_log_width": (None, 0.01)},
    "amip_production_cam6_rad_specified_nc.yaml": {
        "cloud_cam6_rad_specified_nc": (False, True),
        "nc_from_aerosol": (True, False)},
}


def _cfg(deck):
    from scripts.run import run_amip
    p = run_amip.build_arg_parser()
    p.set_defaults(**load_yaml_config(deck, p))
    return run_amip.build_config_from_args(p.parse_args([]))


@pytest.mark.parametrize("arm", sorted(_ARMS))
def test_arm_differs_from_production_in_its_own_rows_only(arm):
    prod, cfg = _cfg(_PROD)._asdict(), _cfg(_AMIP / arm)._asdict()
    diff = {k: (prod[k], cfg[k]) for k in prod if prod[k] != cfg[k]}
    assert diff == _ARMS[arm]
