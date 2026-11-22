"""config/amip/amip_production_{cam6_subgrid_growth,mg2_specified_nc_sed,
positivity_after_zm,cam6_cirrus}.yaml: the production deck with one ported ENSO-fork option
each.  Each RESOLVED ExperimentConfig differs from production in exactly its own
row, and production keeps every option off.
"""
from __future__ import annotations

import pathlib

import pytest

from legoesm.driver.run_config_yaml import load_yaml_config

_AMIP = pathlib.Path(__file__).resolve().parents[2] / "config" / "amip"
_PROD = _AMIP / "amip_production.yaml"
_ARMS = {
    "amip_production_cam6_subgrid_growth.yaml": {
        "cloud_cam6_do_subgrid_growth": (False, True)},
    "amip_production_mg2_specified_nc_sed.yaml": {
        "morrison_specified_nc_droplet_sedimentation": (False, True)},
    "amip_production_positivity_after_zm.yaml": {
        "positivity_after_zm": (False, True)},
    "amip_production_cam6_cirrus.yaml": {
        "morrison_cirrus_nucleation": ("fixed", "cam6_liu_penner"),
        "cirrus_aerosol_file": (None, "inputs/cesm_prescribed_aero/mam3_2000clim_mpas.npz")},
}


def _cfg(deck):
    from scripts.run import run_amip
    p = run_amip.build_arg_parser()
    p.set_defaults(**load_yaml_config(deck, p))
    return run_amip.build_config_from_args(p.parse_args([]))


@pytest.mark.parametrize("arm", sorted(_ARMS))
def test_arm_differs_from_production_in_its_own_row_only(arm):
    prod, cfg = _cfg(_PROD)._asdict(), _cfg(_AMIP / arm)._asdict()
    diff = {k: (prod[k], cfg[k]) for k in prod if prod[k] != cfg[k]}
    assert diff == _ARMS[arm]
