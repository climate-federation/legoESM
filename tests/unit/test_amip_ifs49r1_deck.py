"""config/amip/amip_ifs49r1.yaml: the production deck with OpenIFS 49r1 CLOUDSC,
the faithful CUMASTRN convection (votes VC2a, VC22-VC32) and ecRad 49r1
radiation (votes L1-L28).

Pins that the deck resolves to the IFS lane and adds no validate_strict error
to production's, and that its RESOLVED config differs from
production only in the rows the votes name; and that run_amip refuses
``--clouds resolved`` with any other microphysics (VC31b).
"""
from __future__ import annotations

import pathlib

import pytest

from legoesm.driver.run_config_yaml import load_yaml_config

_AMIP = pathlib.Path(__file__).resolve().parents[2] / "config" / "amip"
_PROD = _AMIP / "amip_production.yaml"
_IFS = _AMIP / "amip_ifs49r1.yaml"

# Resolved-config rows the IFS deck changes relative to production (VC25):
# the scheme selection, the IFS order, and the production rows that error or
# conflict under ifs_cloudsc.
_EXPECTED_DIFF = {
    "microphysics", "convection", "bechtold_use_ifs_ascent", "cld_macmic_num_steps",
    "morrison_ice_snow_d_auto", "morrison_warm_rain_scheme",
    "morrison_warm_rain_incloud",
    "homogeneous_ice_nucleation", "nc_from_aerosol", "use_clubb_cloud_fraction",
    "cloud_scheme", "clubb_liquid_partition",
    "cloud_cam6_rhmini", "cloud_cam6_rhmaxi", "cloud_cam6_rhminis",
    "cloud_cam6_rhmaxis",
    "zm_c0_lnd", "zm_c0_ocn", "zm_ke", "zm_dmpdz", "zm_tau", "zm_capelmt",
    "zm_detrain_phase_width_K", "zm_detrain_number_width_kgkg_s",
    # ecRad 49r1 radiation (votes L1, L18, L20, C2/C3): the scheme, the lane
    # aerosol row it never reads, the RRTMGP-only overhead layer that errors
    # under it, and ecRad's own CAMS aerosol climatology
    "radiation", "aerosol_forcing", "rrtmgp_overhead_layer",
    "ecrad_ifs_aerosol_file",
    # WMS non-orographic drag + METHOX (wms_ozone votes G1, M1-M2)
    "gravity_wave_drag", "ifs_methox", "ifs_ozone_chem",
}


def _args(deck, argv=()):
    from scripts.run import run_amip
    p = run_amip.build_arg_parser()
    p.set_defaults(**load_yaml_config(str(deck), p))
    return p, p.parse_args(list(argv))


def _cfg(deck):
    from scripts.run import run_amip
    return run_amip.build_config_from_args(_args(deck)[1])


def _flat(x, prefix=""):
    if hasattr(x, "_asdict"):
        out = {}
        for k, v in x._asdict().items():
            out.update(_flat(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): x}


def _strict_errors(cfg):
    try:
        cfg.validate_strict()
    except ValueError as e:
        return set(str(e).splitlines()[1:])
    return set()


def test_deck_resolves_to_the_ifs_lane_and_validates():
    cfg = _cfg(_IFS)
    assert cfg.microphysics == "ifs_cloudsc"
    assert cfg.convection == "bechtold" and cfg.bechtold_use_ifs_ascent is True
    assert cfg.cld_macmic_num_steps == 1
    assert cfg.cloud_scheme == "resolved"
    assert cfg.dycore.discretization == "mpas"
    assert cfg.radiation == "ecrad_ifs" and cfg.aerosol_forcing == "off"
    assert cfg.ecrad_ifs_aerosol_file.endswith("aerosol_cams_climatology_43R3a.nc")
    # the launcher supplies the land/IC files both decks need; with them
    # absent, the IFS deck raises no error production does not raise
    # (lane aerosols off: it may raise fewer)
    assert _strict_errors(cfg) <= _strict_errors(_cfg(_PROD))


def test_run_amip_accepts_resolved_clouds_with_cloudsc():
    from scripts.run import run_amip
    p = run_amip.build_arg_parser()
    a = p.parse_args(["--dataset", "analytical", "--clouds", "resolved",
                      "--microphysics", "ifs_cloudsc", "--convection", "bechtold",
                      "--bechtold-use-ifs-ascent"])
    run_amip._postprocess_args(a, p, [])


def test_resolved_config_differs_from_production_only_in_voted_rows():
    prod, ifs = _flat(_cfg(_PROD)), _flat(_cfg(_IFS))
    assert prod.keys() == ifs.keys()
    diff = {k for k in prod if prod[k] != ifs[k]}
    assert diff == _EXPECTED_DIFF


@pytest.mark.parametrize("micro", ["morrison", "sundqvist", "kessler"])
def test_run_amip_refuses_resolved_clouds_without_cloudsc(micro):
    from scripts.run import run_amip
    p = run_amip.build_arg_parser()
    a = p.parse_args(["--dataset", "analytical", "--clouds", "resolved",
                      "--microphysics", micro])
    with pytest.raises(SystemExit):
        run_amip._postprocess_args(a, p, [])


def test_deck_selects_wms_drag_and_methox():
    """Votes G1a/M2a: composite CAM6 orographic + 49r1 WMS, METHOX on."""
    cfg = _cfg(_IFS)
    assert cfg.gravity_wave_drag == "mcfarlane+ifs_wms"
    assert cfg.ifs_methox is True
    assert cfg.ifs_ozone_chem is True
