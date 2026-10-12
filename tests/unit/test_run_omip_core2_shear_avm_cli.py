"""``--tke-shear-avm-weighting``: how K_M enters TKE shear production.

NEMO's zdf_sh2 sums avm on each u/v face (``avm(i+1)+avm(i)``) and multiplies
it into that face's shear before the 0.25 T-point combine
(``zdfsh2.F90:80-94``); the ORCA1 card's default multiplies one T-point K_M
into an already T-collapsed shear (``tpoint``).  The closure carries both
(``TKEConfig.tke_shear_avm_weighting``); this module pins the flag that
selects between them, and the tripole-only guard.
"""
from __future__ import annotations

import sys

import pytest


def _core2():
    import scripts.run.run_omip_core2 as core2
    return core2


def test_card_default_is_unchanged_without_the_flag():
    assert _core2().orca1_zdftke_config().tke_shear_avm_weighting == "tpoint"


def test_flag_reaches_the_closure_config_on_the_rk3_face_geometry():
    cfg = _core2().orca1_zdftke_config(shear_production="nemo_face_native_nbb2",
                                       shear_avm_weighting="nemo_face")
    assert cfg.tke_shear_avm_weighting == "nemo_face"
    assert cfg.tke_shear_production == "nemo_face_native_nbb2"


def test_nemo_face_without_face_native_geometry_raises():
    with pytest.raises(ValueError, match="face-native"):
        _core2().orca1_zdftke_config(shear_avm_weighting="nemo_face")


def test_unknown_value_raises():
    with pytest.raises(ValueError, match="shear_avm_weighting"):
        _core2().orca1_zdftke_config(shear_avm_weighting="face")


def test_it_rides_the_tripole_vmix_builder():
    vm = _core2().build_tripole_vmix_config(
        "tke", tke_shear_production="nemo_face_native_nbb2",
        tke_shear_avm_weighting="nemo_face")
    assert vm.tke.tke_shear_avm_weighting == "nemo_face"


def test_builder_refuses_it_without_the_tke_closure():
    with pytest.raises(ValueError, match="--tke-shear-avm-weighting"):
        _core2().build_tripole_vmix_config("kpp", tke_shear_avm_weighting="nemo_face")


def test_parser_exposes_both_choices_and_defaults_to_none():
    p = _core2()._build_arg_parser()
    assert p.parse_args(["--tke-shear-avm-weighting", "nemo_face"]
                        ).tke_shear_avm_weighting == "nemo_face"
    assert p.parse_args([]).tke_shear_avm_weighting is None
    with pytest.raises(SystemExit):
        p.parse_args(["--tke-shear-avm-weighting", "face"])


@pytest.mark.parametrize("argv", [
    ["--grid", "mpas", "--mpas-vmix", "tke"],
    ["--grid", "fesom", "--fesom-vmix", "legoesm_tke"],
    ["--grid", "tripole", "--tripole-vmix", "kpp"],
])
def test_main_refuses_it_off_the_tripole_tke_lane(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", *argv,
                                      "--tke-shear-avm-weighting", "nemo_face"])
    with pytest.raises(SystemExit, match="--tke-shear-avm-weighting"):
        _core2().main()


# --tke-prandtl-mode: NEMO nn_pdl=1 Prandtl on the avm-weighted p_sh2.

def test_prandtl_default_is_unchanged_without_the_flag():
    assert _core2().orca1_zdftke_config().prandtl_mode == "richardson"


def test_prandtl_flag_reaches_the_closure_config():
    vm = _core2().build_tripole_vmix_config("tke", tke_prandtl_mode="nemo_ri")
    assert vm.tke.prandtl_mode == "nemo_ri"


def test_prandtl_unknown_value_raises():
    with pytest.raises(ValueError, match="prandtl_mode"):
        _core2().orca1_zdftke_config(prandtl_mode="constant")


def test_prandtl_parser_and_builder_guard():
    p = _core2()._build_arg_parser()
    assert p.parse_args(["--tke-prandtl-mode", "nemo_ri"]).tke_prandtl_mode == "nemo_ri"
    assert p.parse_args([]).tke_prandtl_mode is None
    with pytest.raises(ValueError, match="--tke-prandtl-mode"):
        _core2().build_tripole_vmix_config("kpp", tke_prandtl_mode="nemo_ri")


@pytest.mark.parametrize("argv", [
    ["--grid", "mpas", "--mpas-vmix", "kpp"],
    ["--grid", "tripole", "--tripole-vmix", "kpp"],
])
def test_main_refuses_prandtl_off_the_tke_closure(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_omip_core2.py", *argv,
                                      "--tke-prandtl-mode", "nemo_ri"])
    with pytest.raises(SystemExit, match="--tke-prandtl-mode"):
        _core2().main()


def test_prandtl_flag_is_on_the_fesom_stage_allowlist():
    """FESOM's stage gate hard-rejects unlisted args; the flag must be listed
    or the fesom lane refuses it before the builder can receive it."""
    assert "tke_prandtl_mode" in _core2()._FESOM_WIRED_DESTS
