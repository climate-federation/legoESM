"""Unit tests for the --params calibration loader (issue #691).

The loader takes a ``{qualified_name: value}`` calibration mapping (the same
keying the parameter registry + training output use) and splices it into the
matching nested ``*Config`` NamedTuple, validating existence + bounds and
raising loudly on anything it cannot route safely.
"""
from __future__ import annotations

from typing import NamedTuple

import pytest

from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.driver.run_config_yaml import (
    apply_params_to_config,
    build_atm_scalar_param_map,
    load_params_config,
)

_QNAME = "atm.clouds.CloudConfig.q_c_diagnostic"  # bounds (5e-5, 1e-3), tier 1


class _Wrapper(NamedTuple):
    clouds: CloudConfig
    other: int = 5


class _NoCloud(NamedTuple):
    other: int = 5


def test_registry_key_exists():
    """Guard: the qualified name these tests use is really in the registry."""
    from legoesm.training.param_collector import build_registry
    assert _QNAME in {m.qualified_name for m in build_registry()}


def test_apply_routes_override_into_nested_config():
    cfg = _Wrapper(clouds=CloudConfig(scheme="sundqvist"))
    out = apply_params_to_config(cfg, {_QNAME: 3.0e-4}, driver="test")
    assert out.clouds.q_c_diagnostic == 3.0e-4
    assert out.other == 5  # untouched


def test_empty_params_is_noop():
    cfg = _Wrapper(clouds=CloudConfig(scheme="sundqvist"))
    assert apply_params_to_config(cfg, {}, driver="test") is cfg


def test_unknown_parameter_raises():
    cfg = _Wrapper(clouds=CloudConfig(scheme="sundqvist"))
    with pytest.raises(SystemExit):
        apply_params_to_config(cfg, {"atm.clouds.CloudConfig.not_a_field": 1.0},
                               driver="test")


def test_out_of_bounds_value_raises():
    cfg = _Wrapper(clouds=CloudConfig(scheme="sundqvist"))
    with pytest.raises(SystemExit):
        apply_params_to_config(cfg, {_QNAME: 999.0}, driver="test")  # >> 1e-3


def test_absent_target_config_raises():
    # The scheme's config class is not present in this run's config tree.
    with pytest.raises(SystemExit):
        apply_params_to_config(_NoCloud(), {_QNAME: 3.0e-4}, driver="test")


def test_load_params_config_reads_yaml(tmp_path):
    p = tmp_path / "params.yaml"
    p.write_text(f"{_QNAME}: 3.0e-4\n")
    params = load_params_config(str(p))
    assert params[_QNAME] == 3.0e-4


def test_load_params_config_rejects_non_mapping(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(SystemExit):
        load_params_config(str(p))


def test_quoted_numeric_value_is_coerced():
    # A YAML-quoted scalar ("3.0e-4") must coerce, not TypeError on the bounds
    # check (codex #691 round 2).
    cfg = _Wrapper(clouds=CloudConfig(scheme="sundqvist"))
    out = apply_params_to_config(cfg, {_QNAME: "3.0e-4"}, driver="test")
    assert out.clouds.q_c_diagnostic == 3.0e-4


def test_quoted_out_of_bounds_raises_cleanly():
    cfg = _Wrapper(clouds=CloudConfig(scheme="sundqvist"))
    with pytest.raises(SystemExit):  # clean validation error, not TypeError
        apply_params_to_config(cfg, {_QNAME: "999"}, driver="test")


def test_non_numeric_value_raises():
    cfg = _Wrapper(clouds=CloudConfig(scheme="sundqvist"))
    with pytest.raises(SystemExit):
        apply_params_to_config(cfg, {_QNAME: "not-a-number"}, driver="test")


def test_same_class_name_different_module_does_not_cross_route():
    # Two DIFFERENT TKEConfig classes (ocean vertical-mixing vs atmosphere
    # turbulence) must route by (module, class), never by bare name (codex #691).
    from legoesm.atmosphere.physics.turbulence.config import TKEConfig as AtmTKE
    from legoesm.ocean.physics.vertical_mixing.config import TKEConfig as OceanTKE

    class _TwoTKE(NamedTuple):
        ocean_tke: OceanTKE
        atm_tke: AtmTKE

    cfg = _TwoTKE(ocean_tke=OceanTKE(), atm_tke=AtmTKE())
    out = apply_params_to_config(
        cfg, {"ocean.vm.tke.alpha_tke": 50.0}, driver="test")
    assert out.ocean_tke.alpha_tke == 50.0     # ocean one updated
    assert out.atm_tke == cfg.atm_tke          # atmosphere one untouched


# --- atmosphere flat-scalar map (run_amip / run_coupled), issue #691 --------

def test_build_atm_scalar_param_map_is_valid_and_nonempty():
    """Every derived (qualified_name -> ExperimentConfig field) pair maps a real
    registry parameter to a real ExperimentConfig field (drift guard)."""
    from legoesm.driver.config import ExperimentConfig
    from legoesm.training.param_collector import build_registry
    reg = {m.qualified_name for m in build_registry()}
    ec = set(ExperimentConfig._fields)
    amap = build_atm_scalar_param_map()
    assert amap, "atm scalar-param map should not be empty"
    for qname, ec_field in amap.items():
        assert qname in reg, f"{qname} not a registry parameter"
        assert ec_field in ec, f"{ec_field} not an ExperimentConfig field"
    # spot-check the documented cloud mapping
    assert amap.get("atm.clouds.CloudConfig.q_c_diagnostic") == "cloud_q_c_diagnostic"
    # case-variant scalar (ExperimentConfig capitalises the symbol) is covered
    # by the unambiguous case-insensitive fallback (codex #691 atm round).
    assert amap.get("atm.conv.SBMConfig.rh_ref") == "sbm_RH_ref"


def test_scalar_map_applies_atm_param_to_flat_experimentconfig():
    """An atm param routes to the FLAT ExperimentConfig scalar (not a nested
    *Config, which ExperimentConfig doesn't have) via scalar_param_map."""
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
    )
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=5),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"),
        cloud_scheme="sundqvist",
    )
    out = apply_params_to_config(
        cfg, {"atm.clouds.CloudConfig.q_c_diagnostic": 3.0e-4},
        driver="test", scalar_param_map=build_atm_scalar_param_map())
    assert out.cloud_q_c_diagnostic == 3.0e-4
