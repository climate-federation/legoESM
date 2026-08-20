"""#1509: the run manifest must record the --params values a run applied.

Class-routed `--params` (`run_config_yaml._route_overrides_by_class`) land on
NESTED scheme configs, which `_serialize_config` does not reach — so
`resolved_config` showed `turbulence_override: None` and none of the applied
values, and the run's provenance depended on the referenced params FILE still
existing unmodified. These tests pin that the manifest is self-contained.
"""

from __future__ import annotations

import pytest

from legoesm.driver.restart import build_run_manifest
from legoesm.driver.run_config_yaml import apply_params_to_config
from legoesm.driver.config import ExperimentConfig


def _routable_cfg_and_params():
    """A config that actually CARRIES the nested scheme config the router
    targets. A bare ExperimentConfig has no scheme override, so every registry
    key raises SystemExit there and a test built on one is vacuous — which is
    how the first version of this file skipped its own decisive assertion."""
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    cfg = ExperimentConfig(turbulence="clubb_lite",
                           turbulence_override=CLUBBLiteConfig())
    return cfg, {"atm.turb.CLUBBLiteConfig.C_K": 0.2169,
                 "atm.turb.CLUBBLiteConfig.Pr_t": 0.9}


def test_router_records_what_it_applied():
    cfg, params = _routable_cfg_and_params()
    record: dict = {}
    out = apply_params_to_config(cfg, params, driver="test", record=record)
    # The values really were applied to the NESTED config...
    assert float(out.turbulence_override.C_K) == 0.2169
    assert float(cfg.turbulence_override.C_K) != 0.2169, "vacuous: no change"
    # ...and every one of them is recorded, as scalars keyed by qualified name.
    assert record == params, (
        f"router applied {params} but recorded {record} (#1509)")


def test_manifest_carries_params_applied():
    cfg = ExperimentConfig()
    m = build_run_manifest(cfg, params_applied={"clubb.c_K": 0.2169})
    assert m["config"]["params_applied"] == {"clubb.c_K": 0.2169}, (
        "the manifest dropped the applied --params; a reader is back to "
        "trusting the params FILE (#1509)")


def test_manifest_params_field_is_present_and_empty_without_params():
    """Always present, so a reader can distinguish 'no params' from 'old
    manifest that could not record them'."""
    m = build_run_manifest(ExperimentConfig())
    assert m["config"]["params_applied"] == {}


def test_recording_is_opt_in_and_does_not_change_the_config():
    """record=None (every existing caller) must behave exactly as before."""
    cfg, params = _routable_cfg_and_params()
    a = apply_params_to_config(cfg, params, driver="test")
    b = apply_params_to_config(cfg, params, driver="test", record={})
    assert a == b


def test_manifest_of_a_run_with_params_names_the_values_not_just_the_file():
    """The end-to-end property the issue asks for: a reader of the manifest
    ALONE can see which values produced the run."""
    cfg, params = _routable_cfg_and_params()
    record: dict = {}
    cfg2 = apply_params_to_config(cfg, params, driver="test", record=record)
    m = build_run_manifest(cfg2, params_applied=record)
    import json
    blob = json.dumps(m)
    for v in params.values():
        assert str(v) in blob, (
            f"{v} was applied but appears nowhere in the manifest JSON — the "
            f"run's provenance still depends on the --params file (#1509)")
