"""Ocean run-manifest provenance round-trip (#376 Phase 4).

The run-manifest spine is config-kind-aware: an ocean runtime config
(``LatLonCGridOceanConfig`` and friends) must serialize into the manifest,
validate, and reconstruct exactly — including nested NamedTuples — with the
``config_hash`` round-tripping. Atmosphere manifests must keep working with no
``config_kind`` field (back-compat).
"""
from __future__ import annotations

import json

import pytest

from legoesm.ocean.config import (
    ocean_config_to_dict,
    ocean_config_from_dict,
    OceanRunRecord,
)
from legoesm.ocean.state import (
    LatLonCGridOceanConfig,
    OceanConfig,
    SpectralOceanConfig,
)
from legoesm.ocean.eos import LinearEOSConfig
from legoesm.driver.restart import (
    build_run_manifest,
    validate_run_manifest,
    write_run_manifest,
    read_run_manifest,
    record_state_digest,
    recorded_state_digest,
    compute_config_hash,
    detect_config_kind,
)


def _latlon_cfg() -> LatLonCGridOceanConfig:
    # Exercise a nested NamedTuple field (eos_linear) too.
    return LatLonCGridOceanConfig(
        A_h=3.0e4,
        bottom_drag_r=2.5e-3,
        eos="linear",
        eos_linear=LinearEOSConfig(alpha_T=2.0e-4, beta_S=7.6e-4),
        barotropic_solver="implicit_cn",
    )


def test_codec_roundtrip_latlon():
    cfg = _latlon_cfg()
    d = ocean_config_to_dict(cfg)
    # JSON-serializable (no tuples / objects left).
    json.dumps(d)
    assert d["__type__"].endswith(":LatLonCGridOceanConfig")
    assert d["eos_linear"]["__type__"].endswith(":LinearEOSConfig")
    rebuilt = ocean_config_from_dict(d)
    assert rebuilt == cfg
    assert isinstance(rebuilt.eos_linear, LinearEOSConfig)


def test_codec_roundtrip_default_configs():
    for cfg in (LatLonCGridOceanConfig(), OceanConfig(), SpectralOceanConfig()):
        rebuilt = ocean_config_from_dict(ocean_config_to_dict(cfg))
        assert rebuilt == cfg


def test_codec_drops_unknown_field_forward_compat():
    cfg = LatLonCGridOceanConfig(A_h=1.234e4)
    d = ocean_config_to_dict(cfg)
    d["a_since_removed_field"] = 99  # simulate an older manifest
    rebuilt = ocean_config_from_dict(d)
    assert rebuilt.A_h == 1.234e4


def test_codec_rejects_non_legoesm_type():
    bad = {"__type__": "os:system", "x": 1}
    with pytest.raises(ValueError, match="non-legoesm"):
        ocean_config_from_dict(bad)


def test_detect_config_kind():
    assert detect_config_kind(LatLonCGridOceanConfig()) == "ocean"
    assert detect_config_kind(OceanConfig()) == "ocean"
    assert detect_config_kind(SpectralOceanConfig()) == "ocean"


def test_manifest_build_validate_ocean():
    cfg = _latlon_cfg()
    m = build_run_manifest(cfg, command_line="run_omip_core2.py --config x.yaml")
    assert m["config"]["config_kind"] == "ocean"
    # Must validate (config_hash matches resolved_config under the ocean codec).
    validate_run_manifest(m)
    rebuilt = ocean_config_from_dict(m["config"]["resolved_config"])
    assert rebuilt == cfg
    assert compute_config_hash(rebuilt, "ocean") == m["config"]["config_hash"]


def _run_record(**controls) -> OceanRunRecord:
    base = dict(
        runtime_config=_latlon_cfg(), grid="tripole", mesh="mesh.nc",
        nlev=30, dt_seconds=1800.0, total_days=365.0,
        output_path="output/omip", forcing="core2_nyf",
    )
    base.update(controls)
    return OceanRunRecord(**base)


def test_run_record_roundtrips_and_carries_controls():
    rec = _run_record()
    d = ocean_config_to_dict(rec)
    json.dumps(d)
    rebuilt = ocean_config_from_dict(d)
    assert rebuilt == rec
    assert isinstance(rebuilt.runtime_config, LatLonCGridOceanConfig)


def test_run_record_hash_distinguishes_run_controls():
    # The codex HIGH finding: model.config alone cannot identify the experiment.
    # Two records with the SAME runtime_config but different dt / grid / output
    # MUST hash differently now that controls are in the hashed payload.
    base = _run_record()
    assert compute_config_hash(base, "ocean") == compute_config_hash(
        _run_record(), "ocean"
    )
    for changed in (
        _run_record(dt_seconds=3600.0),
        _run_record(total_days=730.0),
        _run_record(grid="latlon_bathy"),
        _run_record(output_path="output/other"),
        _run_record(nlev=20),
    ):
        assert compute_config_hash(changed, "ocean") != compute_config_hash(
            base, "ocean"
        )


def test_manifest_from_run_record_validates():
    rec = _run_record()
    m = build_run_manifest(rec, command_line="run_omip_core2.py --config x.yaml")
    assert m["config"]["config_kind"] == "ocean"
    validate_run_manifest(m)
    rebuilt = ocean_config_from_dict(m["config"]["resolved_config"])
    assert rebuilt == rec


def test_manifest_tamper_detected_ocean():
    cfg = _latlon_cfg()
    m = build_run_manifest(cfg)
    m["config"]["resolved_config"]["A_h"] = 1.0  # tamper without fixing the hash
    with pytest.raises(ValueError, match="config_hash does not match"):
        validate_run_manifest(m)


def test_manifest_write_read_digest_roundtrip_ocean(tmp_path):
    cfg = _latlon_cfg()
    path = write_run_manifest(tmp_path, cfg)
    m = read_run_manifest(path)
    validate_run_manifest(m)
    assert m["config"]["config_kind"] == "ocean"
    # Post-run digest recording (the reproduce reference).
    record_state_digest(path, "deadbeefcafe")
    assert recorded_state_digest(read_run_manifest(path)) == "deadbeefcafe"
