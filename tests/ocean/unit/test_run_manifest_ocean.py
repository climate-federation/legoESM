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
    return LatLonCGridOceanConfig.from_flat(
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
    for cfg in (LatLonCGridOceanConfig.from_flat(), OceanConfig(), SpectralOceanConfig()):
        rebuilt = ocean_config_from_dict(ocean_config_to_dict(cfg))
        assert rebuilt == cfg


def test_codec_drops_unknown_field_forward_compat():
    cfg = LatLonCGridOceanConfig.from_flat(A_h=1.234e4)
    d = ocean_config_to_dict(cfg)
    d["a_since_removed_field"] = 99  # simulate an older manifest
    rebuilt = ocean_config_from_dict(d)
    assert rebuilt.lateral_viscosity.A_h == 1.234e4


def test_codec_rejects_non_legoesm_type():
    bad = {"__type__": "os:system", "x": 1}
    with pytest.raises(ValueError, match="non-legoesm"):
        ocean_config_from_dict(bad)


def test_detect_config_kind():
    assert detect_config_kind(LatLonCGridOceanConfig.from_flat()) == "ocean"
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
        _run_record(forcing_path="/data/other/core2"),
        _run_record(woa_init=True),
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
    m["config"]["resolved_config"]["lateral_viscosity"]["A_h"] = 1.0  # tamper (nested #501)
    with pytest.raises(ValueError, match="config_hash does not match"):
        validate_run_manifest(m)


def test_reproduce_check_on_ocean_manifest_exits_nonzero(tmp_path):
    # codex P2: `reproduce --check` on an ocean manifest must NOT return 0 (the
    # digest comparison cannot be performed via the atmosphere driver) — a green
    # would let CI treat an unperformed check as passed.
    import argparse

    from legoesm.cli import cmd_reproduce

    write_run_manifest(tmp_path, _latlon_cfg(), config_kind="ocean")
    args = argparse.Namespace(
        manifest=str(tmp_path / "run_manifest.json"),
        check=True,
        output=str(tmp_path / "rerun"),
    )
    with pytest.raises(SystemExit) as exc:
        cmd_reproduce(args)
    assert exc.value.code == 2


def test_reproduce_without_check_on_ocean_manifest_validates(tmp_path):
    # Plain `reproduce` (no --check) just validates the ocean manifest and
    # returns (no rerun); must not raise.
    import argparse

    from legoesm.cli import cmd_reproduce

    write_run_manifest(tmp_path, _latlon_cfg(), config_kind="ocean")
    args = argparse.Namespace(
        manifest=str(tmp_path / "run_manifest.json"),
        check=False,
        output=str(tmp_path / "rerun"),
    )
    assert cmd_reproduce(args) is None  # validate-only, clean return


def test_manifest_write_read_digest_roundtrip_ocean(tmp_path):
    cfg = _latlon_cfg()
    path = write_run_manifest(tmp_path, cfg)
    m = read_run_manifest(path)
    validate_run_manifest(m)
    assert m["config"]["config_kind"] == "ocean"
    # Post-run digest recording (the reproduce reference).
    record_state_digest(path, "deadbeefcafe")
    assert recorded_state_digest(read_run_manifest(path)) == "deadbeefcafe"


def test_manifest_write_survives_array_config_leaf(tmp_path):
    """A config carrying an ARRAY leaf (the per-cell NEMO ln_rnf_depth_ini
    runoff_depth_spread_map) must not crash write_run_manifest with
    'Object of type ArrayImpl is not JSON serializable' -- the dump summarises
    array-likes as shape/dtype/min/max (best-effort provenance; the ico7_dm30
    MPAS run lost its whole manifest to this)."""
    import json as _json
    import numpy as np

    from legoesm.driver.restart import write_run_manifest

    cfg = _latlon_cfg()._replace(
        runoff_depth_spread_map=np.linspace(1.0, 150.0, 7))
    rec = _run_record(runtime_config=cfg)
    path = write_run_manifest(tmp_path, rec, config_kind="ocean",
                              runner_tag="test")
    data = _json.loads(path.read_text())
    blob = _json.dumps(data)
    assert "ArrayImpl" not in blob
    assert "__array_summary__" in blob
    # locate the summary and check the numbers survived
    def _find(d):
        if isinstance(d, dict):
            if "__array_summary__" in d:
                return d["__array_summary__"]
            for v in d.values():
                r = _find(v)
                if r is not None:
                    return r
        elif isinstance(d, list):
            for v in d:
                r = _find(v)
                if r is not None:
                    return r
        return None
    s = _find(data)
    assert s is not None and s["shape"] == [7]
    assert s["min"] == 1.0 and s["max"] == 150.0
    # the manifest must validate on its own output (hash consistency under the
    # summarised encoding), not just be JSON-writable (codex)
    from legoesm.driver.restart import validate_run_manifest
    validate_run_manifest(data)


def test_numpy_scalar_config_leaf_encoding_unchanged():
    """A NumPy SCALAR leaf (np.float64 A_h) must keep the plain scalar
    encoding -- not become an __array_summary__ dict -- so configs without
    arrays stay byte-identical and rebuild exactly (codex)."""
    import numpy as np

    cfg = _latlon_cfg()._replace(rho_0=np.float64(1026.5))
    d = ocean_config_to_dict(cfg)
    assert d["rho_0"] == 1026.5 and not isinstance(d["rho_0"], dict)
    rebuilt = ocean_config_from_dict(d)
    assert float(rebuilt.rho_0) == 1026.5
    assert (compute_config_hash(cfg, "ocean")
            == compute_config_hash(cfg._replace(rho_0=1026.5), "ocean"))


def test_array_summary_hash_is_content_sensitive():
    """Two maps with the SAME shape/dtype/min/max but different interior
    values must hash differently (the map affects physics; shape+range alone
    collided -- codex)."""
    import numpy as np

    a = np.array([1.0, 2.0, 150.0])
    b = np.array([1.0, 3.0, 150.0])          # same shape/min/max, different content
    ca = _latlon_cfg()._replace(runoff_depth_spread_map=a)
    cb = _latlon_cfg()._replace(runoff_depth_spread_map=b)
    assert compute_config_hash(ca, "ocean") != compute_config_hash(cb, "ocean")
    # and identical content hashes identically
    assert (compute_config_hash(ca, "ocean")
            == compute_config_hash(_latlon_cfg()._replace(
                runoff_depth_spread_map=a.copy()), "ocean"))
