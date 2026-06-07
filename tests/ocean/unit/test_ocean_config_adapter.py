"""Unit tests for the ocean YAML config adapter (``OceanExperimentConfig``).

Mirrors the atmosphere ``legoesm.config.Config`` boundary tests: YAML load +
deep-merge, dot-notation get/set, runtime-config mapping, typo detection, and
strict validation that delegates to the runtime model validator.
"""
from __future__ import annotations

import textwrap

import pytest

from legoesm.ocean.config import OceanExperimentConfig, DEFAULT_OCEAN_CONFIG
from legoesm.ocean.state import (
    LatLonCGridOceanConfig,
    OceanConfig,
    SpectralOceanConfig,
)


def _write(tmp_path, text: str) -> str:
    p = tmp_path / "ocean.yaml"
    p.write_text(textwrap.dedent(text))
    return str(p)


def test_default_config_resolves_to_latlon_namedtuple():
    cfg = OceanExperimentConfig()
    rt = cfg.to_ocean_config()
    assert isinstance(rt, LatLonCGridOceanConfig)
    # Empty ocean: {} means every runtime default is preserved.
    assert rt == LatLonCGridOceanConfig()


def test_from_yaml_deep_merges_onto_defaults(tmp_path):
    path = _write(
        tmp_path,
        """
        grid:
          type: latlon_cgrid
          n_lat: 180
        ocean:
          A_h: 30000.0
          eos: wright
        time:
          dt_seconds: 1800
        """,
    )
    cfg = OceanExperimentConfig.from_yaml(path)
    # Overridden values present...
    assert cfg.get("grid.n_lat") == 180
    assert cfg.get("ocean.A_h") == 30000.0
    assert cfg.get("time.dt_seconds") == 1800
    # ...and defaults that were not overridden are retained.
    assert cfg.get("grid.n_lon") == DEFAULT_OCEAN_CONFIG["grid"]["n_lon"]
    rt = cfg.to_ocean_config()
    assert rt.A_h == 30000.0
    assert rt.eos == "wright"


def test_dot_get_set_roundtrip():
    cfg = OceanExperimentConfig()
    cfg.set("ocean.bottom_drag_r", 2.5e-3)
    assert cfg.get("ocean.bottom_drag_r") == 2.5e-3
    assert cfg.to_ocean_config().bottom_drag_r == 2.5e-3
    assert cfg.get("nonexistent.key", "fallback") == "fallback"


def test_unknown_ocean_field_raises():
    cfg = OceanExperimentConfig.from_dict({"ocean": {"A_h_typo": 1.0}})
    with pytest.raises(ValueError, match="unknown ocean config field"):
        cfg.to_ocean_config()


def test_unknown_grid_type_raises():
    cfg = OceanExperimentConfig.from_dict({"grid": {"type": "octahedral"}})
    with pytest.raises(ValueError, match="grid.type must be one of"):
        cfg.to_ocean_config()


def test_cubed_sphere_maps_to_oceanconfig():
    cfg = OceanExperimentConfig.from_dict(
        {"grid": {"type": "cubed_sphere"}, "ocean": {"A_h": 1.2e4}}
    )
    rt = cfg.to_ocean_config()
    assert isinstance(rt, OceanConfig)
    assert rt.A_h == 1.2e4


def test_spectral_maps_to_spectraloceanconfig():
    cfg = OceanExperimentConfig.from_dict(
        {"grid": {"type": "spectral"}, "ocean": {"hyperdiff_coeff": 1.0e15}}
    )
    rt = cfg.to_ocean_config()
    assert isinstance(rt, SpectralOceanConfig)


def test_eos_linear_subdict_builds_namedtuple():
    from legoesm.ocean.eos import LinearEOSConfig

    cfg = OceanExperimentConfig.from_dict(
        {"ocean": {"eos": "linear", "eos_linear": {"alpha_T": 2.0e-4}}}
    )
    rt = cfg.to_ocean_config()
    assert rt.eos == "linear"
    assert isinstance(rt.eos_linear, LinearEOSConfig)
    assert rt.eos_linear.alpha_T == 2.0e-4


def test_eos_linear_unknown_field_raises():
    cfg = OceanExperimentConfig.from_dict(
        {"ocean": {"eos": "linear", "eos_linear": {"alpha_typo": 1.0}}}
    )
    with pytest.raises(ValueError, match="unknown ocean.eos_linear field"):
        cfg.to_ocean_config()


def test_nested_physics_block_rejected():
    cfg = OceanExperimentConfig.from_dict(
        {"ocean": {"physics": {"vertical_mixing": {}}}}
    )
    with pytest.raises(ValueError, match="not yet configurable via YAML"):
        cfg.to_ocean_config()


def test_validate_strict_passes_for_default():
    OceanExperimentConfig().validate_strict()  # must not raise


def test_validate_strict_rejects_bad_eos():
    cfg = OceanExperimentConfig.from_dict({"ocean": {"eos": "nonsense"}})
    with pytest.raises(ValueError, match="eos must be one of"):
        cfg.validate_strict()


def test_validate_strict_rejects_bad_barotropic_solver():
    cfg = OceanExperimentConfig.from_dict(
        {"ocean": {"barotropic_solver": "typo_solver"}}
    )
    with pytest.raises(ValueError, match="barotropic_solver must be one of"):
        cfg.validate_strict()


def test_validate_strict_rejects_nonpositive_dt():
    cfg = OceanExperimentConfig.from_dict({"time": {"dt_seconds": 0}})
    with pytest.raises(ValueError, match="time.dt_seconds must be > 0"):
        cfg.validate_strict()


def test_signature_changes_with_runtime_override_and_detects_noop():
    cfg = OceanExperimentConfig()
    sig0 = cfg.signature()
    cfg.set("ocean.A_h", 9.9e4)
    assert cfg.signature() != sig0
    # A non-runtime dot-path is a no-op on the resolved config signature.
    cfg2 = OceanExperimentConfig()
    sig_a = cfg2.signature()
    cfg2.set("ocean.totally_unused_metadata_path", 1)
    # Unknown field makes to_ocean_config raise -> unresolvable signature differs
    # from the resolvable one; the no-op detector keys off byte-identity, so this
    # is correctly NOT byte-identical (it is flagged as a change/typo upstream).
    assert cfg2.signature() != sig_a


def test_time_override_not_a_noop():
    cfg = OceanExperimentConfig()
    sig0 = cfg.signature()
    cfg.set("time.dt_seconds", 900)
    assert cfg.signature() != sig0


def test_run_control_overrides_not_noops():
    # codex P2: overrides the runner consumes outside the runtime config
    # (output.path, grid.nlev, grid.n_lat/n_lon, time.duration_days) must change
    # the signature so init_experiment does not reject them as no-ops.
    for key, value in (
        ("output.path", "output/custom/"),
        ("grid.nlev", 40),
        ("grid.n_lat", 200),
        ("grid.n_lon", 400),
        ("time.duration_days", 999),
    ):
        cfg = OceanExperimentConfig()
        sig0 = cfg.signature()
        cfg.set(key, value)
        assert cfg.signature() != sig0, f"{key} override wrongly a no-op"


def test_forcing_override_is_a_noop():
    # forcing.dataset is advisory (the omip runner always loads CORE-II NYF), so
    # overriding it genuinely does not change the run -> stays a no-op.
    cfg = OceanExperimentConfig()
    sig0 = cfg.signature()
    cfg.set("forcing.dataset", "something_else")
    assert cfg.signature() == sig0


def test_resolve_run_controls_drives_run_from_yaml():
    import argparse

    from legoesm.ocean.config import resolve_ocean_run_controls

    cfg = OceanExperimentConfig.from_dict({
        "grid": {"type": "latlon_cgrid", "n_lat": 180, "n_lon": 360, "nlev": 30},
        "time": {"dt_seconds": 1800, "duration_days": 730},
        "output": {"path": "output/omip_latlon/"},
    })
    args = argparse.Namespace(
        dt=3600.0, years=5.0, output="results/default", nlev=20,
        latlon_res="180x360",
    )
    applied = resolve_ocean_run_controls(cfg, args, cli_given=set())
    assert args.dt == 1800.0
    assert args.years == 2.0          # 730 days / 365
    assert args.output == "output/omip_latlon/"
    assert args.nlev == 30
    assert args.latlon_res == "180x360"
    assert set(applied) == {"dt", "years", "output", "nlev", "latlon_res"}


def test_resolve_run_controls_cli_wins_over_yaml():
    import argparse

    from legoesm.ocean.config import resolve_ocean_run_controls

    cfg = OceanExperimentConfig.from_dict({"time": {"dt_seconds": 1800}})
    args = argparse.Namespace(
        dt=7200.0, years=5.0, output="x", nlev=20, latlon_res="1x1",
    )
    # User passed --dt explicitly -> YAML must NOT override it.
    applied = resolve_ocean_run_controls(cfg, args, cli_given={"dt"})
    assert args.dt == 7200.0
    assert "dt" not in applied


def test_run_command_quotes_paths_with_spaces():
    # codex P2: an absolute config path with spaces/metachars must be shell-safe
    # in the generated run.sh.
    cfg = OceanExperimentConfig()
    cmd = cfg.run_command("/tmp/my runs/exp 1/config.yaml")
    assert "'/tmp/my runs/exp 1/config.yaml'" in cmd
    # And a plain path is unquoted (shlex.quote leaves safe strings bare).
    assert cfg.run_command("config.yaml").endswith("--config config.yaml")


def test_run_command_emits_grid_backend():
    # codex P2: the template's grid backend must be passed to the runner
    # (otherwise it defaults to tripole and ignores n_lat/n_lon).
    assert "--grid latlon_bathy" in OceanExperimentConfig().run_command()
    cube = OceanExperimentConfig.from_dict({"grid": {"type": "cubed_sphere"}})
    assert "--grid cubed_sphere" in cube.run_command()


def test_run_command_unsupported_grid_raises():
    spec = OceanExperimentConfig.from_dict({"grid": {"type": "spectral"}})
    with pytest.raises(ValueError, match="no run_omip_core2.py --grid backend"):
        spec.run_command()


def test_yaml_roundtrip(tmp_path):
    cfg = OceanExperimentConfig.from_dict({"ocean": {"A_h": 2.0e4}})
    out = tmp_path / "out.yaml"
    cfg.to_yaml(str(out))
    reloaded = OceanExperimentConfig.from_yaml(str(out))
    assert reloaded.get("ocean.A_h") == 2.0e4
    assert reloaded.to_ocean_config() == cfg.to_ocean_config()
