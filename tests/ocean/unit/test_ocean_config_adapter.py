"""Unit tests for the ocean YAML config adapter (``OceanExperimentConfig``).

Mirrors the atmosphere ``legoesm.config.Config`` boundary tests: YAML load +
deep-merge, dot-notation get/set, runtime-config mapping, typo detection, and
strict validation that delegates to the runtime model validator.
"""
from __future__ import annotations

import textwrap

import pytest
from legoesm.ocean.config import DEFAULT_OCEAN_CONFIG, OceanExperimentConfig
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
    assert rt == LatLonCGridOceanConfig.from_flat()


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
    # #501: A_h moved into the nested LateralViscosityConfig; the flat YAML key
    # ocean.A_h still routes here via flat_fields + from_flat.
    assert rt.lateral_viscosity.A_h == 30000.0
    assert rt.eos == "wright"


def test_dot_get_set_roundtrip():
    cfg = OceanExperimentConfig()
    cfg.set("ocean.bottom_drag_r", 2.5e-3)
    assert cfg.get("ocean.bottom_drag_r") == 2.5e-3
    # #501: stored nested (config.bottom_drag.bottom_drag_r); the flat YAML key
    # ocean.bottom_drag_r still routes here via flat_fields + from_flat.
    assert cfg.to_ocean_config().bottom_drag.bottom_drag_r == 2.5e-3
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
    with pytest.raises(ValueError,
                       match=r"unknown LinearEOSConfig field.*ocean\.eos_linear"):
        cfg.to_ocean_config()


# ----------------------------------------------------- nested physics YAML (#382)

def test_nested_physics_vertical_mixing_kpp_builds():
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.vertical_mixing.config import (
        KPPConfig,
        VerticalMixingConfig,
    )

    cfg = OceanExperimentConfig.from_dict({
        "ocean": {"physics": {
            "vertical_mixing": {
                "scheme": "kpp",
                "kpp": {"Ri_crit": 0.3},
            },
        }},
    })
    rt = cfg.to_ocean_config()
    assert isinstance(rt.physics, OceanPhysicsConfig)
    assert isinstance(rt.physics.vertical_mixing, VerticalMixingConfig)
    assert rt.physics.vertical_mixing.scheme == "kpp"
    assert isinstance(rt.physics.vertical_mixing.kpp, KPPConfig)
    assert rt.physics.vertical_mixing.kpp.Ri_crit == 0.3
    # untouched sub-configs keep their defaults
    assert rt.physics.vertical_mixing.constant == \
        VerticalMixingConfig().constant


def test_nested_physics_multiple_sections_build():
    cfg = OceanExperimentConfig.from_dict({
        "ocean": {"physics": {
            "convection": {"scheme": "enhanced_diffusion"},
            "shortwave_penetration": {"water_type": "II"},
        }},
    })
    rt = cfg.to_ocean_config()
    assert rt.physics.convection.scheme == "enhanced_diffusion"
    assert rt.physics.shortwave_penetration.water_type == "II"


def test_nested_gm_redi_with_visbeck_and_eke_build():
    from legoesm.ocean.physics.lateral_mixing.config import (
        GMRediConfig,
        VisbeckConfig,
    )
    from legoesm.ocean.physics.lateral_mixing.eke import EKEConfig

    cfg = OceanExperimentConfig.from_dict({
        "ocean": {"gm_redi": {
            "kappa_GM": 1500.0,
            "S_max": 0.02,
            "visbeck": {"enabled": True},   # concrete-default sub-config
            "eke": {"e_min": 1.0e-5},       # Optional (default None) sub-config
        }},
    })
    rt = cfg.to_ocean_config()
    assert isinstance(rt.gm_redi, GMRediConfig)
    assert rt.gm_redi.kappa_GM == 1500.0
    assert rt.gm_redi.S_max == 0.02
    assert isinstance(rt.gm_redi.visbeck, VisbeckConfig)
    assert rt.gm_redi.visbeck.enabled is True
    assert isinstance(rt.gm_redi.eke, EKEConfig)   # resolved from `| None` annotation
    assert rt.gm_redi.eke.e_min == 1.0e-5


def test_nested_typo_detected_at_every_level():
    # top-level physics sub-section field
    with pytest.raises(ValueError, match="unknown OceanPhysicsConfig field"):
        OceanExperimentConfig.from_dict(
            {"ocean": {"physics": {"vertical_mixingg": {}}}}).to_ocean_config()
    # scheme sub-config field (deepest level)
    with pytest.raises(ValueError,
                       match=r"unknown KPPConfig field.*Ri_crite"):
        OceanExperimentConfig.from_dict({"ocean": {"physics": {
            "vertical_mixing": {"scheme": "kpp", "kpp": {"Ri_crite": 0.3}}}}}
        ).to_ocean_config()
    # gm_redi nested sub-config field
    with pytest.raises(ValueError,
                       match=r"unknown VisbeckConfig field.*gm_redi\.visbeck"):
        OceanExperimentConfig.from_dict({"ocean": {"gm_redi": {
            "visbeck": {"enabledd": True}}}}).to_ocean_config()


def test_minimal_latlon_physics_block_validates():
    """A MINIMAL lat-lon physics block (only vertical_mixing) must pass
    validate_strict: the adapter defaults physics.lateral_mixing.scheme to "none"
    (the cube-only "harmonic" runtime default would otherwise be rejected on the
    lat-lon C-grid).  #382 codex P2."""
    cfg = OceanExperimentConfig.from_dict({
        "grid": {"type": "latlon_cgrid"},
        "ocean": {"physics": {
            "vertical_mixing": {"scheme": "kpp", "kpp": {"Ri_crit": 0.3}}}},
    })
    cfg.validate_strict()  # must not raise
    assert cfg.to_ocean_config().physics.lateral_mixing.scheme == "none"


def test_latlon_explicit_harmonic_lateral_mixing_still_rejected():
    """An EXPLICIT physics.lateral_mixing.scheme='harmonic' on the lat-lon grid
    is not silently overridden — it reaches validate_strict and gets the clear
    'cubed-sphere-only' error (respecting the user's explicit choice)."""
    cfg = OceanExperimentConfig.from_dict({
        "grid": {"type": "latlon_cgrid"},
        "ocean": {"physics": {"lateral_mixing": {"scheme": "harmonic"}}},
    })
    assert cfg.to_ocean_config().physics.lateral_mixing.scheme == "harmonic"
    with pytest.raises(ValueError, match="cubed-sphere-only"):
        cfg.validate_strict()


def test_latlon_nonmapping_lateral_mixing_rejected():
    """A non-mapping ocean.physics.lateral_mixing on the lat-lon grid must raise
    the generic 'must be a mapping' error (not a TypeError from the scheme-default
    splat, nor a silent rewrite of a falsey value). #382 codex P2."""
    for bad in ("harmonic", [1], 0):
        with pytest.raises(
                ValueError,
                match=r"ocean\.physics\.lateral_mixing must be a mapping"):
            OceanExperimentConfig.from_dict(
                {"grid": {"type": "latlon_cgrid"},
                 "ocean": {"physics": {"lateral_mixing": bad}}}).to_ocean_config()


def test_nested_section_must_be_mapping():
    with pytest.raises(ValueError, match=r"ocean\.physics must be a mapping"):
        OceanExperimentConfig.from_dict(
            {"ocean": {"physics": 5}}).to_ocean_config()


def test_scalar_in_subconfig_position_rejected():
    """A scalar where a sub-config NamedTuple is expected must fail fast at the
    boundary (else it builds e.g. OceanPhysicsConfig(vertical_mixing=5) that
    crashes later in the factory). #382 codex P2."""
    with pytest.raises(ValueError,
                       match=r"ocean\.physics\.vertical_mixing must be a mapping"):
        OceanExperimentConfig.from_dict(
            {"ocean": {"physics": {"vertical_mixing": 5}}}).to_ocean_config()


def test_null_disables_optional_subconfig():
    """An explicit null on an Optional sub-config (gm_redi.eke) disables it."""
    rt = OceanExperimentConfig.from_dict(
        {"ocean": {"gm_redi": {"eke": None}}}).to_ocean_config()
    assert rt.gm_redi.eke is None


def test_null_on_required_subconfig_rejected():
    """null on a REQUIRED sub-config (physics.vertical_mixing) must fail fast —
    not smuggle a None the factory later dereferences. #382 codex P2."""
    with pytest.raises(ValueError,
                       match=r"ocean\.physics\.vertical_mixing must be a mapping"):
        OceanExperimentConfig.from_dict(
            {"ocean": {"physics": {"vertical_mixing": None}}}).to_ocean_config()


def test_nested_physics_none_is_legacy_mode():
    # ocean.physics: null -> legacy (physics=None) path, not an error.
    rt = OceanExperimentConfig.from_dict(
        {"ocean": {"physics": None}}).to_ocean_config()
    assert rt.physics is None


def test_nested_physics_yaml_roundtrips_through_manifest_codec():
    # The built nested config must survive the run-manifest codec (the #376
    # serialize+reconstruct path) byte-for-byte.
    from legoesm.ocean.config import (
        ocean_config_from_dict,
        ocean_config_to_dict,
    )
    rt = OceanExperimentConfig.from_dict({"ocean": {"physics": {
        "vertical_mixing": {"scheme": "kpp", "kpp": {"Ri_crit": 0.3}}}}}
    ).to_ocean_config()
    rebuilt = ocean_config_from_dict(ocean_config_to_dict(rt))
    assert rebuilt == rt


def test_validate_strict_passes_for_default():
    OceanExperimentConfig().validate_strict()  # must not raise


def test_validate_strict_rejects_bad_eos():
    cfg = OceanExperimentConfig.from_dict({"ocean": {"eos": "nonsense"}})
    with pytest.raises(ValueError, match="eos must be one of"):
        cfg.validate_strict()


def test_validate_strict_rejects_bad_shortwave_penetration_scheme():
    cfg = OceanExperimentConfig.from_dict({"ocean": {"physics": {
        "shortwave_penetration": {"scheme": "ln_qsr_typo"},
    }}})
    with pytest.raises(
        ValueError, match="physics.shortwave_penetration.scheme must be one of"
    ):
        cfg.validate_strict()


def test_validate_strict_rejects_bad_barotropic_solver():
    cfg = OceanExperimentConfig.from_dict(
        {"ocean": {"barotropic_solver": "typo_solver"}}
    )
    with pytest.raises(ValueError, match="barotropic_solver must be one of"):
        cfg.validate_strict()


def test_validate_strict_rejects_nonpositive_dt():
    cfg = OceanExperimentConfig.from_dict({"time": {"dt_seconds": 0}})
    # Routed through the shared setup_selector.require_positive_finite helper
    # (#388), whose message is "<name> must be a finite number > 0, got <v>".
    with pytest.raises(ValueError, match=r"time\.dt_seconds must be a finite number > 0"):
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
        ("forcing.path", "/data/ocean/core2"),
        ("init.woa_init", True),
        ("init.woa_t", "/data/woa/t.nc"),
        ("init.woa_s", "/data/woa/s.nc"),
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
        "init": {"woa_init": True, "woa_t": "d/t.nc", "woa_s": "d/s.nc"},
        "forcing": {"dataset": "core2_nyf", "path": "/data/ocean/core2"},
    })
    args = argparse.Namespace(
        dt=3600.0, years=5.0, output="results/default", nlev=20,
        latlon_res="180x360", woa_init=False, woa_t="x", woa_s="y",
        forcing_path=None,
    )
    applied = resolve_ocean_run_controls(cfg, args, cli_given=set())
    assert args.dt == 1800.0
    assert args.years == 2.0          # 730 days / 365
    assert args.output == "output/omip_latlon/"
    assert args.nlev == 30
    assert args.latlon_res == "180x360"
    assert args.woa_init is True
    assert args.woa_t == "d/t.nc" and args.woa_s == "d/s.nc"
    assert args.forcing_path == "/data/ocean/core2"
    assert set(applied) == {
        "dt", "years", "output", "nlev", "latlon_res",
        "woa_init", "woa_t", "woa_s", "forcing_path",
    }


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


def test_run_command_unsupported_grid_raises():
    # Only latlon_cgrid is wired through the omip --config runner; cube/spectral
    # have no runnable bundle, so run_command raises (init_experiment fails fast).
    for gt in ("cubed_sphere", "spectral"):
        cfg = OceanExperimentConfig.from_dict({"grid": {"type": gt}})
        with pytest.raises(ValueError, match="no runnable run_omip_core2.py"):
            cfg.run_command()


def test_yaml_roundtrip(tmp_path):
    cfg = OceanExperimentConfig.from_dict({"ocean": {"A_h": 2.0e4}})
    out = tmp_path / "out.yaml"
    cfg.to_yaml(str(out))
    reloaded = OceanExperimentConfig.from_yaml(str(out))
    assert reloaded.get("ocean.A_h") == 2.0e4
    assert reloaded.to_ocean_config() == cfg.to_ocean_config()
