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
    # spot-check the documented cloud + convection mappings
    assert amap.get("atm.clouds.CloudConfig.q_c_diagnostic") == "cloud_q_c_diagnostic"
    assert amap.get("atm.conv.SBMConfig.rh_ref") == "sbm_RH_ref"


def test_atm_scalar_map_is_pipeline_threaded():
    """Every atm scalar-map entry is threaded END-TO-END: setting its
    ExperimentConfig scalar changes the resolved scheme config the pipeline
    builds.  This is the guard against the name-convention hazard — many
    ``<prefix>_<field>`` scalars EXIST but are never read (codex #691), so the
    map is a verified allowlist, not a convention.  Also fails if a NEW scalar
    becomes threaded but is missing from the map (extend it)."""
    from legoesm.atmosphere.physics.clouds.config import build_cloud_config
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
    )
    from legoesm.driver.physics_pipeline import build_physics_pipeline
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import make_hybrid_levels
    from legoesm.training.param_collector import build_registry

    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(10)
    reg = {m.qualified_name: m for m in build_registry()}
    # scheme selector that ACTIVATES each config class in the resolver.
    sel = {
        "CloudConfig": {"cloud_scheme": "sundqvist"},
        "SBMConfig": {"convection": "sbm", "microphysics": "kessler"},
        "BechtoldConfig": {"convection": "bechtold", "microphysics": "kessler"},
        "TiedtkeConfig": {"convection": "tiedtke", "microphysics": "kessler"},
        # warm-rain hard-saturation-adjustment overrides: activate each scheme
        # so _resolve_microphysics threads the flat scalar into its sub-config.
        "KesslerConfig": {"microphysics": "kessler"},
        "MorrisonConfig": {"microphysics": "morrison"},
        "P3Config": {"microphysics": "p3"},
        "SeifertBehengConfig": {"microphysics": "seifert_beheng"},
        "ThompsonConfig": {"microphysics": "thompson"},
        # GWD: gwd_config_for overlays the mcfarlane_*/hines_* scalars onto the
        # scheme leaf, and get_gwd_fn hands the pipeline that LEAF as gwd_config.
        "HinesConfig": {"gravity_wave_drag": "hines"},
        "McFarlaneConfig": {"gravity_wave_drag": "mcfarlane"},
        "ZhangMcFarlaneConfig": {"convection": "zhang_mcfarlane"},
        "CLUBBParams": {"turbulence": "clubb", "clubb_prognostic": True},
    }
    resolved_attr = {
        "ZhangMcFarlaneConfig": "convection_config",
        "HinesConfig": "gwd_config",
        "McFarlaneConfig": "gwd_config",
        "SBMConfig": "convection_config",
        "BechtoldConfig": "convection_config",
        "TiedtkeConfig": "convection_config",
        "KesslerConfig": "micro_config",
        "MorrisonConfig": "micro_config",
        "P3Config": "micro_config",
        "SeifertBehengConfig": "micro_config",
        "ThompsonConfig": "micro_config",
    }
    sentinel = 0.123456789
    for qname, ec_field in build_atm_scalar_param_map().items():
        m = reg[qname]
        base = {
            "grid": GridConfig(grid_type="cubed_sphere", resolution=4, nlev=10),
            "dycore": DycoreConfig(model_type="hydrostatic",
                                   discretization="cdgrid"),
            ec_field: sentinel,
            **sel.get(m.config_class, {}),
        }
        pipe = build_physics_pipeline(grid, sigma, ExperimentConfig(**base))
        if m.config_class == "CloudConfig":
            # Reconstruct the SAME build_cloud_config call the pipeline makes
            # (all threaded self._cloud_* attrs), else a genuinely-threaded
            # cloud param (p_xr/alpha_xr) would be falsely dropped.
            cc = build_cloud_config(
                "sundqvist",
                rh_crit=getattr(pipe, "_cloud_rh_crit", None),
                q_c_diagnostic=getattr(pipe, "_cloud_q_c_diagnostic", None),
                conv_cloud_max=getattr(pipe, "_cloud_conv_cloud_max", None),
                conv_cloud_condensate=getattr(
                    pipe, "_cloud_conv_cloud_condensate", None),
                cloud_inhomogeneity_factor=getattr(
                    pipe, "_cloud_inhomogeneity_factor", None),
                cloud_optics_inhomogeneity=getattr(
                    pipe, "_cloud_optics_inhomogeneity", None),
                cloud_fsd=getattr(pipe, "_cloud_fsd", None),
                p_xr=getattr(pipe, "_cloud_p_xr", None),
                alpha_xr=getattr(pipe, "_cloud_alpha_xr", None),
                diagnostic_condensate_scheme=getattr(
                    pipe, "_cloud_diagnostic_condensate_scheme", None),
                adiabatic_lwc_rate=getattr(
                    pipe, "_cloud_adiabatic_lwc_rate", None))
            got = getattr(cc, m.field)
        elif m.config_class == "CLUBBParams":
            got = getattr(pipe.turbulence_config.params, m.field)
        else:
            got = getattr(getattr(pipe, resolved_attr[m.config_class]), m.field)
        assert got == sentinel, (
            f"{qname} -> {ec_field} is in the atm scalar map but the pipeline "
            f"does NOT thread it into {m.config_class} (got {got!r}, not the "
            "sentinel).  Remove it from _ATM_SCALAR_PARAM_MAP or wire the "
            "pipeline to read it."
        )


def test_atm_scalar_map_has_no_under_claim():
    """Reverse of the over-claim test: every production cloud/convection param
    whose convention-named ExperimentConfig scalar the pipeline DOES thread must
    be IN the map — so a newly-threaded scalar can't be silently omitted (codex
    #691).  (The gray-radiation scheme's non-convention scalars are the
    documented conscious exclusion — see _ATM_SCALAR_PARAM_MAP.)"""
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
    )
    from legoesm.driver.physics_pipeline import build_physics_pipeline
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import make_hybrid_levels
    from legoesm.training.param_collector import build_registry

    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(10)
    ec_fields = set(ExperimentConfig._fields)
    amap = build_atm_scalar_param_map()
    sentinel = 0.123456789
    # (config_class, ExperimentConfig scalar prefix, scheme selector, how to
    # read the threaded value from the built pipeline).
    schemes = [
        ("CloudConfig", "cloud_", {"cloud_scheme": "sundqvist"},
         lambda pipe, field: getattr(pipe, f"_cloud_{field}", None)),
        ("SBMConfig", "sbm_", {"convection": "sbm", "microphysics": "kessler"},
         lambda pipe, field: getattr(pipe.convection_config, field, None)),
        ("BechtoldConfig", "bechtold_", {"convection": "bechtold",
                                         "microphysics": "kessler"},
         lambda pipe, field: getattr(pipe.convection_config, field, None)),
        # Tiedtke shares the UNPREFIXED autoconv_* flat scalars with Bechtold
        # (one ExperimentConfig scalar serves whichever mass-flux scheme is
        # active), so its convention prefix is empty.
        ("TiedtkeConfig", "", {"convection": "tiedtke",
                               "microphysics": "kessler"},
         lambda pipe, field: getattr(pipe.convection_config, field, None)),
        # Warm-rain micro families: their threaded flat scalars carry the SAME
        # name as the scheme field (hard_sat_adjust_threshold /
        # hard_sat_max_heating_K), so the convention prefix is empty — the
        # ec_lower lookup below simply skips micro fields with no same-named
        # ExperimentConfig scalar.
        ("KesslerConfig", "", {"microphysics": "kessler"},
         lambda pipe, field: getattr(pipe.micro_config, field, None)),
        ("MorrisonConfig", "", {"microphysics": "morrison"},
         lambda pipe, field: getattr(pipe.micro_config, field, None)),
        ("P3Config", "", {"microphysics": "p3"},
         lambda pipe, field: getattr(pipe.micro_config, field, None)),
        ("SeifertBehengConfig", "", {"microphysics": "seifert_beheng"},
         lambda pipe, field: getattr(pipe.micro_config, field, None)),
        ("ThompsonConfig", "", {"microphysics": "thompson"},
         lambda pipe, field: getattr(pipe.micro_config, field, None)),
        # GWD families: _resolve_gwd -> gwd_config_for overlays the flat
        # scalars onto the scheme leaf, which get_gwd_fn returns as gwd_config.
        ("HinesConfig", "hines_", {"gravity_wave_drag": "hines"},
         lambda pipe, field: getattr(pipe.gwd_config, field, None)),
        ("McFarlaneConfig", "mcfarlane_", {"gravity_wave_drag": "mcfarlane"},
         lambda pipe, field: getattr(pipe.gwd_config, field, None)),
        ("ZhangMcFarlaneConfig", "zm_", {"convection": "zhang_mcfarlane"},
         lambda pipe, field: getattr(pipe.convection_config, field, None)),
        ("CLUBBParams", "clubb_", {"turbulence": "clubb",
                                   "clubb_prognostic": True},
         lambda pipe, field: getattr(pipe.turbulence_config.params, field)),
    ]
    # Companion drift-guard: the family list scanned below must exactly match
    # the config classes present in the verified allowlist map.  The selector /
    # prefix / reader triple is per-family knowledge that CANNOT be derived
    # from the resolver (threading is scattered hand-written getattr code in
    # physics_pipeline), so when a future resolver threads a NEW scheme
    # family's scalars, extend BOTH _ATM_SCALAR_PARAM_MAP and this `schemes`
    # list — this assertion goes red until both agree.
    reg_by_qname = {m.qualified_name: m for m in build_registry()}
    map_classes = {reg_by_qname[q].config_class for q in amap}
    scanned_classes = {cls for cls, _, _, _ in schemes}
    assert map_classes == scanned_classes, (
        f"_ATM_SCALAR_PARAM_MAP covers config classes {sorted(map_classes)} but "
        f"this under-claim scan covers {sorted(scanned_classes)} — extend the "
        "schemes list (selector + prefix + reader) so newly-threaded families "
        "are scanned too."
    )
    ec_lower = {f.lower(): f for f in ec_fields}
    for cls, prefix, sel, reader in schemes:
        params = [m for m in build_registry()
                  if m.config_class == cls and 1 <= m.tunable_tier <= 2]
        for m in params:
            ec_field = ec_lower.get(f"{prefix}{m.field}".lower())
            if ec_field is None:
                continue  # no convention scalar -> genuinely unreachable
            base = {
                "grid": GridConfig(grid_type="cubed_sphere", resolution=4,
                                   nlev=10),
                "dycore": DycoreConfig(model_type="hydrostatic",
                                       discretization="cdgrid"),
                ec_field: sentinel,
                **sel,
            }
            pipe = build_physics_pipeline(grid, sigma, ExperimentConfig(**base))
            threaded = reader(pipe, m.field) == sentinel
            if threaded:
                assert m.qualified_name in amap, (
                    f"{m.qualified_name} is threaded from ExperimentConfig scalar "
                    f"{ec_field!r} but is MISSING from _ATM_SCALAR_PARAM_MAP — add "
                    "it (a newly-threaded scalar must be settable via --params)."
                )


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
