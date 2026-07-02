"""run_coupled --config YAML loader + the canonical CMIP run configs
(config/cmip/cmip_tuned_physics.yaml shared base + cmip_ocean_{slab,3D}.yaml).

Guards that (1) every (include-merged) key in the shipped configs is a real
run_coupled argument dest, (2) the configs carry the tuned ("trained") parameter
values, (3) `include:` merges a base (base < including file), and (4) --config
sets defaults that an explicit CLI flag still overrides.  Pure parser/YAML."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from legoesm.driver.run_config_yaml import load_yaml_config, read_yaml_with_includes

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_run_coupled_mod", REPO / "scripts" / "run" / "run_coupled.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

CMIP_DIR = REPO / "config" / "cmip"
BASE = CMIP_DIR / "cmip_tuned_physics.yaml"
CMIP_CONFIGS = [CMIP_DIR / "cmip_ocean_slab.yaml", CMIP_DIR / "cmip_ocean_3D.yaml"]


def _dests():
    return {a.dest for a in mod.build_parser()._actions}


@pytest.mark.parametrize("cfg", [BASE, *CMIP_CONFIGS], ids=lambda p: p.name)
def test_config_keys_valid_after_include(cfg):
    """Every key (after merging any `include:` base) in the shared base and the
    run configs is a real run_coupled argument dest — a typo would be a silent
    dropped override.  Also exercises that the real `include:` resolves."""
    assert cfg.exists(), f"missing canonical config {cfg}"
    merged = read_yaml_with_includes(str(cfg))
    assert isinstance(merged, dict) and merged, f"{cfg.name} empty/not a mapping"
    assert "include" not in merged, "include meta-key must be stripped"
    unknown = sorted(set(merged) - _dests())
    assert not unknown, f"{cfg.name}: keys not in run_coupled: {unknown}"


def test_configs_carry_the_tuned_parameters():
    """The tuned air-sea / cloud / ocean set is present after the include merge:
    the run configs inherit the physics from the shared base."""
    parser = mod.build_parser()
    slab = load_yaml_config(str(CMIP_CONFIGS[0]), parser)
    assert slab["radiation"] == "rrtmgp"
    assert slab["surface_bulk_scheme"] == "coare3"       # from the base
    assert slab["surface_gustiness_zi"] == 300.0         # from the base
    assert slab["cloud_q_c_diagnostic"] == pytest.approx(3.0e-4)  # from the base
    assert slab["convective_cloud"] is True
    assert slab["snow_albedo_feedback"] is True
    assert slab["ocean"] == "two_layer"                  # slab's own

    d3 = load_yaml_config(str(CMIP_CONFIGS[1]), parser)
    assert d3["surface_bulk_scheme"] == "coare3"         # same base physics
    assert d3["surface_gustiness_zi"] == 300.0
    assert d3["ocean"] == "dynamic"                      # 3D's own
    assert d3["ocean_ic"] == "woa"
    assert d3["ocean_restore_sst_tau_days"] == 5.0
    assert d3["ocean_restore_sss_tau_days"] == 30.0


def test_include_merges_base_then_overrides(tmp_path):
    """`include:` merges the base first; the including file overrides on conflict
    (precedence base < file), and the meta-key is stripped."""
    (tmp_path / "base.yaml").write_text(
        "surface_bulk_scheme: coare3\nocean: two_layer\ndays: 10\n")
    child = tmp_path / "child.yaml"
    child.write_text("include: base.yaml\nocean: dynamic\ndays: 240\n")
    merged = read_yaml_with_includes(str(child))
    assert merged["surface_bulk_scheme"] == "coare3"  # inherited from base
    assert merged["ocean"] == "dynamic"               # child overrides base
    assert merged["days"] == 240                      # child overrides base
    assert "include" not in merged


def test_include_cycle_and_missing_raise(tmp_path):
    (tmp_path / "a.yaml").write_text("include: b.yaml\n")
    (tmp_path / "b.yaml").write_text("include: a.yaml\n")
    with pytest.raises(SystemExit, match="cycle"):
        read_yaml_with_includes(str(tmp_path / "a.yaml"))
    (tmp_path / "c.yaml").write_text("include: nope.yaml\n")
    with pytest.raises(SystemExit, match="not found"):
        read_yaml_with_includes(str(tmp_path / "c.yaml"))


def test_load_yaml_config_rejects_unknown_key(tmp_path):
    """An unknown key is a hard error (no silent typo'd / dropped override)."""
    p = tmp_path / "bad.yaml"
    p.write_text("surface_bulk_scheme: coare3\nnot_a_real_arg: 1\n")
    with pytest.raises(SystemExit, match="unknown key"):
        load_yaml_config(str(p), mod.build_parser())


def test_load_yaml_config_coerces_quoted_scalar(tmp_path):
    """A scalar written as a quoted string is coerced through the arg's type=
    (set_defaults bypasses argparse conversion); a non-numeric value raises."""
    good = tmp_path / "quoted.yaml"
    good.write_text('dt: "300"\nresolution: "24"\n')
    out = load_yaml_config(str(good), mod.build_parser())
    assert out["dt"] == 300.0 and isinstance(out["dt"], float)
    assert out["resolution"] == 24 and isinstance(out["resolution"], int)
    bad = tmp_path / "badnum.yaml"
    bad.write_text("dt: not_a_number\n")
    with pytest.raises(SystemExit, match="not a valid"):
        load_yaml_config(str(bad), mod.build_parser())


def test_orbital_insolation_flag_and_yaml_roundtrip(tmp_path):
    """Coupled CMIP runs can enable Berger-1978 orbital insolation through the
    run interface (codex adversarial review): the --orbital-insolation flag
    parses, and a coupled YAML carrying ``orbital_insolation: true`` is a valid
    key (NOT rejected as unknown) that sets the default; CLI still overrides."""
    # The dest must exist or load_yaml_config rejects the YAML key as unknown.
    assert "orbital_insolation" in _dests()
    parser = mod.build_parser()
    assert parser.parse_args([]).orbital_insolation is False
    assert parser.parse_args(["--orbital-insolation"]).orbital_insolation is True
    # The coupled ExperimentConfig actually carries the field that main()
    # forwards args.orbital_insolation into.
    from legoesm.driver.config import ExperimentConfig
    assert "orbital_insolation" in ExperimentConfig._fields

    p = tmp_path / "orb.yaml"
    p.write_text("orbital_insolation: true\nradiation: rrtmgp\n")
    loaded = load_yaml_config(str(p), mod.build_parser())
    assert loaded["orbital_insolation"] is True
    parser2 = mod.build_parser()
    parser2.set_defaults(**loaded)
    assert parser2.parse_args([]).orbital_insolation is True            # YAML default
    assert parser2.parse_args([]).orbital_insolation is True
    # An explicit on-flag is idempotent; the YAML cannot silently disable it.
    assert parser2.parse_args(["--orbital-insolation"]).orbital_insolation is True


def test_config_sets_defaults_cli_overrides():
    """--config supplies defaults; an explicit CLI flag still wins (precedence
    CLI > config-file > included base > parser default)."""
    parser = mod.build_parser()
    parser.set_defaults(**load_yaml_config(str(CMIP_CONFIGS[0]), parser))
    args = parser.parse_args(["--days", "90"])
    assert args.surface_bulk_scheme == "coare3"   # from the base via include
    assert args.ocean == "two_layer"              # from the run config
    assert args.days == 90                        # explicit CLI overrides config


# --- issue #691: --params calibration across every coupled component --------

def test_params_dest_exists():
    assert "params" in _dests()


def test_params_bundle_routes_ice_coupler_land(tmp_path):
    """The run_coupled --params application (atm scalar map + the
    coupled/coupler/ice/lake bundle) routes an atm, ice, coupler, and land
    parameter each into its component config — the exact split-and-bundle
    contract run_coupled.main() implements (#691)."""
    from typing import NamedTuple

    from legoesm.coupler.config import CouplerConfig
    from legoesm.coupler.lake.config import LakeConfig
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
    )
    from legoesm.driver.coupled_config import CoupledConfig
    from legoesm.driver.run_config_yaml import (
        apply_params_to_config,
        build_atm_scalar_param_map,
        load_params_config,
    )
    from legoesm.ice.config import SeaIceConfig
    from legoesm.land.config import MultiLayerLandConfig

    p = tmp_path / "params.yaml"
    p.write_text(
        "atm.clouds.CloudConfig.q_c_diagnostic: 3.0e-4\n"
        "ice.sea_ice.albedo_ice: 0.7\n"
        "coupler.surface.ocean_albedo: 0.08\n"
        "land.multilayer.Cd_land: 3.0e-3\n"
    )
    params = load_params_config(str(p))
    amap = build_atm_scalar_param_map()
    atm_params = {k: v for k, v in params.items() if k in amap}
    rest_params = {k: v for k, v in params.items() if k not in amap}

    atm_config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=5),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"),
        cloud_scheme="sundqvist",
    )
    atm_config = apply_params_to_config(
        atm_config, atm_params, driver="run_coupled", scalar_param_map=amap)
    assert atm_config.cloud_q_c_diagnostic == 3.0e-4

    class _Bundle(NamedTuple):
        coupled: object
        coupler: object
        ice: object
        lake: object

    bundle = _Bundle(
        coupled=CoupledConfig(land_config=MultiLayerLandConfig()),
        coupler=CouplerConfig(),
        ice=SeaIceConfig(),
        lake=LakeConfig(),
    )
    bundle = apply_params_to_config(bundle, rest_params, driver="run_coupled")
    assert bundle.ice.albedo_ice == 0.7
    assert bundle.coupler.ocean_albedo == 0.08
    assert bundle.coupled.land_config.Cd_land == 3.0e-3


def test_params_clobber_guard():
    """--params overrides that a coupled setup() step would silently overwrite
    are refused loudly, pointing at the effective path (#691 codex audit)."""
    # ocean: rebuilt from mesh/preset -> route to run_omip.
    with pytest.raises(SystemExit, match="run_omip.py --params"):
        mod._check_params_clobber({"ocean.vm.kpp.Ri_crit": 0.3}, "analytical")
    # land carbon: rebuilt in setup regardless of land path -> route to run_lmip.
    with pytest.raises(SystemExit, match="run_lmip.py --params"):
        mod._check_params_clobber({"land.carbon.tor_wood": 1.0e-4}, "analytical")
    # CLM land path (default): ALL config-level land params overridden by the
    # reference maps + per-PFT provider -> refuse every land.* param.
    for qname in ("land.multilayer.Cd_land", "land.soil_thermal.Q_geothermal",
                  "land.stomata.g1_med", "land.multilayer.z0_land"):
        with pytest.raises(SystemExit, match="--land-params clm"):
            mod._check_params_clobber({qname: 1.0}, "clm")
    # analytical land params: non-carbon land honored (config-level land used).
    mod._check_params_clobber({"land.multilayer.Cd_land": 3.0e-3}, "analytical")
    mod._check_params_clobber({"land.soil_thermal.Q_geothermal": 0.05}, "analytical")
    mod._check_params_clobber({"land.stomata.g1_med": 4.0}, "analytical")
    # atmosphere / sea-ice / coupler params always pass (run_coupled owns them).
    mod._check_params_clobber(
        {"ice.sea_ice.albedo_ice": 0.7,
         "coupler.surface.ocean_albedo": 0.08,
         "atm.clouds.CloudConfig.q_c_diagnostic": 3.0e-4}, "clm")
