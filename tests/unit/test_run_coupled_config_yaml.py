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


def test_config_sets_defaults_cli_overrides():
    """--config supplies defaults; an explicit CLI flag still wins (precedence
    CLI > config-file > included base > parser default)."""
    parser = mod.build_parser()
    parser.set_defaults(**load_yaml_config(str(CMIP_CONFIGS[0]), parser))
    args = parser.parse_args(["--days", "90"])
    assert args.surface_bulk_scheme == "coare3"   # from the base via include
    assert args.ocean == "two_layer"              # from the run config
    assert args.days == 90                        # explicit CLI overrides config
