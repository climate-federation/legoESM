"""run_coupled --config YAML loader + the canonical CMIP run configs
(config/cmip/cmip_ocean_{slab,3D}.yaml).

Guards that (1) every key in the shipped configs is a real run_coupled argument
dest (a typo would be a silent dropped override), (2) the configs carry the
tuned ("trained") parameter values, and (3) --config sets defaults that an
explicit CLI flag still overrides.  Pure parser/YAML (no JAX run)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_run_coupled_mod", REPO / "scripts" / "run" / "run_coupled.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

CMIP_CONFIGS = [
    REPO / "config" / "cmip" / "cmip_ocean_slab.yaml",
    REPO / "config" / "cmip" / "cmip_ocean_3D.yaml",
]


def _dests():
    return {a.dest for a in mod.build_parser()._actions}


@pytest.mark.parametrize("cfg", CMIP_CONFIGS, ids=lambda p: p.name)
def test_cmip_config_exists_and_keys_valid(cfg):
    """Each shipped CMIP config exists and every key is a real run_coupled
    argument dest (no key references a flag run_coupled does not have)."""
    assert cfg.exists(), f"missing canonical config {cfg}"
    doc = yaml.safe_load(cfg.read_text())
    assert isinstance(doc, dict) and doc, f"{cfg.name} is not a non-empty mapping"
    unknown = sorted(set(doc) - _dests())
    assert not unknown, f"{cfg.name} has key(s) not in run_coupled: {unknown}"


def test_configs_carry_the_tuned_parameters():
    """The tuned air-sea / cloud / ocean parameter set is encoded as expected."""
    parser = mod.build_parser()
    slab = mod._load_yaml_config(str(CMIP_CONFIGS[0]), parser)
    assert slab["radiation"] == "rrtmgp"
    assert slab["surface_bulk_scheme"] == "coare3"
    assert slab["surface_gustiness_zi"] == 300.0
    assert slab["cloud_q_c_diagnostic"] == pytest.approx(3.0e-4)
    assert slab["convective_cloud"] is True
    assert slab["snow_albedo_feedback"] is True
    assert slab["ocean"] == "two_layer"

    d3 = mod._load_yaml_config(str(CMIP_CONFIGS[1]), parser)
    # 3D twin: identical tuned atm, only the ocean differs.
    assert d3["surface_bulk_scheme"] == "coare3"
    assert d3["surface_gustiness_zi"] == 300.0
    assert d3["ocean"] == "dynamic"
    assert d3["ocean_ic"] == "woa"
    assert d3["ocean_restore_sst_tau_days"] == 5.0
    assert d3["ocean_restore_sss_tau_days"] == 30.0


def test_load_yaml_config_rejects_unknown_key(tmp_path):
    """An unknown key is a hard error (no silent typo'd / dropped override)."""
    p = tmp_path / "bad.yaml"
    p.write_text("surface_bulk_scheme: coare3\nnot_a_real_arg: 1\n")
    with pytest.raises(SystemExit, match="unknown key"):
        mod._load_yaml_config(str(p), mod.build_parser())


def test_load_yaml_config_coerces_quoted_scalar(tmp_path):
    """A scalar written as a quoted string is coerced through the arg's type=
    (set_defaults bypasses argparse type conversion); a non-numeric value for a
    numeric arg is a hard error, not a silent str default."""
    good = tmp_path / "quoted.yaml"
    good.write_text('dt: "300"\nresolution: "24"\n')
    out = mod._load_yaml_config(str(good), mod.build_parser())
    assert out["dt"] == 300.0 and isinstance(out["dt"], float)
    assert out["resolution"] == 24 and isinstance(out["resolution"], int)

    bad = tmp_path / "badnum.yaml"
    bad.write_text("dt: not_a_number\n")
    with pytest.raises(SystemExit, match="not a valid"):
        mod._load_yaml_config(str(bad), mod.build_parser())


def test_config_sets_defaults_cli_overrides():
    """--config supplies defaults; an explicit CLI flag still wins (precedence
    CLI > config > parser default)."""
    parser = mod.build_parser()
    parser.set_defaults(**mod._load_yaml_config(str(CMIP_CONFIGS[0]), parser))
    args = parser.parse_args(["--days", "90"])
    assert args.surface_bulk_scheme == "coare3"   # from the config file
    assert args.ocean == "two_layer"              # from the config file
    assert args.days == 90                        # explicit CLI overrides config
