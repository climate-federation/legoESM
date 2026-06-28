"""Tests for the run_dino.py ``--config`` YAML loader + the committed configs.

The loader maps a YAML of run parameters onto argparse defaults (so explicit
CLI flags still win), rejects unknown keys, and coerces Path-typed fields.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "run_dino", _REPO / "scripts" / "run" / "run_dino.py")
rd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rd)


def _write(tmp_path, text):
    c = tmp_path / "cfg.yaml"
    c.write_text(text)
    return str(c)


def test_config_loads_as_defaults(tmp_path, monkeypatch):
    cfg = _write(tmp_path,
                 "grid: mpas\ndays: 42\nsnapshot_every_days: 5\n"
                 "mpas_eq_visc_boost: 5.0\noutput_dir: results/x\n")
    monkeypatch.setattr(sys, "argv", ["run_dino", "--config", cfg])
    args = rd._parse_args()
    assert args.grid == "mpas"
    assert args.days == 42
    assert args.snapshot_every_days == 5
    assert args.mpas_eq_visc_boost == 5.0
    # Path-typed field coerced from the YAML string (else .mkdir() would crash).
    assert isinstance(args.output_dir, Path)
    assert str(args.output_dir) == "results/x"


def test_cli_overrides_config(tmp_path, monkeypatch):
    cfg = _write(tmp_path, "grid: mpas\ndays: 42\n")
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--config", cfg, "--days", "7"])
    args = rd._parse_args()
    assert args.days == 7        # explicit CLI flag wins
    assert args.grid == "mpas"   # still taken from the config


def test_unknown_key_raises(tmp_path, monkeypatch):
    cfg = _write(tmp_path, "grid: mpas\nbogus_key: 1\n")
    monkeypatch.setattr(sys, "argv", ["run_dino", "--config", cfg])
    with pytest.raises(SystemExit):
        rd._parse_args()


def test_committed_configs_parse_and_are_valid(monkeypatch):
    cfgs = sorted((_REPO / "scripts" / "experiment" / "dino").glob("*.yaml"))
    assert cfgs, "no committed DINO configs found"
    for y in cfgs:
        monkeypatch.setattr(sys, "argv", ["run_dino", "--config", str(y)])
        args = rd._parse_args()        # raises if a key is unknown
        assert args.grid in ("latlon", "mpas")
        assert args.days > 0
        assert isinstance(args.output_dir, Path)


def test_barotropic_solver_flag_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--barotropic-solver", "rigid_lid"])
    args = rd._parse_args()
    assert args.barotropic_solver == "rigid_lid"


def test_barotropic_solver_default_none(monkeypatch):
    """Default None → main() keeps DINOConfig.barotropic_solver (implicit_cn)."""
    monkeypatch.setattr(sys, "argv", ["run_dino"])
    args = rd._parse_args()
    assert args.barotropic_solver is None


def test_barotropic_solver_via_config(tmp_path, monkeypatch):
    cfg = _write(tmp_path, "barotropic_solver: rigid_lid\n")
    monkeypatch.setattr(sys, "argv", ["run_dino", "--config", cfg])
    args = rd._parse_args()
    assert args.barotropic_solver == "rigid_lid"


def test_barotropic_solver_rejects_bad_choice(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--barotropic-solver", "bogus"])
    with pytest.raises(SystemExit):
        rd._parse_args()


def test_rigid_lid_dt_mom_ratio_flag_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--barotropic-solver", "rigid_lid",
                         "--rigid-lid-dt-mom-ratio", "6"])
    args = rd._parse_args()
    assert args.rigid_lid_dt_mom_ratio == 6.0


def test_rigid_lid_dt_mom_ratio_default_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_dino"])
    args = rd._parse_args()
    assert args.rigid_lid_dt_mom_ratio is None
