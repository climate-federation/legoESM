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


def test_eos_flag_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_dino", "--eos", "nemo_seos"])
    args = rd._parse_args()
    assert args.eos == "nemo_seos"


def test_eos_default_none(monkeypatch):
    """Default None → main() keeps DINOConfig.eos (wright)."""
    monkeypatch.setattr(sys, "argv", ["run_dino"])
    args = rd._parse_args()
    assert args.eos is None


def test_eos_via_config(tmp_path, monkeypatch):
    cfg = _write(tmp_path, "eos: nemo_seos\n")
    monkeypatch.setattr(sys, "argv", ["run_dino", "--config", cfg])
    args = rd._parse_args()
    assert args.eos == "nemo_seos"


def test_eos_rejects_bad_choice(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_dino", "--eos", "wrightt"])
    with pytest.raises(SystemExit):
        rd._parse_args()


def test_eos_replaces_dino_config(monkeypatch):
    """The --eos flag must reach DINOConfig.eos (the cfg-replace round-trip)."""
    import dataclasses
    from legoesm.ocean.experiments.dino import DINOConfig
    cfg = DINOConfig()
    assert cfg.eos == "wright"          # default
    cfg = dataclasses.replace(cfg, eos="nemo_seos")
    assert cfg.eos == "nemo_seos"


def test_tke_momentum_visc_bg_flag_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["run_dino", "--tke-momentum-visc-bg", "1.2e-4"])
    args = rd._parse_args()
    assert args.tke_momentum_visc_bg == 1.2e-4


def test_tke_momentum_visc_bg_default_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_dino"])
    args = rd._parse_args()
    assert args.tke_momentum_visc_bg is None


def test_A_v_bg_effective_floors_only_tke():
    """The TKE momentum-viscosity floor must apply ONLY to vmix='tke', leaving
    kpp/constant at the paper A_v_bg (they are stable there)."""
    import dataclasses
    from legoesm.ocean.experiments.dino import DINOConfig
    base = DINOConfig()
    assert base.A_v_bg == 1.2e-4 and base.tke_momentum_visc_bg == 5.0e-4
    # kpp/constant: paper A_v_bg, no floor.
    assert dataclasses.replace(base, vmix_scheme="kpp").A_v_bg_effective == 1.2e-4
    assert dataclasses.replace(
        base, vmix_scheme="constant").A_v_bg_effective == 1.2e-4
    # tke: floored to the stabilizer (max with A_v_bg).
    assert dataclasses.replace(
        base, vmix_scheme="tke").A_v_bg_effective == 5.0e-4
    # A higher A_v_bg override still wins (it's a floor, not an override).
    assert dataclasses.replace(
        base, vmix_scheme="tke", A_v_bg=2.0e-3).A_v_bg_effective == 2.0e-3
    # And the floor can be lowered to run TKE at the unstable paper viscosity.
    assert dataclasses.replace(
        base, vmix_scheme="tke",
        tke_momentum_visc_bg=1.2e-4).A_v_bg_effective == 1.2e-4
