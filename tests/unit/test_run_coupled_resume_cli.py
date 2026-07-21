"""CLI + helper tests for run_coupled checkpoint/resume (dynamic-3D-ocean
ckpt v2 job-chaining): the --resume / --checkpoint-days / --max-wallclock-hours
flags and the _find_latest_checkpoint chain helper.

Pure (no JAX / no config build) so they run in unit CI; the full
run-checkpoint-restart-equivalence is the integration smoke
scripts/tmp/_ocean3d_ckpt_smoke.sbatch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_run_coupled_mod", REPO / "scripts" / "run" / "run_coupled.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_resume_flag_defaults_off():
    """Default run is fresh-start: --resume off, no checkpoint cadence, no
    wallclock budget (byte-identical to the pre-feature behaviour)."""
    args = mod.build_parser().parse_args([])
    assert args.resume is False
    assert args.checkpoint_days == 0
    assert args.max_wallclock_hours == 0.0


def test_resume_flags_roundtrip():
    """The chaining flags round-trip through the parser."""
    args = mod.build_parser().parse_args(
        ["--resume", "--checkpoint-days", "10", "--max-wallclock-hours", "11.5"])
    assert args.resume is True
    assert args.checkpoint_days == 10
    assert args.max_wallclock_hours == 11.5


def test_bulk_thermo_convention_flag_flows_to_config():
    """--bulk-thermo-convention (#762) round-trips through the parser: default
    "legoesm" (constant L_v / dry c_pd, byte-identical), "aerobulk" =
    NEMO/AeroBulk parity.  main() wires args.bulk_thermo_convention into
    ExperimentConfig.surface_thermo_convention, SimpleOceanConfig.
    thermo_convention, and CouplerConfig.thermo_convention (air-sea only)."""
    args = mod.build_parser().parse_args([])
    assert args.bulk_thermo_convention == "legoesm"
    args = mod.build_parser().parse_args(
        ["--surface-bulk-scheme", "coare3",
         "--bulk-thermo-convention", "aerobulk"])
    assert args.bulk_thermo_convention == "aerobulk"


def test_surface_stability_scheme_flag_round_trip():
    """--surface-stability-scheme round-trips; default dyer1974 is the
    byte-identical historical stable branch; unknown names are rejected at
    argparse (dispatch hardening); 'most' stays deliberately absent from
    --surface-bulk-scheme (the atmosphere surface layer treats it as
    constant, so offering it would split the interface)."""
    import pytest

    assert (mod.build_parser().parse_args([]).surface_stability_scheme
            == "dyer1974")
    args = mod.build_parser().parse_args(
        ["--surface-stability-scheme", "gryanik2020"])
    assert args.surface_stability_scheme == "gryanik2020"
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(
            ["--surface-stability-scheme", "dyer1975"])
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--surface-bulk-scheme", "most"])


def test_find_latest_checkpoint_picks_highest_day(tmp_path):
    """_find_latest_checkpoint returns the highest-day checkpoint (numeric, not
    lexicographic) and ignores non-matching files."""
    for day in (10, 30, 20, 5):
        (tmp_path / f"checkpoint_day_{day:04d}.npz").write_bytes(b"x")
    # Decoys that must be ignored.
    (tmp_path / "coupled_day_0030.npz").write_bytes(b"x")
    (tmp_path / "checkpoint_day_final.npz").write_bytes(b"x")
    path, day = mod._find_latest_checkpoint(str(tmp_path))
    assert day == 30
    assert path.endswith("checkpoint_day_0030.npz")


def test_find_latest_checkpoint_empty(tmp_path):
    """No checkpoint present => (None, None) so --resume starts fresh."""
    path, day = mod._find_latest_checkpoint(str(tmp_path))
    assert path is None and day is None
