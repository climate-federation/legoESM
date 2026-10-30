"""run_omip_core2.py CLI flags for short-run / synthetic testing.

Pins the three orchestration flags the ocean-runners tripole lane relies
on (legoESM-ocean-runners patch 002, contributed upstream): short runs on
a day budget without abusing ``--smoke``, explicit opt-in synthetic
forcing, and a forcing amplitude knob.  All three must parse with the
documented defaults; before the flags existed, argparse rejected them
(SystemExit 2) — these tests fail against that parser.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "run" / "run_omip_core2.py"


def _parser():
    spec = importlib.util.spec_from_file_location("run_omip_core2_cli", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._build_arg_parser()


def test_synthetic_forcing_flag_default_off_and_opt_in() -> None:
    """--synthetic-forcing defaults to False; the flag opts in.

    Default-off is the NEMO-faithful contract: a silent synthetic fallback
    would corrupt the comparison invisibly, so synthetic data must be an
    explicit choice.
    """
    parser = _parser()
    args = parser.parse_args([])
    assert args.synthetic_forcing is False
    args = parser.parse_args(["--synthetic-forcing"])
    assert args.synthetic_forcing is True


def test_forcing_scale_flag_default_one() -> None:
    """--forcing-scale defaults to 1.0 (unscaled) and parses a float."""
    parser = _parser()
    args = parser.parse_args([])
    assert args.forcing_scale == 1.0
    args = parser.parse_args(["--forcing-scale", "0.5"])
    assert args.forcing_scale == 0.5


def test_days_flag_overrides_years_and_smoke_budget() -> None:
    """--days defaults to None (fall back to --years / --smoke); when set
    it carries the run budget (main(): days > smoke > years)."""
    parser = _parser()
    args = parser.parse_args([])
    assert args.days is None
    args = parser.parse_args(["--smoke", "--days", "30"])
    assert args.days == 30.0
