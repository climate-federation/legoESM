"""CLI wiring for run_coupled's --sb81-omega-conversion (#1029 ω-side).

Pure config-level tests (no JAX step): the conversion numerics are covered by
tests/unit/test_hybrid_vertical.py (operator identities) and
tests/atmosphere/dycore/regression/test_latlon_pgf_rest_balance_1029.py
(rest balance with the flag ON + gate liveness)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_run_coupled_sb81_mod", REPO / "scripts" / "run" / "run_coupled.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_sb81_flag_defaults_off():
    """Unset flag => False => the legacy arithmetic conversion (bit-identical
    dycore; the SB81 form stays opt-in until the #1029(b) lid treatment)."""
    args = mod.build_parser().parse_args([])
    assert args.sb81_omega_conversion is False


def test_sb81_flag_roundtrip():
    args = mod.build_parser().parse_args(["--sb81-omega-conversion"])
    assert args.sb81_omega_conversion is True


def test_dycore_config_default_off():
    from legoesm.driver.config import DycoreConfig
    assert DycoreConfig().sb81_omega_conversion is False
