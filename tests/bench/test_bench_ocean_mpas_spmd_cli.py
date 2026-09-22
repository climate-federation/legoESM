"""CLI contract of the ocean-MPAS SPMD scaling bench: a PCG flag left unset
keeps the MPASOceanConfig default, so a ladder arm without flags measures
the production solver.  Fails when a bench-side default reappears (the
2026-09-21 defect: --pcg-precond defaulted to "jacobi" and overrode the
adopted "poly", and a whole CPU ladder measured the retired solver)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parents[2] / "scripts" / "bench"
sys.path.insert(0, str(_BENCH_DIR))
_spec = importlib.util.spec_from_file_location(
    "bench_ocean_mpas_spmd", _BENCH_DIR / "bench_ocean_mpas_spmd_scaling.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

from legoesm.ocean.mpas_config import MPASOceanConfig  # noqa: E402

PCG_FIELDS = ("barotropic_implicit_pcg_fixed_iters",
              "barotropic_implicit_pcg_precond",
              "barotropic_implicit_pcg_poly_sweeps")


def _resolved(argv):
    args = mod.build_parser().parse_args(["--n-devices", "1", *argv])
    cfg = mod.apply_pcg_overrides(MPASOceanConfig(), args)
    return {f: getattr(cfg, f) for f in PCG_FIELDS}


def test_no_flag_is_the_config_default():
    defaults = {f: MPASOceanConfig._field_defaults[f] for f in PCG_FIELDS}
    assert _resolved([]) == defaults
    args = mod.build_parser().parse_args(["--n-devices", "1"])
    assert (args.pcg_fixed_iters, args.pcg_precond, args.pcg_poly_sweeps) == (None, None, None)


def test_each_flag_overrides_independently():
    d = MPASOceanConfig._field_defaults
    other = "jacobi" if d["barotropic_implicit_pcg_precond"] == "poly" else "poly"
    r = _resolved(["--pcg-precond", other])
    assert r["barotropic_implicit_pcg_precond"] == other
    assert r["barotropic_implicit_pcg_poly_sweeps"] == d["barotropic_implicit_pcg_poly_sweeps"]
    assert r["barotropic_implicit_pcg_fixed_iters"] == d["barotropic_implicit_pcg_fixed_iters"]
    r = _resolved(["--pcg-poly-sweeps", "7", "--pcg-fixed-iters", "11"])
    assert r["barotropic_implicit_pcg_poly_sweeps"] == 7
    assert r["barotropic_implicit_pcg_fixed_iters"] == 11
    assert r["barotropic_implicit_pcg_precond"] == d["barotropic_implicit_pcg_precond"]


def test_profile_dir_is_off_unless_asked_and_leaves_the_solver_alone():
    """The profiling window must be opt-in and must not touch the solver.

    It replays four extra steps AFTER the timed window, so a default that
    silently switched it on would both inflate the arm's wall time and put
    profiler overhead inside a scaling receipt.
    """
    args = mod.build_parser().parse_args(["--n-devices", "1"])
    assert args.profile_dir is None
    assert _resolved([]) == _resolved(["--profile-dir", "/tmp/does-not-matter"])
    on = mod.build_parser().parse_args(
        ["--n-devices", "1", "--profile-dir", "/tmp/x"])
    assert on.profile_dir == "/tmp/x"
