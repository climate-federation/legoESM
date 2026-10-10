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
              "barotropic_implicit_pcg_poly_sweeps",
              "barotropic_implicit_pcg_variant")

# Owner decisions 2026-10-04/05 and 2026-10-10 (CPU), written out independently of the table.
_PRODUCTION = {"cpu": (40, "jacobi", 4, "chebyshev_deep"),
               "gpu": (20, "gpoly", 4, "standard")}


def _resolved(argv):
    args = mod.build_parser().parse_args(["--n-devices", "1", *argv])
    cfg = mod.apply_pcg_overrides(
        MPASOceanConfig(barotropic_implicit_pcg_variant=args.pcg_variant), args)
    return {f: getattr(cfg, f) for f in PCG_FIELDS}


def _bundle():
    from legoesm.ocean.mpas_config import resolve_barotropic_pcg_defaults
    return resolve_barotropic_pcg_defaults(MPASOceanConfig())


def test_no_flag_is_the_config_default():
    """No flag = the backend's production bundle."""
    import jax
    assert _resolved([]) == dict(zip(PCG_FIELDS, _PRODUCTION[jax.default_backend()]))
    args = mod.build_parser().parse_args(["--n-devices", "1"])
    assert (args.pcg_fixed_iters, args.pcg_precond, args.pcg_poly_sweeps) == (None, None, None)


def test_each_flag_overrides_independently():
    import pytest
    b = _bundle()
    d = {f: getattr(b, f) for f in PCG_FIELDS}
    other = "jacobi" if d["barotropic_implicit_pcg_precond"] == "poly" else "poly"
    # another preconditioner alone would mix bundles: refused before launch
    with pytest.raises(ValueError, match="also pin"):
        _resolved(["--pcg-precond", other])
    r = _resolved(["--pcg-precond", other, "--pcg-fixed-iters", "20",
                   "--pcg-variant", "standard"])
    assert r["barotropic_implicit_pcg_precond"] == other
    assert r["barotropic_implicit_pcg_fixed_iters"] == 20
    assert r["barotropic_implicit_pcg_poly_sweeps"] == d["barotropic_implicit_pcg_poly_sweeps"]
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


def test_the_timed_blocks_are_not_gated_on_the_rank():
    """A fused block is COLLECTIVE, so no rank-local guard may wrap the call
    that runs it.  Only the trace directory handed to it may be rank-local:
    ranks that skip the blocks would not join their halo exchanges or the
    solver's reductions, and the job hangs until its wall limit.  An earlier
    draft of the profiling window did exactly that, and this gate goes red
    against it.
    """
    import ast

    src = (_BENCH_DIR / "bench_ocean_mpas_spmd_scaling.py").read_text()
    tree = ast.parse(src)

    def mentions_the_rank(node):
        return any(isinstance(n, ast.Attribute) and n.attr == "process_index"
                   for n in ast.walk(node))

    timed = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "timed_scan_blocks"]
    assert timed, "the timed-block call is gone; update this gate"

    guards = [n for n in ast.walk(tree)
              if isinstance(n, ast.If) and mentions_the_rank(n.test)]
    assert guards, "no rank guard at all; update this gate"
    for guard in guards:
        for call in timed:
            assert call not in ast.walk(guard), (
                "the timed blocks sit inside a rank-local guard: the ranks "
                "that skip them will not join their collectives and the job "
                "hangs")

    # And the trace directory itself MUST stay rank-local, or all 128 ranks
    # write traces and the analyzer's same-node clock assumption is void.
    assert any(mentions_the_rank(g.test)
               and any(isinstance(n, ast.Name) and n.id == "trace_dir"
                       for n in ast.walk(g))
               for g in guards), "the trace directory is no longer rank-local"


def test_gpoly_flag_and_halo_depth_default():
    """--pcg-precond gpoly reaches the config; --halo-depth is unset by
    default so the layout depth comes from the config's own requirement."""
    from legoesm.parallel.voronoi_spmd_ocean import halo_depth_for_config
    assert _resolved(["--pcg-precond", "gpoly", "--pcg-variant", "standard",
                      "--pcg-fixed-iters", "15"])[
        "barotropic_implicit_pcg_precond"] == "gpoly"
    args = mod.build_parser().parse_args(["--n-devices", "1"])
    assert args.halo_depth is None
    assert mod.build_parser().parse_args(
        ["--n-devices", "1", "--halo-depth", "4"]).halo_depth == 4
    cfg = MPASOceanConfig()
    assert halo_depth_for_config(cfg) == 2
    for k, need in ((2, 2), (4, 2), (6, 4), (8, 6)):
        g = cfg._replace(barotropic_implicit_pcg_precond="gpoly",
                         barotropic_implicit_pcg_variant="standard",
                         barotropic_implicit_pcg_fixed_iters=20,
                         barotropic_implicit_pcg_poly_sweeps=k)
        assert halo_depth_for_config(g) == need
