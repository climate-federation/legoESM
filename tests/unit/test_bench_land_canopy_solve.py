"""Direct test for scripts/bench/bench_land_canopy_solve.py: the timed call
traces on a few synthetic columns and returns per-column outputs (shape only;
no compile)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax


def test_build_call_traces_per_column_outputs():
    script = (Path(__file__).resolve().parents[2]
              / "scripts" / "bench" / "bench_land_canopy_solve.py")
    spec = importlib.util.spec_from_file_location("_bench_land_canopy", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    call, args = mod.build_call(ncol=4, cap=2)
    out = jax.eval_shape(call, *args)
    assert {"shflx", "lhflx", "converged"} <= set(out)
    assert out["shflx"].shape == (4,)
    assert out["converged"].shape == (4,)
