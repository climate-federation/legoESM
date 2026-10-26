"""CLI contract of scripts/bench/run_cpu_mpi_scaling.py that the CPU
profiling campaign relies on: the trace switch exists, defaults off, and
reaches the timing function."""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

_MOD = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "run_cpu_mpi_scaling.py"
_spec = importlib.util.spec_from_file_location("run_cpu_mpi_scaling", _MOD)
rc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rc  # dataclass annotations resolve through sys.modules
_spec.loader.exec_module(rc)


def test_profile_dir_flag_defaults_off_and_reaches_the_timing_function(tmp_path):
    """Without the flag no trace is written (a trace slows the timed scan and
    would silently bias every ladder receipt); with it the directory reaches
    run_single_benchmark, whose signature must carry it."""
    p = rc.build_parser()
    args = p.parse_args(["--grid", "latlon", "--resolution", "64"])
    assert args.profile_dir is None
    args = p.parse_args(["--grid", "latlon", "--resolution", "64",
                         "--profile-dir", str(tmp_path)])
    assert args.profile_dir == str(tmp_path)
    assert "profile_dir" in inspect.signature(rc.run_single_benchmark).parameters
    src = inspect.getsource(rc.run_single_benchmark)
    assert "jax.profiler.start_trace" in src and "jax.profiler.stop_trace" in src
