"""Guard the parity/conservation gates of ``bench_ocean_latlon_spmd_scaling``.

The gates (ported from the retired ``--transport spmd`` lane of
``bench_ocean_mpi_scaling``) are the fail-fast correctness armor the
Derecho/Levante ocean GPU jobs run before any timed ladder:

* wiring tripwire — the gate flags/constants exist (a rename would only
  break at runtime on a cluster);
* an end-to-end single-process 2-virtual-device CPU run with BOTH gates
  armed exits 0, writes the JSONL record, and prints per-field parity lines;
* the smoke-window cap refuses long runs (the re-association floor grows
  with steps).

Multi-controller federation of the same bench:
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py (and the
launcher-gated test_latlon_ocean_spmd_multicontroller.py).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_BENCH = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "bench" / "bench_ocean_latlon_spmd_scaling.py"
)


def _load_bench():
    spec = importlib.util.spec_from_file_location(
        "bench_ocean_latlon_spmd_scaling", _BENCH,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_gate_symbols_and_flags_exist():
    mod = _load_bench()
    assert set(mod.SPMD_PARITY_TOLS) == {"float32", "float64"}
    for rtol, atol in mod.SPMD_PARITY_TOLS.values():
        assert rtol > 0 and atol > 0
    assert mod.SPMD_PARITY_MAX_STEPS >= 4
    # Parser accepts the gate flags (argparse would SystemExit on unknowns).
    import argparse  # noqa: F401  (documents the surface under test)
    for flag in ("--parity-gate", "--check-conservation", "--cons-rtol",
                 "--multicontroller", "--coordinator", "--pcg-precond",
                 "--cheb-degree"):
        assert flag in Path(_BENCH).read_text()


def test_pcg_precond_flags_reach_the_barotropic_config():
    """--pcg-precond / --cheb-degree land on the two config fields the
    implicit_cn solver dispatches on; jacobi (the default) leaves the
    scheme default untouched and the degree refuses to ride any other
    preconditioner."""
    mod = _load_bench()
    m, _ = mod.build_model_and_state(8, 16, 3, tripole=False,
                                     pcg_precond="chebyshev", cheb_degree=3)
    assert m.config.barotropic.barotropic_implicit_preconditioner == "chebyshev"
    assert int(m.config.barotropic.barotropic_chebyshev_degree) == 3
    m, _ = mod.build_model_and_state(8, 16, 3, tripole=False)
    assert m.config.barotropic.barotropic_implicit_preconditioner == "jacobi"
    # chebyshev without a degree runs the scheme default, which is 4, not 0
    m, _ = mod.build_model_and_state(8, 16, 3, tripole=False,
                                     pcg_precond="chebyshev")
    assert int(m.config.barotropic.barotropic_chebyshev_degree) == 4
    m, _ = mod.build_model_and_state(8, 16, 3, tripole=False,
                                     pcg_precond="zonal_line")
    assert m.config.barotropic.barotropic_implicit_preconditioner == "zonal_line"
    with pytest.raises(SystemExit, match="needs --pcg-precond chebyshev"):
        mod.build_model_and_state(8, 16, 3, tripole=False, cheb_degree=3)
    with pytest.raises(SystemExit, match=">= 1"):
        mod.build_model_and_state(8, 16, 3, tripole=False,
                                  pcg_precond="chebyshev", cheb_degree=-3)


# Budget: the smoke compiles the serial reference, the SPMD step, the fused
# scan AND (increment-2) the post-run residual-probe solve — ~9.5 min x64 on
# a shared 4-core CPU allocation (job 8914967); 570 s timed out there.
@pytest.mark.timeout(1300)
def test_single_process_two_virtual_devices_with_gates(tmp_path):
    out = tmp_path / "ocean_spmd.jsonl"
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    proc = subprocess.run(
        [
            sys.executable, str(_BENCH),
            "--n-devices", "2",
            "--n-lat", "16", "--n-lon", "32", "--nlev", "4",
            "--steps", "4", "--warmup", "1", "--dt", "600",
            "--parity-gate", "--check-conservation", "--cons-rtol", "1e-6",
            "--out", str(out),
        ],
        env=env, capture_output=True, text=True, timeout=1200,
    )
    assert proc.returncode == 0, (
        f"rc={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\n"
        f"stderr:\n{proc.stderr[-3000:]}"
    )
    rec = json.loads(out.read_text().strip().splitlines()[-1])
    assert rec["component"] == "ocean"
    assert rec["n_devices"] == 2
    assert "parity" in proc.stdout and "MISMATCH" not in proc.stdout

    # --- increment-2 fields (M1 audit items 4/6/8/9) present + HONEST ---
    # item 6: default implicit_cn at nd=2 runs the fixed-iteration PCG.
    # The canonical solver_residual is an honest NULL (the timed solve's
    # residual is never captured); the standalone zero-slow-forcing probe
    # must have MEASURED a converged residual under its own name (codex
    # batch4: the probe solves a different RHS from the timed step).
    assert rec["solver_iters"] >= 1
    assert rec["solver_iters_mode"].startswith("fixed_pcg")
    assert rec["solver_residual"] is None
    assert rec["residual_reason"]
    assert rec["zero_forcing_probe_measured"] is True
    assert rec["zero_forcing_probe_residual"] is not None
    assert rec["zero_forcing_probe_residual"] < 1e-6
    assert rec["metadata"]["solver_residual"] is None
    assert (rec["metadata"]["extra"]["zero_forcing_probe_measured"]
            is True)
    # item 4: bytes arithmetic consistent; fused scan never gathers; the
    # partial (barotropic-only) census is flagged machine-readably; the
    # split-explicit substep estimator must NOT be published on an
    # implicit-CN row (codex batch4).
    assert rec["full_state_gathers_per_step"] == 0
    assert rec["halo_messages_per_step"] > 0
    assert (rec["halo_bytes_per_step"]
            == rec["halo_messages_per_step"] * rec["halo_bytes_per_message"])
    assert rec["halo_bytes_is_lower_bound"] is True
    assert rec["metadata"]["extra"]["barotropic_halo_messages"] is None
    # item 8: nd=2 without --single-dev-fused-ms AND with the placeholder
    # (uncalibrated) fabric -> the bound must be NULL + named-incomplete
    # + uncalibrated, never fabricated.
    assert rec["t_bound_ms"] is None
    assert rec["measured_over_bound"] is None
    assert rec["bound_calibrated"] is False
    assert "single_device_fused_step_ms" in rec["bound_incomplete_reason"]
    assert any("latency_us" in r for r in rec["bound_incomplete_reason"])
    # item 9: IC-agnostic invariants (the default latlon rest state is
    # flat-bottom but NOT all-wet — the polar land-cap rows are masked, so
    # wet_fraction < 1 here; measured 0.875 on 16 lat rows = 2 cap rows).
    assert 0 < rec["wet_cell_levels"] <= rec["cells"]
    assert rec["wet_cell_levels"] == round(rec["wet_fraction"] * rec["cells"])
    assert (rec["wet_cell_levels_per_device_min"]
            <= rec["wet_cell_levels_per_device_max"])
    # The loud non-informative note fires IFF wet == total.
    assert (("NON-INFORMATIVE" in proc.stdout)
            == bool(rec["wet_equals_total"]))


def test_parity_gate_refuses_long_windows(tmp_path):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    proc = subprocess.run(
        [
            sys.executable, str(_BENCH),
            "--n-devices", "2", "--n-lat", "16", "--n-lon", "32",
            "--nlev", "4", "--steps", "30", "--warmup", "2",
            "--parity-gate", "--out", str(tmp_path / "x.jsonl"),
        ],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0
    assert "smoke gate" in (proc.stdout + proc.stderr)


def test_profile_dir_flag_defaults_off_and_reaches_the_block_timer():
    """A trace on by default would slow every tripole ladder receipt; with
    the flag the directory must reach timed_scan_blocks' trace_dir (the
    shared timer already knows how to trace) on ranks 0-3 only."""
    import inspect
    mod = _load_bench()
    p = mod.build_parser() if hasattr(mod, "build_parser") else None
    src = inspect.getsource(mod)
    assert 'add_argument("--profile-dir"' in src
    assert "trace_dir=_trace_dir" in src
    assert "jax.process_index() < 4" in src
    if p is not None:
        assert p.parse_args(["--n-lat", "8", "--n-lon", "16"]).profile_dir is None
