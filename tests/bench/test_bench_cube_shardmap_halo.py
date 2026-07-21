"""Direct tests for the Blocker-1 cube shard_map halo decisive-experiment harness.

Covers: device-count validation logic (pure, no JAX), the arg parser, an
in-process single-device smoke of the full run, a decisive-mode precondition
(single-device cannot pass a decisive run), and a hermetic 6-emulated-CPU-device
subprocess run that exercises the SPMD ppermute halo path + correctness + a
nonuniform-cotangent AD gate that also asserts a collective lowered in the HLO.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.bench.bench_cube_shardmap_halo as h


# --- Pure logic: device-count validation (no JAX) --------------------------
def test_valid_counts_skips_unavailable():
    runnable, skipped = h._valid_counts([1, 2, 3, 6], available=2, n_grid=24)
    assert runnable == [1, 2]
    assert skipped == [3, 6]


def test_valid_counts_rejects_non_face_divisors():
    runnable, skipped = h._valid_counts([1, 4, 5, 8], available=64, n_grid=24)
    assert runnable == [1]           # 4, 5, 8 are neither face divisors nor 6*kt^2
    assert skipped == [4, 5, 8]


def test_valid_counts_accepts_tiled_meshes():
    # 24 = 6*2^2 (kt=2), 54 = 6*3^2 (kt=3); n_grid=24 is divisible by 2 and 3.
    runnable, skipped = h._valid_counts([6, 24, 54], available=64, n_grid=24)
    assert runnable == [6, 24, 54]
    assert skipped == []


def test_valid_counts_tile_must_be_perfect_square_times_six():
    # 12 = 6*2 and 18 = 6*3 are multiples of 6 but 2, 3 are not perfect squares.
    runnable, skipped = h._valid_counts([12, 18], available=64, n_grid=24)
    assert runnable == []
    assert skipped == [12, 18]


def test_valid_counts_tile_factor_must_divide_n_grid():
    # 24 = 6*2^2 needs kt=2 | n_grid. n_grid=25 is odd -> the face cannot split
    # into 2 tiles, so the count is NOT runnable (would silently mis-shard).
    runnable, skipped = h._valid_counts([24], available=64, n_grid=25)
    assert runnable == []
    assert skipped == [24]


def test_arg_parser_defaults():
    args = h.build_arg_parser().parse_args([])
    assert args.device_counts == "1,2,3,6"
    assert args.pass_efficiency == 0.6
    assert args.gate_efficiency is False
    assert args.precision == "float32"


# --- In-process single-device smoke (1 device always available) ------------
@pytest.mark.slow
def test_single_device_run_smoke_passes(tmp_path):
    args = h.build_arg_parser().parse_args([
        "--device-counts", "1",
        "--n-grid", "12", "--n-lev", "4",
        "--n-warmup", "1", "--n-timing", "3", "--n-correctness-steps", "1",
        "--output-dir", str(tmp_path),
    ])
    result = h.run(args)
    # Non-decisive single-device smoke: correctness trivially exact, AD skipped
    # (needs >1), efficiency not enforced -> overall passes with finite timing.
    assert result["overall_pass"] is True
    assert result["scaling"][0]["n_devices"] == 1
    assert result["scaling"][0]["ms_per_step"] > 0.0
    assert result["correctness"]["worst_max_abs_diff"] == 0.0
    assert result["ad"]["ran"] is False
    assert "ad" not in result["enforced_gates"]


@pytest.mark.slow
def test_decisive_run_rejects_single_device(tmp_path):
    """A decisive run (--gate-efficiency) can never pass on one device: it has
    exercised neither SPMD, AD, nor scaling."""
    args = h.build_arg_parser().parse_args([
        "--device-counts", "1", "--gate-efficiency",
        "--n-grid", "12", "--n-lev", "4",
        "--n-warmup", "1", "--n-timing", "3", "--n-correctness-steps", "1",
        "--output-dir", str(tmp_path),
    ])
    result = h.run(args)
    assert result["overall_pass"] is False
    assert result["decisive_reasons"], "expected a decisive-precondition failure"


# --- Hermetic 6-emulated-device subprocess (real SPMD + AD + scaling) -------
@pytest.mark.slow
def test_spmd_six_device_subprocess(tmp_path):
    """Force 6 emulated CPU devices in a child process and run the full sweep.

    XLA device count must be fixed before JAX initialises, so this cannot run
    in-process alongside the single-device test — a subprocess is the hermetic
    way to exercise the ppermute halo path.
    """
    env = dict(os.environ)
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                        + " --xla_force_host_platform_device_count=6").strip()
    env["JAX_PLATFORMS"] = "cpu"
    out_dir = tmp_path / "spmd6"
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.bench.bench_cube_shardmap_halo",
         "--device-counts", "1,2,3,6",
         "--n-grid", "12", "--n-lev", "4",
         "--n-warmup", "1", "--n-timing", "3", "--n-correctness-steps", "2",
         "--precision", "float64",
         "--output-dir", str(out_dir)],
        env=env, cwd=str(repo_root),
        capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, (
        f"harness exited {proc.returncode}\nSTDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}")

    result = json.loads((out_dir / "results.json").read_text())
    counts = [r["n_devices"] for r in result["scaling"]]
    assert counts == [1, 2, 3, 6]
    # Correctness: SPMD halo moves exactly the right data -> tight match in fp64.
    assert result["gates"]["correctness"] is True
    assert result["correctness"]["worst_max_abs_diff"] <= result["correctness"]["tol"]
    # AD: nonuniform-cotangent VJP matches the single-device reference AND a
    # cross-device collective actually lowered (not a silent local fallback).
    assert result["ad"]["ran"] is True
    assert result["gates"]["ad"] is True, result["ad"]
    ad = result["ad"]["result"]
    assert ad["grad_rel_diff"] <= result["ad"]["tol"]
    assert ad["collective_in_hlo"] is True, (
        "SPMD AD path did not lower a collective — ppermute VJP not exercised")
    # SPMD timed steps confirmed to contain a cross-device collective.
    for r in result["scaling"]:
        if r["n_devices"] > 1:
            assert r["spmd_hlo_collective"] is True
    assert result["overall_pass"] is True


def _run_point(tmp_path, n_devices: int, out_dir: Path):
    """Launch the harness in --point mode with `n_devices` emulated CPU devices
    (single process; distributed init is a no-op, so device_count == n_devices)."""
    env = dict(os.environ)
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                        + f" --xla_force_host_platform_device_count={n_devices}").strip()
    env["JAX_PLATFORMS"] = "cpu"
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.bench.bench_cube_shardmap_halo", "--point",
         "--n-grid", "12", "--n-lev", "4", "--n-warmup", "1", "--n-timing", "3",
         "--precision", "float64", "--output-dir", str(out_dir)],
        env=env, cwd=str(repo_root), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"point n={n_devices} exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return json.loads((out_dir / "results.json").read_text())


@pytest.mark.slow
def test_point_mode_emits_valid_point(tmp_path):
    pt = _run_point(tmp_path, 6, tmp_path / "n6")
    assert pt["mode"] == "point"
    assert pt["n_devices"] == 6
    # Single-process launch: distributed init is a no-op, so one process owns all.
    assert pt["process_count"] == 1
    # The timed model.step lowered the ppermute collective.
    assert pt["spmd_hlo_collective"] is True
    assert pt["ms_per_step"] > 0.0


@pytest.mark.slow
def test_xnode_lane_end_to_end(tmp_path):
    """Full cross-node-lane logic locally: per-N --point launches (emulated single
    process each) feed the aggregator, which builds the curve + applies gates.
    Only real cross-process federation (gloo/NCCL) is cluster-only."""
    import scripts.bench.aggregate_cube_shardmap_scaling as agg

    root = tmp_path / "xnode"
    root.mkdir()
    for n in (1, 2, 3):
        pt = _run_point(tmp_path, n, root / f"n{n}")
        assert pt["n_devices"] == n
        if n > 1:
            assert pt["spmd_hlo_collective"] is True

    out = tmp_path / "agg"
    # Non-decisive: CPU efficiency anti-scales, so gate only correctness of the
    # curve/ppermute/baseline, not the efficiency threshold.
    rc = agg.main(["--root", str(root), "--output-dir", str(out)])
    assert rc == 0
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["device_counts"] == [1, 2, 3]
    assert verdict["gates"]["baseline_present"] is True
    assert verdict["gates"]["ppermute"] is True
    assert verdict["gates"]["config_consistent"] is True
    assert verdict["gates"]["has_multi_point"] is True
    assert verdict["overall_pass"] is True

    # Decisive path exercised end-to-end over the REAL point outputs: threshold 0
    # (CPU anti-scales) + require the exact counts present -> PASS.
    out2 = tmp_path / "agg_decisive"
    rc2 = agg.main(["--root", str(root), "--gate-efficiency", "--pass-efficiency", "0",
                    "--require-counts", "1,2,3", "--output-dir", str(out2)])
    assert rc2 == 0
    v2 = json.loads((out2 / "verdict.json").read_text())
    assert v2["gates"]["required_counts_present"] is True
    assert "efficiency" in v2["enforced_gates"]
    assert v2["overall_pass"] is True
