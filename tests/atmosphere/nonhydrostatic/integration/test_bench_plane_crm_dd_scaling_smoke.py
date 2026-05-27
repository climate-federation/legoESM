"""Smoke test for ``scripts/bench_plane_crm_dd_scaling.py``.

iter-252: the MPI strong/weak scaling benchmark for the plane CRM
step_halo path. Pre iter-252 it had ZERO test coverage — a silent
regression in the benchmark wiring (argparse, single-rank
single-process compose, CSV emit, mpi4py import path) would only
surface when a user tried to launch a real scaling sweep. Closes
that gap with a 1-rank in-process smoke that exercises:

* CLI parse + arg validation.
* PlaneCompressibleEulerModel + DD-layout construction at small mesh.
* step_halo path through warmup + timed window.
* CSV output schema (one row, 10 comma-separated fields).

Multi-rank coverage (``mpirun -np 2``+) is intentionally out of
scope: it requires an MPI launcher in the test runner and is more
appropriately covered by an integration sweep, not a CI smoke.
This test only certifies the script is buildable and runs end-to-end
on the single-rank fast path. CI wall budget: ~10-20 s cold JIT.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "bench_plane_crm_dd_scaling.py"


_EXPECTED_CSV_COLUMNS = 10  # mode, n_ranks, ny_global, nx_global,
                            # ny_local, nx_local, nlev, wall_s,
                            # steps_per_s, wall_per_step_s


@pytest.mark.parametrize("mode", ["strong", "weak"])
def test_bench_plane_crm_dd_scaling_single_rank_smoke(tmp_path, mode):
    """Strong + weak modes must both run end-to-end at np=1 + write
    the expected one-row CSV. The 12x12x8 mesh + 1 warmup + 3 timed
    steps keeps wall budget ~10 s cold JIT.
    """
    out_file = tmp_path / f"bench_{mode}_smoke.csv"
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(SCRIPT),
        "--mode", mode,
        "--nx", "12", "--ny", "12", "--nlev", "8",
        "--warmup-steps", "1",
        "--time-steps", "3",
        "--output", str(out_file),
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        pytest.fail(
            f"bench_plane_crm_dd_scaling.py --mode {mode} exited "
            f"{result.returncode}\n"
            f"stdout tail:\n{result.stdout[-1500:]}\n"
            f"stderr tail:\n{result.stderr[-1500:]}"
        )
    assert out_file.exists(), (
        f"Bench script did not write CSV at {out_file}.\n"
        f"stdout tail:\n{result.stdout[-500:]}"
    )
    # Header row + 1 data row, both 10 comma-separated fields.
    with open(out_file) as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 2, (
        f"Expected exactly 2 CSV rows (header + 1 data), got "
        f"{len(rows)}. rows={rows!r}"
    )
    header, row = rows
    expected_header = [
        "mode", "n_ranks", "ny_global", "nx_global",
        "ny_local", "nx_local", "nlev", "wall_s",
        "steps_per_s", "wall_per_step_s",
    ]
    assert header == expected_header, (
        f"CSV header drift: got {header!r}, expected "
        f"{expected_header!r}."
    )
    assert len(row) == _EXPECTED_CSV_COLUMNS, (
        f"Expected {_EXPECTED_CSV_COLUMNS} CSV columns, got "
        f"{len(row)}. row={row!r}"
    )
    # Mode column matches --mode.
    assert row[0] == mode, (
        f"CSV[0] = {row[0]!r}, expected {mode!r}"
    )
    # n_ranks column == 1 (single-rank smoke).
    assert int(row[1]) == 1, (
        f"CSV[1] (n_ranks) = {row[1]!r}, expected 1 (single-rank)."
    )
    # nlev column == 8 (matches --nlev).
    assert int(row[6]) == 8, (
        f"CSV[6] (nlev) = {row[6]!r}, expected 8."
    )
    # Wall time + throughput must be finite + positive.
    wall_s = float(row[7])
    steps_per_s = float(row[8])
    wall_per_step_s = float(row[9])
    assert wall_s > 0.0 and wall_s < 60.0, (
        f"wall_s={wall_s!r} outside (0, 60) bound for 3-step bench."
    )
    assert steps_per_s > 0.0, (
        f"steps_per_s={steps_per_s!r} must be positive."
    )
    assert wall_per_step_s > 0.0, (
        f"wall_per_step_s={wall_per_step_s!r} must be positive."
    )
    # Cross-check throughput consistency: steps_per_s ≈ 3 / wall_s.
    expected_throughput = 3.0 / wall_s
    assert abs(steps_per_s - expected_throughput) / expected_throughput < 0.02, (
        f"Throughput inconsistent: steps_per_s={steps_per_s:.4f} "
        f"vs 3/wall_s={expected_throughput:.4f}."
    )
