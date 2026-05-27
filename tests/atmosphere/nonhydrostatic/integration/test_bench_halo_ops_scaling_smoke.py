"""Smoke test for ``scripts/bench_halo_ops_scaling.py``.

iter-254: companion to the iter-252 bench_plane_crm_dd_scaling
smoke. bench_halo_ops_scaling benches halo-aware operators (8
operators per "tendency step" — grad_x, grad_y, divergence,
laplacian, 2 cell↔face interps each direction) and decomposes
total cost into local_compute + halo_comm. Useful for diagnosing
where the F9 MPI overhead lives.

Pre iter-254 it had ZERO test coverage; iter-254 adds a single-
rank single-process smoke that exercises argparse + packed halo
exchange + the 8-operator local_compute kernel + stdout schema.

Test does NOT verify halo_comm > 0 (single-rank halo_comm
exercises the local copy path which is fast but nonzero); only
that the stdout reports finite positive values for total / local
/ comm.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "bench_halo_ops_scaling.py"


# iter-256 (Codex iter-252..255 round-1 HIGH#2): end-anchored
# regex so a future PR that adds a trailing field to the bench
# output line fails the schema lock loudly. Pre-iter-256 the regex
# only used match() with no $ — trailing additions would pass.
_LINE_RE = re.compile(
    r"(?P<label>\S+)\s+nranks=\s*(?P<nranks>\d+)\s+"
    r"grid=(?P<ny>\d+)x(?P<nx>\d+)\s+"
    r"ny_local=(?P<ny_local>\d+)\s+"
    r"nx_local=(?P<nx_local>\d+)\s+"
    r"total=\s*(?P<total>[\d.]+)ms\s+"
    r"local_compute=\s*(?P<local>[\d.]+)ms\s+"
    r"halo_comm=\s*(?P<comm>-?[\d.]+)ms\s*$"
)


def test_bench_halo_ops_scaling_single_rank_smoke(tmp_path):
    """Bench script runs the 8-operator halo-aware pipeline + emits
    the standard ``total=X local_compute=Y halo_comm=Z`` line at
    rank 0. Single-rank smoke catches argparse regressions,
    JIT-compile errors, layout-construction bugs, and stdout-format
    drift in O(few-second) cold JIT wall.
    """
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(SCRIPT),
        "--nx", "12", "--ny", "12", "--nlev", "8",
        "--n-warmup", "1",
        "--n-bench", "3",
        "--label", "test_smoke",
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        pytest.fail(
            f"bench_halo_ops_scaling.py exited "
            f"{result.returncode}\n"
            f"stdout tail:\n{result.stdout[-1500:]}\n"
            f"stderr tail:\n{result.stderr[-1500:]}"
        )
    # Find the bench line in stdout.
    lines = [
        L for L in result.stdout.splitlines()
        if L.strip().startswith("test_smoke")
    ]
    assert lines, (
        f"No bench-output line starting with 'test_smoke' found. "
        f"stdout:\n{result.stdout[-1500:]}"
    )
    assert len(lines) == 1, (
        f"Expected exactly 1 bench line, got {len(lines)}:\n"
        f"{lines!r}"
    )
    m = _LINE_RE.match(lines[0].strip())
    assert m is not None, (
        f"Bench output line failed to match schema. Got:\n"
        f"{lines[0]!r}\nRegex:\n{_LINE_RE.pattern}"
    )
    # Schema sanity: lock the per-field values we requested.
    assert m.group("label") == "test_smoke"
    assert int(m.group("nranks")) == 1
    assert int(m.group("ny")) == 12
    assert int(m.group("nx")) == 12
    assert int(m.group("ny_local")) == 12
    assert int(m.group("nx_local")) == 12
    # Timing fields finite + positive (total + local must be > 0;
    # halo_comm CAN be tiny-negative at the noise floor for the
    # single-rank fast path because total - local subtraction
    # straddles zero. Just require it's finite).
    total_ms = float(m.group("total"))
    local_ms = float(m.group("local"))
    comm_ms = float(m.group("comm"))
    assert 0.0 < total_ms < 5000.0, (
        f"total_ms={total_ms} outside (0, 5000) range; bench "
        f"either crashed or hit pathological JIT path."
    )
    assert 0.0 < local_ms < 5000.0, (
        f"local_ms={local_ms} outside (0, 5000) range."
    )
    # iter-256 (Codex iter-252..255 round-1 HIGH#1): no-op detector.
    # Real 8-operator pipeline on 12x12x8 takes ~0.04 ms local
    # compute per step. An elided pipeline would land at <1 us
    # (kernel-launch overhead only) — orders of magnitude below.
    # Floor at 1 us catches a fully-elided kernel while tolerating
    # fast hardware.
    assert local_ms > 1.0e-3, (
        f"local_compute={local_ms} ms below 1us floor; the "
        f"8-operator pipeline likely no-op'd. Real measured "
        f"baseline is ~0.04 ms on this mesh."
    )
    # On a single rank halo_comm should be small (a few percent of
    # total at most). Cap at 4× total to catch a regression where
    # comm dominates due to a bug.
    assert abs(comm_ms) < 4.0 * total_ms, (
        f"halo_comm={comm_ms} suspiciously large vs "
        f"total={total_ms} at single-rank."
    )
