"""Smoke test for ``scripts/bench_dd_scaling.py``.

iter-255: third script in the iter-252 / iter-254 untested-bench
chain. ``bench_dd_scaling.py`` is the original domain-decomposed
scaling bench (per-step wall via step_halo) that
``bench_plane_crm_dd_scaling.py`` later extended with strong/weak
mode separation + CSV output. Pre iter-255 it had ZERO test
coverage; iter-255 adds a single-rank single-process smoke
covering argparse + DD-layout + step_halo timing + stdout schema.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "bench_dd_scaling.py"


# iter-256 (Codex iter-252..255 round-1 HIGH#2): end-anchored.
_LINE_RE = re.compile(
    r"(?P<label>\S+)\s+nranks=\s*(?P<nranks>\d+)\s+"
    r"grid=(?P<ny>\d+)x(?P<nx>\d+)\s+"
    r"ny_local=(?P<ny_local>\d+)\s+"
    r"nx_local=(?P<nx_local>\d+)\s+"
    r"per_step=\s*(?P<per_step>[\d.]+)ms\s*$"
)


def test_bench_dd_scaling_single_rank_smoke():
    """Single-rank smoke catches argparse, factory dispatch, layout
    construction, JIT-compile, and stdout-schema regressions in
    O(few-second) cold JIT wall.
    """
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(SCRIPT),
        "--nx", "12", "--ny", "12", "--nlev", "8",
        "--dt", "1.0",
        "--n-warmup", "1",
        "--n-bench", "3",
        "--label", "test_smoke",
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        pytest.fail(
            f"bench_dd_scaling.py exited {result.returncode}\n"
            f"stdout tail:\n{result.stdout[-1500:]}\n"
            f"stderr tail:\n{result.stderr[-1500:]}"
        )
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
        f"Bench output line failed schema. Got:\n{lines[0]!r}\n"
        f"Regex:\n{_LINE_RE.pattern}"
    )
    assert m.group("label") == "test_smoke"
    assert int(m.group("nranks")) == 1
    assert int(m.group("ny")) == 12
    assert int(m.group("nx")) == 12
    assert int(m.group("ny_local")) == 12
    assert int(m.group("nx_local")) == 12
    per_step_ms = float(m.group("per_step"))
    # step_halo on 12x12x8 should land somewhere between sub-ms
    # (best case JIT'd) and a few seconds (worst case cold JIT
    # before bench window). Cap at 60 s — anything beyond is
    # pathological.
    assert 0.0 < per_step_ms < 60_000.0, (
        f"per_step={per_step_ms} ms outside (0, 60s) range."
    )
    # iter-256 (Codex iter-252..255 round-1 HIGH#1): no-op detector.
    # Real slow_tendency_jit_split on 12x12x8 measures ~6 ms.
    # An elided pipeline would land at microseconds. Floor at
    # 0.05 ms catches a fully-elided kernel.
    assert per_step_ms > 0.05, (
        f"per_step={per_step_ms} ms below 50us floor; the "
        f"slow-tendency pipeline likely no-op'd. Real measured "
        f"baseline is ~6 ms on this mesh."
    )
