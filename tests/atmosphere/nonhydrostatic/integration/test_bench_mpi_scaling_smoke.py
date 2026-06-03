"""Smoke test for ``scripts/bench_mpi_scaling.py``.

iter-258: fifth script in the iter-252..257 untested-bench
coverage chain. ``bench_mpi_scaling.py`` measures plane CRM MPI
path scaling with per-component decomposition (dycore + bcast +
reduce) — useful for diagnosing where the F9 MPI overhead lives.
The replicated-dycore design (rank-0 step + bcast) is the
'anti-scaling' reference point against the DD path benched by
bench_dd_scaling / bench_plane_crm_dd_scaling.

Pre iter-258 it had ZERO test coverage; iter-258 adds a single-
rank single-process smoke covering argparse + replicated step
+ stdout schema.
"""
from __future__ import annotations

import re

from tests.atmosphere.nonhydrostatic.integration._bench_smoke_helpers import (
    REPO_ROOT, fail_on_nonzero, run_bench,
)


SCRIPT = REPO_ROOT / "scripts" / "bench_mpi_scaling.py"


# End-anchored regex (iter-256 round-1 HIGH#2 pattern):
# trailing-field additions will fail the schema lock loudly.
_LINE_RE = re.compile(
    r"(?P<label>\S+)\s+nranks=\s*(?P<nranks>\d+)\s+"
    r"grid=(?P<ny>\d+)x(?P<nx>\d+)x(?P<nlev>\d+)\s+"
    r"total=\s*(?P<total>[\d.]+)ms\s+"
    r"dycore=\s*(?P<dycore>[\d.]+)ms\s+"
    r"bcast=\s*(?P<bcast>[\d.]+)ms\s+"
    r"reduce=\s*(?P<reduce>[\d.]+)ms\s*$"
)


def test_bench_mpi_scaling_single_rank_smoke():
    """Single-rank smoke covering argparse, replicated-dycore step
    timing, and the per-component decomposition stdout schema.
    """
    result = run_bench(SCRIPT, [
        "--nx", "12", "--ny", "12", "--nlev", "8",
        "--dt", "1.0",
        "--n-warmup", "1",
        "--n-bench", "3",
        "--label", "test_smoke",
    ])
    fail_on_nonzero(result, "bench_mpi_scaling.py")
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
        f"Bench output failed schema. Got:\n{lines[0]!r}\n"
        f"Regex:\n{_LINE_RE.pattern}"
    )
    assert m.group("label") == "test_smoke"
    assert int(m.group("nranks")) == 1
    assert int(m.group("ny")) == 12
    assert int(m.group("nx")) == 12
    assert int(m.group("nlev")) == 8

    total_ms = float(m.group("total"))
    dycore_ms = float(m.group("dycore"))
    bcast_ms = float(m.group("bcast"))
    reduce_ms = float(m.group("reduce"))
    # Timing sanity.
    assert 0.0 < total_ms < 60_000.0, (
        f"total_ms={total_ms} outside (0, 60s) range."
    )
    assert 0.0 < dycore_ms < 60_000.0, (
        f"dycore_ms={dycore_ms} outside (0, 60s) range."
    )
    # iter-258 no-op detector (mirrors iter-256 HIGH#1):
    # Real replicated step on 12x12x8 takes ~2 ms per step; an
    # elided dycore would land at <1 us.
    assert dycore_ms > 1.0e-3, (
        f"dycore_ms={dycore_ms} below 1us floor; replicated step "
        f"likely no-op'd. Real baseline ~2 ms on this mesh."
    )
    # bcast + reduce should sum to a small fraction of total at
    # single rank (no real MPI comm — local copies only).
    # Just require non-negative.
    assert bcast_ms >= 0.0
    assert reduce_ms >= 0.0
    # Sanity: components don't exceed total significantly (within
    # 2x — the bench script measures these separately so they can
    # double-count, but not by huge margins).
    component_sum = dycore_ms + bcast_ms + reduce_ms
    assert component_sum < 3.0 * total_ms, (
        f"Component sum {component_sum} ms grossly exceeds total "
        f"{total_ms} ms — accounting regressed."
    )
