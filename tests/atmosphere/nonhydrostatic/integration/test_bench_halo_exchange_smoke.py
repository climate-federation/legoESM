"""Smoke test for ``scripts/bench_halo_exchange.py``.

iter-257: fourth script in the iter-252/254/255 untested-bench
coverage chain. ``bench_halo_exchange.py`` is the MPI halo-exchange
micro-benchmark across grid types (cubed-sphere + voronoi), with
JSON output. Pre iter-257 it had ZERO test coverage; iter-257 adds
a single-rank smoke covering --grid cubed-sphere + JSON output
schema (results.grid, n_ranks, exchanges.{3d,4d}.time_per_exchange_ms).
"""
from __future__ import annotations

import json

from tests.atmosphere.nonhydrostatic.integration._bench_smoke_helpers import (
    REPO_ROOT, fail_on_nonzero, run_bench,
)


SCRIPT = REPO_ROOT / "scripts" / "bench_halo_exchange.py"


def test_bench_halo_exchange_cubed_sphere_single_rank_smoke(tmp_path):
    """Cubed-sphere halo-exchange bench: single-rank smoke covering
    argparse + cubed_sphere grid construction + 3D/4D exchange JIT
    + JSON output schema lock.
    """
    out_file = tmp_path / "bench_halo_smoke.json"
    result = run_bench(SCRIPT, [
        "--grid", "cubed-sphere",
        "--n", "12",
        "--nlev", "8",
        "--n-warmup", "1",
        "--n-iters", "3",
        "--output", str(out_file),
    ])
    fail_on_nonzero(result, "bench_halo_exchange.py")
    assert out_file.exists(), (
        f"Bench script did not write JSON at {out_file}.\n"
        f"stdout tail:\n{result.stdout[-500:]}"
    )
    doc = json.loads(out_file.read_text())
    # Top-level schema.
    assert doc["grid"] == "cubed-sphere", (
        f"JSON grid={doc.get('grid')!r}, expected 'cubed-sphere'."
    )
    assert doc["n"] == 12
    assert doc["nlev"] == 8
    assert doc["n_ranks"] == 1
    assert doc["n_warmup"] == 1
    assert doc["n_iters"] == 3
    # Exchanges sub-dict has both 3d + 4d entries.
    exchanges = doc.get("exchanges", {})
    assert "3d" in exchanges and "4d" in exchanges, (
        f"exchanges missing 3d/4d keys: {list(exchanges.keys())!r}"
    )
    for kind in ("3d", "4d"):
        sub = exchanges[kind]
        for key in ("time_per_exchange_ms", "total_time_s",
                    "msg_bytes_per_edge", "n_edges"):
            assert key in sub, (
                f"exchanges[{kind!r}] missing field {key!r}: {sub!r}"
            )
        t_ms = float(sub["time_per_exchange_ms"])
        total_s = float(sub["total_time_s"])
        msg_bytes = int(sub["msg_bytes_per_edge"])
        n_edges = int(sub["n_edges"])
        # Timing sanity: positive, sub-second.
        assert 0.0 < t_ms < 60_000.0, (
            f"{kind} time_per_exchange_ms={t_ms} outside "
            f"(0, 60s) range."
        )
        assert 0.0 < total_s < 60.0, (
            f"{kind} total_time_s={total_s} outside (0, 60s) range."
        )
        # iter-257 no-op detector (mirrors iter-256 HIGH#1):
        # Real halo exchange on 12x12 cubed-sphere face takes
        # ~1 ms per exchange; an elided exchange would be <1 us.
        # Floor at 1 us catches a full elision.
        assert t_ms > 1.0e-3, (
            f"{kind} time_per_exchange_ms={t_ms} below 1 us "
            f"floor — exchange likely no-op'd."
        )
        # msg_bytes_per_edge MUST be positive (a 0-byte msg means
        # the message-size accounting regressed).
        assert msg_bytes > 0, (
            f"{kind} msg_bytes_per_edge={msg_bytes} not positive."
        )
        assert n_edges > 0, (
            f"{kind} n_edges={n_edges} not positive."
        )
    # Stdout summary table includes the grid label.
    assert "cubed-sphere" in result.stdout, (
        f"Driver stdout missing 'cubed-sphere' marker. "
        f"stdout: {result.stdout[-500:]}"
    )
