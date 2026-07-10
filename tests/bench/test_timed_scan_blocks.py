"""timed_scan_blocks + git_sha: the measurement-contract helper (M1 gap #1/#2).

Runs on CPU with a trivial jitted step — validates the schedule accounting
(compile/probe/blocks executed counts), the separate latency-vs-fused
reporting, single-process imbalance == 1.0, and the fail-open git SHA.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))
from metadata import git_sha, scaling_metadata, timed_scan_blocks  # noqa: E402


def _advance(s):
    return jax.tree_util.tree_map(lambda x: x * 0.999 + 0.001, s)


def test_schedule_executes_exact_step_count():
    s0 = {"a": jnp.ones((4, 4)), "b": jnp.zeros((3,))}
    # count steps via the value: each step is affine, invertible bookkeeping —
    # use a counter leaf instead (exact).
    s0["n"] = jnp.zeros(())
    def adv(s):
        out = dict(s)
        out["n"] = s["n"] + 1.0
        out["a"] = s["a"] * 0.999
        return out
    s, m = timed_scan_blocks(adv, s0, block_steps=5, n_blocks=2, probe_steps=3)
    # 1 compile + 3 probe + 2*5 block = 14 steps
    assert float(s["n"]) == 14.0
    assert m["block_steps"] == 5 and m["n_blocks"] == 2 and m["probe_steps"] == 3
    assert len(m["block_ms"]) == 2


def test_metrics_shape_and_separation():
    s0 = {"x": jnp.ones((8,))}
    s, m = timed_scan_blocks(_advance, s0, block_steps=4, n_blocks=1,
                             probe_steps=2)
    for k in ("compile_ms", "scan_compile_ms", "step_latency_ms",
              "fused_step_ms", "block_ms_max", "block_ms_median",
              "rank_imbalance"):
        assert k in m, k
    # single process: max == median, imbalance exactly 1.0
    assert m["block_ms_max"] == m["block_ms_median"]
    assert m["rank_imbalance"] == 1.0
    # fused number derives from the block, not the probe (both fields are
    # independently rounded — fused to 4 dp, block to 2 dp — so compare at
    # the rounding granularity, not machine precision)
    assert abs(m["fused_step_ms"] - m["block_ms_max"] / 4) < 5e-3


def test_zero_block_steps_no_div0():
    s0 = {"x": jnp.ones(())}
    _, m = timed_scan_blocks(_advance, s0, block_steps=0, n_blocks=1,
                             probe_steps=0)
    assert m["fused_step_ms"] >= 0.0  # finite, no ZeroDivisionError


def test_git_sha_fail_open_and_in_metadata():
    sha = git_sha()
    assert isinstance(sha, str) and len(sha) >= 4  # short sha or "unknown"
    md = scaling_metadata(grid="latlon", component="ocean", resolution="8x16",
                          n_levels=4, precision="float64")
    assert md["git_sha"] == sha
