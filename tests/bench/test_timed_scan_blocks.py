"""timed_scan_blocks + git_sha: the measurement-contract helper (M1 gap #1/#2).

Runs on CPU with a trivial jitted step — validates the schedule accounting
(compile/probe/blocks executed counts, invalid schedules raise instead of
being silently clamped), the separate latency-vs-fused reporting, the
per-block parallel statistics (single-process imbalance == 1.0), the
zero-length parity path's null headline, and the fail-open git SHA.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

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
              "fused_step_ms", "parallel_block_ms", "rank_imbalance",
              "rank_imbalance_per_block"):
        assert k in m, k
    # single process: the parallel block time IS this process's block time,
    # and every per-block max/median ratio is exactly 1.0
    assert m["parallel_block_ms"] == m["block_ms"]
    assert m["rank_imbalance"] == 1.0
    assert m["rank_imbalance_per_block"] == [1.0]
    # fused number derives from the block, not the probe (both fields are
    # independently rounded — fused to 4 dp, block to 2 dp — so compare at
    # the rounding granularity, not machine precision)
    assert abs(m["fused_step_ms"] - m["parallel_block_ms"][0] / 4) < 5e-3


def test_zero_block_steps_parity_path_null_headline():
    """block_steps=0 is the documented zero-length parity path: the blocks
    run (fences/collectives matched) but a zero-step block has NO per-step
    time — the headline must be an honest null, not block_ms/1."""
    s0 = {"x": jnp.ones(())}
    _, m = timed_scan_blocks(_advance, s0, block_steps=0, n_blocks=1,
                             probe_steps=0)
    assert m["fused_step_ms"] is None
    assert m["block_steps"] == 0 and m["n_blocks"] == 1
    assert len(m["block_ms"]) == 1


def test_invalid_schedule_raises_never_clamped():
    """Silent max(1, n_blocks)/max(0, probe_steps) rewrites executed a
    schedule DIFFERENT from the recorded one (codex batch4): invalid values
    must raise so the record always equals the execution."""
    s0 = {"x": jnp.ones(())}
    with pytest.raises(ValueError, match="n_blocks"):
        timed_scan_blocks(_advance, s0, block_steps=2, n_blocks=0,
                          probe_steps=1)
    with pytest.raises(ValueError, match="n_blocks"):
        timed_scan_blocks(_advance, s0, block_steps=2, n_blocks=-2,
                          probe_steps=1)
    with pytest.raises(ValueError, match="probe_steps"):
        timed_scan_blocks(_advance, s0, block_steps=2, n_blocks=1,
                          probe_steps=-1)
    with pytest.raises(ValueError, match="block_steps"):
        timed_scan_blocks(_advance, s0, block_steps=-1, n_blocks=1,
                          probe_steps=0)


def test_seed_state_not_mutated_by_precompile():
    """The scan pre-compile runs on a shallow ALIAS of the seed leaves
    (documented contract: advance must not donate); the seed values used by
    the timed blocks must be byte-identical to freshly advanced ones."""
    import numpy as np
    s0 = {"x": jnp.arange(6.0)}
    ref = np.asarray(s0["x"]).copy()
    s, m = timed_scan_blocks(_advance, dict(s0), block_steps=1, n_blocks=1,
                             probe_steps=0)
    # 1 compile + 0 probe + 1x1 block = 2 steps from the ORIGINAL seed.
    want = ref
    for _ in range(2):
        want = want * 0.999 + 0.001
    # atol at ULP scale: a pre-compile that ADVANCED the live seed (an
    # extra step) would be off by ~1e-3, orders beyond this tolerance.
    assert np.allclose(np.asarray(s["x"]), want, rtol=0, atol=1e-12)


def test_git_sha_fail_open_and_in_metadata():
    sha = git_sha()
    assert isinstance(sha, str) and len(sha) >= 4  # short sha or "unknown"
    md = scaling_metadata(grid="latlon", component="ocean", resolution="8x16",
                          n_levels=4, precision="float64")
    assert md["git_sha"] == sha
