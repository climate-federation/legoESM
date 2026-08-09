"""Direct tests for scripts/bench/bench_halo_collectives.py.

XLA:CPU has no ragged-all-to-all thunk, so the ragged arm itself can
only run on GPU (the bench verifies it in-process there). What IS
CPU-testable, and what these tests pin down:

1. the metadata builder's invariants (the exact contract
   ``ragged_all_to_all`` requires) and its refusal guards;
2. the analytic oracle ``expected_staging`` against a hand-computed
   tiny case (the oracle must not be validated only by the code it is
   meant to check);
3. real cross-strategy parity of the two CPU-capable arms (ppermute,
   uniform) against that oracle on forced host devices — if both match
   on CPU, a GPU ragged mismatch indicts the ragged metadata, not the
   oracle.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_BENCH = _REPO / "scripts" / "bench" / "bench_halo_collectives.py"

_spec = importlib.util.spec_from_file_location("bench_halo_collectives",
                                               _BENCH)
bhc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bhc)


def test_metadata_invariants_ring8():
    n_dev, rows = 8, 3
    offsets = [1, -1, 3, -3]
    meta = bhc.build_offset_graph_metadata(n_dev, offsets, rows)
    ss, rs = meta["send_sizes"], meta["recv_sizes"]
    # the ragged_all_to_all contract
    assert np.array_equal(ss, rs.T)
    # degree rows per device, both directions
    assert (ss.sum(axis=1) == len(offsets) * rows).all()
    assert (rs.sum(axis=1) == len(offsets) * rows).all()
    # every input offset in bounds and block-aligned
    io = meta["input_offsets"]
    assert ((io[ss > 0] % rows) == 0).all()
    assert (io[ss > 0] < len(offsets) * rows).all()
    # output offsets land inside the receiver staging
    oo = meta["output_offsets"]
    assert (oo[ss > 0] < len(offsets) * rows).all()
    # receive order strictly ascending sources
    order = meta["recv_src_order"]
    assert (np.diff(order, axis=1) > 0).all()


def test_metadata_guards_raise():
    with pytest.raises(ValueError, match="duplicate"):
        bhc.build_offset_graph_metadata(8, [1, 1], 2)
    with pytest.raises(ValueError, match="self-send"):
        bhc.build_offset_graph_metadata(8, [8], 2)
    with pytest.raises(ValueError, match="alias"):
        bhc.build_offset_graph_metadata(8, [1, -7], 2)


def test_expected_staging_hand_case():
    """3 devices, offsets {+1,-1}, rows=1, width=1 — computed by hand.

    Payload x[d, j] = d*10 + j (j = offset block: 0 -> d+1, 1 -> d-1).
    Device 0 receives from sources {1, 2} (ascending): source 1's block
    aimed at 0 is its -1 block (j=1, value 11); source 2's block aimed
    at 0 is its +1 block (j=0, value 20).
    """
    n_dev, rows, width = 3, 1, 1
    offsets = [1, -1]
    meta = bhc.build_offset_graph_metadata(n_dev, offsets, rows)
    x = np.zeros((n_dev, 2, width), np.float32)
    for d in range(n_dev):
        x[d, 0] = d * 10 + 0
        x[d, 1] = d * 10 + 1
    exp = bhc.expected_staging(x, meta, rows)
    # d=0 <- {1: its -1 block (11), 2: its +1 block (20)}
    assert exp[0, 0] == 11 and exp[0, 1] == 20
    # d=1 <- {0: its +1 block (0), 2: its -1 block (21)}
    assert exp[1, 0] == 0 and exp[1, 1] == 21
    # d=2 <- {0: its -1 block (1), 1: its +1 block (10)}
    assert exp[2, 0] == 1 and exp[2, 1] == 10


def test_pattern_payload_rows_distinct():
    x = bhc.pattern_payload(4, 2, 3, 2)
    flat = x[:, :, 0].ravel()
    assert len(np.unique(flat)) == flat.size


@pytest.mark.timeout(600)
def test_cpu_arms_match_oracle(tmp_path):
    """ppermute + uniform arms on 4 forced host devices must pass the
    bench's own verify stage and emit timing rows. (ragged is GPU-only;
    its verify runs in the GPU job.)"""
    out = tmp_path / "halo.jsonl"
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    proc = subprocess.run(
        [
            sys.executable, str(_BENCH),
            "--n-devices", "4",
            "--offsets", "1,-1",
            "--rows", "3", "--width", "2",
            "--strategies", "ppermute,uniform",
            "--n-warmup", "1", "--n-iters", "3", "--n-reps", "4",
            "--out", str(out),
        ],
        env=env, capture_output=True, text=True, timeout=570,
    )
    assert proc.returncode == 0, (
        f"rc={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\n"
        f"stderr:\n{proc.stderr[-3000:]}")
    assert "verify ppermute: OK" in proc.stdout
    assert "verify uniform: OK" in proc.stdout
    recs = [json.loads(ln) for ln in out.read_text().strip().splitlines()]
    assert {r["strategy"] for r in recs} == {"ppermute", "uniform"}
    for r in recs:
        assert r["component"] == "halo_collectives_microbench"
        assert r["verified_exact"] is True
        assert r["n_devices"] == 4
        assert np.isfinite(r["per_fill_us"])


@pytest.mark.timeout(600)
def test_verify_failure_aborts(tmp_path):
    """Non-vacuous check of the verify gate: a deliberately corrupted
    strategy table must abort with VERIFY FAILED and write NO rows."""
    out = tmp_path / "halo.jsonl"
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    sabotage = (
        "import bench_halo_collectives as b\n"
        "_orig = b.expected_staging\n"
        "def bad(x, meta, rows):\n"
        "    e = _orig(x, meta, rows); e[0, 0] += 1.0; return e\n"
        "b.expected_staging = bad\n"
        "import sys; sys.argv = ['x', '--n-devices', '4', '--offsets',\n"
        "    '1,-1', '--rows', '3', '--width', '2', '--strategies',\n"
        "    'ppermute', '--n-warmup', '1', '--n-iters', '2',\n"
        f"    '--n-reps', '2', '--out', r'{out}']\n"
        "sys.exit(b.main())\n"
    )
    env["PYTHONPATH"] = str(_BENCH.parent) + os.pathsep + env.get(
        "PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", sabotage],
                          env=env, capture_output=True, text=True,
                          timeout=570)
    assert proc.returncode != 0
    assert "VERIFY FAILED" in (proc.stdout + proc.stderr)
    assert not out.exists()
