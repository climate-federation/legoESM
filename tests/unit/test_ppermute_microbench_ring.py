"""The comm microbenchmark's ring pattern must be a real permutation.

The stride knob exists so the benchmark can move a device's exchange partner
from inside its own node to the far side of the allocation without moving any
process. That only measures what it claims to if every stride still sends each
device's data exactly once and delivers it exactly once -- a pattern that
dropped or duplicated a device would quietly time a smaller exchange and read
as a fabric result.
"""
import importlib.util
import pathlib

import pytest

_PATH = (pathlib.Path(__file__).resolve().parents[2]
         / "scripts" / "bench" / "bench_ppermute_microbench.py")
_spec = importlib.util.spec_from_file_location("_ppermute_microbench", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


@pytest.mark.parametrize("n", [4, 8, 16, 64])
@pytest.mark.parametrize("stride", [1, 3, 4, 5, 8])
def test_ring_is_a_permutation(n, stride):
    if stride % n == 0:
        pytest.skip("degenerate stride is rejected, covered separately")
    perm = _mod._ring(n, stride)
    assert len(perm) == n
    assert sorted(src for src, _ in perm) == list(range(n))
    assert sorted(dst for _, dst in perm) == list(range(n))
    assert all(dst == (src + stride) % n for src, dst in perm)


def test_stride_changes_who_talks_to_whom():
    """A stride that did not move the partner would make the sweep vacuous."""
    assert _mod._ring(16, 1) != _mod._ring(16, 4)
    assert _mod._ring(16, 4) != _mod._ring(16, 8)


def test_self_send_is_rejected():
    with pytest.raises(ValueError, match="send to itself"):
        _mod._ring(16, 16)
    with pytest.raises(ValueError, match="send to itself"):
        _mod._ring(16, 32)


# --- the feature, not just the helper ------------------------------------
# The tests above would all still pass if the command-line option were deleted,
# or if it never reached the pattern that gets timed. These two run the
# benchmark the way a job script runs it.

_CPU_ENV = {
    "JAX_PLATFORMS": "cpu",
    "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
}


def _cpu_env():
    import os
    env = dict(os.environ)
    env.update(_CPU_ENV)
    return env


def test_requested_stride_reaches_the_receipt(tmp_path):
    """A stride that never leaves argparse would time the default pattern."""
    import json
    import subprocess
    import sys

    out = tmp_path / "r.jsonl"
    proc = subprocess.run(
        [sys.executable, str(_PATH), "--n-devices", "8", "--n-iters", "1",
         "--n-warmup", "1", "--n-reps", "4", "--ring-stride", "3",
         "--out", str(out)],
        env=_cpu_env(), capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    rec = json.loads(out.read_text().splitlines()[-1])
    assert rec["ring_stride"] == 3
    assert rec["ring_verified"] is True
    # Provenance: without these a 1-iteration debug row and a real measurement
    # row are indistinguishable once they are globbed together.
    assert rec["n_iters"] == 1 and rec["n_reps"] == 4


def test_the_delivery_check_can_fail(tmp_path):
    """A verifier that passes for a wrong pattern would prove nothing."""
    import subprocess
    import sys

    script = f'''
import importlib.util, numpy as np, jax
from jax.sharding import Mesh
spec = importlib.util.spec_from_file_location("m", {str(_PATH)!r})
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
mesh = Mesh(np.array(jax.devices()[:8]), (m.AXIS,))
assert m.verify_ring(mesh, 8, 3) == 0, "honest pattern must verify"
m._ring = lambda n, stride=1: [(i, (i + stride + 1) % n) for i in range(n)]
bad = m.verify_ring(mesh, 8, 3)
assert bad != 0, "a wrong permutation was accepted"
print("MISMATCH", bad)
'''
    proc = subprocess.run([sys.executable, "-c", script], env=_cpu_env(),
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "MISMATCH" in proc.stdout
