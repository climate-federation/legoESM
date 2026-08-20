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
