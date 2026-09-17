"""Direct tests for write_tracer_oracle_ic.py -- the inverse face map."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "write_tracer_oracle_ic",
    _ROOT / "scripts" / "validate" / "fv3_native" / "write_tracer_oracle_ic.py")
writer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(writer)
parity = writer.parity


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("nm", sorted(parity.DIHEDRAL))
def test_to_oracle_layout_inverts_the_harness_read_path(transposed, nm):
    """Writing a port field into the oracle's (k, j, i) and reading it
    back the way the harness does must give the dihedral-mapped port
    field: oracle_ij(to_oracle_layout(P)) == DIHEDRAL[nm](P). A
    non-square window (n_i != n_j would be a bug elsewhere, but distinct
    k) catches a swapped axis that a cube would hide."""
    rng = np.random.default_rng(0)
    port = rng.standard_normal((5, 5, 3))
    meta = (transposed, nm, 1.0, 1.0)
    stored = writer.to_oracle_layout(port, meta)
    assert stored.shape == (3, 5, 5)
    back = parity.oracle_ij(stored, transposed)
    np.testing.assert_array_equal(back, parity.DIHEDRAL[nm](port))


def test_to_oracle_layout_moves_the_right_axis():
    """(i, j, k) -> (k, j, i) with the identity map: element [i, j, k]
    lands at [k, j, i]."""
    port = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    stored = writer.to_oracle_layout(port, (False, "id", 1.0, 1.0))
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert stored[k, j, i] == port[i, j, k]


def test_iq_zero_is_refused():
    with pytest.raises(SystemExit, match="iq must be >= 1"):
        writer.main(["--zerostep-run", "x", "--out-dir", "y", "--iq", "0"])


def test_compare_oracle_restarts_flags_drift_and_unmoved():
    """compare(): bitwise reads as 0/0/unmoved; a one-cell change reads as
    moved with the right max abs; rel uses the peak scale."""
    spec = importlib.util.spec_from_file_location(
        "compare_oracle_restarts",
        _ROOT / "scripts" / "validate" / "fv3_native" / "compare_oracle_restarts.py")
    cmp_ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp_)
    a = [{"x": np.full((2, 3, 3), 2.0)} for _ in range(6)]
    b = [{"x": t["x"].copy()} for t in a]
    same = cmp_.compare(a, b, ["x"])
    assert same["x"] == (0.0, 0.0, False)
    b[4]["x"][1, 2, 0] += 0.5
    moved = cmp_.compare(a, b, ["x"])
    assert moved["x"][0] == 0.5 and moved["x"][2] is True
    assert abs(moved["x"][1] - 0.5 / 2.5) < 1e-15
