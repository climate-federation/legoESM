"""Direct tests of the round-316 measurement instruments (synthetic cases)."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round316_branch_flips as flips,
    nemo_testcase_l4_orca2_round316_forward_bits as bits,
    nemo_testcase_l4_orca2_round316_fused_grad as fused,
)


def test_forward_bits_compare_flags_one_moved_leaf(tmp_path):
    state = {"T": jnp.arange(4.0), "eta": jnp.asarray([0.0, -0.0])}
    rec = {"case": "gyre", "jaxpr_sha256": "x", "kt": {"1": bits.leaf_hashes(state)}}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(rec))
    moved = dict(rec, kt={"1": bits.leaf_hashes({**state, "eta": jnp.asarray([0.0, 0.0])})})
    b.write_text(json.dumps(moved))
    assert bits.compare(a, a)["verdict"] == "BIT-IDENTICAL"
    assert bits.compare(a, b)["moved"] == ["kt1:['eta']"]  # a signed zero is a move


def _count(f, x, y):
    closed = jax.make_jaxpr(f)(x)
    w = flips.Lockstep()
    got = w.run(closed.jaxpr, list(closed.consts), [(x, y)])
    np.testing.assert_array_equal(np.asarray(got[0][0]), np.asarray(f(x)))
    np.testing.assert_array_equal(np.asarray(got[0][1]), np.asarray(f(y)))
    return w.flipped


def test_lockstep_counts_known_flips_through_jit_scan_cond_and_while():
    x = jnp.asarray([-1.0, -0.1, 0.2, 3.0])
    y = jnp.asarray([-1.0, 0.1, -0.2, 3.0])           # lanes 1 and 2 cross zero
    assert _count(lambda v: jnp.maximum(v, 0.0), x, y) == 2
    assert _count(lambda v: jnp.maximum(v, 0.0), x, x) == 0

    def scanned(v):
        return jax.lax.scan(lambda c, _: (jnp.abs(c) - 0.05, None), v, None, length=3)[0]
    assert _count(jax.jit(scanned), x, x) == 0
    assert _count(jax.jit(scanned), x, y) > 0

    def branchy(v):
        v = jax.lax.cond(v[1] > 0, lambda u: u * 2.0, lambda u: u, v)
        return jax.lax.while_loop(lambda u: u[0] < 0.0, lambda u: u + 0.6, v)
    assert _count(branchy, x, x) == 0
    assert _count(branchy, x, y) >= 2                  # the gt lane + the cond index


def test_fused_grad_parser_requires_a_rung(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(fused, "fused", lambda *a: calls.append(a) or {"ok": 1})
    out = tmp_path / "o.json"
    for argv in (["--output", str(out)], ["--rung", "nope", "--output", str(out)]):
        monkeypatch.setattr("sys.argv", ["fused", *argv])
        with pytest.raises(SystemExit) as exc:
            fused.main()
        assert exc.value.code == 2
    assert calls == []
    monkeypatch.setattr("sys.argv", ["fused", "--rung", "rung0", "--output", str(out)])
    assert fused.main() == 0
    assert calls == [("rung0", 10, ("T", "ssh"))]
    assert json.loads(out.read_text()) == {"ok": 1}
