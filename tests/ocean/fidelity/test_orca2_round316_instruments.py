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


# --- round 317: the fused instrument's cost mode ----------------------------

from typing import NamedTuple  # noqa: E402


class _Field(NamedTuple):
    data: jax.Array

    def replace(self, data):
        return self._replace(data=data)


class _State(NamedTuple):
    T: _Field


def _toy_setup(n_steps=3):
    t0 = jnp.linspace(0.1, 1.2, 12).reshape(2, 2, 3)
    tmask = jnp.ones((2, 2, 3))
    forcing = [(jnp.float64(0.9 + 0.01 * k), jnp.float64(0.05 * k)) for k in range(n_steps)]

    def body(state, f):
        return _State(_Field(jnp.sin(state.T.data) * f[0] + f[1]))

    losses = {"T": lambda s: jnp.mean(s.T.data ** 2)}
    return None, None, _State(_Field(t0)), forcing, losses, tmask, body


def _analytic_grad(t0, forcing, n):
    x, dx = np.asarray(t0), np.ones_like(np.asarray(t0))
    for fw, sf in forcing[:n]:
        dx = dx * np.cos(x) * float(fw)
        x = np.sin(x) * float(fw) + float(sf)
    return 2.0 * x * dx / x.size


@pytest.mark.parametrize("remat", [True, False])
def test_make_run_gradient_is_exact_with_and_without_remat(remat):
    _, _, state0, forcing, losses, _, body = _toy_setup()
    for n in (1, 2, 3):
        run = fused.make_run(body, state0, forcing, n, remat)
        g = jax.jit(jax.grad(lambda t: losses["T"](run(t))))(state0.T.data)
        np.testing.assert_allclose(np.asarray(g), _analytic_grad(state0.T.data, forcing, n),
                                   rtol=1e-13, atol=0)


def test_cost_rows_match_analytic_and_forward_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(fused, "setup", lambda rung: _toy_setup())
    out = tmp_path / "cost.json"
    res = fused.cost("rung0", (1, 3), False, (3,), out)
    assert json.loads(out.read_text())["rows"].keys() == {"1", "3"}
    assert res["remat"] is False
    _, _, state0, forcing, _, tmask, _ = _toy_setup()
    d = np.asarray(fused.locate.probe_direction(state0, tmask))
    for n in (1, 3):
        row = res["rows"][str(n)]
        exact = _analytic_grad(state0.T.data, forcing, n)
        assert row["grad_finite"] and row["nonfinite_entries"] == 0
        np.testing.assert_allclose(row["grad_l2"], np.linalg.norm(exact), rtol=1e-13)
        np.testing.assert_allclose(row["rev_dot_dir0"], np.vdot(exact, d), rtol=1e-13)
        np.testing.assert_allclose(np.load(tmp_path / f"cost_grad_n{n}.npy"), exact, rtol=1e-13)
        assert row["run_peak_rss_gb"] >= row["rss_before_run_gb"] > 0
    assert "fwd_dir0" not in res["rows"]["1"]
    row3 = res["rows"]["3"]
    assert row3["jvp_primal_equals_reverse_primal"] is True
    assert row3["rev_vs_fwd_rel"] < 1e-13


def test_rss_sampler_sees_a_known_allocation():
    with fused.RssSampler(period_s=0.001) as quiet:
        pass
    with fused.RssSampler(period_s=0.001) as loud:
        block = np.ones(50_000_000)                     # 0.4 GB, touched
        block.sum()
        del block
    assert loud.peak - fused.rss_bytes() > 0.3e9
    assert quiet.peak - fused.rss_bytes() < 0.1e9


def test_fused_parser_routes_cost_mode(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(fused, "cost", lambda *a: calls.append(a))
    out = tmp_path / "o.json"
    monkeypatch.setattr("sys.argv", ["fused", "--rung", "rung10", "--cost", "1", "10",
                                     "--no-remat", "--jvp-at", "10", "--output", str(out)])
    assert fused.main() == 0
    monkeypatch.setattr("sys.argv", ["fused", "--rung", "omt2", "--cost", "2",
                                     "--output", str(out)])
    assert fused.main() == 0
    assert calls == [("rung10", (1, 10), False, (10,), out), ("omt2", (2,), True, (), out)]


def test_cost_remat_flag_changes_xla_buffer_estimate(monkeypatch, tmp_path):
    def body(state, f):                                 # several residuals per step
        x = state.T.data
        y = jnp.sin(x)
        z = jnp.exp(-y * y)
        return _State(_Field(jnp.tanh(z * x) * f[0] + jnp.cos(z) * f[1]))

    def setup(rung):
        losses = _toy_setup()[4]
        t0 = jnp.linspace(0.1, 1.2, 64 * 64 * 8).reshape(64, 64, 8)
        forcing = [(jnp.float64(0.9), jnp.float64(0.01))] * 12
        return None, None, _State(_Field(t0)), forcing, losses, jnp.ones(t0.shape), body

    monkeypatch.setattr(fused, "setup", setup)
    kept = fused.cost("rung0", (12,), False, (), tmp_path / "a.json")["rows"]["12"]
    remat = fused.cost("rung0", (12,), True, (), tmp_path / "b.json")["rows"]["12"]
    assert kept["xla_temp_gb"] > 2 * remat["xla_temp_gb"] > 0
    assert kept["grad_l2"] == pytest.approx(remat["grad_l2"], rel=1e-14)
