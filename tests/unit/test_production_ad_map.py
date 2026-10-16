"""Direct test of the production gradient-map probe's instruments
(scripts/validate/production_ad_map.py; the probe itself needs the AMIP
forcing files and runs under sbatch).

The instrument is checked on cases with KNOWN answers before its numbers are
quoted: a linear map (FD/AD exactly 1), a function with a severed gradient
(the check must report AD = 0 against a non-zero FD, not a pass), and the
module spies (record the first concrete call, keep the dispatcher markers).
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "validate"))
import production_ad_map as pam  # noqa: E402


def test_grad_check_on_a_known_linear_map():
    A = jnp.asarray([[1.0, 2.0], [3.0, -1.0]])
    x = {"T": jnp.asarray([0.5, -0.25]), "q_v": jnp.asarray([2.0, 1.0])}
    v = {"T": jnp.asarray([1.0, 0.0]), "q_v": jnp.asarray([0.0, 1.0])}
    rec = pam.grad_check(lambda z: pam._proj_loss(A @ z["T"] + z["q_v"]),
                         x, v, 1e-3, "linear")
    assert rec["finite"] and abs(rec["ratio"] - 1.0) < 1e-10


def test_grad_check_exposes_a_severed_gradient():
    x = {"a": jnp.asarray([1.0, 2.0])}
    v = {"a": jnp.asarray([1.0, 1.0])}
    rec = pam.grad_check(lambda z: jnp.sum(jax.lax.stop_gradient(z["a"]) ** 2)
                         + 0.0 * jnp.sum(z["a"]), x, v, 1e-3, "severed")
    assert rec["ad"] == 0.0 and abs(rec["fd"]) > 1.0


def test_spies_record_first_concrete_call_and_keep_markers(monkeypatch):
    import legoesm.atmosphere.physics.combined as comb

    def fake_factory(*a, **k):
        def fn(x, scale=1.0):
            return x * scale
        fn._wants_forcing = True
        return fn
    for name in pam._SPIED:
        monkeypatch.setattr(comb, name, fake_factory)
    rec = {}
    pam.install_spies(rec)
    fn = comb.make_convection_physics()
    assert getattr(fn, "_wants_forcing", False) is True
    jax.jit(fn)(jnp.ones(2))                 # traced call: not recorded
    assert "make_convection_physics" not in rec
    fn(jnp.ones(2), scale=2.0)
    fn(jnp.zeros(2))                         # only the FIRST concrete call
    _f, args, kw = rec["make_convection_physics"]
    assert float(args[0][0]) == 1.0 and kw == {"scale": 2.0}


def test_state_roundtrip_helpers():
    import collections

    from legoesm.core.field import Field
    T = Field(data=jnp.ones((3, 2)), name="T", units="K")
    q = Field(data=jnp.full((3, 2), 0.01), name="q_v", units="kg/kg")
    S = collections.namedtuple("S", "T tracers")
    st = S(T=T, tracers={"q_v": q})
    x = pam._state_x(st)
    x2 = {"T": x["T"] + 1.0, "q_v": x["q_v"] * 2.0}
    st2 = pam._with_x(st, x2)
    assert float(st2.T.data[0, 0]) == 2.0
    assert float(st2.tracers["q_v"].data[0, 0]) == pytest.approx(0.02)
