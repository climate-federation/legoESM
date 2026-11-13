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


@pytest.mark.parametrize("argv,schedule,checkpoints,storage", [
    (["--stage", "step"], "none", None, "recompute"),
    (["--stage", "step", "--schedule", "binomial", "--checkpoints", "12",
      "--storage", "host"], "binomial", 12, "host"),
])
def test_parser_round_trips_the_schedule_flags(argv, schedule, checkpoints,
                                               storage):
    a = pam.build_parser().parse_args(argv)
    assert (a.schedule, a.checkpoints, a.storage) == (schedule, checkpoints,
                                                      storage)


def test_parser_rejects_an_unknown_schedule():
    with pytest.raises(SystemExit):
        pam.build_parser().parse_args(["--stage", "step", "--schedule", "sqrt"])


def test_peak_gib_is_a_positive_finite_number():
    g = pam.peak_gib()
    assert g > 0.0 and g == g


def test_grad_stash_records_the_adjoint_entering_each_step():
    """Known map: each step multiplies leaf a by 3 (b by 0.5); with the
    loss weighting a by w, the cotangent on a entering step k is
    max(w) * 3**(n-k), its argmax is w's argmax, and the primal recorded
    there is a's value entering step k."""
    n = 5
    ex = ({"a": jnp.ones(3), "b": jnp.ones(3)}, {"c": jnp.ones(2)})
    stash, rec, names = pam.make_grad_stash(n, ex)
    assert names == ["[0]['a']", "[0]['b']", "[1]['c']"]
    w = jnp.asarray([1.0, 1.0, 2.0])             # the argmax cotangent is entry 2

    def step(s, p):
        return {"a": 3.0 * s["a"] + jnp.sin(s["b"]), "b": 0.5 * s["b"]}, p

    def loss(s, p):
        for k in range(n):
            s, p = stash(k, s, p)
            s, p = jax.checkpoint(step)(s, p)
        return jnp.sum(w * s["a"]) + jnp.sum(s["b"]) + jnp.sum(p["c"])
    g = jax.jit(jax.grad(loss))(*ex)
    rows = jnp.asarray(rec["log"][...])
    assert rows.shape == (n, 3)
    assert jnp.allclose(rows[:, 0], 2.0 * 3.0 ** jnp.arange(n, 0, -1))
    assert float(g["a"][2]) == 2.0 * 3.0 ** n
    assert bool(jnp.all(rows[:, 2] == 1.0))      # p is carried untouched
    assert bool(jnp.all(jnp.asarray(rec["loc"][...])[:, 0] == 2))
    # primal of a[2] entering step k: a_k = 3 a_{k-1} + sin(b_{k-1}), a_0 = 1
    a_k, b_k, prim = 1.0, 1.0, []
    for _ in range(n):
        prim.append(a_k)
        a_k, b_k = 3.0 * a_k + float(jnp.sin(b_k)), 0.5 * b_k
    assert jnp.allclose(jnp.asarray(rec["val"][...])[:, 0], jnp.asarray(prim))
    assert rec["shapes"] == [(3,), (3,), (2,)]


def test_parser_round_trips_stash_grads():
    a = pam.build_parser().parse_args(["--stage", "step", "--stash-grads"])
    assert a.stash_grads is True
    assert pam.build_parser().parse_args(["--stage", "step"]).stash_grads is False


def test_parser_round_trips_at_step_variant_and_x_tracers():
    a = pam.build_parser().parse_args(
        ["--stage", "components", "--at-step", "16", "--physics-variant",
         "norad", "--x-tracers", "q_v,q_i"])
    assert (a.at_step, a.physics_variant, a.x_tracers) == (16, "norad", "q_v,q_i")
    d = pam.build_parser().parse_args(["--stage", "step"])
    assert (d.at_step, d.physics_variant, d.x_tracers) == (0, "full", "q_v")


def test_state_helpers_follow_the_tracer_list(monkeypatch):
    import collections

    from legoesm.core.field import Field
    monkeypatch.setattr(pam, "_X_TRACERS", ("q_v", "q_i"))
    S = collections.namedtuple("S", "T tracers")
    st = S(T=Field(data=jnp.ones((3, 2)), name="T", units="K"),
           tracers={"q_v": Field(data=jnp.full((3, 2), 0.01), name="q_v", units="kg/kg"),
                    "q_i": Field(data=jnp.full((3, 2), 1e-6), name="q_i", units="kg/kg")})
    x = pam._state_x(st)
    assert set(x) == {"T", "q_v", "q_i"}
    v = pam._direction(x)
    assert set(v) == set(x) and float(jnp.max(jnp.abs(v["q_i"]))) < 1e-7
    st2 = pam._with_x(st, {**x, "q_i": x["q_i"] * 3.0})
    assert float(st2.tracers["q_i"].data[0, 0]) == pytest.approx(3e-6)


def test_call_stash_records_each_call_in_and_out():
    """Two chained calls of a module that triples x: the cotangent at each
    tap is the downstream Jacobian product (9, 3, 3, 1 for in0, out0, in1,
    out1 of loss = sum(w*out1)), rows follow trace order, reset() renumbers,
    an integer leaf records NaN, and the tap is the identity in value."""
    tap, rec, reset = pam.make_call_stash(max_calls=8, max_leaves=4)
    w = jnp.asarray([1.0, 2.0])

    def mod(x):
        return 3.0 * x

    def f(x):
        reset()
        y = x
        for _ in range(2):
            y, _k = tap("m:in", (y, jnp.int32(1)))
            y = tap("m:out", mod(y))
        return jnp.sum(w * y)
    x = jnp.asarray([1.0, 1.0])
    assert float(jax.jit(f)(x)) == pytest.approx(27.0)
    g = jax.jit(jax.grad(f))(x)
    assert jnp.allclose(g, 9.0 * w)
    rows = jnp.asarray(rec["log"][...])
    assert [rec["labels"][i] for i in range(4)] == ["m:in", "m:out"] * 2
    assert jnp.allclose(rows[:4, 0], jnp.asarray([18.0, 6.0, 6.0, 2.0]))
    assert bool(jnp.isnan(rows[0, 1]))            # the int leaf
    assert bool(jnp.all(jnp.asarray(rec["loc"][...])[:4, 0] == 1))
    assert jnp.allclose(jnp.asarray(rec["val"][...])[:4, 0],
                        jnp.asarray([1.0, 3.0, 3.0, 9.0]))
    assert rec["i"] == 4


def test_parser_round_trips_stash_modules():
    assert pam.build_parser().parse_args(
        ["--stage", "physics", "--stash-modules"]).stash_modules is True


def test_state_save_load_round_trip_is_bitwise(tmp_path):
    tree = ({"T": jnp.linspace(200.0, 300.0, 6).reshape(3, 2),
             "n": jnp.arange(3, dtype=jnp.int32)}, (jnp.asarray(1e-300),))
    f = tmp_path / "s.npz"
    pam.save_state(f, tree, {"at_step": 16})
    like = jax.tree_util.tree_map(jnp.zeros_like, tree)
    got, meta = pam.load_state(f, like)
    assert meta == {"at_step": 16}
    assert pam.state_digest(got) == pam.state_digest(tree)
    assert pam.state_digest(like) != pam.state_digest(tree)
    bad = ({"T": jnp.zeros((2, 3)), "n": tree[0]["n"]}, tree[1])
    with pytest.raises(SystemExit, match="leaf"):
        pam.load_state(f, bad)


def test_eqn_trace_reproduces_grad_and_names_the_floor_division():
    """Known function with a floored division (the suspected microphysics
    pattern) inside a jitted helper (must be recursed into, reported with its
    own source line), a scan and a cond (stay opaque): the
    per-equation reverse pass must reproduce jax.grad exactly, and the
    largest gain must be the division by the 1e-20 floor, with the source
    line of that division."""
    @jax.jit
    def ratio(n, q):
        return n / jnp.maximum(q, 1e-20)          # LINE_FLOOR

    def f(x):
        r = ratio(x["n"], x["q"])
        y, _ = jax.lax.scan(lambda c, z: (c + jnp.tanh(z), None),
                            jnp.zeros(()), r * 1e-25)
        y = jax.lax.cond(y > -1.0, lambda u: u * 2.0, lambda u: u, y)
        return y + jnp.sum(jnp.sin(x["q"]))
    x = {"n": jnp.asarray([1e-30, 2.0, 0.0]), "q": jnp.asarray([0.0, 1.0, 3.0])}
    g_eq, rows = pam.eqn_trace(f, x)
    g = jax.grad(f)(x)
    for k in x:
        assert jnp.allclose(g_eq[k], g[k], rtol=1e-12, atol=0.0), k
    top = max(rows, key=lambda r: r["gain"])
    assert top["prim"] == "div"
    assert top["gain"] >= 1e19
    assert top["src"].split(":")[0].endswith("test_production_ad_map.py")
    assert {"scan", "cond"} <= {r["prim"] for r in rows}
    assert "pjit" not in {r["prim"] for r in rows} and "jit" not in {r["prim"] for r in rows}


def test_time_compiled_separates_compile_from_steady_state_runs():
    calls = []

    def f(x):
        calls.append(1)            # runs at trace time only
        return jnp.sum(jnp.sin(x) ** 2)
    t_c, t_1, reps, out = pam.time_compiled(f, (jnp.ones(1000),), 4)
    assert len(calls) == 1          # traced once, reused for every call
    assert len(reps) == 4 and all(r >= 0.0 for r in reps)
    assert t_c > 0.0 and t_1 >= 0.0
    assert float(out) == pytest.approx(1000 * float(jnp.sin(1.0) ** 2))


def test_parser_round_trips_timing_flags():
    a = pam.build_parser().parse_args(
        ["--stage", "timing", "--timing-steps", "32,128", "--repeats", "2"])
    assert (a.timing_steps, a.repeats) == ("32,128", 2)


def test_grad_stash_rows_are_window_relative_for_a_nonzero_step0(capsys):
    """A window starting at absolute step 16: the tap receives absolute steps,
    rows are filled from 0, nothing is lost or written out of bounds, and the
    printer handles the full table (and an empty row) without crashing."""
    n, step0 = 3, 16
    ex = ({"a": jnp.ones(2)}, {"c": jnp.ones(1)})
    stash, rec, names = pam.make_grad_stash(n, ex, step0=step0)

    def loss(s, p):
        for k in range(n):
            s, p = stash(step0 + k, s, p)
            s = {"a": 2.0 * s["a"]}
        return jnp.sum(s["a"])
    jax.grad(loss)(*ex)
    rows = jnp.asarray(rec["log"][...])
    assert jnp.allclose(rows[:, 0], jnp.asarray([8.0, 4.0, 2.0]))
    pam.print_grad_stash(rec, names, "w")
    rec2 = dict(rec, log=jax.new_ref(jnp.full((n, len(names)), jnp.nan)))
    pam.print_grad_stash(rec2, names, "empty")
    out = capsys.readouterr().out
    assert "ADJOINT w step=0 max=8.000e+00" in out
    assert "ADJOINT empty step=0 no cotangent recorded" in out


def test_code_provenance_names_the_imported_file():
    line = pam.code_provenance(("json",))
    assert line.startswith("json=") and "json" in line and "@" in line
    assert "UNIMPORTABLE" in pam.code_provenance(("no_such_module_xyz",))
