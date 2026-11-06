"""Gradient-access gate (``inert_params.gradient_access_gate``), one process.

Every failure class the gate exists for is planted in a tiny model and must
abort, naming the leaf; the healthy model must pass.  Cross-rank cases
(cancellation, single-rank failures) are in
``tests/distributed/test_gradient_access_gate_mpi.py``.
"""
import jax
import jax.numpy as jnp
import optax
import pytest

from legoesm.ml.training import TrainingConfig, create_optimizer, make_lr_schedule
from legoesm.training import data_parallel as dp
from legoesm.training.inert_params import GradientAccessError, gradient_access_gate

jax.config.update("jax_enable_x64", True)

TC = TrainingConfig(lr=1e-2, optimizer="adamw", warmup_steps=4, total_steps=40,
                    weight_decay=0.1, grad_clip_norm=1.0)
X = jnp.array([0.3, -0.7, 1.1])


def _params():
    return {"w": jnp.array([0.5, -0.2, 0.8]), "head": jnp.array([1.5, 0.4]),
            "emb": jnp.array([0.1, 0.2, 0.3])}


def _loss(p, x):
    h = jnp.tanh(p["w"] * x + p["emb"])
    return jnp.sum((p["head"][0] * h + p["head"][1] - 1.0) ** 2)


def _units(loss=_loss, n=10):
    vg = jax.value_and_grad(loss)
    return [(vg, X * (1.0 + 0.1 * i)) for i in range(n)]


def _gate(params=None, loss=_loss, units=None, opt=None, state=None, **kw):
    params = _params() if params is None else params
    opt = create_optimizer(TC) if opt is None else opt
    state = opt.init(params) if state is None else state
    return gradient_access_gate(
        _units(loss) if units is None else units, [params, state], opt,
        kw.pop("schedule", make_lr_schedule(TC)),
        optimizer_name=kw.pop("optimizer_name", "adamw"), **kw)


def _fails(match, **kw):
    with pytest.raises(GradientAccessError, match=match):
        _gate(**kw)


def test_healthy_model_passes_with_four_steps_on_a_fresh_run():
    rep = _gate()
    assert rep["k"] == 4 and rep["lrs"][0] == 0.0 and all(lr > 0 for lr in rep["lrs"][1:])
    assert all(r["checked"] == r["size"] for r in rep["rows"])


def test_resume_past_step_zero_needs_three_steps():
    opt, p = create_optimizer(TC), _params()
    s = opt.init(p)
    for x in (X, X):
        _, g = jax.value_and_grad(_loss)(p, x)
        u, s = opt.update(g, s, p)
        p = optax.apply_updates(p, u)
    assert _gate(params=p, opt=opt, state=s)["k"] == 3


def test_detached_head_aborts():
    def loss(p, x):
        return _loss({**p, "head": jax.lax.stop_gradient(p["head"])}, x)
    _fails(r"\['head'\].*no_grad=2", loss=loss)


def test_detached_embedding_aborts():
    def loss(p, x):
        return _loss({**p, "emb": jax.lax.stop_gradient(p["emb"])}, x)
    _fails(r"\['emb'\].*no_grad=3", loss=loss)


def test_weight_decay_only_leaf_aborts():
    # Unused leaf: AdamW decay moves it every step; g and du stay exactly 0.
    _fails(r"\['unused'\].*no_grad=2 no_update=2",
           params={**_params(), "unused": jnp.array([0.7, -0.3])})


def test_wrapped_optimizer_refused_even_under_the_adamw_name():
    # An update mask (here zeroing the head's update) changes the optimizer
    # state's structure; the gate refuses anything but the plain chain.
    opt = optax.chain(create_optimizer(TC), optax.masked(
        optax.set_to_zero(), {"w": False, "head": True, "emb": False}))
    _fails("optimizer state is not the clip_by_global_norm", opt=opt)


def test_non_adamw_optimizer_refused():
    _fails("optimizer 'adam' is not one", optimizer_name="adam")


def test_non_finite_loss_aborts():
    def loss(p, x):
        return _loss(p, x) * jnp.nan
    _fails("non-finite local loss", loss=loss)


def test_excluded_element_with_gradient_aborts():
    ex = {"w": jnp.array([False, False, True]), "head": jnp.zeros(2, bool),
          "emb": jnp.zeros(3, bool)}
    p = {**_params(), "w": jnp.array([0.5, -0.2, 0.0])}
    _fails(r"excluded_with_grad=1|excluded elements of \['w'\] moved off 0", params=p, excluded=ex)


def test_excluded_element_must_be_zero():
    ex = {"w": jnp.array([False, False, True]), "head": jnp.zeros(2, bool),
          "emb": jnp.zeros(3, bool)}
    _fails("excluded elements must be exactly 0", excluded=ex)


def test_excluded_dead_element_passes():
    def loss(p, x):   # w[2] has no path to the loss, and is registered
        return _loss({**p, "w": p["w"].at[2].set(0.0)}, x)
    ex = {"w": jnp.array([False, False, True]), "head": jnp.zeros(2, bool),
          "emb": jnp.zeros(3, bool)}
    p = {**_params(), "w": jnp.array([0.5, -0.2, 0.0])}
    rep = _gate(params=p, loss=loss, excluded=ex)
    assert [r["excluded"] for r in rep["rows"] if r["leaf"] == "['w']"] == [1]


def test_a_run_ending_early_rehearses_every_remaining_step():
    rep = _gate(units=_units(n=3))          # lr = [0, a, b], then the end
    assert rep["k"] == 3 and sum(lr > 0 for lr in rep["lrs"]) == 2


def test_no_updating_step_left_refuses():
    _fails("no lr > 0 step left in the run", units=_units(n=1))


def test_lr_zero_too_long_refuses():
    _fails("keeps lr at 0 too long", schedule=lambda c: 0.0)


def test_zero_init_chains_of_depth_two_pass_depth_three_aborts():
    def chain1(p, x):   # w2 zero-init: w1 dead until w2 moves
        return jnp.sum((p["w2"] * jnp.tanh(p["w1"] * x) - 1.0) ** 2)

    def chain2(p, x):   # NeuralGCM shape: zero embedding e AND zero last layer
        return jnp.sum((p["w2"] * jnp.tanh(p["w1"] * x + p["c"] * p["e"])
                        - 1.0) ** 2)

    def chain3(p, x):
        return jnp.sum((p["w3"] * p["w2"] * jnp.tanh(p["w1"] * x) - 1.0) ** 2)
    p1 = {"w1": jnp.ones(3), "w2": jnp.zeros(3)}
    assert _gate(params=p1, loss=chain1)["k"] == 4
    p2 = {**p1, "c": jnp.ones(3), "e": jnp.zeros(3)}
    assert _gate(params=p2, loss=chain2)["k"] == 4
    with pytest.raises(GradientAccessError, match=r"\['w1'\].*no_grad=3"):
        _gate(params={**p1, "w3": jnp.zeros(3)}, loss=chain3)


def test_update_below_precision_everywhere_is_apply_unverified():
    p = {**_params(), "big": jnp.array([1e15, -1e15])}

    def loss(q, x):
        return _loss(q, x) + 1e-3 * jnp.sum(q["big"]) * 1e-9
    _fails(r"\['big'\].*apply_unverified=1", params=p, loss=loss)


def test_update_lost_in_the_movement_probe_aborts(monkeypatch):
    # Per step the first tree-level apply is the movement probe (decay-free
    # update du).  An application that loses du must abort as not_moved, even
    # though the run's own update (second call) is applied correctly.
    real = dp.apply_param_updates
    n = [0]

    def lose_probe(p, u):
        n[0] += 1
        return p if n[0] % 2 == 1 else real(p, u)
    monkeypatch.setattr(dp, "apply_param_updates", lose_probe)
    _fails(r"not_moved=[1-9]")


def test_every_rehearsal_step_matches_the_loop_bitwise(monkeypatch):
    # Record the gate's parameters after each of its run-form updates: per
    # step the first tree-level apply is the du movement probe, the second
    # the run's own update.
    seen = []
    real = dp.apply_param_updates

    def rec(p, u):
        out = real(p, u)
        seen.append(out)
        return out
    monkeypatch.setattr(dp, "apply_param_updates", rec)
    opt, p0 = create_optimizer(TC), _params()
    s0 = opt.init(p0)
    units = _units()
    rep = _gate(params=p0, opt=opt, state=s0, units=units)
    monkeypatch.setattr(dp, "apply_param_updates", real)
    assert len(seen) == 2 * rep["k"]
    seen = seen[1::2]
    p, s = p0, s0
    for j, (vg, x) in enumerate(units[:rep["k"]]):
        p, s, _ = dp.mpi_data_parallel_train_step(vg, p, s, opt, x, 1)
        assert jax.tree.all(jax.tree.map(lambda a, b: bool(jnp.all(a == b)),
                                         p, seen[j])), f"step {j + 1} differs"
    assert jax.tree.all(jax.tree.map(lambda a, b: bool(jnp.all(a == b)),
                                     s, rep["opt_state"]))


def test_loop_and_gate_share_the_apply_function(monkeypatch):
    calls = []
    real = dp.apply_param_updates

    def counted(p, u):
        calls.append(1)
        return real(p, u)
    monkeypatch.setattr(dp, "apply_param_updates", counted)
    rep = _gate()
    n_gate = len(calls)
    # per step: the movement probe and the run-form update, both tree-level
    assert n_gate == 2 * rep["k"]
    opt, p = create_optimizer(TC), _params()
    dp.mpi_data_parallel_train_step(jax.value_and_grad(_loss), p, opt.init(p),
                                    opt, X, 1)
    assert len(calls) == n_gate + 1


def test_run_update_dropped_while_probe_works_aborts(monkeypatch):
    # Per step the gate applies twice: movement probe, then the run's update.
    # Dropping ONLY the run's update must still abort (codex round 3 plant).
    real = dp.apply_param_updates
    n = [0]

    def drop_every_second(p, u):
        n[0] += 1
        return p if n[0] % 2 == 0 else real(p, u)
    monkeypatch.setattr(dp, "apply_param_updates", drop_every_second)
    _fails("the update was not applied")
