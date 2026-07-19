"""Direct tests for ``scripts/bench/bench_ad_overhead.py``.

The benchmark's deliverable is a RATIO (gradient cost / forward cost), so the
things worth pinning are: the loss is a real differentiable scalar, the three
modes are built over the SAME step_fn and step count (a controlled
comparison), a mode failure is recorded rather than raised (a full-BPTT OOM is
a result), and the reported ratio is arithmetically what it claims to be.

Timing values themselves are hardware-dependent and are NOT asserted.
"""
from __future__ import annotations

import importlib.util
import os

import jax
import jax.numpy as jnp
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_MODULE_PATH = os.path.join(
    _REPO_ROOT, "scripts", "bench", "bench_ad_overhead.py")


def _load():
    spec = importlib.util.spec_from_file_location(
        "bench_ad_overhead", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load()


def _toy_state():
    return {"u": jnp.ones((4, 3)), "T": jnp.full((4, 3), 2.0),
            "mask": jnp.array([1, 0, 1, 0])}  # int leaf must be ignored


def _toy_step(state, dt):
    return {"u": state["u"] * (1.0 + dt), "T": state["T"] + dt * state["u"],
            "mask": state["mask"]}


def test_scalar_loss_is_finite_scalar_over_inexact_leaves():
    loss = bench.scalar_loss(_toy_state())
    assert loss.shape == ()
    assert jnp.isfinite(loss)
    # u (1.0) and T (2.0) contribute 1.0 + 4.0; the int mask must not.
    assert float(loss) == pytest.approx(5.0)


def test_scalar_loss_is_real_for_complex_leaves():
    """Spectral states carry complex128 coefficients.

    ``jnp.square`` would leave the loss complex and ``grad`` refuses a
    complex output, so the loss must take the squared MAGNITUDE.  Regression
    for a real failure on --grid spectral.
    """
    state = {"sp": jnp.array([3.0 + 4.0j, 0.0 + 0.0j])}
    loss = bench.scalar_loss(state)
    assert not jnp.iscomplexobj(loss), "loss must be real-valued"
    assert float(loss) == pytest.approx(12.5)  # (|3+4j|^2 + 0) / 2


def test_scalar_loss_differentiable_with_complex_leaves():
    import equinox as eqx

    state = {"sp": jnp.array([3.0 + 4.0j, 1.0 - 2.0j])}
    g = eqx.filter_grad(bench.scalar_loss)(state)
    assert jnp.all(jnp.isfinite(jnp.abs(g["sp"])))


def test_scalar_loss_ignores_integer_leaves():
    """An int leaf in the state must not enter the loss (non-differentiable)."""
    base = _toy_state()
    bumped = dict(base, mask=jnp.array([9, 9, 9, 9]))
    assert float(bench.scalar_loss(base)) == float(bench.scalar_loss(bumped))


def test_scalar_loss_is_differentiable():
    """Must differentiate via ``eqx.filter_grad`` — the path the bench uses.

    Bare ``jax.grad`` raises on the int ``mask`` leaf; filtering to inexact
    leaves is exactly why ``build_runners`` uses equinox here.
    """
    import equinox as eqx

    g = eqx.filter_grad(bench.scalar_loss)(_toy_state())
    assert jnp.all(jnp.isfinite(g["u"]))
    assert jnp.any(g["u"] != 0.0)


def test_bare_jax_grad_rejects_integer_leaves():
    """Pins the reason the bench cannot use bare ``jax.grad``."""
    with pytest.raises(TypeError, match="int"):
        jax.grad(bench.scalar_loss)(_toy_state())


def test_build_runners_exposes_all_three_modes():
    runners = bench.build_runners(_toy_step, dt=0.1, steps=3)
    assert set(runners) == {"forward", "grad", "grad_ckpt"}


def test_forward_runner_matches_hand_rolled_scan():
    """The timed forward path must be the plain integration, not an AD path."""
    runners = bench.build_runners(_toy_step, dt=0.1, steps=3)
    got = runners["forward"](_toy_state())
    ref = _toy_state()
    for _ in range(3):
        ref = _toy_step(ref, 0.1)
    assert jnp.allclose(got["u"], ref["u"])
    assert jnp.allclose(got["T"], ref["T"])


def test_grad_and_grad_ckpt_agree_to_tolerance():
    """Checkpointing changes the schedule, never the gradient value."""
    runners = bench.build_runners(_toy_step, dt=0.1, steps=4)
    state = _toy_state()
    g = runners["grad"](state)
    g_ckpt = runners["grad_ckpt"](state)
    assert jnp.allclose(g["u"], g_ckpt["u"], rtol=1e-6, atol=1e-6)
    assert jnp.allclose(g["T"], g_ckpt["T"], rtol=1e-6, atol=1e-6)


def _big_state(n=96):
    return {"u": jnp.ones((n, n)), "T": jnp.full((n, n), 2.0)}


def _big_step(state, dt):
    # Nonlinear so the backward genuinely needs saved residuals.
    return {"u": jnp.sin(state["u"]) * (1.0 + dt) + 0.01 * state["T"],
            "T": state["T"] + dt * jnp.tanh(state["u"])}


def test_grad_ckpt_actually_remats_resource_gate():
    """NON-VACUOUS gate: remat must show up as a RESOURCE difference.

    Comparing gradient VALUES cannot detect a disabled ``jax.checkpoint`` --
    value-invariance is precisely what remat guarantees, so that assertion
    passes whether or not remat happened.  (Verified: neutering
    ``jax.checkpoint`` to the identity leaves the value tests green.)  A real
    gate must observe the schedule: remat trades memory for recompute, so the
    checkpointed program must use LESS temp memory or MORE flops.
    """
    runners = bench.build_runners(_big_step, dt=0.1, steps=12)
    state = _big_state()
    plain = bench.analyze_mode(runners["grad"], state)
    ckpt = bench.analyze_mode(runners["grad_ckpt"], state)
    if not plain or not ckpt:
        pytest.skip("backend exposes no cost/memory analysis")

    less_memory = (plain.get("temp_bytes") and ckpt.get("temp_bytes")
                   and ckpt["temp_bytes"] < plain["temp_bytes"])
    more_flops = (plain.get("flops") and ckpt.get("flops")
                  and ckpt["flops"] > plain["flops"])
    assert less_memory or more_flops, (
        "grad_ckpt is indistinguishable from grad — jax.checkpoint is not "
        f"taking effect. plain={plain} ckpt={ckpt}")


def test_remat_gate_is_non_vacuous(monkeypatch):
    """Self-test: with jax.checkpoint neutered, the gate above must FAIL.

    Proves the resource gate is a real tripwire and not decoration.
    """
    monkeypatch.setattr(jax, "checkpoint", lambda f, *a, **k: f)
    runners = bench.build_runners(_big_step, dt=0.1, steps=12)
    state = _big_state()
    plain = bench.analyze_mode(runners["grad"], state)
    ckpt = bench.analyze_mode(runners["grad_ckpt"], state)
    if not plain or not ckpt:
        pytest.skip("backend exposes no cost/memory analysis")

    less_memory = (plain.get("temp_bytes") and ckpt.get("temp_bytes")
                   and ckpt["temp_bytes"] < plain["temp_bytes"])
    more_flops = (plain.get("flops") and ckpt.get("flops")
                  and ckpt["flops"] > plain["flops"])
    assert not (less_memory or more_flops), (
        "gate is VACUOUS: it still passes with jax.checkpoint disabled")


def test_grad_runner_returns_nonzero_gradient():
    runners = bench.build_runners(_toy_step, dt=0.1, steps=3)
    g = runners["grad"](_toy_state())
    assert jnp.any(g["u"] != 0.0), "a zero gradient would time an empty tape"


def test_time_modes_warms_once_then_samples_each_mode():
    calls = {"a": 0, "b": 0}

    def _mk(key):
        def fn(state):
            calls[key] += 1
            return state
        return fn

    runners = {"a": _mk("a"), "b": _mk("b")}
    out = bench.time_modes_interleaved(runners, _toy_state(), repeats=4,
                                       modes=("a", "b"))
    assert calls == {"a": 5, "b": 5}, "one warmup + `repeats` timed calls each"
    assert len(out["a"]["runs_s"]) == 4
    assert out["a"]["warmup_s"] >= 0.0


def test_time_modes_interleaves_rather_than_blocking():
    """Ordering must be round-robin: blocked runs put drift on the last mode."""
    order = []
    runners = {
        "a": lambda s: (order.append("a"), s)[1],
        "b": lambda s: (order.append("b"), s)[1],
    }
    bench.time_modes_interleaved(runners, _toy_state(), repeats=3,
                                 modes=("a", "b"))
    timed = order[2:]  # drop the two warmups
    assert timed == ["a", "b", "a", "b", "a", "b"], (
        f"expected interleaved sampling, got {timed}")


def test_non_oom_errors_are_raised_not_recorded():
    """A shape bug must surface loudly, not be filed as 'AD failed'."""
    def _bug(state):
        raise ValueError("shapes (3,) and (4,) are incompatible")

    with pytest.raises(ValueError, match="incompatible"):
        bench.time_modes_interleaved({"x": _bug}, _toy_state(), repeats=1,
                                     modes=("x",))


def test_run_records_mode_failure_instead_of_raising(monkeypatch):
    """A full-BPTT OOM is a RESULT (grad dies, grad_ckpt survives), not a crash."""
    def _boom(state):
        raise RuntimeError("RESOURCE_EXHAUSTED: out of memory")

    def _fake_build(step_fn, dt, steps, state_template=None):
        return {"forward": lambda s: s, "grad": _boom, "grad_ckpt": lambda s: s}

    monkeypatch.setattr(bench, "build_runners", _fake_build)

    import scripts.bench.run_cpu_mpi_scaling as driver
    monkeypatch.setattr(
        driver, "build_amip_step",
        lambda **kw: (_toy_step, _toy_state(), 1.0, 100, 100, None))

    result = bench.run("cubed-sphere", 4, 2, "float32", steps=2, repeats=1,
                       dt=1.0)
    assert result["modes"]["grad"]["ok"] is False
    assert "RESOURCE_EXHAUSTED" in result["modes"]["grad"]["error"]
    assert result["modes"]["forward"]["ok"] is True


def test_run_reports_ratio_relative_to_forward(monkeypatch):
    import scripts.bench.run_cpu_mpi_scaling as driver
    monkeypatch.setattr(
        driver, "build_amip_step",
        lambda **kw: (_toy_step, _toy_state(), 1.0, 100, 100, None))

    result = bench.run("cubed-sphere", 4, 2, "float32", steps=2, repeats=1,
                       dt=1.0)
    fwd = result["modes"]["forward"]["median_s"]
    for mode in ("grad", "grad_ckpt"):
        m = result["modes"][mode]
        if m.get("ok"):
            assert m["ratio_vs_forward"] == pytest.approx(m["median_s"] / fwd)
    assert result["modes"]["forward"]["ratio_vs_forward"] == pytest.approx(1.0)


def test_run_records_single_device_scope(monkeypatch):
    """The scope caveat must travel WITH the number, not just in prose."""
    import scripts.bench.run_cpu_mpi_scaling as driver
    monkeypatch.setattr(
        driver, "build_amip_step",
        lambda **kw: (_toy_step, _toy_state(), 1.0, 100, 100, None))

    result = bench.run("cubed-sphere", 4, 2, "float32", steps=2, repeats=1,
                       dt=1.0)
    assert result["distributed"] is False
    assert result["n_devices"] == 1
    assert result["physics_level"] == "none"


def test_format_report_marks_failed_modes():
    result = {
        "grid": "cubed-sphere", "resolution": 4, "n_levels": 2,
        "precision": "float32", "steps": 2, "physics_level": "none",
        "modes": {
            "forward": {"ok": True, "median_s": 0.1, "min_s": 0.09,
                        "warmup_s": 1.0, "ratio_vs_forward": 1.0},
            "grad": {"ok": False, "error": "RuntimeError: OOM"},
        },
    }
    text = bench.format_report(result)
    assert "FAILED" in text
    assert "1 device" in text


def test_format_report_warns_wall_ratio_is_machine_specific():
    """The bandwidth caveat must travel with the printed number."""
    result = {
        "grid": "latlon", "resolution": 8, "n_levels": 2,
        "precision": "float32", "steps": 2, "physics_level": "none",
        "modes": {"forward": {"ok": True, "median_s": 0.1, "min_s": 0.1,
                              "warmup_s": 1.0, "ratio_vs_forward": 1.0}},
    }
    text = bench.format_report(result)
    assert "FLOP x" in text
    assert "bandwidth-bound" in text
