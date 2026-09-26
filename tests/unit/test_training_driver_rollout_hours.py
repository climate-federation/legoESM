"""The carry-based trainers must accept the caller's supervision horizon.

REGRESSION. ``train_physics_params`` / ``train_neural_gcm`` lost their
``rollout_hours`` and physics kwargs, so BOTH lat-lon AIMIP variants died at the
call with ``TypeError: got an unexpected keyword argument 'rollout_hours'`` —
i.e. classical and column_nn could not be trained on the lat-lon C-grid at all,
while the spectral path (a different trainer) kept working.

The horizon is not cosmetic. ``single_day_rollout`` defaults to ``hours=24``;
the lat-lon driver loads targets at a 6 h lead. Accepting the kwarg but ignoring
it would score a 24 h forecast against a 6 h target and train quietly on the
wrong thing, which is worse than the TypeError.
"""

from __future__ import annotations

import inspect

from legoesm.training import training_driver as td


def test_both_trainers_accept_rollout_hours():
    for fn in (td.train_physics_params, td.train_neural_gcm):
        params = inspect.signature(fn).parameters
        assert "rollout_hours" in params, (
            f"{fn.__name__} dropped rollout_hours; the lat-lon AIMIP driver "
            "passes it and dies at the call")
        assert params["rollout_hours"].kind is inspect.Parameter.KEYWORD_ONLY


def test_both_trainers_forward_physics_kwargs_to_the_segment():
    """`microphysics` / `rad_update_steps` configure the ROLLOUT's physics.

    Silently dropping them builds a different model from the one being tuned,
    so the trainers take **segment_kwargs and hand them to
    ``build_training_segment``.
    """
    for fn in (td.train_physics_params, td.train_neural_gcm):
        params = inspect.signature(fn).parameters
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in params.values()), (
            f"{fn.__name__} takes no **segment_kwargs")
        src = inspect.getsource(fn)
        assert "**segment_kwargs" in src and "build_training_segment" in src


def test_the_horizon_actually_reaches_single_day_rollout():
    """Accepting the kwarg and ignoring it is the dangerous failure mode.

    Asserted on ``_build_train_step``, the function that RUNS the rollout —
    not on the public wrappers, which merely forward.
    """
    src = inspect.getsource(td._build_train_step)
    assert "rollout_hours" in src
    assert "hours=rollout_hours" in src, (
        "single_day_rollout is called without the horizon, so it falls back to "
        "its 24 h default regardless of the target lead")


def test_the_classical_trainer_honours_rad_stop_gradient():
    """`--radiation-as-forcing` must reach the pipeline, not be swallowed."""
    params = inspect.signature(td.train_physics_params).parameters
    assert "rad_stop_gradient" in params
    src = inspect.getsource(td.train_physics_params)
    assert "rad_stop_gradient=rad_stop_gradient" in src


def test_latlon_driver_exit_code_reflects_variant_failures():
    """It returned 0 with every variant holding an 'error' entry."""
    src = (td.__file__.rsplit("packages/", 1)[0]
           + "scripts/run/run_aimip_latlon.py")
    text = open(src).read()
    assert 'failed = sorted(v for v, r in scorecard.items() if "error" in r)' \
        in text
    assert "return 1" in text


def test_trained_params_win_over_caller_segment_kwargs():
    """The two kwarg sources OVERLAP, so they must be merged, not double-splatted.

    ``to_segment_kwargs()`` and the driver's ``segment_kwargs`` both carry
    ``microphysics``. Two ``**`` of the same key is a TypeError at the call —
    which is exactly how the classical lat-lon variant kept dying after the
    signature was restored:

        TypeError: build_segment_fn() got multiple values for keyword
                   argument 'microphysics'

    Precedence matters as much as the merge: the TRAINED value must win. It is
    the quantity being optimised, and letting the caller's static default
    override it would silently cut that parameter out of the rollout.
    """
    src = inspect.getsource(td.train_physics_params)
    assert "**segment_kwargs,\n            **trainable.to_segment_kwargs()" not in src
    assert "{**segment_kwargs, **trainable.to_segment_kwargs()}" in src, (
        "trained values must come SECOND in the merge so they win")


def test_build_training_segment_defaults_are_overridable():
    """The four segment defaults must be DEFAULTS, not hardcodes.

    They were passed by name next to ``**extra_kwargs``, so any caller
    supplying one died with "got multiple values for keyword argument". And
    when nobody supplied them, every carry-based training rollout silently ran
    with microphysics="none" and fix_mass=False no matter what the experiment
    asked for — the tuned model was not the configured model.
    """
    src = inspect.getsource(td.build_training_segment)
    assert "seg_defaults" in src and "seg_defaults.update(extra_kwargs)" in src
    # the collision-prone names must no longer be passed alongside the splat
    for name in ("rad_update_steps=1", 'microphysics="none"',
                 "fix_moisture=False", "fix_mass=False"):
        assert src.count(name) == 1, (
            f"{name} appears more than once — it is both a default and an "
            "explicit call argument again")
    assert "**extra_kwargs,\n    )" not in src


def test_segment_defaults_really_are_applied_when_absent():
    """Non-vacuity: the defaults dict must still carry the original values."""
    src = inspect.getsource(td.build_training_segment)
    for name in ("rad_update_steps=1", 'microphysics="none"',
                 "fix_moisture=False", "fix_mass=False"):
        assert name in src


def test_trainers_route_through_the_shared_multi_step_loss():
    """``multi_step_rollout_loss`` had NO production caller.

    Its docstring says it is "shared by every AIMIP trainer so the rollout+loss
    is defined ONCE", but ``_build_train_step`` did its own
    ``single_day_rollout`` + ``combined_loss``, so the only thing calling the
    shared version was its own unit test. Consequence:
    ``loss_config.multi_step_hours`` was SILENTLY IGNORED — the lat-lon driver
    built tuple-of-lead targets and handed them to a loss that only ever ran one
    rollout.

    Asserted on ``_build_train_step``, the function that RUNS.
    """
    src = inspect.getsource(td._build_train_step)
    assert "_rollout_loss(" in src
    assert "rollout_hours=rollout_hours" in src
    loss_src = inspect.getsource(td._rollout_loss)
    assert "multi_step_rollout_loss(" in loss_src, (
        "the trainer still inlines its own rollout+loss")
    for s_ in (src, loss_src):
        assert "single_day_rollout(" not in s_, (
            "the inline single-horizon rollout is still there, so multi-step "
            "supervision stays dead")


# ---------------------------------------------------------------------------
# EXECUTED tests. Everything above asserts on SOURCE TEXT, which is a tripwire,
# not a proof: it passes if someone renames a variable while breaking the
# behaviour, and it cannot see whether the value actually ARRIVES. Codex flagged
# exactly that. These call the real trainers with lightweight fakes and observe
# what the rollout is handed.
# ---------------------------------------------------------------------------

def _fakes(monkeypatch, seen):
    """Patch the two heavy dependencies and record what the rollout receives."""
    import jax.numpy as jnp

    class _Seg:
        def raw(self, carry, n_steps, forcing):
            return carry

    def fake_build_segment_fn(**kw):
        seen.setdefault("segment_kwargs", []).append(kw)
        return _Seg()

    def fake_rollout(ic, forcing, run_seg_fn, *, dt, hours=24.0):
        seen.setdefault("hours", []).append(float(hours))
        return run_seg_fn(ic, 1, forcing)

    monkeypatch.setattr(td, "build_segment_fn", fake_build_segment_fn)
    monkeypatch.setattr(td, "single_day_rollout", fake_rollout)
    monkeypatch.setattr(
        td, "combined_loss",
        lambda pred, target, sigma_full, grid=None, config=None:
            jnp.sum((pred - target) ** 2))


def _tiny_net(nlev=3):
    """A REAL NeuralPhysics, tiny. A duck-typed stub does not survive
    ``make_neural_step_unified``, which reads ``nlev`` off it — and a fake that
    dodges the real call path would defeat the point of an executed test."""
    import jax
    from legoesm.atmosphere.physics.neural_physics import NeuralPhysics
    return NeuralPhysics(nlev=nlev, hidden_dim=4, n_layers=2,
                         key=jax.random.PRNGKey(0))


def _tiny_inputs():
    import jax.numpy as jnp
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    grid = create_gaussian_grid(6, dealiasing="quadratic")
    sigma = create_sigma_coordinate(3)
    ic = jnp.ones((2,))
    return grid, sigma, [ic], [ic * 2.0], [jnp.zeros((2,))]


def test_rollout_hours_ACTUALLY_reaches_the_rollout(monkeypatch):
    """6 h in must be 6 h at the rollout, not the 24 h default."""
    import optax
    from legoesm.training.losses import LossConfig

    grid, sigma, ics, tgts, forc = _tiny_inputs()
    seen = {}
    _fakes(monkeypatch, seen)
    monkeypatch.setattr(td, "_make_driver_optimizer",
                        lambda *a, **k: optax.sgd(0.0))

    td.train_neural_gcm(
        object(), grid, sigma, _tiny_net(), ics, tgts, forc,
        rollout_hours=6.0, n_epochs=1, dt=600.0,
        loss_config=LossConfig(), log_every=1000)

    assert seen["hours"], "the rollout was never called"
    assert set(seen["hours"]) == {6.0}, (
        f"rollout got hours={set(seen['hours'])}, expected 6.0 — the horizon "
        "is not arriving, so a 6 h target would be scored against a 24 h "
        "forecast")


def test_segment_kwargs_ACTUALLY_reach_build_segment_fn(monkeypatch):
    """microphysics/rad_update_steps must override the built-in defaults."""
    import optax
    from legoesm.training.losses import LossConfig

    grid, sigma, ics, tgts, forc = _tiny_inputs()
    seen = {}
    _fakes(monkeypatch, seen)
    monkeypatch.setattr(td, "_make_driver_optimizer",
                        lambda *a, **k: optax.sgd(0.0))

    td.train_neural_gcm(
        object(), grid, sigma, _tiny_net(), ics, tgts, forc,
        rollout_hours=6.0, n_epochs=1, dt=600.0,
        loss_config=LossConfig(), log_every=1000,
        microphysics="kessler", rad_update_steps=7)

    kws = seen["segment_kwargs"]
    assert kws, "build_segment_fn was never called"
    assert all(k["microphysics"] == "kessler" for k in kws), (
        f"microphysics did not override the default: "
        f"{[k['microphysics'] for k in kws]}")
    assert all(k["rad_update_steps"] == 7 for k in kws)
    # and the untouched defaults survive
    assert all(k["fix_mass"] is False for k in kws)


def test_defaults_survive_when_the_caller_passes_nothing(monkeypatch):
    """Non-vacuity for the test above: absent kwargs keep the old values."""
    import optax
    from legoesm.training.losses import LossConfig

    grid, sigma, ics, tgts, forc = _tiny_inputs()
    seen = {}
    _fakes(monkeypatch, seen)
    monkeypatch.setattr(td, "_make_driver_optimizer",
                        lambda *a, **k: optax.sgd(0.0))

    td.train_neural_gcm(
        object(), grid, sigma, _tiny_net(), ics, tgts, forc,
        n_epochs=1, dt=600.0, loss_config=LossConfig(), log_every=1000)

    kws = seen["segment_kwargs"]
    assert all(k["microphysics"] == "none" for k in kws)
    assert all(k["rad_update_steps"] == 1 for k in kws)
    # ...and with no rollout_hours the documented 24 h default is what runs.
    assert set(seen["hours"]) == {24.0}


def _physics_fakes(monkeypatch):
    """A segment whose rollout reads ONLY ``sbm_tau_c`` (the other default
    trainables reach it but the forward never consumes them)."""
    import optax

    class _Seg:
        def __init__(self, kw):
            self.kw = kw

        def raw(self, carry, n_steps, forcing):
            return carry * self.kw["sbm_tau_c"] / 7200.0

    monkeypatch.setattr(td, "build_segment_fn", lambda **kw: _Seg(kw))
    monkeypatch.setattr(
        td, "single_day_rollout",
        lambda ic, forcing, run_seg_fn, *, dt, hours=24.0:
            run_seg_fn(ic, 1, forcing))
    monkeypatch.setattr(
        td, "combined_loss",
        lambda pred, target, sigma_full, grid=None, config=None:
            ((pred - target) ** 2).sum())
    monkeypatch.setattr(td, "_make_driver_optimizer",
                        lambda *a, **k: optax.sgd(0.0))

    class _Pipeline:
        def build_step_unified(self, rad_stop_gradient=False):
            return None

    return _Pipeline()


def test_classical_trainer_aborts_on_an_inert_parameter(monkeypatch):
    """No-inert-parameters gate: a trainable leaf the rollout never reads
    aborts the run before the first update."""
    import pytest
    from legoesm.training.losses import LossConfig

    grid, sigma, ics, tgts, forc = _tiny_inputs()
    pipe = _physics_fakes(monkeypatch)
    with pytest.raises(ValueError, match="C_E"):
        td.train_physics_params(
            object(), grid, sigma, pipe, ics, tgts, forc, rollout_hours=6.0,
            n_epochs=1, dt=600.0, loss_config=LossConfig(), log_every=1000)


def test_classical_trainer_trains_only_the_given_constraints(monkeypatch):
    """With the inert leaves frozen out the gate passes and only the given
    parameters are trained."""
    from legoesm.training.losses import LossConfig
    from legoesm.training.trainable_params import DEFAULT_TRAINABLE

    grid, sigma, ics, tgts, forc = _tiny_inputs()
    pipe = _physics_fakes(monkeypatch)
    live = [c for c in DEFAULT_TRAINABLE if c.name == "sbm_tau_c"]
    trained, _ = td.train_physics_params(
        object(), grid, sigma, pipe, ics, tgts, forc, rollout_hours=6.0,
        n_epochs=1, dt=600.0, loss_config=LossConfig(), log_every=1000,
        constraints=live)
    assert set(trained.raw_values) == {"sbm_tau_c"}
