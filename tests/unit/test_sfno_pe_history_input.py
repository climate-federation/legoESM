"""Two-snapshot (history) input for the pure-SFNO emulator.

WHY. Measured 2026-08-08 on the muonlr arm (2017, 8 inits): z500 RMSE is 11.6 m
after ONE 6-hour step and 20.5 m at 24 h, so 57 % of the day-1 error is made in
the first step — and that single step already costs 2.9x GraphCast's entire 24 h
forecast (4.06 m). The binding constraint is single-step accuracy. We feed ONE
snapshot, so the network has to infer d/dt from spatial structure; U-Cast and
GraphCast both feed two, which turns the tendency into a finite difference the
network can read directly.

``history_steps`` defaults to 0, so every existing arm is untouched; these tests
pin that as well as the new path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.neural.sfno_pe import (  # noqa: E402
    SFNOPrimitiveEquationConfig,
    SFNOPrimitiveEquationModel,
)
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (  # noqa: E402
    isothermal_rest_state_spectral,
)
from legoesm.grids.gaussian import create_gaussian_grid  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402
from legoesm.ml.sfno import SFNO, SFNOConfig  # noqa: E402

_N_MAX = 8
_N_LEV = 3
_N_STATE = 4 * _N_LEV + 2          # PE3DChannelSpec width
_DT = 21600.0


@pytest.fixture(scope="module")
def env():
    grid = create_gaussian_grid(_N_MAX, dealiasing="quadratic")
    sigma = create_sigma_coordinate(_N_LEV)
    s0 = isothermal_rest_state_spectral(grid, sigma, T_init=280.0,
                                        p_s_init=1.0e5)
    return grid, sigma, s0


def _model(env, history_steps, in_channels=None):
    grid, sigma, _ = env
    in_ch = in_channels if in_channels is not None else _N_STATE * (
        1 + history_steps)
    sfno_cfg = SFNOConfig(in_channels=in_ch, out_channels=_N_STATE,
                          embed_dim=8, n_blocks=1)
    cfg = SFNOPrimitiveEquationConfig(
        sfno_config=sfno_cfg, mode="state_update", dt_sfno=_DT,
        correct_mass=False, use_normalization=False,
        history_steps=history_steps,
    )
    net = SFNO(sfno_cfg, grid, key=jax.random.PRNGKey(0))
    return SFNOPrimitiveEquationModel(
        grid=grid, sigma_coord=sigma, config=cfg, sfno_model=net)


def _perturb(state, grid, amp):
    idx = int(np.argmax(np.asarray(grid.ls) == 2))
    return state._replace(
        lnps_hat=state.lnps_hat.replace(data=state.lnps_hat.data.at[idx].add(amp)))


def test_default_is_no_history_and_needs_no_prev_states(env):
    """The historical single-snapshot input stays the default."""
    _, _, s0 = env
    m = _model(env, 0)
    assert m.config.history_steps == 0
    out = m.step(s0, _DT)
    assert np.all(np.isfinite(np.asarray(out.lnps_hat.data)))


def test_in_channels_mismatch_is_refused_not_mis_sliced(env):
    """A width that does not match the declared history must fail LOUDLY.

    Silently accepting it would either explode deep inside the network or, with
    a compatible-looking width, read the wrong channels and train fine while
    predicting from garbage.
    """
    with pytest.raises(ValueError, match="in_channels"):
        _model(env, 1, in_channels=_N_STATE)   # forgot to double the width


def test_negative_history_is_refused(env):
    with pytest.raises(ValueError, match="history_steps must be >= 0"):
        _model(env, -1)


def test_history_step_runs_and_consumes_the_past_state(env):
    """With history on, the output must actually DEPEND on the past state.

    This is the non-vacuity check: concatenating a tensor the network then
    ignores would pass a smoke test while buying nothing. Two different pasts
    with the SAME current state must give different forecasts.
    """
    grid, _, s0 = env
    m = _model(env, 1)
    prev_a = _perturb(s0, grid, 0.00)
    prev_b = _perturb(s0, grid, 0.05)

    out_a = m.step(s0, _DT, prev_states=(prev_a,))
    out_b = m.step(s0, _DT, prev_states=(prev_b,))
    a = np.asarray(out_a.lnps_hat.data)
    b = np.asarray(out_b.lnps_hat.data)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert not np.allclose(a, b), (
        "forecast is independent of the history channel — the past state is "
        "being packed but not used")


def test_wrong_number_of_prev_states_is_refused(env):
    _, _, s0 = env
    m = _model(env, 1)
    with pytest.raises(ValueError, match="requires that many prev_states"):
        m.step(s0, _DT)                       # history declared, none supplied
    with pytest.raises(ValueError, match="requires that many prev_states"):
        m.step(s0, _DT, prev_states=(s0, s0))  # one too many


def test_history_is_rejected_in_hybrid_mode(env):
    """hybrid_tendencies evaluates the net at RK STAGE states, which have no
    history; reusing the macro-step history per stage is inconsistent."""
    grid, sigma, s0 = env
    sfno_cfg = SFNOConfig(in_channels=_N_STATE * 2, out_channels=_N_STATE,
                          embed_dim=8, n_blocks=1)
    cfg = SFNOPrimitiveEquationConfig(
        sfno_config=sfno_cfg, mode="hybrid_tendencies", dt_sfno=_DT,
        correct_mass=False, use_normalization=False, history_steps=1)
    m = SFNOPrimitiveEquationModel(
        grid=grid, sigma_coord=sigma, config=cfg,
        sfno_model=SFNO(sfno_cfg, grid, key=jax.random.PRNGKey(0)))
    with pytest.raises(ValueError, match="only supported in"):
        m.step(s0, _DT, prev_states=(s0,))


def test_residual_skip_sees_the_CURRENT_state_not_the_past_one(env):
    """Channel order is load-bearing under `residual_prediction` (the DEFAULT).

    SFNO adds ``x_in[..., :out_channels]`` — the FIRST out_channels block of its
    input (ml/sfno.py). If the history were packed first, a residual
    state-update net would predict ``past_state + correction`` instead of
    ``current + correction``: a silently 6 h-stale baseline, on the default
    configuration, with no error anywhere.

    Test: with a residual net and a zeroed decoder contribution the output must
    track the CURRENT state. Comparing the two orderings directly, the residual
    baseline must be the current state — asserted by giving the past a large
    offset and checking the output does not follow it.
    """
    grid, sigma, s0 = env
    sfno_cfg = SFNOConfig(in_channels=_N_STATE * 2, out_channels=_N_STATE,
                          embed_dim=8, n_blocks=1, residual_prediction=True)
    cfg = SFNOPrimitiveEquationConfig(
        sfno_config=sfno_cfg, mode="state_update", dt_sfno=_DT,
        correct_mass=False, use_normalization=False, history_steps=1)
    m = SFNOPrimitiveEquationModel(
        grid=grid, sigma_coord=sigma, config=cfg,
        sfno_model=SFNO(sfno_cfg, grid, key=jax.random.PRNGKey(0)))

    near = _perturb(s0, grid, 0.0)
    far = _perturb(s0, grid, 5.0)     # a wildly different "past"
    out_near = np.asarray(m.step(s0, _DT, prev_states=(near,)).lnps_hat.data)
    out_far = np.asarray(m.step(s0, _DT, prev_states=(far,)).lnps_hat.data)

    idx = int(np.argmax(np.asarray(grid.ls) == 2))
    # The residual baseline is the CURRENT state, identical in both calls, so a
    # 5.0 shift in the PAST must not move the output by anything like 5.0. With
    # the past packed first it would move by ~5.0 exactly.
    assert abs(out_far[idx] - out_near[idx]) < 1.0, (
        "the residual skip is adding the PAST state — channel order is wrong")


def test_history_path_is_differentiable_wrt_the_past_state(env):
    """Gradients must reach the history channels, or training cannot use them."""
    grid, _, s0 = env
    m = _model(env, 1)
    idx = int(np.argmax(np.asarray(grid.ls) == 2))
    base = s0.lnps_hat.data

    def loss(amp):
        prev = s0._replace(
            lnps_hat=s0.lnps_hat.replace(data=base.at[idx].add(amp)))
        out = m.step(s0, _DT, prev_states=(prev,))
        return jnp.sum(jnp.abs(out.lnps_hat.data) ** 2)

    g = float(jax.jit(jax.grad(loss))(0.01))
    assert np.isfinite(g)
    assert abs(g) > 0.0, "no gradient flows to the history input"


# ---------------------------------------------------------------------------
# The training wiring is PARTIAL and must therefore be SAFELY GATED.
# ---------------------------------------------------------------------------

def test_sfno_history_is_gated_not_silently_partial():
    """`sfno_history_steps > 0` must FAIL LOUDLY until training is fully wired.

    Landed so far: the model input, the pairing helper, the history-carrying
    rollout body, the config field and the architecture sizing. NOT landed: the
    three sfno_full loss functions and the sample loop (including the pmap
    batch path) do not yet thread a previous state, so no call site passes
    ``prev_state``.

    A partially-wired feature that quietly did something reasonable — defaulting
    the predecessor to the current state, say — would feed the network a zero
    tendency and train a different model than the config describes. That is the
    exact failure class this file's own history exists to prevent, so the
    half-state raises instead.
    """
    import inspect

    from legoesm.training import neural_gcm_spectral as ngs

    src = inspect.getsource(ngs)
    i = src.index("def _rollout_segment")
    body = src[i:i + 4000]

    assert "prev_state=None" in body, "the rollout takes no predecessor"
    assert "no prev_state was" in body, (
        "history mode does not refuse a missing predecessor — it could train "
        "silently on a fabricated one")
    assert "_body_hist" in body, "the history-carrying scan body is missing"

    # And the gate is REACHABLE: no production call site supplies prev_state,
    # so enabling history today hits the refusal rather than running.
    calls = [ln for ln in src.splitlines()
             if "_rollout_segment(" in ln and "def " not in ln]
    assert len(calls) >= 3
    assert not any("prev_state" in c for c in calls), (
        "a call site now passes prev_state — training may be wired, so this "
        "gate test needs replacing with a real end-to-end history test")


def test_history_default_leaves_the_architecture_untouched():
    """history_steps=0 must size in_channels exactly as before.

    Asserted on the BUILT config rather than on a source string: the sizing
    expression moved into ``training.model_registry`` when both campaigns were
    put on one model builder, and a grep for a literal line in
    ``neural_gcm_spectral`` silently stops testing anything the moment the code
    it names is refactored.
    """
    from legoesm.ml.channel_packing import PE3DChannelSpec
    from legoesm.training.model_registry import sfno_arch_config

    nlev = 8
    n_state = PE3DChannelSpec(nlev=nlev).n_channels
    assert sfno_arch_config("sfno_full", nlev=nlev).in_channels == n_state
    assert sfno_arch_config(
        "sfno_full", nlev=nlev,
        overrides={"sfno_history_steps": 0}).in_channels == n_state
    assert sfno_arch_config(
        "sfno_full", nlev=nlev,
        overrides={"sfno_history_steps": 1}).in_channels == 2 * n_state
