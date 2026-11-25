"""Round 323 (codex HOLD): rung 10's per-step damping target reaches every
public run path (step_checked, integrate, integrate_scan), each path equals the
ladder's states bit for bit, a stale target is refused on every path, and a
scan-based gradient is finite.

Needs the round-5 rung-10 acquisition record; the bitwise and gradient tests
compile rung 10's step, its scan and its pullback (minutes, tens of GB).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not jax.config.jax_enable_x64, reason="needs JAX_ENABLE_X64=1")

N = 3


def _probe():
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round315_grad_probe as probe,
    )
    return probe


def _record_exists():
    return _probe().ROOTS["rung10"].exists()


@pytest.fixture(scope="module")
def rung10():
    if not _record_exists():
        pytest.skip("no rung-10 record")
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    card, state0, forcing = _probe().build("rung10")
    r = card.recipe
    assert r.model_config.rk3_stage3_damping_after_geothermal_nemo_order
    model = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config,
                                  iwm_forcing=r.iwm_forcing)
    inputs = [{"freshwater": fw, "surface_forcing": sf, "sponge": dmp}
              for fw, sf, dmp in forcing[:N]]
    for kt, kw in enumerate(inputs, start=1):
        assert kw["sponge"].nemo_kt == kt
    return card, model, state0, inputs


def _stale(inputs, i=1):
    out = [dict(kw) for kw in inputs]
    out[i]["sponge"] = inputs[0]["sponge"]  # step i+1 handed step 1's target
    return out


def test_stale_or_missing_target_is_refused_on_every_path(rung10):
    card, model, state0, inputs = rung10
    dt, kw = card.dt_s, inputs[1]
    stale = dict(kw, sponge=inputs[0]["sponge"])
    with pytest.raises(ValueError, match="stale"):
        model.step(state0, dt, nemo_kt=2, **stale)
    with pytest.raises(ValueError, match="stale"):
        model.step_checked(state0, dt, nemo_kt=2, **stale)
    with pytest.raises(ValueError, match="requires nemo_kt"):
        model.step_checked(state0, dt, **kw)
    with pytest.raises(ValueError, match="stale"):
        model.integrate(state0, 2 * dt, dt, step_inputs=[stale, stale], nemo_kt0=2)
    with pytest.raises(ValueError, match="requires nemo_kt"):
        model.integrate(state0, dt, dt, step_inputs=inputs[:1])
    with pytest.raises(ValueError, match="stale"):
        model.integrate_scan(state0, N, dt, step_inputs=_stale(inputs), nemo_kt0=1)
    with pytest.raises(ValueError, match="stale"):
        model.integrate_scan(state0, N, dt, step_inputs=inputs, nemo_kt0=2)
    with pytest.raises(ValueError, match="requires nemo_kt0"):
        model.integrate_scan(state0, N, dt, step_inputs=inputs)
    # a traced counter must come with the target that counter selected
    sp = kw["sponge"]
    with pytest.raises(ValueError, match="traced step counter"):
        jax.eval_shape(lambda k: model.step(
            state0, dt, kw["freshwater"], kw["surface_forcing"],
            sp._replace(nemo_kt=k + 0), nemo_kt=k), 2)


def _ladder_states(card, state0, inputs):
    """The round-1 ladder's model (live stage operands exposed) and step call."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    r = card.recipe
    model = LatLonCGridOceanModel(
        r.grid, r.z_coord, r.model_config, iwm_forcing=r.iwm_forcing,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_live_stage_operands=True))
    states, st = [], state0
    for kt, kw in enumerate(inputs, start=1):
        st = model.step(st, dt=card.dt_s, freshwater=kw["freshwater"],
                        surface_forcing=kw["surface_forcing"],
                        sponge=kw["sponge"], nemo_kt=kt).state_after
        states.append(jax.device_get(st))
    return states


def _unequal_leaves(a, b):
    la, ta = jax.tree_util.tree_flatten_with_path(a)
    lb, tb = jax.tree_util.tree_flatten_with_path(b)
    assert ta == tb
    return [(jax.tree_util.keystr(p), float(np.max(np.abs(np.asarray(x) - np.asarray(y)))))
            for (p, x), (_, y) in zip(la, lb)
            if not np.array_equal(np.asarray(x), np.asarray(y))]


@pytest.fixture(scope="module")
def ladder(rung10):
    card, _, state0, inputs = rung10
    states = _ladder_states(card, state0, inputs)
    # non-vacuous: the target moves between steps and the state moves with it
    assert not np.array_equal(inputs[0]["sponge"].T_ref, inputs[1]["sponge"].T_ref)
    assert _unequal_leaves(states[0], states[1])
    return states


def test_integrate_equals_the_ladder_bit_for_bit(rung10, ladder):
    card, model, state0, inputs = rung10
    _, traj = model.integrate(state0, N * card.dt_s, card.dt_s,
                              step_inputs=inputs, nemo_kt0=1)
    for n in range(N):
        assert _unequal_leaves(ladder[n], jax.device_get(traj[n + 1])) == [], n


def test_integrate_scan_equals_the_ladder_bit_for_bit(rung10, ladder):
    card, model, state0, inputs = rung10
    _, stacked = model.integrate_scan(state0, N, card.dt_s, step_inputs=inputs,
                                      nemo_kt0=1)
    diffs = [_unequal_leaves(ladder[n], jax.device_get(
        jax.tree.map(lambda a, n=n: a[n], stacked))) for n in range(N)]
    print("scan vs ladder:", diffs)
    # Every prognostic leaf is bit-identical.  The write-only diagnostic w
    # (nothing reads it back) differs by <= 2.7e-20 m/s from the second
    # in-loop step on: XLA compiles the step differently inside a multi-trip
    # loop (a one-step scan and an outer jit both match exactly).
    assert all(name.startswith(".w[") and err <= 1e-19 for d in diffs for name, err in d), diffs
    assert diffs[0] == []


def test_scan_gradient_through_two_steps_is_finite(rung10):
    card, model, state0, inputs = rung10
    losses, _ = _probe().make_losses(model, card.recipe.z_coord)
    # from the step-1 state (barotropic history carried), so kt 2 and 3 both
    # run inside the scan
    state1 = model.step(state0, card.dt_s, nemo_kt=1, **inputs[0])

    def loss(dT):
        st = state1._replace(T=state1.T.replace(data=state1.T.data + dT))
        final, _ = model.integrate_scan(st, 2, card.dt_s, step_inputs=inputs[1:3],
                                        nemo_kt0=2)
        return losses["T"](final)

    g = np.asarray(jax.grad(loss)(jnp.zeros_like(state1.T.data)))
    print(f"scan kt 2-3 dL/dT: nonfinite {int((~np.isfinite(g)).sum())}, "
          f"max |g| {np.nanmax(np.abs(g)):.6e}")
    assert np.isfinite(g).all() and np.abs(g).max() > 0.0
