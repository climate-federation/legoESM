"""Direct test of the round-315 ORCA2 gradient probe's machinery (no deck needed)."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round315_grad_probe as probe,
)

jax.config.update("jax_enable_x64", True)

DT = 0.1
N_LIST = (1, 2, 3)


def _losses():
    return {"T": lambda st: jnp.sum(st["T"] ** 2)}


def _toy_step(st, s, dt, fw, sf):
    return {"T": st["T"] * (1.0 + s["a"] * dt) + fw, "k": st["k"]}


def _analytic(t0, a, fw, n):
    g = 1.0 + a * DT
    tn = t0 * g ** n + fw * sum(g ** j for j in range(n))
    dtn_da = (t0 * n * g ** (n - 1) * DT
              + fw * sum(j * g ** (j - 1) * DT for j in range(1, n)))
    return 2.0 * tn * g ** n, float(np.sum(2.0 * tn * dtn_da))


def _setup():
    t0 = jnp.asarray([0.3, -1.2, 2.0])
    state = {"T": t0, "k": jnp.asarray([1, 2], dtype=jnp.int32)}
    forcing = [(0.05, None)] * max(N_LIST)
    return t0, state, forcing


@pytest.mark.parametrize("remat", [False, True])
def test_chain_reverse_matches_analytic_and_forward_mode(remat):
    t0, state, forcing = _setup()
    a = 0.7
    step = jax.jit(_toy_step)
    rev = probe.chain_reverse(step, state, {"a": jnp.float64(a)}, forcing,
                              N_LIST, _losses(), DT, remat=remat)
    for n in N_LIST:
        d_t0, d_a = _analytic(np.asarray(t0), a, 0.05, n)
        got = rev[("T", n)]
        np.testing.assert_allclose(np.asarray(got["dstate0"]["T"]), d_t0, rtol=1e-13)
        assert probe.rel_err(got["dscales"]["a"], d_a) < 1e-13
    fwd = probe.chain_forward(step, state, {"a": jnp.float64(a)}, forcing, N_LIST,
                              _losses(), DT, tangent_scales={"a": jnp.float64(1.0)})
    for n in N_LIST:
        assert probe.rel_err(fwd[("T", n)][1], rev[("T", n)]["dscales"]["a"]) < 1e-13


def test_planted_wrong_pullback_fails_the_fd_gate_and_custom_vjp_skips_jvp():
    t0, state, forcing = _setup()

    @jax.custom_vjp
    def scale(x):
        return x

    scale.defvjp(lambda x: (x, None), lambda _, g: (2.0 * g,))  # planted: 2x

    def bad_step(st, s, dt, fw, sf):
        out = _toy_step(st, s, dt, fw, sf)
        return {"T": scale(out["T"]), "k": out["k"]}

    step = jax.jit(bad_step)
    sc = {"a": jnp.float64(0.7)}
    rev = probe.chain_reverse(step, state, sc, forcing, (1,), _losses(), DT)
    d = jnp.asarray([1.0, 0.0, -1.0])
    tangent = {"T": d, "k": np.zeros((2,), dtype=jax.dtypes.float0)}
    # the step holds a custom_vjp: forward mode is unavailable, not a crash
    assert probe._try_jvp(step, state, sc, forcing, (1,), _losses(), DT,
                          tangent_state=tangent) == {}
    fd = {}
    for eps in (1e-3, 1e-4, 1e-5):
        up = probe.chain_forward(step, {"T": t0 + eps * d, "k": state["k"]}, sc,
                                 forcing, (1,), _losses(), DT)
        dn = probe.chain_forward(step, {"T": t0 - eps * d, "k": state["k"]}, sc,
                                 forcing, (1,), _losses(), DT)
        fd[eps] = (up[("T", 1)] - dn[("T", 1)]) / (2 * eps)
    ad = float(jnp.vdot(rev[("T", 1)]["dstate0"]["T"], d))
    row = probe._row("T_dir", "T", 1, ad, None, fd)
    assert row["revfwd_pass"] is None and not row["best"]["pass"]
    good = probe.chain_reverse(jax.jit(_toy_step), state, sc, forcing, (1,), _losses(), DT)
    good_ad = float(jnp.vdot(good[("T", 1)]["dstate0"]["T"], d))
    assert probe._row("T_dir", "T", 1, good_ad, None, fd)["best"]["pass"]


def test_best_fd_picks_the_sweep_minimum_and_applies_tolerance():
    best = probe.best_fd(1.0, {1e-2: 1.01, 1e-4: 1.0 + 1e-9, 1e-6: 1.0 + 1e-5})
    assert best == {"eps": 1e-4, "rel_err": pytest.approx(1e-9, rel=1e-3), "pass": True}
    assert not probe.best_fd(1.0, {1e-3: 1.1})["pass"]
    assert probe.rel_err(0.0, 0.0) == 0.0


class _GM(NamedTuple):
    kappa_Redi: float  # noqa: N815  (mirrors the real config field)


class _LM(NamedTuple):
    gm_redi: _GM


class _TKE(NamedTuple):
    kappaH_min: float  # noqa: N815  (mirrors the real config field)


class _CONST(NamedTuple):
    K_v: float


class _VM(NamedTuple):
    scheme: str
    tke: _TKE
    constant: _CONST


class _PH(NamedTuple):
    lateral_mixing: _LM
    vertical_mixing: _VM


class _BD(NamedTuple):
    bottom_drag_cd0: float


class _CFG(NamedTuple):
    physics: _PH
    bottom_drag: _BD


class _Z(NamedTuple):
    nemo_ldf_ahmt: np.ndarray
    nemo_ldf_ahmf: np.ndarray


@pytest.mark.parametrize("scheme", ["tke", "constant"])
def test_scaled_parts_is_bit_identical_at_one_and_scales_only_its_field(scheme):
    cfg = _CFG(_PH(_LM(_GM(1000.0)), _VM(scheme, _TKE(1.4e-10), _CONST(1.2e-5))),
               _BD(1e-3))
    z = _Z(np.array([1.0e4, 3.3e3]), np.array([2.0e4]))
    same_cfg, same_z = probe.scaled_parts(cfg, z, {k: 1.0 for k in probe.SCALARS})
    assert same_cfg == cfg
    assert np.array_equal(same_z.nemo_ldf_ahmt, z.nemo_ldf_ahmt)
    bumped, _ = probe.scaled_parts(cfg, z, {"kv_background": 2.0})
    vm = bumped.physics.vertical_mixing
    if scheme == "tke":
        assert vm.tke.kappaH_min == 2.8e-10 and vm.constant.K_v == 1.2e-5
    else:
        assert vm.constant.K_v == 2.4e-5 and vm.tke.kappaH_min == 1.4e-10
    assert bumped.bottom_drag == cfg.bottom_drag
    with pytest.raises(probe.GateError):
        probe.scaled_parts(cfg._replace(physics=cfg.physics._replace(
            vertical_mixing=cfg.physics.vertical_mixing._replace(scheme="kpp"))),
            z, {})
