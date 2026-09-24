"""Kessler on the FV3 duo six-face bundle: the pressure-native column body
and the duo layout bridge (``apply_kessler_step_sixface_jax``).

Run alone or with x64: the duo lane is fp64.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
jax.config.update("jax_enable_x64", True)

from legoesm.grids.vertical import (  # noqa: E402
    create_sigma_coordinate, pressure_from_sigma)
from legoesm.thermo import saturation_mixing_ratio  # noqa: E402
from legoesm.atmosphere.physics.microphysics.config import KesslerConfig  # noqa: E402
from legoesm.atmosphere.forcing.idealized.kessler_forcing import (  # noqa: E402
    KESSLER_TRACER_SLOTS, apply_kessler_step_sixface_jax,
    kessler_column_tendencies, kessler_column_tendencies_pressure)

N, NG, KM = 4, 3, 5
M = N + 2 * NG
DT = 300.0


def _columns(ncol=7, nlev=12, seed=3):
    rng = np.random.default_rng(seed)
    T = jnp.asarray(250.0 + 40.0 * rng.random((ncol, nlev)))
    p_s = jnp.asarray(9.0e4 + 2.0e4 * rng.random(ncol))
    sigma = create_sigma_coordinate(nlev, dtype=jnp.float64)
    p_full = pressure_from_sigma(sigma.sigma_full, p_s)
    q_sat = saturation_mixing_ratio(T, p_full)
    q_v = q_sat * jnp.asarray(0.5 + 0.8 * rng.random((ncol, nlev)))  # some supersat
    q_c = jnp.asarray(1e-4 * rng.random((ncol, nlev)))
    q_r = jnp.asarray(1e-4 * rng.random((ncol, nlev)))
    q_c = q_c.at[0, 0].set(-1e-5)                # the floor must still fire
    return T, p_s, sigma, q_v, q_c, q_r


def test_pressure_body_is_the_sigma_adapter_bitwise():
    """The refactor is a pure extraction: sigma adapter == pressure body
    fed the sigma pressures, bytes equal."""
    T, p_s, sigma, q_v, q_c, q_r = _columns()
    cfg = KesslerConfig()
    a = kessler_column_tendencies(T, p_s, q_v, q_c, q_r, sigma, dt=DT,
                                  config=cfg)
    b = kessler_column_tendencies_pressure(
        T, pressure_from_sigma(sigma.sigma_full, p_s),
        pressure_from_sigma(sigma.sigma_half, p_s), q_v, q_c, q_r,
        dt=DT, config=cfg)
    assert any(float(jnp.abs(x).max()) > 0 for x in a), "vacuous: no tendency"
    for x, y in zip(a, b):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


@pytest.fixture(scope="module")
def duo_bundle():
    """A synthetic six-face duo bundle with the lane's OWN pressure
    producer (p_var_hydrostatic), as the HS coupling tests build it."""
    from legoesm.core.fv3_dynamics import p_var_hydrostatic
    from legoesm.grids.fv3_native_gridstruct import FV3_KAPPA
    rng = np.random.default_rng(11)
    ptop = 100.0
    # surface pressure varies per FACE and per COLUMN (codex 2026-09-24:
    # a uniform profile cannot see a pe slice shifted by one cell or a
    # swapped horizontal axis), stretched layers so p_full != sigma * p_s
    ps = 1.0e5 + 2.0e3 * rng.random((6, M, M))
    delp = np.empty((6, M, M, KM))
    for k in range(KM):
        delp[..., k] = (ps - ptop) * (k + 1) / (KM * (KM + 1) / 2)
    press = p_var_hydrostatic(delp, ptop=ptop, akap=FV3_KAPPA, n=N, ng=NG,
                              km=KM)
    press = {nm: np.asarray(press[nm]) for nm in ("pe", "peln", "pkz")}
    pt = 260.0 + 20.0 * rng.random((6, M, M, KM))
    state = dict(u=rng.normal(size=(6, M, M + 1, KM)),
                 v=rng.normal(size=(6, M + 1, M, KM)), pt=pt, delp=delp)
    # q_v near saturation at the layer pressure so Kessler has work to do
    peln6 = np.transpose(press["peln"], (0, 1, 3, 2))
    ci = slice(NG, NG + N)
    p_full_c = delp[:, ci, ci] / (peln6[..., 1:] - peln6[..., :-1])
    q_v = np.zeros((6, M, M, KM))
    q_v[:, ci, ci] = np.asarray(saturation_mixing_ratio(
        jnp.asarray(pt[:, ci, ci]), jnp.asarray(p_full_c))) * 1.2
    q_c = np.zeros_like(q_v); q_c[:, ci, ci] = 2e-4
    q_r = np.zeros_like(q_v); q_r[:, ci, ci] = 1e-4
    return state, press, [jnp.asarray(q_v), jnp.asarray(q_c), jnp.asarray(q_r)]


def test_bridge_matches_a_column_by_column_call(duo_bundle):
    """Layout check: the bridge's (6,n,n,km) transposes/slices reproduce a
    per-column call built independently from pe/peln, bitwise."""
    state, press, q = duo_bundle
    out_state, out_q = apply_kessler_step_sixface_jax(
        state, press, q, dt=DT, n=N, ng=NG, km=KM)
    ci = slice(NG, NG + N)
    cfg = KesslerConfig()
    for t in range(6):
        for i in range(N):
            for j in range(N):
                pe = np.asarray(press["pe"])[t, 1 + i, :, 1 + j]        # (km+1,)
                peln = np.asarray(press["peln"])[t, i, :, j]            # (km+1,)
                # pe carries a one-cell halo, peln does not: the two
                # slices above address the SAME column iff exp(peln)==pe
                # (an off-by-halo pairing cannot satisfy this, GLM)
                assert np.allclose(np.exp(peln), pe, rtol=1e-12, atol=0)
                dp = state["delp"][t, NG + i, NG + j]
                p_full = dp / (peln[1:] - peln[:-1])
                T = state["pt"][t, NG + i, NG + j]
                dT, dqv, dqc, dqr = kessler_column_tendencies_pressure(
                    jnp.asarray(T)[None], jnp.asarray(p_full)[None],
                    jnp.asarray(pe)[None],
                    *(jnp.asarray(np.asarray(q[s])[t, NG + i, NG + j])[None]
                      for s in range(3)), dt=DT, config=cfg)
                exp_pt = T + np.asarray(dT)[0] * DT
                assert np.asarray(out_state["pt"])[t, NG + i, NG + j].tobytes() \
                    == exp_pt.tobytes(), (t, i, j)
                for s, dq in enumerate((dqv, dqc, dqr)):
                    exp_q = np.asarray(q[s])[t, NG + i, NG + j] \
                        + np.asarray(dq)[0] * DT
                    assert np.asarray(out_q[s])[t, NG + i, NG + j].tobytes() \
                        == exp_q.tobytes(), (t, i, j, s)
    moved = np.abs(np.asarray(out_state["pt"]) - state["pt"])[:, ci, ci]
    assert moved.max() > 0.0, "vacuous: Kessler moved nothing"


def test_bridge_touches_only_pt_and_the_three_tracers_on_the_block(duo_bundle):
    state, press, q = duo_bundle
    out_state, out_q = apply_kessler_step_sixface_jax(
        state, press, q, dt=DT, n=N, ng=NG, km=KM)
    for nm in ("u", "v", "delp"):
        assert np.asarray(out_state[nm]).tobytes() == np.asarray(
            state[nm]).tobytes(), nm
    ci = slice(NG, NG + N)
    halo = np.ones((M, M), bool); halo[ci, ci] = False
    assert np.array_equal(np.asarray(out_state["pt"])[:, halo],
                          state["pt"][:, halo])
    for s in range(3):
        assert np.array_equal(np.asarray(out_q[s])[:, halo],
                              np.asarray(q[s])[:, halo])
    assert len(out_q) == len(q)


def test_bridge_jit_equals_eager(duo_bundle):
    state, press, q = duo_bundle
    f = lambda st, pr, qq: apply_kessler_step_sixface_jax(  # noqa: E731
        st, pr, qq, dt=DT, n=N, ng=NG, km=KM)
    a_state, a_q = f(state, press, q)
    b_state, b_q = jax.jit(f)(state, press, q)
    # XLA fusion re-associates (FMA), so jit vs eager is rounding-level,
    # not bitwise -- the HS twin's gate, 1e-13 of peak
    def _close(x, y):
        x, y = np.asarray(x), np.asarray(y)
        return np.abs(x - y).max() <= 1e-13 * max(np.abs(x).max(), 1e-300)
    assert _close(a_state["pt"], b_state["pt"])
    for x, y in zip(a_q, b_q):
        assert _close(x, y)


def test_bridge_refuses_fewer_than_three_tracers(duo_bundle):
    state, press, q = duo_bundle
    with pytest.raises(ValueError, match="Kessler needs 3 tracers"):
        apply_kessler_step_sixface_jax(state, press, q[:2], dt=DT, n=N,
                                       ng=NG, km=KM)
    assert KESSLER_TRACER_SLOTS == ("q_v", "q_c", "q_r")


def test_dry_columns_are_a_no_op(duo_bundle):
    """No vapour, no cloud, no rain -> zero tendency everywhere (a sign or
    layout slip would show up as spurious heating on a dry column)."""
    state, press, _ = duo_bundle
    zeros = [jnp.zeros((6, M, M, KM)) for _ in range(3)]
    out_state, out_q = apply_kessler_step_sixface_jax(
        state, press, zeros, dt=DT, n=N, ng=NG, km=KM)
    assert np.array_equal(np.asarray(out_state["pt"]), state["pt"])
    for x in out_q:
        assert not np.asarray(x).any()
