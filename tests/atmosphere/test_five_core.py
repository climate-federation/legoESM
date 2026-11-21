"""FIVE core (Yamaguchi et al. 2017), ported from Isma Di Carlo's ENSO fork
without isotope passengers and route B: grid nesting, the exchange operators,
the column selection, the moment remap on switch, the wrapper's merge, and the
three RULE 5 tests of the smoothed switches (hard limit = the reference's
code, kept verbatim below; stored per-case gaps at the default widths;
gradients through the switches)."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics import five  # noqa: E402
from legoesm.grids.vertical import create_hybrid_coordinate, make_cam6_l32_levels  # noqa: E402

NCOL = 4
REL = 1e-12
HARD = five.FiveConfig(enabled=True, spread_width=0.0, split_width_kg_kg=0.0,
                       cap_width=0.0, empty_width_kg_kg=0.0)
DEF = five.FiveConfig(enabled=True)


def _host():
    h = make_cam6_l32_levels()
    return create_hybrid_coordinate(h.n_levels, np.asarray(h.A_half, np.float64),
                                    np.asarray(h.B_half, np.float64), p_ref=h.p_ref,
                                    dtype=jnp.float64)


@pytest.fixture(scope="module")
def setup():
    hc = _host()
    cfg = five.FiveConfig(enabled=True)
    G = five.build_five_grid(hc, cfg)
    p_s = jnp.asarray(np.linspace(96000.0, 102500.0, NCOL))
    ph = hc.pressure_at_half(p_s)
    pf = G.coord.pressure_at_half(p_s)
    dpf = G.coord.layer_thickness_dp(p_s)
    dph = hc.layer_thickness_dp(p_s)
    return dict(hc=hc, G=G, cfg=cfg, p_s=p_s, ph=ph, pf=pf, dpf=dpf, dph=dph)


def _rng(seed, shape, lo=0.0, hi=1.0):
    return jnp.asarray(np.random.default_rng(seed).uniform(lo, hi, shape))


def test_grid_nests_and_refines_only_below_742(setup):
    G, hc = setup["G"], setup["hc"]
    np.testing.assert_array_equal(np.asarray(G.coord.A_half)[G.host_iface], np.asarray(hc.A_half))
    np.testing.assert_array_equal(np.asarray(G.coord.B_half)[G.host_iface], np.asarray(hc.B_half))
    ph_ref = np.asarray(hc.A_half + hc.B_half) * hc.p_ref
    assert np.all(G.n_sub[ph_ref[:-1] < setup["cfg"].p_fine_top_pa] == 1)
    assert np.all(G.n_sub[ph_ref[:-1] >= setup["cfg"].p_fine_top_pa] > 1)
    # n_K = ceil(dz_K / target): the mean sublayer depth of each refined layer
    # is at most the target in the standard atmosphere (equal-dp sublayers
    # are slightly deeper at the top of a layer, as in the RF02 column)
    zH = five._std_height(ph_ref, hc.p_ref)
    ref = G.n_sub > 1
    assert np.all(((zH[:-1] - zH[1:]) / G.n_sub)[ref] <= setup["cfg"].dz_target_m)
    # every host interface is a fine interface at any surface pressure
    np.testing.assert_allclose(np.asarray(setup["pf"])[:, G.host_iface], np.asarray(setup["ph"]),
                               rtol=0, atol=1e-9)



def test_f1_exchange_closes_layer_means_and_columns(setup):
    G, ph, pf, dpf = setup["G"], setup["ph"], setup["pf"], setup["dpf"]
    nH, nf = G.n_sub.size, G.owner.size
    q_f = _rng(1, (NCOL, nf), 1e-4, 2e-2)
    inc = _rng(2, (NCOL, nH), -1e-4, 1e-4)
    inc = jnp.maximum(inc, -0.5 * five.to_host(q_f, dpf, G))
    for cfg in (HARD, DEF):
        d, _ = five.exchange_bulk(inc, q_f, ph, pf, dpf, G, cfg)
        back = five.to_host(d, dpf, G)
        assert float(jnp.max(jnp.abs(back - inc) / jnp.max(jnp.abs(inc)))) < REL
        col_f = jnp.sum(d * dpf, 1)
        col_h = jnp.sum(inc * (dpf @ G.onehot), 1)
        assert float(jnp.max(jnp.abs(col_f - col_h) / jnp.sum(jnp.abs(d) * dpf, 1))) < REL
        tau = _rng(3, (NCOL, nH), -1e-3, 1e-3)
        dT = five.spread(tau, ph, pf, dpf, G, cfg.spread_width)
        assert float(jnp.max(jnp.abs(five.to_host(dT, dpf, G) - tau))) < 1e-15


def test_f1_sync_makes_the_host_the_layer_mean(setup):
    G, ph, pf, dpf, dph = setup["G"], setup["ph"], setup["pf"], setup["dpf"], setup["dph"]
    nH, nf = G.n_sub.size, G.owner.size
    fine = {"T": _rng(4, (NCOL, nf), 270, 300), "u": _rng(5, (NCOL, nf), -5, 5),
            "q_v": _rng(6, (NCOL, nf), 1e-3, 1.5e-2), "q_c": _rng(7, (NCOL, nf), 5e-5, 1e-4)}
    host = {k: five.to_host(v, dpf, G) + _rng(8 + i, (NCOL, nH), -1e-5, 1e-5)
            for i, (k, v) in enumerate(fine.items())}
    out = five.sync_increment(fine, host, ph, pf, dpf, G, DEF)
    for k in fine:
        e = jnp.max(jnp.abs(five.to_host(out[k], dpf, G) - host[k]) / jnp.max(jnp.abs(host[k])))
        assert float(e) < REL, k
    W_f = jnp.sum((out["q_v"] + out["q_c"]) * dpf, 1)
    W_h = jnp.sum((host["q_v"] + host["q_c"]) * dph, 1)
    assert float(jnp.max(jnp.abs(W_f - W_h) / W_h)) < 1e-13
    H_f = jnp.sum((constants.c_pd * out["T"] + constants.L_v * out["q_v"]) * dpf, 1)
    H_h = jnp.sum((constants.c_pd * host["T"] + constants.L_v * host["q_v"]) * dph, 1)
    assert float(jnp.max(jnp.abs(H_f - H_h) / H_h)) < 1e-13


def test_cap_shortfall_carried_exactly_to_the_next_sync(setup):
    """Where the cap binds (hard widths), the shortfall stays in host minus
    A(fine) and the next sync exchanges it."""
    G, ph, pf, dpf = setup["G"], setup["ph"], setup["pf"], setup["dpf"]
    nH, nf = G.n_sub.size, G.owner.size
    q = _rng(80, (NCOL, nf), 1e-3, 2e-2)
    X = five.to_host(q, dpf, G)
    k = jnp.arange(nH) == nH - 2
    S = jnp.where(k, -1.5, -0.1) * X
    d, _ = five.exchange_bulk(S, q, ph, pf, dpf, G, HARD)
    q1 = q + d
    short = (X + S) - five.to_host(q1, dpf, G)
    np.testing.assert_allclose(np.asarray(short[:, k]), np.asarray(-0.5 * X[:, k]), rtol=1e-12)
    assert float(jnp.max(jnp.abs(short[:, ~k]) / X[:, ~k])) < 1e-14
    host2 = X + S + jnp.where(k, 2.0, 0.0) * X
    out = five.sync_increment({"q_v": q1}, {"q_v": host2}, ph, pf, dpf, G, HARD)
    e = jnp.abs(five.to_host(out["q_v"], dpf, G) - host2) / jnp.max(jnp.abs(host2))
    assert float(jnp.max(e)) < 1e-12


def test_replay_caps_on_the_increment_over_dt_not_the_rate(setup):
    G, ph, pf, dpf = setup["G"], setup["ph"], setup["pf"], setup["dpf"]
    dt = 1800.0
    q = _rng(90, (NCOL, G.owner.size), 1e-3, 2e-2)
    rate = -2.0 * five.to_host(q, dpf, G) / dt
    out = five.replay_tendencies({"q_v": rate}, {"q_v": q}, ph, pf, dpf, G, dt, HARD)
    assert bool(jnp.all(jnp.abs(q + dt * out["q_v"]) <= 1e-15 * q))


def test_forced_mode_toggles_every_period():
    cfg = five.FiveConfig(enabled=True, mode="forced")
    s = five.init_selection(3, cfg)
    states = []
    for _ in range(int(24 * 3600 / 1800)):
        s, _, _ = five.update_selection(s, 1800.0, cfg)
        states.append(bool(s["active"][0]))
    exp = [(((k + 1) * 0.5) // 6) % 2 == 0 for k in range(len(states))]
    assert states == exp


def test_moments_round_trip_and_spread(setup):
    G, ph, pf, dpf = setup["G"], setup["ph"], setup["pf"], setup["dpf"]
    nH, nf = G.n_sub.size, G.owner.size
    names = five._moment_names()
    mH = _rng(17, (NCOL, len(names), nH + 1), 0.1, 1.0)
    zt = [i for i, nm in enumerate(names) if nm in five._ZT_MOMENTS]
    mH = mH.at[:, zt, nH].set(0.0)
    mf = five.moments_to_fine(mH, ph, pf, G)
    uni = {k: jnp.ones((NCOL, nf)) for k in ("thl", "rt", "u", "v")}
    back = five.moments_to_host(mf, uni, dpf, G)
    # thermodynamic-level moments return exactly; momentum levels at the top and surface too
    np.testing.assert_allclose(np.asarray(back[:, zt, :]), np.asarray(mH[:, zt, :]), rtol=1e-13)
    zm = [i for i in range(len(names)) if i not in zt]
    np.testing.assert_allclose(np.asarray(back[:, zm][:, :, [0, nH]]),
                               np.asarray(mH[:, zm][:, :, [0, nH]]), rtol=1e-13)
    # a uniform field has no between-sublayer variance; a varying one adds it
    rng = {k: (_rng(18, (NCOL, nf), 290, 300) if k == "thl" else v) for k, v in uni.items()}
    b2 = five.moments_to_host(mf, rng, dpf, G)
    i = names.index("thlp2")
    assert float(jnp.min(b2[:, i, 1:nH] - back[:, i, 1:nH])) >= -1e-12
    assert float(jnp.max(b2[:, i, 1:nH] - back[:, i, 1:nH])) > 0.0
    j = names.index("rtp2")
    np.testing.assert_array_equal(np.asarray(b2[:, j]), np.asarray(back[:, j]))




def test_config_validation_refuses_unknown_mode_and_bad_widths():
    with pytest.raises(ValueError):
        five.validate_five_config(five.FiveConfig(mode="route_b"))
    for bad in (-1.0, float("nan"), True):
        with pytest.raises(ValueError):
            five.validate_five_config(five.FiveConfig(cap_width=bad))


# ---------------------------------------------------------------- wrapper
class _F(NamedTuple):
    data: jax.Array

    def replace(self, data):
        return _F(data)


class _State(NamedTuple):
    T: _F
    u: _F
    v: object
    tracers: dict
    p_s: _F


class _Tend(NamedTuple):
    dT_dt: _F
    du_dt: _F
    dv_dt: object
    tracer_tendencies: object


def _wrapper_case(setup, mode):
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    G, hc = setup["G"], setup["hc"]
    nH, nf = G.n_sub.size, G.owner.size
    config = PhysicsConfig()
    dT_fine = _rng(60, (NCOL, nf), -1e-4, 1e-4)

    def host_fn(state, grid, coord, ps, forcing):
        z = jnp.zeros((NCOL, nH))
        return _Tend(_F(z), _F(z), None, {"q_v": _F(z)}), ps

    def fine_fn(state, grid, coord, ps, forcing):
        z = jnp.zeros((NCOL, nf))
        return _Tend(_F(dT_fine), _F(z), None, {"q_v": _F(z)}), ps

    cfg = five.FiveConfig(enabled=True, mode=mode, forced_period_s=600.0)
    fn = five.make_five_physics(host_fn, fine_fn, None, G, cfg, config, 300.0)
    state = _State(_F(_rng(61, (NCOL, nH), 270, 300)), _F(jnp.zeros((NCOL, nH))), None,
                   {"q_v": _F(_rng(62, (NCOL, nH), 1e-3, 1e-2))}, _F(setup["p_s"]))
    ps = init_physics_state(ncol=NCOL, nlev=nH, physics_config=config, dtype=jnp.float64)
    ps = five.five_carry_seed(fn, state, hc, ps)
    return fn, state, ps, five.to_host(dT_fine, setup["dpf"], G)


def test_wrapper_takes_the_layer_mean_of_the_fine_tendencies(setup):
    fn, state, ps, want = _wrapper_case(setup, "always")
    t, ps1 = fn(state, None, setup["hc"], ps)
    np.testing.assert_allclose(np.asarray(t.dT_dt.data), np.asarray(want), rtol=1e-13, atol=1e-20)
    assert ps1.five_carry is not None


def test_wrapper_keeps_the_host_tendencies_when_inactive(setup):
    fn, state, ps, _ = _wrapper_case(setup, "forced")
    t, ps1 = fn(state, None, setup["hc"], ps)            # 300 s: on
    assert float(jnp.max(jnp.abs(t.dT_dt.data))) > 0.0
    t, ps1 = fn(state, None, setup["hc"], ps1)           # 600 s: off
    np.testing.assert_array_equal(np.asarray(t.dT_dt.data), 0.0)


# ---------------------------------------------------------------- RULE 5
def _ref_spread(tau_H, ph_H, pf_f, dp_f, G):
    """The reference fork's spread, verbatim (hard switches)."""
    shape = jax.vmap(five._sz98_column)(tau_H, ph_H, pf_f)
    m = five.to_host(shape, dp_f, G)
    wrong = ((shape * five.expand(tau_H, G)) < 0.0).astype(shape.dtype) @ G.onehot
    ok = (wrong == 0.0) & (m * tau_H > 0.0)
    m_safe = jnp.where(ok, m, 1.0)
    return jnp.where(five.expand(ok, G), shape * five.expand(tau_H / m_safe, G),
                     five.expand(tau_H, G))


def _ref_exchange(inc_H, x_f, ph_H, pf_f, dp_f, G):
    """The reference fork's exchange_bulk, verbatim (hard switches)."""
    S = jnp.minimum(inc_H, 0.0)
    P = inc_H - S
    X = five.to_host(x_f, dp_f, G)
    has = X > 0.0
    f = jnp.where(has, S / jnp.where(has, X, 1.0), 0.0)
    f_eff = jnp.where(f < -1.0, -1.0, f)
    return five.expand(f_eff, G) * x_f + _ref_spread(P, ph_H, pf_f, dp_f, G)


def _cases(setup):
    G, dpf = setup["G"], setup["dpf"]
    nH, nf = G.n_sub.size, G.owner.size
    q = _rng(70, (NCOL, nf), 1e-3, 2e-2)
    X = five.to_host(q, dpf, G)
    q_thin = q.at[:, G.owner == nH - 1].set(4e-10)      # near-empty bottom layer
    return {
        "mixed_sign": (_rng(71, (NCOL, nH), -1e-4, 1e-4), q),
        "source_only": (_rng(72, (NCOL, nH), 1e-6, 1e-4), q),
        "cap_binds": (jnp.where(jnp.arange(nH) == nH - 2, -1.5, -0.1) * X, q),
        "cap_edge": (-0.995 * X, q),
        "tiny_increment": (_rng(73, (NCOL, nH), -2e-9, 2e-9), q),
        "empty_layer": (_rng(74, (NCOL, nH), -1e-5, 1e-5), q_thin),
    }


def test_r5_hard_widths_are_the_reference_bitwise(setup):
    G, ph, pf, dpf = setup["G"], setup["ph"], setup["pf"], setup["dpf"]
    for name, (inc, q) in _cases(setup).items():
        np.testing.assert_array_equal(
            np.asarray(five.exchange_bulk(inc, q, ph, pf, dpf, G, HARD)[0]),
            np.asarray(_ref_exchange(inc, q, ph, pf, dpf, G)), err_msg=name)
        np.testing.assert_array_equal(
            np.asarray(five.spread(inc, ph, pf, dpf, G, 0.0)),
            np.asarray(_ref_spread(inc, ph, pf, dpf, G)), err_msg=name)


# per case: max |default - reference| of the fine water increment over
# max |reference|, measured 2026-11-21 at the voted default widths
_GAP = {"cap_binds": 0.0, "cap_edge": 0.0012562814070352863, "empty_layer": None,
        "mixed_sign": 0.0, "source_only": 0.0, "tiny_increment": 0.05623059957619883}


@pytest.mark.parametrize("name", ["cap_binds", "cap_edge", "empty_layer", "mixed_sign",
                                  "source_only", "tiny_increment"])
def test_r5_default_gap_per_case(setup, name):
    G, ph, pf, dpf = setup["G"], setup["ph"], setup["pf"], setup["dpf"]
    inc, q = _cases(setup)[name]
    ref = _ref_exchange(inc, q, ph, pf, dpf, G)
    got = five.exchange_bulk(inc, q, ph, pf, dpf, G, DEF)[0]
    gap = float(jnp.max(jnp.abs(got - ref)) / jnp.max(jnp.abs(ref)))
    if _GAP[name] is None:
        pytest.fail(f"MEASURED {name!r}: {gap!r}")
    np.testing.assert_allclose(gap, _GAP[name], rtol=1e-6, atol=1e-12)


def test_r5_gradients_through_the_switches(setup):
    G = setup["G"]
    nH, nf = G.n_sub.size, G.owner.size
    ph, pf, dpf = setup["ph"][:1], setup["pf"][:1], setup["dpf"][:1]
    q0 = _rng(15, (1, nf), 0.1, 2.0)
    X0 = five.to_host(1e-2 * q0, dpf, G)
    # increments on every switch: mixed signs, one layer inside the cap band
    inc0 = (jnp.sin(jnp.arange(nH, dtype=jnp.float64)) * 1e-3
            + jnp.where(jnp.arange(nH) == nH - 2, -0.995, 0.0) * X0[0] / 1e-5)[None, :]

    def f(inc, q):
        d, _ = five.exchange_bulk(1e-5 * inc, 1e-2 * q, ph, pf, dpf, G, DEF)
        return 1e5 * jnp.sum(d * jnp.cos(jnp.arange(nf)))

    # eps well inside the 1e-9 kg/kg split band (inc is scaled by 1e-5)
    check_grads(f, (inc0, q0), order=1, modes=("rev",), eps=1e-6, atol=1e-5, rtol=1e-5)
    g = jax.grad(f)(inc0, q0)
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.max(jnp.abs(g))) > 0.0


def test_r5_default_widths_are_the_config_defaults():
    d = five.FiveConfig()
    assert (d.spread_width, d.split_width_kg_kg, d.cap_width, d.empty_width_kg_kg) == (
        0.01, 1.0e-9, 0.01, 1.0e-9)
    assert d.enabled is False
