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
    gm_redi: _GM | None


class _Z(NamedTuple):
    nemo_ldf_ahmt: np.ndarray
    nemo_ldf_ahmf: np.ndarray


@pytest.mark.parametrize("scheme", ["tke", "constant"])
def test_scaled_parts_is_bit_identical_at_one_and_scales_only_its_field(scheme):
    cfg = _CFG(_PH(_LM(_GM(1000.0)), _VM(scheme, _TKE(1.4e-10), _CONST(1.2e-5))),
               _BD(1e-3), _GM(500.0))
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
    redi, _ = probe.scaled_parts(cfg, z, {"kappa_redi": 2.0})
    # the step reads the top-level copy; the physics copy is never read
    assert redi.gm_redi.kappa_Redi == 1000.0
    assert redi.physics == cfg.physics
    none_cfg = cfg._replace(gm_redi=None)
    assert probe.scaled_parts(none_cfg, z, {"kappa_redi": 2.0})[0] == none_cfg
    with pytest.raises(probe.GateError):
        probe.scaled_parts(cfg._replace(physics=cfg.physics._replace(
            vertical_mixing=cfg.physics.vertical_mixing._replace(scheme="kpp"))),
            z, {})


# --- backward non-finite walker (nan_locate --backward) ----------------------

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round315_nan_locate as locate,
)

_H = jnp.asarray([1.0, 0.0, 2.0, 0.0])  # two "dry" lanes of zero thickness
_WET = _H > 0


def _masked_div_step(st):
    # the T update divides by a thickness that is 0 on dry lanes, then masks
    t_new = jnp.where(_WET, st["T"] / _H, 0.0)
    eta_new = st["eta"] + jnp.sum(st["T"])  # the SSH path never divides
    return {"T": t_new, "eta": eta_new}


def _state():
    return {"T": jnp.asarray([0.5, 0.0, 1.5, 0.0]), "eta": jnp.asarray(0.25)}


def test_walker_names_the_backward_division_and_sanitized_gradient_is_the_guarded_one():
    out, pb = jax.vjp(jax.jit(_masked_div_step), _state())
    ct = {"T": jnp.ones(4) * _WET, "eta": jnp.asarray(1.0)}
    assert not np.all(np.isfinite(np.asarray(pb(ct)[0]["T"])))  # the defect is real
    (g_raw,), ev_raw = locate.walk_pullback(pb, ct)
    assert not np.all(np.isfinite(np.asarray(g_raw["T"])))
    first = ev_raw[0]
    assert first["primitive"] == "div" and first["class"] == "generated"
    (g,), ev = locate.walk_pullback(pb, ct, sanitize=True)
    guarded = jnp.where(_WET, 1.0 / jnp.where(_WET, _H, 1.0), 0.0) + 1.0
    np.testing.assert_allclose(np.asarray(g["T"]), np.asarray(guarded), rtol=1e-15)


def test_symbolic_zero_cotangent_reproduces_the_fused_gradient():
    # SSH-only loss: fused grad is finite, the hand-chained dense-zero T
    # cotangent hits 0/0 on the dry lanes
    fused = jax.grad(lambda st: _masked_div_step(st)["eta"])(_state())
    assert np.all(np.isfinite(np.asarray(fused["T"])))
    _, pb = jax.vjp(jax.jit(_masked_div_step), _state())
    ct = {"T": jnp.zeros(4), "eta": jnp.asarray(1.0)}
    (dense,), ev_dense = locate.walk_pullback(pb, ct)
    assert not np.all(np.isfinite(np.asarray(dense["T"]))) and ev_dense
    (sym,), ev_sym = locate.walk_pullback(pb, ct, symbolic_zero_paths=("['T']",))
    np.testing.assert_array_equal(np.asarray(sym["T"]), np.asarray(fused["T"]))
    assert ev_sym == []


def test_walker_finds_a_residual_nan_inside_a_scan_and_reproduces_finite_pullbacks():
    def f(x):
        def body(c, xi):
            return c + jnp.sqrt(xi), c  # sqrt'(0) = inf: residual meets cotangent
        c, ys = jax.lax.scan(body, 0.0, x)
        return c + jnp.sum(ys)

    x = jnp.asarray([4.0, 0.0, 1.0])
    _, pb = jax.vjp(f, x)
    (g,), ev = locate.walk_pullback(pb, jnp.asarray(1.0))
    assert ev and not np.all(np.isfinite(np.asarray(g)))
    # a finite pullback (no zero) is reproduced to rounding by the walker
    _, pb2 = jax.vjp(f, jnp.asarray([4.0, 2.0, 1.0]))
    (g2,), ev2 = locate.walk_pullback(pb2, jnp.asarray(1.0))
    assert ev2 == []
    np.testing.assert_allclose(np.asarray(g2), np.asarray(pb2(jnp.asarray(1.0))[0]), rtol=1e-14)


def test_fd_sweep_defaults_reach_the_reviewed_epsilons():
    # the review measured agreement only below 1e-6 K; the sweep must reach it
    assert min(probe.SWEEP_EPS) <= 1e-7 and max(probe.SWEEP_EPS) >= 1e-3



def test_kappa_redi_scale_moves_the_traced_step_state():
    """No inert parameter: 1 % on kappa_redi changes the one-step T of a
    lat-lon model that runs isoneutral mixing."""
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_full_step_coordinate, create_ocean_z_star

    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())  # a 1 % Redi increment is below fp32's ulp of T
    try:
        grid = create_latlon_grid(n_lat=12, n_lon=8)
        z = create_ocean_z_star(n_levels=6, H_max=3000.0)
        ones = jnp.ones((grid.n_lat, grid.n_lon, z.n_levels))
        z = create_full_step_coordinate(z, bottom_level=np.full(ones.shape[:2], z.n_levels))
        # the probe also scales viscosity/vertical-mixing/drag fields: give it real ones
        z = z._replace(nemo_ldf_ahmt=ones, nemo_ldf_ahmf=ones)
        ph = OceanPhysicsConfig()
        cfg = LatLonCGridOceanConfig.from_flat(
            A_h=1.0e3, A_v=1.0e-3, K_h=0.0, K_v=1.0e-4, bottom_drag_r=1.0e-3,
            gm_redi=GMRediConfig(kappa_GM=0.0, kappa_Redi=1.0e3,
                                 slope_scheme="nemo_iso_lap"),
            outer_integrator="forward_euler")._replace(physics=ph._replace(
                lateral_mixing=ph.lateral_mixing._replace(scheme="none")))
        st = rest_state_latlon_cgrid_ocean(grid, z, T_water_init_C=20.0, T_deep=4.0,
                                           S_uniform=35.0, H_max=3000.0)
        # a meridional T front plus a zonal S contrast: T varies along isopycnals
        wet = st.land_mask.data[:, :, None]
        rows = jnp.arange(grid.n_lat, dtype=st.T.data.dtype)[:, None, None]
        cols = jnp.arange(grid.n_lon, dtype=st.S.data.dtype)[None, :, None]
        st = st._replace(T=st.T.replace(data=st.T.data + 4.0 * jnp.tanh((rows - 6.0) / 2.0) * wet),
                         S=st.S.replace(data=st.S.data + 0.5 * jnp.sin(cols) * wet))
        step = probe.traced_step(LatLonCGridOceanModel(grid, z, cfg))
        one, bump = (step(st, {"kappa_redi": jnp.float64(v)}, 900.0, None, None)
                     for v in (1.0, 1.01))
        assert np.count_nonzero(np.asarray(one.T.data) != np.asarray(bump.T.data)) > 0
    finally:
        set_policy(saved)
