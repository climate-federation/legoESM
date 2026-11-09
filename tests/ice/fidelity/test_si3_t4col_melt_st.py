"""PR 3: melting/internal-melt switches, cascade MIN caps and ice_thd_do (votes Z42-Z49).

Smooth mode keeps NEMO's forward value at every site (straight-through, user ruling 3a
pattern) and takes the gradient from a C1 surrogate.  Checked here on tier-4 fixtures
(NEMO 5.0.1 SI3 ORCA1, kt=17281) with the inputs moved inside each switch's band:
* each helper's forward is the hard value bitwise and its derivative the soft one;
* where one site feeds the output, the gradient equals the central FD of the soft
  surrogate; through chains of sites (AD linearises downstream hard ops at the hard
  primal, vote Z51) the gradient is finite and differs from the hard gradient;
* the soft column's mass, heat and salt identities of ice_thd_dh still close.
Forward bitwise equality with the pre-PR code is the literal-row gate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice.config import SI3SwitchConfig
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as K

from si3_t4col_fixture import dh_args, dt, load, zdf_args

jax.config.update("jax_enable_x64", True)
SMOOTH = SI3SwitchConfig()
TM = SMOOTH.relative_width * SMOOTH.temperature_scale_K * SMOOTH.melt_point_scale_K   # 1e-3 K (Z45)
WF = SMOOTH.relative_width * SMOOTH.flux_scale_W_m2                                    # 0.01 W/m2 (Z46)


@pytest.fixture(autouse=True)
def _release_executables():
    yield
    jax.clear_caches()     # vote Z41


def _with(monkeypatch, mode, f, x):
    """Evaluate f(x) with straight_through returning its soft or hard argument."""
    monkeypatch.setattr(bl, "straight_through", (lambda h, s: s) if mode == "soft" else (lambda h, s: h))
    try:
        return f(x)
    finally:
        monkeypatch.undo()


def _grads(monkeypatch, f, x0):
    g = float(jax.grad(f)(jnp.asarray(x0)))
    g_hard = float(_with(monkeypatch, "hard", jax.grad(f), jnp.asarray(x0)))
    assert np.isfinite(g)
    assert abs(g - g_hard) > 1e-3 * max(abs(g_hard), 1e-30), (g, g_hard)    # surrogate term present
    return g


def _check_single(monkeypatch, f, x0, width):
    """One straight-through site feeds the output: gradient == FD of the soft surrogate."""
    g = _grads(monkeypatch, f, x0)
    eps = 1e-6 * width                                   # Z35 convention
    fd = (float(_with(monkeypatch, "soft", f, x0 + eps)) - float(_with(monkeypatch, "soft", f, x0 - eps))) / (2 * eps)
    np.testing.assert_allclose(g, fd, rtol=1e-4)


def test_helpers_are_hard_forward_soft_derivative():
    x = jnp.asarray([0.2, 0.4999, 0.5, 0.5003, 0.9])
    y, w = jnp.full_like(x, 0.5), 1e-3
    for st_f, hard, soft in ((bl._st_min, jnp.minimum, lambda a, b: bl.min_c1(a, b, 1.0 / w)),
                             (bl._st_max, jnp.maximum, lambda a, b: bl.max_c1(a, b, 1.0 / w))):
        np.testing.assert_array_equal(np.asarray(st_f(x, y, w)), np.asarray(hard(x, y)))
        np.testing.assert_array_equal(np.asarray(st_f(x, y, 0.0)), np.asarray(hard(x, y)))
        g = jax.vmap(jax.grad(lambda a, b: st_f(a, b, w)))(x, y)
        gs = jax.vmap(jax.grad(soft))(x, y)
        np.testing.assert_array_equal(np.asarray(g), np.asarray(gs))
    wt = bl.smoothstep_cubic((x - 0.5 + w) / w)
    a, b = 3.0 * x, -x
    np.testing.assert_array_equal(np.asarray(bl._st_where(x >= 0.5, a, b, wt)), np.asarray(jnp.where(x >= 0.5, a, b)))
    g = jax.grad(lambda z: jnp.sum(bl._st_where(z >= 0.5, 3.0 * z, -z, bl.smoothstep_cubic((z - 0.5 + w) / w))))(x)
    gs = jax.grad(lambda z: jnp.sum(-z + (3.0 * z + z) * bl.smoothstep_cubic((z - 0.5 + w) / w)))(x)
    np.testing.assert_allclose(np.asarray(g), np.asarray(gs), rtol=1e-13)


def test_do_step_open_water_switch_gradient_is_the_surrogate(monkeypatch):
    fx = load("c9_exchange")
    st, _, fz = dh_args(fx)

    def run(q):
        return bl._do_step(st, fz._replace(qlead=jnp.full_like(fz.qlead, q)), dt(fx), K,
                           nemo_version="5.0.1", switches=SMOOTH)
    _check_single(monkeypatch, lambda q: jnp.sum(run(q)[1].hfx_opw), -0.5 * WF, WF)     # zfmdt only
    _grads(monkeypatch, lambda q: jnp.sum(run(q)[0].h_ice * run(q)[0].concentration), -0.5 * WF)


def test_dh_surface_melt_switch_gradient_is_the_surrogate(monkeypatch):
    """Z54: on c4_fullmelt the layer caps bind outside their band (both gradients 0), so
    the check runs on three fixtures and must bite on at least one."""
    bit = 0
    for name in ("c3_qcn_top_k", "c6_melt_order", "c9_exchange"):
        fx = load(name)
        st, zr, fz = dh_args(fx)
        # Net surface melt flux = flux_scale (1 W/m2) in every cell: positive, and small
        # enough that the layer caps do not bind (1000 W/m2 melted whole columns).
        fz = fz._replace(qsr_ice=SMOOTH.flux_scale_W_m2 - zr.qns_ice + fz.qtr_ice_top + zr.qcn_ice_top)
        net = np.asarray(zr.qns_ice + fz.qsr_ice - fz.qtr_ice_top - zr.qcn_ice_top)
        assert np.all(net > 0), "precondition: no positive surface melt flux"

        def f(t, st=st, zr=zr, fz=fz, fx=fx):
            z = zr._replace(T_surface=jnp.full_like(zr.T_surface, t))
            out, fl = bl._dh_step_fluxes(st, z, fz, dt(fx), K, nemo_version="5.0.1", switches=SMOOTH)
            return jnp.sum(out.h_ice) + jnp.sum(out.h_snow)
        x0 = jnp.asarray(K.T0 - 0.5 * TM)
        g = float(jax.grad(f)(x0))
        g_hard = float(_with(monkeypatch, "hard", jax.grad(f), x0))
        assert np.isfinite(g)
        bit += int(g != 0.0 and abs(g - g_hard) > 1e-3 * max(abs(g_hard), 1e-30))
    assert bit >= 1, "surrogate term absent on every fixture (vacuous)"


def test_dh_internal_melt_switch_gradient_is_the_surrogate(monkeypatch):
    fx = load("c4_fullmelt")
    st, zr, fz = dh_args(fx)
    melt = K.T0 - K.liquidus_slope * st.S_layers

    def f(d):
        z = zr._replace(T_ice=melt - d)
        out, fl = bl._dh_step_fluxes(st, z, fz, dt(fx), K, nemo_version="5.0.1", switches=SMOOTH)
        return jnp.sum(fl.hfx_res) + jnp.sum(fl.hfx_sum) + jnp.sum(fl.hfx_bom)
    _grads(monkeypatch, f, 0.5 * TM)


def test_soft_column_dh_budgets_close_inside_the_bands(monkeypatch):
    """Z42-Z49 closure: the soft column's dh mass, heat and salt identities hold with
    every switch inside its band (fractional surrogate weights)."""
    monkeypatch.setattr(bl, "straight_through", lambda h, s: s)
    checked = 0
    for name in ("c4_fullmelt", "c6_melt_order", "c9_exchange", "c3_qcn_top_k", "c5_flood_sst"):
        fx = load(name)
        st, zr, fz = dh_args(fx)
        melt = K.T0 - K.liquidus_slope * st.S_layers
        zr = zr._replace(T_surface=jnp.full_like(zr.T_surface, K.T0 - 0.5 * TM), T_ice=melt - 0.5 * TM)
        out, f = bl._dh_step_fluxes(st, zr, fz, dt(fx), K, nemo_version="5.0.1", switches=SMOOTH)
        a, a1, d = np.asarray(st.concentration), np.asarray(out.concentration), dt(fx)
        mass = (a1 * (K.rho_ice * out.h_ice + K.rho_snow * out.h_snow)
                - a * (K.rho_ice * st.h_ice + K.rho_snow * st.h_snow)) / d
        wsum = sum(np.asarray(getattr(f, k)) for k in bl.SI3ExchangeFluxes._fields
                   if k.startswith("wfx") and k not in ("wfx_err_sub", "wfx_opw"))
        np.testing.assert_allclose(np.asarray(mass), -wsum, rtol=0, atol=1e-11 * max(1.0, np.abs(mass).max()))
        q = lambda s_: (np.sum(np.asarray(s_.e_ice), -1) * np.asarray(s_.h_ice)  # noqa: E731
                        + np.sum(np.asarray(s_.e_snow), -1) * np.asarray(s_.h_snow)) / 3.0
        g = lambda k: np.asarray(getattr(f, k))  # noqa: E731
        rhs = (g("hfx_thd") + g("hfx_res") + g("hfx_sub") + g("hfx_spr")
               - g("hfx_sum") - g("hfx_bom") - g("hfx_bog") - g("hfx_snw"))
        dq = (a1 * q(out) - a * q(st)) / d
        # Soft column: 1.6e-10 relative measured (flood telescoping, mechanism open), vote Z53.
        np.testing.assert_allclose(dq, rhs, rtol=0, atol=1e-9 * max(1.0, np.abs(dq).max()))
        salt = K.rho_ice * (a1 * np.asarray(out.h_ice * out.S_bulk) - a * np.asarray(st.h_ice * st.S_bulk)) / d
        ssum = sum(np.asarray(getattr(f, k)) for k in bl.SI3ExchangeFluxes._fields if k.startswith("sfx"))
        np.testing.assert_allclose(salt, -ssum, rtol=0, atol=1e-11 * max(1.0, np.abs(salt).max()))   # Z58
        checked += 1
    monkeypatch.undo()
    assert checked == 5


def test_zdf_melting_row_switch_gradient_is_finite_and_the_surrogate(monkeypatch):
    fx = load("c4_fullmelt")
    e_i, e_s, sz, h_i, h_s, t_su, fz = zdf_args(fx)

    def f(t):
        r = bl._si3_zdf_bl99_step(e_i, e_s, sz, h_i, h_s, jnp.full_like(t_su, t), fz, dt(fx), K,
                                  nemo_version="5.0.1", switches=SMOOTH._replace(snow_ramp_width_m=0.0))
        return jnp.sum(r.qcn_ice_top)
    _grads(monkeypatch, f, K.T0 - 0.5 * TM)


def test_soft_column_do_budgets_close_inside_the_bands(monkeypatch):
    """Z56-Z58: the soft ice_thd_do column keeps the hard step's mass, salt, heat and snow
    identities (tolerances of test_si3_t4col_thd_do) with the open-water and area-cap
    switches inside their bands."""
    from test_si3_t4col_thd_do import DT, _state
    st, fz = _state()
    c = float(K.rho_ice)
    v_unit = -np.asarray(bl._do_step(st, fz._replace(qlead=-jnp.ones_like(fz.qlead)), DT, K,
                                     nemo_version="5.0.1")[1].wfx_opw) * DT / c          # vnew per -1 W/m2
    room = np.maximum(0.0, bl._MAXIMUM_ICE_CONCENTRATION - np.asarray(st.concentration))
    w_cut = SMOOTH.relative_width * SMOOTH.cut_fraction_scale
    q_cap = -(room + 0.5 * w_cut) * bl._NEW_ICE_THICKNESS / v_unit
    w_ice = SMOOTH.relative_width * SMOOTH.ice_switch_scale
    thin = st._replace(concentration=jnp.full_like(st.concentration, 0.5 * w_ice))   # had/keep band (Z61)
    run = lambda s_, q: bl._do_step(s_, fz._replace(qlead=q), DT, K, nemo_version="5.0.1", switches=SMOOTH)  # noqa: E731
    hard_cap = _with(monkeypatch, "hard", lambda q: run(st, q)[0].h_ice, jnp.asarray(q_cap))
    soft_cap = _with(monkeypatch, "soft", lambda q: run(st, q)[0].h_ice, jnp.asarray(q_cap))
    assert np.all(np.asarray(hard_cap) != np.asarray(soft_cap))       # cap band hit in every cell
    every = np.ones(len(room), bool)
    # Thin arm: only the cells ice_thd_do acts on; below epsi10 the hard step itself
    # zeroes h, a pre-existing behaviour outside PR 3 (vote Z62).
    arms = ((st, jnp.full_like(fz.qlead, -0.5 * WF), every), (st, jnp.asarray(q_cap), every),
            (thin, fz.qlead, np.asarray(fz.qlead) < 0.0))
    for s_, q, _ in arms:      # Z62: real straight_through gradient finite in every arm
        g = jax.grad(lambda q_: jnp.sum(run(s_, q_)[0].h_ice * run(s_, q_)[0].concentration
                                         * (1.0 + jnp.sum(run(s_, q_)[0].e_ice, -1))))(q)
        assert np.all(np.isfinite(np.asarray(g)))
    monkeypatch.setattr(bl, "straight_through", lambda h, s: s)
    for s_, q, m in arms:
        o, fx = run(s_, q)
        a0, a1 = np.asarray(s_.concentration)[m], np.asarray(o.concentration)[m]
        v0, v1 = np.asarray(s_.h_ice)[m] * a0, np.asarray(o.h_ice)[m] * a1
        g_ = lambda k: np.asarray(getattr(fx, k))[m]  # noqa: E731
        # Tiny-flux arm differences O(1e-11 m) out of O(1 m) volumes: atol at the
        # inventory's roundoff scale (vote Z59).
        tol = lambda x: 1e-14 * c * np.abs(x).max() / DT   # noqa: E731
        np.testing.assert_allclose(c * (v1 - v0) / DT, -g_("wfx_opw"), rtol=1e-11, atol=max(1e-18, tol(v0)))
        s0, s1 = np.asarray(s_.S_bulk)[m] * v0, np.asarray(o.S_bulk)[m] * v1
        np.testing.assert_allclose(c * (s1 - s0) / DT, -g_("sfx_opw"), rtol=1e-12, atol=max(1e-16, tol(s0)))
        q0, q1 = np.sum(np.asarray(s_.e_ice)[m], -1) * v0 / 3.0, np.sum(np.asarray(o.e_ice)[m], -1) * v1 / 3.0
        np.testing.assert_allclose((q1 - q0) / DT, g_("hfx_thd") - g_("hfx_opw"), rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(np.asarray(o.h_snow)[m] * a1, np.asarray(s_.h_snow)[m] * a0, rtol=1e-14, atol=0)


def test_do_step_keep_switch_soft_pass_keeps_the_category(monkeypatch):
    """Z63/Z64: below epsi20 total area the hard step empties the category; the soft pass
    keeps its own growth (updated branch, Z61).  The band sits inside the open-water
    ramp (|qlead| ~ 1e-16 W/m2), so the soft growth carries that ramp."""
    from test_si3_t4col_thd_do import DT, _state
    st, fz = _state()
    st = st._replace(concentration=jnp.zeros_like(st.concentration))
    c = float(K.rho_ice)
    v_unit = -np.asarray(bl._do_step(st, fz._replace(qlead=-jnp.ones_like(fz.qlead)), DT, K,
                                     nemo_version="5.0.1")[1].wfx_opw) * DT / c
    q = jnp.asarray(-0.5 * bl._EPS20 * bl._NEW_ICE_THICKNESS / v_unit)
    run = lambda q_: bl._do_step(st, fz._replace(qlead=q_), DT, K, nemo_version="5.0.1",  # noqa: E731
                                 switches=SMOOTH)
    area = lambda q_: run(q_)[0].concentration  # noqa: E731
    assert np.all(np.asarray(_with(monkeypatch, "hard", area, q)) == 0.0)
    soft_area, soft_wfx = _with(monkeypatch, "soft", lambda q_: (area(q_), run(q_)[1].wfx_opw), q)
    grown = -np.asarray(soft_wfx) * DT / (c * bl._NEW_ICE_THICKNESS)
    assert np.all(grown > 0.0)
    np.testing.assert_allclose(np.asarray(soft_area), grown, rtol=1e-12, atol=0)
    g = np.asarray(jax.grad(lambda q_: jnp.sum(area(q_)))(q))
    h = 1e-6 * np.abs(np.asarray(q))
    fd = np.array([(float(jnp.sum(_with(monkeypatch, "soft", area, q.at[k].add(h[k]))))
                    - float(jnp.sum(_with(monkeypatch, "soft", area, q.at[k].add(-h[k]))))) / (2 * h[k])
                   for k in range(q.size)])
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=0)
