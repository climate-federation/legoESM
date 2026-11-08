"""PR 3: melting/internal-melt switches, cascade MIN caps and ice_thd_do (votes Z42-Z49).

Smooth mode keeps NEMO's forward value at every site (straight-through, user ruling 3a
pattern) and takes the gradient from a C1 surrogate.  Checked here on tier-4 fixtures
(NEMO 5.0.1 SI3 ORCA1, kt=17281) with the inputs moved inside each switch's band:
* the gradient equals the central FD of the soft surrogate (straight_through -> soft);
* it differs from the hard gradient (the surrogate term is present: non-vacuous);
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


def _check(monkeypatch, f, x0, width):
    g = float(jax.grad(f)(jnp.asarray(x0)))
    eps = 1e-6 * width                                   # Z35 convention
    fd = (float(_with(monkeypatch, "soft", f, x0 + eps)) - float(_with(monkeypatch, "soft", f, x0 - eps))) / (2 * eps)
    g_hard = float(_with(monkeypatch, "hard", jax.grad(f), jnp.asarray(x0)))
    assert np.isfinite(g)
    np.testing.assert_allclose(g, fd, rtol=1e-4)
    assert abs(g - g_hard) > 1e-3 * max(abs(g_hard), 1e-30), (g, g_hard)    # surrogate term present


def test_do_step_open_water_switch_gradient_is_the_surrogate(monkeypatch):
    st, _, fz = dh_args(load("c9_exchange"))

    def f(q):
        out, fx = bl._do_step(st, fz._replace(qlead=jnp.full_like(fz.qlead, q)), dt(load("c9_exchange")), K,
                              nemo_version="5.0.1", switches=SMOOTH)
        return jnp.sum(out.h_ice * out.concentration) + jnp.sum(fx.hfx_opw)
    _check(monkeypatch, f, -0.5 * WF, WF)


def test_dh_surface_melt_switch_gradient_is_the_surrogate(monkeypatch):
    fx = load("c4_fullmelt")
    st, zr, fz = dh_args(fx)

    def f(t):
        z = zr._replace(T_surface=jnp.full_like(zr.T_surface, t))
        out, fl = bl._dh_step_fluxes(st, z, fz, dt(fx), K, nemo_version="5.0.1", switches=SMOOTH)
        return jnp.sum(out.h_ice) + jnp.sum(out.h_snow)
    _check(monkeypatch, f, K.T0 - 0.5 * TM, TM)


def test_dh_internal_melt_switch_gradient_is_the_surrogate(monkeypatch):
    fx = load("c4_fullmelt")
    st, zr, fz = dh_args(fx)
    melt = K.T0 - K.liquidus_slope * st.S_layers

    def f(d):
        z = zr._replace(T_ice=melt - d)
        out, fl = bl._dh_step_fluxes(st, z, fz, dt(fx), K, nemo_version="5.0.1", switches=SMOOTH)
        return jnp.sum(fl.hfx_res) + jnp.sum(fl.hfx_sum) + jnp.sum(fl.hfx_bom)
    _check(monkeypatch, f, 0.5 * TM, TM)


def test_soft_column_dh_budgets_close_inside_the_bands(monkeypatch):
    """Z42-Z49 closure: the soft column's dh mass, heat and salt identities hold with
    every switch inside its band (fractional surrogate weights)."""
    monkeypatch.setattr(bl, "straight_through", lambda h, s: s)
    checked = 0
    for name in ("c4_fullmelt", "c6_melt_order", "c9_exchange", "c3_qcn_top_k"):
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
        np.testing.assert_allclose(dq, rhs, rtol=0, atol=1e-11 * max(1.0, np.abs(dq).max()))
        checked += 1
    monkeypatch.undo()
    assert checked == 4


def test_zdf_melting_row_switch_gradient_is_finite_and_the_surrogate(monkeypatch):
    fx = load("c4_fullmelt")
    e_i, e_s, sz, h_i, h_s, t_su, fz = zdf_args(fx)

    def f(t):
        r = bl._si3_zdf_bl99_step(e_i, e_s, sz, h_i, h_s, jnp.full_like(t_su, t), fz, dt(fx), K,
                                  nemo_version="5.0.1", switches=SMOOTH._replace(snow_ramp_width_m=0.0))
        return jnp.sum(r.qcn_ice_top)
    _check(monkeypatch, f, K.T0 - 0.5 * TM, TM)
