"""C1: ice_thd_do mixes the new ice's enthalpy and salinity into the category and keeps the snow volume.

Main updated only area and thickness: new ice inherited the old enthalpy/salinity and the snow
thickness was kept on the grown area (snow mass manufactured in 2161 of 2214 new-ice cells).
Oracle = NEMO 5.0.1 SI3 ORCA1 tier-4 U3 record (fixture c1_thd_do: 4 snowy new-ice columns,
3 at the rn_amax cap, 1 open-water column, 4 non-freezing controls), thickness carrier as the
tier-4 literal row (h = v/a, e = e_i/v*nlay, S = sv_i/v).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as K

from si3_t4col_fixture import load, rel

jax.config.update("jax_enable_x64", True)
FX = load("c1_thd_do")
N = len(FX["cell"])
DT = float(FX["rDt_ice"])


def _state(**over):
    a, v, vs = FX["a_i"], FX["v_i"], FX["pas_v_s_pre"]
    pa, pv = a > 0, v > 0
    h = np.where(pa, v / np.where(pa, a, 1.0), 0.0)
    hs = np.where(pa, vs / np.where(pa, a, 1.0), 0.0)
    e = np.where(pv[:, None], FX["e_i"] / np.where(pv, v, 1.0)[:, None] * 3.0, 0.0)
    es = np.where((vs > 0)[:, None], FX["pas_e_s_pre"] / np.where(vs > 0, vs, 1.0)[:, None] * 3.0, 0.0)
    S = np.where(pv, FX["sv_i"] / np.where(pv, v, 1.0), 0.0)
    Sz = np.where(pv[:, None], FX["szv_i"] / np.where(pv, v, 1.0)[:, None] * 3.0, 0.0)
    st = bl.SI3ColumnArrays(*(jnp.asarray(x) for x in (a, h, hs, FX["pas_t_su_pre"], e, es, S, Sz, FX["pas_oa_i_pre"])))
    z = jnp.zeros(N)
    f = {k: z for k in bl.SI3SurfaceForcing._fields if bl.SI3SurfaceForcing._field_defaults.get(k, 0) is not None}
    f.update(t_bottom=jnp.asarray(FX["t_bo"]), sss=jnp.asarray(FX["sss_m"]), qlead=jnp.asarray(FX["qlead"]))
    f.update(over)
    return st, bl.SI3SurfaceForcing(**f)


def _globals(o, fx):
    a = np.asarray(o.concentration); v = np.asarray(o.h_ice) * a
    return {"a_i": a, "v_i": v, "sv_i": np.asarray(o.S_bulk) * v, "e_i": np.asarray(o.e_ice) * v[:, None] / 3.0,
            "szv_i": np.asarray(o.S_layers) * v[:, None] / 3.0, "v_s": np.asarray(o.h_snow) * a,
            **{f"d_{k}": np.asarray(getattr(fx, k)) for k in ("sfx_opw", "wfx_opw", "hfx_thd", "hfx_opw")}}


def test_fixture_new_ice_columns_fail_on_main():
    new = FX["qlead"] < 0
    assert new.sum() == 8 and (~new).sum() == 4 and (FX["a_i"][new] == 0).sum() == 1
    for f in ("e_i", "v_s", "d_sfx_opw"):
        assert np.all(rel(FX[f"main_{f}"], FX[f"nemo_{f}"])[new & (FX["a_i"] > 0)] > FX[f"tol_{f}"][new & (FX["a_i"] > 0)]), f


def test_new_ice_state_and_exchange_match_nemo():
    st, fz = _state()
    o, fx = bl._do_step(st, fz, DT, K)
    port = _globals(o, fx)
    bad = {f: np.flatnonzero(rel(port[f], FX[f"nemo_{f}"]) > FX[f"tol_{f}"]).tolist() for f in port}
    assert not any(bad.values()), bad


def test_mass_salt_heat_snow_budgets_close():
    st, fz = _state()
    o, fx = bl._do_step(st, fz, DT, K)
    a0, a1 = np.asarray(st.concentration), np.asarray(o.concentration)
    v0, v1 = np.asarray(st.h_ice) * a0, np.asarray(o.h_ice) * a1
    np.testing.assert_allclose(K.rho_ice * (v1 - v0) / DT, -np.asarray(fx.wfx_opw), rtol=1e-11, atol=1e-18)
    s0, s1 = np.asarray(st.S_bulk) * v0, np.asarray(o.S_bulk) * v1
    np.testing.assert_allclose(K.rho_ice * (s1 - s0) / DT, -np.asarray(fx.sfx_opw), rtol=1e-12, atol=1e-16)
    q0 = np.sum(st.e_ice, -1) * v0 / 3.0
    q1 = np.sum(o.e_ice, -1) * v1 / 3.0
    np.testing.assert_allclose((q1 - q0) / DT, np.asarray(fx.hfx_thd - fx.hfx_opw), rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(np.asarray(o.h_snow) * a1, np.asarray(st.h_snow) * a0, rtol=1e-14, atol=0)


def test_new_ice_gradient_finite_and_matches_fd():
    st, fz = _state()
    k = int(np.flatnonzero((FX["qlead"] < 0) & (FX["a_i"] > 0) & (FX["a_i"] < 0.9))[0])
    f = lambda q: jnp.sum(bl._do_step(st, fz._replace(qlead=q), DT, K)[0].e_ice[k])  # noqa: E731
    g = jax.grad(f)(fz.qlead)
    h = 1e-4 * abs(float(fz.qlead[k]))
    d = jnp.zeros_like(fz.qlead).at[k].set(h)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g[k])) > 0
    np.testing.assert_allclose(float(g[k]), float((f(fz.qlead + d) - f(fz.qlead - d)) / (2 * h)), rtol=1e-5)
