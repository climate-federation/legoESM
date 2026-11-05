"""C5: snow-ice flooding takes the seawater enthalpy rcp*SST (icethd_dh.F90:454), not rcp*(t_bo-rt0).

Oracle = NEMO 5.0.1 SI3 ORCA1 tier-4 record (fixture c5_flood_sst: 8 flooding columns whose
e_i failed on main, 4 non-flooding controls).  SST - t_bo reaches 6.8 K in the record.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as K

from si3_t4col_fixture import col, dh_args, dt, load, within, within_flux

jax.config.update("jax_enable_x64", True)
FX = load("c5_flood_sst")
FLOOD = ("hfx_thd", "sfx_sni", "wfx_sni", "wfx_snw_sni")


def _run(fz=None):
    st, zr, f0 = dh_args(FX)
    return st, bl._dh_step_fluxes(st, zr, f0 if fz is None else fz, dt(FX), K)


def test_fixture_flooding_columns_fail_on_main():
    fl = FX["is_fail_on_main"]
    assert fl.sum() == 8 and np.all(col(FX, "dh:nemo:dh_snowice")[fl] > 0)
    assert not within(FX, "dh", "e_i", col(FX, "dh:main:e_i"))[fl].any()


def test_flooded_enthalpy_and_flood_fluxes_match_nemo():
    _, (out, fx) = _run()
    for f, v in (("e_i", out.e_ice), ("h_i", out.h_ice), ("h_s", out.h_snow), ("s_i", out.S_bulk)):
        assert within(FX, "dh", f, v).all(), (f, np.flatnonzero(~within(FX, "dh", f, v)))
    for f in FLOOD:
        acc = col(FX, f"dh:in:{f}") + np.asarray(getattr(fx, f))
        assert within_flux(FX, "dh", f, acc).all(), f


def test_heat_budget_closes_with_sst_seawater():
    st, (out, fx) = _run()
    q = lambda s: (np.sum(s.e_ice, -1) * s.h_ice + np.sum(s.e_snow, -1) * s.h_snow) / 3.0  # noqa: E731
    g = lambda k: np.asarray(getattr(fx, k))  # noqa: E731
    rhs = (g("hfx_thd") + g("hfx_res") + g("hfx_sub") + g("hfx_spr")
           - g("hfx_sum") - g("hfx_bom") - g("hfx_bog") - g("hfx_snw"))
    dq = np.asarray(st.concentration) * (q(out) - q(st)) / dt(FX)
    np.testing.assert_allclose(dq, rhs, rtol=0, atol=1e-11 * max(1.0, np.abs(dq).max()))


def test_missing_sst_raises_and_sst_gradient_flows():
    st, zr, fz = dh_args(FX)
    with pytest.raises(ValueError, match="sst"):
        bl._dh_step_fluxes(st, zr, fz._replace(sst=None), dt(FX), K)
    k = int(np.flatnonzero(FX["is_fail_on_main"])[0])
    f = lambda t: jnp.sum(bl._dh_step(st, zr, fz._replace(sst=t), dt(FX), K).e_ice[k])  # noqa: E731
    g = jax.grad(f)(fz.sst)
    d = jnp.zeros_like(fz.sst).at[k].set(1e-3)
    assert float(jnp.abs(g[k])) > 0
    np.testing.assert_allclose(float(g[k]), float((f(fz.sst + d) - f(fz.sst - d)) / 2e-3), rtol=1e-6)
