"""C6: nemo_version="5.0.1" reproduces the ORCA1 oracle's column order (port default stays 5.0.2).

5.0.1 melts snow BEFORE sublimation/deposition (icethd_dh.F90:179-201); 5.0.2 moved the
block after.  The same option carries the other 5.0.1 column differences (no MAX(0,.) on the
basal-growth / new-ice latent term; BL99 cap of 50 iterations), both inactive in this record.
Oracle = NEMO 5.0.1 SI3 ORCA1 tier-4 record (fixture c6_melt_order: 8 snow-melt + sublimation
columns where the 5.0.2 order moves h_i by > 1e-12, plus 4 non-melting controls).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as K

from si3_t4col_fixture import col, dh_args, dt, load, rel, within, within_flux

jax.config.update("jax_enable_x64", True)
FX = load("c6_melt_order")
STATE = (("h_i", "h_ice"), ("h_s", "h_snow"), ("e_i", "e_ice"), ("e_s", "e_snow"), ("s_i", "S_bulk"))
FLUX = ("hfx_sub", "wfx_snw_sub", "wfx_ice_sub", "sfx_sub", "hfx_snw", "wfx_snw_sum", "wfx_err_sub",
        "hfx_sum", "wfx_sum", "sfx_sum", "hfx_thd")


def _run(v):
    st, zr, fz = dh_args(FX)
    return st, bl._dh_step_fluxes(st, zr, fz, dt(FX), K, nemo_version=v)


def test_501_order_matches_nemo_state_and_fluxes():
    _, (out, fx) = _run("5.0.1")
    bad = {f: np.flatnonzero(~within(FX, "dh", f, getattr(out, a))).tolist() for f, a in STATE}
    # hfx_thd also carries the flooding seawater enthalpy (C5 domain: t_bo vs SST, separate PR)
    no_flood = col(FX, "dh:nemo:dh_snowice") == 0
    for f in FLUX:
        ok = within_flux(FX, "dh", f, col(FX, f"dh:in:{f}") + np.asarray(getattr(fx, f)))
        bad[f] = np.flatnonzero(~ok & (no_flood if f == "hfx_thd" else True)).tolist()
    assert (no_flood & FX["is_fail_on_main"]).sum() >= 4
    assert not any(bad.values()), bad


def test_default_502_order_differs_on_the_melt_columns():
    _, (o2, _) = _run("5.0.2")
    m = FX["is_fail_on_main"]
    assert np.all(rel(o2.h_ice, col(FX, "dh:nemo:h_i"))[m] > 1e-12)


def test_budgets_close_in_501_order():
    st, (out, fx) = _run("5.0.1")
    a, d = np.asarray(st.concentration), dt(FX)
    mass = a * (K.rho_ice * (out.h_ice - st.h_ice) + K.rho_snow * (out.h_snow - st.h_snow)) / d
    w = sum(np.asarray(getattr(fx, f)) for f in bl.SI3ExchangeFluxes._fields
            if f.startswith("wfx") and f not in ("wfx_err_sub", "wfx_opw"))
    np.testing.assert_allclose(mass, -w, rtol=0, atol=1e-12 * max(1.0, np.abs(mass).max()))
    q = lambda s: (np.sum(s.e_ice, -1) * s.h_ice + np.sum(s.e_snow, -1) * s.h_snow) / 3.0  # noqa: E731
    g = lambda k: np.asarray(getattr(fx, k))  # noqa: E731
    rhs = (g("hfx_thd") + g("hfx_res") + g("hfx_sub") + g("hfx_spr")
           - g("hfx_sum") - g("hfx_bom") - g("hfx_bog") - g("hfx_snw"))
    dq = a * (q(out) - q(st)) / d
    np.testing.assert_allclose(dq, rhs, rtol=0, atol=1e-11 * max(1.0, np.abs(dq).max()))


def test_unknown_version_raises_and_latent_max_differs_only_when_binding():
    st, zr, fz = dh_args(FX)
    with pytest.raises(ValueError, match="Unknown SI3 column-port variant"):
        bl._dh_step_fluxes(st, zr, fz, dt(FX), K, nemo_version="5.0")
    Tm, T = jnp.asarray(-1.0), jnp.asarray([-2.0, -0.5])       # binding when T > Tm
    l1, l2 = bl._latent_part(Tm, T, "5.0.1"), bl._latent_part(Tm, T, "5.0.2")
    assert float(l1[0]) == float(l2[0]) and float(l1[1]) < 0.0 == float(l2[1])


def test_501_gradient_finite_and_matches_fd():
    st, zr, fz = dh_args(FX)
    k = int(np.flatnonzero(FX["is_fail_on_main"])[0])
    f = lambda e: bl._dh_step(st, zr, fz._replace(evaporation=e), dt(FX), K, nemo_version="5.0.1").h_snow[k]  # noqa: E731
    g = jax.grad(f)(fz.evaporation)
    h = 1e-3 * abs(float(fz.evaporation[k]))
    d = jnp.zeros_like(fz.evaporation).at[k].set(h)
    assert bool(jnp.all(jnp.isfinite(g)))
    np.testing.assert_allclose(float(g[k]), float((f(fz.evaporation + d) - f(fz.evaporation - d)) / (2 * h)),
                               rtol=1e-5, atol=1e-12)
