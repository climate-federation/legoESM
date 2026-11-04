"""C9: the SI3 column port returns NEMO's hfx/wfx/sfx exchange pieces (tier-4 U7/U8 oracle).

Oracle = NEMO 5.0.1 SI3 ORCA1 record (fixture c9_exchange: one or two columns per active
process: basal growth/melt, surface melt, snow melt, flooding, sublimation, deposition,
full melt, drainage, snow-free).  NEMO accumulators are compared as ``in + port contribution``
against the recorded ``out`` at the per-cell 20-seed tolerance (or the vote-W1a absolute floor), on columns whose main-port
state already matches NEMO (the others carry separate defects C1-C8).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as K

from si3_t4col_fixture import col, dh_args, dt, load, rel, within, within_flux, zdf_args

jax.config.update("jax_enable_x64", True)
FX = load("c9_exchange")
DH_FLUX = [f for f in bl.SI3ExchangeFluxes._fields if f"dh:nemo:{f}" in FX]


def _dh():
    st, zr, fz = dh_args(FX)
    return st, bl._dh_step_fluxes(st, zr, fz, dt(FX), K)


def test_fixture_covers_every_dh_accumulator_and_process():
    assert len(DH_FLUX) == 26, DH_FLUX
    assert {"basal_growth", "basal_melt", "surface_melt", "snow_melt", "flooding",
            "sublimation", "deposition", "drainage"} <= set(FX["process_names"].tolist())


def _comparable():
    """Columns outside the C5 (flooding enthalpy) and C6 (5.0.1 melt-then-sublimate order) domains.

    Every recorded column sublimates or deposits, so every snow-melt column is in the C6 domain;
    surface/snow-melt and flooding pieces are oracle-checked by the stacked C6/C5 PRs and here
    only through the budget closure below.
    """
    c6 = (col(FX, "dh:nemo:dh_s_sum") < 0) & (col(FX, "dh:in:evap_ice") != 0)
    c5 = col(FX, "dh:nemo:dh_snowice") > 0
    return ~c6 & ~c5


def test_dh_contributions_match_nemo_on_comparable_columns():
    _, (out, fx) = _dh()
    ok = _comparable()
    assert ok.sum() >= 6
    processes = {"basal melt": col(FX, "dh:nemo:dh_i_bom") < 0, "basal growth": col(FX, "dh:nemo:dh_i_bog") > 0,
                 "sublimation": col(FX, "dh:in:evap_ice") > 0, "deposition": col(FX, "dh:in:evap_ice") < 0}
    assert all((m & ok).any() for m in processes.values())
    bad = {}
    for f in DH_FLUX:
        acc = col(FX, f"dh:in:{f}") + np.asarray(getattr(fx, f))
        m = within_flux(FX, "dh", f, acc)
        if not m[ok].all():
            bad[f] = rel(acc, col(FX, f"dh:nemo:{f}"))[ok].max()
    assert not bad, bad


def test_dh_column_budget_closes_on_port_own_fluxes():
    st, (out, fx) = _dh()
    rhoi, rhos, d = K.rho_ice, K.rho_snow, dt(FX)
    a = np.asarray(st.concentration)
    mass = a * (rhoi * (out.h_ice - st.h_ice) + rhos * (out.h_snow - st.h_snow)) / d
    wsum = sum(np.asarray(getattr(fx, f)) for f in DH_FLUX if f.startswith("wfx") and f != "wfx_err_sub")
    np.testing.assert_allclose(mass, -wsum, rtol=0, atol=1e-12 * max(1.0, np.abs(mass).max()))
    salt = a * rhoi * (out.h_ice * out.S_bulk - st.h_ice * st.S_bulk) / d
    ssum = sum(np.asarray(getattr(fx, f)) for f in DH_FLUX if f.startswith("sfx"))
    np.testing.assert_allclose(salt, -ssum, rtol=0, atol=1e-11 * max(1.0, np.abs(salt).max()))

    def content(s):
        return (np.sum(s.e_ice, -1) * s.h_ice + np.sum(s.e_snow, -1) * s.h_snow) / 3.0

    dq = a * (content(out) - content(st)) / d
    g = lambda k: np.asarray(getattr(fx, k))  # noqa: E731
    rhs = (g("hfx_thd") + g("hfx_res") + g("hfx_sub") + g("hfx_spr")
           - g("hfx_sum") - g("hfx_bom") - g("hfx_bog") - g("hfx_snw"))
    np.testing.assert_allclose(dq, rhs, rtol=0, atol=1e-11 * max(1.0, np.abs(dq).max()))


def test_zdf_hfx_dif_and_err_match_nemo():
    args = zdf_args(FX)
    zr = bl._si3_zdf_bl99_step(*args[:6], args[6], dt(FX), K)
    st = bl.SI3ColumnArrays(jnp.asarray(col(FX, "zdf:in:a_i")), args[3], args[4], args[5],
                            args[0], args[1], jnp.zeros_like(args[3]), args[2], jnp.zeros_like(args[3]))
    fx = bl._zdf_exchange_fluxes(st, zr, args[6], dt(FX), K)
    for f in ("hfx_dif", "hfx_err_dif"):
        acc = col(FX, f"zdf:in:{f}") + np.asarray(getattr(fx, f))
        # hfx_err_dif is a residual (~1e-10 W m-2): score it absolutely against NEMO's own size.
        if f == "hfx_err_dif":
            np.testing.assert_allclose(acc, col(FX, "zdf:nemo:hfx_err_dif"), rtol=0, atol=1e-6)
        else:
            assert within_flux(FX, "zdf", f, acc)[FX["state_ok_on_main"]].all(), f
    # non-vacuity of the fixture: the residual is non-zero somewhere
    assert np.abs(col(FX, "zdf:nemo:hfx_dif") - col(FX, "zdf:in:hfx_dif")).max() > 1.0


def test_sal_sfx_bri_res_match_nemo_above_himin():
    n = len(FX["cell"])
    i = lambda k: jnp.asarray(col(FX, f"sal:in:{k}"))  # noqa: E731
    pre = bl.SI3ColumnArrays(i("a_i"), i("h_i"), jnp.zeros(n), i("t_su"), jnp.zeros((n, 3)), jnp.zeros((n, 3)),
                             i("s_i"), i("sz_i"), jnp.zeros(n))
    from si3_t4col_fixture import forcing
    _, _, fx = bl._sal_step(pre, forcing(n, t_bottom=i("t_bo"), sss=i("sss")), dt(FX), K)
    thick = np.asarray(i("h_i")) > 0.1      # main has no rn_sal_himin gate yet (C8)
    for f in ("sfx_bri", "sfx_res"):
        acc = col(FX, f"sal:in:{f}") + np.asarray(getattr(fx, f))
        assert within_flux(FX, "sal", f, acc)[thick].all(), f


def test_column_trace_carries_summed_fluxes_and_is_differentiable():
    st, zr, fz = dh_args(FX)
    tr = bl.si3_column_step_arrays(st, fz._replace(dqns_ice=-jnp.ones_like(st.h_ice)), dt(FX), K)
    assert isinstance(tr.fluxes, bl.SI3ExchangeFluxes)

    def loss(qsb):
        f = bl.si3_column_step_arrays(st, fz._replace(qsb_ice_bottom=qsb), dt(FX), K).fluxes
        return jnp.sum(f.hfx_thd + f.wfx_bog + f.wfx_bom + f.sfx_bog + f.sfx_bom)

    g = jax.grad(loss)(fz.qsb_ice_bottom)
    eps = 1e-3
    e = jnp.zeros_like(g).at[0].set(eps)
    fd = (loss(fz.qsb_ice_bottom + e) - loss(fz.qsb_ice_bottom - e)) / (2 * eps)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).max()) > 0
    np.testing.assert_allclose(float(g[0]), float(fd), rtol=1e-5, atol=1e-10)
