"""C4: total melt sends the snow to the ocean and resets a_i = 0, h_s = 0, t_su = rt0.

NEMO icethd_dh.F90:427-440 (snow mass/enthalpy -> wfx_res/hfx_res when h_i == 0) and
:528-532 (WHERE h_i == 0).  Oracle = NEMO 5.0.1 SI3 ORCA1 tier-4 record (fixture c4_fullmelt:
8 fully melting columns + 4 basal-melt controls).  NEMO's own 1-ulp ensemble flips the
full-melt branch in some of these columns (spread 1.0), so the reset values are asserted
EXACTLY where both NEMO and the port melt the column out.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as K

from si3_t4col_fixture import col, dh_args, dt, load, within, within_flux

jax.config.update("jax_enable_x64", True)
FX = load("c4_fullmelt")


def _run(**kw):
    st, zr, fz = dh_args(FX)
    return st, bl._dh_step_fluxes(st, zr, fz, dt(FX), K, **kw)


def test_full_melt_reset_matches_nemo_exactly():
    _, (out, fx) = _run(full_melt_width=0.0)
    melt = (col(FX, "dh:nemo:h_i") == 0) & (np.asarray(out.h_ice) == 0)
    assert melt.sum() >= 6
    for f, v in (("a_i", out.concentration), ("h_s", out.h_snow), ("t_su", out.T_surface)):
        np.testing.assert_array_equal(np.asarray(v)[melt], col(FX, f"dh:nemo:{f}")[melt], err_msg=f)
    # main kept the area / snow / cold surface there
    assert np.any(col(FX, "dh:main:a_i")[melt] != 0) and np.any(col(FX, "dh:main:t_su")[melt] != K.T0)


def test_snow_to_ocean_fluxes_and_controls_match_nemo():
    _, (out, fx) = _run(full_melt_width=0.0)
    for f in ("wfx_res", "hfx_res"):
        acc = col(FX, f"dh:in:{f}") + np.asarray(getattr(fx, f))
        assert within_flux(FX, "dh", f, acc).all(), f
    ctl = ~FX["is_fail_on_main"]
    for f, v in (("a_i", out.concentration), ("h_s", out.h_snow), ("t_su", out.T_surface), ("h_i", out.h_ice)):
        assert within(FX, "dh", f, v)[ctl].all(), f


def test_mass_and_heat_budget_close_with_snow_removal():
    st, (out, fx) = _run()
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


def test_smooth_snow_removal_band_is_differentiable():
    """Thin a basal-melting snowy control column until its residual ice sits inside the 1e-3 m band.

    (The fully melting fixture columns melt by internal melting, which removes whole layers at
    any thickness, so they cannot be tuned into the band.)
    """
    st, zr, fz = dh_args(FX)
    ctl = (~FX["is_fail_on_main"]) & (col(FX, "dh:in:h_s") > 0) & (col(FX, "dh:nemo:dh_i_bom") < 0)
    k = int(np.flatnonzero(ctl)[0])
    h0 = float(st.h_ice[k])

    def run(h, **kw):
        return bl._dh_step(st._replace(h_ice=st.h_ice.at[k].set(h)), zr, fz, dt(FX), K, **kw)

    w = bl._FULL_MELT_SNOW_WIDTH_M
    assert float(run(h0).h_ice[k]) > w
    lo, hi = 0.0, h0                           # bisect the entry thickness: residual = w/2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if float(run(mid).h_ice[k]) < 0.5 * w else (lo, mid)
    x0 = jnp.asarray(hi)
    assert 0.3 * w < float(run(x0).h_ice[k]) < 0.7 * w
    f = lambda x: run(x).h_snow[k]  # noqa: E731
    g = jax.grad(f)(x0)
    fd = (f(x0 + 1e-8) - f(x0 - 1e-8)) / 2e-8
    assert bool(jnp.isfinite(g)) and float(jnp.abs(g)) > 0
    np.testing.assert_allclose(float(g), float(fd), rtol=1e-3)
    # hard form keeps all of that snow while any ice remains
    assert float(run(x0, full_melt_width=0.0).h_snow[k]) > float(f(x0))
