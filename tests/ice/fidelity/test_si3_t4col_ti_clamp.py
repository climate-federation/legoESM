"""C2: ice temperature from enthalpy carries NEMO's glo2eqv bounds (icevar.F90:393-395).

Oracle = NEMO 5.0.1 SI3 ORCA1 tier-4 record (fixture c2_ti_clamp: the one column where the
bound binds on main, cell 1567, SH, h_s 0.78 mm, plus 4 controls).  Hard form (width 0)
must reproduce NEMO's BL99 outputs; the default C1 form stays within width/8 of it and never above Tmelt.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as K

from si3_t4col_fixture import col, dt, load, within, zdf_args

jax.config.update("jax_enable_x64", True)
FX = load("c2_ti_clamp")
FIELDS = ("t_su", "t_i", "t_s", "e_i", "e_s", "qcn_ice_bot")


def _zdf(**kw):
    a = zdf_args(FX)
    r = bl._si3_zdf_bl99_step(*a[:6], a[6], dt(FX), K, **kw)
    return {"t_su": r.T_surface, "t_i": r.T_ice, "t_s": r.T_snow, "e_i": r.e_ice, "e_s": r.e_snow,
            "qcn_ice_bot": r.qcn_ice_bottom}


def test_fixture_contains_the_binding_column():
    assert FX["is_fail_on_main"].sum() == 1 and 1567 in FX["cell"].tolist()
    # main's recorded port value is outside the 20-seed spread there (the defect)
    assert not within(FX, "zdf", "t_i", col(FX, "zdf:main:t_i"))[FX["is_fail_on_main"]].any()


def test_hard_bounds_reproduce_nemo_bl99_outputs():
    out = _zdf(t_clamp_width=0.0)
    bad = {f: np.flatnonzero(~within(FX, "zdf", f, out[f])).tolist() for f in FIELDS}
    assert not any(bad.values()), bad


def test_default_smooth_bounds_stay_within_c1_band_of_hard():
    e, S = jnp.asarray(col(FX, "zdf:in:e_i")), jnp.asarray(col(FX, "zdf:in:sz_i"))
    hard = bl.ice_temperature_from_enthalpy(e, S, K, width=0.0)
    soft = bl.ice_temperature_from_enthalpy(e, S, K)
    gap = float(jnp.max(jnp.abs(soft - hard)))
    assert gap <= bl._T_CLAMP_WIDTH_K / 8.0 + 1e-12, gap
    below = np.asarray(hard) < np.asarray(K.T0 - K.liquidus_slope * S) - 0.0051
    assert np.array_equal(np.asarray(soft)[below], np.asarray(hard)[below])   # exact outside the band


def test_bounds_gradient_finite_nonzero_across_band_and_matches_fd():
    T_m = -0.27
    T_c = T_m + jnp.array([-0.02, -0.006, -0.003, 0.0, 0.003, 0.02])
    g = jax.vmap(jax.grad(lambda x: bl._glo2eqv_bounds(x, T_m, bl._T_CLAMP_WIDTH_K)))(T_c)
    assert bool(jnp.all(jnp.isfinite(g)))
    np.testing.assert_allclose(np.asarray(g[jnp.array([0, 1])]), 1.0)   # below the band: identity
    assert float(g[-1]) == 0.0 and bool(jnp.all((g[2:5] > 0) & (g[2:5] < 1)))   # band: monotone transfer
    # through the full inverse at the binding fixture column (cell 1567, inside the band)
    k = FX["cell"].tolist().index(1567)
    e, S = jnp.asarray(col(FX, "zdf:in:e_i"))[k], jnp.asarray(col(FX, "zdf:in:sz_i"))[k]
    f = lambda x: jnp.sum(bl.ice_temperature_from_enthalpy(x, S, K))  # noqa: E731
    ge = jax.grad(f)(e)
    assert bool(jnp.all(jnp.isfinite(ge))) and float(jnp.abs(ge).max()) > 0
    for m in range(e.shape[0]):
        h = 1e-6 * float(jnp.abs(e[m]))
        d = jnp.zeros_like(e).at[m].set(h)
        np.testing.assert_allclose(float(ge[m]), float((f(e + d) - f(e - d)) / (2 * h)), rtol=1e-4, atol=1e-16)


def test_zdf_heat_content_change_matches_nemo_hfx_dif():
    """Conservation (vote D11a): -a*d(sum e*h)/dt of the port equals NEMO's hfx_dif contribution."""
    out = _zdf(t_clamp_width=0.0)
    i = lambda k: col(FX, f"zdf:in:{k}")  # noqa: E731
    q = lambda e_i, e_s: (np.sum(e_i, -1) * i("h_i") + np.sum(e_s, -1) * i("h_s")) / 3.0  # noqa: E731
    port = -(q(np.asarray(out["e_i"]), np.asarray(out["e_s"])) - q(i("e_i"), i("e_s"))) / dt(FX) * i("a_i")
    nemo = col(FX, "zdf:nemo:hfx_dif") - i("hfx_dif")
    np.testing.assert_allclose(port, nemo, rtol=1e-9, atol=1e-9)


def test_smooth_min_max_c1_helper_contract():
    from legoesm.core.smooth import smooth_max_c1, smooth_min_c1
    x = jnp.linspace(-0.02, 0.02, 401)
    w = 0.01
    m = smooth_min_c1(x, 0.0, w)
    assert bool(jnp.all(m <= jnp.minimum(x, 0.0) + 1e-15))                 # never above the bound
    assert float(jnp.max(jnp.minimum(x, 0.0) - m)) <= w / 8 + 1e-15
    assert bool(jnp.all(jnp.diff(m) >= -1e-15))                           # monotone
    out = jnp.abs(x) >= w / 2
    assert bool(jnp.all(m[out] == jnp.minimum(x, 0.0)[out]))              # exact outside the band
    assert bool(jnp.all(smooth_min_c1(x, 0.0, 0) == jnp.minimum(x, 0.0)))
    assert bool(jnp.all(smooth_max_c1(x, 0.0, w) == -smooth_min_c1(-x, 0.0, w)))
    g = jax.vmap(jax.grad(lambda v: smooth_min_c1(v, 0.0, w)))(x)
    assert bool(jnp.all(jnp.isfinite(g))) and bool(jnp.all((g >= 0) & (g <= 1)))
