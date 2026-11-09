"""PR 4: ice_cor(kt, 2) after ice_thd_do in the SI3 column chain (coordinator ruling, votes Z65-Z72).

NEMO 5.0.1 keeps the ice volume in ice_thd_do (`icethd_do.F90:286-291`) and lets
ice_cor -> ice_var_zapsmall (`icecor.F90:114`, `icevar.F90:611-700`) empty categories
with MIN(a_i, v_i, h_i) < epsi10, returning their water, salt and heat through
wfx/sfx/hfx_res.  Checked here on the tier-4 c1_thd_do fixture (NEMO 5.0.1 ORCA1):
* entry -> exit water, salt and heat close with only ice_thd_do and ice_cor acting,
  on the recorded state and on a thin (area 5e-13) state whose ice the zap removes;
* the smooth surrogate of the correction passes check_grads inside its bands.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice.config import SI3SwitchConfig
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as K

from test_si3_t4col_thd_do import DT, FX, _state

jax.config.update("jax_enable_x64", True)
SMOOTH = SI3SwitchConfig()


@pytest.fixture(autouse=True)
def _release_executables():
    yield
    jax.clear_caches()     # vote Z41


def _forcing(fz):
    return fz._replace(sst=jnp.asarray(FX["t_bo"]) - K.T0)       # vote Z72


def _do_and_cor_only(monkeypatch):
    """Stages before ice_thd_do hand their input through with zero exchange (Z70)."""
    def zdf(e_i, e_s, sz, h_i, h_s, t_su, *a, **k):
        z = jnp.zeros_like(h_i)
        return bl.SI3ZDFResult(t_su, jnp.zeros_like(e_i), jnp.zeros_like(e_s), e_i, e_s,
                               z, z, z, z, jnp.zeros(h_i.shape, jnp.int32))
    monkeypatch.setattr(bl, "_si3_zdf_bl99_step", zdf)
    monkeypatch.setattr(bl, "_zdf_exchange_fluxes", lambda s, *a, **k: bl._zero_fluxes(s.h_ice))
    monkeypatch.setattr(bl, "_dh_step_fluxes", lambda s, *a, **k: (s, bl._zero_fluxes(s.h_ice)))
    monkeypatch.setattr(bl, "_sal_step", lambda s, *a, **k: (s.S_bulk, s.S_layers, bl._zero_fluxes(s.h_ice)))


def _inventories(s):
    a, h, hs = (np.asarray(x) for x in (s.concentration, s.h_ice, s.h_snow))
    water = a * (K.rho_ice * h + K.rho_snow * hs)
    salt = K.rho_ice * a * h * np.asarray(s.S_bulk)
    heat = a * (np.sum(np.asarray(s.e_ice), -1) * h + np.sum(np.asarray(s.e_snow), -1) * hs) / 3.0
    return water, salt, heat


@pytest.mark.parametrize("arm", ["recorded", "thin"])
def test_do_and_cor_budgets_close(monkeypatch, arm):
    st, fz = _state()
    if arm == "thin":
        st = st._replace(concentration=jnp.full_like(st.concentration, 5e-13))
    _do_and_cor_only(monkeypatch)
    tr = bl.si3_column_step_arrays(st, _forcing(fz), DT, K, nemo_version="5.0.1")
    f = tr.fluxes
    g = lambda k: np.asarray(getattr(f, k))  # noqa: E731
    (w0, s0, q0), (w1, s1, q1) = _inventories(st), _inventories(tr.exit)
    wsum = sum(g(k) for k in bl.SI3ExchangeFluxes._fields if k.startswith("wfx") and k != "wfx_err_sub")
    ssum = sum(g(k) for k in bl.SI3ExchangeFluxes._fields if k.startswith("sfx"))
    tol = lambda x: 1e-14 * np.abs(x).max() / DT   # noqa: E731   (Z59 convention)
    np.testing.assert_allclose((w1 - w0) / DT, -wsum, rtol=1e-11, atol=max(1e-18, tol(w0)))
    np.testing.assert_allclose((s1 - s0) / DT, -ssum, rtol=1e-12, atol=max(1e-16, tol(s0)))
    np.testing.assert_allclose((q1 - q0) / DT, g("hfx_thd") - g("hfx_opw") + g("hfx_res"), rtol=1e-10, atol=1e-8)
    if arm == "thin":           # no growth (qlead >= 0): the zap empties the cell and returns its water
        idle = np.asarray(fz.qlead) >= 0.0
        assert idle.any() and np.all(np.asarray(tr.exit.concentration)[idle] == 0.0)
        assert np.all(g("wfx_res")[idle] > 0.0)


@pytest.mark.parametrize("band", ["thin_ice", "zap"])
def test_cor_soft_pass_check_grads(monkeypatch, band):
    """Z71: order-2 check_grads of the correction's soft pass inside each band, FD step
    1e-3 x band width, tolerances of test_si3_ice_cor_orca1_oracle."""
    st, fz = _state()
    k = int(np.flatnonzero(np.asarray(st.concentration) > 0.0)[0])
    one = lambda x: x[k:k + 1]  # noqa: E731
    post = jax.tree_util.tree_map(one, st)
    fk = jax.tree_util.tree_map(lambda x: one(x) if getattr(x, "ndim", 0) else x, _forcing(fz))
    a = post.concentration
    if band == "thin_ice":
        width = bl._MINIMUM_ICE_THICKNESS / 100.0
        v0 = a * (bl._MINIMUM_ICE_THICKNESS - 0.5 * width)
    else:
        width = bl._EPS10 / 100.0
        v0 = jnp.full_like(a, bl._EPS10 - 0.5 * width)
    monkeypatch.setattr(bl, "straight_through", lambda h, s: s)

    def f(v):
        vs = post.h_snow * a
        vols = (a, v, vs, post.S_bulk * v, post.e_ice * (v / 3.0)[..., None], post.e_snow * (vs / 3.0)[..., None])
        out, fl = bl._cor_step(post, vols, fk, DT, K, switches=SMOOTH)
        return jnp.concatenate([out.concentration, out.h_ice, out.S_bulk, jnp.ravel(out.e_ice),
                                fl.wfx_res, fl.sfx_res, fl.hfx_res])
    check_grads(f, (v0,), order=2, modes=("fwd", "rev"), eps=1e-3 * width, atol=1e-6, rtol=1e-3)
