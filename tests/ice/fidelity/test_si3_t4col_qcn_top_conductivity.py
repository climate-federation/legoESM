"""C3: snow-free top conductive flux uses the conductivity of the START of the last BL99 iteration.

NEMO evaluates ztcond_i once per iteration from that iteration's starting temperatures and
reuses it for qcn_ice_top after the loop (icethd_zdf_bl99.F90:283,325,771); main recomputed it
from the final T_i.  Oracle = NEMO 5.0.1 SI3 ORCA1 tier-4 record (fixture c3_qcn_top_k: 8
snow-free columns failing on main + 4 snow-covered controls).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ice import bitz_lipscomb as bl
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG as K

from si3_t4col_fixture import col, dt, load, within, zdf_args

jax.config.update("jax_enable_x64", True)
FX = load("c3_qcn_top_k")


def _zdf(**over):
    a = zdf_args(FX)
    fz = a[6]._replace(**over)
    return bl._si3_zdf_bl99_step(*a[:6], fz, dt(FX), K)


def test_fixture_snow_free_columns_fail_on_main():
    fail = FX["is_fail_on_main"]
    assert fail.sum() == 8 and np.all(col(FX, "zdf:in:h_s")[fail] == 0)
    assert not within(FX, "zdf", "qcn_ice_top", col(FX, "zdf:main:qcn_ice_top"))[fail].any()


def test_qcn_ice_top_and_bl99_state_match_nemo():
    r = _zdf()
    out = {"qcn_ice_top": r.qcn_ice_top, "t_su": r.T_surface, "t_i": r.T_ice, "t_s": r.T_snow,
           "e_i": r.e_ice, "e_s": r.e_snow, "qcn_ice_bot": r.qcn_ice_bottom}
    bad = {f: np.flatnonzero(~within(FX, "zdf", f, v)).tolist() for f, v in out.items()}
    assert not any(bad.values()), bad


def test_zdf_heat_content_change_matches_nemo_hfx_dif():
    """Conservation (vote D11a): -a*d(sum e*h)/dt equals NEMO's hfx_dif contribution."""
    r = _zdf()
    i = lambda k: col(FX, f"zdf:in:{k}")  # noqa: E731
    q = lambda e_i, e_s: (np.sum(e_i, -1) * i("h_i") + np.sum(e_s, -1) * i("h_s")) / 3.0  # noqa: E731
    port = -(q(np.asarray(r.e_ice), np.asarray(r.e_snow)) - q(i("e_i"), i("e_s"))) / dt(FX) * i("a_i")
    np.testing.assert_allclose(port, col(FX, "zdf:nemo:hfx_dif") - i("hfx_dif"), rtol=1e-9, atol=1e-9)


def test_qcn_ice_top_gradient_finite_and_matches_fd():
    a = zdf_args(FX)
    q0 = a[6].qns_ice

    def f(q):
        return jnp.sum(_zdf(qns_ice=q).qcn_ice_top)

    g = jax.grad(f)(q0)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).max()) > 0
    d = jnp.zeros_like(q0).at[0].set(1e-4)
    np.testing.assert_allclose(float(g[0]), float((f(q0 + d) - f(q0 - d)) / 2e-4), rtol=1e-4, atol=1e-8)
