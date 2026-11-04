"""C7: layer remap divides by MAX(zhnew, epsi20) like NEMO's ice_var_vremap (icevar.F90:1005-1031).

Main returned 0 enthalpy for any new layer <= 1e-10 m.  In the tier-4 record (fixture
c7_remap_eps) NEMO keeps e_i up to 8e10 J m-3 in columns of 3e-18..2e-17 m; the port reaches
h_i = 0 exactly in those columns (roundoff-level thickness inside the C6 melt-order domain), so
the literal row cannot isolate C7 there.  The oracle here is therefore a line-by-line numpy
transliteration of ice_var_vremap, fed the fixture's recorded layer enthalpies on thin columns.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ice import bitz_lipscomb as bl

from si3_t4col_fixture import col, load

jax.config.update("jax_enable_x64", True)
FX = load("c7_remap_eps")
EPSI20 = 1.0e-20


def nemo_vremap(ph_old, pts_old):
    """icevar.F90 ice_var_vremap, nlay_i = 3, ph_old/pts_old over 0:nlay_i+1 (content per segment)."""
    nlay = 3
    zh_cum0 = np.concatenate([[0.0], np.cumsum(ph_old)])
    zts_cum0 = np.concatenate([[0.0], np.cumsum(pts_old)])
    zhnew = np.sum(ph_old) * (1.0 / nlay)
    zh_cum1 = np.array([zhnew * k for k in range(nlay + 1)])
    zts_cum1 = np.zeros(nlay + 1)
    for jk0 in range(1, nlay + 3):
        for jk1 in range(1, nlay):
            if zh_cum1[jk1] <= zh_cum0[jk0] and zh_cum1[jk1] > zh_cum0[jk0 - 1]:
                zts_cum1[jk1] = (zts_cum0[jk0 - 1] * (zh_cum0[jk0] - zh_cum1[jk1])
                                 + zts_cum0[jk0] * (zh_cum1[jk1] - zh_cum0[jk0 - 1])) / (zh_cum0[jk0] - zh_cum0[jk0 - 1])
    zts_cum1[nlay] = zts_cum0[nlay + 2]
    return np.array([max(0.0, zts_cum1[k] - zts_cum1[k - 1]) / max(zhnew, EPSI20) for k in range(1, nlay + 1)])


def _cases():
    e = col(FX, "dh:in:e_i")                     # recorded per-volume layer enthalpies [J m-3]
    for total in (2.08e-17, 6.9e-18, 1e-12, 1e-10, 0.37):
        for row in e:
            h = np.array([0.0, total / 3, total / 3, total / 3 * 0.5, total / 3 * 0.5])   # last = basal new ice
            yield h, np.array([0.0, row[0], row[1], row[2], row[2] * 1.01])


def test_remap_matches_nemo_vremap_on_thin_and_normal_columns():
    for h, e in _cases():
        port = np.asarray(bl._piecewise_remap(jnp.asarray(h), jnp.asarray(e), jnp.asarray(h.sum())))
        ref = nemo_vremap(h, h * e)
        np.testing.assert_allclose(port, ref, rtol=1e-12, atol=0.0)
        assert ref.max() > 0                       # the thin columns keep their enthalpy


def test_remap_conserves_column_enthalpy_including_thin_columns():
    for h, e in _cases():
        port = np.asarray(bl._piecewise_remap(jnp.asarray(h), jnp.asarray(e), jnp.asarray(h.sum())))
        np.testing.assert_allclose(np.sum(port) * h.sum() / 3, np.sum(h * e), rtol=1e-12)


def test_remap_zero_column_is_zero_and_gradient_finite_at_thin_column():
    z = jnp.zeros(5)
    assert float(jnp.max(jnp.abs(bl._piecewise_remap(z, z + 3e8, jnp.asarray(0.0))))) == 0.0
    h0, e0 = next(_cases())
    f = lambda t: jnp.sum(bl._piecewise_remap(jnp.asarray(h0) * t, jnp.asarray(e0), jnp.sum(jnp.asarray(h0) * t)))  # noqa: E731
    g = jax.grad(f)(1.0)
    assert bool(jnp.isfinite(g))
    ge = jax.grad(lambda e: jnp.sum(bl._piecewise_remap(jnp.asarray(h0), e, jnp.asarray(h0.sum()))))(jnp.asarray(e0))
    assert bool(jnp.all(jnp.isfinite(ge))) and float(jnp.abs(ge).max()) > 0
    gz = jax.grad(lambda t: jnp.sum(bl._piecewise_remap(z * t, z + 3e8, jnp.sum(z * t))))(1.0)
    assert bool(jnp.isfinite(gz))
