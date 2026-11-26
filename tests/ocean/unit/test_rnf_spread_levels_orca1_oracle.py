"""Oracle test: NEMO whole-level runoff spread vs NEMO 5.0.1 ORCA1 (tier-3 RNF unit, defect P1).

NEMO ``ln_rnf_depth_ini`` (sbcrnf.F90:415-441) spreads runoff over ``nk_rnf``
WHOLE levels (first ``jk >= 2`` whose ``gdept`` reaches ``h``, capped at
``mbkt``) with ``h_rnf = sum e3t(1..nk_rnf, Kmm)``, and ``sbc_rnf_div`` adds
``rnf*r1_rho0/h_rnf`` to every one of those levels.  Fixture: 70 ORCA1 columns
(10 deepest spreads, 50 random runoff columns, 10 no-runoff) with NEMO's
recorded zrnfcl_max, gdept_1d, mbkt, nk_rnf, e3t(Kmm), tmask, rnf, h_rnf;
provenance in ``rnf_spread_levels.json``.  Integers exact; reals at rel 1e-12
(C2: NEMO spread 5.5e-16 < 1e-12).
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.forcing.runoff_depth import nemo_runoff_level_count
from legoesm.ocean.freshwater import (
    FreshwaterForcing,
    RunoffSpreadLevels,
    resolve_runoff_spread_arg,
    runoff_entry_profile,
    runoff_spread_layer_fractions,
    runoff_spread_virtual_salt_tendency_3d,
)

jax.config.update("jax_enable_x64", True)

_FIX = Path(__file__).parent / "fixtures" / "orca1_t3" / "rnf_spread_levels.npz"


@pytest.fixture(scope="module")
def fx():
    return dict(np.load(_FIX))


def _nk(fx):
    return nemo_runoff_level_count(
        fx["zrnfcl_max"][None, :], fx["gdept_1d"],
        dep_max=float(fx["rn_dep_max"]), rnf_max=float(fx["rn_rnf_max"]))


def test_level_count_matches_nemo_nk_rnf(fx):
    nk = _nk(fx)
    # NEMO caps at mbkt; the runtime weights apply that cap (wet levels only).
    np.testing.assert_array_equal(np.minimum(nk, fx["mbkt"]), fx["nemo_nk_rnf"])
    assert fx["nemo_nk_rnf"].max() >= 10       # fixture exercises deep spreads


def test_whole_level_source_matches_nemo(fx):
    nk = jnp.asarray(_nk(fx))
    h_k = jnp.asarray(fx["e3t"] * fx["tmask"])
    mask = jnp.asarray(fx["tmask"][:, 0])
    rho_0 = 1.0 / float(fx["r1_rho0"])
    w, h_rnf = runoff_spread_layer_fractions(h_k, mask, RunoffSpreadLevels(nk))
    run = fx["zrnfcl_max"] > 0
    np.testing.assert_allclose(np.asarray(h_rnf)[run], fx["nemo_h_rnf"][run],
                               rtol=1e-12, atol=0.0)
    z = jnp.zeros_like(mask)
    fw = FreshwaterForcing(precip=z, evap=z, runoff=jnp.asarray(fx["rnf"]),
                           ice_fw=z, restoring=None)
    ent = np.asarray(runoff_entry_profile(fw, h_k, mask, rho_0,
                                          RunoffSpreadLevels(nk)))
    hk = np.asarray(h_k)
    port = np.where(hk > 0, ent / np.where(hk > 0, hk, 1.0), 0.0)
    lev = np.arange(hk.shape[-1])[None, :]
    act = (fx["rnf"] != 0)[:, None] & (lev < fx["nemo_nk_rnf"][:, None])
    nemo = np.where(act, (fx["rnf"] * fx["r1_rho0"] / fx["nemo_h_rnf"])[:, None], 0.0)
    np.testing.assert_allclose(port, nemo, rtol=1e-12, atol=0.0)


def test_whole_level_virtual_salt_conserves_column(fx):
    nk = jnp.asarray(_nk(fx))
    h_k = jnp.asarray(fx["e3t"] * fx["tmask"])
    mask = jnp.asarray(fx["tmask"][:, 0])
    z = jnp.zeros_like(mask)
    R = jnp.asarray(fx["rnf"])
    fw = FreshwaterForcing(precip=z, evap=z, runoff=R, ice_fw=z, restoring=z)
    dS = runoff_spread_virtual_salt_tendency_3d(
        fw, 35.0, h_k, 1026.0, mask, runoff_spread_m=RunoffSpreadLevels(nk))
    col = np.asarray(jnp.sum(dS * h_k, axis=-1))
    np.testing.assert_allclose(col, -35.0 * np.asarray(R) / 1026.0 * np.asarray(mask),
                               rtol=1e-12, atol=1e-30)


def test_resolver_levels_exclusive():
    class C:
        runoff_depth_spread_m = 0.0
        runoff_depth_spread_map = None
        runoff_depth_spread_levels = np.array([2, 3], np.int32)
    out = resolve_runoff_spread_arg(C)
    assert isinstance(out, RunoffSpreadLevels)
    C.runoff_depth_spread_map = np.array([10.0, 20.0])
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_runoff_spread_arg(C)


def test_level_count_rejects_bad_ladder():
    with pytest.raises(ValueError):
        nemo_runoff_level_count(np.ones((12, 2)), np.array([5.0, 1.0]))
    with pytest.raises(ValueError):
        nemo_runoff_level_count(np.ones(2), np.array([1.0, 5.0]))


def _nemo_literal(zr, gdept_0, mbkt, e3t, dep_max, rnf_max):
    """sbcrnf.F90:415-441 transcribed loop-for-loop (per column); returns (nk, h_rnf)."""
    h = dep_max / rnf_max * zr if zr > 0 else 1.0
    if zr > 0:
        h = min(h, gdept_0[mbkt - 1])
        jk = 2
        while jk < mbkt and gdept_0[jk - 1] < h:
            jk += 1
        nk = jk
    else:
        nk = 1
    return nk, float(sum(e3t[k] for k in range(nk)))


def test_whole_level_cap_partial_bottom_and_land(fx):
    """Synthetic columns that OVERFLOW the seafloor (uncapped nk > mbkt), with a
    partial bottom cell, a land column and a one-level column (KNOWN
    DIFFERENCE: NEMO counts its dry level 2; here all runoff stays in level 1)."""
    g = fx["gdept_1d"]
    deep = np.argmax(fx["nemo_nk_rnf"])
    e3_full = fx["e3t"][deep].copy()
    cols = []                                   # (mbkt, partial-bottom factor, wet)
    for mb in (2, 3, 5):
        cols += [(mb, 1.0, True), (mb, 0.4, True)]
    cols += [(4, 1.0, False), (1, 1.0, True)]
    n = len(cols)
    nlev = e3_full.size
    zr = np.full(n, 0.05)                       # h = 150 m: overflows every column above
    e3 = np.zeros((n, nlev)); tm = np.zeros((n, nlev))
    for c, (mb, fac, wet) in enumerate(cols):
        e3[c, :mb] = e3_full[:mb]; e3[c, mb - 1] *= fac
        tm[c, :mb] = 1.0 if wet else 0.0
    nk = jnp.asarray(nemo_runoff_level_count(
        zr[None, :], g, dep_max=float(fx["rn_dep_max"]), rnf_max=float(fx["rn_rnf_max"])))
    assert np.all(np.asarray(nk)[:-2] > np.array([c[0] for c in cols[:-2]]))  # overflow exercised
    h_k = jnp.asarray(e3 * tm); mask = jnp.asarray(tm[:, 0])
    w, h_rnf = runoff_spread_layer_fractions(h_k, mask, RunoffSpreadLevels(nk))
    w = np.asarray(w); h_rnf = np.asarray(h_rnf)
    for c, (mb, fac, wet) in enumerate(cols):
        if not wet:
            assert np.all(w[c] == 0.0) and h_rnf[c] == 0.0
            continue
        if mb == 1:                              # KNOWN DIFFERENCE (vote V3)
            assert w[c, 0] == 1.0 and np.all(w[c, 1:] == 0.0)
            np.testing.assert_allclose(h_rnf[c], e3[c, 0], rtol=1e-15)
            continue
        gd0 = g.copy()
        gd0[mb - 1] = g[mb - 2] + 0.5 * (e3[c, mb - 2] + e3[c, mb - 1])   # partial-cell T-depth
        nk_ref, h_ref = _nemo_literal(zr[c], gd0, mb, e3[c], float(fx["rn_dep_max"]),
                                      float(fx["rn_rnf_max"]))
        assert nk_ref == mb
        np.testing.assert_array_equal(w[c], (np.arange(nlev) < nk_ref).astype(float))
        np.testing.assert_allclose(h_rnf[c], h_ref, rtol=1e-12)
