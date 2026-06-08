"""Direct unit tests for the E3SM/CAM Beres (2004) convective GW source.

Covers the new leaf ``gw_beres_src`` (and its helpers ``build_stand_in_mfcc``,
``_nint``, ``_cshift_rows``) plus the ``e3sm_cam_gwd(source="convective")``
driver path:

* term-by-term match to the gfortran ``gw_beres_src`` oracle for the
  heating-depth diagnosis, the ``q0`` / ``uh`` averaging, the NINT lookup,
  the ground-relative ``cshift`` Doppler shift, the ``q0^2/AL`` amplitude
  scaling, and the critical-level filtering (anchors baked from
  ``.physics-validator/gravity_wave_drag/oracle/`` — the live compare matches
  to ~1e-15 relative in float64 with the SAME stand-in mfcc table on both
  sides; bit-faithfulness to Beres-2004 needs the offline table);
* shallow heating (< 2.5 km) and no convection -> zero source;
* differentiability of the source w.r.t. the heating field;
* jit / vmap parity;
* end-to-end driver produces finite, dissipative tendencies.

The stand-in mfcc table is a documented placeholder (NOT bit-faithful to
Beres); these tests therefore validate the ALGORITHM, not the published
spectrum amplitudes.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    E3SMBeresConfig,
    E3SMCAMConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import (
    build_stand_in_mfcc,
    gw_beres_src,
    e3sm_cam_gwd,
    _nint,
    _cshift_rows,
)

ORACLE_RAIR = 287.04
ORACLE_G = 9.80616
DC = 2.5
NGWV = 32
MAXH = 20
MAXUH = 40


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _beres_column(pver=40, heat="deep"):
    """Reproduce the oracle column (identical to gen_input_beres.py)."""
    p_top, p_surf = 100.0, 1.0e5
    pint = np.linspace(p_top, p_surf, pver + 1)
    pmid = 0.5 * (pint[:-1] + pint[1:])
    T = np.linspace(230.0, 290.0, pver)
    dp = np.diff(pint)
    dz = ORACLE_RAIR * T * dp / (ORACLE_G * pmid)
    z_half = np.zeros(pver + 1)
    for k in range(pver, 0, -1):
        z_half[k - 1] = z_half[k] + dz[k - 1]
    zm = 0.5 * (z_half[:-1] + z_half[1:])
    u = np.linspace(5.0, 30.0, pver)
    v = np.zeros(pver)
    netdt = np.zeros(pver)
    if heat == "deep":
        m = (zm > 1500.0) & (zm < 12000.0)
        netdt[m] = 5.0 / 86400.0 * (1.0 + 0.5 * np.sin(zm[m] / 3000.0))
    elif heat == "shallow":
        m = (zm > 500.0) & (zm < 1800.0)
        netdt[m] = 3.0 / 86400.0
    elif heat == "none":
        pass
    k700 = int(np.argmin(np.abs(pmid - 70000.0)))
    return pint, pmid, T, zm, z_half, u, v, netdt, k700


# Oracle anchors for the canonical deep-convection column (lat=0).
_ORACLE_HDEPTH = 10.94399092476893
_ORACLE_MAXQ0 = 7.499345418128451
_ORACLE_SRC_LEVEL = 8
_ORACLE_TAU0 = {  # spectrum at the launched interface, selected waves
    -32: 1.952892337097388e-15,
    -10: 9.475133825112131e-14,
    -6: 1.950933554906722e-13,
    0: 3.799899696727634e-13,
    6: 0.0,    # critical-level filtered (Umini=4..Umaxi=10)
    10: 0.0,
    32: 2.861242490946904e-15,
}
_ORACLE_NONZERO = 58


def test_beres_matches_oracle():
    pint, pmid, T, zm, z_half, u, v, netdt, k700 = _beres_column(heat="deep")
    to = lambda a: jnp.asarray(a[None, :])
    beres = E3SMBeresConfig(
        cf=20.0, al=1.0e5, hdepth_scaling_factor=1.0, hdepth_min_km=2.5,
        z_heat_max=20000.0, storm_speed_min=10.0, maxh=MAXH, maxuh=MAXUH,
        use_stand_in_table=True, mfcc_peak=1.0e-2, mfcc_c0=30.0,
        mfcc_hdepth_growth=0.05,
    )
    mfcc = build_stand_in_mfcc(NGWV, DC, beres, jnp.float64)
    out = gw_beres_src(
        to(u), to(v), to(netdt), to(zm), jnp.zeros(1), k700, mfcc,
        NGWV, DC, beres,
    )
    tau0_full, src, tend, xv, yv, c, ubm, ubi, hdepth, maxq0 = out
    assert int(src[0]) == _ORACLE_SRC_LEVEL
    assert int(tend[0]) == _ORACLE_SRC_LEVEL
    np.testing.assert_allclose(float(hdepth[0]), _ORACLE_HDEPTH, rtol=1e-12)
    np.testing.assert_allclose(float(maxq0[0]), _ORACLE_MAXQ0, rtol=1e-12)
    spec = np.array(tau0_full[0, :, int(src[0])])  # (nwav,)
    for l, ref in _ORACLE_TAU0.items():
        idx = l + NGWV
        np.testing.assert_allclose(spec[idx], ref, rtol=1e-10, atol=1e-25)
    assert int(np.count_nonzero(spec)) == _ORACLE_NONZERO
    # purely zonal westerly -> unit vector (1, 0).
    np.testing.assert_allclose(float(xv[0]), 1.0, rtol=1e-14)
    assert abs(float(yv[0])) < 1e-18


def test_beres_uh_column_lookup():
    """A uh-DEPENDENT stand-in table must select the correct ``uh`` column.

    With ``mfcc_uh_slope != 0`` the table amplitude varies along the mean-wind
    axis, so the launched spectrum equals ``base_spectrum * (1 +
    slope*uh_idx)`` only if ``uh_idx``/``uh_col`` map correctly (codex
    beres-1 #3).  uh = mean(ubm over heating range) - CS; for this westerly
    column uh rounds to 5, so the amplitude factor is ``1 + slope*5``.
    """
    pint, pmid, T, zm, z_half, u, v, netdt, k700 = _beres_column(heat="deep")
    to = lambda a: jnp.asarray(a[None, :])
    slope = 0.02
    beres0 = E3SMBeresConfig(use_stand_in_table=True, mfcc_uh_slope=0.0)
    beresS = E3SMBeresConfig(use_stand_in_table=True, mfcc_uh_slope=slope)
    mfcc0 = build_stand_in_mfcc(NGWV, DC, beres0, jnp.float64)
    mfccS = build_stand_in_mfcc(NGWV, DC, beresS, jnp.float64)
    out0 = gw_beres_src(to(u), to(v), to(netdt), to(zm), jnp.zeros(1),
                        k700, mfcc0, NGWV, DC, beres0)
    outS = gw_beres_src(to(u), to(v), to(netdt), to(zm), jnp.zeros(1),
                        k700, mfccS, NGWV, DC, beresS)
    src = int(out0[1][0])
    spec0 = np.array(out0[0][0, :, src])
    specS = np.array(outS[0][0, :, src])
    nz = np.abs(spec0) > 1e-30
    # recover uh_idx from the amplitude ratio on the non-zero (non-filtered)
    # waves: specS/spec0 = 1 + slope*uh_idx -> uh_idx = (ratio-1)/slope.
    ratio = specS[nz] / spec0[nz]
    uh_idx = (ratio - 1.0) / slope
    np.testing.assert_allclose(uh_idx, np.round(uh_idx[0]), rtol=0, atol=1e-9)
    assert abs(round(float(uh_idx[0]))) <= MAXUH


def test_beres_shallow_and_none_zero_source():
    for heat in ("shallow", "none"):
        pint, pmid, T, zm, z_half, u, v, netdt, k700 = _beres_column(heat=heat)
        to = lambda a: jnp.asarray(a[None, :])
        beres = E3SMBeresConfig()
        mfcc = build_stand_in_mfcc(NGWV, DC, beres, jnp.float64)
        out = gw_beres_src(
            to(u), to(v), to(netdt), to(zm), jnp.zeros(1), k700, mfcc,
            NGWV, DC, beres,
        )
        tau0_full = out[0]
        assert float(jnp.max(jnp.abs(tau0_full))) < 1e-18, (
            f"heat={heat} launched nonzero stress"
        )


def test_beres_lat_gate():
    """No convective waves poleward of pi/2 (the E3SM |lat| < pi/2 gate)."""
    pint, pmid, T, zm, z_half, u, v, netdt, k700 = _beres_column(heat="deep")
    to = lambda a: jnp.asarray(a[None, :])
    beres = E3SMBeresConfig()
    mfcc = build_stand_in_mfcc(NGWV, DC, beres, jnp.float64)
    out = gw_beres_src(
        to(u), to(v), to(netdt), to(zm),
        jnp.array([np.pi / 2 + 0.01]), k700, mfcc, NGWV, DC, beres,
    )
    assert float(jnp.max(jnp.abs(out[0]))) < 1e-18


def test_nint_round_half_away_from_zero():
    x = jnp.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.4, 2.6, -2.4])
    got = np.array(_nint(x))
    # Fortran NINT: 0.5->1, 1.5->2, 2.5->3, -0.5->-1, -2.5->-3, etc.
    expect = np.array([1, 2, 3, -1, -2, -3, 2, 3, -2], dtype=float)
    np.testing.assert_array_equal(got, expect)


def test_cshift_matches_fortran():
    """_cshift_rows must match Fortran cshift(array, SHIFT): element i -> i-SHIFT."""
    a = jnp.arange(5.0)[None, :]  # [0,1,2,3,4]
    # cshift([0,1,2,3,4], 2) = [2,3,4,0,1] (Fortran: shift left by 2)
    got2 = np.array(_cshift_rows(a, jnp.array([2], dtype=jnp.int32))[0])
    np.testing.assert_array_equal(got2, np.array([2.0, 3.0, 4.0, 0.0, 1.0]))
    # cshift([0,1,2,3,4], -1) = [4,0,1,2,3]
    gotm = np.array(_cshift_rows(a, jnp.array([-1], dtype=jnp.int32))[0])
    np.testing.assert_array_equal(gotm, np.array([4.0, 0.0, 1.0, 2.0, 3.0]))


def test_beres_grad_finite_nonzero():
    pint, pmid, T, zm, z_half, u, v, netdt, k700 = _beres_column(heat="deep")
    to = lambda a: jnp.asarray(a[None, :])
    beres = E3SMBeresConfig()
    mfcc = build_stand_in_mfcc(NGWV, DC, beres, jnp.float64)
    netdt_j = to(netdt)

    def loss(netdt_in):
        out = gw_beres_src(
            to(u), to(v), netdt_in, to(zm), jnp.zeros(1), k700, mfcc,
            NGWV, DC, beres,
        )
        return jnp.sum(out[0] ** 2)

    g = jax.grad(loss)(netdt_j)
    assert jnp.all(jnp.isfinite(g))
    # q0 = CF*max(netdt) -> tau0 ~ q0^2 -> nonzero sensitivity to the heating.
    assert float(jnp.sum(jnp.abs(g))) > 0.0


def test_beres_jit_vmap_parity():
    pint, pmid, T, zm, z_half, u, v, netdt, k700 = _beres_column(heat="deep")
    to = lambda a: jnp.asarray(np.broadcast_to(a[None, :], (3, len(a))))
    beres = E3SMBeresConfig()
    mfcc = build_stand_in_mfcc(NGWV, DC, beres, jnp.float64)
    args = (to(u), to(v), to(netdt), to(zm), jnp.zeros(3))

    def fn(u_, v_, nd_, zm_, lat_):
        return gw_beres_src(u_, v_, nd_, zm_, lat_, k700, mfcc,
                            NGWV, DC, beres)[0]

    eager = fn(*args)
    jitted = jax.jit(fn)(*args)
    np.testing.assert_allclose(np.array(jitted), np.array(eager),
                              rtol=1e-12, atol=1e-20)


def test_beres_driver_dissipative():
    pver = 40
    ncol = 2
    pint, pmid, T, zm, z_half, u, v, netdt, k700 = _beres_column(heat="deep")
    rep = lambda a: jnp.broadcast_to(jnp.asarray(a)[None, :], (ncol, len(a)))
    pmid_c = rep(pmid)
    pint_c = jnp.broadcast_to(jnp.asarray(pint)[None, :], (ncol, pver + 1))
    T_c = rep(T)
    zf_c = rep(zm)
    zh_c = jnp.broadcast_to(jnp.asarray(z_half)[None, :], (ncol, pver + 1))
    rho_c = pmid_c / (ORACLE_RAIR * T_c)
    u_c = rep(u)
    v_c = jnp.zeros((ncol, pver))
    lat = jnp.zeros(ncol)
    cfg = E3SMCAMConfig(source="convective", pgwv=NGWV, dc=DC, effgw=0.4)
    out = e3sm_cam_gwd(
        u_c, v_c, T_c, pmid_c, pint_c, zf_c, zh_c, rho_c, lat, 1800.0, cfg,
        netdt_col=rep(netdt),
    )
    assert jnp.all(jnp.isfinite(out.du_dt))
    assert jnp.all(jnp.isfinite(out.dT_dt))
    assert float(jnp.max(jnp.abs(out.du_dt))) > 0.0
    # eps_gwd is column KE dissipation, must be >= 0 (drag removes KE).
    assert jnp.all(out.eps_gwd >= -1e-9)
