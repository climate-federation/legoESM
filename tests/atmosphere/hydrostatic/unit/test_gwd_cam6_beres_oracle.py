"""CAM6 oracle pins for the Beres convective gravity-wave source.

1. ``gw_beres_src`` with the CAM6 kernel flags (``spectrum_shift="end_off"``,
   ``storm_speed_truncate=False``, ``hdepth_min_km=1.0``) reproduces a NumPy
   transcription of CAM6 ``gw_convect.F90::gw_beres_src`` (loop form:
   ``index_of_nearest``, real ``CS``, ``eoshift``, the ``min_hdepth`` gate)
   on a small synthetic table, rel 1e-10 — and the E3SM flags do NOT
   (non-vacuity).
2. ``load_mfcc_table`` orients the netcdf ``mfcc(PS, MW, HD)`` exactly as
   ``gw_init_beres`` reads it into ``mfcc(HD, -maxuh:maxuh, -ngwv:ngwv)``
   with the ``start=[1,1,ngwv_file-ngwv+1]`` phase-speed subset.
3. ``_source_level_index("interface_below_p")`` = CAM6's ``desc%k`` loop.
4. A ``+``-joined ``E3SMCAMConfig.source`` sums the per-source solves with
   their own efficiencies (CAM ``gw_tend`` accumulation).

Run with ``JAX_ENABLE_X64=1``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    E3SMBeresConfig,
    E3SMCAMConfig,
    E3SMFrontalConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import (
    _eoshift_rows,
    _source_level_index,
    e3sm_cam_gwd,
    gw_beres_src,
    load_mfcc_table,
)

from legoesm import constants

jax.config.update("jax_enable_x64", True)

PGWV, DC = 12, 2.5
MAXH, MAXUH = 20, 8
CF, AL, Z_HEAT_MAX, STORM_MIN = 20.0, 1.0e5, 20000.0, 10.0


def _fnint(x):
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def _index_of_nearest(x, grid):
    interfaces = (grid[:-1] + grid[1:]) / 2.0
    idx = 1
    for i, itf in enumerate(interfaces):
        if x > itf:
            idx = i + 2
    return idx


def _cam6_gw_beres_src_ref(u, v, netdt, zm, k, mfcc, hd_m, min_hdepth,
                           ngwv, dc, storm_shift=True):
    """gw_convect.F90 gw_beres_src, 1-based Fortran indexing transcribed.

    ``mfcc[hd_idx-1, uh+maxuh, l+ngwv]``; returns tau (ncol, 2ngwv+1, pver+1)
    at 0-based interface index ``topi`` (= Fortran interface topi+1).
    """
    ncol, pver = u.shape
    nwav = 2 * ngwv + 1
    maxuh = (mfcc.shape[1] - 1) // 2
    tau = np.zeros((ncol, nwav, pver + 1))
    hdepth = np.zeros(ncol)
    q0 = np.zeros(ncol)
    xv = np.zeros(ncol)
    yv = np.zeros(ncol)
    ubm = np.zeros((ncol, pver))
    ubi = np.zeros((ncol, pver + 1))
    # get_unit_vector at the source level (1-based k -> 0-based k-1)
    uc, vc = u[:, k - 1], v[:, k - 1]
    mag = np.sqrt(uc * uc + vc * vc)
    for i in range(ncol):
        if mag[i] > 0:
            xv[i], yv[i] = uc[i] / mag[i], vc[i] / mag[i]
    ubi[:, k] = mag                                  # ubi(:, desc%k+1)
    for kk in range(pver):
        ubm[:, kk] = u[:, kk] * xv + v[:, kk] * yv
    ubi[:, 0] = ubm[:, 0]
    ubi[:, 1:pver] = 0.5 * (ubm[:, :-1] + ubm[:, 1:])
    boti = np.zeros(ncol, dtype=int)
    topi = np.zeros(ncol, dtype=int)
    for kk in range(pver, 0, -1):                    # 1-based, surface up
        for i in range(ncol):
            if boti[i] == 0:
                if zm[i, kk - 1] >= Z_HEAT_MAX:
                    boti[i] = kk
                    topi[i] = kk
                elif netdt[i, kk - 1] > 0.0:
                    boti[i] = kk
            elif topi[i] == 0:
                if zm[i, kk - 1] >= Z_HEAT_MAX:
                    topi[i] = kk
                elif not (netdt[i, kk - 1] > 0.0):
                    topi[i] = kk
        if np.all(topi != 0):
            break
    assert np.all(topi != 0), "oracle precondition: every heating run terminates"
    for i in range(ncol):
        hdepth[i] = zm[i, topi[i] - 1] - zm[i, boti[i] - 1]
    hd_idx = np.array([_index_of_nearest(h, hd_m) for h in hdepth])
    hd_idx[hdepth < max(min_hdepth, hd_m[0])] = 0
    for kk in range(topi.min(), boti.max() + 1):
        sel = (kk >= topi) & (kk <= boti)
        q0[sel] = np.maximum(q0[sel], netdt[sel, kk - 1])
    maxq0 = q0 * 24.0 * 3600.0
    q0 = q0 * CF
    if storm_shift:
        cs = np.sign(ubm[:, k - 1]) * np.maximum(np.abs(ubm[:, k - 1]) - STORM_MIN, 0.0)
        cs = np.where(ubm[:, k - 1] >= 0, np.abs(cs), -np.abs(cs))
        uh = np.zeros(ncol)
        for kk in range(topi.min(), boti.max() + 1):
            sel = (kk >= topi) & (kk <= boti)
            uh[sel] += ubm[sel, kk - 1] / (boti[sel] - topi[sel] + 1)
        uh = uh - cs
    else:
        uh = ubm[:, k - 1]
    uh = np.minimum(uh, float(maxuh))
    uh = np.maximum(uh, -float(maxuh))
    umini = np.full(ncol, ngwv)
    umaxi = np.full(ncol, -ngwv)
    for kk in range(topi.min(), boti.max() + 1):
        sel = (kk >= topi) & (kk <= boti)
        for i in np.where(sel)[0]:
            umini[i] = min(umini[i], _fnint(ubm[i, kk - 1] / dc))
            umaxi[i] = max(umaxi[i], _fnint(ubm[i, kk - 1] / dc))
    umini = np.maximum(umini, -ngwv)
    umaxi = np.minimum(umaxi, ngwv)
    for i in range(ncol):
        if hd_idx[i] > 0:
            tau0 = mfcc[hd_idx[i] - 1, _fnint(uh[i]) + maxuh, :].copy()
            if storm_shift:
                shift = -_fnint(cs[i] / dc)
                # eoshift: out(j) = in(j+shift) if in range else 0
                out = np.zeros_like(tau0)
                for j in range(nwav):
                    if 0 <= j + shift < nwav:
                        out[j] = tau0[j + shift]
                tau0 = out
            tau0 = tau0 * q0[i] * q0[i] / AL
            lo, hi = umini[i] + ngwv, umaxi[i] + ngwv
            if lo <= hi:
                tau0[lo:hi + 1] = 0.0
            tau[i, :, topi[i]] = tau0
    cref = np.arange(-ngwv, ngwv + 1) * dc
    return dict(tau=tau, src_level=topi, xv=xv, yv=yv, ubm=ubm, ubi=ubi,
                c=np.broadcast_to(cref, (ncol, nwav)), hdepth=hdepth, maxq0=maxq0)


def _synthetic(pver=30, seed=3):
    rng = np.random.default_rng(seed)
    ncol = 8
    zm = np.linspace(26000.0, 100.0, pver)[None, :] * np.ones((ncol, 1))
    zm = zm * rng.uniform(0.97, 1.03, (ncol, 1))
    u = rng.uniform(-6.0, 6.0, (ncol, pver))
    v = rng.uniform(-3.0, 3.0, (ncol, pver))
    netdt = np.zeros((ncol, pver))
    for i in range(ncol):
        zlo, zhi = rng.uniform(200.0, 1500.0), rng.uniform(4000.0, 14000.0)
        sel = (zm[i] > zlo) & (zm[i] < zhi)
        netdt[i, sel] = rng.uniform(1e-5, 8e-4, sel.sum())
    netdt[1, :] = 0.0                                          # no convection
    sel = (zm[2] > 200.0) & (zm[2] < 900.0)                    # too shallow
    netdt[2, :] = 0.0
    netdt[2, sel] = 3e-4
    netdt[3, zm[3] > 300.0] = 2e-4                             # reaches 20 km window
    netdt[6, :] = 0.0                                          # 1-2.5 km: CAM6 launches,
    netdt[6, (zm[6] > 300.0) & (zm[6] < 2100.0)] = 4e-4        # E3SM's 2.5 km floor not
    # uniform wind through the heating range -> Umini == Umaxi: Fortran's
    # tau0(Umini:Umaxi)=0 still zeroes ONE bin (codex: the port skipped it)
    u[7, :], v[7, :] = 4.0, 0.0
    # strong source-level winds in both directions -> real CS, big shifts
    k = 22                                                     # 1-based source level
    u[4, k - 1], v[4, k - 1] = 23.7, 1.0
    u[5, k - 1], v[5, k - 1] = -19.3, -2.0
    mfcc = rng.uniform(0.1, 1.0, (MAXH, 2 * MAXUH + 1, 2 * PGWV + 1))
    return ncol, pver, u, v, netdt, zm, mfcc, k


def _beres(**kw):
    base = dict(cf=CF, al=AL, z_heat_max=Z_HEAT_MAX, storm_speed_min=STORM_MIN,
                maxh=MAXH, maxuh=MAXUH, use_stand_in_table=False)
    base.update(kw)
    return E3SMBeresConfig(**base)


CAM6 = dict(spectrum_shift="end_off", storm_speed_truncate=False, hdepth_min_km=1.0,
            hd_index_rule="nearest_grid")


def _run_kernel(beres, u, v, netdt, zm, mfcc, k):
    ncol = u.shape[0]
    lat = jnp.zeros(ncol)
    return gw_beres_src(
        jnp.asarray(u), jnp.asarray(v), jnp.asarray(netdt), jnp.asarray(zm),
        lat, k - 1, jnp.asarray(mfcc), PGWV, DC, beres)


def test_cam6_flags_pin_gw_convect_transcription():
    ncol, pver, u, v, netdt, zm, mfcc, k = _synthetic()
    hd_m = 1000.0 * np.arange(1, MAXH + 1)
    ref = _cam6_gw_beres_src_ref(u, v, netdt, zm, k, mfcc, hd_m, 1000.0, PGWV, DC)
    out = _run_kernel(_beres(**CAM6), u, v, netdt, zm, mfcc, k)
    tau, src_level, tend_level, xv, yv, c, ubm, ubi, hdepth, maxq0 = out
    assert ref["tau"].any(), "synthetic case launches nothing; oracle vacuous"
    np.testing.assert_allclose(np.asarray(tau), ref["tau"], rtol=1e-10, atol=0.0)
    np.testing.assert_array_equal(np.asarray(src_level), ref["src_level"])
    np.testing.assert_allclose(np.asarray(xv), ref["xv"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(yv), ref["yv"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(ubm), ref["ubm"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(ubi)[:, :-1], ref["ubi"][:, :-1], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(c), ref["c"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(hdepth) * 1000.0, ref["hdepth"], rtol=1e-10)
    np.testing.assert_allclose(np.asarray(maxq0), ref["maxq0"], rtol=1e-12)
    # the regime coverage the pin relies on
    assert (ref["tau"][1] == 0).all() and (ref["tau"][2] == 0).all()
    assert ref["tau"][4].any() and ref["tau"][5].any()
    # equal-bounds critical-level column: exactly one bin zeroed at the launch level
    t7 = ref["tau"][7, :, ref["src_level"][7]]
    assert t7.any() and (t7 == 0.0).sum() == 1


def test_e3sm_flags_are_not_the_cam6_law():
    """Non-vacuity: circular shift / truncated CS / 2.5 km floor differ."""
    ncol, pver, u, v, netdt, zm, mfcc, k = _synthetic()
    hd_m = 1000.0 * np.arange(1, MAXH + 1)
    ref = _cam6_gw_beres_src_ref(u, v, netdt, zm, k, mfcc, hd_m, 1000.0, PGWV, DC)
    for flags in (dict(spectrum_shift="circular"), dict(storm_speed_truncate=True),
                  dict(hdepth_min_km=2.5)):
        kw = dict(CAM6)
        kw.update(flags)
        tau = np.asarray(_run_kernel(_beres(**kw), u, v, netdt, zm, mfcc, k)[0])
        assert not np.allclose(tau, ref["tau"], rtol=1e-6, atol=0.0), flags


def test_half_km_tie_follows_index_of_nearest_not_nint():
    """hdepth exactly 1.5 km: CAM6's index_of_nearest takes row 1 (x > 1500 is
    false), E3SM's NINT takes row 2.  Random depths never hit the tie."""
    pver = 30
    ncol = 2
    zm = (16000.0 - 500.0 * np.arange(pver))[None, :] * np.ones((ncol, 1))   # 16 km .. 1.5 km
    u = np.full((ncol, pver), 3.0)
    v = np.zeros((ncol, pver))
    netdt = np.zeros((ncol, pver))
    netdt[:, (zm[0] >= 2000.0) & (zm[0] <= 3000.0)] = 2e-4   # topi at 3500 m
    rng = np.random.default_rng(1)
    mfcc = rng.uniform(0.1, 1.0, (MAXH, 2 * MAXUH + 1, 2 * PGWV + 1))
    k = 25
    hd_m = 1000.0 * np.arange(1, MAXH + 1)
    ref = _cam6_gw_beres_src_ref(u, v, netdt, zm, k, mfcc, hd_m, 1000.0, PGWV, DC)
    assert ref["hdepth"][0] == 1500.0
    cam = _run_kernel(_beres(**CAM6), u, v, netdt, zm, mfcc, k)
    np.testing.assert_allclose(np.asarray(cam[0]), ref["tau"], rtol=1e-10, atol=0.0)
    e3sm = _run_kernel(_beres(**dict(CAM6, hd_index_rule="nint")), u, v, netdt, zm, mfcc, k)
    assert not np.allclose(np.asarray(e3sm[0]), ref["tau"], rtol=1e-6, atol=0.0)
    with pytest.raises(ValueError, match="hd_index_rule"):
        _run_kernel(_beres(**dict(CAM6, hd_index_rule="bogus")), u, v, netdt, zm, mfcc, k)


def test_equal_bounds_filter_also_pinned_on_the_e3sm_flags():
    """Fortran tau0(Umini:Umaxi)=0 zeroes one bin for Umini == Umaxi in E3SM
    too (same slice); pin it on the E3SM flag set so a revert to ``>`` fails."""
    ncol, pver, u, v, netdt, zm, mfcc, k = _synthetic()
    out = _run_kernel(_beres(), u, v, netdt, zm, mfcc, k)      # E3SM defaults
    tau = np.asarray(out[0])
    lvl = int(np.asarray(out[1])[7])
    t7 = tau[7, :, lvl]
    assert t7.any() and (t7 == 0.0).sum() == 1


_REAL_TABLE = ("/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/docs/references/"
               "cam6_inputdata/newmfspectra40_dc25.nc")


@pytest.mark.skipif(not __import__("os").path.exists(_REAL_TABLE),
                    reason="real CAM6 Beres table not available here")
def test_real_cam6_table_loads_with_the_default_beres_dimensions():
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import e3sm_mfcc_table
    tbl = load_mfcc_table(_REAL_TABLE, 32)
    assert tbl.shape == (20, 81, 65) and np.isfinite(tbl).all() and not tbl.flags.writeable
    g = GravityWaveDragConfig(
        scheme="e3sm_cam",
        e3sm_cam=E3SMCAMConfig(source="convective", pgwv=32,
                               beres=E3SMBeresConfig(mfcc_table_path=_REAL_TABLE)))
    assert e3sm_mfcc_table(g) is tbl                    # defaults maxh=20/maxuh=40 match
    # and the kernel accepts it end to end
    args, kw = _column_state()
    cfg = E3SMCAMConfig(source="convective", pgwv=32,
                        beres=E3SMBeresConfig(hdepth_min_km=1.0))
    out = e3sm_cam_gwd(*args, 900.0, cfg, mfcc_table=tbl, netdt_col=kw["netdt_col"])
    assert np.all(np.isfinite(np.asarray(out.du_dt)))


def test_eoshift_rows_matches_fortran_semantics():
    a = jnp.asarray(np.arange(1.0, 6.0)[None, :].repeat(3, 0))
    out = np.asarray(_eoshift_rows(a, jnp.asarray([0, 2, -1])))
    np.testing.assert_array_equal(out[0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(out[1], [3, 4, 5, 0, 0])
    np.testing.assert_array_equal(out[2], [0, 1, 2, 3, 4])


def test_unknown_shift_and_level_rule_raise():
    ncol, pver, u, v, netdt, zm, mfcc, k = _synthetic()
    with pytest.raises(ValueError, match="spectrum_shift"):
        _run_kernel(_beres(spectrum_shift="bogus"), u, v, netdt, zm, mfcc, k)
    with pytest.raises(ValueError, match="source_level_rule"):
        _source_level_index(jnp.ones((2, 5)), jnp.ones((2, 6)), 7e4, "bogus")


def test_source_level_interface_rule_matches_cam6_loop():
    # pref_edge top-down; CAM: do k=0,pver: if pref_edge(k+1) < 70000 desc%k = k+1
    pint = np.array([[100.0, 5000.0, 30000.0, 60000.0, 69000.0, 75000.0, 100000.0]])
    pmid = 0.5 * (pint[:, 1:] + pint[:, :-1])
    desc_k = 0
    for kk in range(0, pmid.shape[1] + 1):
        if pint[0, kk] < 70000.0:           # pref_edge(k+1) 1-based == pint[k] 0-based
            desc_k = kk + 1
    got = int(_source_level_index(jnp.asarray(pmid), jnp.asarray(pint), 7e4,
                                  "interface_below_p"))
    assert got == desc_k - 1 == 4
    near = int(_source_level_index(jnp.asarray(pmid), jnp.asarray(pint), 7e4,
                                   "nearest_midpoint"))
    assert near == 4  # 72000 is the nearest midpoint here (same level, by chance)
    # a case where the two rules differ: interfaces 62000/69500/90000 ->
    # CAM picks the 69500-90000 layer (midpoint 79750); "nearest" picks 65750
    pint2 = np.array([[100.0, 5000.0, 30000.0, 62000.0, 69500.0, 90000.0, 100000.0]])
    pmid2 = 0.5 * (pint2[:, 1:] + pint2[:, :-1])
    assert int(_source_level_index(jnp.asarray(pmid2), jnp.asarray(pint2), 7e4,
                                   "interface_below_p")) == 4
    assert int(_source_level_index(jnp.asarray(pmid2), jnp.asarray(pint2), 7e4,
                                   "nearest_midpoint")) == 3


def test_load_mfcc_table_orientation_matches_gw_init_beres(tmp_path):
    import xarray as xr
    n_ps, n_mw, n_hd = 29, 17, MAXH                         # ngwv_file = 14, maxuh = 8
    ps_i, mw_i, hd_i = np.meshgrid(np.arange(n_ps), np.arange(n_mw), np.arange(n_hd),
                                   indexing="ij")
    data = ps_i * 1e4 + mw_i * 1e2 + hd_i                    # encodes its own indices
    path = tmp_path / "mfcc_synth.nc"
    xr.Dataset(
        {"mfcc": (("PS", "MW", "HD"), data.astype(np.float64))},
        coords={"HD": np.arange(1, n_hd + 1, dtype=np.float64),
                "PS": np.arange(n_ps, dtype=np.float64),
                "MW": np.arange(n_mw, dtype=np.float64)},
    ).to_netcdf(path)
    tbl = load_mfcc_table(str(path), PGWV)
    ngwv_file = (n_ps - 1) // 2
    assert tbl.shape == (MAXH, 2 * MAXUH + 1, 2 * PGWV + 1)
    for h in range(1, MAXH + 1):
        for uh in range(-MAXUH, MAXUH + 1):
            for ell in range(-PGWV, PGWV + 1):
                # Fortran: mfcc(h, uh, l) <- file(PS = ngwv_file-ngwv+1 + (l+ngwv),
                #                                 MW = uh+maxuh+1, HD = h)
                ps_1based = (ngwv_file - PGWV + 1) + (ell + PGWV)
                want = (ps_1based - 1) * 1e4 + (uh + MAXUH) * 1e2 + (h - 1)
                assert tbl[h - 1, uh + MAXUH, ell + PGWV] == want
    with pytest.raises(ValueError, match="does not cover"):
        load_mfcc_table(str(path), ngwv_file + 1)
    bad = tmp_path / "bad.nc"
    xr.Dataset({"mfcc": (("HD", "MW", "PS"), np.zeros((n_hd, n_mw, n_ps)))},
               coords={"HD": np.arange(1, n_hd + 1, dtype=np.float64)}).to_netcdf(bad)
    assert load_mfcc_table(str(bad), 1).shape == (n_hd, n_mw, 3)   # dim ORDER free
    bad2 = tmp_path / "bad2.nc"
    xr.Dataset({"mfcc": (("X", "MW", "PS"), np.zeros((n_hd, n_mw, n_ps)))}).to_netcdf(bad2)
    with pytest.raises(ValueError, match="expected dims"):
        load_mfcc_table(str(bad2), 1)


def test_real_table_index_arithmetic_through_the_kernel(tmp_path):
    """The loaded table row the kernel reads for (hdepth, uh) is the file's
    (HD=hdepth, MW=uh+maxuh, PS=ngwv_file+l) element -> whole-chain check."""
    import xarray as xr
    n_ps, n_mw = 29, 2 * MAXUH + 1
    ps_i, mw_i, hd_i = np.meshgrid(np.arange(n_ps), np.arange(n_mw), np.arange(MAXH),
                                   indexing="ij")
    data = 1.0 + ps_i * 1e4 + mw_i * 1e2 + hd_i
    path = tmp_path / "t.nc"
    xr.Dataset({"mfcc": (("PS", "MW", "HD"), data)},
               coords={"HD": np.arange(1, MAXH + 1, dtype=np.float64)}).to_netcdf(path)
    tbl = load_mfcc_table(str(path), PGWV)
    ncol, pver, u, v, netdt, zm, _, k = _synthetic()
    hd_m = 1000.0 * np.arange(1, MAXH + 1)
    ref = _cam6_gw_beres_src_ref(u, v, netdt, zm, k, tbl, hd_m, 1000.0, PGWV, DC)
    tau = np.asarray(_run_kernel(_beres(**CAM6), u, v, netdt, zm, tbl, k)[0])
    np.testing.assert_allclose(tau, ref["tau"], rtol=1e-10, atol=0.0)


# --- multi-source driver ------------------------------------------------------

def _column_state(ncol=6, nlev=30, seed=11):
    rng = np.random.default_rng(seed)
    pint = np.linspace(100.0, 1.0e5, nlev + 1)[None, :] * np.ones((ncol, 1))
    pmid = 0.5 * (pint[:, 1:] + pint[:, :-1])
    T = 220.0 + 70.0 * (pmid / 1.0e5) + rng.uniform(-2, 2, (ncol, nlev))
    z_half = 7000.0 * np.log(1.0e5 / np.maximum(pint, 1.0))
    z_full = 0.5 * (z_half[:, 1:] + z_half[:, :-1])
    rho = pmid / (constants.R_d * T)
    u = rng.uniform(-20, 20, (ncol, nlev))
    v = rng.uniform(-5, 5, (ncol, nlev))
    lat = np.linspace(-1.2, 1.2, ncol)
    netdt = np.zeros((ncol, nlev))
    netdt[:, (z_full[0] > 1000) & (z_full[0] < 9000)] = 3e-4
    frontgf = rng.uniform(0.0, 5e-15, (ncol, nlev))
    h_topo = rng.uniform(50.0, 400.0, ncol)
    args = [jnp.asarray(a) for a in (u, v, T, pmid, pint, z_full, z_half, rho, lat)]
    return args, dict(h_topo_col=jnp.asarray(h_topo), frontgf_col=jnp.asarray(frontgf),
                      netdt_col=jnp.asarray(netdt))


def test_multi_source_sums_single_sources_with_own_effgw():
    args, kw = _column_state()
    beres = E3SMBeresConfig(effgw=0.4, hdepth_min_km=1.0)
    cfg = E3SMCAMConfig(source="orographic+frontal+convective", pgwv=PGWV,
                        effgw=0.125, frontal=E3SMFrontalConfig(effgw=1.0),
                        beres=beres)
    dt = 900.0
    out = e3sm_cam_gwd(*args, dt, cfg, **kw)
    parts = [e3sm_cam_gwd(*args, dt, cfg._replace(source=s, effgw=e), **kw)
             for s, e in (("orographic", 0.125), ("frontal", 1.0), ("convective", 0.4))]
    # momentum and dissipation are plain sums (the fixers' momentum part does
    # not depend on the accumulated wind)
    for f in ("du_dt", "dv_dt", "eps_gwd"):
        want = sum(np.asarray(getattr(p, f)) for p in parts)
        np.testing.assert_allclose(np.asarray(getattr(out, f)), want, rtol=1e-12, atol=1e-30)
        assert np.any(want != 0.0), f
    for p in parts:
        assert np.any(np.asarray(p.du_dt) != 0.0)
    # heating is NOT a plain sum: CAM's energy fixer acts on the accumulated
    # ptend, so the multi-source column total-energy budget must close while
    # the naive sum of separately-closed sources leaves dt*du_a*du_b open.
    u, v, T, pmid, pint = (np.asarray(a) for a in args[:5])
    mass = np.diff(pint, axis=1) / constants.g
    def budget(du, dv, dT):
        return np.sum(mass * (constants.c_pd * dT + du * (u + 0.5 * dt * du)
                              + dv * (v + 0.5 * dt * dv)), axis=1)
    b_multi = budget(*(np.asarray(getattr(out, f)) for f in ("du_dt", "dv_dt", "dT_dt")))
    scale = np.max(np.abs(mass * constants.c_pd * np.asarray(out.dT_dt)))
    assert np.max(np.abs(b_multi)) < 1e-10 * scale, (b_multi, scale)
    # two OVERLAPPING spectral sources (codex probe): the naive sum of the
    # separately-closed sources leaves the dt*du_a*du_b cross term open
    c2 = cfg._replace(source="frontal+background", effgw=1.0,
                      frontal=E3SMFrontalConfig(effgw=1.0))
    o2 = e3sm_cam_gwd(*args, dt, c2, **kw)
    p2 = [e3sm_cam_gwd(*args, dt, c2._replace(source=s), **kw)
          for s in ("frontal", "background")]
    b2 = budget(*(np.asarray(getattr(o2, f)) for f in ("du_dt", "dv_dt", "dT_dt")))
    b2_naive = budget(np.asarray(o2.du_dt), np.asarray(o2.dv_dt),
                      sum(np.asarray(p.dT_dt) for p in p2))
    scale2 = np.max(np.abs(mass * constants.c_pd * np.asarray(o2.dT_dt)))
    assert np.max(np.abs(b2)) < 1e-10 * scale2, (b2, scale2)
    assert np.max(np.abs(b2_naive)) > 1e4 * np.max(np.abs(b2)), (b2_naive, b2)
    # per-source effgw matters: dropping it changes the frontal part
    alt = e3sm_cam_gwd(*args, dt, cfg._replace(frontal=E3SMFrontalConfig()), **kw)
    assert not np.allclose(np.asarray(alt.du_dt), np.asarray(out.du_dt))
    # landfrac scales ONLY the orographic source (gw_drag.F90:904-906): the
    # frontal+convective drag over a pure-ocean column must survive
    ocean = e3sm_cam_gwd(*args, dt, cfg._replace(source="frontal+convective"),
                         land_frac_col=jnp.zeros(u.shape[0]), **kw)
    assert np.any(np.asarray(ocean.du_dt) != 0.0)
    oro_ocean = e3sm_cam_gwd(*args, dt, cfg._replace(source="orographic"),
                             land_frac_col=jnp.zeros(u.shape[0]), **kw)
    assert not np.any(np.asarray(oro_ocean.du_dt))
    # effgw stays differentiable through the multi-source driver
    def loss(eff):
        c2 = cfg._replace(frontal=cfg.frontal._replace(effgw=eff))
        return jnp.sum(e3sm_cam_gwd(*args, dt, c2, **kw).du_dt ** 2)
    g = jax.grad(loss)(jnp.asarray(1.0))
    assert np.isfinite(float(g)) and float(g) != 0.0


def test_multi_source_rejects_duplicates_and_unknown():
    args, kw = _column_state()
    with pytest.raises(ValueError, match="Duplicate"):
        e3sm_cam_gwd(*args, 900.0, E3SMCAMConfig(source="frontal+frontal", pgwv=PGWV), **kw)
    with pytest.raises(ValueError, match="Unknown E3SM GWD source"):
        e3sm_cam_gwd(*args, 900.0, E3SMCAMConfig(source="frontal+bogus", pgwv=PGWV), **kw)
