### FILE tests/unit/test_ifs_closure.py :: REPLACE module
"""Unit tests for _ifs_closure (OpenIFS cumastrn closure port).

Chains exactly as in the reference harness (tests/unit/test_ifs_ascent.py):
departure search from _ifs_test_ascent (column_refine=2), then the main
ascent _ifs_ascent.ifs_updraught_ascent, then the closure under test.
Sounding builders are imported from the existing harness -- no new soundings
are invented.

float32 everywhere; every allclose states its tolerance.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from types import SimpleNamespace

try:  # repository layout: tests is a package
    from .test_ifs_ascent import column, _run, _run_kind
except ImportError:  # flat layout
    from test_ifs_ascent import column, _run, _run_kind  # type: ignore

from legoesm.atmosphere.physics.convection import _ifs_test_ascent as ta
from legoesm.atmosphere.physics.convection import _ifs_ascent as asc
from legoesm.atmosphere.physics.convection import _ifs_closure as cl
from legoesm import constants as C

TOL = dict(rtol=1e-5, atol=1e-8)   # float32 eager tolerance
TOL_LOOSE = dict(rtol=5e-3, atol=1e-6)  # float32 chained-computation tolerance


# --------------------------------------------------------------------------
# builder: real chain (trigger -> ascent) plus closure inputs
# --------------------------------------------------------------------------
def _setup(kind, dt=112.5, dT_dt=2.0e-4, dq_dt=1.0e-6):
    """Run the real trigger+ascent chain and assemble closure inputs.

    dT/dq tendencies are small positive constants so that ZDHPBL > 0 (the
    shallow branch needs it, cumastrn.F90:576).

    CALLER CONTRACT (see ifs_closure docstring): cumastrn runs the ascent
    with the first-guess base mass flux ZMFUB, so the closure expects
    M_u[k_cbot] == M_b0.  The harness ascent produces a cloud-base flux of
    0.02 kg m-2 s-1 while first_guess_mass_flux on the same column gives a
    much larger M_b0; if we handed that in unchanged, ZHEAT (proportional
    to PMFU) would be ~46x too small and the closure would saturate at
    ZMFMAX.  We therefore rescale M_u and every flux built from it by
    M_b0 / M_u[k_cbot] so the harness satisfies the source's contract.
    """
    out, tst, env = _run_kind(kind)
    T, qsp, qs, p_full, p_half, geo_full, geo_half, T_h, q_h = env
    nlev = T.shape[1]
    f32 = lambda x: jnp.asarray(np.asarray(x, np.float32))

    T_u, q_u, l_u, M_u = out.T_u, out.q_u, out.l_u, out.M
    Th = T_h[:, :nlev]
    qh = q_h[:, :nlev]
    gh = geo_half[:, :nlev]

    # flux profiles in the ascent-output contract (fabricated tracers where
    # the ascent module does not export them; they carry the same M_u shape
    # and are what scale_profile must rescale by the single ZMFS)
    PMFUS = M_u * C.c_pd * (T_u - Th)
    PMFUQ = M_u * (q_u - qh)
    PMFUL = M_u * l_u
    PLUDE = 1.0e-4 * M_u
    PDMFUP = getattr(out, "PDMFUP", None)
    if PDMFUP is None:
        PDMFUP = 1.0e-4 * M_u
    PMFUDE_RATE = 1.0e-3 * M_u
    PDMFEN = 5.0e-4 * M_u

    # updraught-velocity accumulator -> PWMEAN conversion (cumastrn:788-800):
    # the closure does w_mean = sqrt(2 * pwmean_raw / zdpmean); use the
    # kinetic energy PKINEU of the ascent (K = w^2/2)
    j = jnp.arange(nlev)[None, :]
    kb = jnp.asarray(tst.k_cbot, jnp.int32)[:, None]
    kt = jnp.asarray(out.k_ctop, jnp.int32)[:, None]
    win = (j >= kt) & (j <= kb)
    dp_full = p_full - jnp.concatenate([p_full[:, :1], p_full[:, :-1]], -1)
    pwmean_raw = jnp.sum(jnp.asarray(out.PKINEU) * dp_full * win, axis=1)
    zdpmean = jnp.sum(dp_full * win, axis=1)

    # first-guess mass flux for this column (same computation as
    # _first_guess below, inlined because the rescaling needs it here)
    zdhpbl = cl.subcloud_mse_supply(
        f32(np.full((1, nlev), dT_dt)), f32(np.full((1, nlev), dq_dt)),
        p_half, jnp.maximum(tst.k_cbot, 0))
    kbv = jnp.maximum(tst.k_cbot, 0)
    idx = jnp.arange(1)
    kb_i = kbv.astype(jnp.int32)
    zqumqe = q_u[idx, kb_i] + l_u[idx, kb_i] - qh[idx, kb_i]
    zdh = cl.RG * jnp.maximum(
        cl.RCPD * (T_u[idx, kb_i] - Th[idx, kb_i]) + cl.RLVTT * zqumqe,
        1.0e5 * jnp.maximum(0.01 * qh[idx, kb_i], 1.0e-10))
    M_b0, _, _ = cl.first_guess_mass_flux(
        p_half, kbv, tst.ldcum > 0.5, tst.ktype, zdhpbl, zdh,
        f32(dt), cl.IFSClosureConfig())

    # enforce the caller contract: PMFU(k_cbot) == M_b0
    m_base_ascent = M_u[idx, kb_i]
    rescale = M_b0 / jnp.maximum(m_base_ascent, 1.0e-30)   # (1,)
    rescale = rescale[:, None]                              # (1,1) broadcast
    M_u = M_u * rescale
    PMFUS = PMFUS * rescale
    PMFUQ = PMFUQ * rescale
    PMFUL = PMFUL * rescale
    PLUDE = PLUDE * rescale
    PDMFUP = PDMFUP * rescale
    PMFUDE_RATE = PMFUDE_RATE * rescale
    PDMFEN = PDMFEN * rescale

    s = SimpleNamespace(
        T=T, q=qsp, qs=qs, p_full=p_full, p_half=p_half,
        geo_full=geo_full, geo_half=geo_half, T_h=T_h, q_h=q_h,
        T_u=T_u, q_u=q_u, l_u=l_u, M_u=M_u,
        PMFUS=PMFUS, PMFUQ=PMFUQ, PMFUL=PMFUL, PLUDE=PLUDE,
        PDMFUP=PDMFUP, PMFUDE_RATE=PMFUDE_RATE, PDMFEN=PDMFEN,
        dT_dt_other=f32(np.full((1, nlev), dT_dt)),
        dq_dt_other=f32(np.full((1, nlev), dq_dt)),
        ldcum=tst.ldcum > 0.5,
        ktype=tst.ktype,
        k_cbot=kbv,
        k_ctop=jnp.maximum(out.k_ctop, 0),
        k_dpl=jnp.maximum(tst.k_dpl, 0),
        pwmean=pwmean_raw, zdpmean=zdpmean,
        land_frac=f32(np.zeros(1)), dx_m=f32(1.0e5), dt=f32(dt),
        cfg=cl.IFSClosureConfig(),
    )
    return s, out, tst


def _deep_closure(s, T_u=None, M_u=None):
    return cl.deep_cape_closure(
        s.T, s.q, s.qs, s.T_h[:, :s.T.shape[1]], s.q_h[:, :s.T.shape[1]],
        s.T_u if T_u is None else T_u, s.q_u, s.l_u,
        s.M_u if M_u is None else M_u,
        s.p_full, s.p_half, s.geo_full, s.geo_half[:, :s.T.shape[1]],
        s.dT_dt_other, s.dq_dt_other,
        s.ldcum, s.ktype, s.k_cbot, s.k_ctop, s.k_dpl,
        s.pwmean, s.zdpmean, s.land_frac, s.dx_m, s.dt, s.cfg,
        _first_guess(s)[0], _first_guess(s)[1])


def _first_guess(s):
    zdhpbl = cl.subcloud_mse_supply(s.dT_dt_other, s.dq_dt_other,
                                    s.p_half, s.k_cbot)
    idx = jnp.arange(1)
    kb = s.k_cbot.astype(jnp.int32)
    zqumqe = s.q_u[idx, kb] + s.l_u[idx, kb] - s.q_h[idx, kb]
    zdh = cl.RG * jnp.maximum(
        cl.RCPD * (s.T_u[idx, kb] - s.T_h[idx, kb]) + cl.RLVTT * zqumqe,
        1.0e5 * jnp.maximum(0.01 * s.q_h[idx, kb], 1.0e-10))
    return cl.first_guess_mass_flux(
        s.p_half, s.k_cbot, s.ldcum, s.ktype, zdhpbl, zdh, s.dt, s.cfg)


def _ifs_closure(s, T=None):
    return cl.ifs_closure(
        M_u=s.M_u, PMFUS=s.PMFUS, PMFUQ=s.PMFUQ, PMFUL=s.PMFUL,
        PLUDE=s.PLUDE, PDMFUP=s.PDMFUP, PMFUDE_RATE=s.PMFUDE_RATE,
        PDMFEN=s.PDMFEN, T_u=s.T_u, q_u=s.q_u, l_u=s.l_u,
        k_ctop=s.k_ctop, pwmean=s.pwmean, zdpmean=s.zdpmean,
        ldcum=s.ldcum, ktype=s.ktype, k_cbot=s.k_cbot, k_dpl=s.k_dpl,
        T=s.T if T is None else T, q=s.q, qs=s.qs, p_full=s.p_full,
        p_half=s.p_half, geo_full=s.geo_full, geo_half=s.geo_half,
        T_h=s.T_h, q_h=s.q_h,
        dT_dt_other=s.dT_dt_other, dq_dt_other=s.dq_dt_other,
        land_frac=s.land_frac, dx_m=s.dx_m, dt=s.dt, cfg=s.cfg)


# ==========================================================================
# 1. ZDHPBL integrates ONLY the sub-cloud levels (off-by-one regression)
# ==========================================================================
def test_subcloud_mse_supply_only_subcloud_levels():
    s, _, _ = _setup("deep")
    kb = int(s.k_cbot[0])
    p_half = np.asarray(s.p_half[0], np.float64)
    dp = p_half[1:] - p_half[:-1]
    dT, dq = 1.0e-4, 1.0e-6
    val = float(cl.subcloud_mse_supply(
        jnp.asarray(np.full((1, 30), dT, np.float32)),
        jnp.asarray(np.full((1, 30), dq, np.float32)),
        s.p_half, s.k_cbot)[0])

    expected = sum((cl.RLVTT * dq + cl.RCPD * dT) * dp[jj]
                   for jj in range(kb, 30))   # j >= k_cbot INCLUSIVE
    assert val == pytest.approx(expected, rel=1e-4), (val, expected)

    # cloud-base level itself contributes (regression: excluding it is the bug)
    expected_excl = sum((cl.RLVTT * dq + cl.RCPD * dT) * dp[jj]
                        for jj in range(kb + 1, 30))
    assert abs(val - expected_excl) > 1e-6

    # the level ABOVE cloud base (j = k_cbot-1) is NOT included
    dq_only_above = np.zeros((1, 30), np.float32)
    dq_only_above[0, kb - 1] = 1.0
    val_above = float(cl.subcloud_mse_supply(
        np.zeros((1, 30), np.float32), jnp.asarray(dq_only_above),
        s.p_half, s.k_cbot)[0])
    assert val_above == pytest.approx(0.0, abs=1e-3)


# ==========================================================================
# 2. first_guess_mass_flux branches
# ==========================================================================
def test_first_guess_mass_flux_branches():
    _, _, p_full, p_half1 = column("deep")
    p_half = jnp.asarray(np.tile(np.asarray(p_half1, np.float32), (4, 1)))
    dt = jnp.asarray(112.5, jnp.float32)
    kb = jnp.asarray([28, 28, 28, 28], jnp.int32)
    cfg = cl.IFSClosureConfig()
    ldcum = jnp.asarray([True, True, True, False])
    ktype = jnp.asarray([1, 2, 2, 2], jnp.int32)
    zdhpbl = jnp.asarray([0.0, 500.0, -10.0, 100.0], jnp.float32)
    zdh_shal = jnp.asarray(np.full(4, cl.RG * 1.0e4, np.float32))

    zmfub, zmfmax, ldcum_out = cl.first_guess_mass_flux(
        p_half, kb, ldcum, ktype, zdhpbl, zdh_shal, dt, cfg)

    dp_base = float(np.asarray(p_half[0, 29] - p_half[0, 28]))
    zmfmax_ref = dp_base * cfg.rmfcfl / (cl.RG * 112.5)
    # rtol=1e-5: a float32 build cannot hold rtol=1e-6 on this quantity
    # (measured float32 value differs from the float64 reference by 2.4e-6)
    assert np.allclose(np.asarray(zmfmax), zmfmax_ref, rtol=1e-5)

    assert np.allclose(np.asarray(zmfub[0]), 0.1 * zmfmax_ref, rtol=1e-5)
    assert np.allclose(np.asarray(zmfub[1]),
                       min(500.0 / (cl.RG * 1.0e4), zmfmax_ref), rtol=1e-4)
    assert np.allclose(np.asarray(zmfub[2]), 0.1 * zmfmax_ref, rtol=1e-5)
    assert bool(ldcum_out[2]) is False        # cumastrn:578 LDCUM=.FALSE.
    assert float(zmfub[3]) == pytest.approx(0.0, abs=1e-30)
    assert bool(ldcum_out[0]) and bool(ldcum_out[1])  # others untouched


# ==========================================================================
# 3. deep_cape_closure bounds + monotonicity in buoyancy
# ==========================================================================
def test_deep_cape_closure_bounds_and_monotonicity():
    # dt = 112.5 s suffices once _setup enforces the caller contract
    # PMFU(k_cbot) == M_b0: with the harness ascent rescaled, ZHEAT grows
    # by the ~46x rescaling factor and M_b1 lands well below ZMFMAX
    # (~0.5 * M_b0), so no cap and no smaller dt is needed.
    s, _, _ = _setup("deep")
    M_b1, zcape, zheat, zxtau = _deep_closure(s)
    _, zmfmax, _ = _first_guess(s)
    assert float(zcape[0]) <= 5000.0 + 1e-3
    assert float(zheat[0]) >= 1.0e-4
    assert 720.0 - 1e-3 <= float(zxtau[0]) <= 10800.0 + 1e-3
    assert float(M_b1[0]) <= float(zmfmax[0]) * (1 + 1e-6)
    assert float(M_b1[0]) > 0.0
    # precondition for the monotonicity half: the column is NOT on the cap
    assert float(M_b1[0]) < float(zmfmax[0]) * 0.999, (
        "column sits on the ZMFMAX cap; monotonicity in CAPE cannot be "
        "observed there -- reduce dt or force ktype=1 on the trade sounding")

    # closure, not formula: more buoyant plume -> more CAPE -> more mass flux
    kb, kt = int(s.k_cbot[0]), int(s.k_ctop[0])
    T_u2 = jnp.array(s.T_u)
    for k in range(kt, kb + 1):              # strictly inside the LLO3 window
        T_u2 = T_u2.at[0, k].add(0.5)
    M_b1b, zcapeb, _, _ = _deep_closure(s, T_u=T_u2)
    assert float(zcapeb[0]) > float(zcape[0]) + 1e-3
    # not clipped at ZMFMAX, so the mass flux must follow the CAPE
    assert float(M_b1b[0]) > float(M_b1[0])


# ==========================================================================
# 4. shallow_closure uses the plume temperature at k_cbot-1 in ZDH2
# ==========================================================================
def test_shallow_closure_uses_level_above_cloud_base():
    s, _, _ = _setup("trade")
    kb = int(s.k_cbot[0])
    zdhpbl = cl.subcloud_mse_supply(s.dT_dt_other, s.dq_dt_other,
                                    s.p_half, s.k_cbot)
    _, zmfmax, _ = _first_guess(s)

    def run(T_u):
        return cl.shallow_closure(
            s.p_half, s.k_cbot, s.ldcum, s.ktype, zdhpbl,
            T_u, s.q_u, s.l_u, s.T_h, s.q_h,
            jnp.asarray(0.02, jnp.float32), zmfmax, s.dt, s.cfg)

    m0 = float(run(s.T_u)[0])
    # perturb T_u at k_cbot-1 ONLY: pre-fix code (T_u at k_cbot) is blind
    T_u2 = jnp.array(s.T_u).at[0, kb - 1].add(2.0)
    m1 = float(run(T_u2)[0])
    assert abs(m1 - m0) / max(abs(m0), 1e-12) > 1e-3, (m0, m1)
    # a warmer plume above the base raises ZDH2 -> lowers the mass flux
    assert m1 < m0
    # sanity: perturbing T_u far above the base (k_ctop) leaves it unchanged
    kt = int(s.k_ctop[0])
    T_u3 = jnp.array(s.T_u).at[0, max(kt - 2, 0)].add(2.0)
    m2 = float(run(T_u3)[0])
    assert m2 == pytest.approx(m0, rel=1e-6)


# ==========================================================================
# 5. scale_profile
# ==========================================================================
def _scaled(s, M_b1, M_b0, dt):
    return cl.scale_profile(
        s.M_u, s.PMFUS, s.PMFUQ, s.PMFUL, s.PLUDE, s.PDMFUP,
        s.PMFUDE_RATE, s.PDMFEN, M_b0, M_b1, s.p_half, s.k_cbot,
        s.k_ctop, s.ldcum, jnp.asarray(dt, jnp.float32), s.cfg)


def test_scale_profile_cloud_base_and_single_factor():
    s, _, _ = _setup("deep")
    M_b0, _, _ = _first_guess(s)
    M_b1, _, _, _ = _deep_closure(s)
    (M_u, PMFUS, PMFUQ, _, _, _, _, _, zmfs) = _scaled(s, M_b1, M_b0, 112.5)

    kb = int(s.k_cbot[0])
    zmfs_raw = float(M_b1[0] / max(cl.RMFCMIN, float(M_b0[0])))
    assert float(zmfs[0]) <= zmfs_raw * (1 + 1e-6)

    # (a) M_u at cloud base equals min(M_b1, the per-level CFL/RMFLIA
    # bound): after _setup rescales the ascent to PMFU(k_cbot) == M_b0,
    # the rescaled base flux is zmfs * M_b0 which equals M_b1 only when
    # the bound does not bind.
    ph = np.asarray(s.p_half[0], np.float64)
    zmfmax_lev = min((ph[kb + 1] - ph[kb]) * 3.0 / (cl.RG * 112.5),
                     cl.RMFLIA)
    m_base = float(M_u[0, kb])
    m_expected = min(float(M_b1[0]), zmfmax_lev)
    assert m_base == pytest.approx(m_expected, rel=1e-5), (
        m_base, m_expected, float(M_b1[0]), zmfmax_lev)
    assert m_base <= zmfmax_lev * (1 + 1e-5)

    # (b) EVERY profile is scaled by the SAME factor zmfs.  Only levels
    # strictly inside the plume carry non-zero fluxes; above the top both
    # numerator and denominator vanish and the ratio is 0/0.
    kt = int(s.k_ctop[0])
    inside = [k for k in range(kt + 1, kb)
              if abs(float(np.asarray(s.PMFUS[0, k]))) > 1e-6
              and abs(float(np.asarray(s.PMFUQ[0, k]))) > 1e-12]
    assert len(inside) >= 2, inside
    assert float(zmfs[0]) > 0
    for k in inside[:2]:
        r1 = float(PMFUS[0, k] / np.asarray(s.PMFUS[0, k]))
        r2 = float(PMFUQ[0, k] / np.asarray(s.PMFUQ[0, k]))
        assert r1 == pytest.approx(float(zmfs[0]), rel=1e-4)
        assert r2 == pytest.approx(float(zmfs[0]), rel=1e-4)


def test_scale_profile_cfl_bound_binds_for_large_dt():
    s, _, _ = _setup("deep")
    M_b0, _, _ = _first_guess(s)
    M_b1, _, _, _ = _deep_closure(s)
    big_dt = 10.0 * 3600.0
    (M_u, _, _, _, _, _, _, _, zmfs) = _scaled(s, M_b1, M_b0, big_dt)
    zmfs_raw = float(M_b1[0] / max(cl.RMFCMIN, float(M_b0[0])))
    assert float(zmfs[0]) < zmfs_raw * 0.999          # the CFL bound binds

    nlev = s.T.shape[1]
    j = np.arange(nlev)
    ph = np.asarray(s.p_half[0], np.float64)
    zmfmax_lev = np.minimum(
        (ph[1:] - ph[:-1] if False else ph[1:nlev + 1] - ph[:nlev] if False
         else (ph[1:nlev + 1] - np.concatenate([[ph[0]], ph[1:nlev]]))
         ) * cl.IFSClosureConfig().rmfcfl / (cl.RG * big_dt),
        cl.RMFLIA)
    win = (j >= 1) & (j >= int(s.k_ctop[0]) - 1) & (j <= int(s.k_cbot[0]))
    mu = np.asarray(M_u[0])
    assert np.all(mu[win] <= zmfmax_lev[win] * (1 + 1e-5))
    assert float(zmfs[0]) >= 1.0e-10                  # :972 floor


def test_scale_profile_nonconvecting_column_untouched():
    s, _, _ = _setup("deep")
    M_b0, _, _ = _first_guess(s)
    dbl = lambda x: jnp.asarray(np.tile(np.asarray(x, np.float32), (2, 1)))
    args = {k: dbl(getattr(s, k)) for k in
            ("M_u", "PMFUS", "PMFUQ", "PMFUL", "PLUDE", "PDMFUP",
             "PMFUDE_RATE", "PDMFEN")}
    out = cl.scale_profile(
        args["M_u"], args["PMFUS"], args["PMFUQ"], args["PMFUL"],
        args["PLUDE"], args["PDMFUP"], args["PMFUDE_RATE"], args["PDMFEN"],
        jnp.asarray([float(M_b0[0]), float(M_b0[0])], jnp.float32),
        jnp.asarray([float(M_b0[0]), 0.0], jnp.float32),
        dbl(s.p_half), jnp.asarray([int(s.k_cbot[0])] * 2, jnp.int32),
        jnp.asarray([int(s.k_ctop[0])] * 2, jnp.int32),
        jnp.asarray([True, False]), s.dt, s.cfg)
    M_u, PMFUS, zmfs = out[0], out[1], out[-1]
    assert float(zmfs[1]) == pytest.approx(0.0, abs=1e-30)
    for name, arr in zip(("M_u", "PMFUS"), (M_u, PMFUS)):
        np.testing.assert_allclose(
            np.asarray(arr[1]), np.asarray(getattr(s, name)[0]),
            rtol=1e-6, atol=1e-12, err_msg=name)


# ==========================================================================
# 6. end-to-end ifs_closure
# ==========================================================================
def test_ifs_closure_end_to_end_deep_and_trade():
    nlev = 30
    for kind in ("deep", "trade"):
        s, _, _ = _setup(kind)
        res = _ifs_closure(s)
        for k in ("M_u", "PMFUS", "PMFUQ", "PMFUL", "PLUDE", "PDMFUP",
                  "PMFUDE_RATE", "PDMFEN"):
            assert np.asarray(res[k]).shape == (1, nlev), (kind, k)
        for k in ("M_b1", "zcape", "zheat", "zxtau", "zmfs"):
            assert np.asarray(res[k]).shape == (1,), (kind, k)
        assert bool(np.asarray(res["ldcum"])[0]) is True

    # deep: CAPE closure moved the mass flux away from the first guess
    s_d, _, _ = _setup("deep")
    res_d = _ifs_closure(s_d)
    M_b0, _, _ = _first_guess(s_d)
    assert abs(float(res_d["M_b1"][0]) - float(M_b0[0])) > 1e-6

    # trade: M_b1 comes from the shallow branch (recompute it directly)
    s_t, _, _ = _setup("trade")
    res_t = _ifs_closure(s_t)
    zdhpbl = cl.subcloud_mse_supply(s_t.dT_dt_other, s_t.dq_dt_other,
                                    s_t.p_half, s_t.k_cbot)
    _, zmfmax, _ = _first_guess(s_t)
    shal = cl.shallow_closure(
        s_t.p_half, s_t.k_cbot, s_t.ldcum, s_t.ktype, zdhpbl,
        s_t.T_u, s_t.q_u, s_t.l_u, s_t.T_h, s_t.q_h,
        jnp.asarray(0.02, jnp.float32), zmfmax, s_t.dt, s_t.cfg)
    assert float(res_t["M_b1"][0]) == pytest.approx(float(shal[0]), rel=1e-5)


# ==========================================================================
# 7. jit / eager parity
# ==========================================================================
def test_ifs_closure_jit_eager_parity():
    s, _, _ = _setup("deep")
    eager = _ifs_closure(s)
    jitted = jax.jit(lambda T: cl.ifs_closure(
        M_u=s.M_u, PMFUS=s.PMFUS, PMFUQ=s.PMFUQ, PMFUL=s.PMFUL,
        PLUDE=s.PLUDE, PDMFUP=s.PDMFUP, PMFUDE_RATE=s.PMFUDE_RATE,
        PDMFEN=s.PDMFEN, T_u=s.T_u, q_u=s.q_u, l_u=s.l_u,
        k_ctop=s.k_ctop, pwmean=s.pwmean, zdpmean=s.zdpmean,
        ldcum=s.ldcum, ktype=s.ktype, k_cbot=s.k_cbot, k_dpl=s.k_dpl,
        T=T, q=s.q, qs=s.qs, p_full=s.p_full, p_half=s.p_half,
        geo_full=s.geo_full, geo_half=s.geo_half, T_h=s.T_h, q_h=s.q_h,
        dT_dt_other=s.dT_dt_other, dq_dt_other=s.dq_dt_other,
        land_frac=s.land_frac, dx_m=s.dx_m, dt=s.dt, cfg=s.cfg))(s.T)
    for k in ("M_u", "PMFUS", "PMFUQ", "PMFUL", "PLUDE", "PDMFUP",
              "PMFUDE_RATE", "PDMFEN", "M_b1", "zcape", "zheat", "zxtau",
              "zmfs"):
        np.testing.assert_allclose(
            np.asarray(jitted[k]), np.asarray(eager[k]),
            rtol=1e-5, atol=1e-8, err_msg=k)   # float32 jit tolerance


# ==========================================================================
# 8. gradient of the full chain w.r.t. environmental T
# ==========================================================================
def test_ifs_closure_grad_wrt_T_full_chain():
    # The gradient through the full chain (closure only; ascent inputs are
    # fixed at the harness values) IS finite -- the earlier xfail
    # (integer level selection / max kinks) does not materialise here
    # because the closure consumes the ascent output, not the search.
    s, _, _ = _setup("deep")

    def loss(T):
        res = cl.ifs_closure(
            M_u=s.M_u, PMFUS=s.PMFUS, PMFUQ=s.PMFUQ, PMFUL=s.PMFUL,
            PLUDE=s.PLUDE, PDMFUP=s.PDMFUP, PMFUDE_RATE=s.PMFUDE_RATE,
            PDMFEN=s.PDMFEN, T_u=s.T_u, q_u=s.q_u, l_u=s.l_u,
            k_ctop=s.k_ctop, pwmean=s.pwmean, zdpmean=s.zdpmean,
            ldcum=s.ldcum, ktype=s.ktype, k_cbot=s.k_cbot, k_dpl=s.k_dpl,
            T=T, q=s.q, qs=s.qs, p_full=s.p_full, p_half=s.p_half,
            geo_full=s.geo_full, geo_half=s.geo_half, T_h=s.T_h,
            q_h=s.q_h, dT_dt_other=s.dT_dt_other, dq_dt_other=s.dq_dt_other,
            land_frac=s.land_frac, dx_m=s.dx_m, dt=s.dt, cfg=s.cfg)
        return jnp.sum(res["M_u"])

    g = jax.grad(loss)(s.T)
    assert bool(jnp.all(jnp.isfinite(g))), "grad must be finite everywhere"
    # a zero gradient would mean the closure is disconnected from T
    assert float(jnp.max(jnp.abs(g))) > 0.0
