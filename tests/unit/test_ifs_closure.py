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
def _setup(kind, dt=112.5,
           dT_dt_other=2.0e-4, dq_dt_other=1.0e-6,
           dT_dt_adv=-1.5e-4, dq_dt_adv=2.0e-7):
    """Run the real trigger+ascent chain and assemble closure inputs.

    TWO tendency pairs with DIFFERENT values (deliberately different in
    sign/magnitude so a future collapse of the two pairs into one fails
    test_tendency_pairs_not_interchangeable):
      * dT_dt_other / dq_dt_other  -- TOTAL physics tendencies (PTENT/PTENQ),
        used by ZDHPBL and ZCAPPBL; small positive so ZDHPBL > 0 (the
        shallow branch needs it, cumastrn.F90:576).
      * dT_dt_adv / dq_dt_adv      -- ADVective tendencies (PTENTA/PTENQA),
        used by ZCAPE2 and ZDQCV.

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
    # the closure does w_mean = sqrt(2 * max(1e-2, pwmean/max(1,zdpmean)));
    # use the kinetic energy PKINEU of the ascent (K = w^2/2)
    j = jnp.arange(nlev)[None, :]
    kb = jnp.asarray(tst.k_cbot, jnp.int32)[:, None]
    kt = jnp.asarray(out.k_ctop, jnp.int32)[:, None]
    win = (j >= kt) & (j <= kb)
    dp_full = p_full - jnp.concatenate([p_full[:, :1], p_full[:, :-1]], -1)
    pwmean_raw = jnp.sum(jnp.asarray(out.PKINEU) * dp_full * win, axis=1)
    zdpmean = jnp.sum(dp_full * win, axis=1)

    # first-guess mass flux for this column (same computation as
    # _first_guess below, inlined because the rescaling needs it here).
    # NOTE: ZDHPBL uses the TOTAL pair (PTENT/PTENQ), never the advective.
    k_start = _k_start(p_full)
    zdhpbl = cl.subcloud_mse_supply(
        f32(np.full((1, nlev), dT_dt_other)),
        f32(np.full((1, nlev), dq_dt_other)),
        p_half, jnp.maximum(tst.k_cbot, 0), k_start)
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
        dT_dt_other=f32(np.full((1, nlev), dT_dt_other)),
        dq_dt_other=f32(np.full((1, nlev), dq_dt_other)),
        dT_dt_adv=f32(np.full((1, nlev), dT_dt_adv)),
        dq_dt_adv=f32(np.full((1, nlev), dq_dt_adv)),
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
        s.dT_dt_adv, s.dq_dt_adv,
        s.ldcum, s.ktype, s.k_cbot, s.k_ctop, s.k_dpl,
        s.pwmean, s.zdpmean, s.land_frac, s.dx_m, s.dt, s.cfg,
        _first_guess(s)[0], _first_guess(s)[1])


def _first_guess(s):
    zdhpbl = cl.subcloud_mse_supply(s.dT_dt_other, s.dq_dt_other,
                                    s.p_half, s.k_cbot,
                                    _k_start(s.p_full))
    idx = jnp.arange(1)
    kb = s.k_cbot.astype(jnp.int32)
    zqumqe = s.q_u[idx, kb] + s.l_u[idx, kb] - s.q_h[idx, kb]
    zdh = cl.RG * jnp.maximum(
        cl.RCPD * (s.T_u[idx, kb] - s.T_h[idx, kb]) + cl.RLVTT * zqumqe,
        1.0e5 * jnp.maximum(0.01 * s.q_h[idx, kb], 1.0e-10))
    return cl.first_guess_mass_flux(
        s.p_half, s.k_cbot, s.ldcum, s.ktype, zdhpbl, zdh, s.dt, s.cfg)


def _ifs_closure(s, T=None, ktype=None, ktype_first_guess=None):
    return cl.ifs_closure(
        ktype_first_guess=ktype_first_guess,
        M_u=s.M_u, PMFUS=s.PMFUS, PMFUQ=s.PMFUQ, PMFUL=s.PMFUL,
        PLUDE=s.PLUDE, PDMFUP=s.PDMFUP, PMFUDE_RATE=s.PMFUDE_RATE,
        PDMFEN=s.PDMFEN, T_u=s.T_u, q_u=s.q_u, l_u=s.l_u,
        k_ctop=s.k_ctop, pwmean=s.pwmean, zdpmean=s.zdpmean,
        ldcum=s.ldcum, ktype=s.ktype if ktype is None else ktype,
        k_cbot=s.k_cbot, k_dpl=s.k_dpl,
        T=s.T if T is None else T, q=s.q, qs=s.qs, p_full=s.p_full,
        p_half=s.p_half, geo_full=s.geo_full, geo_half=s.geo_half,
        T_h=s.T_h, q_h=s.q_h,
        dT_dt_other=s.dT_dt_other, dq_dt_other=s.dq_dt_other,
        dT_dt_adv=s.dT_dt_adv, dq_dt_adv=s.dq_dt_adv,
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
    # k_start = 0 -> the full sub-cloud window; the NJKT2 floor is tested
    # separately in test_njkt2_cutoff_window.
    val = float(cl.subcloud_mse_supply(
        jnp.asarray(np.full((1, 30), dT, np.float32)),
        jnp.asarray(np.full((1, 30), dq, np.float32)),
        s.p_half, s.k_cbot, jnp.asarray([0], jnp.int32))[0])

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
        s.p_half, s.k_cbot, jnp.asarray([0], jnp.int32))[0])
    assert val_above == pytest.approx(0.0, abs=1e-3)


# ==========================================================================
# 2. first_guess_mass_flux branches
# ==========================================================================
def test_first_guess_mass_flux_branches():
    # STRETCHED sigma grid (sigma**1.3): the cloud-base layer thickness
    # p_half[kb]-p_half[kb-1] deliberately differs from the layer ABOVE,
    # p_half[kb+1]-p_half[kb], so a regression back to the wrong interface
    # pair cannot hide on a uniform grid.
    sig = np.linspace(0.02, 1.0, 31) ** 1.3
    ph = 1013.2e2 * sig
    p_half = jnp.asarray(np.tile(ph.astype(np.float32), (4, 1)))
    kb = 28
    dp_base = ph[kb] - ph[kb - 1]          # PAPH(IKB)-PAPH(IKB-1)  <-- correct
    dp_above = ph[kb + 1] - ph[kb]         # the WRONG pair used pre-fix
    assert dp_base != dp_above and abs(dp_base - dp_above) / dp_base > 0.01

    dt = jnp.asarray(112.5, jnp.float32)
    kb_v = jnp.asarray([kb] * 4, jnp.int32)
    cfg = cl.IFSClosureConfig()
    ldcum = jnp.asarray([True, True, True, False])
    ktype = jnp.asarray([1, 2, 2, 2], jnp.int32)
    zdhpbl = jnp.asarray([0.0, 500.0, -10.0, 100.0], jnp.float32)
    zdh_shal = jnp.asarray(np.full(4, cl.RG * 1.0e4, np.float32))

    zmfub, zmfmax, ldcum_out = cl.first_guess_mass_flux(
        p_half, kb_v, ldcum, ktype, zdhpbl, zdh_shal, dt, cfg)

    zmfmax_ref = dp_base * cfg.rmfcfl / (cl.RG * 112.5)
    # rtol=1e-4: holds in float32 and float64 (single division, no chaining)
    assert np.allclose(np.asarray(zmfmax), zmfmax_ref, rtol=1e-4)
    # the wrong interface pair would give a measurably different bound
    zmfmax_wrong = dp_above * cfg.rmfcfl / (cl.RG * 112.5)
    assert abs(zmfmax_wrong - zmfmax_ref) / zmfmax_ref > 0.01

    assert np.allclose(np.asarray(zmfub[0]), 0.1 * zmfmax_ref, rtol=1e-4)
    assert np.allclose(np.asarray(zmfub[1]),
                       min(500.0 / (cl.RG * 1.0e4), zmfmax_ref), rtol=1e-4)
    assert np.allclose(np.asarray(zmfub[2]), 0.1 * zmfmax_ref, rtol=1e-4)
    assert bool(ldcum_out[2]) is False        # cumastrn:578 LDCUM=.FALSE.
    assert float(zmfub[3]) == pytest.approx(0.0, abs=1e-30)
    assert bool(ldcum_out[0]) and bool(ldcum_out[1])  # others untouched


# ==========================================================================
# 3. deep_cape_closure bounds + monotonicity in buoyancy
# ==========================================================================
def test_deep_cape_closure_bounds_and_monotonicity():
    # Integrand regression (not a clip bound): with the ADVective pair set
    # to zero, ZCAPE2 == ZCAPE, ZDQCV == 0, and (land_frac = 0) ZCAPDCYCL
    # == 0, so the fast branch collapses to ZCAPE = min(5000, ZCAPE_raw)
    # with ZCAPE_raw the plain pressure integral of the buoyancy -- which
    # we recompute here in float64 numpy.  Likewise ZHEAT against the
    # stability integrand with ZORCPD = 1/c_pd.
    s, _, _ = _setup("deep", dT_dt_adv=0.0, dq_dt_adv=0.0)
    M_b1, zcape, zheat, zxtau = _deep_closure(s)
    M_b0, zmfmax, _ = _first_guess(s)

    nlev = s.T.shape[1]
    kb, kt = int(s.k_cbot[0]), int(s.k_ctop[0])
    RETV = float(cl.RETV)
    T = np.asarray(s.T[0], np.float64)
    q = np.asarray(s.q[0], np.float64)
    qs = np.asarray(s.qs[0], np.float64)
    Tu = np.asarray(s.T_u[0], np.float64)
    qu = np.asarray(s.q_u[0], np.float64)
    lu = np.asarray(s.l_u[0], np.float64)
    Th = np.asarray(s.T_h[0, :nlev], np.float64)
    qh = np.asarray(s.q_h[0, :nlev], np.float64)
    Mu = np.asarray(s.M_u[0], np.float64)
    pf = np.asarray(s.p_full[0], np.float64)
    phv = np.asarray(s.p_half[0], np.float64)
    gf = np.asarray(s.geo_full[0, :nlev], np.float64)
    dpf = pf - np.concatenate([[pf[0]], pf[:-1]])
    dph = phv[1:] - phv[:-1]
    jj = np.arange(nlev)
    win = (jj >= 1) & (jj <= kb) & (jj > kt)          # LLO3 on our indices

    # branch selection computed independently: ZSATFR <= 0.94 (fast branch)
    zsatfr = float((q / qs * dph)[kt:].sum() / (phv[-1] - phv[kt]))
    assert zsatfr <= 0.94, "expected the ZSATFR<=0.94 (fast) branch"

    buoy = (Tu - Th) / Th + RETV * (qu - qh) - lu
    zcape_raw = float((buoy * dpf)[win].sum())
    assert 0.0 < zcape_raw < 5000.0, zcape_raw     # not vacuous: no clip
    assert np.allclose(np.asarray(zcape[0]), min(5000.0, zcape_raw),
                       rtol=1e-4, atol=1e-6)  # rtol=1e-4: f32/f64 safe

    dzg = gf[:-1] - gf[1:]
    stable = (((T[:-1] - T[1:] + dzg * float(cl.ZORCPD)) / Th[1:]
               + RETV * (q[:-1] - q[1:])) * float(cl.RG) * Mu[1:])
    zheat_ref = max(1.0e-4,
                    float(np.maximum(0.0, stable)[win[1:]].sum()))
    assert np.allclose(np.asarray(zheat[0]), zheat_ref,
                       rtol=1e-4, atol=1e-8)   # rtol=1e-4: f32 sum of ~30 terms

    assert 720.0 - 1e-3 <= float(zxtau[0]) <= 10800.0 + 1e-3
    assert float(M_b1[0]) <= float(zmfmax[0]) * (1 + 1e-6)
    assert float(M_b1[0]) > 0.0
    # precondition for the monotonicity half: the column is NOT on the cap
    assert float(M_b1[0]) < float(zmfmax[0]) * 0.999, (
        "column sits on the ZMFMAX cap; monotonicity in CAPE cannot be "
        "observed there -- reduce dt or force ktype=1 on the trade sounding")

    # closure, not formula: more buoyant plume -> more CAPE -> more mass flux
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
    # subcloud_mse_supply now takes the NJKT2 cutoff index (5th argument);
    # compute it with the shared min-form helper.
    zdhpbl = cl.subcloud_mse_supply(
        s.dT_dt_other, s.dq_dt_other, s.p_half, s.k_cbot,
        _k_start(s.p_full))
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
    # Built on the STRETCHED synthetic column (_synth_deep, sigma**1.3):
    # the _setup harness column is uniform in sigma at cloud base, so the
    # two candidate interface pairs differ by ~0.008 Pa in ~3376 Pa and
    # cannot be distinguished.  On the synthetic grid they genuinely
    # differ (guard below keeps this test honest).
    s = _synth_deep()
    M_b0 = jnp.asarray(np.asarray(s.zmfub, np.float32))
    M_b1, _, _, _ = _synth_deep_run(s)
    PMFUS0 = s.M_u * jnp.asarray(np.float32(100.0))
    PMFUQ0 = s.M_u * jnp.asarray(np.float32(0.01))
    dt = jnp.asarray(112.5, jnp.float32)
    (M_u, PMFUS, PMFUQ, _, _, _, _, _, zmfs) = cl.scale_profile(
        s.M_u, PMFUS0, PMFUQ0, s.M_u, s.M_u, s.M_u, s.M_u, s.M_u,
        M_b0, M_b1, s.p_half, s.k_cbot, s.k_ctop, s.ldcum, dt, s.cfg)

    kb = int(s.k_cbot[0])
    kt = int(s.k_ctop[0])
    zmfs_raw = float(M_b1[0]) / max(cl.RMFCMIN, float(M_b0[0]))
    assert float(zmfs[0]) <= zmfs_raw * (1 + 1e-6)

    # cloud-base layer thickness is PAPH(IKB)-PAPH(IKB-1) =
    # p_half[kb]-p_half[kb-1] (cumastrn:963), NOT p_half[kb+1]-p_half[kb].
    ph = np.asarray(s.p_half[0], np.float64)
    dp_base = ph[kb] - ph[kb - 1]
    dp_above = ph[kb + 1] - ph[kb]
    assert abs(dp_base - dp_above) / dp_base > 1e-3, (
        "synthetic grid is uniform at cloud base; the two interface pairs "
        "cannot be distinguished here")

    zmfmax_lev = min(dp_base * s.cfg.rmfcfl / (cl.RG * 112.5), cl.RMFLIA)
    m_base = float(M_u[0, kb])
    m_expected = min(float(M_b1[0]), zmfmax_lev)
    assert m_base == pytest.approx(m_expected, rel=1e-4), (   # rel=1e-4: f32/f64
        m_base, m_expected, float(M_b1[0]), zmfmax_lev)
    assert m_base <= zmfmax_lev * (1 + 1e-5)

    # (b) EVERY profile is scaled by the SAME factor zmfs.  Check levels
    # strictly inside the plume (k_ctop < k < k_cbot), lower half of the
    # window where the per-level CFL bound cannot bind; below the cloud
    # base scale_profile REPLACES the mass flux with the taper, and above
    # the top the fluxes vanish, so both regions are excluded.
    inside = [k for k in range(kt + 1, kb)
              if abs(float(np.asarray(PMFUS0[0, k]))) > 1e-6
              and abs(float(np.asarray(PMFUQ0[0, k]))) > 1e-12]
    assert len(inside) >= 2, inside
    assert float(zmfs[0]) > 0
    for k in inside[-2:]:
        r1 = float(PMFUS[0, k] / np.asarray(PMFUS0[0, k]))
        r2 = float(PMFUQ[0, k] / np.asarray(PMFUQ0[0, k]))
        assert r1 == pytest.approx(float(zmfs[0]), rel=1e-4)  # f32 division
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
    # per-level bound at OUR index j is PAPH(JK)-PAPH(JK-1) =
    # p_half[j]-p_half[j-1] (cumastrn:968), the SAME interface pair as the
    # cloud-base ZMFMAX.
    dp_lev = ph[:nlev] - np.concatenate([[ph[0]], ph[:nlev - 1]])
    zmfmax_lev = np.minimum(dp_lev * s.cfg.rmfcfl / (cl.RG * big_dt),
                            cl.RMFLIA)
    kt, kb = int(s.k_ctop[0]), int(s.k_cbot[0])
    mu = np.asarray(M_u[0], np.float64)
    mu0 = np.asarray(s.M_u[0], np.float64)
    zf = float(zmfs[0])

    # TWO disjoint index sets, exactly as in the source:
    #   scaling window: j >= k_ctop-1 AND j <= k_cbot  -- profile is
    #                   zmfs * input, clamped by the per-level bound
    #   below base:     j > k_cbot                      -- REPLACED by the
    #                   source taper (cumastrn:970-973), NOT multiplied
    #                   by zmfs, so the ratio to the input is not zmfs
    win = (j >= kt - 1) & (j <= kb)
    below = j > kb
    assert not np.any(win & below)

    # (a) inside the scaling window: zmfs * input, clamped by the
    #     per-level CFL bound.  rtol=1e-4, atol=1e-12: f32 pipeline vs a
    #     float64 reference.
    # The module does not clamp per level: it lowers the SINGLE zmfs until no
    # level in the window exceeds its bound (cumastrn:975-978), so inside the
    # window the profile is exactly zmfs * input.
    np.testing.assert_allclose(
        mu[win], zf * mu0[win], rtol=1e-4, atol=1e-12)
    assert np.all(mu[win] <= zmfmax_lev[win] * (1 + 1e-5))

    # (b) below-base levels follow the taper formula exactly
    #     (cumastrn:970-973): PMFU(IKB)*(PAPH(KLEV+1)-PAPH(JK)) /
    #     (PAPH(KLEV+1)-PAPH(IKB)) with PMFU(IKB) = min(M_b1, ZMFMAX).
    #     rtol=1e-4, atol=1e-10: f32 pipeline vs float64 reference.
    # cumastrn:970-973 builds the taper from the UNSCALED PMFU(IKB) of the
    # input profile (the taper is written before ZMFS is applied, and the
    # below-base levels are then excluded from the multiplication).
    m_base = mu0[kb]
    taper = m_base * (ph[-1] - ph[j[below]]) / (ph[-1] - ph[kb])
    np.testing.assert_allclose(mu[below], taper, rtol=1e-4, atol=1e-10)

    assert float(zmfs[0]) >= 1.0e-10                  # :975-979 floor


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
    M_b0_d, _, _ = _first_guess(s_d)
    assert abs(float(res_d["M_b1"][0]) - float(M_b0_d[0])) > 1e-6

    # trade: reference built by calling the module's own shallow_closure
    # with the same inputs (no third hand re-derivation of ZDH).  The
    # end-to-end result must equal that reference.
    s_t, _, _ = _setup("trade")
    res_t = _ifs_closure(s_t)
    zdhpbl_t = cl.subcloud_mse_supply(
        s_t.dT_dt_other, s_t.dq_dt_other, s_t.p_half, s_t.k_cbot,
        _k_start(s_t.p_full))
    M_b0_t, zmfmax_t, _ = _first_guess(s_t)
    ref_t = cl.shallow_closure(
        s_t.p_half, s_t.k_cbot, s_t.ldcum, s_t.ktype, zdhpbl_t,
        s_t.T_u, s_t.q_u, s_t.l_u, s_t.T_h, s_t.q_h,
        M_b0_t, zmfmax_t, s_t.dt, s_t.cfg)
    # rtol=1e-5, atol=1e-30: identical f32 pipeline, end-to-end vs direct
    np.testing.assert_allclose(
        np.asarray(res_t["M_b1"]), np.asarray(ref_t),
        rtol=1e-5, atol=1e-30,
        err_msg=str((float(res_t["M_b1"][0]), float(np.asarray(ref_t).ravel()[0]))))

    # ONE hand-computed check of ZDH on a TRIVIAL column (constant
    # profiles): with T_u - T_h = dT uniformly, q_u = q_h and l_u = 0,
    # both ZDH terms (kb and kb-1) equal RCPD*dT, so
    #   ZDH = RG * max(RCPD * dT, ZDH_FLOOR_FRAC_RCPD * RCPD)
    # unambiguously, and (away from the weak branch and the cap)
    #   M_b1 = min(ZDHPBL / ZDH, ZMFMAX).
    dT_off, dT, dq = 1.0, 2.0e-4, 1.0e-6
    s_c = _synth_deep()
    s_c.ktype = jnp.asarray([2], jnp.int32)
    s_c.T_u = s_c.T + jnp.asarray(np.float32(dT_off))   # constant offset
    s_c.dT_dt_other = jnp.asarray(np.full((1, 30), dT, np.float32))
    s_c.dq_dt_other = jnp.asarray(np.full((1, 30), dq, np.float32))
    ph_c = np.asarray(s_c.p_half[0], np.float64)
    kbc = int(s_c.k_cbot[0])
    zdhpbl_c = (cl.RLVTT * dq + cl.RCPD * dT) * (ph_c[kbc:] - ph_c[kbc - 1:-1]).sum()
    zdh_c = cl.RG * max(cl.RCPD * dT_off,
                        cl.ZDH_FLOOR_FRAC_RCPD * cl.RCPD)
    zmfmax_c = (ph_c[kbc] - ph_c[kbc - 1]) * s_c.cfg.rmfcfl / (cl.RG * 112.5)
    zmfub_c = min(zdhpbl_c / zdh_c, zmfmax_c)
    # precondition: off the weak branch (ZDH is large) and not capped
    assert zdh_c > cl.ZDH_WEAK_FRAC * cl.RG * cl.RCPD
    assert zmfub_c < zmfmax_c * (1 - 1e-6)
    m_c = cl.shallow_closure(
        s_c.p_half, s_c.k_cbot, s_c.ldcum, s_c.ktype,
        jnp.asarray(np.asarray([zdhpbl_c], np.float32)),
        s_c.T_u, s_c.q_u, s_c.l_u, s_c.T_h, s_c.q_h,
        jnp.asarray(np.asarray([0.01], np.float32)),
        jnp.asarray(np.asarray([zmfmax_c], np.float32)),
        s_c.dt, s_c.cfg)[0]
    # rtol=1e-4: float64 hand reference vs the f32 pipeline
    assert float(np.asarray(m_c).ravel()[0]) == pytest.approx(zmfub_c, rel=1e-4), (
        float(m_c[0]), zmfub_c)


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
        dT_dt_adv=s.dT_dt_adv, dq_dt_adv=s.dq_dt_adv,
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
    def grad_of(s, key="M_u"):
        def loss(T):
            res = cl.ifs_closure(
                M_u=s.M_u, PMFUS=s.PMFUS, PMFUQ=s.PMFUQ, PMFUL=s.PMFUL,
                PLUDE=s.PLUDE, PDMFUP=s.PDMFUP, PMFUDE_RATE=s.PMFUDE_RATE,
                PDMFEN=s.PDMFEN, T_u=s.T_u, q_u=s.q_u, l_u=s.l_u,
                k_ctop=s.k_ctop, pwmean=s.pwmean, zdpmean=s.zdpmean,
                ldcum=s.ldcum, ktype=s.ktype, k_cbot=s.k_cbot,
                k_dpl=s.k_dpl, T=T, q=s.q, qs=s.qs, p_full=s.p_full,
                p_half=s.p_half, geo_full=s.geo_full, geo_half=s.geo_half,
                T_h=s.T_h, q_h=s.q_h, dT_dt_other=s.dT_dt_other,
                dq_dt_other=s.dq_dt_other, dT_dt_adv=s.dT_dt_adv,
                dq_dt_adv=s.dq_dt_adv, land_frac=s.land_frac, dx_m=s.dx_m,
                dt=s.dt, cfg=s.cfg)
            return jnp.sum(res[key])
        return jax.grad(loss)(s.T)

    # CASE 1 -- the harness deep column at the default dt = 112.5 s is
    # NOT capped (measured M_b1 ~ 2.29 against ZMFMAX ~ 9.18), so the
    # closure output genuinely depends on the environmental T and the
    # gradient must be finite AND non-zero.
    s1, _, _ = _setup("deep")
    res1 = _ifs_closure(s1)
    _, zmfmax1, _ = _first_guess(s1)
    # assert the cap state so CASE 1 and CASE 2 can never silently swap
    assert 0.0 < float(res1["M_b1"][0]) < float(zmfmax1[0]) * (1 - 1e-6), (
        "CASE 1 precondition: harness deep column must be UNCAPPED "
        f"(M_b1={float(res1['M_b1'][0])}, ZMFMAX={float(zmfmax1[0])})")
    # sum(M_u) can be flat in T even on an uncapped column: when the per-level
    # RMFLIA bound (not the CAPE closure) sets ZMFS, the scaled profile no
    # longer depends on M_b1.  The closure's own output M_b1 is the quantity
    # that must carry the T dependence.
    g1 = grad_of(s1, "M_b1")
    g1_mu = grad_of(s1, "M_u")
    assert bool(jnp.all(jnp.isfinite(g1))), "grad must be finite everywhere"
    assert bool(jnp.all(jnp.isfinite(g1_mu))), "grad must be finite everywhere"
    # a zero gradient would mean the closure is disconnected from T
    assert float(jnp.max(jnp.abs(g1))) > 0.0

    # CASE 2 -- deliberately capped column: ZMFMAX = dp*RMFCFL/(RG*dt)
    # shrinks as dt grows, so raise dt until M_b1 sits ON the cap (to
    # within a relative 1e-6).  There the closure output is locally
    # constant in T, so the gradient may LEGITIMATELY be zero -- all we
    # demand is finiteness (never NaN/inf through the integer level
    # selection and max() kinks).
    s2 = None
    zmfmax2 = None
    for f in (2.0, 4.0, 8.0, 16.0, 32.0):
        cand, _, _ = _setup("deep", dt=112.5 * f)
        res = _ifs_closure(cand)
        _, zm, _ = _first_guess(cand)
        if float(res["M_b1"][0]) >= float(zm[0]) * (1 - 1e-6):
            s2, zmfmax2 = cand, zm
            break
    assert s2 is not None, (
        "could not drive the deep column onto the ZMFMAX cap with dt up "
        "to 3600 s; the cap logic has changed")
    res2 = _ifs_closure(s2)
    # assert the cap state so CASE 1 and CASE 2 can never silently swap
    assert float(res2["M_b1"][0]) >= float(zmfmax2[0]) * (1 - 1e-6), (
        "CASE 2 precondition: the column must sit ON the ZMFMAX cap")
    g2 = grad_of(s2)
    # finite only: at the cap d(sum M_u)/dT may legitimately be zero
    assert bool(jnp.all(jnp.isfinite(g2))), "grad must be finite everywhere"


def _k_start(p_full, njkt2_pa=60.0e2):
    """NJKT2 as OUR 0-based full-level index: the SMALLEST j with
    p_full[j] > njkt2_pa, floored at j = 1 (IFS default NJKT2 = 2).

    sucumf.F90:280-285 scans JLEV = NFLEVG..2 and keeps the LAST
    assignment, i.e. the topmost (smallest-index) level whose pressure
    still exceeds the cutoff -- the min form, not the max/surface form.
    """
    p = np.asarray(p_full)
    j = np.arange(p.shape[-1])
    cand = np.where(p > njkt2_pa, j, p.shape[-1])   # sentinel where p <= cutoff
    return jnp.asarray(np.maximum(cand.min(axis=-1), 1), jnp.int32)


def _synth_deep(dq_dt_adv=None, T=None, dt=112.5, nlev=30,
                lapse_frac=1.0, rh=0.8, mfu=0.01):
    """Synthetic deep-convective column on a STRETCHED sigma grid.

    lapse_frac = 1.0 gives a dry-static-energy-CONSTANT column: T drops by
    exactly g*dz/c_pd per level, so the ZHEAT stability integrand
    (T[j-1]-T[j] + dz_geo/c_pd)/T_h vanishes level by level with
    ZORCPD = 1/c_pd.  With the old ZORCPD = g/c_pd the same column gives a
    large positive integrand -- that asymmetry is the regression target.

    rh < 0.94 keeps the closure on the fast (ZSATFR <= 0.94) branch so the
    ZDQCV term is active (needed by test_njkt2_cutoff_window).
    """
    g = float(C.g)
    cpd = float(C.c_pd)
    f32 = lambda a: jnp.asarray(np.asarray(a, np.float32))

    sig = np.linspace(0.02, 1.0, nlev + 1) ** 1.3      # stretched sigma
    ph = 1013.2e2 * sig                                 # surface last
    pf = 0.5 * (ph[1:] + ph[:-1])
    dz = 150.0
    z_half = dz * (nlev - np.arange(nlev + 1))          # height, top = index 0
    geo_half = g * z_half
    geo_full = g * dz * (nlev - np.arange(nlev) - 0.5)

    T_col = 300.0 - lapse_frac * g * dz * (nlev - np.arange(nlev) - 0.5) / cpd
    if T is not None:
        T_col = T
    q_col = np.full(nlev, 0.01)
    qs_col = q_col / rh

    adv = np.zeros((1, nlev)) if dq_dt_adv is None else dq_dt_adv
    kb = jnp.asarray([nlev - 2], jnp.int32)
    zmfmax = (ph[nlev - 2] - ph[nlev - 3]) * 3.0 / (g * dt)
    return SimpleNamespace(
        T=f32(T_col[None]), q=f32(q_col[None]), qs=f32(qs_col[None]),
        p_full=f32(pf[None]), p_half=f32(ph[None]),
        geo_full=f32(geo_full[None]), geo_half=f32(geo_half[None]),
        T_h=f32(T_col[None]), q_h=f32(q_col[None]),
        T_u=f32(T_col[None]), q_u=f32(q_col[None]),
        l_u=f32(np.zeros((1, nlev))), M_u=f32(np.full((1, nlev), mfu)),
        dT_dt_other=f32(np.zeros((1, nlev))),
        dq_dt_other=f32(np.zeros((1, nlev))),
        dT_dt_adv=f32(np.zeros((1, nlev))),
        dq_dt_adv=f32(adv),
        ldcum=jnp.asarray([True]), ktype=jnp.asarray([1], jnp.int32),
        k_cbot=kb, k_ctop=jnp.asarray([2], jnp.int32),
        k_dpl=jnp.asarray([nlev - 1], jnp.int32),
        pwmean=f32([1.0]), zdpmean=f32([1.0e4]),
        land_frac=f32([0.0]), dx_m=f32([1.0e5]), dt=f32([dt]),
        cfg=cl.IFSClosureConfig(),
        zmfub=f32([0.01]), zmfmax=f32([zmfmax]),
    )


def _synth_deep_run(s):
    return cl.deep_cape_closure(
        s.T, s.q, s.qs, s.T_h, s.q_h, s.T_u, s.q_u, s.l_u, s.M_u,
        s.p_full, s.p_half, s.geo_full, s.geo_half,
        s.dT_dt_other, s.dq_dt_other, s.dT_dt_adv, s.dq_dt_adv,
        s.ldcum, s.ktype, s.k_cbot, s.k_ctop, s.k_dpl,
        s.pwmean, s.zdpmean, s.land_frac, s.dx_m, s.dt, s.cfg,
        s.zmfub, s.zmfmax)


def test_zheat_zorcpd_neutral_column_regression():
    # Neutral column (constant dry static energy): the ZHEAT integrand is
    # zero level by level with ZORCPD = 1/c_pd -> ZHEAT sits on its 1e-4
    # floor.  With the pre-fix ZORCPD = g/c_pd the same column produces a
    # large positive integrand and ZHEAT far above the floor, so this
    # test FAILS on the old coefficient.
    s = _synth_deep(lapse_frac=1.0)
    M_b1, zcape, zheat, zxtau = _synth_deep_run(s)
    assert float(zheat[0]) == pytest.approx(1.0e-4, rel=1e-3, abs=1e-9), (
        float(zheat[0]),)   # rel=1e-3: f32 cancellation residue ~1e-7/1e-4

    # NON-VACUITY control, made quantitative: with lapse_frac = L the
    # per-level ZHEAT integrand is (1-L)*g*dz/(c_pd*T_h[j]) * RG * M_u
    # (all constants as in the module), summed over the module's cloud
    # window k_ctop < j <= k_cbot and floored at 1e-4.  Recompute that
    # sum from the fixture's own numbers in float64 numpy and require the
    # module to reproduce it; additionally require the sub-adiabatic
    # column to beat the neutral (floored) value by a factor >= 10.
    L, dz, Mu = 0.5, 150.0, 0.01
    g = float(C.g)
    cpd = float(C.c_pd)
    RG = float(cl.RG)
    s2 = _synth_deep(lapse_frac=L)
    _, _, zheat2, _ = _synth_deep_run(s2)
    kt2, kb2 = int(s2.k_ctop[0]), int(s2.k_cbot[0])
    Th = np.asarray(s2.T_h[0], np.float64)
    expected2 = max(
        1.0e-4,
        float(np.sum((1.0 - L) * g * dz / (cpd * Th[kt2 + 1: kb2 + 1])
                     * RG * Mu)))
    # rtol=1e-4: float64 reference vs a f32 pipeline summing ~26 terms
    assert float(zheat2[0]) == pytest.approx(expected2, rel=1e-4, abs=1e-9), (
        float(zheat2[0]), expected2)
    assert expected2 > 1.0e-4            # the control is not on the floor
    assert float(zheat2[0]) >= 10.0 * float(zheat[0])


def test_tendency_pairs_not_interchangeable():
    zero = dict(dT_dt_other=0.0, dq_dt_other=0.0,
                dT_dt_adv=0.0, dq_dt_adv=0.0)
    only_adv = dict(dT_dt_other=0.0, dq_dt_other=0.0,
                    dT_dt_adv=-2.0e-4, dq_dt_adv=1.0e-6)
    only_tot = dict(dT_dt_other=2.0e-4, dq_dt_other=1.0e-6,
                    dT_dt_adv=0.0, dq_dt_adv=0.0)

    # --- deep column: ZCAPE2/ZDQCV consume the ADV pair only -----------
    s0, _, _ = _setup("deep", **zero)
    sa, _, _ = _setup("deep", **only_adv)
    st, _, _ = _setup("deep", **only_tot)
    _, zcape0, _, _ = _deep_closure(s0)
    _, zcapea, _, _ = _deep_closure(sa)
    _, zcapet, _, _ = _deep_closure(st)
    # adv pair moves ZCAPE (via ZCAPE2_mix and ZDQCV); total pair does not
    assert abs(float(zcapea[0]) - float(zcape0[0])) \
        > 1e-3 * max(abs(float(zcape0[0])), 1.0), (
        float(zcape0[0]), float(zcapea[0]))
    assert np.allclose(np.asarray(zcapet[0]), np.asarray(zcape0[0]),
                       rtol=1e-6, atol=1e-8)   # rtol=1e-6: identical integrand

    # --- trade column: ZDHPBL (shallow first guess) consumes the TOTAL
    # pair only (PTENT/PTENQ, cumastrn:484-496) -------------------------
    t0, _, _ = _setup("trade", **zero)
    ta, _, _ = _setup("trade", **only_adv)
    tt, _, _ = _setup("trade", **only_tot)
    M_b0_0, _, ldcum0 = _first_guess(t0)
    M_b0_a, _, ldcum_a = _first_guess(ta)
    M_b0_t, _, ldcum_t = _first_guess(tt)
    # total pair: ZDHPBL > 0 -> min(ZDHPBL/ZDH, ZMFMAX); with the total
    # pair zero ZDHPBL = 0 -> fallback 0.1*ZMFMAX and LDCUM=.FALSE.
    assert abs(float(M_b0_t[0]) - float(M_b0_0[0])) \
        > 1e-3 * max(abs(float(M_b0_0[0])), 1e-30)
    assert bool(ldcum_t[0]) is True and bool(ldcum0[0]) is False
    # adv pair: ZDHPBL identical -> bit-identical first guess
    assert float(M_b0_a[0]) == float(M_b0_0[0])
    assert bool(ldcum_a[0]) == bool(ldcum0[0])


def test_njkt2_cutoff_window():
    # A large spurious ADVective humidity tendency (PTENQA, the ZDQCV
    # integrand) ABOVE the 60 hPa NJKT2 cutoff must be invisible to the
    # closure; the same tendency BELOW the cutoff must move ZDQCV, hence
    # ZCAPE and M_b1.  The cutoff index is reproduced EXACTLY as the
    # module computes it (the min form, floored at 1) -- no independent
    # re-derivation that can drift by one level.
    s0 = _synth_deep()                                   # adv pair = 0
    M_b1_0, zcape_0, _, _ = _synth_deep_run(s0)

    p = np.asarray(s0.p_full[0], np.float64)
    nlev = p.size
    jj = np.arange(nlev)
    # module form: j = arange(nlev); j_njkt2 = min(where(p > njkt2, j, nlev-1))
    #              j_njkt2 = max(j_njkt2, 1)
    j_njkt2 = int(max(np.where(p > 60.0e2, jj, nlev - 1).min(), 1))

    # j_above is the first level AT OR ABOVE the cutoff: by the min form,
    # every j < j_njkt2 has p <= 60 hPa.
    # NOTE: ZQENH2(JK) averages PTENQA(JK) with PTENQA(JK-1) (cumastrn:757-758),
    # so a spike at exactly j_njkt2-1 legitimately reaches the first in-window
    # level.  "Outside the window" therefore means at least two levels above.
    j_above = max(j_njkt2 - 2, 0)
    assert 0 <= j_above < nlev
    assert p[j_above] <= 60.0e2, (j_above, p[j_above])

    kt, kb = int(s0.k_ctop[0]), int(s0.k_cbot[0])
    j_below = min(j_njkt2 + 5, kb - 1)
    assert kt < j_below < kb, (kt, j_below, kb)   # inside the ZDQCV mask
    assert p[j_below] > 60.0e2, (j_below, p[j_below])

    spike = np.zeros((1, nlev))
    spike[0, j_above] = 1.0e-2        # 1e-2 kg kg-1 s-1: enormous
    s_a = _synth_deep(dq_dt_adv=spike)
    M_b1_a, zcape_a, _, _ = _synth_deep_run(s_a)
    # above the cutoff: invisible -- exactly the baseline computation
    assert float(zcape_a[0]) == float(zcape_0[0])
    assert float(M_b1_a[0]) == float(M_b1_0[0])

    spike_b = np.zeros((1, nlev))
    spike_b[0, j_below] = 1.0e-2
    s_b = _synth_deep(dq_dt_adv=spike_b)
    M_b1_b, zcape_b, _, _ = _synth_deep_run(s_b)
    # below the cutoff: ZDQCV feeds ZCAPE and (pre-cap) M_b1
    assert abs(float(zcape_b[0]) - float(zcape_0[0])) > 1.0, (
        float(zcape_0[0]), float(zcape_b[0]))
    assert abs(float(M_b1_b[0]) - float(M_b1_0[0])) \
        > 1e-3 * max(abs(float(M_b1_0[0])), 1e-3), (
        float(M_b1_0[0]), float(M_b1_b[0]))


def test_pwmean_floors_finite_gradient():
    # Empty accumulator: pwmean_raw = 0, zdpmean = 0.  The source's two
    # floors (cuascn.F90:863: MAX(1e-2, pwmean/MAX(1, zdpmean)) then
    # sqrt(2*.)) give w_mean = sqrt(2e-2), a finite ZTAU, and -- crucially
    # -- a FINITE gradient of the closure w.r.t. pwmean_raw (the old bare
    # sqrt(pwmean) had d/dp sqrt(2p)|_0 = infinity).
    s = _synth_deep()
    s.pwmean = jnp.asarray(np.zeros(1, np.float32))
    s.zdpmean = jnp.asarray(np.zeros(1, np.float32))

    _, _, _, zxtau = _synth_deep_run(s)
    assert bool(jnp.isfinite(zxtau)[0])
    assert 720.0 - 1e-3 <= float(zxtau[0]) <= 10800.0 + 1e-3  # :861 clip

    def ztau_of(pw):
        s2 = _synth_deep()
        s2.pwmean = pw
        s2.zdpmean = jnp.asarray(np.zeros(1, np.float32))
        return _synth_deep_run(s2)[3][0]

    g = jax.grad(ztau_of)(jnp.asarray(np.zeros(1, np.float32)))
    assert bool(jnp.isfinite(g)[0]), float(g[0])   # finite in f32 and f64
    # and through M_b1 as well (ZTAU enters ZMFUB1 via ZXTAU)
    def mb_of(pw):
        s2 = _synth_deep()
        s2.pwmean = pw
        s2.zdpmean = jnp.asarray(np.zeros(1, np.float32))
        return _synth_deep_run(s2)[0][0]
    g2 = jax.grad(mb_of)(jnp.asarray(np.zeros(1, np.float32)))
    assert bool(jnp.isfinite(g2)[0]), float(g2[0])


def test_zmfs_floor_branches():
    # (:975-979) the 1e-10 floor lives ONLY inside the binding branch.
    # Branch A: honest ratio below 1e-10 with NO binding level -> the
    # ratio is kept, not raised to 1e-10.
    # Branch B: a level binds and zmfmax/PMFU there is < 1e-10 -> the
    # floored value 1e-10.
    s = _synth_deep()                       # stretched grid, M_u = 0.01
    M_b0 = jnp.asarray(np.array([0.01], np.float32))
    dt = jnp.asarray(np.array([112.5], np.float32))

    # Branch A: ratio = 1e-12, and M_u * 1e-12 << any per-level bound
    M_b1 = jnp.asarray(np.array([1.0e-14], np.float32))
    (_, _, _, _, _, _, _, _, zmfs_a) = cl.scale_profile(
        s.M_u, s.M_u, s.M_u, s.M_u, s.M_u, s.M_u, s.M_u, s.M_u,
        M_b0, M_b1, s.p_half, s.k_cbot, s.k_ctop, s.ldcum, dt, s.cfg)
    assert float(M_b1[0] / float(M_b0[0])) < 1.0e-10     # precondition
    assert float(zmfs_a[0]) == pytest.approx(1.0e-12, rel=1e-3, abs=0.0), (
        float(zmfs_a[0]),)   # rel=1e-3: f32 division of 1e-14/1e-2

    # Branch B: giant mass flux at one in-window level (j = 5) makes
    # zmfmax/PMFU there ~ 1e-12 < 1e-10, and the huge raw ratio binds.
    # Build the perturbed profile in numpy, then CONVERT to a JAX array
    # before using the .at[...] setter (numpy arrays have no .at).
    mu_big_np = np.asarray(s.M_u, np.float64).copy()
    mu_big_np[0, 5] = 1.0e12
    M_u_big = jnp.asarray(mu_big_np, jnp.float32)
    M_b1_big = jnp.asarray(np.array([100.0], np.float32))  # ratio 1e4
    (_, _, _, _, _, _, _, _, zmfs_b) = cl.scale_profile(
        M_u_big, M_u_big, M_u_big, M_u_big, M_u_big, M_u_big, M_u_big,
        M_u_big, M_b0, M_b1_big, s.p_half, s.k_cbot, s.k_ctop,
        s.ldcum, dt, s.cfg)
    assert float(M_b1_big[0] / float(M_b0[0])) > 1.0e-10  # precondition: binds
    assert float(zmfs_b[0]) == pytest.approx(1.0e-10, rel=1e-3, abs=0.0), (
        float(zmfs_b[0]),)


def test_first_guess_follows_the_pre_reclassification_type():
    """cumastrn builds ZMFUB at :563-576, BEFORE the ascent, and forms
    ZMFS = ZMFUB1/ZMFUB at :963 with that same number -- but it reclassifies
    KTYPE in between, at :635-641.  So a caller that reclassifies has to tell
    the closure which type the first guess was built with.  Deep and shallow
    take different branches there (ZMFMAX*0.1 against ZDHPBL/ZDH), so M_b0
    must follow ktype_first_guess and NOT the type used by the closure
    branches.  Deleting the kwarg makes the two calls below identical.
    """
    s, _, _ = _setup("deep")
    one = jnp.ones_like(s.ktype)
    as_deep = _ifs_closure(s, ktype=2 * one, ktype_first_guess=one)
    as_shallow = _ifs_closure(s, ktype=2 * one, ktype_first_guess=2 * one)
    assert float(as_deep["M_b0"][0]) != float(as_shallow["M_b0"][0])
    # the deep first guess is ZMFMAX*0.1 (cumastrn.F90:564), independent of
    # the sub-cloud supply, and much larger than the shallow ZDHPBL/ZDH here
    assert float(as_deep["M_b0"][0]) > float(as_shallow["M_b0"][0])
