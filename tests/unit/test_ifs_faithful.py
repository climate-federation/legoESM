"""The faithful IFS chain behind BechtoldConfig.use_ifs_ascent.

The switch is OFF by default, so the first test is the one that matters for
every production run: turning the field on must be the ONLY thing that changes
behaviour, and leaving it alone must change nothing at all.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection
from legoesm.atmosphere.physics.convection.config import BechtoldConfig
from legoesm.atmosphere.physics.convection import _ifs_faithful as F
from legoesm.atmosphere.physics.convection import _ifs_test_ascent as TA
from legoesm import constants
from legoesm.thermo import saturation_specific_humidity
from legoesm.atmosphere.physics.thermodynamics import compute_moist_adiabat


def _deep_column(nlev=30, ps=101300.0):
    """The frozen deep sounding of the other IFS tests (same builder)."""
    sig_h = np.linspace(0.0, 1.0, nlev + 1)
    p_half = (sig_h * ps)[None, :].astype(np.float32).copy()
    p_half[0, 0] = max(p_half[0, 0], 1000.0)
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    pf = p_full[0].astype(np.float64)
    T0, q0 = 300.0, 0.017
    rcpl = constants.R_d / constants.c_pd
    Tad = np.asarray(compute_moist_adiabat(
        jnp.array([T0]), jnp.array(pf)[None, :], jnp.array([q0])))[0]
    T = np.where(pf > 95000, T0 * (pf / ps) ** rcpl, Tad - 1.0)
    T = np.where(pf < 15000, np.maximum(T, 200.0), T)
    qs = np.asarray(saturation_specific_humidity(jnp.array(T), jnp.array(pf)))
    q = np.minimum(np.where(pf > 95000, q0, 0.8 * qs), q0)
    return (jnp.asarray(T[None, :], jnp.float32), jnp.asarray(q[None, :], jnp.float32),
            jnp.asarray(p_full), jnp.asarray(p_half))


def _kwargs(ncol, nlev):
    z = jnp.zeros((ncol, nlev))
    return dict(u=z, v=z, conv_prog_profile=z, conv_stoch_state=jnp.zeros(ncol),
                prng_key=None, dt=112.5, land_frac=jnp.zeros(ncol),
                shf_w_m2=jnp.full((ncol,), -20.0),
                lhf_w_m2=jnp.full((ncol,), -100.0),
                dT_dt_dyn=z, dq_dt_dyn=z)


def test_default_off_is_byte_identical():
    """The default must select the legacy scheme EXACTLY -- the switch is the
    only difference between the two configs, and an explicit False must be
    bit-identical to not setting it at all."""
    T, q, p_full, p_half = _deep_column()
    kw = _kwargs(*T.shape)
    a, mua, _ = bechtold_convection(T, q, p_full, p_half,
                                    config=BechtoldConfig(), **kw)
    b, mub, _ = bechtold_convection(T, q, p_full, p_half,
                                    config=BechtoldConfig(use_ifs_ascent=False), **kw)
    assert BechtoldConfig().use_ifs_ascent is False
    np.testing.assert_array_equal(np.asarray(a.dT_dt), np.asarray(b.dT_dt))
    np.testing.assert_array_equal(np.asarray(a.dq_v_dt), np.asarray(b.dq_v_dt))
    np.testing.assert_array_equal(np.asarray(mua), np.asarray(mub))


def test_faithful_chain_runs_and_is_finite():
    """Switching it on runs the whole ported chain (trigger -> first-guess ->
    ascent -> closure -> tendencies) and returns finite fields of the right
    shape.  It does NOT assert a non-zero tendency: see the declared gap in
    test_shallow_supply_gap_is_declared below."""
    T, q, p_full, p_half = _deep_column()
    kw = _kwargs(*T.shape)
    out, M_u, stoch = bechtold_convection(
        T, q, p_full, p_half, config=BechtoldConfig(use_ifs_ascent=True), **kw)
    for name, arr in (("dT_dt", out.dT_dt), ("dq_v_dt", out.dq_v_dt),
                      ("M_u", M_u)):
        assert np.asarray(arr).shape == T.shape, name
        assert bool(jnp.all(jnp.isfinite(arr))), name
    assert out.du_dt_conv is None and out.dv_dt_conv is None   # no CMT yet


def test_trigger_fires_on_this_column():
    """Guard against a silent no-op: the chain's FIRST stage must actually
    diagnose convection on this sounding, otherwise the finiteness test above
    would pass on a column that never convects."""
    T, q, p_full, p_half = _deep_column()
    geo_full, geo_half = F._hydrostatic_geopotential(T, q, p_half)
    ncol = T.shape[0]
    trig = TA.ifs_departure_search_refined(
        T, q, p_full, p_half, geo_full, geo_half,
        jnp.full((ncol,), -20.0), jnp.full((ncol,), -100.0),
        jnp.full((ncol,), 0.1), jnp.zeros(ncol), jnp.zeros_like(T),
        TA.IFSTestAscentConfig())
    assert float(np.asarray(trig.ldcum)[0]) > 0.5
    assert int(np.asarray(trig.ktype)[0]) in (1, 2)
    assert int(np.asarray(trig.k_cbot)[0]) > int(np.asarray(trig.k_ctop)[0])


def test_shallow_supply_gap_is_declared():
    """DECLARED GAP, pinned so it cannot be forgotten.

    cumastrn:570-581 sets the shallow cloud-base mass flux from ZDHPBL, the
    sub-cloud integral of the TOTAL physics tendencies (PTENT/PTENQ), and sets
    LDCUM=.FALSE. when that supply is not positive (:578).  This call site does
    not carry a total-physics tendency pair yet, so shallow columns are
    switched off and the chain returns zero for them.  When the pipeline
    plumbing lands, this test should FAIL and be replaced by a real
    shallow-convection assertion.
    """
    T, q, p_full, p_half = _deep_column()
    kw = _kwargs(*T.shape)
    out, _, _ = bechtold_convection(
        T, q, p_full, p_half, config=BechtoldConfig(use_ifs_ascent=True), **kw)
    geo_full, geo_half = F._hydrostatic_geopotential(T, q, p_half)
    trig = TA.ifs_departure_search_refined(
        T, q, p_full, p_half, geo_full, geo_half,
        jnp.full((1,), -20.0), jnp.full((1,), -100.0), jnp.full((1,), 0.1),
        jnp.zeros(1), jnp.zeros_like(T), TA.IFSTestAscentConfig())
    if int(np.asarray(trig.ktype)[0]) == 2:
        assert float(jnp.max(jnp.abs(out.dT_dt))) == 0.0, (
            "shallow column with no total-physics tendency must be inert "
            "(cumastrn:578); if this fires, the PTENT/PTENQ plumbing landed "
            "and this test needs replacing")


def test_no_cloud_edge_dipole():
    """REGRESSION for the +-11000 K/day dipole (2026-09-17).

    Two stages were missing between the closure and the tendencies: CUFLXN's
    environmental subtraction (cuflxn.F90:250-251) and the absolute-flux
    reconstruction cumastrn does before CUDTDQN when RMFSOLTQ > 0
    (cumastrn.F90:1194-1216). Without them the environment is subtracted twice
    and the two lowest levels exchange ~1e4 K/day.  Convective heating on a
    single column cannot physically reach that; bound it well below, and check
    the bottom pair is not an equal-and-opposite pair.
    """
    T, q, p_full, p_half = _deep_column()
    ncol, nlev = T.shape
    kw = _kwargs(ncol, nlev)
    kw["dq_dt_dyn"] = jnp.full((ncol, nlev), 1e-8)      # a real sub-cloud supply
    out, _, _ = bechtold_convection(
        T, q, p_full, p_half, config=BechtoldConfig(use_ifs_ascent=True),
        dT_dt_rad=jnp.full((ncol, nlev), -1.5 / 86400.0), **kw)
    dT_day = np.asarray(out.dT_dt[0]) * 86400.0
    assert np.all(np.abs(dT_day) < 500.0), (
        "cloud-edge dipole is back", dT_day[np.abs(dT_day) > 500.0])
    # the failure mode was specifically equal-and-opposite at the bottom pair
    bottom_pair = abs(dT_day[-1] + dT_day[-2])
    bottom_size = abs(dT_day[-1]) + abs(dT_day[-2])
    assert not (bottom_size > 100.0 and bottom_pair < 0.05 * bottom_size), (
        dT_day[-2], dT_day[-1])
    # not vacuous: the chain must actually be convecting here.  The bound is
    # 0.1 and not 1 K/day because this column is typed SHALLOW by the trigger
    # on 30 levels, so its mass flux is ZDHPBL/ZDH, and ZDH is built on the
    # HALF-level environment (cumastrn.F90:570 uses ZTENH/ZQENH).  The fixture
    # has a sharp humidity step at 950 hPa, so the half-level humidity at cloud
    # base is 0.0129 against 0.017 at the full level -- a much larger ZQUMQE
    # and a correspondingly smaller mass flux than the full-level shortcut this
    # chain used until 2026-09-17.
    # MEASURED 2026-09-17 on this fixture: 0.33 K/day, with ktype 2 -> 2 (the
    # realised cloud depth is 6.8 kPa, far under the 20 kPa split) and
    # ZMFS = 0.96, so the retype and the closure rescale are both inert here
    # and the value is set by the half-level ZDH alone.  A band, not a floor,
    # so drift in EITHER direction trips.
    # RE-BANDED 2026-09-29 (specific-basis LCL moved the fixture sounding):
    # MEASURED 0.150 K/day, ktype still 2; band kept at the same +-40 %.
    assert 0.09 < float(np.max(np.abs(dT_day))) < 0.21


def test_ktype_reclassified_against_ascent_top():
    """Pins cumastrn.F90:635-641.

    Between the single CUASCN ascent (:611-625) and the final closure (:845
    deep, :894 shallow) the source reclassifies KTYPE against the ACTUAL cloud
    depth ZPBMPT = PAPH(KCBOT) - PAPH(KCTOP), with RDEPTHS =
    IFSTestAscentConfig.depth_split_pa: a deep column (1) whose realised depth
    fell short of the split becomes shallow (2), a shallow column (2) that
    reached the split becomes deep (1), and an INACTIVE column (LDCUM false)
    keeps its incoming type.  Deleting the helper fails this test.
    """
    _, _, _, p_half = _deep_column(nlev=30)
    ncol = 5
    p_half = jnp.broadcast_to(jnp.asarray(p_half), (ncol, p_half.shape[-1]))
    cfg = TA.IFSTestAscentConfig()
    kb = jnp.full((ncol,), 27, jnp.int32)                 # PAPH(IKB)
    k_ctop = jnp.asarray([10, 25, 10, 25, 25], jnp.int32)
    ktype_in = jnp.asarray([1, 1, 2, 2, 1], jnp.int32)
    ldcum = jnp.asarray([True, True, True, True, False])
    # geometry pinned so the case table cannot silently go vacuous if the
    # fixture grid ever changes
    depth = np.asarray(p_half[0, kb[0]] - p_half[0, k_ctop])
    np.testing.assert_array_equal(depth >= float(cfg.depth_split_pa),
                                  [True, False, True, False, False])
    ktype_out = F._reclassify_ktype(ktype_in, ldcum, p_half, kb, k_ctop,
                                    cfg.depth_split_pa)
    np.testing.assert_array_equal(np.asarray(ktype_out), [1, 2, 1, 2, 1])


def test_closure_divides_by_the_mass_flux_that_launched_the_ascent(monkeypatch):
    """ZMFS = ZMFUB1/ZMFUB (cumastrn.F90:963) has to divide by the SAME ZMFUB
    that launched CUASCN, so the chain's first guess and the one ifs_closure
    rebuilds internally must be the same number.  Two ways that breaks, both
    gated here: the closure rebuilding it from the RECLASSIFIED type
    (cumastrn.F90:635-641 runs after the ascent, the first guess at :563-576
    before it), and the two sites disagreeing on half- vs full-level
    environment at cloud base.  Also asserts the reclassification is actually
    wired into the chain: deleting only the call site fails here.
    """
    seen = {}
    real_asc, real_clo = F.ifs_updraught_ascent, F.ifs_closure
    real_rec = F._reclassify_ktype

    def spy_asc(*a, **k):
        seen["M_b_launch"] = np.asarray(a[19])      # positional M_b
        return real_asc(*a, **k)

    def spy_clo(*a, **k):
        out = real_clo(*a, **k)
        seen["M_b0"] = np.asarray(out["M_b0"])
        return out

    def spy_rec(*a, **k):
        seen["reclassified"] = seen.get("reclassified", 0) + 1
        return real_rec(*a, **k)

    monkeypatch.setattr(F, "ifs_updraught_ascent", spy_asc)
    monkeypatch.setattr(F, "ifs_closure", spy_clo)
    monkeypatch.setattr(F, "_reclassify_ktype", spy_rec)

    T, q, p_full, p_half = _deep_column()
    ncol, nlev = T.shape
    kw = _kwargs(ncol, nlev)
    kw["dq_dt_dyn"] = jnp.full((ncol, nlev), 1e-8)   # a real sub-cloud supply,
    kw["dT_dt_rad"] = jnp.full((ncol, nlev), -1.5 / 86400.0)   # so the column
    bechtold_convection(T, q, p_full, p_half,                  # is ACTIVE: an
                        config=BechtoldConfig(use_ifs_ascent=True), **kw)
    assert seen["reclassified"] == 1
    assert float(seen["M_b_launch"].max()) > 0.0
    # this fixture does NOT cross the depth split (6.8 kPa against 20 kPa), so
    # it does not exercise ktype_first_guess; that seam is gated separately by
    # test_ifs_closure.test_first_guess_follows_the_pre_reclassification_type
    np.testing.assert_allclose(seen["M_b0"], seen["M_b_launch"], rtol=1e-6)

    # Second phase: FORCE a retype, which this fixture does not do on its own.
    # The closure branches now see KTYPE=1 while the first guess was built on
    # KTYPE=2, so without ktype_first_guess the closure would rebuild M_b0 as
    # the deep ZMFMAX*0.1 and the identity would break by orders of magnitude.
    seen.clear()
    monkeypatch.setattr(F, "_reclassify_ktype",
                        lambda ktype, *a, **k: jnp.ones_like(ktype))
    bechtold_convection(T, q, p_full, p_half,
                        config=BechtoldConfig(use_ifs_ascent=True), **kw)
    np.testing.assert_allclose(seen["M_b0"], seen["M_b_launch"], rtol=1e-6)


@pytest.fixture
def _x64():
    """The budget residual is a difference of telescoping flux terms, so it is
    float32 roundoff that has to be small, not the budget; the check needs x64
    to be meaningful at the stated tolerance (codex review)."""
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


def test_convective_water_is_conserved(_x64):
    """Column water budget: what cudtdqn sinks as vapour must come back.

    cudtdqn.F90:343-347 removes PLUDE (detrained condensate) and PDMFUP
    (precipitation generation) from PTENQ, scaled by ZDP = RG/dp; the host
    contract returns them as dq_c_conv_dt = PLUDE*g/dp (anvil cloud water,
    added to q_c next step) and dq_r_conv_dt = PDMFUP*g/dp (rain, whose
    column integral leaves as surface precipitation).  With m_k = dp_k/g the
    closed budget is sum_k m_k*(dq_v_dt + dq_c_conv_dt + dq_r_conv_dt) = 0:
    the tendency module's interior rows telescope against its JK = KLEV row,
    so only the returned sources are left.  The tolerance is RELATIVE to the
    summed size of the terms; an absolute epsilon would pass on an inert
    column, which the non-vacuity assert also rules out.
    """
    T, q, p_full, p_half = _deep_column()
    ncol, nlev = T.shape
    kw = _kwargs(ncol, nlev)
    kw["dq_dt_dyn"] = jnp.full((ncol, nlev), 1e-8)   # the active-column recipe
    out, _, _ = bechtold_convection(                 # of test_no_cloud_edge_dipole
        T, q, p_full, p_half,
        config=BechtoldConfig(use_ifs_ascent=True),
        dT_dt_rad=jnp.full((ncol, nlev), -1.5 / 86400.0), **kw)

    # bookkeeping in float64 so the check adds no roundoff of its own
    dp = (np.asarray(p_half[0, 1:], np.float64)
          - np.asarray(p_half[0, :-1], np.float64))
    mass = dp / float(constants.g)                    # m_k = dp_k/g [kg m-2]
    dq_v = np.asarray(out.dq_v_dt[0], np.float64)
    dq_c = np.asarray(out.dq_c_conv_dt[0], np.float64)
    dq_r = np.asarray(out.dq_r_conv_dt[0], np.float64)

    sink = float(np.sum(dq_v * mass))
    detrained = float(np.sum(dq_c * mass))
    rained = float(np.sum(dq_r * mass))
    scale = float(np.sum((np.abs(dq_v) + np.abs(dq_c) + np.abs(dq_r)) * mass))

    assert detrained + rained > 0.0, (detrained, rained)   # not vacuous
    assert np.all(dq_c >= 0.0)      # the driver relies on both being sources
    assert np.all(dq_r >= 0.0)
    residual = sink + detrained + rained
    assert abs(residual) <= 1.0e-5 * scale, (
        "convective water not conserved", residual, scale, sink, detrained,
        rained)
