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
                shf_w_m2=jnp.full((ncol,), 20.0),     # positive upward (model sign)
                lhf_w_m2=jnp.full((ncol,), 100.0),
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


def test_default_bechtold_reproduces_the_ifs_source_ascent():
    """At the defaults the mapped ascent config IS the IFS source one (so the
    production output does not move); every mapped field has a counterpart."""
    from legoesm.atmosphere.physics.convection._ifs_ascent import IFSAscentConfig
    got = F.ifs_ascent_config(BechtoldConfig(use_ifs_ascent=True))
    ref = IFSAscentConfig()
    for f in ref._fields:
        np.testing.assert_allclose(getattr(got, f), getattr(ref, f),
                                   rtol=1e-12, err_msg=f)


def test_driver_set_bechtold_values_reach_the_ifs_ascent(monkeypatch):
    """A BechtoldConfig value set by the run must reach the ascent the
    faithful chain launches (it used to be built from IFS defaults)."""
    seen = {}
    real = F.ifs_updraught_ascent

    def spy(*a, **k):
        seen["cfg"] = k["cfg"]
        return real(*a, **k)

    monkeypatch.setattr(F, "ifs_updraught_ascent", spy)
    T, q, p_full, p_half = _deep_column()
    kw = _kwargs(*T.shape)
    cfg = BechtoldConfig(use_ifs_ascent=True, epsilon_deep=2.5e-3,
                         delta_deep=1.0e-4, epsilon_shallow=6.0e-3,
                         rprcon=2.0e-3, dnoprc=5.0e-4)
    bechtold_convection(T, q, p_full, p_half, config=cfg, **kw)
    a = seen["cfg"]
    assert a.entrorg == 2.5e-3 and a.detrpen == 1.0e-4
    assert a.rprcon == 2.0e-3 and a.dnoprc == 5.0e-4
    np.testing.assert_allclose(a.entshalp, 6.0e-3 / 2.5e-3, rtol=1e-12)


def test_default_bechtold_reproduces_the_ifs_source_closure_and_test_ascent():
    """At the Bechtold defaults the closure and test-parcel configs ARE the
    IFS source ones (default output does not move)."""
    from legoesm.atmosphere.physics.convection._ifs_closure import IFSClosureConfig
    cfg = BechtoldConfig(use_ifs_ascent=True)
    for got, ref in ((F.ifs_closure_config(cfg), IFSClosureConfig()),
                     (F.ifs_test_ascent_config(cfg), TA.IFSTestAscentConfig())):
        for f in ref._fields:
            g, r = getattr(got, f), getattr(ref, f)
            if isinstance(r, str):
                assert g == r, f
            else:
                np.testing.assert_allclose(g, r, rtol=1e-12, err_msg=f)


def test_driver_set_bechtold_values_reach_the_ifs_closure_and_trigger(monkeypatch):
    """Run-set cape_qadv_weight reaches the closure (RCAPQADV) and run-set
    epsilon_deep the trigger's test parcel (ENTRORG)."""
    seen = {}
    real_trig, real_clo = F.ifs_departure_search_refined, F.ifs_closure

    def spy_trig(*a, **k):
        seen["ta"] = k["cfg"]
        return real_trig(*a, **k)

    def spy_clo(*a, **k):
        seen["clo"] = a[-1]
        return real_clo(*a, **k)

    monkeypatch.setattr(F, "ifs_departure_search_refined", spy_trig)
    monkeypatch.setattr(F, "ifs_closure", spy_clo)
    T, q, p_full, p_half = _deep_column()
    kw = _kwargs(*T.shape)
    cfg = BechtoldConfig(use_ifs_ascent=True, epsilon_deep=2.5e-3,
                         cape_qadv_weight=0.6)
    bechtold_convection(T, q, p_full, p_half, config=cfg, **kw)
    assert seen["ta"].entr_deep_base == 2.5e-3
    assert seen["clo"].rcapqadv == 0.6


def _param_grad_and_fd(nlev, field):
    """d(sum dT_dt)/d(field) on the deep sounding: reverse mode and a central
    difference with relative step 1e-4."""
    T, q, p_full, p_half = [a.astype(jnp.float64) for a in _deep_column(nlev=nlev)]
    kw = _kwargs(*T.shape)
    base = BechtoldConfig(use_ifs_ascent=True)
    v = float(getattr(base, field))

    def f(x):
        out, _, _ = bechtold_convection(T, q, p_full, p_half,
                                        config=base._replace(**{field: x}), **kw)
        return jnp.sum(out.dT_dt)

    g = float(jax.grad(f)(jnp.asarray(v)))
    h = 1e-4 * v
    fd = (float(f(jnp.asarray(v + h))) - float(f(jnp.asarray(v - h)))) / (2 * h)
    return g, fd


@pytest.mark.parametrize("field", ["epsilon_deep", "rprcon"])
def test_reverse_grad_matches_fd_on_convecting_column(_x64, field):
    """At 90 levels the deep sounding convects and both parameters move the
    heating; reverse mode was NaN here (sqrt of a clamped zero)."""
    g, fd = _param_grad_and_fd(90, field)
    assert fd != 0.0
    assert g == pytest.approx(fd, rel=1e-3)


@pytest.mark.parametrize("field", ["epsilon_deep", "rprcon"])
def test_reverse_grad_finite_on_non_convecting_column(_x64, field):
    """At 30 levels the column is shallow and neither parameter matters: the
    gradient must be the finite zero the finite difference measures."""
    g, fd = _param_grad_and_fd(30, field)
    assert np.isfinite(g)
    assert g == fd == 0.0


def test_faithful_chain_receives_ifs_signed_surface_fluxes(monkeypatch):
    """bechtold takes positive-upward fluxes; the IFS trigger reads PAHFS /
    PQHFL positive downward (cubasen.F90:341), so the chain must see them
    negated."""
    T, q, p_full, p_half = _deep_column()
    kw = _kwargs(*T.shape)
    seen = {}
    orig = F.ifs_departure_search_refined

    def spy(*a, **k):
        seen["shf"], seen["lhf"] = k["shf_w_m2"], k["lhf_w_m2"]
        return orig(*a, **k)

    monkeypatch.setattr(F, "ifs_departure_search_refined", spy)
    bechtold_convection(T, q, p_full, p_half,
                        config=BechtoldConfig(use_ifs_ascent=True), **kw)
    np.testing.assert_array_equal(np.asarray(seen["shf"]), -np.asarray(kw["shf_w_m2"]))
    np.testing.assert_array_equal(np.asarray(seen["lhf"]), -np.asarray(kw["lhf_w_m2"]))


def test_upward_surface_fluxes_launch_the_surface_parcel(monkeypatch):
    """Upward surface heat and moisture fluxes, as the surface layer passes
    them (positive upward), must let the surface test parcel go (the 30-level
    sounding departs from the surface); with the sign unflipped the buoyancy
    flux had the wrong sign and convection was switched off."""
    T, q, p_full, p_half = _deep_column()
    kw = _kwargs(*T.shape)
    got = {}
    orig = F.ifs_departure_search_refined

    def spy(*a, **k):
        got["trig"] = orig(*a, **k)
        return got["trig"]

    monkeypatch.setattr(F, "ifs_departure_search_refined", spy)
    bechtold_convection(T, q, p_full, p_half,
                        config=BechtoldConfig(use_ifs_ascent=True), **kw)
    trig = got["trig"]
    assert float(np.asarray(trig.ldcum)[0]) > 0.5
    assert int(np.asarray(trig.k_dpl)[0]) == T.shape[1] - 1


# ---------------------------------------------------------------------------
# Shallow closure with no sub-cloud supply (BechtoldConfig.ifs_shallow_nosupply)
# 60-level deep sounding: first triggered deep, reclassified shallow.
# ---------------------------------------------------------------------------

_NOSUPPLY_CASES = {   # case -> (dq_dt_dyn [kg/kg/s], dT_dt_rad [K/day] or None)
    "no_supply": (0.0, None),
    "rad_cooling": (0.0, -1.5),
    "supply": (1e-7, None),
}
# measured 2026-10-29 (x64): (cloud-base flux M_b1 [kg/m2/s], peak |dT/dt| [K/day])
_NOSUPPLY_FIRST_GUESS = {
    "no_supply": (0.45912254813980874, 292.0799722680822),
    "rad_cooling": (0.45912254813980874, 292.0799722680822),
    "supply": (0.20041796379722074, 129.81216574308928),
}
# Re-measured 2026-10-29 when the faithful chain gained below-cloud rain
# evaporation: in these two weak cases the peak is now the sub-cloud cooling
# (was 0.7339 and 1.2954e-4 K/day); M_b1 unchanged.
_NOSUPPLY_DEFAULT = {   # KNOWN DIFFERENCE: default "zero", width 1 W/m2
    "no_supply": (0.0010753948197864782, 1.2393810288776876),
    "rad_cooling": (1.8977114269968982e-07, 0.000248384298174642),
    "supply": (0.20041796379722074, 129.81216574308928),
}


def _nosupply_run(case, monkeypatch, **cfg_kw):
    from legoesm.atmosphere.physics.convection import _ifs_closure as C
    T, q, p_full, p_half = [a.astype(jnp.float64) for a in _deep_column(nlev=60)]
    kw = _kwargs(*T.shape)
    dq, rad = _NOSUPPLY_CASES[case]
    kw["dq_dt_dyn"] = jnp.full(T.shape, dq)
    if rad is not None:
        kw["dT_dt_rad"] = jnp.full(T.shape, rad / 86400.0)
    seen = {}
    orig = C.shallow_closure

    def spy(*a):
        seen["args"] = a
        seen["mb1"] = orig(*a)
        return seen["mb1"]

    monkeypatch.setattr(C, "shallow_closure", spy)
    out, _, _ = bechtold_convection(
        T, q, p_full, p_half,
        config=BechtoldConfig(use_ifs_ascent=True, **cfg_kw), **kw)
    peak = float(np.max(np.abs(np.asarray(out.dT_dt[0])))) * 86400.0
    return float(np.asarray(seen["mb1"])[0]), peak, seen["args"]


@pytest.mark.parametrize("case", sorted(_NOSUPPLY_CASES))
def test_shallow_nosupply_first_guess_is_the_source(_x64, case, monkeypatch):
    """'first_guess' = cumastrn.F90:912: with ZDHPBL <= 0 the closure keeps
    the first guess (0.1*ZMFMAX for a deep-triggered column); the OpenIFS
    single-column oracle reproduces 0.45910 kg/m2/s and 292.30 K/day here
    (the 5e-5 offset is the two models' gravity constants)."""
    mb1, peak, args = _nosupply_run(case, monkeypatch, ifs_shallow_nosupply="first_guess")
    zmfub, zdhpbl = np.asarray(args[10])[0], float(np.asarray(args[4])[0])
    if zdhpbl <= 0.0:
        assert mb1 == zmfub
    np.testing.assert_allclose([mb1, peak], _NOSUPPLY_FIRST_GUESS[case], rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("case", sorted(_NOSUPPLY_CASES))
def test_shallow_nosupply_hard_limit_is_max_supply_zero(_x64, case, monkeypatch):
    """Width 0 is the hard max(ZDHPBL, 0): no supply gives exactly no
    cloud-base flux; a positive supply is the source's ZDHPBL/ZDH."""
    mb1, _, _ = _nosupply_run(case, monkeypatch, ifs_shallow_supply_width_w_m2=0.0)
    if _NOSUPPLY_CASES[case][0] == 0.0:
        assert mb1 == 0.0
    else:
        assert mb1 == _NOSUPPLY_FIRST_GUESS[case][0]


@pytest.mark.parametrize("case", sorted(_NOSUPPLY_CASES))
def test_shallow_nosupply_default_gap_per_case(_x64, case, monkeypatch):
    """KNOWN DIFFERENCE at the default ('zero', width 1 W/m2), per case."""
    mb1, peak, _ = _nosupply_run(case, monkeypatch)
    np.testing.assert_allclose([mb1, peak], _NOSUPPLY_DEFAULT[case], rtol=1e-6, atol=1e-12)


def test_shallow_nosupply_gradient_through_the_supply_ramp(_x64, monkeypatch):
    """Finite, non-zero derivative of the cloud-base flux with respect to the
    supply at the default width, on both sides of zero supply."""
    from jax.test_util import check_grads
    from legoesm.atmosphere.physics.convection import _ifs_closure as C
    _, _, args = _nosupply_run("no_supply", monkeypatch)
    g = constants.g

    def f(z):
        a = list(args)
        a[4] = z
        return C.shallow_closure(*a)

    for z0 in (-g, 0.0, g):
        z = jnp.asarray([z0])
        check_grads(f, (z,), order=2, modes=("fwd", "rev"))
        d = float(jax.grad(lambda x: f(x).sum())(z)[0])
        assert np.isfinite(d) and d > 0.0


# --- below-cloud rain evaporation (cuflxn.F90:443-500) on the faithful chain ---
# OpenIFS single-column oracle on the port_deep column (scripts/validate/oracle/
# ifs_cumastrn, run.sh + compare_subcloud_evap.py, 2026-10-29): its production
# sum, PQSEN and inputs at the levels the evaporation uses (0-based: 48 = KCTOP,
# 57 = KCBOT, 58-59 evaporate), and PDMFUP after CUFLXN at 58, 59.
_ORC_PROD = 0.0004629992212176128              # sum of PDMFUP > 0 [kg/m2/s]
_ORC_DT = 112.5
_ORC_Q = {48: 0.010755470022559166, 57: 0.017000000923871994,
          58: 0.017000000923871994, 59: 0.017000000923871994}
_ORC_QSEN = {48: 0.013439665915219487, 57: 0.01843255585663878,
             58: 0.019799489482692344, 59: 0.021232030435932126}
_ORC_PH = (96235.0, 97923.3359375, 99611.6640625, 101300.0)   # PAPH 57..60
_ORC_EVAP_NODSNOW = (-3.0949592889734816e-05, -4.5176587955529223e-05)  # LMFDSNOW=F
_ORC_EVAP_DEFAULT = (-2.9062759385935807e-05, -4.2339741776857620e-05)  # source defaults
# leaf/oracle(LMFDSNOW=F) - 1: constants.g vs the oracle's RG in RCPECONS=5.44e-4/RG
_LEAF_FORMULA_GAP = (4.996859e-05, 4.790063e-05)
# KNOWN DIFFERENCE: chain/oracle(default) - 1 in PDMFUP at 58, 59; detrained
# precipitation (LMFDSNOW, PSNDE) is not removed from the falling flux.
_CHAIN_EVAP_GAP = (5.465891e-02, 5.511759e-02)
# measured 2026-10-29 (x64): chain PDMFUP into CUDTDQN at 58, 59 [kg/m2/s]
_EVAP_PER_CASE = {
    ("no_supply", "first_guess"): (-3.065129820791043e-05, -4.467340613930167e-05),
    ("rad_cooling", "first_guess"): (-3.065129820791043e-05, -4.467340613930167e-05),
    ("supply", "first_guess"): (-1.8988099923941898e-05, -2.7185109827724597e-05),
    ("no_supply", "zero"): (-9.265940506730531e-07, -1.3440622970114357e-07),
    ("rad_cooling", "zero"): (-1.8723098894160612e-10, 0.0),
    ("supply", "zero"): (-1.8988099923941898e-05, -2.7185109827724597e-05),
}


def _leaf_on_oracle_column(q58, q59):
    """The shared leaf on a 3-layer stand-in for the oracle column: layer 0
    carries the whole cloud-layer production (not evaporated, it is above
    KCBOT), layers 1-2 are the oracle's levels 58-59; the half levels are the
    oracle's PAPH, the last one its surface."""
    from legoesm.atmosphere.physics.convection.bechtold import (
        ifs_subcloud_rain_evaporation)
    ph = jnp.asarray([_ORC_PH], dtype=jnp.float64)
    dp = ph[:, 1:] - ph[:, :-1]
    q = jnp.stack([jnp.asarray(_ORC_Q[57]), q58, q59])[None]
    qs = jnp.asarray([[_ORC_QSEN[57], _ORC_QSEN[58], _ORC_QSEN[59]]])
    src = jnp.asarray([[_ORC_PROD, 0.0, 0.0]]) * constants.g / dp
    rh_b = _ORC_Q[57] / _ORC_QSEN[57]
    rh_t = _ORC_Q[48] / _ORC_QSEN[48]
    ev, _, _ = ifs_subcloud_rain_evaporation(
        q, qs, ph, dp, src, jnp.asarray([[0.0, 1.0, 1.0]]),
        jnp.asarray([rh_b]), jnp.asarray([rh_t]), jnp.asarray([0.0]), _ORC_DT,
        rmfcfl=F.RMFCFL)
    return -ev[0, 1:] * dp[0, 1:] / constants.g      # as PDMFUP increments


def test_subcloud_evap_leaf_is_the_openifs_formula(_x64):
    """The leaf on the oracle's own production and PQSEN reproduces OpenIFS
    with detrained precipitation off; the stored gap is the gravity constant."""
    got = np.asarray(_leaf_on_oracle_column(jnp.asarray(_ORC_Q[58]),
                                            jnp.asarray(_ORC_Q[59])))
    gap = got / np.asarray(_ORC_EVAP_NODSNOW) - 1.0
    np.testing.assert_allclose(gap, _LEAF_FORMULA_GAP, rtol=1e-6, atol=1e-12)


def _evap_run(case, mode, monkeypatch):
    T, q, p_full, p_half = [a.astype(jnp.float64) for a in _deep_column(nlev=60)]
    kw = _kwargs(*T.shape)
    dq, rad = _NOSUPPLY_CASES[case]
    kw["dq_dt_dyn"] = jnp.full(T.shape, dq)
    if rad is not None:
        kw["dT_dt_rad"] = jnp.full(T.shape, rad / 86400.0)
    seen = {}
    orig = F.ifs_convective_tendencies

    def spy(*a, **k):
        seen["pdmfup"] = np.asarray(a[-3])[0]
        return orig(*a, **k)

    monkeypatch.setattr(F, "ifs_convective_tendencies", spy)
    out, _, _ = bechtold_convection(
        T, q, p_full, p_half,
        config=BechtoldConfig(use_ifs_ascent=True, ifs_shallow_nosupply=mode), **kw)
    return seen["pdmfup"], out


def test_faithful_subcloud_evap_gap_vs_openifs(_x64, monkeypatch):
    """KNOWN DIFFERENCE per level: in the source's no-supply mode the chain
    evaporates 5.5 % more than OpenIFS (detrained precipitation unported)."""
    up, _ = _evap_run("no_supply", "first_guess", monkeypatch)
    gap = up[58:60] / np.asarray(_ORC_EVAP_DEFAULT) - 1.0
    np.testing.assert_allclose(gap, _CHAIN_EVAP_GAP, rtol=1e-6, atol=1e-12)
    assert np.all(up[:58] >= 0.0)        # evaporation only below cloud base


@pytest.mark.parametrize("case,mode", sorted(_EVAP_PER_CASE))
def test_faithful_subcloud_evap_per_case(_x64, case, mode, monkeypatch):
    """Stored per-case PDMFUP at the evaporating levels; host rain stays a
    non-negative source and carries the column-wide evaporation debit."""
    up, out = _evap_run(case, mode, monkeypatch)
    np.testing.assert_allclose(up[58:60], _EVAP_PER_CASE[(case, mode)],
                               rtol=1e-6, atol=1e-12)
    assert np.all(np.asarray(out.dq_r_conv_dt) >= 0.0)
    # column water closes with the evaporation in it (budget and tolerance of
    # test_convective_water_is_conserved)
    _, _, _, p_half = _deep_column(nlev=60)
    mass = np.diff(np.asarray(p_half[0], np.float64)) / float(constants.g)
    terms = [np.asarray(x[0], np.float64) for x in
             (out.dq_v_dt, out.dq_c_conv_dt, out.dq_r_conv_dt)]
    residual = sum(float(np.sum(t * mass)) for t in terms)
    scale = float(np.sum(sum(np.abs(t) for t in terms) * mass))
    assert abs(residual) <= 1.0e-5 * scale, (residual, scale)


def test_subcloud_evap_gradient(_x64):
    """Finite, non-zero derivative of the evaporation with respect to the
    sub-cloud humidity through the faithful-path leaf call."""
    from jax.test_util import check_grads

    def f(q58, q59):
        return _leaf_on_oracle_column(q58, q59).sum()

    args = (jnp.asarray(_ORC_Q[58]), jnp.asarray(_ORC_Q[59]))
    check_grads(f, args, order=2, modes=("fwd", "rev"))
    d = jax.grad(f, argnums=(0, 1))(*args)
    assert all(np.isfinite(float(x)) and float(x) > 0.0 for x in d)
