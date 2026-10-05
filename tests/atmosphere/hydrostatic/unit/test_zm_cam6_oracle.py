"""Oracle pins for the CAM6 Zhang-McFarlane port (``_zm_cam6.py``).

Every ported routine is compared, column by column, against the loop-for-loop
Python transcription of ``zm_conv.F90`` in ``_zm_cam6_oracle.py`` at rel
1e-10 (``zm_convr`` end to end, then ``zm_conv_evap``, ``momtran`` and
``convtran`` driven by the ported mass fluxes).  Plus: the column water and
static-energy budgets, momentum conservation, jit parity, finite gradients
through the state and the CAM6 namelist parameters, a non-vacuity check that
the pin fails when the oracle is run with a different parameter, and the
measured size of the saturation-curve departure (Goff-Gratch vs Tetens).
"""

from __future__ import annotations

import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio, saturation_vapor_pressure
from legoesm.atmosphere.physics.convection import _zm_cam6 as Z
from legoesm.atmosphere.physics.convection._zm_dilute import LWMAX
from legoesm.atmosphere.physics.convection.config import ZhangMcFarlaneConfig
from legoesm.atmosphere.physics.convection.mass_flux import compute_column_geometry
from legoesm.atmosphere.physics.convection.zhang_mcfarlane import (
    zhang_mcfarlane_convection,
)

from tests.atmosphere.hydrostatic.unit import _zm_cam6_oracle as O

CFG = ZhangMcFarlaneConfig()
# columns called without a land fraction: the explicit aquaplanet choice
CFG_AQUA = CFG._replace(land_fraction="none")
DT = 1800.0
NLEV = 30
KW = dict(tpert=0.0, capelmt=CFG.capelmt, tau=CFG.tau, num_cin=CFG.num_cin,
          dmpdz=CFG.dmpdz, tiedke_add=CFG.tiedke_add, lwmax=LWMAX, c0_lnd=CFG.c0_lnd,
          c0_ocn=CFG.c0_ocn, alfa=CFG.alfa, limcnv_p_pa=CFG.limcnv_p_pa,
          pbl_top_pa=CFG.pbl_top_pa)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _column(T_sfc, lapse, rh, p_s=1.0e5, p_top=2.0e3, nlev=NLEV):
    ph = jnp.linspace(p_top, p_s, nlev + 1)[None, :]
    pf = 0.5 * (ph[:, :-1] + ph[:, 1:])
    z = -8000.0 * jnp.log(pf / p_s)
    T = jnp.maximum(T_sfc - lapse * 1e-3 * z, 200.0)
    q = rh * saturation_mixing_ratio(T, pf)
    return T, q, pf, ph


@pytest.fixture(scope="module")
def cols():
    """Four columns: deep ocean, deep land, marginal, stable-dry (no convection)."""
    parts = [_column(300.0, 6.5, 0.85), _column(303.0, 7.0, 0.70),
             _column(298.0, 5.9, 0.80), _column(290.0, 4.0, 0.30)]
    T = jnp.concatenate([p[0] for p in parts])
    q = jnp.concatenate([p[1] for p in parts])
    pf = jnp.concatenate([p[2] for p in parts])
    ph = jnp.concatenate([p[3] for p in parts])
    land = jnp.array([0.0, 1.0, 0.4, 0.0])
    u = jnp.linspace(-5.0, 25.0, NLEV)[None, :] * jnp.array([1.0, 0.5, -1.0, 2.0])[:, None]
    v = jnp.linspace(3.0, -8.0, NLEV)[None, :] * jnp.array([1.0, 2.0, 0.3, 1.0])[:, None]
    dz, _, z = compute_column_geometry(T, pf, ph, q_v=q)
    zf = jnp.concatenate([jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1], jnp.zeros((4, 1))], axis=1)
    cld = jnp.clip(0.3 * jnp.sin(jnp.arange(NLEV) / 3.0)[None, :] + 0.2, 0.0, 0.6) \
        * jnp.ones((4, 1))
    return dict(T=T, q=q, pf=pf, ph=ph, z=z, zf=zf, land=land, u=u, v=v, cld=cld)


def _port(c, pblh=None, pref_edge=None, **over):
    return Z.zm_convr(c["T"], c["q"], c["pf"], c["ph"], c["z"], c["zf"], c["land"], DT,
                      pblh=pblh, pref_edge=pref_edge, **{**KW, **over})


def _oracle(c, i, pblh=None, pref_edge=None, **over):
    kw = dict(capelmt=CFG.capelmt, tau=CFG.tau, num_cin=CFG.num_cin, dmpdz=CFG.dmpdz,
              c0_lnd=CFG.c0_lnd, c0_ocn=CFG.c0_ocn, alfa=CFG.alfa, pbl_top_pa=CFG.pbl_top_pa,
              pref_edge=None if pref_edge is None else np.asarray(pref_edge))
    kw.update(over)
    n = lambda a: np.asarray(a[i])
    return O.zm_convr(n(c["T"]), n(c["q"]), n(c["pf"]), n(c["ph"]), n(c["z"]), n(c["zf"]),
                      float(c["land"][i]), 0.5 * DT,
                      None if pblh is None else float(pblh[i]), 0.0, **kw)


def _close(a, b, rtol=1e-10, atol=0.0, what=""):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    scale = max(np.max(np.abs(b)), 1e-300)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol + rtol * scale, err_msg=what)


# ---------------------------------------------------------------------------
# zm_convr end to end
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_pblh", [False, True])
@pytest.mark.parametrize("alfa", [CFG.alfa, 0.2], ids=["alfa_default", "alfa_0.2"])
def test_zm_convr_matches_fortran_transcription(cols, use_pblh, alfa):
    """``alfa=0.2`` activates the ``ratmjb`` downdraft cap on the land column
    (inactive at the default), the branch that used a stale cloud-base
    downdraft flux in ``totevp`` (codex round 1, #4)."""
    pblh = jnp.array([900.0, 1400.0, 700.0, 500.0]) if use_pblh else None
    P = _port(cols, pblh=pblh, alfa=alfa)
    n_deep = 0
    for i in range(4):
        Oi = _oracle(cols, i, pblh=pblh, alfa=alfa)
        assert int(P.limcnv[i]) == Oi["limcnv"] and int(P.msg[i]) == Oi["msg"]
        _close(P.cape[i], Oi["cape"], what=f"cape col{i}")
        assert bool(P.ideep[i]) == bool(Oi["ideep"])
        assert int(P.maxg[i]) == Oi["mx"], f"mx col{i}"
        if not Oi["ideep"]:
            for name in ("dqdt", "heat", "dlf", "rprd", "mu", "md", "du", "eu", "ed"):
                assert np.all(np.asarray(getattr(P, name)[i]) == 0.0), name
            assert float(P.mb[i]) == 0.0 and float(P.prec[i]) == 0.0
            continue
        n_deep += 1
        assert int(P.jt[i]) == Oi["jt"], f"jt col{i}"
        _close(P.dsubcld[i], Oi["dsubcld"], what="dsubcld")
        _close(P.mb[i], Oi["mb"], what=f"mb col{i}")
        for name in ("dqdt", "heat", "dlf", "rprd", "mu", "md", "du", "eu", "ed", "mc",
                     "cmeg", "ql", "pflx", "dp"):
            _close(getattr(P, name)[i], Oi[name], what=f"{name} col{i}")
        _close(P.prec[i], Oi["prec"], what="prec")
    assert n_deep >= 2, "the fixture must exercise the deep branch"
    assert not bool(P.ideep[3]), "the stable column must not convect"


def test_buoyan_dilute_indices_and_parcel_match(cols):
    P = _port(cols)
    K = jnp.arange(1, NLEV + 1)
    # re-run the port's buoyancy stage alone with the oracle's pblt/msg
    for i in range(4):
        Oi = _oracle(cols, i)
        msg = jnp.array([Oi["msg"]], jnp.int32)
        p = cols["pf"][i:i + 1] * 0.01
        pf = cols["ph"][i:i + 1] * 0.01
        in_pbl = (cols["pf"][i:i + 1] >= CFG.pbl_top_pa) & (K[None, :] >= msg + 1)
        pblt = jnp.min(jnp.where(in_pbl, K[None, :], NLEV), axis=1).astype(jnp.int32)
        bd = Z.buoyan_dilute(cols["q"][i:i + 1], cols["T"][i:i + 1], p, cols["z"][i:i + 1],
                             pf, pblt, jnp.zeros(1), msg, num_cin=CFG.num_cin,
                             dmpdz=CFG.dmpdz, tiedke_add=CFG.tiedke_add, lwmax=LWMAX)
        assert int(bd.mx[0]) == Oi["mx"]
        assert int(bd.lcl[0]) == Oi["lcl"], f"lcl col{i}"
        assert int(bd.lel[0]) == Oi["lel"], f"lel col{i}"
        _close(bd.tl[0], Oi["tl"], what="tl")
        _close(bd.cape[0], Oi["cape"], what="cape")
        _close(bd.tp[0], Oi["tp"], what="tp")
        _close(bd.qstp[0], Oi["qstp"], what="qstp")


# ---------------------------------------------------------------------------
# evaporation, momentum and tracer transport driven by the ported fluxes
# ---------------------------------------------------------------------------
def test_zm_conv_evap_matches_transcription(cols):
    P = _port(cols)
    T1 = cols["T"] + P.heat / constants.c_pd * DT
    q1 = cols["q"] + P.dqdt * DT
    pdel = cols["ph"][:, 1:] - cols["ph"][:, :-1]
    E = Z.zm_conv_evap(T1, cols["pf"], pdel, q1, P.rprd, cols["cld"], DT, P.prec, ke=CFG.ke)
    for i in range(4):
        n = lambda a: np.asarray(a[i])
        Oi = O.zm_conv_evap(n(T1), n(cols["pf"]), n(pdel), n(q1), n(P.rprd), n(cols["cld"]),
                            DT, float(P.prec[i]), CFG.ke)
        for name in ("tend_s", "tend_q", "ntprprd", "ntsnprd", "flxprec", "flxsnow"):
            _close(getattr(E, name)[i], Oi[name], what=f"{name} col{i}")
        _close(E.prec[i], Oi["prec"], what="prec")
        _close(E.snow[i], Oi["snow"], what="snow")
    assert float(jnp.abs(E.tend_q).max()) > 0.0, "evaporation must be active on a deep column"
    assert float(E.snow[0]) < float(E.prec[0])


def test_momtran_matches_transcription(cols):
    P = _port(cols)
    M = Z.momtran(cols["u"], cols["v"], P.mu, P.md, P.du, P.eu, P.ed, P.dp, P.jt, P.maxg,
                  P.msg, DT, momcu=CFG.momcu, momcd=CFG.momcd)
    for i in range(4):
        if not bool(P.ideep[i]):
            assert np.all(np.asarray(M.dudt[i]) == 0.0)
            continue
        n = lambda a: np.asarray(a[i])
        du_o, dv_o, se_o = O.momtran(n(cols["u"]), n(cols["v"]), n(P.mu), n(P.md), n(P.du),
                                     n(P.eu), n(P.ed), n(P.dp), int(P.jt[i]), int(P.maxg[i]),
                                     DT, CFG.momcu, CFG.momcd)
        _close(M.dudt[i], du_o, what=f"dudt col{i}")
        _close(M.dvdt[i], dv_o, what=f"dvdt col{i}")
        _close(M.seten[i], se_o, what=f"seten col{i}")
    assert float(jnp.abs(M.dudt).max()) > 0.0


def test_convtran_matches_transcription(cols):
    P = _port(cols)
    qc = 1.0e-4 * jnp.exp(-((jnp.arange(NLEV) - 12.0) / 5.0) ** 2)[None, :] * jnp.ones((4, 1))
    qc = qc.at[:, :4].set(0.0)   # a zero stretch exercises the geometric-mean floor
    C = Z.convtran(qc, P.mu, P.md, P.du, P.eu, P.ed, P.dp, P.jt, P.maxg, P.msg)
    for i in range(4):
        if not bool(P.ideep[i]):
            assert np.all(np.asarray(C[i]) == 0.0)
            continue
        n = lambda a: np.asarray(a[i])
        Ci = O.convtran(n(qc), n(P.mu), n(P.md), n(P.du), n(P.eu), n(P.ed), n(P.dp),
                        int(P.jt[i]), int(P.maxg[i]))
        _close(C[i], Ci, what=f"convtran col{i}")
    # mass conservation of the flux-form transport (hPa-weighted column sum)
    for i in range(4):
        assert abs(float((C[i] * P.dp[i]).sum())) < 1e-12 * float(jnp.abs(C[i] * P.dp[i]).max() + 1e-300)


# ---------------------------------------------------------------------------
# budgets
# ---------------------------------------------------------------------------
def test_zm_convr_static_energy_and_water_budgets(cols):
    P = _port(cols)
    dpp = cols["ph"][:, 1:] - cols["ph"][:, :-1]
    # cp*dsdt + L*dqdt is a pure flux divergence: column integral vanishes
    e = (dpp * (P.heat + constants.L_v * P.dqdt)).sum(1) / constants.g
    scale = (dpp * jnp.abs(P.heat)).sum(1) / constants.g + 1.0
    assert float(jnp.max(jnp.abs(e) / scale)) < 1e-10
    # prec (before evaporation) closes the vapour + detrained-liquid budget
    w = -(dpp * (P.dqdt + P.dlf)).sum(1) / constants.g
    _close(P.prec, jnp.maximum(w, 0.0), what="prec budget")
    # ... and equals the column integral of the cloud model's rain production
    # (an independent quantity: q1q2_pjr's fluxes must close against cldprp's
    # rprd, which is how the Fortran defines prec at zm_conv.F90:1310-1320)
    _close(P.prec, (dpp * P.rprd).sum(1) / constants.g, rtol=1e-9, what="prec vs rprd")


def test_scheme_water_closure_and_rain_field(cols):
    out, carry = zhang_mcfarlane_convection(
        cols["T"], cols["q"], cols["pf"], cols["ph"], cols["u"], cols["v"],
        jnp.zeros_like(cols["T"]), DT, CFG, land_frac=cols["land"], cld_frac=cols["cld"])
    dpp = cols["ph"][:, 1:] - cols["ph"][:, :-1]
    sink = -(dpp * (out.dq_v_dt + out.dq_c_conv_dt)).sum(1) / constants.g
    rain = (dpp * out.dq_r_conv_dt).sum(1) / constants.g
    _close(rain, sink, rtol=1e-9, what="rain vs vapour+cloud sink")
    assert float(rain[0]) > 0.0 and float(rain[3]) == 0.0
    assert float(jnp.abs(out.dq_c_conv_dt).max()) > 0.0
    assert bool((out.dq_c_conv_dt >= 0.0).all())
    # momentum: flux-form transport conserves the column momentum exactly
    for comp in (out.du_dt_conv, out.dv_dt_conv):
        m = (dpp * comp).sum(1)
        assert float(jnp.abs(m).max()) < 1e-12 * float((dpp * jnp.abs(comp)).sum(1).max() + 1e-300)
    assert carry.shape == cols["T"].shape and bool((carry[:, :-1] == 0.0).all())
    assert float(carry[0, -1]) > 0.0


# ---------------------------------------------------------------------------
# jit / AD
# ---------------------------------------------------------------------------
def _scheme_scalar(T, q, cfg, c):
    out, _ = zhang_mcfarlane_convection(
        T, q, c["pf"], c["ph"], c["u"], c["v"], jnp.zeros_like(T), DT, cfg,
        land_frac=c["land"], cld_frac=c["cld"])
    return (out.dT_dt ** 2).sum() + (out.dq_v_dt ** 2).sum() * 1e6 + (out.du_dt_conv ** 2).sum()


def test_jit_matches_eager(cols):
    f = lambda T, q: zhang_mcfarlane_convection(
        T, q, cols["pf"], cols["ph"], cols["u"], cols["v"], jnp.zeros_like(T), DT, CFG,
        land_frac=cols["land"], cld_frac=cols["cld"])[0]
    o_e = f(cols["T"], cols["q"])
    o_j = jax.jit(f)(cols["T"], cols["q"])
    # 1e-9: XLA fusion re-association in the level scans (float64, ~1e-11 seen)
    for name in ("dT_dt", "dq_v_dt", "dq_c_conv_dt", "dq_r_conv_dt", "du_dt_conv", "cape"):
        _close(getattr(o_j, name), getattr(o_e, name), rtol=1e-9, what=name)


def test_gradients_finite_and_nonzero(cols):
    gT, gq = jax.grad(_scheme_scalar, argnums=(0, 1))(cols["T"], cols["q"], CFG, cols)
    assert bool(jnp.isfinite(gT).all()) and bool(jnp.isfinite(gq).all())
    assert float(jnp.abs(gT[:2]).max()) > 0.0 and float(jnp.abs(gq[:2]).max()) > 0.0
    assert bool((gT[3] == 0.0).all()), "a non-convecting column has no sensitivity"
    for name in ("tau", "c0_ocn", "c0_lnd", "ke", "momcu", "momcd", "dmpdz", "alfa",
                 "tiedke_add", "capelmt"):
        g = jax.grad(lambda x: _scheme_scalar(
            cols["T"], cols["q"], CFG._replace(**{name: x}), cols))(
            jnp.asarray(getattr(CFG, name), jnp.float64))
        assert bool(jnp.isfinite(g)), name
        assert float(jnp.abs(g)) > 0.0, f"zero gradient through {name}"


# ---------------------------------------------------------------------------
# non-vacuity and measured departures
# ---------------------------------------------------------------------------
def test_pin_detects_a_parameter_change(cols):
    P = _port(cols)
    Oi = _oracle(cols, 0, alfa=0.2)           # downdraft fraction changed in the oracle only
    assert not np.allclose(np.asarray(P.heat[0]), Oi["heat"], rtol=1e-6, atol=0.0)
    Oi = _oracle(cols, 0, c0_ocn=0.045)
    assert not np.allclose(np.asarray(P.rprd[0]), Oi["rprd"], rtol=1e-6, atol=0.0)


def _goff_gratch_water_pa(t):
    tboil = 373.16
    return 10.0 ** (-7.90298 * (tboil / t - 1.0) + 5.02808 * math.log10(tboil / t)
                    - 1.3816e-7 * (10.0 ** (11.344 * (1.0 - t / tboil)) - 1.0)
                    + 8.1328e-3 * (10.0 ** (-3.49149 * (tboil / t - 1.0)) - 1.0)
                    + math.log10(1013.246)) * 100.0


def test_saturation_curve_departure_is_small():
    """CAM qsat_water is Goff-Gratch; the port uses the shared Tetens curve."""
    ts = np.linspace(230.0, 315.0, 60)
    rel = np.array([abs(float(saturation_vapor_pressure(jnp.float64(t))) / _goff_gratch_water_pa(t) - 1.0)
                    for t in ts])
    # measured: < 1 % above 250 K, ~4 % at 230 K (Tetens over-estimates cold-air e_s)
    assert rel[ts >= 250.0].max() < 0.01, rel[ts >= 250.0].max()
    assert rel.max() < 0.06, rel.max()


def test_qsat_is_cam_specific_humidity_not_mixing_ratio():
    """CAM ``wv_sat_svp_to_qsat``: ``ε·es/(p − (1−ε)·es)``, written out here
    independently of any thermo conversion.  A port feeding the mixing ratio
    ``ε·es/(p − es)`` is 0.08-2.3 % high over this range and fails."""
    ts = np.array([240.0, 280.0, 300.0])
    ps = np.array([300.0, 850.0, 1000.0])
    es = np.array([float(saturation_vapor_pressure(jnp.float64(t))) for t in ts])
    cam = constants.epsilon * es / (ps * 100.0 - (1.0 - constants.epsilon) * es)
    port = np.asarray(Z._qsat_hpa(jnp.asarray(ts), jnp.asarray(ps)))
    # 1e-6: the shared smooth cap shifts q by ~1e-10 kg/kg (measured <= 1.4e-7 relative)
    np.testing.assert_allclose(port, cam, rtol=1e-6, atol=0.0)


# ---------------------------------------------------------------------------
# codex / GLM round-1 findings, each pinned by a test that fails on revert
# ---------------------------------------------------------------------------
def test_non_convecting_column_with_zero_humidity_is_finite(cols):
    """A zero next to a finite humidity in a NON-convecting column must not
    reach the interface log-mean (the Fortran forms it on deep columns only):
    the unmasked version returned NaN moisture tendencies."""
    q = cols["q"].at[3, 15].set(0.0)
    P = Z.zm_convr(cols["T"], q, cols["pf"], cols["ph"], cols["z"], cols["zf"], cols["land"],
                   DT, pblh=None, **KW)
    assert not bool(P.ideep[3])
    for name in ("dqdt", "heat", "dlf", "rprd", "mb", "prec"):
        a = getattr(P, name)[3]
        assert bool(jnp.isfinite(a).all()), name
        assert float(jnp.abs(a).max()) == 0.0, name
    out, _ = zhang_mcfarlane_convection(cols["T"], q, cols["pf"], cols["ph"], cols["u"],
                                        cols["v"], jnp.zeros_like(q), DT, CFG_AQUA)
    assert bool(jnp.isfinite(out.dq_v_dt).all()) and bool(jnp.isfinite(out.dT_dt).all())


def test_float32_reverse_mode_gradient_is_finite(cols):
    """The downdraft recursions divide by ``min(md, -1e-20)``; the literal
    form's VJP is ``0 * 1e40`` = NaN in float32 (codex round 1, #2)."""
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    c = {k: f32(v) for k, v in cols.items()}

    def loss(T, q):
        out, _ = zhang_mcfarlane_convection(T, q, c["pf"], c["ph"], c["u"], c["v"],
                                            jnp.zeros_like(T), DT, CFG,
                                            land_frac=c["land"], cld_frac=c["cld"])
        assert out.dT_dt.dtype == jnp.float32, "the scheme must run in float32 here"
        return (out.dT_dt ** 2).sum() + (out.du_dt_conv ** 2).sum()

    gT, gq = jax.grad(loss, argnums=(0, 1))(c["T"], c["q"])
    assert gT.dtype == jnp.float32
    assert bool(jnp.isfinite(gT).all()) and bool(jnp.isfinite(gq).all())
    assert float(jnp.abs(gT[:2]).max()) > 0.0


def test_limcnv_is_fixed_from_the_reference_interfaces(cols):
    """CAM caps deep convection at the interface where the REFERENCE pressures
    cross 40 hPa, once at init (zm_conv_intr.F90:334-345); a column whose own
    interfaces cross elsewhere must still get the reference index (codex
    round 1, #5).  The pin transcribes the same init."""
    T, q, pf, ph = _column(296.0, 6.0, 0.75, p_s=7.0e4, p_top=1.0e3)
    ref = jnp.linspace(1.0e3, 1.0e5, NLEV + 1)     # the coordinate at p_ref
    dz, _, z = compute_column_geometry(T, pf, ph, q_v=q)
    zf = jnp.concatenate([jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1], jnp.zeros((1, 1))], axis=1)
    c = dict(T=T, q=q, pf=pf, ph=ph, z=z, zf=zf, land=jnp.zeros((1,)))
    local, fixed = _port(c), _port(c, pref_edge=ref)
    k_ref = int(jnp.sum(ref < CFG.limcnv_p_pa))
    assert 1 <= k_ref <= NLEV and float(ref[k_ref - 1]) < CFG.limcnv_p_pa <= float(ref[k_ref])
    assert int(fixed.limcnv[0]) == k_ref
    assert int(local.limcnv[0]) != k_ref, "fixture must make the local and reference caps differ"
    assert int(fixed.limcnv[0]) == _oracle(c, 0, pref_edge=ref)["limcnv"]
    assert int(local.limcnv[0]) == _oracle(c, 0)["limcnv"]


def test_cldprp_saturation_guard_matches_fortran():
    """``qst = 1`` where ``p - es <= 0`` (zm_conv.F90:2937-2941), the one
    guarded qsat in the oracle (GLM round 1, #1)."""
    t = jnp.array([[300.0, 330.0, 350.0, 280.0]])
    p_hpa = jnp.array([[1000.0, 150.0, 300.0, 5.0]])
    got = Z._qst_cldprp(t, p_hpa)
    want = np.array([[O.qsat_hpa(300.0, 1000.0), 1.0, 1.0, 1.0]])
    fires = np.asarray(p_hpa * 100.0 - saturation_vapor_pressure(t) <= 0.0)
    assert fires.tolist() == [[False, True, True, True]], "fixture must fire the guard"
    _close(got, want, what="qst guard")


def test_closure_eb_nonpositive_is_visible_only_in_deep_columns():
    """A launch-level vapour pressure ``eb <= 0`` is ``log(0)`` in the Fortran
    (GLM round 1, #2): NaN in a deep column, 0 in a masked one."""
    nlev = 6
    one = jnp.ones((2, nlev))
    zero = jnp.zeros((2, nlev))
    q = one * 0.01
    q = q.at[:, nlev - 1].set(0.0)                  # launch level dry
    p = jnp.linspace(100.0, 1000.0, nlev)[None, :] * jnp.ones((2, 1))
    t = 280.0 * one
    zf = jnp.concatenate([jnp.linspace(15000.0, 0.0, nlev + 1)[None, :]] * 2)
    z = 0.5 * (zf[:, :-1] + zf[:, 1:])
    ivec = lambda v: jnp.full((2,), v, jnp.int32)
    mb = Z.closure(q, t, p, z, t + constants.g / constants.c_pd * z, t, q, q, t, one, one, one, -one, q, t,
                   q, t, 100.0 * one, q, zf, zero, jnp.array([500.0, 500.0]),
                   jnp.array([500.0, 500.0]), jnp.array([275.0, 275.0]), ivec(4), ivec(2),
                   ivec(2), ivec(nlev), ivec(0), capelmt=CFG.capelmt, tau=CFG.tau,
                   deep=jnp.array([True, False]))
    assert bool(jnp.isnan(mb[0])) and bool(jnp.isfinite(mb[1]))


def test_defaults_match_cam6_hardcodes_and_namelist():
    """The values the Fortran hard-codes (alfa 3483, dmpdz 4328, lwmax 4331,
    tau/capelmt/tiedke_add in zm_convi) and the CAM6 f09 namelist; both
    sides of the pin read them from this config, so they need a canary."""
    cfg = ZhangMcFarlaneConfig()
    assert (cfg.alfa, cfg.dmpdz, LWMAX) == (0.1, -1.0e-3, 1.0e-3)
    assert (cfg.tau, cfg.capelmt, cfg.tiedke_add, cfg.num_cin) == (3600.0, 70.0, 0.5, 1)
    assert (cfg.c0_lnd, cfg.c0_ocn, cfg.ke) == (0.0075, 0.03, 5.0e-6)
    assert (cfg.momcu, cfg.momcd) == (0.7, 0.7)
    assert cfg.limcnv_p_pa == 4000.0 and cfg.parcel_tpert == 0.0
    from legoesm.atmosphere.physics.convection.config import __param_spec__
    assert __param_spec__["ZhangMcFarlaneConfig"]["params"]["ke"]["units"] == "s^-1 (kg/m^2/s)^-1/2"


def test_deepcu_inputs_match_transcription_and_feed_the_consumer(cols):
    """``mass_flux_up`` / ``icwmr`` are CAM's CMFMC / ICWMRDP (the pbuf fields
    clubb_intr's deepcu reads): mc scattered to the top face of each layer,
    bottom face 0, kg/m^2/s (zm_conv_intr.F90:661); in-cloud ql, zero outside
    convecting columns.  Pinned against the loop transcription and shown to
    yield a non-zero CAM6 deep-convective cloud fraction."""
    from legoesm.atmosphere.physics.clouds.cloud_fraction import cam6_deep_convective_fraction
    from legoesm.atmosphere.physics.clouds.config import CloudConfig

    out, _ = zhang_mcfarlane_convection(
        cols["T"], cols["q"], cols["pf"], cols["ph"], cols["u"], cols["v"],
        jnp.zeros_like(cols["T"]), DT, CFG, land_frac=cols["land"], cld_frac=cols["cld"])
    assert out.mass_flux_up.shape == (4, NLEV + 1) and out.icwmr.shape == (4, NLEV)
    assert bool((out.mass_flux_up[:, -1] == 0.0).all())
    assert bool(jnp.isfinite(out.mass_flux_up).all()) and bool(jnp.isfinite(out.icwmr).all())
    n_deep = 0
    for i in range(4):
        Oi = _oracle(cols, i)
        if not Oi["ideep"]:
            assert float(jnp.abs(out.mass_flux_up[i]).max()) == 0.0
            assert float(jnp.abs(out.icwmr[i]).max()) == 0.0
            continue
        n_deep += 1
        _close(out.mass_flux_up[i, :NLEV], Oi["mc"] * 100.0 / constants.g, what=f"cmfmc col{i}")
        _close(out.icwmr[i], Oi["ql"], what=f"icwmrdp col{i}")
        # cloud base (maxg, 1-based) carries a positive net mass flux
        assert float(out.mass_flux_up[i, Oi["mx"] - 1]) > 0.0
    assert n_deep >= 2
    deepcu = cam6_deep_convective_fraction(out.mass_flux_up, out.icwmr,
                                           CloudConfig(scheme="cam6_clubb"))
    assert float(deepcu[0].max()) > 0.0 and float(deepcu[3].max()) == 0.0
    assert bool(jnp.isfinite(deepcu).all())


# ---------------------------------------------------------------------------
# real AMIP columns with an exact zero in the humidity profile (2026-09-22 blowup)
# ---------------------------------------------------------------------------
_REAL_FIX = pathlib.Path(__file__).parent / "fixtures" / "zm_cam6_real_zero_humidity_columns.npz"


@pytest.mark.parametrize("tag", ["32", "36"])
def test_real_columns_with_zero_humidity_are_finite_and_pinned(tag):
    """The CAM6 60-day arm went NaN in every field within 38 steps: real MPAS
    columns carry an exact q_v = 0 layer (the host's tracer clip), on which
    the interface log-mean log(q(k-1)/q(k)) is NaN -- in the Fortran too.
    CAM never sees a zero because physics_update floors Q at qmin = 1e-12
    (qneg3) before and after zm_convr; the wrapper applies the same floor.
    Raw columns through the wrapper must be finite AND match the transcription
    fed the floored humidity (the floor is the only thing between them)."""
    d = np.load(_REAL_FIX)
    T, q, pf, ph = (jnp.asarray(d[k + tag]) for k in ("T", "q", "p_full", "p_half"))
    pref = jnp.asarray(d["pref_edge" + tag])
    ncol, nlev = T.shape
    assert bool((q == 0.0).any()), "fixture must contain an exact zero"
    u = jnp.zeros_like(T)
    out, carry = zhang_mcfarlane_convection(T, q, pf, ph, u, u, jnp.zeros_like(T), 112.5, CFG_AQUA,
                                            pref_edge=pref)
    for name in ("dT_dt", "dq_v_dt", "dq_c_conv_dt", "dq_r_conv_dt", "cape", "du_dt_conv",
                 "mass_flux_up", "icwmr"):
        assert bool(jnp.isfinite(getattr(out, name)).all()), name
    assert bool(jnp.isfinite(carry).all())
    assert bool(out.convective_mask.any()), "fixture columns must convect (they did in the run)"
    # the tendency is a RATE: a 1800 s physics step (CAM's dtime, the cadence
    # arm) returns the same dT_dt as a 112.5 s one except where the mumax CFL
    # cap on mb binds (then smaller), never larger (2026-09-22 cadence bisect)
    out_h, _ = zhang_mcfarlane_convection(T, q, pf, ph, u, u, jnp.zeros_like(T), 1800.0, CFG_AQUA,
                                          pref_edge=pref)
    assert float(jnp.abs(out_h.dT_dt).max()) <= 1.05 * float(jnp.abs(out.dT_dt).max())
    assert float(jnp.abs(out_h.dq_v_dt).max()) <= 1.05 * float(jnp.abs(out.dq_v_dt).max())
    # the run's second physics call: the carry from the first call plus a
    # carried cloud fraction (the wrapper reads neither into the kernels)
    out2, carry2 = zhang_mcfarlane_convection(
        T, q, pf, ph, u, u, carry, 112.5, CFG_AQUA, cld_frac=jnp.full(T.shape, 0.5), pref_edge=pref)
    assert bool(jnp.isfinite(out2.dT_dt).all()) and bool(jnp.isfinite(out2.dq_v_dt).all())
    assert bool(jnp.isfinite(carry2).all())
    dz, _, z = compute_column_geometry(T, pf, ph, q_v=jnp.maximum(q, Z.Q_MIN_VAPOR))
    zf = jnp.concatenate([jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1], jnp.zeros((ncol, 1))], axis=1)
    c = dict(T=T, q=jnp.maximum(q, Z.Q_MIN_VAPOR), pf=pf, ph=ph, z=z, zf=zf, land=jnp.zeros((ncol,)))
    P = _port(c, pref_edge=pref)
    for i in range(ncol):
        Oi = _oracle(c, i, pref_edge=pref)
        assert bool(P.ideep[i]) == bool(Oi["ideep"])
        for name in ("dqdt", "heat", "dlf", "rprd", "mu", "md"):
            _close(getattr(P, name)[i], Oi[name], what=f"{name} col{i}")
