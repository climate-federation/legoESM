"""Oracle match: WRF v4.8.0 Kain-Fritsch (``_kf_eta.kf_eta_para`` and the
``kain_fritsch_convection`` wrapper).

Fixture ``oracle_fixtures/kain_fritsch.npz``: the 15 audit columns (3 real L32
MPAS AMIP columns, 12 constructed soundings) and the outputs of RUNNING WRF
v4.8.0 06d4240ae9 ``phys/module_cu_kfeta.F`` ``KF_eta_PARA`` (gfortran 8.5,
fp64), in 12 arms that vary the constant set (legoESM / WRF), the winds, the
grid-scale w, the time step, ``kf_trigger`` (3 / 1) and the hydrometeor
feedback branch (warm_rain / ice / mixed phase), plus 117 columns of a seeded
randomised sweep (perturbed soundings, winds, w, dt 60-900 s, DX 3-60 km) picked
to run the branches the arms do not reach.  Generator:
``scripts/validate/oracle/kain_fritsch/{build.sh,make_fixture.py}``.  The
oracle was fed exactly what the wrapper feeds the port (see make_fixture.py).

Error metric (as tests/atmosphere/test_tiedtke_wrf_oracle.py): per field,
max|ours - oracle| / max|oracle| over all columns and levels.
"""

from __future__ import annotations

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.convection._kf_eta import KFConstants, kf_eta_para
from legoesm.atmosphere.physics.convection.config import KainFritschConfig
from legoesm.atmosphere.physics.convection.kain_fritsch import (
    _lego_constants,
    kain_fritsch_convection,
)

from legoesm import constants

_FIX = Path(__file__).parent / "oracle_fixtures" / "kain_fritsch.npz"
_TOL = 1.0e-10  # user-approved oracle-match tolerance (max relative error, fp64)
_OUT = ("dtdt", "dqdt", "dqcdt", "dqrdt", "dqidt", "dqsdt")
_DIAG_SCALARS = ("ltop", "let", "lc", "kpbl", "lcl", "lfs", "ldb", "ncount", "nstep", "abe",
                 "ainc", "timec", "tder", "trppt", "pptflx", "cpr", "fbfrc", "frc2", "vmflcl",
                 "wlcl", "rad", "chmin", "cldhgt", "fabe")
_DIAG_ARRAYS = ("umf", "uer", "udr", "dmf", "der", "ddr", "tu", "qu", "qliq", "qice", "detlq",
                "detic", "pptliq", "pptice", "tg", "qg", "eqfrc", "ems")
_DIAG_ARMS = ("L_nat_w05", "L_u20_w05")


@pytest.fixture(scope="module")
def fix():
    return dict(np.load(_FIX))


def _arms():
    return [str(a) for a in np.load(_FIX)["arm_names"]]


def _consts(fix, cs):
    v = dict(zip([str(k) for k in fix["const_order"]], fix[f"consts_{cs}"]))
    return KFConstants(g=v["G"], cp=v["CP"], r=v["R"], xlv0=v["XLV0"], xlv1=v["XLV1"],
                       svp1=v["SVP1"], svp2=v["SVP2"], svp3=v["SVP3"], svpt0=v["SVPT0"])


def _bu(a, dtype=jnp.float64):
    return jnp.asarray(np.asarray(a)[:, ::-1], dtype)


def _meta(fix, arm):
    w, dt, trig, warm, fqi, fqs, u20 = fix[f"{arm}_meta"]
    return dict(w=float(w), dt=float(dt), trigger=int(trig), warm_rain=bool(warm),
                f_qi=bool(fqi), f_qs=bool(fqs), u20=bool(u20), cs=str(fix[f"{arm}_consts"][0]))


def _winds_td(fix, m):
    T = fix["T"]
    if m["u20"]:
        return np.full_like(T, 20.0), np.zeros_like(T)
    return fix["u_nat"], fix["v_nat"]


def _run_core(fix, arm, dtype=jnp.float64):
    m = _meta(fix, arm)
    u, v = _winds_td(fix, m)
    T = _bu(fix["T"], dtype)
    return kf_eta_para(
        _bu(u, dtype), _bu(v, dtype), T, _bu(fix["q"], dtype), _bu(fix["p_full"], dtype),
        jnp.asarray(fix["dz_bu"], dtype), jnp.full_like(T, m["w"]),
        jnp.asarray(fix[f"rhoe_{m['cs']}_bu"], dtype), dt=m["dt"], dx=float(fix["dx"]),
        constants=_consts(fix, m["cs"]), trigger=m["trigger"], warm_rain=m["warm_rain"],
        f_qi=m["f_qi"], f_qs=m["f_qs"], cudt=float(fix["cudt"]))


def _rel(ours, ref):
    ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    den = np.max(np.abs(ref))
    err = np.max(np.abs(ours - ref))
    return err / den if den > 0 else err


@pytest.mark.parametrize("arm", _arms())
def test_kf_eta_para_matches_wrf(fix, arm):
    o = _run_core(fix, arm)
    np.testing.assert_array_equal(np.asarray(o["fired"]).astype(int), fix[f"{arm}_fired"].astype(int))
    np.testing.assert_array_equal(np.asarray(o["ishall"]), fix[f"{arm}_ishall"].astype(int))
    np.testing.assert_array_equal(np.asarray(o["cutop"]), fix[f"{arm}_cutop"].astype(int))
    np.testing.assert_array_equal(np.asarray(o["cubot"]), fix[f"{arm}_cubot"].astype(int))
    for k in _OUT:
        assert _rel(o[k], fix[f"{arm}_{k}"]) <= _TOL, (arm, k, _rel(o[k], fix[f"{arm}_{k}"]))
    assert _rel(o["pratec"], fix[f"{arm}_pratec"]) <= _TOL
    f = fix[f"{arm}_fired"] == 1
    assert _rel(np.asarray(o["nca"])[f], fix[f"{arm}_nca"][f]) <= _TOL
    assert fix[f"{arm}_nfatal"].sum() == 0


@pytest.mark.parametrize("arm", _DIAG_ARMS)
def test_kf_eta_internal_state_matches_wrf(fix, arm):
    """End-of-routine UMF/UER/UDR/DMF/... and the closure scalars, levels 1..LTOP."""
    o = _run_core(fix, arm)
    f = fix[f"{arm}_fired"] == 1
    for k in _DIAG_SCALARS:
        assert _rel(np.asarray(o[k])[f], fix[f"{arm}_{k}"][f]) <= _TOL, k
    for k in _DIAG_ARRAYS:
        for i in np.where(f)[0]:
            lt = int(fix[f"{arm}_ltop"][i])
            assert _rel(np.asarray(o[k])[i, :lt], fix[f"{arm}_{k}"][i, :lt]) <= _TOL, (k, i)


def test_sweep_columns_match_wrf(fix):
    """Randomised-sweep columns (make_fixture.SWEEP_PICK): perturbed soundings,
    winds, w, dt in 60-900 s and DX in 3-60 km over six trigger / constant /
    hydrometeor combos, chosen so the closure iteration, moisture borrowing,
    the grid-area mass cap and the other branches the arms never reach run.
    Metric as the arms, per combo over its picked columns."""
    combos = fix["sweep_combos"]
    assert fix["sweep_combo"].size >= 1
    for ci in np.unique(fix["sweep_combo"]):
        cs, trig, warm, fqi, fqs = combos[ci]
        sel = np.where(fix["sweep_combo"] == ci)[0]
        kc = _consts(fix, cs)

        def one(u, v, t, q, p, dz, w, rh, dt, dx):
            o = kf_eta_para(u[None], v[None], t[None], q[None], p[None], dz[None], w[None],
                            rh[None], dt=dt, dx=dx, constants=kc, trigger=int(trig),
                            warm_rain=warm == "1", f_qi=fqi == "1", f_qs=fqs == "1",
                            cudt=float(fix["cudt"]))
            return {k: o[k][0] for k in _OUT + ("pratec", "fired", "ishall", "nca")}

        args = [jnp.asarray(fix[f"sweep_{k}"][sel]) for k in
                ("u", "v", "T", "q", "p", "dz", "w", "rhoe", "dt", "dx")]
        o = jax.jit(jax.vmap(one))(*args)
        np.testing.assert_array_equal(np.asarray(o["fired"]).astype(int),
                                      fix["sweep_fired"][sel].astype(int))
        np.testing.assert_array_equal(np.asarray(o["ishall"]), fix["sweep_ishall"][sel].astype(int))
        for k in _OUT + ("pratec",):
            assert _rel(o[k], fix[f"sweep_{k}"][sel]) <= _TOL, (ci, k, _rel(o[k], fix[f"sweep_{k}"][sel]))
        f = fix["sweep_fired"][sel] == 1
        assert _rel(np.asarray(o["nca"])[f], fix["sweep_nca"][sel][f]) <= _TOL, ci
    assert (fix["sweep_fired"] == 1).sum() >= 90 and (fix["sweep_ishall"] == 1).sum() >= 10


def test_lookup_beyond_table_is_clamped_edge_continuous():
    """Where WRF reads its KF_LUTAB tables out of bounds (undefined behaviour;
    the oracle segfaults on some such columns) the port clamps the table
    coordinate: finite, constant beyond the edge and continuous at it."""
    from legoesm.atmosphere.physics.convection import _kf_eta as K
    kc = _lego_constants()
    lut = tuple(jnp.asarray(x) for x in K.kf_lookup_tables(kc.svp1, kc.svp2, kc.svp3, kc.svpt0))
    plutop, rdpr, rdthk = (float(x) for x in lut[4:7])
    p_edge = plutop + (K._KFNP - 1) / rdpr            # top of the pressure axis
    for p in (p_edge, p_edge + 5.0e3):
        th_edge = float(lut[2][-1]) + (K._KFNT - 1) / rdthk
        a = np.asarray(K._tab_interp(lut, p, th_edge))
        b = np.asarray(K._tab_interp(lut, p, th_edge + 50.0))
        np.testing.assert_allclose(a, b, rtol=1e-12)
        assert np.all(np.isfinite(b))
    a = np.asarray(K._tab_interp(lut, p_edge - 1.0e-6, 330.0))
    b = np.asarray(K._tab_interp(lut, p_edge + 5.0e3, 330.0))
    np.testing.assert_allclose(a, b, rtol=1e-8)
    top = K._ALU_ASTRT + (K._ALU_N - 1) * K._ALU_AINC
    np.testing.assert_allclose(K._tlog_lookup(lut[3], top), K._tlog_lookup(lut[3], 10.0 * top),
                               rtol=1e-12)


def test_fixture_exercises_the_scheme(fix):
    """The arms cover deep, shallow (USL re-pass with FBFRC = 1), elevated
    source layers, downdraughts, glaciation, rain feedback and no convection."""
    a = "L_nat_w05"
    ish = fix[f"{a}_ishall"]
    assert (ish == 0).sum() >= 5 and (ish == 1).sum() >= 1 and (ish == 2).sum() >= 3
    assert (fix["L_nat_w0_ishall"] == 1).sum() >= 3
    f = fix[f"{a}_fired"] == 1
    assert np.any(fix[f"{a}_lc"][f] > 1)                   # elevated USL
    assert np.any(fix[f"{a}_tder"][f] > 0)                 # downdraught
    assert np.any(fix[f"{a}_pptice"][f] > 0)               # glaciation
    assert np.any(fix[f"{a}_fbfrc"][f] == 1)               # shallow re-pass
    assert np.max(np.abs(fix[f"{a}_dqrdt"])) > 0
    assert np.max(np.abs(fix["L_nat_w05_mixi_dqidt"])) > 0
    assert np.max(np.abs(fix["L_nat_w05_mixs_dqsdt"])) > 0


def _wrapper_inputs(fix, arm, dtype=jnp.float64):
    m = _meta(fix, arm)
    u, v = _winds_td(fix, m)
    T = jnp.asarray(fix["T"], dtype)
    return m, dict(T=T, q_v=jnp.asarray(fix["q"], dtype), p_full=jnp.asarray(fix["p_full"], dtype),
                   p_half=jnp.asarray(fix["p_half"], dtype), w_grid=jnp.full_like(T, m["w"]),
                   conv_prog_profile=jnp.zeros_like(T), dt=m["dt"], u=jnp.asarray(u, dtype),
                   v=jnp.asarray(v, dtype))


def test_wrapper_constants_are_the_fixture_arm(fix):
    assert tuple(_lego_constants()) == tuple(_consts(fix, "L"))


@pytest.mark.parametrize("arm,cfg", [
    ("L_nat_w05", {}), ("L_u20_w05", {}), ("L_nat_w0", {}), ("L_nat_w05_dt360", {}),
    ("L_nat_w05_t1", {"enable_rh_trigger_perturb": False}),
])
def test_wrapper_matches_wrf(fix, arm, cfg):
    """kain_fritsch_convection end to end (flip, geometry, rhoe, constants) vs the
    oracle run on the same columns."""
    m, x = _wrapper_inputs(fix, arm)
    out, carry = kain_fritsch_convection(**x, config=KainFritschConfig()._replace(**cfg))
    td = lambda a: np.asarray(a)[:, ::-1]  # noqa: E731
    assert _rel(out.dT_dt, td(fix[f"{arm}_dtdt"])) <= _TOL
    assert _rel(out.dq_v_dt, td(fix[f"{arm}_dqdt"])) <= _TOL
    assert _rel(out.dq_c_conv_dt, td(fix[f"{arm}_dqcdt"])) <= _TOL
    np.testing.assert_array_equal(np.asarray(out.convective_mask), fix[f"{arm}_fired"])
    dp = fix["p_half"][:, 1:] - fix["p_half"][:, :-1]
    # convective precipitation leaves through dq_r: its column integral is PRATEC
    # plus the oracle's resolved-rain tendency
    p_col = np.sum(np.asarray(out.dq_r_conv_dt) * dp, axis=1) / constants.g
    want = fix[f"{arm}_pratec"] + np.sum(td(fix[f"{arm}_dqrdt"]) * dp, axis=1) / constants.g
    assert _rel(p_col, want) <= _TOL
    if arm in _DIAG_ARMS:
        f = fix[f"{arm}_fired"] == 1
        prod = (fix[f"{arm}_pptliq"] + fix[f"{arm}_pptice"]) * (
            fix[f"{arm}_ainc"] * (1.0 - fix[f"{arm}_fbfrc"]) * fix[f"{arm}_frc2"])[:, None]
        ems = fix[f"{arm}_ems"]
        rain = np.where(f[:, None] & (ems > 0), prod / np.where(ems > 0, ems, 1.0), 0.0)
        rain = np.nan_to_num(rain) + fix[f"{arm}_dqrdt"]
        assert _rel(out.dq_r_conv_dt, td(rain)) <= _TOL
        assert _rel(np.asarray(out.cape)[f], fix[f"{arm}_abe"][f]) <= _TOL


@pytest.mark.parametrize("arm", ["L_nat_w05", "L_nat_w0"])
def test_column_water_closes_as_the_oracle(fix, arm):
    """Vapour + cloud + precipitation sources integrate to zero in each column
    (the oracle's own moisture budget); measured residual is round-off."""
    m, x = _wrapper_inputs(fix, arm)
    out, _ = kain_fritsch_convection(**x)
    dp = fix["p_half"][:, 1:] - fix["p_half"][:, :-1]
    tot = (np.asarray(out.dq_v_dt) + np.asarray(out.dq_c_conv_dt)
           + np.asarray(out.dq_r_conv_dt)) * dp / constants.g
    scale = np.sum(np.abs(np.asarray(out.dq_v_dt)) * dp / constants.g, axis=1)
    f = fix[f"{arm}_fired"] == 1
    assert np.all(np.abs(tot.sum(axis=1))[f] <= 1.0e-9 * scale[f])


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_wrapper_gradients_finite(fix, dtype):
    m, x = _wrapper_inputs(fix, "L_nat_w05", dtype)

    def loss(T, q, u, v, w):
        out, carry = kain_fritsch_convection(**{**x, "T": T, "q_v": q, "u": u, "v": v,
                                                "w_grid": w})
        return (jnp.sum(out.dT_dt) * 1e2 + jnp.sum(out.dq_v_dt) * 1e6
                + jnp.sum(out.dq_r_conv_dt) * 1e6 + jnp.sum(carry))

    g = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))(x["T"], x["q_v"], x["u"], x["v"],
                                                         x["w_grid"])
    for gg in g:
        assert bool(jnp.all(jnp.isfinite(gg)))
    assert float(jnp.max(jnp.abs(g[0]))) > 0.0


def test_eager_matches_jit(fix):
    m, x = _wrapper_inputs(fix, "L_nat_w05")
    e, _ = kain_fritsch_convection(**x)
    j, _ = jax.jit(lambda kw: kain_fritsch_convection(**kw, dt=m["dt"]))(
        {k: v for k, v in x.items() if k != "dt"})
    for a, b in ((e.dT_dt, j.dT_dt), (e.dq_v_dt, j.dq_v_dt), (e.dq_r_conv_dt, j.dq_r_conv_dt)):
        assert _rel(b, a) <= _TOL


def test_surrogate_fields_and_missing_winds_raise(fix):
    m, x = _wrapper_inputs(fix, "L_nat_w05")
    with pytest.raises(ValueError, match="M_b_max"):
        kain_fritsch_convection(**x, config=KainFritschConfig(M_b_max=0.1))
    with pytest.raises(ValueError, match="winds"):
        kain_fritsch_convection(**{**x, "u": None})
    T = jnp.asarray(fix["T"])
    with pytest.raises(ValueError, match="trigger"):
        kf_eta_para(T, T, T, T, T, T, T, T, dt=300.0, dx=25e3,
                    constants=_lego_constants(), trigger=2)
