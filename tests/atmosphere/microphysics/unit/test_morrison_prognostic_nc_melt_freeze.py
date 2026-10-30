"""MorrisonConfig.prognostic_droplet_number, stage S10.4: MG2 instant ice melt
into droplets and homogeneous droplet freezing (micro_mg2_0.F90:2615-2685),
with the smooth switch-on (user decision 2026-10-20), against the RUNNING MG2.

Fixture ``oracle_fixtures/mg2_prognostic_nc_melt.npz``: ``make_fixture.py
--prognostic-nc-melt`` through ``mg2_tap`` (build.sh: probe arrays tap_pre just
before the ice melt, M:2616, and tap_post after homogeneous freezing, before
M:2687; assignment-only lines).  Cold columns carry MG2 ice processes this port
has not reached, so the comparison is function-level: the helper fed the tapped
post-sedimentation state must reproduce the tapped change.  With a vanishing
ramp width the helper is exactly CAM's hard form.  x64.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm import constants as C  # noqa: E402
from legoesm.atmosphere.physics.microphysics import _warm_rain as W  # noqa: E402
from legoesm.atmosphere.physics.microphysics import morrison as M  # noqa: E402
from legoesm.core.smooth import smoothstep_quintic  # noqa: E402

from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import TOL, _rel  # noqa: E402
from tests.atmosphere.microphysics.unit.test_morrison_prognostic_nc import _cfg  # noqa: E402
from tests.atmosphere.microphysics.unit import test_morrison_prognostic_nc_act as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/mg2_prognostic_nc_melt.npz"
HARD = 1.0e-12     # ramp width [K] below every fixture cell's distance to its threshold


def _fx():
    return dict(np.load(FIXTURE))


def _tapped(fx):
    dt = float(fx["dt"])
    raw = dict(q_c=fx["col_q_c"] + fx["ora_pre_qctend"] * dt,
               N_c=fx["col_nc_kg"] + fx["ora_pre_nctend"] * dt,
               q_i=fx["col_q_i"] + fx["ora_pre_qitend"] * dt,
               N_i=fx["col_ni_kg"] + fx["ora_pre_nitend"] * dt,
               T=fx["ora_pre_Tp"])
    end = dict(q_c=fx["col_q_c"] + fx["ora_post_qctend"] * dt,
               N_c=fx["col_nc_kg"] + fx["ora_post_nctend"] * dt,
               q_i=fx["col_q_i"] + fx["ora_post_qitend"] * dt,
               N_i=fx["col_ni_kg"] + fx["ora_post_nitend"] * dt,
               T=fx["col_T"] + fx["ora_post_tlat"] / C.c_pd * dt)
    return raw, end


def _helper(raw, melt_w=HARD, frz_w=HARD, T_h=233.15, cap=W.MG2_CAP_SHARPNESS):
    return W.mg2_instant_ice_melt_homog_freeze(
        *(jnp.asarray(raw[k]) for k in ("q_c", "N_c", "q_i", "N_i", "T")), T_h, melt_w, frz_w,
        cap, presence_top=W.MG2_QSMALL)


def _activity(fx):
    raw, end = _tapped(fx)
    names = list(fx["col_names"])
    m, h = names.index("meltice"), names.index("homog")
    dqi = end["q_i"] - raw["q_i"]
    melt = (raw["T"][m] > C.T_freeze) & (fx["ora_pre_dumi"][m] > 0)
    frz = (raw["T"][h] < 233.15) & (fx["ora_pre_dumc"][h] > 0)
    fm = -dqi[m] / np.where(fx["ora_pre_dumi"][m] > 0, fx["ora_pre_dumi"][m], 1.0)
    fh = dqi[h] / np.where(fx["ora_pre_dumc"][h] > 0, fx["ora_pre_dumc"][h], 1.0)
    return m, h, melt, frz, fm, fh


def test_fixture_inputs_and_coverage():
    fx = _fx()
    assert np.allclose(fx["col_rho"], fx["col_p_full"] / (C.R_d * fx["col_T"]), rtol=1e-14)
    assert not any(np.isnan(v).any() for v in fx.values() if v.dtype.kind == "f")
    assert (fx["col_q_r"] == 0).all()           # no rain/snow: the instant snow melt and
    m, h, melt, frz, fm, fh = _activity(fx)     # rain freezing cannot touch the pools
    assert (melt & (fm > 0.999)).sum() >= 3 and (melt & (fm > 0) & (fm < 0.99)).any()
    assert (frz & (fh > 0.999)).sum() >= 3 and (frz & (fh > 0) & (fh < 0.99)).any()
    # the ice pool the melt reads is the clipped raw state (no rain froze to ice there)
    raw, _ = _tapped(fx)
    assert np.allclose(fx["ora_pre_dumi"][m], np.maximum(raw["q_i"][m], 0.0), rtol=1e-13, atol=0)


def test_hard_limit_matches_mg2():
    fx = _fx()
    raw, end = _tapped(fx)
    # vanishing ramp width and a 1e12 cap sharpness: CAM's hard form
    q_c, N_c, q_i, N_i, T, melt, frz, chg = _helper(raw, cap=1.0e12)
    got = dict(q_c=q_c, N_c=N_c, q_i=q_i, N_i=N_i, T=T)
    errs = {}
    for k, scale in (("q_c", 1e-3), ("N_c", 1e8), ("q_i", 1e-3), ("N_i", 1e8), ("T", 300.0)):
        d = np.abs(np.asarray(got[k]) - end[k])
        errs[k] = float(d.max() / scale)
    print("S10.4 helper vs MG2 (abs err / scale)", errs, "changed", int(np.asarray(chg).sum()))
    assert max(errs.values()) <= TOL, errs
    assert int(np.asarray(chg).sum()) >= 8


def test_closures_and_mass_based_ice_number():
    fx = _fx()
    raw, _ = _tapped(fx)
    q_c, N_c, q_i, N_i, T, melt, frz, chg = _helper(raw, 2.0, 0.2)
    dqc = np.asarray(q_c) - raw["q_c"]
    dqi = np.asarray(q_i) - raw["q_i"]
    clipped = (raw["q_c"] < 0) | (raw["q_i"] < 0)
    w = ~clipped
    assert np.abs(dqc[w] + dqi[w]).max() <= 1e-15                      # water
    dT = np.asarray(T) - raw["T"]
    assert np.abs(C.c_pd * dT - C.L_f * (np.asarray(frz) - np.asarray(melt))).max() <= 1e-9
    # frozen-droplet ice number follows the frozen MASS (25 um, rho 500), not N_c
    raw2 = dict(raw, N_c=2.0 * raw["N_c"])
    N_i2 = np.asarray(_helper(raw2, 2.0, 0.2)[3])
    _, h, _, frz_m, _, _ = _activity(fx)
    assert np.array_equal(N_i2[h][frz_m], np.asarray(N_i)[h][frz_m])
    k = np.asarray(frz)[h] > 0
    want = np.asarray(frz)[h][k] * 3.0 / (4.0 * 3.14 * 1.563e-14 * 500.0)
    assert np.allclose(np.asarray(N_i)[h][k] - raw["N_i"][h][k], want, rtol=1e-13)


def test_smooth_switch_on():
    """Zero at and beyond the threshold's wrong side, CAM's amount after the
    width W, C2 in between (decision 2026-10-20: W = 4/sharpness)."""
    q = jnp.full(7, 1.0e-3)
    x = jnp.array([-0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
    T = C.T_freeze + x
    z = jnp.zeros(7)
    out = W.mg2_instant_ice_melt_homog_freeze(z, z, q, z + 1e5, T, 233.15, 2.0, 0.2, presence_top=W.MG2_QSMALL)
    melt = np.asarray(out[5])
    hard = np.asarray(W.mg2_instant_ice_melt_homog_freeze(z, z, q, z + 1e5, T, 233.15, HARD, HARD, presence_top=W.MG2_QSMALL)[5])
    ramp = np.asarray(smoothstep_quintic(x / 2.0))
    assert melt[0] == 0.0 and melt[1] == 0.0
    assert np.allclose(melt, ramp * hard, rtol=1e-14, atol=0.0)
    assert np.allclose(melt[5:], hard[5:], rtol=1e-14)               # full strength past W
    assert 0.0 < melt[3] < hard[3]
    assert np.asarray(smoothstep_quintic(jnp.array(0.5))) == 0.5
    # widths in the production setup: 4 / existing sharpness (the oracle tests'
    # HARD_LIMIT sharpness 1e6 makes them ~4e-6 K, i.e. CAM's hard form)
    from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import _production_morrison
    cfg = _production_morrison()
    assert W.RAMP_WIDTH_PER_SHARPNESS / cfg.melt_sharpness == 2.0
    assert np.isclose(W.RAMP_WIDTH_PER_SHARPNESS / cfg.homogeneous_freeze_sharpness, 0.2)


def test_rounded_cap():
    """smooth_min(1, r) at sharpness 100 (decision 2026-10-20, 2b): never above
    the hard cap, ln(2)/100 below it at r = 1, < 7e-5 at 5 % from r = 1."""
    q = 1.0e-3
    r = jnp.array([0.5, 0.95, 1.0, 1.05, 2.0])
    x = r * C.L_f * q / C.c_pd                       # excess so heat allows r*q
    z = jnp.zeros(5)
    melt = np.asarray(W.mg2_instant_ice_melt_homog_freeze(
        z, z, z + q, z + 1e5, C.T_freeze + x, 233.15, HARD, HARD, presence_top=W.MG2_QSMALL)[5]) / q
    hard = np.minimum(np.asarray(r), 1.0)
    assert (melt <= hard * (1.0 + 1e-13)).all()
    assert np.isclose(hard[2] - melt[2], np.log(2.0) / 100.0, rtol=1e-6)
    assert (np.abs(melt - hard)[[0, 4]] < 1e-13).all()
    assert (np.abs(melt - hard)[[1, 3]] < 7e-5).all()     # 5 % off the corner: < 0.007 %


def test_gradients():
    """check_grads (order 2) inside the ramps and in the heat-limited band; inputs
    in K offsets and g/kg so the finite-difference step is meaningful."""
    def melt_fn(dT, q_gkg):
        z = jnp.zeros_like(dT)
        o = W.mg2_instant_ice_melt_homog_freeze(z + 1e-4, z + 1e8, 1e-3 * q_gkg, z + 1e5,
                                                C.T_freeze + dT, 233.15, 2.0, 0.2, presence_top=W.MG2_QSMALL)
        return 1e3 * o[0] + (o[4] - C.T_freeze)

    def frz_fn(dT, q_gkg):
        z = jnp.zeros_like(dT)
        o = W.mg2_instant_ice_melt_homog_freeze(1e-3 * q_gkg, z + 1e8, z, z, 233.15 - dT,
                                                233.15, 2.0, 0.2, presence_top=W.MG2_QSMALL)
        return 1e3 * o[2] + (o[4] - 233.15)

    # in the ramp (twice), past it, and AT the rounded cap's corner (r = 1)
    dT_m = jnp.array([0.45, 1.25, 2.85, 3.0])
    q_m = jnp.array([0.1, 5.0, 2.0, 9.03])       # 3 K melts 9.03 g/kg: r = 1
    check_grads(melt_fn, (dT_m, q_m), order=2, modes=("rev",), eps=1e-5, atol=1e-5, rtol=1e-4)
    dT_h = jnp.array([0.08, 0.17, 0.65, 0.3324])
    q_h = jnp.array([0.01, 0.6, 3.0, 1.0])
    check_grads(frz_fn, (dT_h, q_h), order=2, modes=("rev",), eps=1e-5, atol=1e-5, rtol=1e-4)
    g = jax.grad(lambda d: melt_fn(d, q_m).sum())(dT_m)
    assert np.isfinite(np.asarray(g)).all() and np.abs(np.asarray(g)).min() > 0


def test_gradient_finite_at_sub_qsmall_mass():
    """User decision 2026-10-26 (b): the heat-limit ratio is computed only where
    the mass is >= qsmall, fraction 1 below.  The failing cell is the one the
    whole-call gradient test hit (test_mg2_ice_rest_oracle, cell 3): ice melt
    called with dumi = 1.03e-219 kg/kg at 25.15 K below 0 C, where excess/mass
    overflowed to ~-8e217 and the gradient through smooth_min was NaN."""
    T = C.T_freeze - 25.15282158
    m = 1.03282319e-219

    def frac(x, mass):
        return W.mg2_phase_change_fraction(x, mass, 2.0, W.MG2_CAP_SHARPNESS)

    g = jax.grad(frac, argnums=(0, 1))(jnp.asarray(T - C.T_freeze), jnp.asarray(m))
    assert all(np.isfinite(float(gi)) for gi in g), g

    def melt(q_i, T_):
        z = jnp.zeros_like(q_i)
        o = W.mg2_instant_ice_melt_homog_freeze(z, z, q_i, z + 1.0, T_, 233.15, 2.0, 0.2,
                                                presence_top=W.MG2_QSMALL)
        return jnp.sum(o[2]) + jnp.sum(o[4])
    g = jax.grad(melt, argnums=(0, 1))(jnp.asarray([m]), jnp.asarray([T]))
    assert all(np.isfinite(np.asarray(gi)).all() for gi in g), g
    # below qsmall the fraction is the ramp alone (cap fraction 1), above unchanged
    x = jnp.asarray([0.5, 0.5])
    q = jnp.asarray([5.0e-19, 1.0e-4])
    f = np.asarray(frac(x, q))
    assert f[0] == float(smoothstep_quintic(0.25))
    r = 0.5 * C.c_pd / (C.L_f * 1.0e-4)
    assert np.isclose(f[1], float(smoothstep_quintic(0.25)) * min(r, 1.0), rtol=1e-3)


def test_recipient_raw_and_donor_overwrite():
    """Negative raw recipient is incremented, not clipped; donor overwritten from
    its clipped pool (MG2 += / = semantics, M:2634/2645, M:2672/2680)."""
    one = jnp.ones(1)
    q_c, N_c, q_i, N_i, T, melt, frz, chg = W.mg2_instant_ice_melt_homog_freeze(
        -1e-6 * one, -5.0 * one, 1e-4 * one, 1e5 * one, 280.0 * one, 233.15, 2.0, 0.2, presence_top=W.MG2_QSMALL)
    assert np.isclose(float(q_c[0]), -1e-6 + 1e-4, rtol=1e-12) and float(q_i[0]) == 0.0
    q_c, N_c, q_i, N_i, T, melt, frz, chg = W.mg2_instant_ice_melt_homog_freeze(
        1e-4 * one, 1e8 * one, -1e-7 * one, -3.0 * one, 220.0 * one, 233.15, 2.0, 0.2, presence_top=W.MG2_QSMALL)
    assert np.isclose(float(q_i[0]), -1e-7 + 1e-4, rtol=1e-12) and float(q_c[0]) == 0.0
    # gates on the mass strictly > 0 (not qsmall): 0 < q_i < qsmall still melts
    tiny = W.mg2_instant_ice_melt_homog_freeze(0 * one, 0 * one, 5e-19 * one, 0 * one,
                                               280.0 * one, 233.15, 2.0, 0.2, presence_top=W.MG2_QSMALL)
    assert float(tiny[5][0]) == 5e-19 and bool(tiny[7][0])
    # no co-firing: a melt cell never freezes
    assert float(W.mg2_instant_ice_melt_homog_freeze(
        1e-3 * one, 1e8 * one, 1e-3 * one, 1e5 * one, 274.0 * one, 233.15, 2.0, 0.2, presence_top=W.MG2_QSMALL)[6][0]) == 0.0


def test_morrison_wires_the_post_sedimentation_state(monkeypatch):
    """The scheme feeds the helper exactly the post-sedimentation state and its
    targets reach the outputs; Morrison's own melt and freezing rates are off."""
    fx = A._fx()
    seen = {}
    real = M.mg2_instant_ice_melt_homog_freeze

    def spy(*a, **k):
        seen["in"] = a
        seen["out"] = real(*a, **k)
        return seen["out"]

    def ident(q_c, N_c, q_i, N_i, T, *a, **k):
        z = jnp.zeros_like(q_c)
        return q_c, N_c, q_i, N_i, T, z, z, jnp.zeros(q_c.shape, bool)

    cloudy = jnp.asarray(fx["col_q_c"]) > 0
    even = jnp.arange(cloudy.shape[1]) % 2 == 0
    T = jnp.where(cloudy, jnp.where(even, 274.5, 228.0), jnp.asarray(fx["col_T"]))
    # ice only in the melting cells, so no Bergeron uptake empties the freezing ones
    hyd = dict(q_i=jnp.where(cloudy & even, 2e-4, 0.0), N_i=jnp.where(cloudy & even, 1e5, 0.0))
    # near-ice-saturated vapour in the freezing cells (the warm fixture's humidity at
    # 228 K would deposit enough ice to warm them past -40 C)
    fx["col_q_v"] = np.where(np.asarray(cloudy & ~even), 1.0e-5, fx["col_q_v"])
    monkeypatch.setattr(M, "mg2_instant_ice_melt_homog_freeze", ident)
    base = A._run(fx, T=T, hyd_kw=hyd)
    monkeypatch.setattr(M, "mg2_instant_ice_melt_homog_freeze", spy)
    out = A._run(fx, T=T, hyd_kw=hyd)
    dt = float(fx["dt"])
    q_c_in, _, q_i_in, _, T_in = (np.asarray(v) for v in seen["in"][:5])
    cfg = _cfg()
    assert seen["in"][5:] == (cfg.homogeneous_freeze_T, 4.0 / cfg.melt_sharpness,
                              4.0 / cfg.homogeneous_freeze_sharpness)
    assert np.allclose(q_c_in, fx["col_q_c"] + np.asarray(base.dq_c_dt) * dt, rtol=1e-13, atol=1e-20)
    assert np.allclose(q_i_in, np.asarray(hyd["q_i"]) + np.asarray(base.dq_i_dt) * dt,
                       rtol=1e-13, atol=1e-20)
    assert np.allclose(T_in, np.asarray(T) + np.asarray(base.dT_dt) * dt, rtol=1e-14)
    chg = np.asarray(seen["out"][7])
    assert chg.any()
    # increments reproduce the targets to rounding of the start state (~1e-19 here)
    assert np.allclose((fx["col_q_c"] + np.asarray(out.dq_c_dt) * dt)[chg],
                       np.asarray(seen["out"][0])[chg], rtol=1e-12, atol=W.MG2_QSMALL)
    assert np.allclose((np.asarray(T) + np.asarray(out.dT_dt) * dt)[chg],
                       np.asarray(seen["out"][4])[chg], rtol=1e-14)
    qb = {k: np.asarray(v) for k, v in out.qc_budget.items()}
    resid = sum(qb.values()) - np.asarray(out.dq_c_dt)
    assert np.abs(resid).max() <= 1e-12 * np.abs(np.asarray(out.dq_c_dt)).max()
    assert (qb["ice_melt"] > 0).any() and (qb["homogeneous_freezing"] < 0).any()


def test_warm_ice_melts_into_droplets_not_rain():
    """Ice-only cells at +7 C in ice-saturated air: under the switch melted ice
    becomes cloud water (MG2), none of it rain (Morrison's melt rate is off)."""
    fx = A._fx()
    cloudy = jnp.asarray(fx["col_q_c"]) > 0
    z = jnp.zeros_like(cloudy, dtype=float)
    T = jnp.where(cloudy, 280.0, jnp.asarray(fx["col_T"]))
    # ice-saturated vapour: neither deposition nor sublimation moves the ice
    from legoesm.thermo import saturation_specific_humidity_ice
    qs = np.asarray(saturation_specific_humidity_ice(T, jnp.asarray(fx["col_p_full"])))
    fx["col_q_v"] = np.where(np.asarray(cloudy), qs, fx["col_q_v"])
    fx["col_q_r"] = np.zeros_like(fx["col_q_r"])
    out = A._run(fx, T=T, q_c=z, npccn_kg=z,
                 hyd_kw=dict(q_r=z, N_r=z, q_i=jnp.where(cloudy, 1e-4, 0.0),
                             N_i=jnp.where(cloudy, 1e5, 0.0)))
    c = np.asarray(cloudy)
    dt = float(fx["dt"])
    melt = np.asarray(out.qc_budget["ice_melt"])[c] * dt
    assert (melt > 0.2e-4).all()          # most of the 1e-4 (some ice sediments out)
    assert np.abs(np.asarray(out.dq_r_dt)[c] * dt).max() <= 1e-3 * melt.min()


def test_readjust_qsmall_edge_keeps_a_negative_number():
    """At in-cloud q == qsmall exactly (cf = 1) MG2's re-adjust gate passes but the
    size routine (qcic > qsmall) leaves the number, so a negative post-step number
    survives; just above qsmall it is reset into the size bounds."""
    one = jnp.ones(2)
    rho, dt = 1.0 * one, 600.0
    N_c, dN = 1.0e6 * one, -2.0e6 / dt * one
    q_post = jnp.array([W.MG2_QSMALL, 2.0 * W.MG2_QSMALL])
    out = np.asarray(W.mg2_droplet_number_readjust(q_post, N_c + dN * dt, N_c, one, rho, dt, dN, presence_top=W.MG2_QSMALL))
    assert out[0] == float(dN[0])                    # untouched: number ends at -1e6
    assert (1.0e6 + out[1] * dt) > 0.0               # reset to a positive number
    # an in-range number is left bit-identical (no round-trip through (N*cf - N_c)/dt)
    q_ok, n_ok = jnp.array([1e-4]), jnp.array([5e7])
    d = jnp.array([1.2345e3])
    assert np.asarray(W.mg2_droplet_number_readjust(q_ok, n_ok + d * dt, n_ok, jnp.ones(1),
                                                    jnp.ones(1), dt, d, presence_top=W.MG2_QSMALL))[0] == 1.2345e3


def test_no_net_droplet_floor_under_the_switch(monkeypatch):
    """Whatever the re-adjust returns (here a number that would end negative) reaches
    the output: MG2 has no net floor (decision 2026-10-20)."""
    fx = A._fx()

    def neg(q_post, N_post, N_c, *a, **k):
        return -3.0 * jnp.maximum(N_c, 1.0) / float(fx["dt"])

    monkeypatch.setattr(M, "mg2_droplet_number_readjust", neg)
    out = A._run(fx)
    want = np.asarray(neg(None, None, jnp.asarray(fx["col_nc_kg"] * fx["col_rho"])))
    q_post = fx["col_q_c"] + np.asarray(out.dq_c_dt) * float(fx["dt"])
    keep = q_post >= W.MG2_QSMALL                     # the number clear does not apply
    assert keep.any() and np.array_equal(np.asarray(out.dN_c_dt)[keep], want[keep])
