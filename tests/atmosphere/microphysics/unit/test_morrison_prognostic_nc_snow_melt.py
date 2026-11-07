"""MorrisonConfig.prognostic_droplet_number, stage S10.4b: MG2 instant snow melt
above +2 C at its two points (pre-applied on the start state, micro_mg2_0.F90:
1145-1175, moved back into the tendencies at :2024-2040; and after
sedimentation, :2537-2565), with the smooth switch-on and rounded heat cap
(user decisions 2026-10-20), against the RUNNING MG2.

Fixture ``oracle_fixtures/mg2_prognostic_nc_snow.npz``: ``make_fixture.py
--prognostic-nc-snow`` through ``mg2_tap`` (build.sh: tap_snow1 after the
pre-apply, tap_snow2/3 around the post-sedimentation melt; assignment-only
lines).  Function-level at both points; a vanishing ramp width and a 1e12 cap
sharpness give CAM's hard form.  x64.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm import constants as C  # noqa: E402
from legoesm.atmosphere.physics.microphysics import _warm_rain as W  # noqa: E402
from legoesm.atmosphere.physics.microphysics import morrison as M  # noqa: E402
from legoesm.core.smooth import smoothstep_quintic  # noqa: E402

from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import TOL  # noqa: E402
from tests.atmosphere.microphysics.unit.test_morrison_prognostic_nc import _cfg  # noqa: E402
from tests.atmosphere.microphysics.unit import test_morrison_prognostic_nc_act as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/mg2_prognostic_nc_snow.npz"
HARD, HARD_CAP = 1.0e-12, 1.0e12
T_SNOW = C.T_freeze + 2.0


def _qvi_mg2(T, p):
    """MG2's ice-saturation humidity on the scheme's own CAM Goff-Gratch curves:
    qvl at or above 0 C, qvi below (micro_mg2_0.F90:909-916)."""
    from legoesm.thermo import (saturation_specific_humidity_goff_gratch as qvl,
                                saturation_specific_humidity_ice_goff_gratch as qvi)
    return jnp.where(T >= C.T_freeze, qvl(T, p), qvi(T, p))


def _fx():
    return dict(np.load(FIXTURE))


def _melt(q, n, T, w=HARD, cap=HARD_CAP):
    m, k = W.mg2_instant_snow_melt(jnp.asarray(T), jnp.asarray(q), jnp.asarray(n), w, cap)
    return np.asarray(m), np.asarray(k)


def test_fixture_inputs_and_coverage():
    fx = _fx()
    assert np.allclose(fx["col_rho"], fx["col_p_full"] / (C.R_d * fx["col_T"]), rtol=1e-14)
    assert not any(np.isnan(v).any() for v in fx.values() if v.dtype.kind == "f")
    assert (fx["col_q_i"] == 0).all() and (fx["col_T"] > C.T_freeze).any()
    qs, T, dt = fx["col_q_s"], fx["col_T"], float(fx["dt"])
    f1 = np.where(qs > 0, (qs - fx["ora_snow1_qs"]) / np.where(qs > 0, qs, 1.0), 0.0)
    assert (f1 > 0.999).sum() >= 10 and ((f1 > 0) & (f1 < 0.99)).any()
    assert ((T > T_SNOW) & (qs > 0) & (qs < W.MG2_QSMALL)).any()       # gate > 0, not qsmall
    d = fx["ora_snow2_dums"]
    f2 = np.where(d > 0, (fx["ora_snow2_qstend"] - fx["ora_snow3_qstend"]) * dt
                  / np.where(d > 0, d, 1.0), 0.0)
    assert (f2 > 0.999).sum() >= 2 and ((f2 > 0) & (f2 < 0.99)).any()


def test_pre_apply_matches_mg2():
    fx = _fx()
    dt = float(fx["dt"])
    qs, ns, T = fx["col_q_s"], fx["col_ns_kg"], fx["col_T"]
    m, n = _melt(qs, ns, T)
    nr0 = fx["col_N_r"] / fx["col_rho"]
    errs = {
        "qs": np.abs(np.maximum(qs - m, 0.0) - fx["ora_snow1_qs"]).max() / 1e-3,
        "ns": np.abs(np.maximum(ns - n, 0.0) - fx["ora_snow1_ns"]).max() / 1e4,
        "qr": np.abs(np.maximum(fx["col_q_r"] + m, 0.0) - fx["ora_snow1_qr"]).max() / 1e-3,
        "nr": np.abs(np.maximum(nr0 + n, 0.0) - fx["ora_snow1_nr"]).max() / 1e4,
        "tlat": np.abs(-C.L_f * m / dt - fx["ora_snow1_tlat"]).max() / (C.L_f * 1e-3 / dt),
    }
    print("S10.4b pre-apply vs MG2", errs)
    assert max(errs.values()) <= TOL, errs


def test_post_sedimentation_matches_mg2():
    fx = _fx()
    dt = float(fx["dt"])
    m, n = _melt(fx["ora_snow2_dums"], fx["ora_snow2_dumns"], fx["ora_snow2_Tp"])
    d = {k: fx[f"ora_snow3_{k}"] - fx[f"ora_snow2_{k}"]
         for k in ("qstend", "nstend", "qrtend", "nrtend", "tlat")}
    errs = {
        "qs": np.abs(-m / dt - d["qstend"]).max() / (1e-3 / dt),
        "ns": np.abs(-n / dt - d["nstend"]).max() / (1e4 / dt),
        "qr": np.abs(m / dt - d["qrtend"]).max() / (1e-3 / dt),
        "nr": np.abs(n / dt - d["nrtend"]).max() / (1e4 / dt),
        "tlat": np.abs(-C.L_f * m / dt - d["tlat"]).max() / (C.L_f * 1e-3 / dt),
    }
    print("S10.4b post-sedimentation vs MG2", errs)
    assert max(errs.values()) <= TOL, errs


def test_smooth_switch_on_and_cap():
    x = np.array([-0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
    q = np.full(6, 1.0e-4)
    m, _ = _melt(q, q * 0 + 1e4, T_SNOW + x, w=2.0, cap=W.MG2_CAP_SHARPNESS)
    hard, _ = _melt(q, q * 0 + 1e4, T_SNOW + x)
    assert m[0] == 0.0 and m[1] == 0.0
    assert np.allclose(m, np.asarray(smoothstep_quintic(jnp.asarray(x / 2.0))) * hard,
                       rtol=1e-12, atol=0.0)
    # heat cap rounded: at r = 1 ln(2)/100 below the hard cap, never above it
    qq = 3.0 * C.c_pd / C.L_f
    m1, _ = _melt(np.array([qq]), np.array([1e4]), np.array([T_SNOW + 3.0]), w=2.0,
                  cap=W.MG2_CAP_SHARPNESS)
    assert np.isclose(1.0 - m1[0] / qq, np.log(2.0) / 100.0, rtol=1e-6)
    from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import _production_morrison
    assert W.RAMP_WIDTH_PER_SHARPNESS / _production_morrison().melt_sharpness == 2.0


def test_gradients():
    def fn(dT, q_gkg):
        m, n = W.mg2_instant_snow_melt(T_SNOW + dT, 1e-3 * q_gkg, 1e4 * q_gkg, 2.0)
        return 1e3 * m + 1e-4 * n

    dT = jnp.array([0.4, 1.3, 2.8, 3.0])
    q = jnp.array([0.1, 5.0, 2.0, 9.03])        # last: r = 1 past the ramp
    check_grads(fn, (dT, q), order=2, modes=("rev",), eps=1e-5, atol=1e-5, rtol=1e-4)


def _snow_run(fx, q_s, N_s, T, cfg=None, hyd=None):
    z = jnp.zeros_like(jnp.asarray(fx["col_T"]))
    kw = dict(q_s=q_s, N_s=N_s, q_c=z, q_i=z, N_i=z)
    kw.update(hyd or {})
    return A._run(fx, cfg=cfg, T=T, npccn_kg=z, q_c=z, hyd_kw=kw)


def test_pre_apply_restore_is_the_premelted_run_plus_the_melt():
    """Every snow cell melts fully at the pre point (f = 1), so the pre-melted
    state S' has no snow left: scheme(S) == scheme(S') + (S' - S)/dt."""
    fx = A._fx()
    dt = float(fx["dt"])
    rho = jnp.asarray(fx["col_rho"])
    T = jnp.maximum(jnp.asarray(fx["col_T"]), 290.0)
    snow = jnp.asarray(fx["col_q_c"]) > 0
    q_s = jnp.where(snow, 1.0e-4, 0.0)
    N_s = jnp.where(snow, 3.0e4, 0.0)
    out = _snow_run(fx, q_s, N_s, T)
    q_r0, N_r0 = jnp.asarray(fx["col_q_r"]), jnp.asarray(fx["col_N_r"])
    z = jnp.zeros_like(q_s)
    ref = _snow_run(fx, z, z, T, hyd=dict(q_r=q_r0 + q_s, N_r=N_r0 + N_s * rho))
    assert np.allclose(np.asarray(out.dq_s_dt), np.asarray(ref.dq_s_dt) - np.asarray(q_s) / dt,
                       rtol=1e-12, atol=1e-22)
    assert np.allclose(np.asarray(out.dN_s_dt), np.asarray(ref.dN_s_dt) - np.asarray(N_s) / dt,
                       rtol=1e-12, atol=1e-12)
    assert np.allclose(np.asarray(out.dq_r_dt), np.asarray(ref.dq_r_dt) + np.asarray(q_s) / dt,
                       rtol=1e-12, atol=1e-20)
    assert np.allclose(np.asarray(out.dN_r_dt),
                       np.asarray(ref.dN_r_dt) + np.asarray(N_s * rho) / dt, rtol=1e-12,
                       atol=1e-9)
    dT_pre = -C.L_f * np.asarray(q_s) / (C.c_pd * dt)
    assert np.allclose(np.asarray(out.dT_dt), np.asarray(ref.dT_dt) + dT_pre,
                       rtol=1e-12, atol=1e-15)
    assert np.abs(dT_pre).max() > 0


def test_morrison_snow_melt_rate_off_and_number_needed():
    fx = A._fx()
    z = jnp.zeros_like(jnp.asarray(fx["col_T"]))
    with pytest.raises(ValueError, match="prognostic snow number"):
        A._run(fx, hyd_kw=dict(N_s=None))
    # a cell between 0 and +2 C keeps its snow under the switch (Morrison's
    # PSMLT rate is off; MG2 only melts above +2 C)
    snow = jnp.asarray(fx["col_q_c"]) > 0
    T = jnp.where(snow, C.T_freeze + 1.0, jnp.asarray(fx["col_T"]))
    fx["col_q_v"] = np.asarray(_qvi_mg2(T, jnp.asarray(fx["col_p_full"])))
    out = _snow_run(fx, jnp.where(snow, 1e-4, 0.0), jnp.where(snow, 3e4, 0.0), T,
                    hyd=dict(q_r=z, N_r=z))
    assert np.abs(np.asarray(out.dq_r_dt)[np.asarray(snow)]).max() < 1e-12


def test_graupel_heat_cap_reads_the_premelted_temperature(monkeypatch):
    """Snow pre-melt cools the cell; graupel melting is capped by the heat left
    after that cooling (user decision 4a).  Graupel fall speed 0, ice-saturated
    vapour, no cloud, and a melt demand of all the graupel, so the cap binds."""
    fx = A._fx()
    dt = float(fx["dt"])
    cell = jnp.asarray(fx["col_q_c"]) > 0
    T = jnp.where(cell, T_SNOW + 1.0, jnp.asarray(fx["col_T"]))
    q_s = jnp.where(cell, 5.0e-4, 0.0)          # pre-apply melts half of it (ramp at +1 K)
    q_g = jnp.where(cell, 1.0e-2, 0.0)
    fx["col_q_v"] = np.asarray(_qvi_mg2(T, jnp.asarray(fx["col_p_full"])))
    monkeypatch.setattr(M, "graupel_melting_pgmlt",
                        lambda q_g, *a, **k: jnp.clip(q_g, 0.0) / dt)
    # graupel vapour exchange runs on the SAM saturation curve, the vapour here
    # on MG2's Goff-Gratch one: switch it off so only melting moves heat
    # graupel vapour exchange runs on the SAM saturation curve, the vapour here
    # on MG2's Goff-Gratch one: off, so only melting moves heat
    cfg = _cfg(fall_a_g=0.0, do_graupel_deposition=False)
    z = jnp.zeros_like(q_s)
    out = _snow_run(fx, q_s, jnp.where(cell, 3e4, 0.0), T, cfg=cfg,
                    hyd=dict(q_g=q_g, q_r=z, N_r=z))
    m_pre, _ = _melt(np.asarray(q_s), np.full(q_s.shape, 3e4), np.asarray(T), w=2.0,
                     cap=W.MG2_CAP_SHARPNESS)
    c = np.asarray(cell)
    assert (m_pre[c] > 0).all()
    # the melt demand exceeds the heat, so the cap binds: snow pre-melt plus
    # graupel melt together cool the cell exactly to 0 C, not below (a cap
    # reading the start T instead ends the cell 0.17 K below 0 C, measured)
    T_end = np.asarray(T) + np.asarray(out.dT_dt) * dt
    assert np.allclose(T_end[c], C.T_freeze, rtol=0.0, atol=1e-6)   # rounded cap: ~1e-8 K


def test_post_sedimentation_point_is_wired(monkeypatch):
    """Snow above +2 C air falls into it: the second (post-sedimentation) call
    reads the post-step snow pool (number cleared below qsmall) and the post-step
    temperature, and its melt reaches snow, rain (number x rho) and T."""
    fx = A._fx()
    dt = float(fx["dt"])
    nlev = fx["col_T"].shape[1]
    lev = np.arange(nlev)[None, :] * np.ones_like(fx["col_T"])
    T = jnp.asarray(np.where(lev < nlev // 2, 268.0, 281.0))
    fx["col_q_v"] = np.asarray(_qvi_mg2(T, jnp.asarray(fx["col_p_full"])))
    top = jnp.asarray((lev >= nlev // 2 - 3) & (lev < nlev // 2))
    # snow in the warm cells too: their pre-apply melt cools them 0.33 K before the
    # post point, so the post point's temperature differs from the start T
    warm = jnp.asarray(lev >= nlev // 2)
    q_s = jnp.where(top, 2.0e-3, jnp.where(warm, 1.0e-3, 0.0))
    N_s = jnp.where(top | warm, 1.0e4, 0.0)
    calls = []
    real = M.mg2_instant_snow_melt

    def spy(*a, **k):
        calls.append(a)
        out = real(*a, **k)
        calls.append(out)
        return out

    def zero_post(*a, **k):
        out = real(*a, **k)
        if len(calls) == 2:              # pre call recorded; zero the post one
            out = (0.0 * out[0], 0.0 * out[1])
        calls.append(a)
        calls.append(out)
        return out

    z = jnp.zeros_like(q_s)
    monkeypatch.setattr(M, "mg2_instant_snow_melt", zero_post)
    base = _snow_run(fx, q_s, N_s, T, hyd=dict(q_r=z, N_r=z))
    calls.clear()
    monkeypatch.setattr(M, "mg2_instant_snow_melt", spy)
    out = _snow_run(fx, q_s, N_s, T, hyd=dict(q_r=z, N_r=z))
    (Tp, dums, dumns, _w), (m, n) = calls[2], calls[3]
    m, n = np.asarray(m), np.asarray(n)
    assert (m > 0).sum() >= 3                              # snow fell into warm air
    assert np.allclose(np.asarray(dums),
                       np.maximum(np.asarray(q_s) + np.asarray(base.dq_s_dt) * dt, 0.0),
                       rtol=1e-12, atol=1e-20)
    assert np.allclose(np.asarray(Tp), np.asarray(T) + np.asarray(base.dT_dt) * dt, rtol=1e-14)
    assert np.abs(np.asarray(Tp) - np.asarray(T))[np.asarray(warm)].min() > 0.3   # 1 g/kg melted
    small = np.asarray(dums) < W.MG2_QSMALL
    assert np.all(np.asarray(dumns)[small] == 0.0)
    assert np.allclose(np.asarray(out.dq_s_dt) - np.asarray(base.dq_s_dt), -m / dt,
                       rtol=1e-9, atol=1e-18)
    assert np.allclose(np.asarray(out.dq_r_dt) - np.asarray(base.dq_r_dt), m / dt,
                       rtol=1e-9, atol=1e-18)
    assert np.allclose(np.asarray(out.dT_dt) - np.asarray(base.dT_dt),
                       -C.L_f * m / (C.c_pd * dt), rtol=1e-9, atol=1e-15)
    # rain number: the melted flakes arrive with MG2's slow number fall speed, so
    # the new rain sits on its size bound and the bound sets N_r; the x rho of
    # the post point is pinned by main's whole-column MG2 oracle
    # (test_mg2_ice_snow_oracle), not here.


def test_pre_apply_leaves_cells_it_does_not_enter():
    """A cold cell with a slightly negative snow and rain amount: MG2 does not
    enter the pre-apply there, so nothing is clipped or restored."""
    fx = A._fx()
    T = jnp.full_like(jnp.asarray(fx["col_T"]), 250.0)
    fx["col_q_v"] = np.asarray(_qvi_mg2(T, jnp.asarray(fx["col_p_full"])))
    neg = jnp.full_like(T, -1.0e-9)
    z = jnp.zeros_like(T)
    out = _snow_run(fx, neg, z, T, hyd=dict(q_r=neg, N_r=z))
    m, _ = W.mg2_instant_snow_melt(T, neg, z, 2.0)
    assert not np.asarray(m).any()
    assert np.abs(np.asarray(out.dq_s_dt)).max() < 1e-15


def test_graupel_melt_switches_on_smoothly(monkeypatch):
    """Under the switch the graupel melt is scaled by the smooth step (0 at 0 C,
    full after 2 K) times the rounded heat cap (user decision 2026-10-21 b):
    with a small demand (heat never binds) the melt is exactly step x demand."""
    fx = A._fx()
    dt = float(fx["dt"])
    cell = jnp.asarray(fx["col_q_c"]) > 0
    dTs = np.array([-0.3, 0.0, 0.5, 1.0, 1.5, 2.5])
    k = np.asarray(cell).nonzero()
    T = np.asarray(fx["col_T"]).copy()
    for j, (a, b) in enumerate(zip(*k)):
        T[a, b] = C.T_freeze + dTs[j % len(dTs)]
    T = jnp.asarray(T)
    fx["col_q_v"] = np.asarray(_qvi_mg2(T, jnp.asarray(fx["col_p_full"])))
    demand = 1.0e-8                                   # kg/kg/s: 6e-6 per step, heat ample
    monkeypatch.setattr(M, "graupel_melting_pgmlt",
                        lambda q_g, *a, **kw: jnp.where(q_g > 0, demand, 0.0))
    q_g = jnp.where(cell, 1.0e-3, 0.0)
    z = jnp.zeros_like(q_g)
    # production melt sharpness 2 /K (W = 2 K); the oracle tests' hard limit makes W ~ 4e-6 K
    cfg = _cfg(fall_a_g=0.0, do_graupel_deposition=False, do_graupel_rain_accretion=False)._replace(melt_sharpness=2.0)
    out = _snow_run(fx, z, z, T, cfg=cfg,
                    hyd=dict(q_g=q_g, q_r=z, N_r=z))
    c = np.asarray(cell)
    x = np.asarray(T)[c] - C.T_freeze
    want = np.asarray(smoothstep_quintic(jnp.asarray(x / 2.0))) * demand
    got = -np.asarray(out.dq_g_dt)[c]
    assert np.allclose(got, want, rtol=1e-6, atol=1e-20)
    assert (got[x <= 0] == 0).all() and (got[(x > 0) & (x < 2)] < demand).all()


def test_graupel_melt_gradient_is_smooth_through_0C():
    """Finite-difference check of d(graupel melt)/dT through the whole scheme at
    T - 0 C = 0.5 K (inside the step) and 1e-3 K (just above the threshold)."""
    fx = A._fx()
    cell = jnp.asarray(fx["col_q_c"]) > 0
    q_g = jnp.where(cell, 1.0e-3, 0.0)
    z = jnp.zeros_like(q_g)
    cfg = _cfg(fall_a_g=0.0, do_graupel_deposition=False)._replace(melt_sharpness=2.0)

    def melt(dT):
        T = jnp.where(cell, C.T_freeze + dT, jnp.asarray(fx["col_T"]))
        f2 = dict(fx, col_q_v=_qvi_mg2(T, jnp.asarray(fx["col_p_full"])))
        out = _snow_run(f2, z, z, T, cfg=cfg, hyd=dict(q_g=q_g, q_r=z, N_r=z))
        return -jnp.sum(jnp.where(cell, out.dq_g_dt, 0.0))

    for x in (0.5, 1.0e-3):
        g = float(jax.grad(melt)(x))
        h = 1e-4 * max(x, 1e-2)
        fd = (float(melt(x + h)) - float(melt(x - h))) / (2 * h)
        assert np.isfinite(g) and g > 0 and np.isclose(g, fd, rtol=2e-3), (x, g, fd)


def test_graupel_caps_are_rounded_not_hard(monkeypatch):
    """Under the switch PGMLT is called without dt (its hard heat and
    all-of-the-graupel caps off) and both limits are smooth_min(1, r, 100):
    heat-limited and graupel-limited cells match the rounded forms exactly."""
    fx = A._fx()
    dt = float(fx["dt"])
    cell = jnp.asarray(fx["col_q_c"]) > 0
    seen = {}
    k = np.asarray(cell).nonzero()
    T = np.asarray(fx["col_T"]).copy()
    q_g = np.zeros_like(T)
    demand = 3.0e-5                            # 1.8e-2 per step
    dt_ = float(fx["dt"])
    # past the 2 K step, so only the caps act: heat-limited, tiny graupel, and
    # graupel exactly at its rounded corner (q_g/dt == demand) with ample heat
    for j, (a, b) in enumerate(zip(*k)):
        T[a, b] = C.T_freeze + (3.0, 3.0, 10.0)[j % 3]
        q_g[a, b] = (1.0e-2, 1.0e-6, demand * dt_)[j % 3]
    T, q_g = jnp.asarray(T), jnp.asarray(q_g)
    fx["col_q_v"] = np.asarray(_qvi_mg2(T, jnp.asarray(fx["col_p_full"])))

    def pg(q, *a, dt=None, **kw):
        seen["dt"] = dt
        return jnp.where(q > 0, demand, 0.0)

    monkeypatch.setattr(M, "graupel_melting_pgmlt", pg)
    z = jnp.zeros_like(q_g)
    out = _snow_run(fx, z, z, T, cfg=_cfg(fall_a_g=0.0, do_graupel_deposition=False, do_graupel_rain_accretion=False),
                    hyd=dict(q_g=q_g, q_r=z, N_r=z))
    assert seen["dt"] is None
    c = np.asarray(cell)
    from legoesm.core.smooth import smooth_min
    r_q = np.asarray(q_g)[c] / dt / demand
    m1 = demand * np.asarray(smooth_min(jnp.ones_like(jnp.asarray(r_q)), jnp.asarray(r_q), 100.0))
    r_h = C.c_pd * (np.asarray(T)[c] - C.T_freeze) / (C.L_f * m1 * dt)
    want = m1 * np.asarray(smooth_min(jnp.ones_like(jnp.asarray(r_h)), jnp.asarray(r_h), 100.0))
    got = -np.asarray(out.dq_g_dt)[c]
    assert np.allclose(got, want, rtol=1e-10)
    assert (r_h < 1).any() and (r_q < 1).any() and (np.abs(r_q - 1.0) < 1e-12).any()
