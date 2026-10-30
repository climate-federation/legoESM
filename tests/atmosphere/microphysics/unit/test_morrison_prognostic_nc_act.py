"""MorrisonConfig.prognostic_droplet_number, stage S10.2: the MG2 droplet-number
budget (activation pre-apply and re-apply, nprc1/npra/riming number sinks from
the unlimited mass rates, the nc limiter with omsm), against the RUNNING MG2.

Fixture ``oracle_fixtures/mg2_prognostic_nc_act.npz``: ``make_fixture.py
--prognostic-nc-act`` through the probe build ``mg2_tap`` (build.sh: a copy of
micro_mg2_0.F90 of cam_cesm2_1_rel_60 with three added lines recording nctend
right after the activation re-apply, micro_mg2_0.F90:2022).  Bound: 1e-10
relative, hard-switch limit.  Where MG2's number would end the step negative
(N_c + npccn*dt < 0), MG2's end-of-step droplet re-adjust repairs it (ported in
S10.4, which also removed the interim net floor).
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants as C  # noqa: E402
from legoesm.atmosphere.physics.microphysics import _warm_rain as W  # noqa: E402
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics  # noqa: E402
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState  # noqa: E402

from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import TOL, _rel  # noqa: E402
from tests.atmosphere.microphysics.unit.test_morrison_prognostic_nc import _cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/mg2_prognostic_nc_act.npz"
OMSM = 1.0 - 1.0e-5


def _fx():
    return dict(np.load(FIXTURE))


def _run(fx, cfg=None, npccn_kg=None, N_c=None, q_c=None, hyd_kw=None, T=None):
    g = {k: jnp.asarray(fx[f"col_{k}"]) for k in
         ("T", "q_v", "q_c", "cf", "q_r", "N_r", "p_full", "p_half", "rho", "nc_kg", "relvar",
          "npccn")}
    dz = (g["p_half"][:, 1:] - g["p_half"][:, :-1]) / (C.g * g["rho"])
    z = jnp.zeros_like(g["T"])
    kw = dict(q_c=g["q_c"] if q_c is None else q_c, q_r=g["q_r"], q_i=z, q_s=z, q_g=z,
              N_c=g["nc_kg"] * g["rho"] if N_c is None else N_c, N_r=g["N_r"], N_i=z,
              N_s=z)   # the switch needs a snow number (decision 1a)
    kw.update(hyd_kw or {})
    npc = g["npccn"] if npccn_kg is None else npccn_kg
    return morrison_microphysics(
        g["T"] if T is None else T, g["q_v"], HydrometeorState(**kw), g["p_full"],
        g["p_half"], g["rho"], dz, float(fx["dt"]), cfg or _cfg(),
        cloud_fraction=g["cf"], relvar=g["relvar"], npccn=npc * g["rho"])


def _mg2_masks(fx):
    nc, npc, qc = fx["col_nc_kg"], fx["col_npccn"], fx["col_q_c"]
    dt = float(fx["dt"])
    pre = np.where(qc >= 1e-18, np.maximum(nc + npc * dt, 0.0), nc)
    fin = nc + fx["ora_nctend_stage"] * dt
    neg = fin < 0.0
    bound = (pre > 0) & (np.abs(fin / np.where(pre > 0, pre, 1.0) - (1.0 - OMSM)) < 1e-6)
    return pre, neg, bound


def test_droplet_number_budget_matches_mg2_stage():
    """Compared with MG2's WHOLE-CALL nctend on every cell (S10.4: warm columns,
    so after sedimentation only the final droplet re-adjust, :2835, and the
    number clear, :2913, write it -- both ported; no net floor).  The S10.2
    coverage masks use the probe right after the activation re-apply."""
    fx = _fx()
    out = _run(fx)
    dt = float(fx["dt"])
    dN = np.asarray(out.dN_c_dt) / fx["col_rho"]          # per kg, as MG2
    _, _, bound = _mg2_masks(fx)
    nc = fx["col_nc_kg"]
    clear = fx["col_q_c"] + fx["ora_qctend"] * dt < W.MG2_QSMALL
    neg = ~clear & (nc + fx["ora_nctend_sed"] * dt < 0.0)
    err = _rel(dN, fx["ora_nctend"])
    print("S10.2 nctend max rel err", float(err.max()), "cells", err.size,
          "repaired-negative cells", int(neg.sum()), "clear cells", int(clear.sum()))
    assert float(err.max()) <= TOL
    # the removed interim floor (-nc/dt) would differ from MG2 in the cells
    # MG2's re-adjust repairs
    assert neg.any() and (np.abs(-nc[neg] / dt - fx["ora_nctend"][neg])
                          > 1e-6 * np.abs(nc[neg] / dt)).all()
    qb = out.qc_budget
    for k, ora in (("autoconversion", "ora_prctot"), ("accretion", "ora_pratot")):
        e = float(_rel(-np.asarray(qb[k]), fx[ora]).max())
        assert e <= TOL, (k, e)
    # coverage, all on the MG2 side
    npc, qc = fx["col_npccn"], fx["col_q_c"]
    assert bound.any()                                           # number limiter binds
    witness = (fx["ora_qcrat"] < 1.0) & ~bound & (npc == 0) & (qc > 0)
    assert witness.any()                                         # mass limited, number not
    assert ((npc > 0) & (qc > 0) & ~neg).any() and ((npc < 0) & (qc > 0) & ~neg).any()
    # clear air: MG2 re-applies npccn (:2022) and its number clear (:2913) then
    # overrides it; both sides end at -nc/dt there
    assert ((npc != 0) & (qc == 0) & (fx["ora_nctend_stage"] == npc)).any() and clear.any()
    capped = (qc / np.maximum(fx["col_cf"], 1e-4) > 5.0e-3) & ~neg & ~bound
    assert capped.any()                       # MG2's 5e-3 in-cloud cap, number not limited
    assert neg.sum() < (~neg).sum()


def test_activation_pre_apply_moves_the_rates():
    fx = _fx()
    with_act = np.asarray(_run(fx).qc_budget["autoconversion"])
    without = np.asarray(_run(fx, npccn_kg=jnp.zeros_like(jnp.asarray(fx["col_npccn"])))
                         .qc_budget["autoconversion"])
    moved = np.abs(with_act - without) > 1e-6 * np.abs(without)
    assert (moved & (fx["col_npccn"] != 0)).any()
    assert not (moved & (fx["col_npccn"] == 0)).any()


def _no_bergeron(monkeypatch):
    """MG2 Bergeron onto snow and onto cloud ice also drain cloud water here;
    these tests isolate the riming number sinks, so both are zeroed."""
    from legoesm.atmosphere.physics.microphysics import morrison as M
    monkeypatch.setattr(M, "mg2_bergeron_snow", lambda T, q_cic, *a: jnp.zeros_like(q_cic))
    real = M.mg2_ice_deposition_sublimation

    def no_berg(*a, **k):
        dep, subl, berg = real(*a, **k)
        return dep, subl, jnp.zeros_like(berg)
    monkeypatch.setattr(M, "mg2_ice_deposition_sublimation", no_berg)


def test_cold_riming_number_sinks_and_no_ice_riming(monkeypatch):
    _no_bergeron(monkeypatch)
    """Cold cloudy cells with snow, graupel and ice: riming by cloud ice is 0;
    the droplet-number sink is (snow + graupel riming) x the size-reset ncic/qcic
    where autoconversion and accretion vanish (q_r = 0, qcic < icsmall)."""
    fx = _fx()
    qc0 = jnp.asarray(fx["col_q_c"])
    cloudy = qc0 > 0
    cf = jnp.maximum(jnp.asarray(fx["col_cf"]), 1e-4)
    q_c = jnp.where(cloudy, 5.0e-9 * cf, 0.0)                    # qcic = 5e-9 < 1e-8
    T = jnp.where(cloudy, 258.0, jnp.asarray(fx["col_T"]))
    sol = jnp.where(cloudy, 1.0e-4, 0.0)
    cfg = _cfg()
    out = _run(fx, cfg=cfg, npccn_kg=jnp.zeros_like(qc0), q_c=q_c, T=T,
               hyd_kw=dict(q_r=jnp.zeros_like(qc0), q_s=sol, q_g=sol, q_i=0.1 * sol,
                           N_i=jnp.where(cloudy, 1.0e5, 0.0)))
    qb = {k: np.asarray(v) for k, v in out.qc_budget.items()}
    assert np.all(qb["riming_ice"] == 0.0)
    rim = -(qb["riming_snow"] + qb["riming_graupel"])
    c = np.asarray(cloudy)
    assert (rim[c] > 0).all() and np.all(qb["autoconversion"][c] == 0) \
        and np.all(qb["accretion"][c] == 0)
    sinks = -sum(v for k, v in qb.items() if k not in ("condensation", "sedimentation"))
    dt = float(fx["dt"])
    assert (sinks[c] * dt < 0.99 * np.asarray(q_c)[c]).all()   # qc mass limiter not binding
    rho = jnp.asarray(fx["col_rho"])
    qcic = q_c / cf
    ncic = W.mg2_size_bounded_nc(qcic, jnp.asarray(fx["col_nc_kg"]) * rho / cf, rho)
    pre = -rim * np.asarray(ncic / qcic)
    got = np.asarray(out.dN_c_dt)
    nc_vol = np.asarray(jnp.asarray(fx["col_nc_kg"]) * rho)
    has_n = _top(c) & (nc_vol > 0)
    assert has_n.sum() >= 3
    assert (-pre[has_n] * dt < nc_vol[has_n]).all()             # nc limiter not binding
    want = _with_sedimentation(fx, out, q_c, jnp.asarray(nc_vol),
                               jnp.asarray(np.where(has_n, pre, 0.0)), T)
    assert float(_rel(got[has_n], want[has_n]).max()) <= 1e-12
    # a cloudy cell with no droplets: the nc limiter removes nothing (MG2 :1628-1642);
    # only MG2's final size re-adjust (S10.4) gives it droplets
    empty = c & (nc_vol == 0)
    assert empty.any()
    want0 = _with_sedimentation(fx, out, q_c, jnp.asarray(nc_vol), jnp.zeros_like(q_c), T)
    assert float(_rel(got[empty], want0[empty]).max()) <= 1e-12 and (got[empty] > 0).all()


def _top(c):
    """Top cloudy level of each cloud layer: no droplets fall into it, so its
    sedimentation depends on its own state only."""
    return c & ~np.concatenate([np.zeros_like(c[:, :1]), c[:, :-1]], axis=1)


def _with_sedimentation(fx, out, q_c, N_c, dN_pre, T):
    """``dN_pre`` (per volume, before droplet sedimentation and the number clear)
    carried through the S10.3 steps with the oracle-tested port, as Morrison does."""
    rho = jnp.asarray(fx["col_rho"])
    dt = float(fx["dt"])
    cf = jnp.maximum(jnp.asarray(fx["col_cf"]), 1e-4)
    dqc_pre = out.dq_c_dt - out.qc_budget["sedimentation"]
    _, sed_n, _, _, _, _ = W.mg2_cloud_sedimentation(
        q_c + dqc_pre * dt, (N_c + dN_pre * dt) / rho, cf, T, rho,
        jnp.asarray(fx["col_p_half"]), dt, _cfg().sed_cfl_substeps_max, presence_top=W.MG2_QSMALL)
    dN = dN_pre + sed_n * rho
    # S10.4 final droplet re-adjust (no ice melt / homogeneous freezing at 258 K)
    q_ic = jnp.minimum(jnp.maximum(q_c + out.dq_c_dt * dt, 0.0) / cf, W.KK2000_CAM6_QC_MAX)
    n_ic = jnp.maximum(N_c + dN * dt, 0.0) / cf
    n_adj = W.mg2_size_bounded_nc(q_ic, n_ic, rho)
    dN = jnp.where((q_ic >= W.MG2_QSMALL) & (n_adj != n_ic), (n_adj * cf - N_c) / dt, dN)
    return np.asarray(jnp.where(q_c + out.dq_c_dt * dt < W.MG2_QSMALL, -N_c / dt, dN))


def _cold(fx, qcic, q_r, nc_scale=1.0):
    qc0 = jnp.asarray(fx["col_q_c"])
    cloudy = qc0 > 0
    cf = jnp.maximum(jnp.asarray(fx["col_cf"]), 1e-4)
    q_c = jnp.where(cloudy, qcic * cf, 0.0)
    T = jnp.where(cloudy, 258.0, jnp.asarray(fx["col_T"]))
    sol = jnp.where(cloudy, 1.0e-4, 0.0)
    rho = jnp.asarray(fx["col_rho"])
    N_c = jnp.asarray(fx["col_nc_kg"]) * rho * nc_scale
    out = _run(fx, npccn_kg=jnp.zeros_like(qc0), q_c=q_c, T=T, N_c=N_c,
               hyd_kw=dict(q_r=jnp.where(cloudy, q_r, 0.0), q_s=sol, q_g=sol, q_i=0.1 * sol,
                           N_i=jnp.where(cloudy, 1.0e5, 0.0),
                           N_r=jnp.where(cloudy, q_r * rho / 6.5e-8, 0.0)))
    qb = {k: np.asarray(v) for k, v in out.qc_budget.items()}
    ncic = W.mg2_size_bounded_nc(q_c / cf, N_c / cf, rho)
    return (out, qb, np.asarray(cloudy), np.asarray(ncic / (q_c / cf)), np.asarray(N_c),
            np.asarray(q_c), T)


def test_cold_full_sum_and_graupel_under_the_number_cap(monkeypatch):
    _no_bergeron(monkeypatch)
    """(1) qcic = 1e-6 with rain: autoconversion + accretion + snow + graupel riming
    together set the number sink; (2) droplet number x 1e-6: the riming-dominated
    sink is capped by MG2's nc limiter at N_c/dt x omsm (user decision 7a)."""
    fx = _fx()
    dt = float(fx["dt"])
    out, qb, c, r, N_c, q_c, T = _cold(fx, 1.0e-6, 1.0e-5)
    mass = -(qb["autoconversion"] + qb["accretion"] + qb["riming_snow"] + qb["riming_graupel"])
    sinks = -sum(v for k, v in qb.items() if k not in ("condensation", "sedimentation"))
    ok = _top(c) & (N_c > 0) & (sinks * dt < 0.99 * q_c) & (mass * r * dt < 0.99 * N_c)
    assert ok.any() and (qb["autoconversion"][ok] < 0).all() and (qb["riming_graupel"][ok] < 0).all()
    want = _with_sedimentation(fx, out, jnp.asarray(q_c), jnp.asarray(N_c),
                               jnp.asarray(np.where(ok, -(mass * r), 0.0)), T)
    assert float(_rel(np.asarray(out.dN_c_dt)[ok], want[ok]).max()) <= 1e-12
    out, qb, c, r, N_c, q_c, T = _cold(fx, 1.0e-6, 1.0e-5, nc_scale=1.0e-6)
    mass = -(qb["autoconversion"] + qb["accretion"] + qb["riming_snow"] + qb["riming_graupel"])
    capd = _top(c) & (N_c > 0) & (mass * r * dt > 1.01 * N_c)
    assert capd.any() and (qb["riming_graupel"][capd] < 0).all()
    want = _with_sedimentation(fx, out, jnp.asarray(q_c), jnp.asarray(N_c),
                               jnp.asarray(np.where(capd, -N_c / dt * OMSM, 0.0)), T)
    assert float(_rel(np.asarray(out.dN_c_dt)[capd], want[capd]).max()) <= 1e-12


def test_npccn_guards():
    fx = _fx()
    g = {k: jnp.asarray(fx[f"col_{k}"]) for k in
         ("T", "q_v", "q_c", "cf", "q_r", "N_r", "p_full", "p_half", "rho", "nc_kg", "relvar")}
    dz = (g["p_half"][:, 1:] - g["p_half"][:, :-1]) / (C.g * g["rho"])
    z = jnp.zeros_like(g["T"])
    hyd = HydrometeorState(q_c=g["q_c"], q_r=g["q_r"], q_i=z, q_s=z, q_g=z,
                           N_c=g["nc_kg"] * g["rho"], N_r=g["N_r"], N_i=z, N_s=z)
    args = (g["T"], g["q_v"], hyd, g["p_full"], g["p_half"], g["rho"], dz, float(fx["dt"]))
    with pytest.raises(ValueError, match="npccn= with shape"):
        morrison_microphysics(*args, _cfg(), cloud_fraction=g["cf"], relvar=g["relvar"])
    with pytest.raises(ValueError, match="npccn= with shape"):
        morrison_microphysics(*args, _cfg(), cloud_fraction=g["cf"], relvar=g["relvar"],
                              npccn=jnp.asarray(0.0))
    with pytest.raises(ValueError, match="read only under"):
        morrison_microphysics(*args, _cfg()._replace(prognostic_droplet_number=False),
                              cloud_fraction=g["cf"], npccn=z)


def test_gradients_through_activation():
    fx = _fx()
    _, neg, _ = _mg2_masks(fx)
    keep = jnp.asarray(~neg)
    rho = jnp.asarray(fx["col_rho"])

    def f(npc, nc, qc):
        out = _run(fx, npccn_kg=npc, N_c=nc, q_c=qc)
        return jnp.sum(jnp.where(keep, out.dN_c_dt, 0.0)) * 1e-6 + jnp.sum(out.dq_r_dt)

    g = jax.grad(f, argnums=(0, 1, 2))(jnp.asarray(fx["col_npccn"]),
                                         jnp.asarray(fx["col_nc_kg"]) * rho,
                                         jnp.asarray(fx["col_q_c"]))
    for a in g:
        assert np.isfinite(np.asarray(a)).all()
    assert np.abs(np.asarray(g[0])[~neg]).max() > 0.0
