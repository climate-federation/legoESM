"""MorrisonConfig.prognostic_droplet_number, stage S10.3: MG2 cloud-droplet
sedimentation (mass, number, evaporation of the share falling into clear air,
surface flux) and the droplet-number clear, against the RUNNING MG2.

Fixture ``oracle_fixtures/mg2_prognostic_nc_sed.npz``: ``make_fixture.py
--prognostic-nc-sed`` through the probe build ``mg2_tap`` (build.sh: a copy of
micro_mg2_0.F90 of cam_cesm2_1_rel_60 with 8 added lines that only record
nctend after the activation re-apply, nctend after sedimentation (before
:2501) and the droplet sub-step count).  All columns are warm and ice-free, so
MG2's whole-call qctend, qcsedten, qvlat, tlat and surface fluxes contain no
process this port has not reached yet.  Bound 1e-10 relative, x64.
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

from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import TOL, _rel  # noqa: E402
from tests.atmosphere.microphysics.unit.test_morrison_prognostic_nc import _cfg  # noqa: E402
from tests.atmosphere.microphysics.unit import test_morrison_prognostic_nc_act as A  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/mg2_prognostic_nc_sed.npz"


def _fx():
    return dict(np.load(FIXTURE))


def _rel_flux(a, b, gross):
    """Relative error against max(|value|, gross): where the cloud-water limiter
    binds, the sedimenting state q_c + dq_c*dt = q_c*(1 - omsm) is a 1e-5
    residue of q_c, so its absolute round-off is eps*q_c, not eps*residue."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.maximum(np.maximum(np.abs(a), np.abs(b)), np.asarray(gross))
    return np.where(d > 0, np.abs(a - b) / np.where(d > 0, d, 1.0), 0.0)


def test_fixture_inputs():
    fx = _fx()
    assert np.allclose(fx["col_rho"], fx["col_p_full"] / (C.R_d * fx["col_T"]), rtol=1e-14)
    wet = (fx["col_q_c"] > 0) | (fx["col_q_r"] > 0)
    assert (fx["col_T"][wet] > C.T_freeze).all()          # warm water: no unported cold process


def _gross(fx, out):
    """Outgoing term at the START-of-step water, fc(post state) |q_c| / pdel."""
    dt = float(fx["dt"])
    rho = jnp.asarray(fx["col_rho"])
    q0 = jnp.asarray(fx["col_q_c"])
    q = q0 + (out.dq_c_dt - out.qc_budget["sedimentation"]) * dt
    fc, _ = W.mg2_droplet_fall_coefficients(
        q, jnp.asarray(fx["col_nc_kg"]), jnp.maximum(jnp.asarray(fx["col_cf"]), 1e-4),
        jnp.asarray(fx["col_T"]), rho, presence_top=W.MG2_QSMALL)
    gross = np.abs(np.asarray(q0 * fc)) / np.diff(fx["col_p_half"], axis=1)
    # only where MG2's cloud-water limiter binds (the residue cells); plain
    # relative error everywhere else
    return np.where(fx["ora_qcrat"] < 1.0, gross, 0.0)


def test_sedimentation_matches_mg2():
    fx = _fx()
    out = A._run(fx)
    dt = float(fx["dt"])
    rho = fx["col_rho"]
    qb = out.qc_budget
    nc = fx["col_nc_kg"]
    errs = {
        "qctend": _rel(out.dq_c_dt, fx["ora_qctend"]),
        "qcsedten": _rel_flux(qb["sedimentation"], fx["ora_qcsedten"], _gross(fx, out)),
        "qvlat": _rel(out.dq_v_dt, fx["ora_qvlat"]),
        "tlat": _rel(np.asarray(out.dT_dt) * C.c_pd, fx["ora_tlat"]),
        "precip": _rel(out.precipitation,
                       fx["ora_rflx_bot"][:, -1] + fx["ora_lflx_bot"][:, -1]),
    }
    clear = fx["col_q_c"] + fx["ora_qctend"] * dt < W.MG2_QSMALL
    dN = np.asarray(out.dN_c_dt) / rho
    # whole-call nctend: after sedimentation only the S10.4 droplet re-adjust and
    # the number clear write it in these warm columns
    errs["nctend"] = _rel(dN, fx["ora_nctend"])
    worst = {k: float(v.max()) for k, v in errs.items()}
    print("S10.3 max rel err", worst, "clear", int(clear.sum()))
    assert max(worst.values()) <= TOL, worst
    # coverage, on the MG2 side
    names = list(fx["col_names"])
    thin = names.index("thin")
    assert fx["ora_nstepc"][thin, 0] > 1                       # sub-stepping active
    cf = fx["col_cf"][thin]
    shrink = np.r_[False, (cf[1:] < cf[:-1]) & (fx["col_q_c"][thin][:-1] > 0)]
    assert (fx["ora_qvlat"][thin][shrink] > 0).any()           # inflow evaporates (no rain here)
    assert (fx["col_q_r"][thin] == 0).all()
    clr = names.index("clearnc")
    assert (clear[clr] & (nc[clr] > 0)).any()                  # number clear with droplets left
    assert np.abs(fx["ora_qcsedten"]).max() > 0
    # the sub-step count is reported with the other species'
    assert int(np.asarray(out.sed_substeps_required)[thin]) >= int(fx["ora_nstepc"][thin, 0])


def test_column_water_closes():
    fx = _fx()
    out = A._run(fx)
    pdel = np.diff(fx["col_p_half"], axis=1)
    col = ((np.asarray(out.dq_v_dt) + np.asarray(out.dq_c_dt) + np.asarray(out.dq_r_dt))
           * pdel / C.g).sum(axis=1) + np.asarray(out.precipitation)
    scale = (np.abs(np.asarray(out.dq_c_dt)) * pdel / C.g).sum(axis=1).max()
    assert np.abs(col).max() <= 1e-12 * scale


def test_gradients_through_sedimentation():
    fx = _fx()
    rho = jnp.asarray(fx["col_rho"])
    thin = list(fx["col_names"]).index("thin")

    def f(nc, qc):
        out = A._run(fx, N_c=nc, q_c=qc)
        return jnp.sum(out.qc_budget["sedimentation"][thin]) * 1e6 + jnp.sum(out.dq_v_dt) * 1e6

    g = jax.grad(f, argnums=(0, 1))(jnp.asarray(fx["col_nc_kg"]) * rho, jnp.asarray(fx["col_q_c"]))
    for a in g:
        assert np.isfinite(np.asarray(a)).all()
    assert np.abs(np.asarray(g[0])[thin]).max() > 0 and np.abs(np.asarray(g[1])[thin]).max() > 0


def test_switch_cap_is_the_shared_one():
    cfg = _cfg()
    assert cfg.sed_cfl_substeps and int(cfg.sed_cfl_substeps_max) >= int(_fx()["ora_nstepc"].max())


def _thin(fx):
    """The fixture restricted to the thin-layer column (MG2 nstep 17)."""
    i = list(fx["col_names"]).index("thin")
    return {k: (v[i:i + 1] if k.startswith(("col_", "ora_")) else v) for k, v in fx.items()}


def test_cap_overflow_is_reported_and_strict_aborts():
    """Autoconversion off, so the step makes no rain (whose 4 m-layer count, 73,
    would otherwise set the maximum) and only the droplets can exceed the cap."""
    fx = _thin(_fx())
    n_mg2 = int(fx["ora_nstepc"][0, 0])
    base = _cfg()._replace(autocon_fact=0.0)
    out = A._run(fx, cfg=base)
    assert int(np.asarray(out.sed_substeps_required)[0]) == n_mg2   # droplet count, as MG2
    out = A._run(fx, cfg=base._replace(sed_cfl_substeps_max=4))
    assert int(np.asarray(out.sed_substeps_required)[0]) == n_mg2 > 4
    with pytest.raises(Exception, match="cloud-droplet sedimentation"):
        A._run(fx, cfg=base._replace(sed_cfl_substeps_max=4, sed_cfl_substeps_strict=True))


def test_substeps_off_is_one_pass():
    """sed_cfl_substeps=False: one pass (user decision 2026-10-20), checked against
    an independent single-pass evaluation of MG2's loop body on the same state."""
    fx = _thin(_fx())
    out = A._run(fx, cfg=_cfg()._replace(sed_cfl_substeps=False))
    assert out.sed_substeps_required is None
    dt = float(fx["dt"])
    rho = jnp.asarray(fx["col_rho"])
    lcl = np.maximum(fx["col_cf"], 1e-4)
    sed = np.asarray(out.qc_budget["sedimentation"])
    q = jnp.asarray(fx["col_q_c"]) + (out.dq_c_dt - out.qc_budget["sedimentation"]) * dt
    # droplet number before sedimentation: no activation, no rain, so only the
    # (tiny) autoconversion number sink acts; recover it from the published
    # autoconversion with the oracle-tested size reset
    qcic = np.asarray(fx["col_q_c"]) / lcl
    r_ic = np.asarray(W.mg2_size_bounded_nc(jnp.asarray(np.minimum(qcic, 5e-3)),
                                            jnp.asarray(fx["col_nc_kg"] / lcl) * rho, rho))
    r = np.where(qcic >= W.MG2_QSMALL, r_ic / np.where(qcic > 0, np.minimum(qcic, 5e-3), 1.0), 0.0)
    n_pre = fx["col_nc_kg"] + np.asarray(out.qc_budget["autoconversion"]) * r / fx["col_rho"] * dt
    fc, _ = W.mg2_droplet_fall_coefficients(q, jnp.asarray(n_pre), jnp.asarray(lcl),
                                            jnp.asarray(fx["col_T"]), rho, presence_top=W.MG2_QSMALL)
    F = np.asarray(fc * q)
    pdel = np.diff(fx["col_p_half"], axis=1)
    d = np.c_[np.zeros((1, 1)), np.minimum(lcl[:, 1:] / lcl[:, :-1], 1.0)]
    Fup = np.c_[np.zeros((1, 1)), F[:, :-1]]
    want = -(F - d * Fup) / pdel
    assert float(_rel(sed, want).max()) <= 1e-12
    # and it is not the sub-stepped answer
    full = np.asarray(A._run(fx).qc_budget["sedimentation"])
    assert float(_rel(sed, full).max()) > 1e-3


def test_cloud_water_exactly_at_qsmall_is_finite():
    """MG2 gates the fall speed at qcic >= qsmall and the size distribution at
    qcic > qsmall; exactly at qsmall the slope is 0 (codex review)."""
    q = jnp.asarray([[W.MG2_QSMALL, 1.0e-4]])
    fc, fnc = W.mg2_droplet_fall_coefficients(q, jnp.asarray([[1.0e6, 1.0e8]]), jnp.ones((1, 2)),
                                              jnp.full((1, 2), 280.0), jnp.ones((1, 2)), presence_top=W.MG2_QSMALL)
    assert np.isfinite(np.asarray(fc)).all() and np.isfinite(np.asarray(fnc)).all()
    assert float(fc[0, 0]) == 0.0 and float(fc[0, 1]) > 0.0
    out = W.mg2_cloud_sedimentation(q, jnp.asarray([[1.0e6, 1.0e8]]), jnp.ones((1, 2)),
                                    jnp.full((1, 2), 280.0), jnp.ones((1, 2)),
                                    jnp.asarray([[80000.0, 80500.0, 81000.0]]), 600.0, 16, presence_top=W.MG2_QSMALL)
    for a in out[:5]:
        assert np.isfinite(np.asarray(a)).all()
