#!/usr/bin/env python
"""ONE momentum vertical-diffusion step (dyn_zdf), OURS vs NEMO, from NEMO's own
stage-3 operands on the equatorial strip (2S-2N, 225.5-255.5E).

NEMO RK3 stage 3 (stprk3_stg.F90:218-222: Kbb=N, Kmm=N+1/2, Kaa=N+1, rDt=rn_Dt)
calls dyn_zdf (stprk3_stg.F90:430) which, with ln_dynadv_vec:
  dynzdf.F90:121      ua = (ub + rDt*rhs3) * umask          (rhs3 = stage-3 3-D RHS)
  dynzdf.F90:150      ua = ua - uu_b(Kaa)                     (ln_drgimp .AND. ln_dynspg_ts)
  dynzdf.F90:182-195  tridiagonal: zwi_k = -rDt/2*(avm_i+avm_i+1)_k / (e3u(Kaa)_k e3uw(Kmm)_k),
                      zws_k likewise at w-level k+1, zwd = 1 - zwi - zws; zwi_1 = 0
  dynzdf.F90:303-305  row 1 RHS += rDt*utauU/(e3u(Kaa)_1 rho0)    (key_RK3)
then stprk3_stg.F90:440-444 re-imposes the depth mean: ua += uu_b(Kaa) - SUM(e3u_0 ua) r1_hu_0.

Operands (all NEMO, step kt+1 of a run ending at kt+1 with ln_dyn_trd):
  ub      = strip uoce, record kt-1   (end of step kt  = Kbb of step kt+1)
  rhs3    = SUM of the stage-3 instant trends utrd_{hpg,pvo,rvo,keg,zad,ldf}[+spg]
            (record kt; the instant value is the last trd_dyn write = stage 3)
  avm     = strip avm, record kt (zdf_phy at step entry)
  utauU   = 0.5*(utau_oce_i + utau_oce_i+1) (sbcblk.F90:917, sbcmod.F90:560), record kt
  uu_b(Kaa) = SUM(e3u_0 un(kt+1)) r1_hu_0  (inferred from the answer: tests shape, not
            barotropic evolution; depth-uniform rhs errors are invisible)
  e3u(Kaa) = e3u_0 (1+r3u(ssh N+1)); e3uw(Kmm) = e3uw_0 (1+r3u(mean ssh N, N+1)) APPROX (domqco.F90:166)
NEMO's strip output is float32 (u quantum ~1e-8 m/s): GATE 1 is also scored on
vertical differences, which cancel the depth-uniform part.

GATE 1 (instrument): the NumPy transcription, fed the TREND-FREE RHS
x = un(kt+1) - dt*utrd_zdf - uu_b + stress row (exact by dynzdf.F90:121,527 and
stprk3_stg.F90:440 because the dynzdf rows sum to 1), returns un(kt+1).  This uses
no RHS reconstruction; the summed-trend rhs3 is reported as secondary only.  GATE 2 (claim): our production solver
``implicit_vertical_diffusion_ocean`` on the IDENTICAL solve input and
coefficients reproduces the same; GATE 2b repeats it with the production interface spacing
0.5*(dz_k+dz_k+1) instead of e3uw.  Aimp ``wi`` is not in the matrix (production
shared_thomas omits it); GATE 1 tests that omission against NEMO.  PLANT: avm x(1+plant) at one interior
interface and at the surface row must each move our output beyond the
tolerance.  Numbers only -- no verdict.  Bottom-friction rows are not modelled
(deep columns only; their influence on the top --kmax levels is printed).

Usage (compute node only):
  python onestep_momentum_twin.py --run-dir <RUN_DT150_TWIN_k192_DYN> --kt 191 \
      --domain-cfg <domain_cfg.nc>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

STEP_DT = 150.0
RHO0 = 1026.0          # const-ok: NEMO rho0 (phycst.F90), the oracle's literal, not ours
TRENDS = ("hpg", "pvo", "rvo", "keg", "zad", "ldf")
J0, I0, NJ, NI = 180, 152, 13, 31   # strip zoom in NEMO output index space (domain_def eqs*)


def _strip(run_dir: Path, grid: str):
    import netCDF4 as nc
    f = sorted(run_dir.glob(f"ORCA1_1ts_*_eqs1ts_{grid}.nc"))
    if len(f) != 1:
        raise SystemExit(f"need exactly one eqs1ts_{grid} file in {run_dir}, got {f}")
    return nc.Dataset(f[0])


def _rec(d, var, rec):
    a = np.ma.filled(d[var][rec], np.nan).astype(np.float64)
    return np.where(np.abs(a) > 1e10, np.nan, a)


def _check_time(d, step):
    t = np.asarray(d["time_instant"][:], dtype=np.float64)
    if abs((t[0] % 86400.0) - STEP_DT) > 1e-6:
        raise SystemExit("strip record 0 is not the end of step 1")
    if abs(t[step - 1] - t[0] - (step - 1) * STEP_DT) > 1e-6:
        raise SystemExit(f"strip record {step - 1} is not step {step}")


def nemo_solve(x, avm_u, e3u_a, e3uw_m, dt):
    """dynzdf.F90:182-195 matrix + :297-315 recurrences, one column, levels 0..n-1.
    avm_u[k] lives on w-level k (k=0 surface, unused); zero at/below the bottom."""
    n = x.size
    zwi = np.zeros(n); zws = np.zeros(n); zwd = np.ones(n)
    for k in range(n):
        if k > 0:
            zwi[k] = -dt * avm_u[k] / (e3u_a[k] * e3uw_m[k])
        if k + 1 < n:
            zws[k] = -dt * avm_u[k + 1] / (e3u_a[k] * e3uw_m[k + 1])
        zwd[k] = 1.0 - zwi[k] - zws[k]
    for k in range(1, n):
        zwd[k] = zwd[k] - zwi[k] * zws[k - 1] / zwd[k - 1]
    y = x.copy()
    for k in range(1, n):
        y[k] = y[k] - zwi[k] / zwd[k - 1] * y[k - 1]
    y[n - 1] = y[n - 1] / zwd[n - 1]
    for k in range(n - 2, -1, -1):
        y[k] = (y[k] - zws[k] * y[k + 1]) / zwd[k]
    return y


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--kt", type=int, required=True, help="Kbb step; the twin step is kt+1")
    p.add_argument("--domain-cfg", required=True)
    p.add_argument("--kmax", type=int, default=30)
    p.add_argument("--min-levels", type=int, default=45)
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--plant", type=float, default=0.10)
    p.add_argument("--plant-k", type=int, default=9, help="w-level of the interior plant (9 = 12.8 m)")
    p.add_argument("--with-spg", action="store_true", help="add utrd_spg to rhs3 (closure arm)")
    a = p.parse_args(argv)

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import netCDF4 as nc
    from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
        implicit_vertical_diffusion_ocean)

    st = a.kt + 1
    U, W, D2 = _strip(a.run_dir, "U"), _strip(a.run_dir, "W"), _strip(a.run_dir, "2D")
    for d in (U, W, D2):
        _check_time(d, st)
    missing = [f"utrd_{t}" for t in TRENDS if f"utrd_{t}" not in U.variables]
    if missing:
        raise SystemExit(f"strip U file lacks {missing}: run with ln_dyn_trd + utrd_* fields")
    nrec = U["uoce"].shape[0]
    print(f"[strip] records {nrec} (expect >= {st}); utrd records {U['utrd_zdf'].shape[0]}")
    ub = _rec(U, "uoce", a.kt - 1)              # (z, y, x)
    un1 = _rec(U, "uoce", st - 1)
    rhs = sum(_rec(U, f"utrd_{t}", st - 1) for t in TRENDS)
    if a.with_spg:
        rhs = rhs + _rec(U, "utrd_spg", st - 1)
    trd_zdf = _rec(U, "utrd_zdf", st - 1)
    avm = _rec(W, "avm", st - 1)                # T-point w-levels (z, y, x)

    dc = nc.Dataset(a.domain_cfg)
    sl = (slice(J0, J0 + NJ), slice(I0, I0 + NI + 1))     # +1 column for the i+1 neighbour
    g = lambda v: np.asarray(dc[v][0], dtype=np.float64)
    e3u0 = g("e3u_0")[:, sl[0], sl[1]][..., :NI]
    e3uw0 = g("e3uw_0")[:, sl[0], sl[1]][..., :NI]
    e1e2t = (g("e1t") * g("e2t"))[sl]
    e1e2u = (g("e1u") * g("e2u"))[sl][:, :NI]
    blev = np.asarray(dc["bottom_level"][0])[sl].astype(int)
    nz = e3u0.shape[0]
    kidx = np.arange(nz)[:, None, None]
    tmask = (kidx < blev[None]).astype(float)
    umask = tmask[..., :NI] * tmask[..., 1:]
    hu0 = np.sum(e3u0 * umask, axis=0)
    r1_hu0 = np.where(hu0 > 0, 1.0 / np.where(hu0 > 0, hu0, 1.0), 0.0)
    # qco (domqco.F90:166): r3u = 0.5*(e1e2t_i ssh_i + e1e2t_i+1 ssh_i+1) r1_hu_0 r1_e1e2u.
    # Kaa = ssh at end of step st; Kmm (stage-2 ssh, N+1/2) is not dumped -> APPROX as
    # the mean of ssh at N and N+1.
    def r3u(ssh):
        r = np.full((NJ, NI), np.nan)
        r[:, :NI - 1] = (0.5 * (e1e2t[:, :NI - 1] * ssh[:, :NI - 1] + e1e2t[:, 1:NI] * ssh[:, 1:NI])
                         * r1_hu0[:, :NI - 1] / e1e2u[:, :NI - 1])
        return r
    ssh_a, ssh_b = _rec(D2, "ssh", st - 1), _rec(D2, "ssh", a.kt - 1)
    r3u_a, r3u_m = r3u(ssh_a), r3u(0.5 * (ssh_a + ssh_b))
    e3u_a = e3u0 * (1.0 + r3u_a[None])
    e3uw_m = e3uw0 * (1.0 + r3u_m[None])
    print(f"[geom] qco r3u(Kaa) |max| {np.nanmax(np.abs(r3u_a)):.2e}; Kmm = mean(ssh N, N+1) APPROX")

    # avm at U faces (w-levels); columns are sliced to their n wet levels, so
    # interfaces 1..n-1 are exactly NEMO's wumask=1 set
    avm_u = 0.5 * (avm[:, :, :NI] + avm[:, :, 1:NI + 1]) if avm.shape[2] > NI else None
    if avm_u is None:
        # the strip has exactly NI T columns: last U column lacks its east neighbour
        avm_u = np.full((avm.shape[0], NJ, NI), np.nan)
        avm_u[:, :, :NI - 1] = 0.5 * (avm[:, :, :NI - 1] + avm[:, :, 1:NI])
    avm_u = avm_u[:nz]

    # Under key_RK3 "utau" is only iom_put in the MLF filter (dynatf_qco.F90:256-259);
    # the open-ocean T-point stress is utau_oce (sbcblk.F90:917).  utauU = 0.5*(i + i+1)
    # (sbcmod.F90:560).  Consistency: strip taum (final, after sbcmod) vs |tau_oce|.
    tu, tv, tm = (_rec(D2, v, st - 1) for v in ("utau_oce", "vtau_oce", "taum"))
    print(f"[stress] max|taum - |tau_oce|| / taum = {np.nanmax(np.abs(tm - np.hypot(tu, tv)) / tm):.2e}")
    utauU = np.full((NJ, NI), np.nan)
    utauU[:, :NI - 1] = 0.5 * (tu[:, :NI - 1] + tu[:, 1:NI])

    uub = np.sum(e3u0 * np.nan_to_num(un1) * umask, axis=0) * r1_hu0

    cols = [(j, i) for j in range(NJ) for i in range(NI - 1)
            if blev[j, i] >= a.min_levels and blev[j, i + 1] >= a.min_levels
            and np.isfinite(utauU[j, i]) and np.isfinite(r3u_a[j, i])]
    bad = [(j, i) for (j, i) in cols if not all(
        np.all(np.isfinite(f[:int(np.sum(umask[:, j, i])), j, i])) for f in (ub, un1, rhs, trd_zdf))
        or not np.all(np.isfinite(avm_u[1:int(np.sum(umask[:, j, i])), j, i]))]
    if bad:
        raise SystemExit(f"NaN inside {len(bad)} wet columns, e.g. {bad[:3]} -- operands not trusted")
    print(f"[cols] {len(cols)} deep wet U columns (>= {a.min_levels} levels both sides)")
    if len(cols) < 20:
        raise SystemExit("too few columns")

    def build(j, i, avm_scale=None, tau_scale=1.0, summed=False):
        # Default (trend-free) RHS: ua_bc - ub = dt*(rhs3 + utrd_zdf) (dynzdf.F90:121,527),
        # so x = ua_bc - dt*utrd_zdf - uu_b + stress row; ua_bc = un1 + const and the
        # dynzdf rows sum to 1, so solve(x) must return un1 up to that const, which
        # finish() removes.  summed=True rebuilds rhs3 from the trend sum instead.
        n = int(np.sum(umask[:, j, i]))
        if summed:
            x = (ub[:n, j, i] + STEP_DT * rhs[:n, j, i]) - uub[j, i]
        else:
            x = (un1[:n, j, i] - STEP_DT * trd_zdf[:n, j, i]) - uub[j, i]
        x[0] += tau_scale * STEP_DT * utauU[j, i] / (e3u_a[0, j, i] * RHO0)
        av = avm_u[:n, j, i].copy()
        av[0] = 0.0
        if avm_scale is not None:
            av[avm_scale[0]] *= avm_scale[1]
        return n, x, av

    def finish(j, i, n, y):
        zub = uub[j, i] - np.sum(e3u0[:n, j, i] * y) * r1_hu0[j, i]
        return y + zub

    err_nemo, err_ours, chg, err_prod, full_nemo, full_ours, raw_ours, shear = [], [], [], [], [], [], [], []
    for (j, i) in cols:
        n, x, av = build(j, i)
        yn_raw = nemo_solve(x, av, e3u_a[:n, j, i], e3uw_m[:n, j, i], STEP_DT)
        yn = finish(j, i, n, yn_raw)
        yo_raw = np.asarray(implicit_vertical_diffusion_ocean(
            jnp.asarray(x), jnp.asarray(av[1:]), jnp.asarray(e3u_a[:n, j, i]),
            jnp.asarray(e3uw_m[1:n, j, i]), STEP_DT))
        raw_ours.append(np.max(np.abs(yo_raw - yn_raw)))
        yo = finish(j, i, n, yo_raw)
        shear.append(np.diff(yn[:a.kmax]) - np.diff(un1[:a.kmax, j, i]))
        e = e3u_a[:n, j, i]
        yb = finish(j, i, n, np.asarray(implicit_vertical_diffusion_ocean(
            jnp.asarray(x), jnp.asarray(av[1:]), jnp.asarray(e),
            jnp.asarray(0.5 * (e[:-1] + e[1:])), STEP_DT)))
        err_prod.append(yb[:a.kmax] - yn[:a.kmax])
        err_nemo.append(yn[:a.kmax] - un1[:a.kmax, j, i])
        full_nemo.append(np.max(np.abs(yn - un1[:n, j, i])))
        full_ours.append(np.max(np.abs(yo - yn)))
        err_ours.append(yo[:a.kmax] - yn[:a.kmax])
        chg.append(un1[:a.kmax, j, i] - ub[:a.kmax, j, i])
    err_nemo, err_ours, chg, err_prod, shear = (np.array(v) for v in (err_nemo, err_ours, chg, err_prod, shear))
    print(f"[dtype] solver output {yo.dtype}, inputs float64")
    print(f"GATE1 transcription vs NEMO un({st}) top {a.kmax}: max|d| {np.abs(err_nemo).max():.3e} m/s, "
          f"median|d| {np.median(np.abs(err_nemo)):.3e}; NEMO |du| per step median {np.median(np.abs(chg)):.3e}")
    print(f"GATE1-shear (APPROX: no Aimp wi, no bottom drag; depth-uniform offsets cancel) "
          f"max|d(du)| {np.abs(shear).max():.3e} m/s, median {np.median(np.abs(shear)):.3e}; "
          f"NEMO |du_k - du_k+1| median {np.median(np.abs(np.diff(chg, axis=1))):.3e}")
    print(f"GATE2-raw ours vs transcription before mean re-imposition, full column: max|d| {max(raw_ours):.3e} m/s")
    print(f"GATE2 ours vs transcription top {a.kmax}: max|d| {np.abs(err_ours).max():.3e} m/s, "
          f"median|d| {np.median(np.abs(err_ours)):.3e}  (tol {a.tol:g})")
    print(f"[full depth] GATE1 max|d| {max(full_nemo):.3e} m/s; GATE2 max|d| {max(full_ours):.3e} m/s "
          "(depth-uniform rhs errors are cancelled by the uu_b re-imposition: invisible here, irrelevant to shear)")
    print(f"GATE2b production spacing 0.5(dz_k+dz_k+1) vs transcription: max|d| {np.abs(err_prod).max():.3e} m/s, "
          f"median|d| {np.median(np.abs(err_prod)):.3e}")
    zt = np.cumsum(e3u0[:a.kmax, cols[0][0], cols[0][1]]) - 0.5 * e3u0[:a.kmax, cols[0][0], cols[0][1]]
    print(" lev  z[m] | GATE1 median|d|  max|d| | GATE2 max|d| | GATE2b max|d|")
    for k in range(a.kmax):
        print(f" {k:3d} {zt[k]:5.1f} | {np.median(np.abs(err_nemo[:, k])):.2e} {np.abs(err_nemo[:, k]).max():.2e} | "
              f"{np.abs(err_ours[:, k]).max():.2e} | {np.abs(err_prod[:, k]).max():.2e}")
    # secondary: the summed-trend RHS (utrd_hpg carries isolated spikes uoce does not show)
    sm = []
    for (j, i) in cols:
        n, x, av = build(j, i, summed=True)
        sm.append(np.max(np.abs(finish(j, i, n, nemo_solve(x, av, e3u_a[:n, j, i], e3uw_m[:n, j, i], STEP_DT))[:a.kmax]
                                - un1[:a.kmax, j, i])))
    sm = np.array(sm)
    print(f"[secondary] summed-trend RHS vs un({st}) top {a.kmax}: median col max|d| {np.median(sm):.3e}; "
          f"cols with max|d| < 1e-6: {np.mean(sm < 1e-6):.3f}")
    # trend closure: NEMO utrd_zdf vs (ua_bc - ub)/dt - rhs3, with ua_bc from the transcription
    tz = []
    for (j, i) in cols:
        n, x, av = build(j, i, summed=True)
        ybc = nemo_solve(x, av, e3u_a[:n, j, i], e3uw_m[:n, j, i], STEP_DT)
        tz.append(((ybc - ub[:n, j, i]) / STEP_DT - rhs[:n, j, i])[:a.kmax] - trd_zdf[:a.kmax, j, i])
    tz = np.array(tz)
    print(f"[closure] utrd_zdf residual top {a.kmax}: median|r| {np.median(np.abs(tz)):.3e} m/s2, "
          f"max {np.abs(tz).max():.3e}; NEMO |utrd_zdf| median {np.median(np.abs(trd_zdf[:a.kmax])):.3e}")
    # plants: our output must move beyond tol when avm is perturbed
    rc = 0
    for lab, kk in (("interior avm", a.plant_k), ("top-interface avm", 1), ("deep avm", 35), ("surface stress", None)):
        moved = []
        for (j, i) in cols:
            n, x, av = build(j, i)
            if kk is None:
                _, xp, avp = build(j, i, tau_scale=1.0 + a.plant)
            else:
                _, xp, avp = build(j, i, avm_scale=(kk, 1.0 + a.plant))
            y0 = implicit_vertical_diffusion_ocean(jnp.asarray(x), jnp.asarray(av[1:]),
                                                   jnp.asarray(e3u_a[:n, j, i]), jnp.asarray(e3uw_m[1:n, j, i]), STEP_DT)
            y1 = implicit_vertical_diffusion_ocean(jnp.asarray(xp), jnp.asarray(avp[1:]),
                                                   jnp.asarray(e3u_a[:n, j, i]), jnp.asarray(e3uw_m[1:n, j, i]), STEP_DT)
            moved.append(float(np.max(np.abs(finish(j, i, n, np.asarray(y1)) - finish(j, i, n, np.asarray(y0))))))
        frac = float(np.mean(np.array(moved) > a.tol))
        print(f"[plant {lab} (w-level {kk}) x{1 + a.plant:g}] fraction of columns moved > tol: {frac:.3f}; "
              f"median move {np.median(moved):.2e}")
        if frac < 0.5 and lab != "deep avm":   # deep avm ~background: weak by physics, report only
            rc = 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
