"""Shallow-cumulus diagnostic: production CLUBB in the SCM vs our own LES.

Reuses the LES-SCM tuning harness (case build, surface, forcing, window) and
records what that harness does not: liquid water, cloud top, and CLUBB's
turbulent fluxes w'rt' and w'thl' over the LES analysis window. Arms differ in
ONE of: nlev, dt, surface-flux-to-closure (printed per arm).

Usage (compute node, x64):
  python scripts/tmp/diag_shallow_clubb_scm.py --case bomex \
      --les-dir results/les_ref/bomex --nlev 64 --dt 60 --out out.npz
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
FAC = float(os.environ.get("MF_FAC", "10"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "harness", REPO / "scripts/run/run_scm_les_turbulence_tuning.py")
H = importlib.util.module_from_spec(_spec)
sys.modules["harness"] = H
_spec.loader.exec_module(H)
# run_bomex_les.py labels BOMEX output "bomex_cu" (its --case-label default);
# the reference guard's alias table lacks it. Same deck (gSAM CASES/BOMEX).
import legoesm.training.les_reference as _lr  # noqa: E402
_lr._CASE_ALIASES["bomex"] = frozenset({"bomex", "bomex_cu"})
# Since b024f1c706 prognostic CLUBB reads grid.lat; SCMGrid names it grid_lat,
# so the SCM crashes. Alias for this probe only (same value).
from legoesm.atmosphere.forcing.scm.scm import SCMGrid  # noqa: E402
SCMGrid.latCell = property(lambda g: g.grid_lat)

MOM = ("rtm", "thlm", "um", "vm", "wp2", "wp3", "up2", "vp2", "wprtp",
       "wpthlp", "upwp", "vpwp", "rtp2", "thlp2", "rtpthlp")


_G_MF_LOG = []


def rollout(cfg, arm):
    MF_LOG = _G_MF_LOG
    scm = H._create_scm(cfg, arm.case, arm.dt)
    step_fn, tend_fn, dt = scm._step_fn, scm._tend_fn, arm.dt
    print("tracers:", sorted(scm.state.tracers.keys()), flush=True)
    liq = [k for k in ("q_c", "q_l", "q_cl") if k in scm.state.tracers][0]
    if os.environ.get("STEPWISE"):
        import time
        f = jax.jit(lambda st, ps, t: step_fn(st, ps, tend_fn, dt, t))
        st, ps = scm.state, scm.phys_state
        for k in range(int(os.environ["STEPWISE"])):
            t0 = time.time()
            st, ps = f(st, ps, jnp.float64(k * dt))
            T = np.asarray(st.T.data); q = np.asarray(st.tracers["q_v"].data)
            print(f"STEP {k} {time.time()-t0:.2f}s T[{T.min():.2f},{T.max():.2f}] "
                  f"qv_max {q.max():.4g} finite={np.all(np.isfinite(T)) and np.all(np.isfinite(q))} "
                  f"mfcf_max {float(np.nanmax(np.asarray(ps.mf_cloudfrac))):.3g}", flush=True)
            _z = np.asarray(ps.mf_ztopm1)
            print(f"  ztopm1 finite={np.all(np.isfinite(_z))} range [{np.nanmin(_z):.1f},{np.nanmax(_z):.1f}] "
                  f"moments finite={np.all(np.isfinite(np.asarray(ps.clubb_moments)))} "
                  f"mf_qc finite={np.all(np.isfinite(np.asarray(ps.mf_qc)))} "
                  f"MFLOG last finite={(np.all(np.isfinite(MF_LOG[-1][1])) and np.all(np.isfinite(MF_LOG[-1][2]))) if MF_LOG else None} "
                  f"MFLOG nonfinite calls={sum(1 for x in MF_LOG if not np.all(np.isfinite(x[1])))}/{len(MF_LOG)}", flush=True)
            if not np.all(np.isfinite(T)):
                np.savez("/burg/glab/users/pg2328/enso_logs/rico_nan_state.npz", k=k)
                break
        raise SystemExit(0)

    def body(carry, k):
        state, ps = carry
        t = k.astype(jnp.float64) * dt
        ns, nps = step_fn(state, ps, tend_fn, dt, t)
        _z = jnp.zeros_like(ns.T.data[0, 0, 0])
        _cf = nps.cloud_fraction if getattr(nps, "cloud_fraction", None) is not None else None
        out = (ns.T.data[0, 0, 0], ns.tracers["q_v"].data[0, 0, 0],
               ns.tracers[liq].data[0, 0, 0], nps.clubb_moments[0],
               _z if nps.mf_cloudfrac is None else jnp.reshape(nps.mf_cloudfrac, (-1,)),
               _z if nps.mf_qc is None else jnp.reshape(nps.mf_qc, (-1,)),
               _z if _cf is None else jnp.reshape(_cf, (-1,)),
               _z if nps.mf_qc is None else jnp.reshape(nps.mf_cloudfrac * nps.mf_qc, (-1,)))
        return (ns, nps), out

    nsteps = int(round(arm.hours * 3600.0 / dt))
    print("TRACE nsteps", nsteps, flush=True)
    nan = int(round(arm.analysis_hours * 3600.0 / dt))
    _, (T, qv, ql, mom, mfcf, mfqc, ccf, mfaq) = jax.jit(
        lambda c: lax.scan(body, c, jnp.arange(nsteps)))((scm.state, scm.phys_state))
    w = slice(nsteps - nan, nsteps)
    if os.environ.get("MF_STEPS"):
        np.savez(os.environ["MF_STEPS"], mfcf=np.asarray(mfcf[w]), mfqc=np.asarray(mfqc[w]),
                 ccf=np.asarray(ccf[w]), mfaq=np.asarray(mfaq[w]))
    return {k: np.asarray(v[w]).mean(0) for k, v in
            dict(T=T, qv=qv, ql=ql, mom=mom, mfcf=mfcf, mfqc=mfqc, ccf=ccf, mfaq=mfaq).items()}, np.asarray(ql)


def les_window(les_dir, hours, analysis_hours):
    acc = {}
    for f in sorted(glob.glob(str(Path(les_dir) / "profiles/prof_*.npz"))):
        d = np.load(f)
        th = float(d["t_hours"])
        if hours - analysis_hours - 1e-6 <= th <= hours + 1e-6:
            for k in ("z", "qc", "cloud_frac", "wqt", "wtheta", "qv", "theta"):
                acc.setdefault(k, []).append(np.asarray(d[k], float))
    return {k: np.mean(v, 0) for k, v in acc.items()}, len(acc.get("z", []))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--les-dir", required=True)
    p.add_argument("--nlev", type=int, default=64)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--surface-flux-to-closure", action="store_true")
    p.add_argument("--prod-liquid", action="store_true",
                   help="production CLUBB liquid coupling: liquid_partition + "
                        "liquid_from_closure + cld_macmic_num_steps=3 (shim)")
    p.add_argument("--lib-params", action="store_true",
                   help="CLUBB library defaults for every coefficient CAM6 overrides")
    p.add_argument("--clubb-mf", action="store_true",
                   help="CAM7 CLUBB-MF on; its CFL-limited plume flux is recorded "
                        "each call and added to CLUBB's w'rt'/w'thl'")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    hargs = ["--case", a.case, "--les-dir", a.les_dir, "--schemes", "clubb",
             "--nlev", str(a.nlev), "--skip-tuning"]
    if a.dt is not None:
        hargs += ["--dt", str(a.dt)]
    if a.surface_flux_to_closure:
        hargs += ["--surface-flux-to-closure"]
    args = H.parse_args(hargs)
    args.scored_override = None
    arm = H._build_arms(args, [a.case], {a.case: Path(a.les_dir)})[0]
    cfg = H._arm_config("clubb", arm, args)
    if a.prod_liquid:
        import legoesm.atmosphere.forcing.scm.scm as _scm
        from legoesm.atmosphere.physics.microphysics.config import (
            apply_microphysics_experiment_flags)
        # SingleColumnModel exposes no macmic count; production runs 3.
        # and only the MPAS lane routes CLUBB's liquid tendency.
        _orig = _scm.make_physics
        _scm.make_physics = lambda c, **kw: _orig(
            c, **{**kw, "model_type": "mpas", "cld_macmic_num_steps": 3})
        cfg = cfg._replace(
            turbulence=cfg.turbulence._replace(liquid_partition=True),
            microphysics=cfg.microphysics._replace(
                morrison=apply_microphysics_experiment_flags(
                    cfg.microphysics.morrison, cfg.microphysics.scheme,
                    liquid_from_closure=True)))
    if a.lib_params:
        lib = dict(C2rt=2.0, C2thl=2.0, C2rtthl=2.0, C4=2.0, C_uu_shr=0.4,
                   C6rt=2.0, C6rtb=2.0, C6thl=2.0, C6thlb=2.0, C8=0.5, C8b=0.02,
                   C11=0.4, C11b=0.4, C14=1.0, C_wp3_pr_turb=0.0,
                   C_wp2_splat=2.0, c_K1=0.2, c_K2=0.025, nu2=1.0, c_K8=5.0,
                   c_K9=0.1, nu9=10.0, c_K10=1.0, c_K10h=1.0, gamma_coef=0.25,
                   gamma_coefb=0.25, Skw_denom_coef=4.0, Skw_max_mag=10.0,
                   beta=1.0, lmin_coef=0.5, lambda0_stability_coef=0.03,
                   mult_coef=0.5, C_invrs_tau_bkgnd=1.1, C_invrs_tau_shear=0.15,
                   C_invrs_tau_N2=0.4, C_invrs_tau_N2_xp2=0.05,
                   C_invrs_tau_N2_clear_wp3=1.0, up2_sfc_coef=4.0)
        cl = cfg.turbulence.clubb
        cfg = cfg._replace(turbulence=cfg.turbulence._replace(
            clubb=cl._replace(params=cl.params._replace(**lib))))
    MF_LOG = _G_MF_LOG
    if a.clubb_mf:
        import legoesm.atmosphere.physics.turbulence.clubb_mf as _mf
        _orig_f = _mf.clubb_mf_forcing

        def _rec(*args, **kw):
            if os.environ.get("MF_MEM") and kw.get("ztopm1") is not None:
                kw["ztopm1"] = jnp.full_like(kw["ztopm1"], float(os.environ["MF_MEM"]))
            fthl, fqt, out = _orig_f(*args, **kw)
            if os.environ.get("MF_DUMP"):
                def _dump(c, z, q, *rest):
                    if not np.all(np.isfinite(np.asarray(q))) and not os.path.exists(os.environ["MF_DUMP"]):
                        np.savez(os.environ["MF_DUMP"], ztopm1=np.asarray(z),
                                 args=np.array([np.asarray(r) for r in rest], dtype=object),
                                 **{f"c_{k}": np.asarray(v) for k, v in c.items()})
                        print("DUMPED nonfinite MF call inputs", flush=True)
                jax.debug.callback(_dump, args[0], kw.get("ztopm1"), out["qtflx"], *args[1:5])
            jax.debug.callback(lambda zm, q, t: MF_LOG.append((np.asarray(zm).reshape(-1), np.asarray(q).reshape(-1), np.asarray(t).reshape(-1))),
                               args[0]["zm"], out["cflfac"] * out["qtflx"], out["cflfac"] * out["thlflx"])
            return fthl, fqt, out
        _mf.clubb_mf_forcing = _rec
        if os.environ.get("MF_L0"):
            _mf.L0 = float(os.environ["MF_L0"])
        if os.environ.get("MF_PWSCALE"):
            _f = float(os.environ["MF_PWSCALE"])
            _mf.PWMIN, _mf.PWMAX = _mf.PWMIN * _f, _mf.PWMAX * _f
        print("MF knobs L0", _mf.L0, "PW", _mf.PWMIN, _mf.PWMAX, flush=True)
        if os.environ.get("LOOPCAP"):
            _CAP = int(os.environ["LOOPCAP"])

            def _hit(name, lam, k):
                print(f"LOOPCAP {name} hit cap lam={float(lam):.6g} iters={int(k)}", flush=True)

            def _knuth_c(seeds, lam):
                explam = jnp.exp(-1.0 * lam)
                def body(c):
                    sd, k, puni, n = c
                    sd, u = _mf._kiss_next(sd)
                    return sd, k + 1, puni * u, n + 1
                sd, k, _, n = jax.lax.while_loop(lambda c: (c[2] > explam) & (c[3] < _CAP), body,
                                                 (seeds, jnp.int32(0), jnp.float64(1.0), jnp.int32(0)))
                jax.lax.cond(n >= _CAP, lambda: jax.debug.callback(_hit, "knuth", lam, n), lambda: None)
                return sd, k - 1
            _orig_h = _mf._hormann

            def _hormann_c(seeds, lam):
                import inspect
                return _orig_h(seeds, lam)
            pass
            # hormann: re-implement with cap by wrapping lax.while_loop locally
            _wl = jax.lax.while_loop

            def _capped_wl(cond, body, init):
                def c2(c):
                    return cond(c[0]) & (c[1] < _CAP)
                def b2(c):
                    return body(c[0]), c[1] + 1
                out, n = _wl(c2, b2, (init, jnp.int32(0)))
                jax.lax.cond(n >= _CAP, lambda: jax.debug.callback(_hit, "while", jnp.float64(-1), n), lambda: None)
                return out
            _mf.jax = type("J", (), {"lax": type("L", (), {k: getattr(jax.lax, k) for k in dir(jax.lax) if not k.startswith("__")} | {"while_loop": staticmethod(_capped_wl)}), **{k: getattr(jax, k) for k in ("numpy", "scipy", "debug", "tree_util", "nn", "vmap", "jit", "custom_jvp", "custom_vjp", "random") if hasattr(jax, k)}})
        if os.environ.get("MF_TRACE"):
            _orig_d = _mf.entrainment_draws
            _ncall = [0]

            def _pr(lam, u):
                _ncall[0] += 1
                lam = np.asarray(lam)
                if _ncall[0] <= 3:
                    np.savez(f"/burg/glab/users/pg2328/enso_logs/draw_in_{_ncall[0]}.npz", lam=lam, u=np.asarray(u))
                if _ncall[0] < 20 or _ncall[0] % 50 == 0 or not np.all(np.isfinite(lam)):
                    print(f"TRACE call {_ncall[0]} lam finite={np.all(np.isfinite(lam))} "
                          f"max={np.nanmax(lam):.3g} nnan={np.sum(~np.isfinite(lam))}", flush=True)

            def _draws(lam, u):
                print("TRACE tracing entrainment_draws", flush=True)
                jax.debug.callback(_pr, lam, u, ordered=True)
                return _orig_d(lam, u)
            _mf.entrainment_draws = _draws
        cfg = cfg._replace(turbulence=cfg.turbulence._replace(clubb_mf=True))
        if os.environ.get("MF_ALPHTURB"):
            cfg = cfg._replace(turbulence=cfg.turbulence._replace(clubb=cfg.turbulence.clubb._replace(
                mf_alphturb=float(os.environ["MF_ALPHTURB"]))))
            print("MF alphturb", cfg.turbulence.clubb.mf_alphturb, flush=True)
        if os.environ.get("MF_HARD"):
            import math as _m
            _cl = cfg.turbulence.clubb
            _hk = {f: _m.inf for f in _cl._fields if f.startswith("mf_bsort_") and f.endswith("_sharpness")}
            if os.environ["MF_HARD"] == "all":
                _hk = {f: _m.inf for f in _cl._fields if f.startswith("mf_") and f.endswith("_sharpness")}
            cfg = cfg._replace(turbulence=cfg.turbulence._replace(clubb=_cl._replace(**_hk)))
            print("MF hard", sorted(_hk), flush=True)
        if os.environ.get("MF_BSORT"):
            cfg = cfg._replace(turbulence=cfg.turbulence._replace(clubb_mf_bsort=True))
        print("MF bsort", cfg.turbulence.clubb_mf_bsort, flush=True)
        _cc = getattr(cfg.radiation, "cloud_config", None)
        if getattr(_cc, "scheme", None) == "cam6_clubb":
            cfg = cfg._replace(radiation=cfg.radiation._replace(
                cloud_config=_cc._replace(clubb_mf_cloud=True, mf_cldfrac_fac=FAC)))
            print("MF cldfrac_fac", cfg.radiation.cloud_config.mf_cldfrac_fac, flush=True)
        print("cloud scheme", getattr(_cc, "scheme", None),
              "radiation", getattr(cfg.radiation, "scheme", None), flush=True)
    print(f"ARM clubb_mf={a.clubb_mf} lib_params={a.lib_params} case={a.case} nlev={arm.case.nlev} dt={arm.dt} hours={arm.hours} "
          f"window={arm.analysis_hours} h flux_to_closure={a.surface_flux_to_closure} "
          f"prod_liquid={a.prod_liquid}",
          flush=True)
    m, ql_hist = rollout(cfg, arm)
    les, nprof = les_window(a.les_dir, arm.hours, arm.analysis_hours)
    z = np.asarray(arm.case.z_full)
    p_full = np.asarray(arm.case.p_full)
    theta = m["T"] / np.asarray(H.exner_function(jnp.asarray(p_full)))
    rho = p_full / (H.constants.R_d * m["T"])
    zi = np.concatenate([[float(arm.case.les_domain_top_m) * 1.2],
                         0.5 * (z[:-1] + z[1:]), [0.0]])
    dz = zi[:-1] - zi[1:]
    lwp_scm = float(np.sum(rho * m["ql"] * dz)) * 1e3
    rho_les = np.interp(les["z"], z[::-1], rho[::-1])
    lwp_les = float(np.trapezoid(rho_les * les["qc"], les["z"])) * 1e3
    thr = 1e-6
    top_scm = float(z[m["ql"] > thr].max()) if np.any(m["ql"] > thr) else np.nan
    top_les = float(les["z"][les["qc"] > thr].max()) if np.any(les["qc"] > thr) else np.nan
    mom = {n: m["mom"][i] for i, n in enumerate(MOM)}
    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"git={git} LES profiles in window: {nprof}")
    # plume cloud as CAM7 radiation sees it: deep fraction 10*cf (cap 0.8),
    # in-cloud water mf_qc -> grid water 10*cf*mf_qc (one step lagged carry)
    dc = np.minimum(FAC * m["mfcf"], 0.8)
    lwp_mf = float(np.sum(rho * FAC * m["mfcf"] * m["mfqc"] * dz)) * 1e3
    print(f"LWP g/m2: scm CLUBB {lwp_scm:.3f}  plume {lwp_mf:.3f}  total {lwp_scm + lwp_mf:.3f}  les {lwp_les:.3f}")
    print(f"COVER (max-in-column cloud fraction): CLUBB {m['ccf'].max():.4f}  plume {dc.max():.4f}  "
          f"combined {np.max(np.minimum(m['ccf'] + dc, 1.0)):.4f}  les {les['cloud_frac'].max():.4f}")
    print(f"cloud top (qc>{thr}) m: scm {top_scm:.0f}  les {top_les:.0f}")
    print(f"max qc g/kg: scm {1e3*m['ql'].max():.4f} at {z[m['ql'].argmax()]:.0f} m"
          f"  les {1e3*les['qc'].max():.4f} at {les['z'][les['qc'].argmax()]:.0f} m")
    print(f"LES max cloud_frac {les['cloud_frac'].max():.4f} at "
          f"{les['z'][les['cloud_frac'].argmax()]:.0f} m")
    print("wprtp first/last:", mom["wprtp"][:3], mom["wprtp"][-3:])
    zm_asc = (MF_LOG[0][0] if MF_LOG else None)
    if MF_LOG:
        n_w = int(round(len(MF_LOG) * arm.analysis_hours / arm.hours))
        mf_q = np.mean([x[1] for x in MF_LOG[-n_w:]], 0)
        mf_t = np.mean([x[2] for x in MF_LOG[-n_w:]], 0)
        print(f"MF calls {len(MF_LOG)}, window calls {n_w}")
        _n = len(MF_LOG)
        for _h in (0.25, 0.5, 1, 2, 3, 4, 5, arm.hours):
            _i = min(_n - 1, int(round(_n * _h / arm.hours)) - 1)
            _blk = MF_LOG[max(0, _i - 20):_i + 1]
            _q = np.mean([x[1] for x in _blk], 0); _zm = MF_LOG[0][0]
            _top = _zm[np.abs(_q) > 1e-3 * np.abs(_q).max()].max() if np.abs(_q).max() > 0 else np.nan
            print(f"MFTOP t={_h:g} h flux-top {_top:.0f} m", flush=True)
    else:
        zi_asc = zi[::-1].copy(); zi_asc[-1] = 2 * zi_asc[-2] - zi_asc[-3]
        zm_asc, mf_q, mf_t = zi_asc, 0.0, 0.0
    tot_q = mom["wprtp"] + mf_q
    print("FLUXRATIO z | clubb w'rt'/LES | mf/LES | total/LES")
    for zz in (300, 500, 700, 900, 1100, 1300, 1500):
        lw = np.interp(zz, les["z"], les["wqt"])
        cq = np.interp(zz, zm_asc, mom["wprtp"]); mq = np.interp(zz, zm_asc, np.broadcast_to(mf_q, zm_asc.shape))
        print(f"FLUXRATIO {zz:5d} | {cq/lw:6.2f} | {mq/lw:6.2f} | {(cq+mq)/lw:6.2f}")
    print("LES wqt profile (z, wqt):")
    for zz in (100, 300, 500, 700, 1000, 1300, 1600, 1900, 2200):
        print(f"  z={zz:5d} les wqt={np.interp(zz, les['z'], les['wqt']):.3e} "
              f"wth={np.interp(zz, les['z'], les['wtheta']):.3e} "
              f"qc={np.interp(zz, les['z'], les['qc']):.3e} | scm qc="
              f"{np.interp(zz, z[::-1], m['ql'][::-1]):.3e} theta="
              f"{np.interp(zz, z[::-1], theta[::-1]):.2f} les theta="
              f"{np.interp(zz, les['z'], les['theta']):.2f} scm qv="
              f"{np.interp(zz, z[::-1], m['qv'][::-1]):.5f} les qv="
              f"{np.interp(zz, les['z'], les['qv']):.5f}")
    np.savez(a.out, z=z, theta=theta, qv=m["qv"], ql=m["ql"], mom=m["mom"],
             mom_names=np.array(MOM), les_z=les["z"], les_qc=les["qc"],
             les_cf=les["cloud_frac"], les_wqt=les["wqt"], les_wth=les["wtheta"],
             les_theta=les["theta"], les_qv=les["qv"], lwp_scm=lwp_scm,
             lwp_les=lwp_les, ql_hist_max=np.max(ql_hist, 1), git=git,
             nlev=arm.case.nlev, dt=arm.dt, rho=rho, dz=dz, mfcf=m["mfcf"],
             mfqc=m["mfqc"], mfaq=m["mfaq"], ccf=m["ccf"])


if __name__ == "__main__":
    main()
