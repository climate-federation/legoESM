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
SCMGrid.lat = property(lambda g: g.grid_lat)

MOM = ("rtm", "thlm", "um", "vm", "wp2", "wp3", "up2", "vp2", "wprtp",
       "wpthlp", "upwp", "vpwp", "rtp2", "thlp2", "rtpthlp")


def rollout(cfg, arm):
    scm = H._create_scm(cfg, arm.case, arm.dt)
    step_fn, tend_fn, dt = scm._step_fn, scm._tend_fn, arm.dt
    print("tracers:", sorted(scm.state.tracers.keys()), flush=True)
    liq = [k for k in ("q_c", "q_l", "q_cl") if k in scm.state.tracers][0]

    def body(carry, k):
        state, ps = carry
        t = k.astype(jnp.float64) * dt
        ns, nps = step_fn(state, ps, tend_fn, dt, t)
        out = (ns.T.data[0, 0, 0], ns.tracers["q_v"].data[0, 0, 0],
               ns.tracers[liq].data[0, 0, 0], nps.clubb_moments[0])
        return (ns, nps), out

    nsteps = int(round(arm.hours * 3600.0 / dt))
    nan = int(round(arm.analysis_hours * 3600.0 / dt))
    _, (T, qv, ql, mom) = jax.jit(
        lambda c: lax.scan(body, c, jnp.arange(nsteps)))((scm.state, scm.phys_state))
    w = slice(nsteps - nan, nsteps)
    return {k: np.asarray(v[w]).mean(0) for k, v in
            dict(T=T, qv=qv, ql=ql, mom=mom).items()}, np.asarray(ql)


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
        import functools
        import legoesm.atmosphere.forcing.scm.scm as _scm
        from legoesm.atmosphere.physics.microphysics.config import (
            apply_microphysics_experiment_flags)
        # SingleColumnModel exposes no macmic count; production runs 3.
        _scm.make_physics = functools.partial(_scm.make_physics,
                                              cld_macmic_num_steps=3)
        cfg = cfg._replace(
            turbulence=cfg.turbulence._replace(liquid_partition=True),
            microphysics=apply_microphysics_experiment_flags(
                cfg.microphysics, cfg.microphysics.scheme,
                liquid_from_closure=True))
    print(f"ARM case={a.case} nlev={arm.case.nlev} dt={arm.dt} hours={arm.hours} "
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
    print(f"LWP g/m2: scm {lwp_scm:.3f}  les {lwp_les:.3f}")
    print(f"cloud top (qc>{thr}) m: scm {top_scm:.0f}  les {top_les:.0f}")
    print(f"max qc g/kg: scm {1e3*m['ql'].max():.4f} at {z[m['ql'].argmax()]:.0f} m"
          f"  les {1e3*les['qc'].max():.4f} at {les['z'][les['qc'].argmax()]:.0f} m")
    print(f"LES max cloud_frac {les['cloud_frac'].max():.4f} at "
          f"{les['z'][les['cloud_frac'].argmax()]:.0f} m")
    print("wprtp first/last:", mom["wprtp"][:3], mom["wprtp"][-3:])
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
             nlev=arm.case.nlev, dt=arm.dt)


if __name__ == "__main__":
    main()
