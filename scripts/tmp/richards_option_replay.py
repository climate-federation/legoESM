"""Offline replay of Richards solver options on four cases (probe, not production).

Patch anchors target the solver at commit 99742ec40 (before option 1 landed);
run it against that tree. Results: PR decision record for the Richards fix.

Variants patch the PR worktree's richards.py source in memory:
  main : undamped, 10 iterations, no convergence freeze (= origin/main loop)
  fix  : PR as committed (psi-space damping 0.1, 30 it, freeze)
  opt1 : fix + give back a negative residual into unsaturated layers
  opt2 : theta-space damping 0.1 (bound enforced), 30 it, freeze
  opt3 : undamped, 30 it, freeze
Per solve, a debug callback records: pre-correction error (created - refill,
before main's take-back), post-correction error (water_created), n_iter,
converged, top-layer theta and head.
"""
import os, subprocess, sys, tempfile, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
W = "/glade/derecho/scratch/pg2328/tmp/wt-richards"
import legoesm.land.richards as R
import legoesm.land.multilayer_land as ML
assert R.__file__.startswith(W), R.__file__
SHA = subprocess.check_output(["git", "-C", W, "rev-parse", "--short", "HEAD"]).decode().strip()
DIRTY = subprocess.check_output(["git", "-C", W, "status", "--porcelain", "--", "packages"]).decode().strip()
print(f"PROBE git={SHA} dirty_packages={bool(DIRTY)} land={R.__file__}", flush=True)

src = open(R.__file__).read()
def sub(a, b):
    global src
    assert src.count(a) == 1, a[:60]
    src = src.replace(a, b)

sub("""        lam = richards_config.max_dse_per_iter / jnp.maximum(
            dse_try, richards_config.max_dse_per_iter)
        dpsi = dpsi * lam[:, None]
        dh_s = dh_s * lam
""", """        lam = richards_config.max_dse_per_iter / jnp.maximum(
            dse_try, richards_config.max_dse_per_iter)
        if not _PV["damp"]:
            lam = jnp.ones_like(lam)
        _psi_full = jnp.maximum(psi_m + dpsi, _psi_dry_floor)
        dpsi = dpsi * lam[:, None]
        dh_s = dh_s * lam
""")
sub("""        psi_new = jnp.maximum(psi_m + dpsi, _psi_dry_floor)
""", """        psi_new = jnp.maximum(psi_m + dpsi, _psi_dry_floor)
        if _PV["theta_damp"]:
            _th_d = theta_m + lam[:, None] * (theta_try - theta_m)
            _ok = ((lam < 1.0)[:, None] & (_th_d < 0.999 * hydro_config.theta_sat)
                   & (theta_try < 0.999 * hydro_config.theta_sat)
                   & (theta_m < 0.999 * hydro_config.theta_sat))
            _th_safe = jnp.where(_ok, _th_d, 0.5 * (hydro_config.theta_r + hydro_config.theta_sat))
            psi_new = jnp.where(_ok, jnp.maximum(psi_from_theta(_th_safe, hydro_config),
                                                 _psi_dry_floor), psi_new)
""")
sub("""        conv = (lam >= 1.0) & (""", """        conv = (_PV["freeze"]) & (lam >= 1.0) & (""")
sub("""        0, richards_config.max_iter,""", """        0, _PV["max_iter"],""")
sub("""    water_created = _budget(theta_final) - refill
""", """    water_created = _budget(theta_final) - refill
    _pre = created - refill
    if _PV["giveback"]:
        _deficit = jnp.maximum(-water_created, 0.0)
        _room = jnp.where(psi_final < 0.0, jnp.maximum(
            (0.999 * hydro_config.theta_sat - theta_final) * dz[None, :], 0.0), 0.0)
        _gf = jnp.minimum(_deficit / jnp.maximum(jnp.sum(_room, axis=1), 1e-30), 1.0)
        _give = _room * _gf[:, None]
        theta_final = theta_final + _give / dz[None, :]
        _ts = jnp.where(_give > 0.0, theta_final, 0.5 * (theta_floor + hydro_config.theta_sat))
        psi_final = jnp.where(_give > 0.0, psi_from_theta(_ts, hydro_config), psi_final)
        water_created = _budget(theta_final) - refill
    jax.debug.callback(_record, _pre, water_created, n_iter_final, converged,
                       theta_final[:, 0], psi_final[:, 0])
""")
src = src.replace("from __future__ import annotations\n", "from __future__ import annotations\n", 1)
REC = []
def _record(pre, post, nit, conv, th0, ps0):
    REC.append(tuple(np.asarray(x) for x in (pre, post, nit, conv, th0, ps0)))
VARIANTS = {
    "main": dict(damp=False, theta_damp=False, freeze=False, max_iter=10, giveback=False),
    "fix":  dict(damp=True, theta_damp=False, freeze=True, max_iter=30, giveback=False),
    "opt1": dict(damp=True, theta_damp=False, freeze=True, max_iter=30, giveback=True),
    "opt2": dict(damp=True, theta_damp=True, freeze=True, max_iter=30, giveback=False),
    "opt3": dict(damp=False, theta_damp=False, freeze=True, max_iter=30, giveback=False),
}
R.__dict__["_record"] = _record
R.__dict__["_PV"] = {}
exec(compile(src, R.__file__, "exec"), R.__dict__)
ML.solve_richards = R.solve_richards

sys.path.insert(0, W + "/tests/land/unit")
import test_richards_dry_convergence as DC
import test_land_evap_supply_limit as EV
import test_ic_soil_hydraulics as IC
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.surface_scheme import TwoLeafCanopyConfig
from legoesm.land.canopy.interception import InterceptionConfig
for m in (DC, EV, IC):
    if hasattr(m, "solve_richards"):
        m.solve_richards = R.solve_richards

def summarize(label, extra=""):
    pre = np.concatenate([np.atleast_1d(r[0]) for r in REC]); post = np.concatenate([np.atleast_1d(r[1]) for r in REC])
    nit = np.concatenate([np.atleast_1d(r[2]) for r in REC]); cv = np.concatenate([np.atleast_1d(r[3]) for r in REC])
    th = np.concatenate([np.atleast_1d(r[4]) for r in REC]); ps = np.concatenate([np.atleast_1d(r[5]) for r in REC])
    if not (np.all(np.isfinite(pre)) and np.all(np.isfinite(post)) and np.all(np.isfinite(th))):
        raise SystemExit(f"NaN in {label}")
    print(f"ROW {label:34s} solves={len(cv):4d} conv={int(cv.sum())}/{len(cv)} nit_max={int(nit.max()):2d} "
          f"theta_top_max={th.max():.3f} head_top_max={ps.max():.3g}m "
          f"pre_err_m[min,max]=[{pre.min():.2e},{pre.max():.2e}] post_err_m[min,max]=[{post.min():.2e},{post.max():.2e}] "
          f"post_sum_m={post.sum():.2e} {extra}", flush=True)

tmp = tempfile.mkdtemp()
cfg = MultiLayerLandConfig().richards
for vname, pv in VARIANTS.items():
    R._PV.clear(); R._PV.update(pv)
    for cname, col in (("spikeA", DC._COL_A), ("spikeB", DC._COL_B)):
        REC.clear(); out, res = DC._solve(col, cfg)
        summarize(f"{cname:7s} {vname}", f"theta_top_final={float(out.theta_new[0,0]):.3f}")
    REC.clear()
    res, et, gap = EV._day(TwoLeafCanopyConfig(), top=3.0e-3, deep=3.0e-3, precip=2.0e-6,
                           interception=InterceptionConfig())
    summarize(f"drizzle {vname}", f"LAND_res_sum_kgm2={res.sum():.3e} LAND_res_max={np.abs(res).max():.2e}")
    REC.clear()
    st0, conv, rep, residual_mm = IC._first_step(__import__("pathlib").Path(tempfile.mkdtemp(dir=tmp)), jnp.float64, True)
    mm = residual_mm(conv)
    summarize(f"wetdry  {vname}", f"IC_test_res_mm_max={mm.max():.2e}")
