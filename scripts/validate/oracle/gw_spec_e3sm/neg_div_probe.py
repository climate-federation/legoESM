"""#2567 probe: E3SM v3.0.1 ``gw_drag_prof`` vs the port where a layer's raw
stress divergence ``ubtl`` is negative.

E3SM applies ``gwut = sign(ubtl, c-u)`` (= ``|ubtl|*sgn(c-u)``) and carries the
signed ``ubtl`` to the interface below (gw_common.F90:649, :671); the port
applies ``sgn(c-u)*ubtl``.  They differ only when ``ubtl < 0``, and there
E3SM's column deposits momentum the stress profile does not supply.

(1) Constructed columns (:func:`column`): the launched stress below the source
grows upward (``tend_level > src_level``, which E3SM's interface allows), so
the tendency region holds negative divergences; ``control`` launches uniform
stress below the source (every real E3SM source).  The E3SM side is
``gw_neg_div_oracle`` from ``build.sh`` (``neg_div_driver.F90``, unmodified
gw_common.F90).  Printed per case: the port-vs-E3SM du/dt gap relative to the
column maximum and each side's column momentum budget (:func:`budget`).
(2) Reachability: the most negative raw divergence the port meets in every
arm/dataset of the E3SM oracle fixture, hard limit and smooth default.

Usage (x64, repo root on PYTHONPATH):
  bash scripts/validate/oracle/gw_spec_e3sm/build.sh <E3SM v3.0.1> <bld>
  python scripts/validate/oracle/gw_spec_e3sm/neg_div_probe.py <bld>/gw_neg_div_oracle <out_dir>
"""
from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from legoesm import constants

CASES = ("neg", "neg_umcfac", "control")


def column(case: str) -> dict:
    """30-level isothermal column, one wave, ``u = 10 m/s`` everywhere."""
    pver = 30
    pint = np.geomspace(100.0, 1.0e5, pver + 1)
    src, tend, c0 = 20, 26, 40.0
    tau = np.zeros(pver + 1)
    if case in ("neg", "neg_umcfac"):         # stress grows upward below the source
        tau[src:tend + 1] = np.linspace(0.02, 0.01, tend - src + 1)
        tau[tend:] = 0.01
        if case == "neg_umcfac":              # c near u: the umcfac limit is close
            c0 = 10.5
    elif case == "control":                   # uniform below the source
        tau[src:] = 0.02
    else:
        raise ValueError(f"unknown case {case!r}; expected one of {CASES}")
    return dict(pver=pver, pint=pint, pmid=0.5 * (pint[1:] + pint[:-1]),
                t=np.full(pver, 240.0), ubm=np.full(pver, 10.0), ubi=np.full(pver + 1, 10.0),
                tau=tau, alpha=np.full(pver + 1, 1.0e-6), src=src, tend=tend, c=c0,
                dt=1800.0, effgw=1.0)


def budget(col: dict, du_dt, tau) -> tuple[float, float]:
    """(column deposit sum du/dt dp/g over the tendency region, stress
    divergence effgw*sgn(c-u)*(tau(tend) - tau(top))) for the zonal column."""
    dpm = np.diff(col["pint"])
    n = col["tend"]
    dep = float(np.sum(np.asarray(du_dt)[:n] * dpm[:n] / constants.g))
    div = float(np.sign(col["c"] - col["ubm"][0]) * col["effgw"] * (tau[n] - tau[0]))
    return dep, div


def _fortran(exe, col, tag, wd, cfg):
    fin, fout = wd / f"in_{tag}.txt", wd / f"out_{tag}.txt"
    head = [col["pver"], col["dt"], col["effgw"], col["c"], col["src"], col["tend"], 0.0,
            1.0, 0.0, cfg.kwv, cfg.fcrit2, constants.g, constants.R_d, constants.c_pd]
    with open(fin, "w") as f:
        f.write(" ".join(map(repr, head)) + "\n")
        for k in ("pint", "pmid", "t", "ubm", "ubi", "tau", "alpha"):
            f.write(" ".join(f"{x:.17e}" for x in col[k]) + "\n")
    r = subprocess.run([exe, str(fin), str(fout)], capture_output=True, text=True)
    if "E3SM_DIRECT_OK" not in r.stdout:
        raise RuntimeError(r.stdout + r.stderr)
    return {p[0]: np.array([float(x) for x in p[1:]]) for p in map(str.split, open(fout))}


def port(col, prof, cfg, smooth):
    """The port's ``gw_drag_prof`` on :func:`column` ``col``; ``prof`` holds
    the (nlev+1)/(nlev) profiles ``ti, rhoi, nm, ni`` of one column (the
    oracle's, or the port's ``gw_prof``).  Returns (du/dt, final stress)."""
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import gw_drag_prof
    fo = {k: np.asarray(prof[k]).reshape(-1) for k in ("ti", "rhoi", "nm", "ni")}
    a = lambda x: jnp.asarray(x)[None]
    dpm = np.diff(col["pint"])
    kw = {} if not smooth else dict(
        softmin_sharpness=cfg.softmin_sharpness, crit_level_sharpness=cfg.crit_level_sharpness,
        crit_level_floor=cfg.crit_level_floor, umcfac_abs_width=cfg.umcfac_abs_width,
        umcfac_sign_width=cfg.umcfac_sign_width, taumin_ramp_width=cfg.taumin_ramp_width)
    tau, ut, *_ = gw_drag_prof(
        jnp.asarray(col["tau"])[None, None], jnp.array([[col["c"]]]),
        jnp.array([col["src"]]), jnp.array([col["tend"]]), a(col["t"]), a(fo["ti"]),
        a(np.log(col["pint"])), a(fo["rhoi"]), a(fo["nm"]), a(fo["ni"]), a(col["ubm"]),
        a(col["ubi"]), jnp.array([1.0]), jnp.array([0.0]), a(dpm), a(1.0 / dpm),
        jnp.array([0.0]), col["effgw"], col["dt"], cfg, orographic_only=False,
        alpha_iface=a(col["alpha"]), **kw)
    return np.asarray(ut[0]), np.asarray(tau[0, 0])


def _reach(repo: Path) -> dict:
    """Most negative raw layer divergence in the tendency region of every
    oracle-fixture arm (port at the hard limit and the smooth default)."""
    import jax
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.gravity_wave_drag import e3sm_cam as E
    sys.path.insert(0, str(repo / "tests/atmosphere/hydrostatic/unit"))
    import test_gw_spec_e3sm_oracle as T
    z = dict(np.load(T.FIXTURE))
    meta = json.loads(str(z.pop("meta")))
    m = meta["mfcc"]
    tbl = np.random.default_rng(m["seed"]).uniform(
        0.0, m["hi"], size=(m["maxh"], 2 * m["maxuh"] + 1, 2 * m["pgwv"] + 1))
    tbl[:, :, m["pgwv"]] = 0.0
    if hashlib.sha256(tbl.tobytes()).hexdigest() != m["sha256"]:
        raise RuntimeError("regenerated Beres harness table does not match the fixture")
    rec, orig = [], E.gw_drag_prof

    sig = inspect.signature(orig)

    def wrapped(*args, **kw):
        out = orig(*args, **kw)
        bound = sig.bind(*args, **kw).arguments
        tau_f, tau0, tend_level, rdpm = out[0], out[4], bound["tend_level"], bound["rdpm"]
        raw = constants.g * (tau0[:, :, 1:] - tau_f[:, :, :-1]) * rdpm[:, None, :]
        within = jnp.arange(rdpm.shape[1])[None, None, :] < tend_level[:, None, None]
        jax.debug.callback(lambda n, p: rec.append((float(n), float(p))),
                           jnp.min(jnp.where(within, raw, jnp.inf)),
                           jnp.max(jnp.where(within, jnp.abs(raw), 0.0)))
        return out

    E.gw_drag_prof = wrapped
    try:
        res = {}
        for arm, ds in T._cases():
            for hard in (True, False):
                rec.clear()
                T._arm((z, meta, tbl), arm, ds, hard=hard)
                res[f"{arm}/{ds}/{'hard' if hard else 'default'}"] = dict(
                    min_raw=min(r[0] for r in rec), max_abs=max(r[1] for r in rec))
    finally:
        E.gw_drag_prof = orig
    return res


def main(exe: str, out_dir: str) -> dict:
    import jax
    if not jax.config.jax_enable_x64:
        raise SystemExit("run with JAX_ENABLE_X64=1")
    from legoesm.atmosphere.physics.gravity_wave_drag.config import E3SMCAMConfig
    repo = Path(__file__).resolve().parents[4]
    wd = Path(out_dir)
    wd.mkdir(parents=True, exist_ok=True)
    cfg = E3SMCAMConfig()
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    res = dict(provenance=dict(git_sha=sha, oracle_md5=hashlib.md5(
        Path(exe).read_bytes()).hexdigest(), cases=list(CASES)))
    for case in CASES:
        col = column(case)
        fo = _fortran(exe, col, case, wd, cfg)
        uf, tf = fo["utgw"], fo["tau"]
        r = dict(min_raw_divergence=float(np.min(
            constants.g * np.diff(col["tau"])[:col["tend"]] / np.diff(col["pint"])[:col["tend"]])),
            budget_e3sm=budget(col, uf, tf))
        for name, smooth in (("hard", False), ("default", True)):
            up, tp = port(col, fo, cfg, smooth)
            r[f"gap_rel_{name}"] = float(np.abs(up - uf).max() / np.abs(uf).max())
            r[f"budget_port_{name}"] = budget(col, up, tp)
        res[case] = r
        print(case, r)
    res["reach"] = _reach(repo)
    for k, v in res["reach"].items():
        print("reach", k, v)
    json.dump(res, open(wd / "neg_div_probe.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
