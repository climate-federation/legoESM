"""Run the OpenIFS CUMASTRN single-column oracle and the JAX port on the same
column and write a level-by-level comparison.

usage: make_fixture.py <driver.x> <work dir> <case> [--nlev N]
  case = port_deep : the port's own inputs on the deep test sounding
         known     : isothermal dry column at the triple point (no convection;
                     saturation humidity has a closed form)
Writes <work dir>/<case>_inputs.txt, <case>_dump.txt and
results/ifs_cumastrn_oracle/<case>.npz plus a printed table.  It reports
numbers only; interpretation belongs in the analysis.
"""
import argparse
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, REPO)

from tests.unit.test_ifs_faithful import _deep_column, _kwargs  # noqa: E402
from legoesm.atmosphere.physics.convection import _ifs_faithful as F  # noqa: E402
from legoesm.atmosphere.physics.convection import _ifs_closure as C  # noqa: E402
from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection  # noqa: E402
from legoesm.atmosphere.physics.convection.config import BechtoldConfig  # noqa: E402
from legoesm import constants  # noqa: E402


def _row(a):
    return " ".join(f"{float(x):.17e}" for x in np.ravel(a))


def write_inputs(path, T, q, u, v, pf, ph, gf, gh, dT, dq, dTa, dqa, dt, dx, shf, lhf):
    with open(path, "w") as fh:
        fh.write(f"{T.size}\n{dt:.17e} {dx:.17e} {shf:.17e} {lhf:.17e}\n")
        for a in (T, q, u, v, pf, ph, gf, gh, dT, dq, dTa, dqa):
            fh.write(_row(a) + "\n")


def read_dump(path):
    out, lev = {}, []
    for line in open(path):
        tok = line.split()
        if tok[0] in ("KLEV",):
            out["klev"] = int(tok[1])
        elif tok[0] == "CONST":
            out.update(zip(("rlvtt", "rd", "rv", "retv", "rtt", "rcpd", "rg"), map(float, tok[8:])))
        elif tok[0] == "FG":
            vals = tok[6:]
            out.update(ktype_fg=int(vals[0]), ktype_rc=int(vals[1]), kcbot=int(vals[2]),
                       kctop=int(vals[3]), ldcum_fg=vals[4] == "T")
        elif tok[0] == "MB":
            out.update(zip(("zmfub_fg", "zmfub1", "zdhpbl"), map(float, tok[4:])))
        elif tok[0] == "OUT":
            out.update(ldcum=tok[5] == "T", ktype=int(tok[6]), kcbot_out=int(tok[7]), kctop_out=int(tok[8]))
        elif tok[0] == "LEV":
            names = tok[1:]
        elif tok[0].isdigit():
            lev.append([float(x) for x in tok[1:]])
    lev = np.asarray(lev)
    for i, n in enumerate(names):
        out[n] = lev[:, i]
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            assert np.all(np.isfinite(v)), f"non-finite oracle field {k}"
    return out


def port_case(nlev):
    """Capture exactly what the port's faithful chain receives and produces."""
    T, q, pf, ph = [a.astype(jnp.float64) for a in _deep_column(nlev=nlev)]
    kw = _kwargs(*T.shape)
    cap = {}

    def spy(mod, name, fn):
        orig = getattr(mod, name)

        def w(*a, **k):
            r = orig(*a, **k)
            fn(a, k, r)
            return r
        setattr(mod, name, w)
        return orig

    keep = []
    keep.append((F, "ifs_faithful_convection", spy(F, "ifs_faithful_convection", lambda a, k, r: cap.update(
        call_args=a, call_kw=k))))
    keep.append((F, "_hydrostatic_geopotential", spy(F, "_hydrostatic_geopotential", lambda a, k, r: cap.update(
        geo=r))))
    keep.append((F, "ifs_departure_search_refined", spy(F, "ifs_departure_search_refined", lambda a, k, r: cap.update(
        trig=r))))
    keep.append((C, "first_guess_mass_flux", spy(C, "first_guess_mass_flux", lambda a, k, r: cap.update(
        fg=(r, a[4])))))
    keep.append((F, "ifs_updraught_ascent", spy(F, "ifs_updraught_ascent", lambda a, k, r: cap.update(asc=r))))
    keep.append((F, "_reclassify_ktype", spy(F, "_reclassify_ktype", lambda a, k, r: cap.update(ktype_rc=r))))
    keep.append((F, "ifs_convective_fluxes", spy(F, "ifs_convective_fluxes", lambda a, k, r: cap.update(
        flx=r))))
    keep.append((C, "shallow_closure", spy(C, "shallow_closure", lambda a, k, r: cap.update(mb1=r))))
    cfg = BechtoldConfig(use_ifs_ascent=True)
    out = bechtold_convection(T, q, pf, ph, config=cfg, **kw)[0]
    for mod, name, orig in keep:
        setattr(mod, name, orig)
    a, k = cap["call_args"], cap["call_kw"]
    T_in, q_in, pf_in, ph_in, u_in, v_in = (np.asarray(x)[0] for x in a[:6])
    dt = float(a[9])
    gf, gh = (np.asarray(x)[0] for x in cap["geo"])
    z = np.zeros_like(T_in)
    to = lambda key: np.asarray(k[key])[0] if k.get(key) is not None else z  # noqa: E731
    inputs = dict(T=T_in, q=q_in, u=u_in, v=v_in, pf=pf_in, ph=ph_in, gf=gf, gh=gh,
                  dT=to("dT_dt_other"), dq=to("dq_dt_other"), dTa=to("dT_dt_adv"), dqa=to("dq_dt_adv"),
                  dt=dt, dx=float(a[10].dx_m), shf=float(np.asarray(k["shf_w_m2"])[0]),
                  lhf=float(np.asarray(k["lhf_w_m2"])[0]))
    fg, zdhpbl = cap["fg"]
    asc = cap["asc"]
    port = dict(ktype_fg=int(np.asarray(cap["trig"].ktype)[0]), kcbot=int(np.asarray(cap["trig"].k_cbot)[0]),
                ktype_rc=int(np.asarray(cap["ktype_rc"])[0]), kctop=int(np.asarray(asc.k_ctop)[0]),
                zmfub_fg=float(np.asarray(fg[0])[0]), zdhpbl=float(np.asarray(zdhpbl)[0]),
                zmfub1=float(np.asarray(cap["mb1"])[0]),
                mfu_asc=np.asarray(asc.M)[0], mfus_asc=np.asarray(asc.PMFUS)[0],
                mful_asc=np.asarray(asc.PMFUL)[0], lude_asc=np.asarray(asc.PLUDE)[0],
                dmfup_asc=np.asarray(asc.PDMFUP)[0], dT_dt=np.asarray(out.dT_dt)[0],
                dq_dt=np.asarray(out.dq_v_dt)[0])
    for n, v in zip(("mfu_flx", "mfus_flx", "mfuq_flx", "mful_flx", "lude_flx", "dmfup_flx"), cap["flx"]):
        port[n] = np.asarray(v)[0]
    return inputs, port


def known_case(nlev):
    sig = np.linspace(0.0, 1.0, nlev + 1)
    ph = sig * 101300.0
    ph[0] = 1000.0
    pf = 0.5 * (ph[1:] + ph[:-1])
    rtt = 273.16   # const-ok: OpenIFS RTT (oracle known-answer input, checked against the dump)
    # stable (temperature rises upward), dry; pure-ice (< RTICECU = RTT-38 after SUCUMF),
    # triple-point and pure-liquid thirds so both saturation branches are exercised
    T = np.where(np.arange(nlev) < nlev // 3, rtt - 40.0,
                 np.where(np.arange(nlev) < 2 * nlev // 3, rtt, rtt + 20.0))[::-1].copy()
    q = np.full(nlev, 1.0e-6)
    gh = np.zeros(nlev + 1)
    for j in range(nlev - 1, -1, -1):   # isothermal hydrostatic; only consistency matters here
        gh[j] = gh[j + 1] + constants.R_d * T[j] * np.log(ph[j + 1] / ph[j])
    gf = 0.5 * (gh[1:] + gh[:-1])
    z = np.zeros(nlev)
    return dict(T=T, q=q, u=z, v=z, pf=pf, ph=ph, gf=gf, gh=gh, dT=z, dq=z, dTa=z, dqa=z,
                dt=112.5, dx=1.0e5, shf=0.0, lhf=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("driver")
    ap.add_argument("work")
    ap.add_argument("case", choices=["port_deep", "known"])
    ap.add_argument("--nlev", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    if args.case == "port_deep":
        inputs, port = port_case(args.nlev)
    else:
        inputs, port = known_case(args.nlev), None
    fin = os.path.join(args.work, f"{args.case}_inputs.txt")
    fout = os.path.join(args.work, f"{args.case}_dump.txt")
    write_inputs(fin, **inputs)
    r = subprocess.run([args.driver, fin, fout], cwd=args.work, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    assert r.returncode == 0 and "DRIVER_OK" in r.stdout, "oracle driver failed"
    o = read_dump(fout)
    res = os.path.join(REPO, "results", "ifs_cumastrn_oracle")
    os.makedirs(res, exist_ok=True)
    sha = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    save = {f"oracle_{k}": np.asarray(v) for k, v in o.items()}
    save.update({f"in_{k}": np.asarray(v) for k, v in inputs.items()})
    if port:
        save.update({f"port_{k}": np.asarray(v) for k, v in port.items()})
    np.savez(os.path.join(res, f"{args.case}.npz"), repo_sha=sha, **save)
    print(f"CASE {args.case} nlev={args.nlev} repo_sha={sha}")
    print("ORACLE scalars:", {k: o[k] for k in ("ldcum_fg", "ktype_fg", "ktype_rc", "kcbot", "kctop", "zmfub_fg",
                                                 "zmfub1", "zdhpbl", "ldcum", "ktype")})
    print("ORACLE const:", {k: o[k] for k in ("rlvtt", "rd", "rv", "retv", "rtt", "rcpd", "rg")})
    if args.case == "known":
        eps = o["rd"] / o["rv"]
        r2es = 611.21 * eps   # const-ok: satcurve-ok: suphec.F90:231 R2ES=611.21*RD/RV
        T, rtt = inputs["T"], o["rtt"]
        liq = r2es * np.exp(17.502 * (T - rtt) / (T - 32.19))   # satcurve-ok: suphec.F90 R3LES, R4LES
        ice = r2es * np.exp(22.587 * (T - rtt) / (T + 0.7))     # satcurve-ok: suphec.F90 R3IES, R4IES
        es = np.where(T >= rtt, liq, ice)   # FOEALFCU = 1 at T >= RTWAT=RTT, 0 below RTICECU (only these regimes present)
        zqs = np.minimum(0.5, es / inputs["pf"])
        qs_closed = zqs / (1.0 - o["retv"] * zqs)
        # SATUR fills JK = NJKT2..KLEV only (satur.F90 loop; sucumf.F90:285 NJKT2 = lowest
        # level with STPRE > 60 hPa); levels above keep the input humidity, as in CUCALLN
        sat = inputs["pf"] > 6000.0   # const-ok: sucumf.F90:285 60.E2 threshold
        rel = np.abs(o["qsen"][sat] - qs_closed[sat]) / qs_closed[sat]
        print(f"KNOWN levels checked {int(sat.sum())}/{sat.size}; max|qsen-closed|/closed = {float(np.max(rel)):.3e}")
        print("KNOWN levels above NJKT2 equal input q:", bool(np.all(o["qsen"][~sat] == inputs["q"][~sat])))
        print("KNOWN max|dT_dt| max|dq_dt| max|mfu| =", float(np.max(np.abs(o["dT_dt"]))),
              float(np.max(np.abs(o["dq_dt"]))), float(np.max(np.abs(o["mfu"]))))
        return
    print("PORT scalars:", {k: port[k] for k in ("ktype_fg", "ktype_rc", "kcbot", "kctop", "zmfub_fg",
                                                 "zmfub1", "zdhpbl")})
    print("index note: oracle KCBOT/KCTOP are 1-based (port = oracle - 1)")
    print(f"{'k':>3} {'p_hPa':>7} | {'M_asc ifs':>11} {'port':>11} | {'MFUS ifs':>12} {'port':>12} | "
          f"{'dTdt ifs K/d':>12} {'port':>12}")
    for j in range(len(inputs["pf"])):
        if max(abs(o["mfu_asc"][j]), abs(port["mfu_asc"][j]), abs(o["dT_dt"][j]), abs(port["dT_dt"][j])) == 0.0:
            continue
        print(f"{j:3d} {inputs['pf'][j]/100:7.1f} | {o['mfu_asc'][j]:11.4e} {port['mfu_asc'][j]:11.4e} | "
              f"{o['mfus_asc'][j]:12.5e} {port['mfus_asc'][j]:12.5e} | {o['dT_dt'][j]*86400:12.4f} "
              f"{port['dT_dt'][j]*86400:12.4f}")
    dpg = np.diff(inputs["ph"]) / o["rg"]
    for who, src in (("oracle", o), ("port", port)):
        print(f"BUDGET {who}: column sum(cp*dT + L*dq) dp/g = "
              f"{float(np.sum((o['rcpd'] * src['dT_dt'] + o['rlvtt'] * src['dq_dt']) * dpg)):.4e} W/m2 "
              f"(sum cp*dT = {float(np.sum(o['rcpd'] * src['dT_dt'] * dpg)):.4e})")
    print(f"{'k':>3} {'p_hPa':>7} | flux-stage M ifs/port | MFUS ifs/port | DMFUP ifs/port | DMFDP ifs")
    for j in range(len(inputs["pf"]) - 14, len(inputs["pf"])):
        print(f"{j:3d} {inputs['pf'][j]/100:7.1f} | {o['mfu_flx'][j]:10.4e} {port['mfu_flx'][j]:10.4e} | "
              f"{o['mfus_flx'][j]:11.4e} {port['mfus_flx'][j]:11.4e} | {o['dmfup_flx'][j]:10.3e} "
              f"{port['dmfup_flx'][j]:10.3e} | {o['dmfdp_flx'][j]:10.3e}")
    for n in ("mfu_asc", "mfus_asc", "mful_asc", "lude_asc", "dmfup_asc", "mfu_flx", "mfus_flx", "mfuq_flx",
              "mful_flx", "lude_flx", "dmfup_flx", "dT_dt", "dq_dt"):
        d = np.abs(o[n] - port[n])
        sc = max(float(np.max(np.abs(o[n]))), 1e-300)
        print(f"DIFF {n}: max_abs={float(np.max(d)):.3e} at k={int(np.argmax(d))}  max_abs/max|ifs|={float(np.max(d))/sc:.3e}")


if __name__ == "__main__":
    main()
