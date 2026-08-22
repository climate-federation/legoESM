#!/usr/bin/env python
"""#1455 PHASE-2 MEASUREMENT 1: the RETENTION FACTOR, measured rather than assumed.

WHY THIS EXISTS
===============
The substep walk measures an INJECTION: the per-step barotropic transport
difference a matched state deposits in one parent step (+5.56e-03 Sv/step at
day 180, of which 64% is the frozen slow forcing and 36% a state-insensitive
in-loop constant).  The budget row measures an ACCUMULATION: -0.401 Sv of
realized ACC gap over 90 days.

Those are different objects.  Write the error recursion

    e_{n+1} = M_n e_n + d_n

The walk measures ``d_n``.  The budget measures the accumulation of the
``d_n`` along two free-running trajectories.  The two agree ONLY if the
retention ``M`` is 1, and every "x times the budget row" ratio this campaign
published silently assumed exactly that -- which is why all of them were
retracted (eb3f6d23d, 89dadfeb4).  ``M`` has never been measured.  This
module measures it.

WHAT IS MEASURED
================
A free-running 90-day twin gets the measured deposit's OWN depth-uniform
barotropic velocity field added ONCE at t=0, and its ACC transport is tracked
daily against a byte-identical unperturbed arm.  The injected section
transport at t=0+ IS the deposit, so

    R(t) = [ACC(perturbed, t) - ACC(control, t)] / (injected transport)

is the retention factor, in the deposit's own units, as a ratio of one
functional to itself.

THE TRAP THIS MODULE EXISTS TO AVOID
====================================
A response measured at ONE amplitude cannot tell LINEAR RETENTION from
CHAOTIC DIVERGENCE.  Both produce a non-zero difference that persists for 90
days; only the first is a retention factor, and only the first may be
multiplied by a per-step injection rate to predict an accumulation.  They are
separated by AMPLITUDE SCALING: a linear response scales exactly with the
injection, a chaotic one saturates and decorrelates.  Arms at 0.5x, 1x, 100x
and -100x are therefore required, and the linearity criterion is
pre-registered.  Without them this module would report a retention factor
that is really a Lyapunov exponent.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

#: the ACC difference floor between two 90-day arms of this card, measured by
#: the 4-member perturbed ensemble on branch fidelity/dino-floor90 [Sv].
FLOOR_SV = 1.146e-05

#: readability bar: a response below this many floors carries no information.
READABLE_FLOORS = 3.0

#: pre-registered linearity tolerance (see PREREG): the amplitude-scaled
#: responses must agree to this fraction on the days where both are readable.
LINEARITY_TOL = 0.25

#: the accumulation this campaign has to explain [Sv], from the full-section
#: accumulated stage budget (c987db464).
GAP_FULL_SECTION = -0.401
GAP_CHANNEL = +0.063

STEPS_PER_DAY = 32
N_DAYS = 90

#: the measured per-step deposits, mean over the ten bridged states of the
#: 90-day window (results/dino_1455/baro_deposit_time_walk.npz).  Loaded from
#: that artifact when present -- these are the fallback for a bare checkout and
#: the run REFUSES to use them silently.
_DEPOSIT_ARTIFACT = os.path.join(
    _THIS_DIR, "..", "..", "..", "..", "results", "dino_1455",
    "baro_deposit_time_walk.npz")


def provenance(tag: str) -> str:
    """git SHA + the environment variables that can change a number."""
    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=_THIS_DIR, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:
            return "?"
    sha = _git("rev-parse", "HEAD")
    # TRACKED dirt only: untracked files cannot change a number, and a bare
    # `git status --porcelain` counts them, which made an earlier stamp on this
    # branch permanently read "+dirty" and carry no information.
    dirty = len([ln for ln in _git("status", "--porcelain", "--untracked-files=no")
                 .splitlines() if ln.strip()])
    env = " ".join(f"{k}={os.environ.get(k)!r}" for k in
                   ("JAX_ENABLE_X64", "LEGOESM_NEMO_E3T", "CUDA_VISIBLE_DEVICES"))
    return f"[provenance {tag}] git={sha}+dirtyTracked{dirty}  {env}"


def load_deposits() -> dict:
    """The measured per-step deposits, from the committed walk artifact."""
    p = os.path.abspath(_DEPOSIT_ARTIFACT)
    if not os.path.exists(p):
        raise SystemExit(
            f"FATAL: {p} is missing.  The retention correction multiplies the "
            "MEASURED per-step deposits by the MEASURED retention; inventing "
            "either half is how this campaign produced the ratios it had to "
            "retract.  Run baro_deposit_time_walk.py first.")
    d = np.load(p, allow_pickle=True)
    return {"total": np.asarray(d["dep_total"], dtype=np.float64),
            "forcing": np.asarray(d["dep_forcing"], dtype=np.float64),
            "in_loop": np.asarray(d["dep_in_loop"], dtype=np.float64)}


def load_arm(path: str) -> dict:
    """One twin arm, with the checks that decide whether it may be used."""
    if not os.path.exists(path):
        raise SystemExit(f"FATAL: missing arm {path}")
    d = np.load(path, allow_pickle=True)
    for k in ("acc_dep_daily", "acc_gate_daily", "perturb_baro_scale",
              "perturb_baro_key", "stable", "blew_up_at_step"):
        if k not in d.files:
            raise SystemExit(
                f"FATAL: {path} has no {k!r}.  It was not produced by a "
                "--daily-acc run, so it cannot enter a retention curve.")
    acc = np.asarray(d["acc_dep_daily"], dtype=np.float64)
    gate = np.asarray(d["acc_gate_daily"], dtype=np.float64)
    # C1/C2: NaN is FATAL, never averaged away.
    if not bool(d["stable"]) or int(d["blew_up_at_step"]) != -1:
        raise SystemExit(f"FATAL: arm {path} is not stable")
    if not (np.isfinite(acc).all() and np.isfinite(gate).all()):
        raise SystemExit(f"FATAL: non-finite transport in {path}")
    return {"path": path, "acc": acc, "gate": gate,
            "scale": float(d["perturb_baro_scale"]),
            "key": str(d["perturb_baro_key"])}


def retention(arm: dict, ctrl: dict, injected_sv: float) -> dict:
    """R(t) for one arm, with its readability mask."""
    resp = arm["acc"] - ctrl["acc"]
    resp_gate = arm["gate"] - ctrl["gate"]
    inj = injected_sv * arm["scale"]
    if inj == 0.0:
        raise SystemExit("FATAL: retention of an arm with zero injection")
    return {"resp": resp, "resp_gate": resp_gate, "inj": inj,
            "R": resp / inj, "R_gate": resp_gate / inj,
            "floors": np.abs(resp) / FLOOR_SV,
            "readable": np.abs(resp) > READABLE_FLOORS * FLOOR_SV}


def linearity(a: dict, b: dict) -> dict:
    """Do two amplitudes give the SAME R?  The linear/chaotic discriminator.

    Scored only where BOTH arms are readable.  A linear response has R
    independent of amplitude; a chaotic divergence does not.
    """
    m = a["readable"] & b["readable"]
    if not m.any():
        return {"n": 0, "median_ratio": float("nan"), "pass": False,
                "reason": "no day where both arms are readable"}
    ratio = a["R"][m] / b["R"][m]
    med = float(np.median(ratio))
    # correlation of the two RESPONSE time series is the sharper statistic:
    # a linear response is the SAME curve rescaled, so corr == 1.
    corr = float(np.corrcoef(a["resp"][m], b["resp"][m])[0, 1])
    ok = abs(med - 1.0) <= LINEARITY_TOL
    return {"n": int(m.sum()), "median_ratio": med, "corr": corr, "pass": bool(ok),
            "reason": ""}


def accumulate(R: np.ndarray, per_step_sv: float) -> float:
    """The retention-corrected accumulation S = sum_n d_n R(t_end - t_n) [Sv].

    A constant per-step injection d applied at every one of the 2880 steps,
    each surviving to day 90 with the retention its own age implies.  With a
    daily R this is d * STEPS_PER_DAY * sum over days of R.
    """
    return float(per_step_sv * STEPS_PER_DAY * np.nansum(R))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="/tmp/dino_1455_ret",
                    help="directory holding the arm npz files")
    ap.add_argument("--injected", type=float, default=5.5555e-03,
                    help="the day-180 total deposit [Sv], the transport the "
                         "scale=1 arm injects at t=0+")
    ap.add_argument("--key", default="dU_avg",
                    help="which injected pattern to score (dU_avg = total "
                         "deposit, dU_sub = in-loop share)")
    ap.add_argument("--arms", default="m_p05:0.5,m_p1:1.0,m_p100:100.0,m_m100:-100.0",
                    help="comma-separated name:scale list")
    ap.add_argument("--ctrl", default="m_ctrl")
    ap.add_argument("--out", default=os.path.join(
        _THIS_DIR, "..", "..", "..", "..", "results", "dino_1455",
        "baro_retention_walk.npz"))
    args = ap.parse_args(argv)

    print(provenance("baro_retention_walk"))
    print(f"floor={FLOOR_SV:.3e} Sv   readable bar={READABLE_FLOORS} floors   "
          f"linearity tol={LINEARITY_TOL}")

    ctrl = load_arm(os.path.join(args.dir, f"{args.ctrl}.npz"))
    if ctrl["scale"] != 0.0:
        raise SystemExit(
            f"FATAL: the control arm has scale={ctrl['scale']}, not 0 -- read "
            "from its own stamp, not its filename")
    arms, rets = {}, {}
    for spec in args.arms.split(","):
        nm, sc = spec.split(":")
        a = load_arm(os.path.join(args.dir, f"{nm}.npz"))
        # C4: the arms differ ONLY in the scale, verified from the stamps.
        if abs(a["scale"] - float(sc)) > 1e-12:
            raise SystemExit(
                f"FATAL: {nm} stamps scale={a['scale']}, the arm list says {sc}")
        if a["key"] != args.key:
            raise SystemExit(
                f"FATAL: {nm} injected {a['key']!r}, this scoring is for "
                f"{args.key!r} -- refusing to mix two patterns into one curve")
        arms[nm] = a
        rets[nm] = retention(a, ctrl, args.injected)

    print(f"\n=== R(t), injected pattern {args.key!r}, "
          f"scale-1 transport {args.injected:+.4e} Sv ===")
    names = list(rets)
    print("  day | " + " | ".join(f"{n:>10s}" for n in names)
          + " |   floors(1x)")
    ref = "m_p1" if "m_p1" in rets else names[0]
    for i in list(range(6)) + list(range(9, N_DAYS, 10)) + [N_DAYS - 1]:
        row = " | ".join(f"{rets[n]['R'][i]:+10.5f}" for n in names)
        print(f"  {i+1:3d} | {row} | {rets[ref]['floors'][i]:9.2f}")

    # ---- the linear/chaotic discriminator -------------------------------
    print("\n=== LINEARITY (does R depend on amplitude?) ===")
    lin = {}
    for nm in names:
        if nm == ref:
            continue
        L = linearity(rets[nm], rets[ref])
        lin[nm] = L
        # -100x is scored against its own sign flip: R should be IDENTICAL,
        # because R already divides by the signed injection.
        print(f"  {nm:>10s} vs {ref}: n={L['n']:3d} days both readable, "
              f"median R-ratio={L['median_ratio']:+.4f}, "
              f"response corr={L.get('corr', float('nan')):+.4f}  "
              f"-> {'LINEAR' if L['pass'] else 'NOT LINEAR'}")
    all_linear = bool(lin) and all(v["pass"] for v in lin.values())
    print(f"  VERDICT (built from the measured ratios): "
          f"{'LINEAR' if all_linear else 'NOT LINEAR'}")
    if not all_linear:
        print("  => the post-transient difference between the arms is NOT a "
              "linear response to the injection.  It is trajectory divergence, "
              "and it MUST NOT be multiplied by a per-step injection rate: "
              "doing so would report a Lyapunov exponent as a retention "
              "factor.  Only the days where the arms agree on R carry "
              "retention information; the rest is bounded, not curved.")

    # ---- the coherent (linear) window ------------------------------------
    Rref = rets[ref]["R"]
    coherent = []
    for i in range(N_DAYS):
        vals = [rets[n]["R"][i] for n in names if rets[n]["readable"][i]]
        if len(vals) < 2:
            continue
        v = np.asarray(vals)
        if np.abs(v).max() > 0 and (np.abs(v - v.mean()).max()
                                    <= LINEARITY_TOL * np.abs(v.mean())):
            coherent.append(i)
    print(f"\n=== THE COHERENT WINDOW: days where every readable arm agrees on "
          f"R to {int(100*LINEARITY_TOL)}% ===")
    print(f"  days = {[i+1 for i in coherent] if coherent else 'NONE beyond day 1'}")
    n_coh = len(coherent)
    R_coh = float(np.sum([Rref[i] for i in coherent])) if coherent else 0.0
    print(f"  sum of R over the coherent window = {R_coh:+.5f} "
          f"({n_coh} day(s))")

    # ---- the retention-corrected accounting ------------------------------
    dep = load_deposits()
    print("\n=== THE RETENTION-CORRECTED ACCOUNTING ===")
    print(f"  target: full-section gap {GAP_FULL_SECTION:+.3f} Sv, "
          f"channel {GAP_CHANNEL:+.3f} Sv")
    print("  survivor      | mean d_n [Sv/step] | S coherent-window "
          "| S all-days (UPPER BOUND, contaminated by divergence)")
    acct = {}
    for nm_s, arr in (("total", dep["total"]), ("forcing", dep["forcing"]),
                      ("in-loop", dep["in_loop"])):
        dbar = float(arr.mean())
        s_coh = accumulate(np.array([Rref[i] for i in coherent])
                           if coherent else np.zeros(1), dbar)
        s_all = accumulate(Rref, dbar)
        acct[nm_s] = {"d_bar": dbar, "S_coherent": s_coh, "S_all": s_all}
        print(f"  {nm_s:13s} | {dbar:+.4e}        | {s_coh:+10.4f} Sv      "
              f"| {s_all:+10.4f} Sv")

    # ---- the verdict, BUILT from the measured values ---------------------
    print("\n=== VERDICT (every word below is computed from the numbers "
          "above, none of it is hardcoded) ===")
    verdicts = {}
    for nm_s in ("forcing", "in-loop"):
        s_coh = acct[nm_s]["S_coherent"]
        s_all = acct[nm_s]["S_all"]
        need = GAP_FULL_SECTION
        # sign first: a positive S cannot pay a negative gap at any size.
        sign_ok = (s_all * need) > 0 or (s_coh * need) > 0
        big_enough = max(abs(s_coh), abs(s_all)) >= abs(need) / 2.0
        if not sign_ok and not big_enough:
            v = "EXONERATED (wrong sign AND too small)"
        elif not sign_ok:
            v = "REFUTED BY SIGN (right size, wrong sign)"
        elif not big_enough:
            v = "EXONERATED (right sign, too small)"
        else:
            v = "CANDIDATE (right sign and size)"
        short = max(abs(need) / max(abs(s_all), 1e-300), 0.0)
        verdicts[nm_s] = v
        print(f"  {nm_s:8s}: S(all-days upper bound)={s_all:+.4f} Sv vs "
              f"needed {need:+.3f} Sv -> {short:.1f}x short -> {v}")
    print(f"\n  retention at day 1 = {Rref[0]:+.4f}; "
          f"at day 2 = {Rref[1]:+.4f}; "
          f"|R| falls below 1% of the injection after day "
          f"{next((i+1 for i in range(N_DAYS) if abs(Rref[i]) < 0.01), None)}")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, days=np.arange(1, N_DAYS + 1),
             ctrl_acc=ctrl["acc"], ctrl_gate=ctrl["gate"],
             coherent_days=np.asarray([i + 1 for i in coherent], dtype=np.int32),
             summary_json=np.array(json.dumps({
                 "provenance": provenance("baro_retention_walk"),
                 "injected_sv": args.injected, "key": args.key,
                 "floor_sv": FLOOR_SV,
                 "R_day1": float(Rref[0]), "R_day2": float(Rref[1]),
                 "linearity": {k: {kk: vv for kk, vv in v.items()
                                   if kk != "reason"} for k, v in lin.items()},
                 "all_linear": all_linear,
                 "accounting": acct, "verdicts": verdicts,
                 "gap_full_section": GAP_FULL_SECTION})),
             **{f"R_{n}": rets[n]["R"] for n in names},
             **{f"resp_{n}": rets[n]["resp"] for n in names})
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
