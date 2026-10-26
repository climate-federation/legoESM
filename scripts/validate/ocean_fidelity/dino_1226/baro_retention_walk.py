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

#: ...AND their response time series must actually be the SAME CURVE.  This
#: second bar is not decoration, it is the one that works.  Scored on the first
#: run, the median-ratio test alone PASSED an arm pair whose response series
#: correlate at 0.06 -- i.e. it certified two essentially unrelated curves as a
#: linear response, because with sign-oscillating divergence the pointwise
#: ratio distribution is heavy-tailed and its MEDIAN lands near 1 by accident.
#: A ratio-of-noise is a statistic that cannot fail the way this test needs to.
#: The correlation cannot be fooled that way: a linear response is one curve
#: rescaled, so it correlates at 1 by construction.
LINEAR_CORR_MIN = 0.90

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
    # The MEASURED injected transport, stamped by the twin.  Older artifacts
    # predate the stamp; they are refused rather than silently normalised by a
    # hand-typed default, which is the defect this replaced.
    if "injected_sv" not in d.files:
        raise SystemExit(
            f"FATAL: {path} carries no injected_sv stamp.  It predates the "
            "measured normaliser, and every retention factor divides by that "
            "number -- re-run the arm rather than supplying it by hand.")
    return {"path": path, "acc": acc, "gate": gate,
            "scale": float(d["perturb_baro_scale"]),
            "key": str(d["perturb_baro_key"]),
            "injected_sv": float(d["injected_sv"])}


def retention(arm: dict, ctrl: dict, injected_sv: float | None = None) -> dict:
    """R(t) for one arm, with its readability mask.

    The normaliser is the arm's OWN measured injected transport.  It used to be
    a CLI float defaulting to the total deposit, which would have divided an
    in-loop-pattern response by a number 2.75x too big without any guard
    firing.  ``injected_sv`` remains only as an override for synthetic tests.
    """
    resp = arm["acc"] - ctrl["acc"]
    resp_gate = arm["gate"] - ctrl["gate"]
    inj = (arm["injected_sv"] if injected_sv is None
           else injected_sv * arm["scale"])
    if inj == 0.0:
        raise SystemExit("FATAL: retention of an arm with zero injection")
    return {"resp": resp, "resp_gate": resp_gate, "inj": inj,
            "scale": arm["scale"],
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
        return {"n": 0, "median_ratio": float("nan"), "corr": float("nan"),
                "ratio_ok": False, "corr_ok": False, "pass": False,
                "reason": "no day where both arms are readable"}
    ratio = a["R"][m] / b["R"][m]
    med = float(np.median(ratio))
    # Correlate the R series, NOT the raw responses.  R already divides by the
    # SIGNED injection, so a perfect linear response gives corr = +1 for every
    # arm including the negative-scale one -- whereas the raw responses of a
    # -100x and a +1x arm anti-correlate at -1 when the response is perfectly
    # linear, which would have been printed next to a "LINEAR" verdict and read
    # as its opposite.
    corr = float(np.corrcoef(a["R"][m], b["R"][m])[0, 1])
    ratio_ok = abs(med - 1.0) <= LINEARITY_TOL
    corr_ok = corr >= LINEAR_CORR_MIN
    return {"n": int(m.sum()), "median_ratio": med, "corr": corr,
            "ratio_ok": bool(ratio_ok), "corr_ok": bool(corr_ok),
            "pass": bool(ratio_ok and corr_ok), "reason": ""}


def _distinct_amplitudes(rets: dict, names: list, i: int) -> int:
    """How many DISTINCT injection magnitudes are readable on day ``i``.

    A +100x and a -100x arm agree on R whenever the response is merely
    sign-antisymmetric, which says nothing about AMPLITUDE linearity -- the
    only property this window exists to certify.  Counting arms rather than
    magnitudes let that pair certify the window on its own once the smaller
    arms dropped under the floor.
    """
    return len({round(abs(rets[n]["scale"]), 12)
                for n in names if rets[n]["readable"][i]})


def linear_window(rets: dict, names: list) -> list:
    """The CONTIGUOUS leading run of days on which the amplitudes agree.

    Contiguity is the point.  A response to a t=0 injection is a decaying
    transient: once the arms have decorrelated, any later day on which three
    sign-oscillating series happen to land within tolerance is a coincidence,
    not a resumption of linear response.  Scoring every such day (the first
    version of this function) inflated the retention sum SEVENFOLD.
    """
    # Length comes from the DATA, not from the module's N_DAYS constant: a
    # shorter series must not index past its end.
    n_days = min(len(rets[n]["R"]) for n in names)
    last_certified = -1
    for i in range(n_days):
        vals = [rets[n]["R"][i] for n in names if rets[n]["readable"][i]]
        n_amp = _distinct_amplitudes(rets, names, i)
        if len(vals) >= 2 and n_amp >= 2:
            v = np.asarray(vals)
            mean = float(np.abs(v.mean()))
            if mean <= 0.0 or np.abs(v - v.mean()).max() > LINEARITY_TOL * mean:
                break            # a REAL disagreement ends the window
            last_certified = i
        # Days with fewer than two DISTINCT injection magnitudes readable carry
        # no information -- which is not the same as evidence that linearity
        # ended, so they do not break the run.  They also cannot EXTEND it:
        # the window ends at the last day actually certified, so a trailing
        # run of uninformative days can never be scored as linear response.
    return list(range(last_certified + 1))


def accumulate(R: np.ndarray, per_step_sv: float) -> float:
    """The retention-corrected accumulation S = sum_n d_n R(age of step n) [Sv].

    A constant per-step injection ``d`` is applied at every one of the 2880
    steps and each survives to day 90 with the retention its own AGE implies,
    so S = d * (steps/day) * integral of R over ages 0..90 days.

    THE AGE-ZERO BIN IS NOT OPTIONAL.  ``R`` is sampled at ages 1,2,...,90
    days, but the 32 steps injected in the final 24 hours have age under one
    day, where R runs from 1 (by construction: at t=0+ the injected transport
    IS the deposit) down to R(1 day).  A plain sum over the daily samples drops
    that bin entirely -- worth up to d * STEPS_PER_DAY = 0.15 Sv against a
    0.401 Sv target, i.e. up to 37% of the quantity being explained, and always
    in the direction that makes a suspect look too small.  Both reviews caught
    it independently.

    R(0) = 1 is prepended and the integral is a trapezoid.  Over the first day
    that OVERESTIMATES, because the barotropic adjustment is hours and the
    decay inside day 1 is far faster than linear -- which is the right
    direction for a bound used to EXONERATE.
    """
    if not np.all(np.isfinite(R)):
        raise SystemExit(
            "FATAL: non-finite retention in accumulate() -- a NaN must never "
            "be summed away into an accumulation")
    ages = np.concatenate([[1.0], R])          # R(age=0) == 1 by construction
    return float(per_step_sv * STEPS_PER_DAY * np.trapezoid(ages))


def verdict(s_coh: float, s_all: float, need: float,
            superposes_beyond_window: bool = True) -> str:
    """Classify a survivor from its retention-corrected accumulation.

    Extracted from ``main`` so it can be table-tested: an inline version of
    this had its sign discriminator inverted by mutation and every test stayed
    green, which is the "test that cannot fail" class this campaign bans.

    ``need`` is the accumulation to be explained.  SIGN IS CHECKED FIRST and on
    its own: the deposits are positive at every measured state and the gap is
    negative, so an S of the right SIZE but the wrong SIGN refutes the
    injection picture just as firmly as one that is too small.
    """
    # SUPERPOSITION IS A PRECONDITION, NOT A DETAIL.  S = sum_n d_n R(age)
    # adds the responses of 2880 separate injections, which is only valid
    # while the response is LINEAR.  Once the perturbed and control
    # trajectories have decorrelated, their difference is no longer a
    # response to the injection at all -- it is two chaotic trajectories
    # drifting apart, and its magnitude reflects when divergence started
    # rather than how big the injection was.  Summing it as if it
    # superposed is not a loose upper bound, it is a category error, and it
    # flipped this verdict from EXONERATED to CANDIDATE on real data purely
    # on the strength of two late-window chaotic excursions.
    #
    # So when the arms have decorrelated, the all-days figure is not
    # admitted as an accumulation at all and the verdict rests on the linear
    # window alone.  The chaotic range is reported as UNMEASURABLE BY THIS
    # INSTRUMENT, which is the honest description of it.
    if superposes_beyond_window:
        sign_ok = (s_all * need) > 0 or (s_coh * need) > 0
        big_enough = max(abs(s_coh), abs(s_all)) >= abs(need) / 2.0
    else:
        sign_ok = (s_coh * need) > 0
        big_enough = abs(s_coh) >= abs(need) / 2.0
    if not sign_ok and not big_enough:
        return "EXONERATED (wrong sign AND too small)"
    if not sign_ok:
        return "REFUTED BY SIGN (right size, wrong sign)"
    if not big_enough:
        return "EXONERATED (right sign, too small)"
    return "CANDIDATE (right sign and size)"


def lyapunov_growth(resp: np.ndarray, floor: float) -> dict:
    """Fit an exponential envelope to |response| -- the discriminator that works.

    The amplitude ladder CANNOT separate retention from chaos on its own, and
    this is the review finding that matters most: tangent-linear chaotic growth
    is EXACTLY proportional to the perturbation amplitude until it saturates,
    so 0.5x and 1x arms agree to many digits under BOTH hypotheses.  What
    separates them is the SHAPE in time.  A retention factor is bounded and
    decays; a chaotic divergence GROWS exponentially.  A positive fitted
    exponent is therefore positive evidence of divergence, not merely an
    absence of evidence for retention.
    """
    n = len(resp)
    t = np.arange(1, n + 1, dtype=np.float64)
    a = np.abs(resp)
    m = a > floor            # log of a sub-floor value is noise, not signal
    if m.sum() < 10:
        return {"n": int(m.sum()), "rate_per_day": float("nan"),
                "e_folding_days": float("nan"), "growing": False}
    sl, _ = np.polyfit(t[m], np.log(a[m]), 1)
    return {"n": int(m.sum()), "rate_per_day": float(sl),
            "e_folding_days": float(1.0 / sl) if sl != 0 else float("inf"),
            "growing": bool(sl > 0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="/tmp/dino_1455_ret",
                    help="directory holding the arm npz files")
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
        rets[nm] = retention(a, ctrl)
        print(f"  {nm:>10s}: scale {a['scale']:+8.2f}  pattern {a['key']:8s}  "
              f"MEASURED injected transport {a['injected_sv']:+.6e} Sv")

    print(f"\n=== R(t), injected pattern {args.key!r} "
          f"(normaliser is each arm's OWN measured injected transport) ===")
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
    coherent = linear_window(rets, names)
    print(f"\n=== THE LINEAR WINDOW: the CONTIGUOUS run of days from day 1 on "
          f"which every readable arm agrees on R to {int(100*LINEARITY_TOL)}% ===")
    print("  Contiguous, and that matters: taking every day on which three "
          "sign-oscillating series happen to agree is a cherry-pick, and it "
          "inflated this sum by 7x on the first run.  Once the arms decorrelate "
          "they can re-agree by chance at any later date without any of it "
          "being a response to the injection.")
    print(f"  days = {[i+1 for i in coherent] if coherent else 'NONE'}")
    n_coh = len(coherent)
    R_coh = float(np.sum([Rref[i] for i in coherent])) if coherent else 0.0
    print(f"  sum of R over the linear window = {R_coh:+.5f} ({n_coh} day(s))")

    # ---- the discriminator the amplitude ladder cannot provide -----------
    print("\n=== IS THE POST-TRANSIENT SIGNAL GROWING?  (the shape-in-time "
          "discriminator) ===")
    print("  A retention factor is bounded and decays.  A chaotic divergence "
          "grows exponentially.\n  Tangent-linear chaos is EXACTLY proportional "
          "to the injection amplitude until it\n  saturates, so the amplitude "
          "ladder alone cannot tell the two apart -- this can.")
    lyap = {}
    for nm in names:
        gl = lyapunov_growth(rets[nm]["resp"], FLOOR_SV)
        lyap[nm] = gl
        print(f"  {nm:>10s}: fitted rate {gl['rate_per_day']:+.4f} /day  "
              f"e-folding {gl['e_folding_days']:+.1f} d  on {gl['n']} "
              f"above-floor days -> "
              f"{'GROWING (divergence)' if gl['growing'] else 'decaying'}")
    n_grow = sum(1 for v in lyap.values() if v["growing"])
    print(f"  {n_grow} of {len(lyap)} arms GROW.  Where the signal grows it "
          f"carries no retention information at any amplitude.")

    # Does the response still SUPERPOSE beyond the linear window?  Measured,
    # not assumed: the R series of two different amplitudes are correlated
    # inside the window and uncorrelated outside it if the arms have
    # decorrelated.  This is the precondition for summing d_n R(age) at all.
    _post = slice(len(coherent), None)
    _pc = []
    for _i, _a in enumerate(names):
        for _b in names[_i + 1:]:
            if abs(abs(rets[_a]["scale"]) - abs(rets[_b]["scale"])) < 1e-12:
                continue          # same magnitude proves nothing about linearity
            _x, _y = rets[_a]["R"][_post], rets[_b]["R"][_post]
            if len(_x) > 3:
                _pc.append(float(np.corrcoef(_x, _y)[0, 1]))
    _post_corr = float(np.median(_pc)) if _pc else float("nan")
    _superposes = bool(_pc) and _post_corr >= LINEAR_CORR_MIN
    print(f"\n=== DOES THE RESPONSE STILL SUPERPOSE PAST THE LINEAR WINDOW? ===")
    print(f"  median R-correlation across amplitudes, days "
          f"{len(coherent)+1}..{N_DAYS}: {_post_corr:+.4f}  "
          f"(bar {LINEAR_CORR_MIN})")
    print(f"  -> {'superposes' if _superposes else 'DOES NOT SUPERPOSE'}: "
          + ("the all-days sum is admissible."
             if _superposes else
             "the all-days sum is NOT an accumulation and is NOT scored.  "
             "Summing d_n*R(age) over a decorrelated range adds responses "
             "that never superposed; the range is UNMEASURABLE by an impulse "
             "experiment, which is the pre-registered UNREADABLE outcome."))

    # ---- the retention-corrected accounting ------------------------------
    dep = load_deposits()
    print("\n=== THE RETENTION-CORRECTED ACCOUNTING ===")
    print(f"  target: full-section gap {GAP_FULL_SECTION:+.3f} Sv, "
          f"channel {GAP_CHANNEL:+.3f} Sv")
    print("  survivor      | mean d_n [Sv/step] | S linear-window "
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
        v = verdict(s_coh, s_all, need, superposes_beyond_window=_superposes)
        verdicts[nm_s] = v
        _scored = s_all if _superposes else s_coh
        _lbl = "all-days" if _superposes else "linear-window (all-days NOT scored)"
        print(f"  {nm_s:8s}: S({_lbl})={_scored:+.4f} Sv vs "
              f"needed {need:+.3f} Sv -> "
              f"{abs(need)/max(abs(_scored),1e-300):.1f}x short -> {v}")
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
                 "injected_sv": {n: arms[n]["injected_sv"] for n in names},
                 "key": args.key,
                 "floor_sv": FLOOR_SV,
                 "R_day1": float(Rref[0]), "R_day2": float(Rref[1]),
                 "linearity": {k: {kk: vv for kk, vv in v.items()
                                   if kk != "reason"} for k, v in lin.items()},
                 "all_linear": all_linear, "lyapunov": lyap,
                 "accounting": acct, "verdicts": verdicts,
                 "post_window_corr": _post_corr, "superposes": _superposes,
                 "gap_full_section": GAP_FULL_SECTION})),
             **{f"R_{n}": rets[n]["R"] for n in names},
             **{f"resp_{n}": rets[n]["resp"] for n in names})
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
