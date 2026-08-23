#!/usr/bin/env python
"""#1455 KICK-ASYMMETRY DISCRIMINATOR: is legoESM's ~100x larger perturbation
spread the leapfrog COMPUTATIONAL MODE, or real error growth?

Pre-registration: ``PREREG_kick_asymmetry.md``, committed BEFORE any member of
the two-level arm ran. Read it before citing any number this prints.

THE QUESTION.  ``dino_verdict360_result.md`` recorded, and did not
investigate, that legoESM amplifies an identical 1e-14 relative temperature
kick 78-435x more than NEMO by day 90 while the two GROWTH RATES match -- so
the offset is born in the first 30 days.  The recorded kick perturbs the NOW
leapfrog level only, which is the maximal excitation of the computational
mode; NEMO's spread even DECAYS from day 30 to 60, the signature of a filter
removing that mode.  If the asymmetry is the computational mode, kicking BOTH
time levels with the same draw removes it from both models and the ratio
collapses.

THE ONE VARIABLE.  Same restart, same card, same namelist, same certified
binary, same seeds, same 90-day horizon, same reductions, same scorer.  Only
the perturbation convention differs:

  arm ONE-LEVEL  (recorded, already on disk, NOT re-run)
      legoESM  /tmp/dino_verdict360/m{0..3}   days 0-90 of the 360-day members
      NEMO     RUN_VERDICT360_M{0..3}         kt 5760-8640
  arm TWO-LEVEL  (new)
      legoESM  /tmp/dino_kick2/m{0..3}        kamm_twin_90d --perturb-both-levels
      NEMO     RUN_KICK2_M{0..3}              perturb_nemo_tn_90d --both-levels

The first 90 days of a 360-day integration are the same trajectory as a 90-day
integration; control C4 tests exactly that by comparing the two arms'
UNPERTURBED member 0, which is the same run specification on both.

EVERYTHING IS IMPORTED.  Metrics ``verdict360.all_metrics`` (which is itself
the acceptance gate's five plus the two band reductions plus the three
latitude groups), loaders ``verdict360.lego_state`` / ``nemo_state``, spread
``floor90_ensemble.spread``, the NEMO kick ``perturb_nemo_tn_90d.perturb``.
Nothing numeric is re-derived here.

WHAT IT IS NOT.  It prints tables and the ONE mechanical comparison the
pre-registration registered.  It edits no floor, no recipe, no card.

Usage
-----
  kick_asymmetry.py --self-check              # arithmetic, no runs needed
  kick_asymmetry.py --setup-nemo              # build the 4 two-level NEMO dirs
  kick_asymmetry.py --run-nemo                # launch them (serial, ~8 min each)
  kick_asymmetry.py --run-lego --gpu 0        # launch the 4 legoESM members
  kick_asymmetry.py                           # score both arms
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))
import acc_thermal_wind as A  # noqa: E402  mesh + masks
import acceptance_gate_90d as G  # noqa: E402  KT_RESTART / STEPS_PER_DAY
import floor90_ensemble as F  # noqa: E402  spread(), head_sha(), member_sha()
import perturb_nemo_tn_90d as P  # noqa: E402  the committed NEMO kick
import verdict360 as V  # noqa: E402  all_metrics, loaders, KEYS

N_DAYS = 90
SEEDS = V.SEEDS                        # (None, 1, 2, 3) -- the recorded convention
N_MEM = len(SEEDS)
CURVE_DAYS = tuple(range(10, N_DAYS + 1, 10))
SNAP_GRID = tuple(range(0, N_DAYS + 1, 10))
SCORE_DAY = 90                         # the day the recorded asymmetry was quoted at
DECAY_PAIR = (30, 60)                  # the recorded NEMO spread decay
KEYS = V.KEYS
LABELS = V.LABELS

KT_END = G.KT_RESTART + N_DAYS * G.STEPS_PER_DAY          # 8640
LEGO_DIR_DEFAULT = "/tmp/dino_kick2"
NEMO_SRC = V.NEMO_SRC                  # RUN_90D_TWIN: namelist donor, ALREADY 90 days
NEMO_CERT = V.NEMO_CERT
MPIRUN = V.MPIRUN
N_RANK = V.N_RANK

# The bars, from PREREG_kick_asymmetry.md section 4.  At n=4 the log of a
# spread estimate has sd 1/sqrt(2(n-1)) = 0.408; F is built from FOUR
# independent spread estimates, so log F carries sqrt(4) x 0.408 = 0.816 and a
# two-sided 95% band is exp(1.96 x 0.8165) = 4.955.  An F below that is
# NOT distinguishable from no change.  The pre-registration rounded this
# to '5.0' in prose and was amended to the exact value before any member
# of the two-level arm ran; the bar is COMPUTED here, never typed.
LOG_SD_SPREAD = float(1.0 / np.sqrt(2 * (N_MEM - 1)))
F_RESOLVABLE = float(np.exp(1.96 * np.sqrt(4.0) * LOG_SD_SPREAD))
R2_TWO_SIDED = 10.0 ** V.ONE_SIDED_DECADES     # 10x: the band where an RSS floor is two-sided
QUANTUM_MARGIN = V.QUANTUM_MARGIN              # 10x storage quantum or the metric is flagged


def member_name(i):
    return V.member_name(i)


def nemo_dir(i):
    return f"{A.DINO}/RUN_KICK2_M{i}"


def lego_npz(out_dir, i):
    return f"{out_dir}/{member_name(i)}.npz"


def lego_log(out_dir, i):
    return f"{out_dir}/{member_name(i)}.log"


# ------------------------------------------------------------- legoESM runs ---
def lego_cmd(out_dir, i):
    """``floor90_ensemble``'s own member command at 90 days, plus the 10-day
    snapshot grid this probe reads and, for the perturbed members, the
    two-level kick.  No other option flags: the card's defaults are what is
    being measured."""
    cmd = [sys.executable, f"{_DIR}/run_fp64.py", f"{_DIR}/kamm_twin_90d.py",
           "nemo_dino_kamm_mlf", lego_npz(out_dir, i),
           "--days", str(N_DAYS), "--bridge-before", "--save-3d",
           "--snap-days", ",".join(str(d) for d in SNAP_GRID)]
    if SEEDS[i] is not None:
        cmd += ["--perturb-seed", str(SEEDS[i]), "--perturb-both-levels"]
    return cmd


def run_lego(out_dir, gpu):
    """Sequential, one GPU.  90 days is ~3 min per member, so the two-GPU
    slot machinery verdict360 needs at 360 days buys nothing here and its
    refill bug class is not worth re-importing."""
    os.makedirs(out_dir, exist_ok=True)
    want = V.launch_sha(out_dir)
    for i in range(N_MEM):
        npz = lego_npz(out_dir, i)
        if os.path.exists(npz):
            got = F.member_sha(lego_log(out_dir, i))
            if got != want:
                raise SystemExit(
                    f"{npz} exists but its log records HEAD={got}, not the "
                    f"ensemble's launch HEAD={want}. Members at different "
                    f"source revisions measure the revision as well as the "
                    f"noise. Delete it to regenerate.")
            print(f"[skip] {npz} exists at the ensemble's launch HEAD", flush=True)
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_ENABLE_X64="1",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false")
        cmd = lego_cmd(out_dir, i)
        print("RUN:", " ".join(cmd), f"[CUDA_VISIBLE_DEVICES={gpu}]", flush=True)
        with open(lego_log(out_dir, i), "w") as fh:
            rc = subprocess.run(cmd, env=env, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        print(f"[done] {member_name(i)} rc={rc}", flush=True)


def lego_completed(log_path):
    """The harness's OWN success line -- an exit code is not evidence."""
    if not os.path.exists(log_path):
        return False, "no log"
    with open(log_path, errors="replace") as fh:
        txt = fh.read()
    if f"DONE nsteps={G.STEPS_PER_DAY * N_DAYS} STABLE=True" in txt:
        return True, "OK"
    for ln in txt.splitlines():
        if ln.startswith("BLOWUP") or ln.startswith("DONE "):
            return False, ln.strip()
    return False, "no DONE line (still running or killed)"


# ---------------------------------------------------------------- NEMO runs ---
def setup_nemo():
    """Build the four two-level NEMO run dirs.

    ZERO namelist edits.  ``RUN_90D_TWIN``'s ``namelist_cfg`` is already the
    90-day configuration (``nn_it000=5761``, ``nn_itend=8640``,
    ``nn_stock=320`` -> 10-day restart dumps), which is exactly this arm's
    horizon and sampling.  verdict360 had to rewrite ``nn_itend`` because it
    ran 360 days; this arm does not, so the namelist is COPIED and the
    'one variable' claim needs no line-editing to be true.
    """
    for i in range(N_MEM):
        d = nemo_dir(i)
        stale = (glob.glob(f"{d}/DINO_*_restart_*.nc") + glob.glob(f"{d}/time.step")
                 + glob.glob(f"{d}/ocean.output"))
        if stale:
            raise SystemExit(
                f"{d} already carries {len(stale)} output file(s) from an "
                f"earlier run (e.g. {os.path.basename(stale[0])}). Scoring a "
                f"mix of old and new states is a silent confound -- clear the "
                f"directory deliberately, or point at a fresh one.")
        os.makedirs(d, exist_ok=True)
        for f in ("namelist_cfg", "namelist_ref"):
            shutil.copy2(f"{NEMO_SRC}/{f}", f"{d}/{f}")
        link = f"{d}/nemo"
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(NEMO_CERT, link)
        if SEEDS[i] is None:
            shutil.copy2(P.SRC, f"{d}/{P.SRC.name}")
            print(f"[{member_name(i)}] {d}: unperturbed restart copied")
            continue
        rep = P.perturb(SEEDS[i], Path(d), both_levels=True)
        # C5, the perturbation receipt, printed AND written next to the run it
        # justifies -- a receipt that only ever existed on a terminal is not a
        # receipt.
        lines = [f"targets={rep['targets']}",
                 f"other_vars_bit_identical={rep['other_vars_bit_identical']}",
                 f"max_other_var_diff={rep['max_other_var_diff']:.3e}",
                 f"attrs_bit_identical={rep['attrs_bit_identical']}"]
        for name, pt in rep["per_target"].items():
            lines.append(f"{name}: max|d|={pt['max_abs_diff']:.6e} "
                         f"max_rel={pt['max_rel_diff']:.6e} "
                         f"n_changed={pt['n_changed']}/{pt['n_total']}")
        with open(f"{d}/PERTURB_RECEIPT.txt", "w") as fh:
            fh.write(f"seed={SEEDS[i]} both_levels=True\n"
                     + "\n".join(lines) + "\n")
        print(f"[{member_name(i)}] {d}: " + "  ".join(lines))
    print(f"{N_MEM} dirs built; namelist_cfg/namelist_ref copied BYTE-IDENTICAL "
          f"from {NEMO_SRC} (already the 90-day configuration)")


def run_nemo():
    """Serial, 16 ranks -- measured 24x faster in aggregate than 4-wide on this
    machine (``verdict360.run_nemo``'s recorded measurement) and the same
    configuration the recorded 90-day twin ran under."""
    for i in range(N_MEM):
        d = nemo_dir(i)
        if not os.path.exists(f"{d}/nemo"):
            raise SystemExit(f"{d} not set up -- run --setup-nemo first")
        cmd = [MPIRUN, "-np", str(N_RANK), "./nemo"]
        print("RUN:", " ".join(cmd), f"[cwd={d}]", flush=True)
        with open(f"{d}/run_kick2_m{i}.log", "w") as fh:
            rc = subprocess.run(cmd, cwd=d, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        print(f"[done] NEMO {member_name(i)} rc={rc}", flush=True)


def nemo_completed(run_dir, log_glob):
    ts = f"{run_dir}/time.step"
    if not os.path.exists(ts):
        return False, "no time.step"
    with open(ts) as fh:
        got = int(fh.read().strip())
    if got < KT_END:
        return False, f"time.step={got}, expected >= {KT_END}"
    oo = f"{run_dir}/ocean.output"
    if os.path.exists(oo):
        with open(oo, errors="replace") as fh:
            if "E R R O R" in fh.read():
                return False, "E R R O R in ocean.output"
    logs = glob.glob(f"{run_dir}/{log_glob}")
    if not logs:
        return False, f"no run log matching {log_glob}"
    with open(logs[0], errors="replace") as fh:
        tail = fh.read()[-200:]
    if "STOP 0" not in tail:
        return False, f"no 'STOP 0' in {os.path.basename(logs[0])} tail: {tail!r}"
    return True, "OK"


# ------------------------------------------------------------------ scoring ---
class Arm:
    """One perturbation convention: where its eight runs live and how they are
    checked for completion.  ``time.step`` is compared with ``>=`` because the
    one-level arm's NEMO members ran on to day 360 -- their day-90 restart dump
    is the same state either way."""

    def __init__(self, name, lego_dir, nemo_dir_fn, lego_nsteps, nemo_log_glob):
        self.name = name
        self.lego_dir = lego_dir
        self.nemo_dir_fn = nemo_dir_fn
        self.lego_nsteps = lego_nsteps
        self.nemo_log_glob = nemo_log_glob

    def lego_done(self, i):
        path = lego_log(self.lego_dir, i)
        if not os.path.exists(path):
            return False, "no log"
        with open(path, errors="replace") as fh:
            txt = fh.read()
        if f"DONE nsteps={self.lego_nsteps} STABLE=True" in txt:
            return True, "OK"
        for ln in txt.splitlines():
            if ln.startswith("BLOWUP") or ln.startswith("DONE "):
                return False, ln.strip()
        return False, "no DONE line"

    def nemo_done(self, i):
        return nemo_completed(self.nemo_dir_fn(i), self.nemo_log_glob)


def arms(lego_dir):
    one = Arm("one-level", V.LEGO_DIR_DEFAULT, V.nemo_dir,
              G.STEPS_PER_DAY * V.N_DAYS, "run_verdict360_m*.log")
    two = Arm("two-level", lego_dir, nemo_dir,
              G.STEPS_PER_DAY * N_DAYS, "run_kick2_m*.log")
    return one, two


def collect(arm, days):
    """rows[side][member][day] -> metric dict.  NaN anywhere is FATAL."""
    wet = A.tmask
    rows = {"lego": {}, "nemo": {}}
    for i in range(N_MEM):
        rows["lego"][i], rows["nemo"][i] = {}, {}
        npz = lego_npz(arm.lego_dir, i)
        for day in days:
            ml = V.all_metrics(V.lego_state(npz, day), wet)
            mn = V.all_metrics(V.nemo_state(arm.nemo_dir_fn(i), day), wet)
            for tag, m in (("lego", ml), ("nemo", mn)):
                bad = [k for k in KEYS if not np.isfinite(m[k])]
                if bad:
                    raise SystemExit(f"non-finite {bad} for {tag} "
                                     f"{member_name(i)} day {day} in arm "
                                     f"{arm.name} -- a FINDING, not a member "
                                     f"to drop")
            rows["lego"][i][day] = ml
            rows["nemo"][i][day] = mn
        print(f"  [{arm.name}] scored {member_name(i)} "
              f"({len(days)} days, both sides)", flush=True)
    return rows


def spread_at(rows, side, key, day, stat=1):
    """stat 0 = max-pairwise range, 1 = sample std (the primary)."""
    return F.spread([rows[side][i][day][key] for i in range(N_MEM)])[stat]


def ratio(rows, key, day, stat=1):
    """legoESM spread / NEMO spread.  NaN if NEMO's spread is exactly zero --
    never a large finite number standing in for a division by zero."""
    ns = spread_at(rows, "nemo", key, day, stat)
    ls = spread_at(rows, "lego", key, day, stat)
    return float("nan") if ns == 0.0 else ls / ns


def _self_check():
    """The arithmetic this probe owns, none of which is imported: the
    resolvability bar, the ratio, and the median-with-exclusions rule."""
    assert abs(LOG_SD_SPREAD - 0.40825) < 1e-4, LOG_SD_SPREAD
    assert abs(F_RESOLVABLE - 4.9547) < 1e-3, F_RESOLVABLE
    assert 4.5 < F_RESOLVABLE < 5.5, F_RESOLVABLE
    assert R2_TWO_SIDED == 10.0, R2_TWO_SIDED

    # ratio(): a zero NEMO spread must give NaN, not inf and not a big float.
    fake = {"lego": {i: {90: {"acc": float(i)}} for i in range(N_MEM)},
            "nemo": {i: {90: {"acc": 7.0}} for i in range(N_MEM)}}
    assert np.isnan(ratio(fake, "acc", 90)), ratio(fake, "acc", 90)
    fake["nemo"] = {i: {90: {"acc": 7.0 + i}} for i in range(N_MEM)}
    assert abs(ratio(fake, "acc", 90) - 1.0) < 1e-12, ratio(fake, "acc", 90)
    fake["lego"] = {i: {90: {"acc": 3.0 * i}} for i in range(N_MEM)}
    assert abs(ratio(fake, "acc", 90) - 3.0) < 1e-12, ratio(fake, "acc", 90)

    # verdict(): the three registered outcomes, and only those.
    assert verdict(9.0, 5.0) == "CONFIRMS H1"
    assert verdict(9.0, 50.0) == "PARTIAL"
    assert verdict(4.9, 3.0) == "REFUTES H1"      # F below the bar decides alone
    assert verdict(4.9, 500.0) == "REFUTES H1"
    assert verdict(float("nan"), 5.0) == "INDETERMINATE"
    assert verdict(9.0, float("nan")) == "INDETERMINATE"
    # exactly at the bar counts as resolvable (>=, as registered)
    assert verdict(F_RESOLVABLE, 10.0) == "CONFIRMS H1"
    assert verdict(F_RESOLVABLE, 10.0001) == "PARTIAL"

    # median_over(): NaN must PROPAGATE out of an included metric rather than
    # being skipped -- np.median does that, np.nanmedian would not.
    assert abs(median_over({"a": 1.0, "b": 3.0, "c": 2.0}, set()) - 2.0) < 1e-12
    assert np.isnan(median_over({"a": 1.0, "b": float("nan")}, set()))
    # ...but a metric EXCLUDED for sitting at the storage quantum is dropped
    # before the median, which is the only sanctioned exclusion.
    assert abs(median_over({"a": 1.0, "b": float("nan"), "c": 3.0}, {"b"}) - 2.0) < 1e-12
    try:
        median_over({"a": 1.0}, {"a"})
    except SystemExit:
        pass
    else:                                          # pragma: no cover
        raise AssertionError("an empty median must raise, not return NaN")
    print(f"SELF-CHECK OK: n={N_MEM} -> log-spread sd {LOG_SD_SPREAD:.4f}, "
          f"collapse factor resolvable at {F_RESOLVABLE:.2f}x, two-sided band "
          f"{R2_TWO_SIDED:.0f}x; ratio() returns NaN on a zero denominator; "
          f"median propagates NaN from included metrics and raises when empty")
    return 0


def verdict(med_f, med_r2):
    """PREREG section 4, spelled ONCE and unit-tested."""
    if not (np.isfinite(med_f) and np.isfinite(med_r2)):
        return "INDETERMINATE"
    if med_f < F_RESOLVABLE:
        return "REFUTES H1"
    return "CONFIRMS H1" if med_r2 <= R2_TWO_SIDED else "PARTIAL"


def median_over(values, excluded):
    """Median over the metrics that are not excluded, NaN-propagating."""
    keep = [v for k, v in values.items() if k not in excluded]
    if not keep:
        raise SystemExit("every metric was excluded -- there is nothing to "
                         "take a median of, and a median of nothing is not a "
                         "verdict")
    return float(np.median(np.asarray(keep, dtype=np.float64)))


# ------------------------------------------------------------------- tables ---
def controls(one, two, rows_one, rows_two, quantum):
    print("\n" + "=" * 110)
    print("CONTROLS (PREREG section 7) -- read before any number below")
    print("=" * 110)

    # C1 completion, both arms, both sides, each side's OWN success line.
    for arm in (one, two):
        for i in range(N_MEM):
            ok_l, why_l = arm.lego_done(i)
            ok_n, why_n = arm.nemo_done(i)
            print(f"[C1 {arm.name:<9} {member_name(i):<11}] "
                  f"lego {'COMPLETE' if ok_l else 'INCOMPLETE'} ({why_l})   "
                  f"NEMO {'COMPLETE' if ok_n else 'INCOMPLETE'} ({why_n})")
            if not (ok_l and ok_n):
                raise SystemExit(f"{arm.name} {member_name(i)} did not "
                                 f"complete: lego={why_l} nemo={why_n}. A "
                                 f"member that did not finish is a FINDING.")

    # C2 one source revision per legoESM ensemble.
    for arm in (one, two):
        shas = {member_name(i): F.member_sha(lego_log(arm.lego_dir, i))
                for i in range(N_MEM)}
        if len(set(shas.values())) != 1 or None in shas.values():
            raise SystemExit(f"{arm.name} legoESM members sit at different "
                             f"source revisions: {shas}")
        print(f"[C2 {arm.name:<9}] all {N_MEM} legoESM members at HEAD="
              f"{list(shas.values())[0][:12]}")

    # C2b: the same certified NEMO binary everywhere, both arms.
    for arm, dfn in ((one, V.nemo_dir), (two, nemo_dir)):
        for i in range(N_MEM):
            real = os.path.realpath(f"{dfn(i)}/nemo")
            if real != os.path.realpath(NEMO_CERT):
                raise SystemExit(f"{arm.name} {member_name(i)} nemo -> {real}, "
                                 f"not the certified {NEMO_CERT}")
        print(f"[C2b {arm.name:<8}] all {N_MEM} NEMO members on the certified "
              f"binary")

    # C3 no metric identical across a whole ensemble.
    for arm, rows in ((one, rows_one), (two, rows_two)):
        for side in ("lego", "nemo"):
            for k in KEYS:
                vals = [rows[side][i][SCORE_DAY][k] for i in range(N_MEM)]
                if len(set(vals)) < 2:
                    raise SystemExit(
                        f"{arm.name}/{side}: metric {k!r} is IDENTICAL across "
                        f"ALL {N_MEM} members ({vals}) -- the kick never "
                        f"reached it, so its spread is zero by construction")
    print("[C3] every metric separates at least two members in all four "
          "ensembles")

    # C4 the tool change is inert on the unperturbed path: the two arms'
    # member 0 is the SAME run specification.  Compared against the two-level
    # arm's own spread, because that is the quantity every ratio below is
    # built from -- a control-to-control difference of that size means the
    # arms differ in something other than the kick.
    print(f"\n[C4] arm-to-arm control drift at day {SCORE_DAY} "
          f"(same run specification on both arms)")
    print(f"     {'metric':<38}{'one-level m0':>18}{'two-level m0':>18}"
          f"{'|drift|':>13}{'two-level std':>15}{'drift/std':>11}")
    worst = 0.0
    for k in KEYS:
        a = rows_one["lego"][0][SCORE_DAY][k]
        b = rows_two["lego"][0][SCORE_DAY][k]
        sd = spread_at(rows_two, "lego", k, SCORE_DAY)
        r = float("nan") if sd == 0 else abs(a - b) / sd
        if np.isfinite(r):
            worst = max(worst, r)
        print(f"     {LABELS[k]:<38}{a:>18.9f}{b:>18.9f}{abs(a - b):>13.4e}"
              f"{sd:>15.4e}{r:>11.3f}")
    for k in KEYS:
        a = rows_one["nemo"][0][SCORE_DAY][k]
        b = rows_two["nemo"][0][SCORE_DAY][k]
        sd = spread_at(rows_two, "nemo", k, SCORE_DAY)
        r = float("nan") if sd == 0 else abs(a - b) / sd
        if np.isfinite(r):
            worst = max(worst, r)
    print(f"     worst drift/std over both sides and all {len(KEYS)} metrics: "
          f"{worst:.3f}")
    if worst >= 1.0:
        raise SystemExit(
            f"C4 FAILED: the two arms' UNPERTURBED controls differ by "
            f"{worst:.2f}x the two-level ensemble spread. The arms then differ "
            f"in something other than the kick convention and no ratio between "
            f"them is interpretable. Find that difference before reading any "
            f"number in this run.")
    print("[C4] PASS -- the arms' controls agree to well within one ensemble "
          "spread, so the only difference between them is the kick")

    # C5 receipts: the legoESM PERTURB stamp from each member's own log, and
    # the NEMO receipt file written at setup.
    print("\n[C5] perturbation receipts, as recorded by the runs themselves")
    for arm in (one, two):
        for i in range(1, N_MEM):
            stamp = "(none)"
            path = lego_log(arm.lego_dir, i)
            if os.path.exists(path):
                with open(path, errors="replace") as fh:
                    for ln in fh:
                        if ln.startswith("PERTURB seed="):
                            stamp = ln.strip()
            print(f"     lego {arm.name:<9} {member_name(i):<11} {stamp}")
    for i in range(1, N_MEM):
        rp = f"{nemo_dir(i)}/PERTURB_RECEIPT.txt"
        if os.path.exists(rp):
            with open(rp) as fh:
                body = " | ".join(ln.strip() for ln in fh if ln.strip())
            print(f"     NEMO two-level {member_name(i):<11} {body}")
        else:
            print(f"     NEMO two-level {member_name(i):<11} (no receipt file)")

    # C7 storage quantum: which metrics are resolved at all.
    print(f"\n[C7] float32 storage quantum of the legoESM npz at day "
          f"{SCORE_DAY}, and the two-level spread against it")
    print(f"     {'metric':<38}{'quantum':>13}{'two-level lego std':>21}"
          f"{'std/quantum':>13}  flag")
    excluded = set()
    for k in KEYS:
        sd = spread_at(rows_two, "lego", k, SCORE_DAY)
        q = quantum[k]
        r = float("inf") if q == 0 else sd / q
        flag = ""
        if r < QUANTUM_MARGIN:
            excluded.add(k)
            flag = f"q  EXCLUDED from the median (below {QUANTUM_MARGIN:g}x)"
        print(f"     {LABELS[k]:<38}{q:>13.3e}{sd:>21.4e}{r:>13.2f}  {flag}")
    return excluded


def growth_table(arm):
    """C3: did the kick reach the integrated state?  max|dT| of each perturbed
    member against its OWN side's control, by day."""
    from rebuild_nemo_restart import rebuild
    days = (10, 30, 60, 90)
    print(f"\n--- GROWTH CONTROL, arm {arm.name}: max|dT| vs own-side control [K] ---")
    print("    (legoESM snapshots are float32: quantum ~2e-06 K on a ~20 K "
          "field, so an early 0.0 is storage, not physics)")
    for side in ("lego", "nemo"):
        print(f"  {side}")
        print(f"    {'member':<14}" + "".join(f"{'day ' + str(d):>16}" for d in days))
        for i in range(1, N_MEM):
            cells = []
            for day in days:
                if side == "lego":
                    key = f"T3d_day{day}"
                    da = np.load(lego_npz(arm.lego_dir, i))
                    db = np.load(lego_npz(arm.lego_dir, 0))
                    if key not in da.files or key not in db.files:
                        cells.append(f"{'(absent)':>16}")
                        continue
                    a = np.asarray(da[key], dtype=np.float64)
                    b = np.asarray(db[key], dtype=np.float64)
                else:
                    kt = G.KT_RESTART + day * G.STEPS_PER_DAY
                    a = np.asarray(rebuild(f"{arm.nemo_dir_fn(i)}/DINO_{kt:08d}"
                                           f"_restart_*.nc", ["tn"])["tn"],
                                   dtype=np.float64)
                    b = np.asarray(rebuild(f"{arm.nemo_dir_fn(0)}/DINO_{kt:08d}"
                                           f"_restart_*.nc", ["tn"])["tn"],
                                   dtype=np.float64)
                v = np.abs(a - b)
                if not np.isfinite(v).all():
                    raise SystemExit(f"non-finite T at day {day}, {side} "
                                     f"{member_name(i)}, arm {arm.name} -- a "
                                     f"blown member is a finding")
                cells.append(f"{float(v.max()):>16.4e}")
            print(f"    {member_name(i):<14}" + "".join(cells))


def spread_curves(rows_one, rows_two):
    print("\n--- SPREAD(t): single-run ensemble spread by day, both arms ---")
    print("    sample std (the primary statistic).  At n=4 every value carries "
          f"~{100 * LOG_SD_SPREAD:.0f}% relative standard error, so read a")
    print("    factor of two, not three digits.  legoESM states are stored "
          "float32, NEMO states are read float64.")
    for side in ("lego", "nemo"):
        for tag, rows in (("one-level", rows_one), ("two-level", rows_two)):
            print(f"\n  {side}  /  {tag}")
            print(f"    {'metric':<38}"
                  + "".join(f"{'d' + str(d):>12}" for d in CURVE_DAYS))
            for k in KEYS:
                cells = "".join(f"{spread_at(rows, side, k, d):>12.4e}"
                                for d in CURVE_DAYS)
                print(f"    {LABELS[k]:<38}{cells}")


def ratio_table(rows_one, rows_two, excluded):
    print("\n" + "=" * 110)
    print(f"THE RATIO TABLE -- legoESM spread / NEMO spread at day {SCORE_DAY}")
    print("=" * 110)
    print("F = R1/R2 is the COLLAPSE FACTOR: how much removing the leapfrog "
          "time-level mismatch shrank the")
    print(f"asymmetry.  PREREG section 4: F below {F_RESOLVABLE:.2f} is not "
          f"distinguishable from no change at n={N_MEM}.")
    print(f"{'metric':<38}{'R1 one-level':>15}{'R2 two-level':>15}"
          f"{'F = R1/R2':>13}  flag")
    r1, r2, ff = {}, {}, {}
    for k in KEYS:
        r1[k] = ratio(rows_one, k, SCORE_DAY)
        r2[k] = ratio(rows_two, k, SCORE_DAY)
        ff[k] = float("nan") if not np.isfinite(r2[k]) or r2[k] == 0 else r1[k] / r2[k]
        print(f"{LABELS[k]:<38}{r1[k]:>15.4g}{r2[k]:>15.4g}{ff[k]:>13.4g}"
              f"  {'q (excluded)' if k in excluded else ''}")
    med_r1 = median_over(r1, excluded)
    med_r2 = median_over(r2, excluded)
    med_f = median_over(ff, excluded)
    n_used = len(KEYS) - len(excluded)
    print(f"{'MEDIAN over the ' + str(n_used) + ' scored metrics':<38}"
          f"{med_r1:>15.4g}{med_r2:>15.4g}{med_f:>13.4g}")
    return med_r1, med_r2, med_f


def decay_table(rows_one, rows_two):
    d0, d1 = DECAY_PAIR
    print("\n" + "=" * 110)
    print(f"THE DECAY SIGNATURE -- NEMO spread ratio s({d1})/s({d0}), "
          f"independent of the ratio table")
    print("=" * 110)
    print("PREREG section 5: a spread that DECAYS and then grows is a damped "
          "mode plus physical growth.  If the")
    print("one-level kick's computational mode is what NEMO's filter removes, "
          "the decay must not survive a kick")
    print("that carries no time-level mismatch.  legoESM is printed beside it "
          "for contrast, not as a criterion.")
    print(f"{'metric':<38}{'NEMO one-lvl':>14}{'NEMO two-lvl':>14}"
          f"{'lego one-lvl':>14}{'lego two-lvl':>14}  decay -> ?")
    flips, persists = [], []
    for k in KEYS:
        def _r(rows, side):
            a = spread_at(rows, side, k, d0)
            b = spread_at(rows, side, k, d1)
            return float("nan") if a == 0 else b / a
        n1, n2 = _r(rows_one, "nemo"), _r(rows_two, "nemo")
        l1, l2 = _r(rows_one, "lego"), _r(rows_two, "lego")
        note = ""
        if np.isfinite(n1) and n1 < 1.0 and np.isfinite(n2):
            if n2 >= 1.0:
                note = "decay GONE"
                flips.append(k)
            else:
                note = "decay PERSISTS"
                persists.append(k)
        print(f"{LABELS[k]:<38}{n1:>14.4g}{n2:>14.4g}{l1:>14.4g}{l2:>14.4g}"
              f"  {note}")
    print(f"\nNEMO metrics that decayed under the one-level kick: "
          f"{len(flips) + len(persists)} of {len(KEYS)}; of those, "
          f"{len(flips)} lost the decay under the two-level kick and "
          f"{len(persists)} kept it.")
    return flips, persists


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=LEGO_DIR_DEFAULT,
                   help="directory for the two-level legoESM members")
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for --run-lego")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--setup-nemo", action="store_true")
    p.add_argument("--run-nemo", action="store_true")
    p.add_argument("--run-lego", action="store_true")
    args = p.parse_args(argv)

    if args.self_check:
        return _self_check()
    if args.setup_nemo:
        setup_nemo()
        return 0
    if args.run_nemo:
        run_nemo()
        return 0
    if args.run_lego:
        run_lego(args.dir, args.gpu)
        return 0

    _self_check()
    one, two = arms(args.dir)
    print("\nscoring both arms (this loads every state exactly once)")
    rows_one = collect(one, CURVE_DAYS)
    rows_two = collect(two, CURVE_DAYS)

    quantum = V.fp32_quantum(V.lego_state(lego_npz(two.lego_dir, 0), SCORE_DAY),
                             A.tmask)
    excluded = controls(one, two, rows_one, rows_two, quantum)
    for arm in (one, two):
        growth_table(arm)
    spread_curves(rows_one, rows_two)
    med_r1, med_r2, med_f = ratio_table(rows_one, rows_two, excluded)
    flips, persists = decay_table(rows_one, rows_two)

    print("\n" + "=" * 110)
    print("PRE-REGISTERED COMPARISON (PREREG_kick_asymmetry.md section 4)")
    print("=" * 110)
    print(f"  median R1 (one-level lego/NEMO spread ratio) = {med_r1:.4g}")
    print(f"  median R2 (two-level lego/NEMO spread ratio) = {med_r2:.4g}")
    print(f"  median F  (collapse factor)                  = {med_f:.4g}"
          f"    resolvable at >= {F_RESOLVABLE:.2f}")
    print(f"  two-sided band for R2                        = <= "
          f"{R2_TWO_SIDED:.0f}")
    print(f"  registered outcome: {verdict(med_f, med_r2)}")
    print(f"  decay signature: {len(flips)} of {len(flips) + len(persists)} "
          f"decaying NEMO metrics lost the decay under the two-level kick")
    print("\n(the outcome above is the mechanical application of a rule fixed "
          "before these numbers existed;")
    print(" its interpretation belongs in the result document, not here)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
