"""#1029: read the exponent-2 vs exponent-3 survival ensemble and apply the
pre-registered rule from scripts/cluster/dycore_1029/exponent_ensemble.sbatch.

Each arm directory ``e{2,3}_s{seed}`` holds the matrix runner's output and the
arm's stdout in ``e{2,3}_s{seed}.log`` next to it.  An arm is VALID only if:
its log shows the exponent override actually reached the model; exactly one
``results.txt`` exists; and it either records a BLOWUP (survival = that day) or
has ``status: PASS`` (right-censored at its ``days``).  A FAIL without a
BLOWUP, a crash, or anything unreadable is INVALID -- never a survivor.

Usage: python issue_1029_exponent_survival.py <job output dir>
Exit status is nonzero when any arm is INVALID or a control fails.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

_BLOWUP = re.compile(r"BLOWUP at step (\d+) \(day ([0-9.]+)\)")
_REF_SEED42_DAYS = 54.6     # 2026-09-23 exp-3 seed-42 record (issue #1029)
SEEDS = (42, 1234, 2718, 31415, 65537, 7, 99, 2024, 8191, 104729)


def read_arm(d: Path, exponent: int) -> tuple[float, bool] | str:
    """(survival days, censored) for one arm dir, or the reason it is INVALID."""
    log = d.parent / f"{d.name}.log"
    try:
        if f"B = eta**{exponent}" not in log.read_text(errors="replace"):
            return "override line missing from log"
        hits = list(d.rglob("results.txt"))
        if len(hits) != 1:
            return f"{len(hits)} results.txt files"
        text = hits[0].read_text()
        m = _BLOWUP.search(text)
        if m:
            return float(m.group(2)), False
        rows = dict(l.split(": ", 1) for l in text.splitlines() if ": " in l)
        if rows.get("status") != "PASS":
            return f"status {rows.get('status')!r} without a BLOWUP"
        return float(rows["days"]), True
    except (OSError, ValueError, KeyError) as exc:
        return f"unreadable: {exc}"


def sign_test_p(wins: int, losses: int) -> float:
    """One-sided exact sign test P(X >= wins), ties dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def verdict(arms: dict[tuple[int, int], tuple[float, bool]]) -> str:
    paired = sorted(s for e, s in arms if e == 3 and (2, s) in arms)
    n = len(paired)
    surv = {e: sum(arms[(e, s)][1] for s in paired) for e in (2, 3)}
    wins = sum(arms[(2, s)][0] > arms[(3, s)][0] for s in paired)
    losses = sum(arms[(2, s)][0] < arms[(3, s)][0] for s in paired)
    p = sign_test_p(wins, losses)

    def fmt(a):
        return f">{a[0]:.0f}" if a[1] else f"{a[0]:.2f}"

    lines = ["seed      exp3      exp2"]
    lines += [f"{s:>6}  {fmt(arms[(3, s)]):>8}  {fmt(arms[(2, s)]):>8}"
              for s in paired]
    lines.append(f"pairs={n} survivors(200 d) exp3={surv[3]} exp2={surv[2]}; "
                 f"exp2 longer in {wins}, shorter in {losses}; "
                 f"one-sided sign test p={p:.3g}")
    if n < 10:
        v = "INCONCLUSIVE (fewer than 10 valid pairs)"
    elif surv[2] >= 8 and surv[3] <= 2:
        v = "CONFIRMED (exponent 2 cures it)"
    elif surv[2] < 8 and p <= 0.05:
        v = "NOT CURED, but exponent 2 delays the blow-up"
    elif sign_test_p(losses, wins) <= 0.05:
        v = "WORSE: exponent 2 brings the blow-up forward"
    elif surv[2] <= 2 and p > 0.2:
        v = "REFUTED (no cure, no detectable effect)"
    else:
        v = "INCONCLUSIVE"
    lines.append(f"VERDICT: {v}")
    return "\n".join(lines)


def main(out: Path) -> int:
    """Every one of the 21 pre-registered arms must be present and valid.

    Exit codes are printed but are not a validity test: a blown-up arm exits
    nonzero by design, and results.txt is written only after the time loop
    finishes, so a run killed later (plotting) still holds its true outcome.
    """
    arms, bad = {}, []
    for s in SEEDS:
        for e in (3, 2):
            r = read_arm(out / f"e{e}_s{s}", e)
            if isinstance(r, str):
                bad.append(f"e{e}_s{s}: {r}")
            else:
                arms[(e, s)] = r
    dup = read_arm(out / "e3_s42dup", 3)
    if isinstance(dup, str):
        bad.append(f"e3_s42dup: {dup}")
    for b in bad:
        print(f"INVALID {b}")
    codes = out / "exit_codes.txt"
    if codes.exists():
        print("exit codes: " + " ".join(codes.read_text().split()))
    same = not isinstance(dup, str) and dup == arms.get((3, 42))
    print(f"CONTROL determinism (exp3 seed 42 run twice): "
          f"{'PASS' if same else 'FAIL'} ({dup} vs {arms.get((3, 42))})")
    if (3, 42) in arms:
        d42 = arms[(3, 42)][0]
        rel = abs(d42 / _REF_SEED42_DAYS - 1)
        print(f"CONTROL drift (reported, not a verdict input): exp3 seed 42 = "
              f"{d42:.2f} d vs {_REF_SEED42_DAYS} d on 2026-09-23 "
              f"({'within' if rel <= 0.3 else 'OUTSIDE'} 30%)")
    print(verdict(arms))
    return 0 if (not bad and same) else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
