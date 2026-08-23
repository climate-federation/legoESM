"""Aggregate N seeded LES->SCM tuning runs into per-scheme mean +/- std.

Each seed jittered the initial parameters (``run_scm_les_turbulence_tuning.py
--seed``); the spread of the tuned joint score across seeds is the
optimization/landscape uncertainty. Reads the ``ranking.csv`` written by each
seed's run and reports, per scheme, the default score (seed-invariant, a
control) and the tuned score as mean +/- std with the min/max envelope.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def _read_ranking(d: Path) -> dict[str, dict]:
    f = d / "ranking.csv"
    if not f.exists():
        return {}
    out = {}
    with f.open() as fh:
        for row in csv.DictReader(fh):
            out[row["scheme"]] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path,
                    help="one output dir per seed")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    per_scheme: dict[str, dict] = {}
    for d in args.dirs:
        for scheme, row in _read_ranking(d).items():
            rec = per_scheme.setdefault(
                scheme, {"tuned": [], "default": [], "status": []})
            # A non-finite / failed arm leaves score_tuned blank; skip it in the
            # stats but count it, so a scheme that failed on some seeds is not
            # silently reported as a clean mean over the survivors.
            st = row.get("score_tuned", "")
            rec["status"].append(row.get("status", "?"))
            if st not in ("", "None"):
                rec["tuned"].append(float(st))
            dv = row.get("score_default", "")
            if dv not in ("", "None"):
                rec["default"].append(float(dv))

    def stats(xs):
        if not xs:
            return None
        n = len(xs)
        m = sum(xs) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / n) if n > 1 else 0.0
        return {"mean": m, "std": sd, "min": min(xs), "max": max(xs), "n": n}

    rows = []
    for scheme, rec in per_scheme.items():
        t = stats(rec["tuned"])
        d0 = stats(rec["default"])
        n_seeds = len(rec["status"])
        n_ok = len(rec["tuned"])
        rows.append({
            "scheme": scheme,
            "default": d0["mean"] if d0 else None,
            "tuned_mean": t["mean"] if t else None,
            "tuned_std": t["std"] if t else None,
            "tuned_min": t["min"] if t else None,
            "tuned_max": t["max"] if t else None,
            "n_seeds": n_seeds, "n_tuned_ok": n_ok,
        })
    rows.sort(key=lambda r: (r["tuned_mean"] is None, r["tuned_mean"] or 0.0))

    print(f"{'scheme':<18}{'default':>9}{'tuned mean':>12}{'std':>9}"
          f"{'[min':>9}{'max]':>9}{'seeds':>7}")
    for r in rows:
        if r["tuned_mean"] is None:
            print(f"{r['scheme']:<18}{'--':>9}{'FAILED all seeds':>30}")
            continue
        flag = "" if r["n_tuned_ok"] == r["n_seeds"] else \
            f"  ({r['n_tuned_ok']}/{r['n_seeds']} ok)"
        print(f"{r['scheme']:<18}{r['default']:>9.4f}{r['tuned_mean']:>12.4f}"
              f"{r['tuned_std']:>9.4f}{r['tuned_min']:>9.4f}{r['tuned_max']:>9.4f}"
              f"{r['n_seeds']:>7}{flag}")

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
