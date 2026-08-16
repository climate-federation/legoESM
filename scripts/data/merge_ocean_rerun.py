#!/usr/bin/env python
"""Merge the phase-1 ocean summary with rerun summaries.

Each rerun wrote its own summary.json under results/ocean_rerun{,/<case>}/.
We overwrite phase-1 records for cases that were rerun, then save the merged
summary back to results/ocean/summary.json for the final validator pass.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("results")
BASE = ROOT / "ocean" / "summary.json"
BACKUP = ROOT / "ocean" / "summary_phase1.json"
OUT = BASE  # overwrite

RERUN_DIRS = [
    ROOT / "ocean_rerun" / "rest_state_uniform",
    ROOT / "ocean_rerun" / "barotropic_double_gyre",
    ROOT / "ocean_rerun" / "global_barotropic_wind",
    ROOT / "ocean_rerun" / "lock_exchange",
    ROOT / "ocean_rerun" / "stommel_gyre_tracer",
    ROOT / "ocean_rerun2" / "phillips",
    ROOT / "ocean_rerun2" / "igw",
    ROOT / "ocean_rerun_eady5",
]


def key(r: dict) -> tuple:
    return (r.get("test") or r.get("case"), r.get("grid"), r.get("resolution"))


def main() -> None:
    if not BASE.is_file():
        raise SystemExit(f"Missing {BASE}")
    phase1 = json.loads(BACKUP.read_text() if BACKUP.is_file()
                        else BASE.read_text())
    by_key = {key(r): r for r in phase1["results"]}
    replaced = []
    for d in RERUN_DIRS:
        sj = d / "summary.json"
        if not sj.is_file():
            print(f"[warn] missing {sj}")
            continue
        data = json.loads(sj.read_text())
        for r in data["results"]:
            k = key(r)
            if k in by_key:
                by_key[k] = r
                replaced.append(k)
            else:
                by_key[k] = r
    merged = {
        "results": list(by_key.values()),
        "total_wall_time": sum(float(r.get("wall_time") or 0)
                               for r in by_key.values()),
        "n_pass": sum(1 for r in by_key.values() if r["status"] == "PASS"),
        "n_fail": sum(1 for r in by_key.values() if r["status"] == "FAIL"),
        "n_skip": sum(1 for r in by_key.values() if r["status"] == "SKIP"),
        "n_error": sum(1 for r in by_key.values() if r["status"] == "ERROR"),
        # XFAIL/XPASS must be carried too: a count-based consumer that sees
        # only n_fail/n_error would report a clean merge while an XPASS -- a
        # known-failure entry that has started passing, i.e. a stale waiver --
        # sits unaccounted for (codex review, #1609).
        "n_xfail": sum(1 for r in by_key.values() if r["status"] == "XFAIL"),
        "n_xpass": sum(1 for r in by_key.values() if r["status"] == "XPASS"),
        "quick_mode": True,
        "merged_from_reruns": True,
    }
    OUT.write_text(json.dumps(merged, indent=2))
    print(f"Replaced {len(replaced)} records; wrote {OUT}")
    print(f"  PASS={merged['n_pass']}  FAIL={merged['n_fail']}  "
          f"ERROR={merged['n_error']}  SKIP={merged['n_skip']}")


if __name__ == "__main__":
    main()
