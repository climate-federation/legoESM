#!/usr/bin/env python
"""Consolidate atmosphere + ocean matrix summary.json results.

Reads results/atmosphere/summary.json and results/ocean/summary.json
and prints a compact PASS/FAIL/ERROR table grouped by grid.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def _load(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _emit(label, summary):
    if summary is None:
        print(f"{label}: no summary.json found")
        return
    rows = summary.get("results", summary) if isinstance(summary, dict) else summary
    if isinstance(rows, dict) and "tests" in rows:
        rows = rows["tests"]
    if not isinstance(rows, list):
        # Some matrices nest differently; try common keys.
        for k in ("tests", "results", "cases"):
            if isinstance(summary, dict) and isinstance(summary.get(k), list):
                rows = summary[k]
                break
    by_grid = Counter()
    fail_rows = []
    for r in rows:
        g = r.get("grid", "?")
        st = r.get("status", "?")
        by_grid[(g, st)] += 1
        if st in ("FAIL", "ERROR"):
            fail_rows.append((r.get("case") or r.get("test", "?"), g,
                              r.get("resolution", "?"), st,
                              (r.get("notes") or "")[:100]))
    print(f"\n=== {label} ===")
    grids = sorted({g for g, _ in by_grid})
    statuses = sorted({s for _, s in by_grid})
    print(f"{'grid':<24s}", " ".join(f"{s:>6s}" for s in statuses), " TOTAL")
    for g in grids:
        cells = [by_grid.get((g, s), 0) for s in statuses]
        print(f"{g:<24s}", " ".join(f"{c:>6d}" for c in cells), f"{sum(cells):>6d}")
    if fail_rows:
        print(f"\n{label} non-PASS rows:")
        for row in fail_rows:
            print(f"  [{row[3]}] {row[0]}/{row[1]}/{row[2]}: {row[4]}")


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "results"
    _emit("ATMOSPHERE", _load(root / "atmosphere" / "summary.json"))
    _emit("OCEAN", _load(root / "ocean" / "summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
