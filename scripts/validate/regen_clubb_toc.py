#!/usr/bin/env python
"""Regenerate (or check) the line-numbered table of contents in ``clubb.py``.

WHY THIS EXISTS. ``clubb.py`` is ~6700 lines and opens with a TOC whose rows
cite the line number of each section header, plus a pointer to the flag
reference table. ``tests/unit/test_clubb_config.py::test_toc_line_numbers_accurate``
asserts every one of them is exact, so any edit that shifts the file turns that
test red -- correctly, because a stale TOC misleads readers, but the remedy was
hand-editing twenty numbers. On 2026-09-23 all 19 rows and the pointer had
drifted together (row 1 cited 328 against an actual 343; row 19 cited 5497
against 5935), which is what a hand-maintained derived artifact does.

So the numbers are DERIVED here instead of maintained. Regenerating by hand and
stopping there would have re-armed the same trap with fresh bait.

    python scripts/validate/regen_clubb_toc.py --check   # exit 1 if stale
    python scripts/validate/regen_clubb_toc.py           # rewrite in place

``--check`` is the form for CI and for the test's failure message; the bare
form is the one-command fix.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

_DEFAULT = (pathlib.Path(__file__).resolve().parents[2]
            / "packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py")
#: The TOC lives in the module docstring; no section header can appear that early.
_TOC_SCAN_LINES = 80
_FLAG_BANNER = "CAM-default CLUBB model-flag values"


def true_positions(lines: list[str]) -> tuple[dict[int, int], int | None]:
    """Where the section headers and the flag banner ACTUALLY are (1-based).

    A section header is a top-level comment ``# N. Title``. The FIRST match for
    each number wins: the TOC cites the header, and a later line that merely
    mentions ``# 3.`` inside prose must not steal the row.
    """
    headers: dict[int, int] = {}
    for i, line in enumerate(lines, 1):
        m = re.match(r"# (\d+)\. ", line)
        if m:
            headers.setdefault(int(m.group(1)), i)
    banner = next((i for i, l in enumerate(lines, 1) if _FLAG_BANNER in l), None)
    return headers, banner


#: How many TOC rows the file is expected to carry. A checker that matches
#: ZERO rows reports "in sync" after validating nothing, which is strictly
#: worse than drift -- so the count is asserted, not discovered (review).
EXPECTED_TOC_ROWS = 19


def retarget(lines: list[str]) -> tuple[list[str], list[str]]:
    """Return (rewritten lines, list of human-readable drifts found)."""
    headers, banner = true_positions(lines)
    out = list(lines)
    drift: list[str] = []
    for i, line in enumerate(lines[:_TOC_SCAN_LINES]):
        m = re.match(r"(  (\d+)\.\s+\[line\s+)(\d+)(\])", line)
        if m:
            n, cited = int(m.group(2)), int(m.group(3))
            actual = headers.get(n)
            if actual is None:
                drift.append(f"TOC row {n} has no `# {n}. ` header in the file")
                continue
            if actual != cited:
                drift.append(f"row {n}: cites {cited}, header at {actual}")
                out[i] = (f"{m.group(1)}{actual}{m.group(4)}"
                          f"{line[m.end():]}")
        m2 = re.search(r"(flag reference table at line )(\d+)", line)
        if m2:
            cited = int(m2.group(2))
            if banner is None:
                drift.append("flag reference table banner not found in the file")
            elif banner != cited:
                drift.append(f"flag pointer: cites {cited}, banner at {banner}")
                out[i] = line[:m2.start(2)] + str(banner) + line[m2.end(2):]
    return out, drift


def check_toc(path=None) -> list[str]:
    """Drift in the committed file, as a list of human-readable strings.

    The importable form, so PYTEST is the invoker. A ``--check`` flag that
    nothing runs is not a tripwire -- this repo has no CI, so the only thing
    that reliably executes is the test suite. One implementation, two doors.

    REFUSES a file whose TOC it cannot see: matching zero rows and reporting
    "in sync" would be a checker that validates nothing while looking green.
    """
    path = pathlib.Path(path or _DEFAULT)
    lines = path.read_text().split("\n")
    n_rows = sum(1 for l in lines[:_TOC_SCAN_LINES]
                 if re.match(r"  (\d+)\.\s+\[line\s+(\d+)\]", l))
    if n_rows != EXPECTED_TOC_ROWS:
        return [f"parsed {n_rows} TOC rows, expected {EXPECTED_TOC_ROWS} — the "
                f"TOC format changed, so this checker is validating nothing; "
                f"fix the pattern or EXPECTED_TOC_ROWS before trusting a green"]
    return retarget(lines)[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=str(_DEFAULT))
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit 1 without writing")
    args = ap.parse_args(argv)

    path = pathlib.Path(args.path)
    # One implementation, two doors: the CLI goes through the same row-count
    # guard as pytest, so ``--check`` cannot pass on a TOC it failed to parse.
    drift = check_toc(path)
    out = retarget(path.read_text().split("\n"))[0]

    if not drift:
        print(f"{path.name}: TOC is exact")
        return 0
    for d in drift:
        print(f"  {d}")
    if args.check:
        print(f"{path.name}: {len(drift)} stale TOC reference(s) -- run this "
              f"script without --check to fix")
        return 1
    path.write_text("\n".join(out))
    print(f"{path.name}: rewrote {len(drift)} stale reference(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
