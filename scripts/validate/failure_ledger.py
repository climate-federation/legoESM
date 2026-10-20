"""Failure ledger for issue #2055: junit XML -> one row per failing test, sorted by cause.

Usage:
    python scripts/validate/failure_ledger.py JOB=path.xml [JOB=path.xml ...] [--md out.md]

Each row is (job, test id, kind, category, reason). The category is a FIRST GUESS
from the test path and the failure text; the reason is the failure message's first
line, so every row can be re-checked by eye. Categories follow the issue's table.
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter

# Ordered: first match wins. (category, test-id regex or None, failure-text regex or None)
RULES: list[tuple[str, str | None, str | None]] = [
    ("timeout", None, r"Timeout \(?>?\s*\d|Failed: Timeout"),
    ("missing optional package", None, r"No module named '(dask|mpi4py|mpi4jax|zarr|xarray|netCDF4)"),
    ("macOS libm", None, r"libm\.so\.6"),
    ("XLA_FLAGS pollution", None, r"Unknown flags? in XLA_FLAGS"),
    ("seg_precip_frozen", None, r"seg_precip_frozen"),
    ("MPAS CMOR feed fakes", r"cmor.*mpas|mpas.*cmor", None),
    ("config round-trip", r"(round_?trip|roundtrip).*config|config.*(round_?trip|roundtrip)", None),
    ("stale fake driver", None, r"'(_?Fake|_?Dummy|_?Stub|Mock|SimpleNamespace)\w*' object has no attribute"),
    ("missing data/doc", None, r"FileNotFoundError|No such file or directory|does not exist"),
    ("policy drift", r"^tests/test_(no_|param_|physics_contracts|dispatch|validate_strict|scripts_|"
                     r"federation|inline|ratchet|import)", None),
    ("stale-after-rename", None,
     r"unexpected keyword argument|cannot import name|module '[\w.]+' has no attribute|"
     r"missing \d+ required (positional|keyword)|has no field|got multiple values"),
    ("oracle/fixture mismatch", r"oracle|fixture|fv3_native|gnomonic", None),
    ("pinned reference changed", None, r"Not equal to tolerance|Mismatched elements|assert_allclose|approx"),
    ("new safety check vs old test", None, r"(refus|requires?|must be|not supported|is required)"),
]


def classify(test_id: str, text: str) -> str:
    for cat, id_re, txt_re in RULES:
        if id_re and not re.search(id_re, test_id, re.I):
            continue
        if txt_re and not re.search(txt_re, text):
            continue
        return cat
    return "other"


def _test_id(case: ET.Element) -> str:
    cls = case.get("classname", "")
    f = case.get("file") or (cls.replace(".", "/") + ".py" if cls else "")
    # classname is dotted module[.Class]; rebuild a pytest-like id
    parts = cls.split(".")
    for i in range(len(parts), 0, -1):
        if parts[i - 1].startswith("test_"):
            f = "/".join(parts[:i]) + ".py"
            rest = parts[i:]
            return "::".join([f, *rest, case.get("name", "")])
    return f"{f}::{case.get('name', '')}"


def rows_from_junit(job: str, path: str) -> list[dict]:
    out = []
    for case in ET.parse(path).getroot().iter("testcase"):
        for kind in ("failure", "error"):
            el = case.find(kind)
            if el is None:
                continue
            msg = el.get("message") or ""
            text = msg + "\n" + (el.text or "")
            tid = _test_id(case)
            first = next((ln.strip() for ln in msg.splitlines() if ln.strip()), "") or \
                next((ln.strip() for ln in (el.text or "").splitlines()[::-1] if ln.strip()), "")
            out.append({"job": job, "id": tid, "kind": kind,
                        "category": classify(tid, text), "reason": first[:160]})
    return out


def to_markdown(rows: list[dict]) -> str:
    counts = Counter(r["category"] for r in rows)
    lines = [f"# Failure ledger ({len(rows)} rows)", "", "| Category | Count |", "|---|---|"]
    lines += [f"| {c} | {n} |" for c, n in counts.most_common()]
    for cat, _ in counts.most_common():
        lines += ["", f"## {cat}", "", "| Job | Test | Kind | Reason |", "|---|---|---|---|"]
        for r in sorted((r for r in rows if r["category"] == cat), key=lambda r: (r["job"], r["id"])):
            reason = r["reason"].replace("|", "\\|")
            lines.append(f"| {r['job']} | `{r['id']}` | {r['kind']} | {reason} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", help="JOB=path/to/junit.xml")
    ap.add_argument("--md", help="write the markdown ledger here")
    a = ap.parse_args(argv)
    rows = []
    for item in a.inputs:
        job, sep, path = item.partition("=")
        if not sep:
            ap.error(f"expected JOB=path, got {item!r}")
        rows += rows_from_junit(job, path)
    md = to_markdown(rows)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(md)
    print(md.split("\n## ")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
