"""Failure ledger for issue #2055: junit XML -> one row per failing test, sorted by cause.

Usage:
    python scripts/validate/failure_ledger.py JOB=path.xml [JOB=path.xml ...] [--md out.md]

Each row is one failing test id per job: (job, test id, kinds, category, reason).
A test that fails AND errors in teardown is one row with kinds "failure+error".
The category is a FIRST GUESS from the test path and the failure text, never a
diagnosis; the reason is the first failure message line, so every row can be
re-checked by eye. Categories follow the issue's table.
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
    # ci.yml: the top-level tests/test_*.py files are the AST/source ratchets.
    ("policy drift", r"^tests/test_[^/]+\.py(::|$)", None),
    ("stale-after-rename", None,
     r"unexpected keyword argument|cannot import name|module '[\w.]+' has no attribute|"
     r"missing \d+ required (positional|keyword)|has no field|got multiple values"),
    ("oracle/fixture mismatch", r"oracle|fv3_native|gnomonic", None),
    ("pinned reference changed", None, r"Not equal to tolerance|Mismatched elements|\u00b1"),
    # Only a guard the model RAISES: an exception type, then refuse/require wording.
    ("new safety check vs old test", None,
     r"(ValueError|RuntimeError|SystemExit|NotImplementedError)\b[^\n]*(\brefus|\brequires\b|not supported)"),
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
    if not cls:  # collection error: pytest puts the dotted module path in name
        return case.get("file") or case.get("name", "").replace(".", "/") + ".py"
    f = case.get("file") or (cls.replace(".", "/") + ".py" if cls else "")
    # classname is dotted module[.Class]; rebuild a pytest-like id
    parts = cls.split(".")
    for i in range(len(parts), 0, -1):
        if parts[i - 1].startswith("test_"):
            f = "/".join(parts[:i]) + ".py"
            rest = parts[i:]
            return "::".join([f, *rest, case.get("name", "")])
    return f"{f}::{case.get('name', '')}"


def _reason(texts: list[tuple[str, str]]) -> str:
    for msg, _ in texts:  # the whole message, on one line
        if msg.strip():
            return " ".join(msg.split())[:160]
    for _, body in texts:  # no message: pytest's "E   ..." line
        for ln in body.splitlines():
            if ln.startswith("E "):
                return " ".join(ln[1:].split())[:160]
    return ""


def rows_from_junit(job: str, paths: str | list[str]) -> list[dict]:
    rows: dict[str, dict] = {}
    for path in [paths] if isinstance(paths, str) else paths:
        for case in ET.parse(path).getroot().iter("testcase"):
            bad = case.findall("failure") + case.findall("error")
            if not bad:
                continue
            tid = _test_id(case)
            row = rows.setdefault(tid, {"kinds": [], "texts": []})
            for el in bad:
                if el.tag not in row["kinds"]:
                    row["kinds"].append(el.tag)
                row["texts"].append((el.get("message") or "", el.text or ""))
    return [{"job": job, "id": tid, "kind": "+".join(r["kinds"]),
             "category": classify(tid, "\n".join(m + "\n" + b for m, b in r["texts"])),
             "reason": _reason(r["texts"])}
            for tid, r in rows.items()]


def to_markdown(rows: list[dict]) -> str:
    counts = Counter(r["category"] for r in rows)
    lines = [f"# Failure ledger ({len(rows)} rows)", "", "| Category | Count |", "|---|---|"]
    lines += [f"| {c} | {n} |" for c, n in counts.most_common()]
    for cat, _ in counts.most_common():
        lines += ["", f"## {cat} (first-guess category)", "", "| Job | Test | Kind | Reason |", "|---|---|---|---|"]
        for r in sorted((r for r in rows if r["category"] == cat), key=lambda r: (r["job"], r["id"])):
            reason = r["reason"].replace("|", "\\|")
            lines.append(f"| {r['job']} | `{r['id']}` | {r['kind']} | {reason} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", help="JOB=path/to/junit.xml")
    ap.add_argument("--md", help="write the markdown ledger here")
    a = ap.parse_args(argv)
    by_job: dict[str, list[str]] = {}
    for item in a.inputs:
        job, sep, path = item.partition("=")
        if not sep:
            ap.error(f"expected JOB=path, got {item!r}")
        by_job.setdefault(job, []).append(path)
    rows = [r for job, paths in by_job.items() for r in rows_from_junit(job, paths)]
    md = to_markdown(rows)
    if a.md:
        with open(a.md, "w") as fh:
            fh.write(md)
    print(md.split("\n## ")[0] if a.md else md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
