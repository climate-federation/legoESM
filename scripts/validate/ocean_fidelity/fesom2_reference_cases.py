#!/usr/bin/env python
"""Inventory FESOM2's own CI reference cases and their committed truth values.

Why
---
FESOM2 ships `fcheck` truth values with each setup in ``setups/<case>/setup.yml``
-- 1-day reference values for sst / temp / salt / u / v (and a_ice where sea ice
is on). Those are the reference numbers a paired legoESM-vs-FESOM2 experiment
should be judged against, and they are already committed upstream, so no FESOM2
rerun is needed to obtain them.

This EXTRACTS them rather than transcribing them into prose. A hand-copied
table goes stale silently; this re-reads the source of truth and stamps
provenance (path, mtime, git SHA of the FESOM2 checkout when available).

**A truth value is only comparable against a legoESM run whose CONFIGURATION
matches**: same geometry (`cyclic_length`, `which_toy`), same EOS
(`state_equation`), same closures (`Fer_GM`, `Redi`, `mix_scheme`), same
run length, same forcing/climatology. This script therefore emits the
config discriminants ALONGSIDE each truth value, so a mismatch is visible
instead of assumed away.

Known mismatch, do not paper over it
------------------------------------
FESOM2 `test_neverworld2` uses `which_toy: "neverworld2"` with
`cyclic_length: 60.0` (a sector). legoESM's `ocean/experiments/neverworld2_lite.py`
is a DIFFERENT configuration -- a DINO-geometry port that is "doubly periodic
in longitude (full globe, ~360 deg zonal extent)" and self-described as a
"Phase D smoke-grade port". Same name, different case. Comparing their numbers
term-by-term would be a confound, not a result.

Usage
-----
    python scripts/validate/ocean_fidelity/fesom2_reference_cases.py \\
        --fesom2-root /work/ab0995/a270301/fesom2 --json-out cases.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

# Config keys that decide whether a truth value is comparable at all.
DISCRIMINANTS = (
    "cyclic_length", "which_toy", "toy_ocean", "state_equation",
    "Fer_GM", "Redi", "mix_scheme", "use_ice", "run_length",
    "run_length_unit", "ntasks",
)


def _scalar(text: str):
    t = text.strip().strip('"').strip("'")
    for cast in (int, float):
        try:
            return cast(t)
        except ValueError:
            pass
    if t in ("True", "true"):
        return True
    if t in ("False", "false"):
        return False
    return t


def parse_setup(path: Path) -> dict:
    """Pull ``fcheck`` values and the config discriminants out of a setup.yml.

    Deliberately a small hand parser rather than a YAML dependency: we need
    only two shapes (the ``fcheck:`` block and ``key: value`` lines), and the
    files are upstream-controlled, so a parse failure must be loud rather than
    silently returning {}.
    """
    lines = path.read_text().splitlines()
    fcheck: dict = {}
    cfg: dict = {}
    in_fcheck = False
    fcheck_indent = None
    for ln in lines:
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        indent = len(ln) - len(ln.lstrip())
        if ln.strip().startswith("fcheck:"):
            in_fcheck, fcheck_indent = True, indent
            continue
        if in_fcheck:
            if indent <= fcheck_indent:
                in_fcheck = False
            elif ":" in ln:
                k, v = ln.split(":", 1)
                if v.strip():
                    fcheck[k.strip()] = _scalar(v)
                continue
        if ":" in ln:
            k, v = ln.split(":", 1)
            k = k.strip()
            if k in DISCRIMINANTS and v.strip():
                cfg[k] = _scalar(v)
    return {"fcheck": fcheck, "config": cfg}


def _git_sha(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def collect(fesom2_root: Path) -> dict:
    setups = fesom2_root / "setups"
    if not setups.is_dir():
        raise SystemExit(f"no setups/ under {fesom2_root}")
    cases = {}
    for d in sorted(p for p in setups.iterdir() if p.is_dir()):
        f = d / "setup.yml"
        if not f.is_file():
            continue
        rec = parse_setup(f)
        if not rec["fcheck"]:
            continue                      # no committed truth values
        rec["setup_yml"] = str(f)
        rec["mtime"] = f.stat().st_mtime
        cases[d.name] = rec
    return {
        "_provenance": {
            "fesom2_root": str(fesom2_root),
            "fesom2_git_sha": _git_sha(fesom2_root),
            "probe": "scripts/validate/ocean_fidelity/fesom2_reference_cases.py",
        },
        "cases": cases,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fesom2-root", default="/work/ab0995/a270301/fesom2")
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data = collect(Path(args.fesom2_root))
    for name, rec in data["cases"].items():
        c = rec["config"]
        geom = ", ".join(f"{k}={c[k]}" for k in
                         ("which_toy", "cyclic_length", "use_ice") if k in c)
        print(f"{name}  [{geom or 'realistic'}]")
        for k, v in rec["fcheck"].items():
            print(f"    {k:6s} = {v!r}")
    print(f"\n{len(data['cases'])} cases with committed truth values")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(data, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
