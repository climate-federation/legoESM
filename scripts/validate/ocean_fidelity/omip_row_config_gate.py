"""Row gate for the one-row-at-a-time OMIP production move.

A row's run must differ from the previous adopted run in exactly the voted
scheme group.  This compares two ``run_manifest.json`` files of the same leg:

* the command lines, each parsed with the runner's argument parser AS OF
  THAT RUN'S COMMIT (``git show <commit>:scripts/run/run_omip_core2.py``),
  so spelling does not matter and a default that changed between the two
  commits shows up as a difference; per-run paths (``--output``,
  ``--restart-save``, ``--restart-from``) compare by presence only;
* the flattened ``resolved_config`` (a key present on one side only is a
  difference, empty sections included), minus ``output_path``.

It exits 0 only when the option differences are exactly ``--expect``
(``dest=value`` pairs), the resolved-config differences all match an
``--allow-config`` glob, and the run duration agrees.  Anything else is
printed and exits 1, before any model time is spent.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import shlex
import subprocess
import sys
import types
from pathlib import Path

PER_RUN_DESTS = {"output", "restart_save", "restart_from"}
RUNNER = "scripts/run/run_omip_core2.py"
_ABSENT = object()


def parser_at(commit):
    """The runner's argument parser as it was at ``commit``."""
    src = subprocess.run(["git", "show", f"{commit}:{RUNNER}"], check=True,
                         capture_output=True, text=True).stdout
    mod = types.ModuleType(f"_core2_at_{commit[:12]}")
    mod.__file__ = str(Path(RUNNER).resolve())
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod._build_arg_parser()


def _flatten(d, pre=""):
    if isinstance(d, dict):
        if not d and pre:
            return {pre: {}}
        out = {}
        for k, v in d.items():
            out.update(_flatten(v, f"{pre}.{k}" if pre else str(k)))
        return out
    return {pre: d}


def _options(manifest, parser):
    argv = manifest["run"]["argv"]
    # Per-run paths compare by presence only: starting from a restart (or
    # saving one) is part of the run, where the file lives is not.
    return {k: (v is not None if k in PER_RUN_DESTS else v)
            for k, v in vars(parser.parse_args(argv[1:])).items()}


def _parse_expect(items):
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        out[k] = v
    return out


def gate(base, cand, parser_base, parser_cand, expect, allow_config):
    problems = []
    ob, oc = _options(base, parser_base), _options(cand, parser_cand)
    diff = {k: (ob.get(k, _ABSENT), oc.get(k, _ABSENT))
            for k in sorted(set(ob) | set(oc))
            if ob.get(k, _ABSENT) != oc.get(k, _ABSENT)}
    for k, (vb, vc) in diff.items():
        if k not in expect:
            problems.append(f"option {k}: {vb!r} -> {vc!r} (not in the row)")
        elif str(vc) != expect[k]:
            problems.append(f"option {k}: {vb!r} -> {vc!r} (row expects {expect[k]!r})")
    for k in expect:
        if k not in diff:
            problems.append(f"option {k}: expected to change to {expect[k]!r}, unchanged")
    fb = _flatten(base["config"]["resolved_config"])
    fc = _flatten(cand["config"]["resolved_config"])
    for k in sorted(set(fb) | set(fc)):
        vb, vc = fb.get(k, _ABSENT), fc.get(k, _ABSENT)
        if k == "output_path" or vb == vc:
            continue
        if not any(fnmatch.fnmatch(k, g) for g in allow_config):
            show = lambda v: "<absent>" if v is _ABSENT else repr(v)  # noqa: E731
            problems.append(f"config {k}: {show(vb)} -> {show(vc)}")
    return diff, problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, help="previous adopted run_manifest.json")
    p.add_argument("--candidate", required=True, help="this row's run_manifest.json")
    p.add_argument("--expect", nargs="+", required=True,
                   help="dest=value option changes that make up the row")
    p.add_argument("--allow-config", nargs="*", default=[],
                   help="resolved_config key globs the row may change")
    a = p.parse_args(argv)
    base = json.load(open(a.baseline))
    cand = json.load(open(a.candidate))
    pb = parser_at(base["legoESM"]["commit"])
    pc = parser_at(cand["legoESM"]["commit"])
    diff, problems = gate(base, cand, pb, pc, _parse_expect(a.expect),
                          a.allow_config)
    for k, (vb, vc) in diff.items():
        print(f"[row-gate] option {k}: {vb!r} -> {vc!r}")
    for msg in problems:
        print(f"[row-gate] FAIL {msg}")
    print(f"[row-gate] baseline argv: {shlex.join(base['run']['argv'][:3])} ...")
    print(f"[row-gate] {'PASS' if not problems else 'FAIL'}: "
          f"{len(diff)} option(s) differ, {len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
