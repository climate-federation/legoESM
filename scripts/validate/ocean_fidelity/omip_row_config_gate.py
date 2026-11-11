"""Row gate for the one-row-at-a-time OMIP production move.

A row's run must differ from the previous adopted run in exactly the voted
scheme group.  This compares two ``run_manifest.json`` files of the same leg:

* the command lines, parsed with the runner's own argument parser, so
  spelling and defaults do not matter; per-run paths (``--output``,
  ``--restart-save``, ``--restart-from``) are ignored;
* the flattened ``resolved_config``, minus ``output_path``.

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
import sys

PER_RUN_DESTS = {"output", "restart_save", "restart_from"}


def _flatten(d, pre=""):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            out.update(_flatten(v, f"{pre}.{k}" if pre else str(k)))
        return out
    return {pre: d}


def _options(manifest, parser):
    argv = manifest["run"]["argv"]
    return {k: v for k, v in vars(parser.parse_args(argv[1:])).items()
            if k not in PER_RUN_DESTS}


def _parse_expect(items):
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        out[k] = v
    return out


def gate(base, cand, parser, expect, allow_config):
    problems = []
    ob, oc = _options(base, parser), _options(cand, parser)
    diff = {k: (ob.get(k), oc.get(k)) for k in sorted(set(ob) | set(oc))
            if ob.get(k) != oc.get(k)}
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
        if k == "output_path" or fb.get(k) == fc.get(k):
            continue
        if not any(fnmatch.fnmatch(k, g) for g in allow_config):
            problems.append(f"config {k}: {fb.get(k)!r} -> {fc.get(k)!r}")
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
    from scripts.run.run_omip_core2 import _build_arg_parser
    base = json.load(open(a.baseline))
    cand = json.load(open(a.candidate))
    diff, problems = gate(base, cand, _build_arg_parser(), _parse_expect(a.expect),
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
