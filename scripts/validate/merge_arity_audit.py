#!/usr/bin/env python
"""Find tuple-unpack arity mismatches — the silent merge-regression class.

WHY.  Merging PR #1638 was conflict-free and still broke the tripole ocean's
barotropic path: ``compute_filter_weights`` returns four values, and the merged
lat-lon C-grid caller unpacked three.  Git's merge is line-based and has no
notion of symbols, so a callee's return arity in one module and a caller's
unpack in another are unrelated text to it; adjacent non-overlapping hunks in
the same if/elif chain merged cleanly and dropped the fourth name.

That one was recoverable because the branch raises when it runs.  The dangerous
variant is the "obvious" minimal fix: keeping the three-value unpack and
setting the missing value locally.  Here ``n_loop`` is ``2 * n_substeps - 1``
(23 for 12), so a local ``n_loop = n_substeps`` would have run the barotropic
filter over half its window and returned a plausible wrong answer instead of
crashing.

WHAT THIS DOES.  For every function in the tree that returns a tuple literal of
a fixed size, find every assignment-unpack call site of that name and compare
counts.  Reports only disagreements.  Deliberately simple and syntactic: it
needs no type stubs and no import of the scientific stack, so it runs on a
login node in seconds.

LIMITS, stated so the output is not over-read:
  * matches on the bare function NAME (unqualified calls only, so ``x.split()``
    does not collide with a repo ``split``), and two repo functions sharing a
    name can still produce a false pair -- check the file:line before believing
    it;
  * only sees ``return a, b, c`` style tuple literals, not a returned variable
    that happens to hold a tuple, nor ``-> Tuple[...]`` annotations alone;
  * starred targets (``a, *rest = f()``) are skipped, since any arity fits.
A clean report is therefore weak evidence, not proof.  It is a tripwire.
"""
from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path


def _returns(tree):
    """{func_name: {tuple sizes it returns}} for fixed-size tuple returns."""
    out = defaultdict(set)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            # Only returns belonging to THIS function, not a nested one.
            if not isinstance(sub, ast.Return) or sub.value is None:
                continue
            owner = None
            for cand in ast.walk(node):
                if isinstance(cand, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if any(sub is r for r in ast.walk(cand)
                           if isinstance(r, ast.Return)):
                        owner = cand
            if owner is not node:
                continue
            if isinstance(sub.value, ast.Tuple) and not any(
                    isinstance(e, ast.Starred) for e in sub.value.elts):
                out[node.name].add(len(sub.value.elts))
    return out


def _unpacks(tree):
    """[(func_name, n_targets, lineno)] for `a, b = func(...)` call sites."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Tuple):
            continue
        if any(isinstance(e, ast.Starred) for e in tgt.elts):
            continue                      # any arity fits a starred target
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        fn = call.func
        # UNQUALIFIED names only. Matching `x.split()` against a repo function
        # named `split` produced 14 false positives on the first run -- every
        # hit was str.split or np.split. A cross-module repo call is imported
        # by name, so restricting to ast.Name keeps the real signal and drops
        # the library-method noise.
        if isinstance(fn, ast.Name):
            out.append((fn.id, len(tgt.elts), node.lineno))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=["packages", "scripts", "src"])
    a = ap.parse_args()

    files = []
    for r in a.roots:
        files.extend(Path(r).rglob("*.py"))
    returns, sites = defaultdict(set), []
    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for k, v in _returns(tree).items():
            returns[k] |= v
        for name, n, ln in _unpacks(tree):
            sites.append((name, n, f, ln))

    bad = []
    for name, n, f, ln in sites:
        sizes = returns.get(name)
        # Only flag when the callee is unambiguous: exactly one known tuple
        # size, and the site disagrees with it. A function with several return
        # shapes is a legitimate (if unpleasant) pattern, not a merge bug.
        if sizes and len(sizes) == 1 and n not in sizes:
            bad.append((f, ln, name, n, next(iter(sizes))))

    print(f"scanned {len(files)} files, {len(sites)} unpack sites, "
          f"{len(returns)} functions returning fixed tuples\n")
    if not bad:
        print("no arity disagreements found (tripwire clean, not a proof)")
        return 0
    print("ARITY DISAGREEMENTS (verify each -- matching is by bare name):")
    for f, ln, name, got, want in sorted(bad):
        print(f"  {f}:{ln}: unpacks {got} from {name}(), which returns {want}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
