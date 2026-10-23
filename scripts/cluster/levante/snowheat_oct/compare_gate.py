"""Bit-identity gate: every array of two checkpoints must match exactly.

Usage: compare_gate.py <ckpt_a.npz> <ckpt_b.npz>   (exit 0 = identical)
       compare_gate.py --expect-differ <ckpt_on.npz> <ckpt_off.npz>
           (exit 0 = some entry present in BOTH differs: the switch acts on the
            shared state, not only by adding its own field; 1 = none; 2 = error)
Used before the snow-node A/B: the explicit switch-OFF run and the plain
production-default run (same tree, one day from the same restart) must write
identical checkpoints for the OFF arms to stand as the production default.
"""
from __future__ import annotations

import sys

import numpy as np


def checkpoint_differences(path_a, path_b):
    """Names of entries that differ (or exist on one side only)."""
    a, b = np.load(path_a, allow_pickle=False), np.load(path_b, allow_pickle=False)
    diff = sorted(set(a.files) ^ set(b.files))
    for k in sorted(set(a.files) & set(b.files)):
        x, y = a[k], b[k]
        # Bytes, not values: +0.0 vs -0.0 and NaN payloads are differences.
        if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
            diff.append(k)
    return diff


def shared_differences(path_a, path_b):
    """Differing numeric state arrays present in both checkpoints.

    Scalars and strings (step, scheme names, conventions) never count, so a
    stored label cannot make a no-op switch look active.
    """
    a, b = np.load(path_a, allow_pickle=False), np.load(path_b, allow_pickle=False)
    state = {k for k in set(a.files) & set(b.files)
             if all(z[k].ndim >= 1 and z[k].dtype.kind in "fciub" for z in (a, b))}
    return [k for k in checkpoint_differences(path_a, path_b) if k in state]


if __name__ == "__main__":
    args = sys.argv[1:]
    expect_differ = args[:1] == ["--expect-differ"]
    paths = args[1:] if expect_differ else args
    if len(paths) != 2:
        print("usage: compare_gate.py [--expect-differ] <ckpt_a.npz> <ckpt_b.npz>")
        sys.exit(2)
    path_a, path_b = paths
    try:
        d = (shared_differences if expect_differ else checkpoint_differences)(path_a, path_b)
    except Exception as exc:
        print(f"ERROR: {exc!r}")
        sys.exit(2)
    if expect_differ:
        print(f"DIFFER (shared): {d}" if d else "NO SHARED ENTRY DIFFERS")
        sys.exit(0 if d else 1)
    print("IDENTICAL" if not d else f"DIFFER: {d}")
    sys.exit(1 if d else 0)
