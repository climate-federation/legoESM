"""Bit-identity gate: every array of two checkpoints must match exactly.

Usage: compare_gate.py <ckpt_a.npz> <ckpt_b.npz>   (exit 0 = identical)
Used before the snow-node A/B: the switch-OFF branch run and the base-commit
run, both one day from the same restart, must write identical checkpoints for
the base-commit controls to be shared.
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


if __name__ == "__main__":
    d = checkpoint_differences(sys.argv[1], sys.argv[2])
    print("IDENTICAL" if not d else f"DIFFER: {d}")
    sys.exit(1 if d else 0)
