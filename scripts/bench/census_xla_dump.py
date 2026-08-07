#!/usr/bin/env python
"""Count the collectives the COMPILED step actually contains, from an XLA dump.

REPLACES the nsys-based census, which was found unreliable on the MPAS lane:
those traces silently omit ``ncclDevKernel_SendRecv`` in a subset of runs —
187k kernels recorded, zero collectives, no error, while the run's own receipt
says the ppermute halo ran (see the campaign doc, 2026-08-07). A sampling
profiler reports what it happened to catch. This reports what the executable
contains, which is deterministic.

It also replaces the in-process census in ``metadata.hlo_collective_census``,
which cannot work for the runs that matter. That one re-lowers the step, and
under multicontroller the step closes over sharded schedule arrays, so JAX
refuses:

    RuntimeError: Closing over jax.Array that spans non-addressable (non
    process local) devices is not allowed. Got jax.Array: int32[2,2,1643]

Every GPU row is multicontroller, which is why every one of them recorded
``hlo_collectives: null``. Dumping sidesteps re-lowering entirely: the
compiler writes the module while the REAL run compiles it, with the REAL
flags and device count, so GPU-only collective combining and pipelined-p2p
are reflected as executed rather than as emitted.

USE
  1. Run the bench with ``XLA_FLAGS=--xla_dump_to=<dir>`` (per rank, use a
     per-rank dir — every process dumps).
  2. ``python scripts/bench/census_xla_dump.py --dump-dir <dir>``

WHICH MODULE. A run dumps hundreds of modules (jit_add, jit_where, ...); only
the step matters. ``--module-re`` selects it, default ``jit__step``. If the
pattern matches several distinct modules the script REFUSES rather than
guessing or summing — two different compiled steps in one directory means the
run compiled more than one thing and the caller must say which.

WHAT THE NUMBERS ARE. Counts of collective ops in the post-optimization HLO,
per type, for one execution of that module. ``collective-permute`` is the
halo round count that binds MPAS strong scaling. Multiply by the tendency
evaluations per step of the integrator actually run — this counts the module,
not the step.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

# Post-optimization HLO op names. `collective-permute-start/done` are the
# async split form; counting the START avoids double-counting a single
# logical exchange.
_OPS = ("collective-permute-start", "collective-permute", "all-reduce-start",
        "all-reduce", "all-gather-start", "all-gather", "all-to-all",
        "reduce-scatter")


def count_ops(text: str) -> dict[str, int]:
    """Collective ops in one HLO module, by type.

    Matches ``= <op>(`` so an op only counts where it is the instruction
    being produced, not where it appears inside a name or a comment. The
    async ``-start``/``-done`` pair is collapsed onto the base name and the
    ``-done`` half ignored, so one logical exchange counts once.
    """
    counts: dict[str, int] = {}
    for op in _OPS:
        n = len(re.findall(rf"=\s*(?:f32|bf16|s32|u32|pred|\(|\w)[^=\n]*?\b"
                           rf"{re.escape(op)}\(", text))
        if not n:
            n = len(re.findall(rf"\b{re.escape(op)}\(", text))
        if n:
            base = op.replace("-start", "")
            counts[base] = counts.get(base, 0) + n
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    return counts


def find_module(dump_dir: Path, module_re: str) -> Path:
    """The one post-optimization module matching *module_re*."""
    cands = sorted(p for p in dump_dir.glob("*after_optimizations.txt")
                   if re.search(module_re, p.name))
    if not cands:
        available = sorted({re.sub(r"^module_\d+\.", "", p.name)
                            .replace(".cpu_after_optimizations.txt", "")
                            .replace(".gpu_after_optimizations.txt", "")
                            for p in dump_dir.glob("*after_optimizations.txt")})
        raise SystemExit(
            f"no post-optimization module matches {module_re!r} in "
            f"{dump_dir}.\nDumped modules: {available[:20]}"
            f"{' ...' if len(available) > 20 else ''}")
    # Several files can be the SAME module recompiled; distinct module names
    # are the ambiguity that matters.
    names = {re.sub(r"^module_\d+\.", "", p.name) for p in cands}
    if len(names) > 1:
        raise SystemExit(
            f"{module_re!r} matches {len(names)} distinct modules in "
            f"{dump_dir}: {sorted(names)}. Narrow --module-re; summing them "
            f"would report a count no single execution ever pays.")
    return cands[-1]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump-dir", required=True, type=Path,
                   help="Directory given to --xla_dump_to")
    p.add_argument("--module-re", default="jit__step",
                   help="Regex selecting the step module (default jit__step)")
    p.add_argument("--out", default=None, help="Optional JSON output path")
    args = p.parse_args()

    if not args.dump_dir.is_dir():
        raise SystemExit(f"not a directory: {args.dump_dir}")
    module = find_module(args.dump_dir, args.module_re)
    counts = count_ops(module.read_text())

    print(f"module: {module.name}")
    for op in sorted(k for k in counts if k != "total"):
        print(f"  {op:24s} {counts[op]:6d}")
    print(f"  {'TOTAL':24s} {counts['total']:6d}")
    if not counts["total"]:
        print("\nNo collectives in this module. For a multi-device run that is "
              "a RESULT worth checking, not a default: confirm the run really "
              "was sharded and that --module-re selected the step.")

    if args.out:
        outdir = os.path.dirname(args.out)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"module": module.name, "dump_dir": str(args.dump_dir),
                       "counts": counts}, f, indent=2)
        print(f"JSON: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
