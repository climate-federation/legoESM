#!/usr/bin/env python3
"""Copy an MPAS checkpoint with every ``physstate_*`` entry removed.

The driver refuses to restart a prognostic physics carry (e.g. prognostic
CLUBB's 15 moments) from a checkpoint that holds another scheme's carry, and
it refuses a PARTIAL physics carry too.  Its one sanctioned escape is a
checkpoint with NO ``physstate_*`` entries at all: every physics-state field
(turbulence energy, convection memory, gravity-wave spectrum, held radiation,
...) is then freshly seeded, and the run log says so.  A controlled pair that
needs that escape must give BOTH arms the same stripped file so they differ
only in the scheme under test.

Usage:
  strip_physics_state.py <checkpoint.npz> <out.npz>
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np


def strip_physics_state(src: str, dst: str) -> list[str]:
    """Write ``dst`` = ``src`` minus its ``physstate_*`` keys; return the dropped keys.

    Never in place (an interrupted write would destroy the only copy), never
    via pickle (a checkpoint holds plain numeric/string arrays only), and the
    output is written to a temporary file next to ``dst`` and renamed into
    place so a partial file can never be mistaken for a checkpoint.
    """
    if os.path.realpath(src) == os.path.realpath(dst):
        raise SystemExit("refusing to strip a checkpoint in place; give a new destination")
    with np.load(src, allow_pickle=False) as z:
        kept = {k: z[k] for k in z.files if not k.startswith("physstate_")}
        dropped = sorted(k for k in z.files if k.startswith("physstate_"))
    if not dropped:
        raise SystemExit(f"{src}: no physstate_* entries to strip")
    fd, tmp = tempfile.mkstemp(prefix=".strip_", suffix=".npz",
                               dir=os.path.dirname(os.path.abspath(dst)) or ".")
    os.close(fd)
    try:
        np.savez(tmp, **kept)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dropped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args(argv)
    dropped = strip_physics_state(args.src, args.dst)
    print(f"{args.dst}: dropped {len(dropped)} physics-state entries: {dropped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
