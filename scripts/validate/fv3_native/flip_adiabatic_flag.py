#!/usr/bin/env python3
"""Flip `adiabatic .true. -> .false.` in a deck's input.nml, and refuse
anything else.

A separate FILE rather than a heredoc inside the build sbatch: the
build script already runs one heredoc, and nesting a second broke the
quoting badly enough that fragments executed in the wrong shell. It is
also independently testable, which a heredoc is not.

The assertions are the point. This edit is what makes a deck moist
(driver/solo/atmosphere.F90:156-161 derives zvir and moist_phys from
this one flag), so a silent no-op would publish a DRY deck under a
moist name -- the worst available outcome for a generated oracle.
"""
from __future__ import annotations

import re
import sys

_PAT = re.compile(r'^\s*adiabatic\s*=\s*\.(\w+)\.', re.M)


def flip(text: str) -> str:
    """`.true.` -> `.false.`, asserting it was the only such assignment."""
    hits = _PAT.findall(text)
    if hits != ["true"]:
        raise SystemExit(
            f"expected exactly one 'adiabatic = .true.' assignment, "
            f"found {hits!r}. The source deck drifted; refusing to guess "
            f"which one makes it moist.")
    out = re.sub(r'(^\s*adiabatic\s*=\s*)\.true\.', r'\1.false.', text,
                 flags=re.M)
    got = _PAT.findall(out)
    if got != ["false"]:
        raise SystemExit(f"edit did not take: adiabatic now {got!r}")
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: flip_adiabatic_flag.py <input.nml>")
    with open(argv[0]) as fh:
        text = fh.read()
    with open(argv[0], "w") as fh:
        fh.write(flip(text))
    print("adiabatic -> .false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
