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


def _live(text: str, key: str):
    """Every UNCOMMENTED assignment of ``key``, in file order.

    `!` starts a comment in a namelist, and these decks routinely carry
    a commented alternative beside the live value -- so a raw regex
    would count two and could pick either.
    """
    out = []
    for raw in text.splitlines():
        line = raw.split("!", 1)[0]
        m = re.search(rf'^\s*{key}\s*=\s*(\S+)', line, re.I)
        if m:
            out.append(m.group(1).strip())
    return out


def set_key(text: str, key: str, value: str, expect: str) -> str:
    """Set ``key`` to ``value``, asserting exactly one live assignment
    currently reading ``expect``.

    The assertion is the point: a silent no-op publishes a deck that is
    not the experiment its name claims, which for a generated oracle is
    the worst available outcome.
    """
    hits = _live(text, key)
    if hits != [expect]:
        raise SystemExit(
            f"expected exactly one live '{key} = {expect}', found "
            f"{hits!r}. The source deck drifted; refusing to guess which "
            f"assignment defines the experiment.")
    out = re.sub(rf'(^\s*{key}\s*=\s*)\S+', rf'\g<1>{value}', text,
                 flags=re.M | re.I, count=0)
    got = _live(out, key)
    if got != [value]:
        raise SystemExit(f"edit did not take: {key} now {got!r}")
    return out


def flip(text: str) -> str:
    """`adiabatic .true. -> .false.`, the moist switch."""
    return set_key(text, "adiabatic", ".false.", ".true.")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 1:
        key, value, expect = "adiabatic", ".false.", ".true."
    elif len(argv) == 4:
        key, value, expect = argv[1], argv[2], argv[3]
    else:
        raise SystemExit(
            "usage: flip_adiabatic_flag.py <input.nml> "
            "[<key> <new> <expected-current>]")
    with open(argv[0]) as fh:
        text = fh.read()
    with open(argv[0], "w") as fh:
        fh.write(set_key(text, key, value, expect))
    print(f"{key}: {expect} -> {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
