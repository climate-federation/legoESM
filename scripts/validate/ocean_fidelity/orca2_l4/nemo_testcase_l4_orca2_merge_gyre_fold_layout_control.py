#!/usr/bin/env python3
"""One-variable control for the 2026-09-25 GYRE-lane merge: the fold layout.

The merge brought GitHub main's EXACT tripolar storage-layout classifier
(`92dedb6ab8`, PR #1749).  That classifier reclassifies the ORCA2 mesh as
``pivot_row_stored``, which switches four fold consumers -- the north-fold
ghost-row source, the meridional derivative's beyond-fold partner, the
density-Jacobian pressure gradient's partner column, and the per-stagger
permutation accessors -- from the stored top row to the row BELOW it.

This control changes ONE thing: it puts the fold descriptor back into the
shape the ORCA2 lane parent produced (``pivot_row_stored`` false and no
per-stagger maps, so the accessors fall back exactly as they did when those
fields did not exist).  Nothing else in the tree is touched.  Run the ORCA2
ten-step ladder under it and compare with the unpatched ladder: if the
reclassification owns the movement, the patched ladder reproduces round 20.

Usage mirrors the ladder gate; every argument is forwarded to it.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
for package in (REPO, REPO / "packages/core", REPO / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

import legoesm.grids.tripole as tripole


def install_parent_layout() -> None:
    """Force every detected fold back to the lane parent's descriptor shape."""
    original = tripole._detect_fold

    def detect(*args, **kwargs):
        fold = original(*args, **kwargs)
        if bool(getattr(fold, "is_active", False)):
            fold = fold._replace(
                pivot_row_stored=False, perm_u=None, perm_f=None)
        return fold

    detect.__wrapped__ = original
    tripole._detect_fold = detect


def main(argv: list[str] | None = None) -> int:
    install_parent_layout()
    sys.path.insert(0, str(HERE.parent))
    import nemo_testcase_l4_orca2_round1_ladder_gate as gate
    return gate.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
