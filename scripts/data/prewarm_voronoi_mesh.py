"""Prewarm the on-disk Voronoi mesh cache for big subdivision levels.

Levels above the routine cap (subdiv > 8) are cache-or-prewarm only
(``voronoi.py`` big-mesh policy, codex round-19): an MPI launch REFUSES a
cache miss so that N ranks can never each rebuild a multi-hour mesh. This
script is the designated single builder — run it ONCE per (level,
lloyd_iterations) on one process, then launch the parallel job.

Mesh flavours
-------------
``--lloyd 50`` (default) is the production SCVT. ``--lloyd 0`` is the
LABELLED SYNTHETIC SCALING MESH — a bisected icosahedron whose Voronoi
dual is valid for TRiSK but under-relaxed: measured at subdiv-6, area CV
0.084 vs 0.061 and 128-part imbalance 1.148 vs 1.095 against lloyd=50
(within the codex round-19 comparability gate). Use it for scaling
receipts, never for physics claims.

Cache location: $LEGOESM_MESH_CACHE_DIR — put it on PROJECT /work space
(scratch purges) and export the same path in the parallel job.

Usage
-----
    LEGOESM_MESH_CACHE_DIR=/work/.../mesh_cache \\
    python scripts/data/prewarm_voronoi_mesh.py --level 9 --lloyd 0
"""
from __future__ import annotations

import argparse
import os
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", type=int, required=True,
                    help="Icosahedral subdivision level (9 or 10 need this "
                         "script; <=8 build routinely without it).")
    ap.add_argument("--lloyd", type=int, default=50,
                    help="Lloyd iterations: 50 = production SCVT, 0 = "
                         "labelled synthetic scaling mesh (minutes, not "
                         "hours).")
    args = ap.parse_args()

    if not os.environ.get("LEGOESM_MESH_CACHE_DIR"):
        raise SystemExit(
            "Set LEGOESM_MESH_CACHE_DIR to a shared PROJECT path first "
            "(the parallel job must export the same one).")

    os.environ["LEGOESM_ALLOW_BIG_MESH_BUILD"] = "1"
    from legoesm.grids.voronoi import create_voronoi_mesh, prewarm_voronoi_cache

    t0 = time.time()
    if args.level > 8:
        mesh = create_voronoi_mesh(args.level, lloyd_iterations=args.lloyd)
        path = "(big-mesh policy cache)"
    else:
        path = prewarm_voronoi_cache(args.level,
                                     lloyd_iterations=args.lloyd)
        mesh = None
    dt = time.time() - t0
    n = mesh.nCells if mesh is not None else 10 * 4 ** args.level + 2
    print(f"prewarmed level={args.level} lloyd={args.lloyd}: {n} cells "
          f"in {dt/60:.1f} min -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
