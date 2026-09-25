"""Rebuild an MPAS run's ``areaCell`` so an OLD ledger artifact can be
area-weighted (#1354).

The budget-ledger reporter refuses to print unless it can establish the
reduction, because the ledger used to average columns EQUALLY while the energy
store weighted them by area -- the mismatch that made the reported
non-closure unattributable.  Artifacts written before the driver started
shipping ``areaCell`` therefore cannot be read at all.

They can still be READ, because the mesh is generated, not loaded: the run
records its subdivision level, and ``create_voronoi_mesh`` rebuilds the same
tessellation from it.  This writes that mesh's ``areaCell`` to a ``.npy`` the
reporter accepts via ``--areacell``.

CAVEAT, stated because it bounds what the recomputed number means: the mesh is
rebuilt from the subdivision level with this repo's CURRENT Lloyd-relaxation
defaults.  If the run used different ones the cell areas differ slightly -- a
smooth O(%) perturbation of the weights, not a different mesh -- so treat the
result as a re-reduction of the same columns, not as a bit-exact replay.

Usage
-----
    python scripts/validate/mpas_ledger_areacell.py --level 5 --out area.npy
    python scripts/validate/mpas_ledger_report.py <artifact.npz> \
        --areacell area.npy
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", type=int, required=True,
                   help="MPAS subdivision level the run used "
                        "(ExperimentConfig.grid.resolution); "
                        "nCells = 10*4^level + 2")
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--expect-ncells", type=int, default=None,
                   help="refuse unless the rebuilt mesh has this many cells "
                        "(pass the artifact's column count)")
    args = p.parse_args(argv)

    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(args.level)
    area = np.asarray(mesh.areaCell, dtype=np.float64)
    if args.expect_ncells is not None and area.size != args.expect_ncells:
        raise SystemExit(
            f"rebuilt level-{args.level} mesh has {area.size} cells, but the "
            f"artifact has {args.expect_ncells}: this is not the same mesh, "
            "and weighting one by the other would be worse than refusing.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, area)
    print(f"[areacell] level {args.level}: {area.size} cells, "
          f"area {area.min():.4e}..{area.max():.4e} m2 "
          f"(max/min {area.max() / area.min():.3f}), wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
