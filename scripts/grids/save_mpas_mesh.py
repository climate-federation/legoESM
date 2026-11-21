"""Build a global MPAS (SCVT) mesh once and save it as a hash-pinned file.

Runs that set ``mpas_mesh_file`` / ``mpas_mesh_sha256`` load this file
instead of regenerating the mesh, so every machine steps the same cells in
the same order (a regenerated level 4 mesh put 1,973 of 2,562 cells at other
indices on Snellius than on a Mac).

The mesh is built fresh (the on-disk Voronoi cache is bypassed) under the
float64 storage policy, so the file holds the full float64 geometry; a run
loads it in its own storage dtype.  The script prints the SHA-256 to put in
the deck as ``mpas_mesh_sha256``.

Usage
-----
    python scripts/grids/save_mpas_mesh.py 4 data/grids/mpas_l4_lloyd50.npz
    python scripts/grids/save_mpas_mesh.py 6 mesh6.npz --lloyd-iterations 50
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("resolution", type=int,
                    help="subdivision level (nCells = 10 * 4**level + 2)")
    ap.add_argument("out", help="output .npz path")
    ap.add_argument("--lloyd-iterations", type=int, default=50,
                    help="Lloyd relaxation iterations (default 50, the "
                         "model driver's value)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output file")
    args = ap.parse_args(argv)

    if os.path.exists(args.out) and not args.force:
        ap.error(f"{args.out} exists; pass --force to overwrite")

    # Fresh build: a cached mesh may come from another code version or another
    # storage policy.  Set before legoesm is imported.
    os.environ["LEGOESM_MESH_CACHE_DISABLE"] = "1"

    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm import constants
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.voronoi_mesh_file import (
        load_voronoi_mesh, save_voronoi_mesh)

    previous = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        t0 = time.time()
        mesh = create_voronoi_mesh(
            args.resolution, lloyd_iterations=args.lloyd_iterations)
        t_build = time.time() - t0
        digest = save_voronoi_mesh(
            mesh, args.out, resolution=args.resolution,
            lloyd_iterations=args.lloyd_iterations, omega=constants.Omega)
        # read back through the verifying loader before reporting success
        back = load_voronoi_mesh(args.out, expected_sha256=digest)
        if back.nCells != mesh.nCells:
            raise SystemExit("read-back check failed")
    finally:
        set_policy(previous)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"mesh resolution {args.resolution}, lloyd_iterations "
          f"{args.lloyd_iterations}: {mesh.nCells} cells, {mesh.nEdges} "
          f"edges, {mesh.nVertices} vertices (built in {t_build:.1f} s)")
    print(f"wrote {args.out} ({size_mb:.2f} MB)")
    print(f"mpas_mesh_file: {args.out}")
    print(f"mpas_mesh_sha256: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
