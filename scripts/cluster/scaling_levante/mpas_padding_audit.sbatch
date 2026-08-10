#!/bin/bash -l
#SBATCH --job-name=mpas_pad_audit
#SBATCH --account=bb1596
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=mpas_pad_audit.%j.log
# ===========================================================================
# PADDING-AWARE COLOURING audit (codex broad-consult lever #6, offline):
# each coloured ppermute round moves the ROUND MAXIMUM payload for every
# participating pair (the index arrays are (n_dev, max_r) and ppermute
# ships the whole padded buffer). Inflation = padded bytes / actual
# bytes over the s9@64 schedule. Lever is real if >~1.15; below that a
# byte-optimal recolouring cannot pay for itself on a latency-dominated
# lane.
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export JAX_PLATFORMS=cpu
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
cd "$SUBMIT_DIR" || exit 2

"$PY" - <<'EOF'
import numpy as np
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
from legoesm.parallel.sharded_dynamics import (
    SPMD_HALO_DEPTH, _build_voronoi_partition_infra,
    _build_halo_send_maps, _build_ppermute_schedule,
)

mesh = create_voronoi_mesh(subdivision_level=9, lloyd_iterations=0)
mesh = reorder_voronoi_for_sharding(mesh, 128)   # the census/prod reorder
n_dev = 64
nC, nE = int(mesh.nCells), int(mesh.nEdges)
cells_per, edges_per = nC // n_dev, nE // n_dev

(_, _, _, _, _, max_lc, max_le, parts, owner) = \
    _build_voronoi_partition_infra(mesh, n_dev, halo_depth=SPMD_HALO_DEPTH)
(pairs, cs, cr, es, er) = _build_halo_send_maps(
    parts, owner, n_dev, cells_per, edges_per)
sched = _build_ppermute_schedule(parts, owner, n_dev, cells_per,
                                 edges_per, max_lc, max_le)

nlev = 26
cell_w = (nlev + 2) * 4           # dry cell record, f32 (T,nlev + ps + phis)
edge_w = nlev * 4

# actual bytes: every directed transfer's true row count
actual = sum(len(v) for v in cs.values()) * cell_w \
       + sum(len(v) for v in es.values()) * edge_w

# padded bytes: per round, every ACTIVE pair ships the round max
n_rounds = sched["n_rounds"]
hc, he = sched["halo_cells_per_round"], sched["halo_edges_per_round"]
# pairs per round from the perms: count directed sends in round r
padded = 0
for r in range(n_rounds):
    n_pairs_r = sum(1 for (a, b) in sched["ppermute_perms"][r] if a != b)
    padded += n_pairs_r * (hc[r] * cell_w + he[r] * edge_w)

print(f"rounds={n_rounds} pairs={len(pairs)}")
print(f"actual bytes/fill  = {actual/1e6:.2f} MB")
print(f"padded bytes/fill  = {padded/1e6:.2f} MB")
print(f"INFLATION = {padded/actual:.3f}")
if padded / actual > 1.15:
    print("VERDICT: padding-aware recolouring is a REAL byte lever")
else:
    print("VERDICT: inflation below the 1.15 bar — lever refuted offline")
EOF
