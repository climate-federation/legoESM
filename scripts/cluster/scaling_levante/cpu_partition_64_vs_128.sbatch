#!/bin/bash -l
#SBATCH --job-name=part_ab
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=part_ab_%j.log
# ===========================================================================
# IS THE 128-RANK PARTITION WORSE THAN THE 64-RANK ONE?
#
# The CPU atmosphere gets SLOWER going 64 -> 128 ranks (822 -> 1240 ms, one
# code revision, one job). The leading alternative to a thread-pool
# explanation is that the ownership split simply degrades at 128: space-
# filling-curve partitions get worse in discrete jumps, and a step costs what
# its SLOWEST rank costs, so a load imbalance or an extra exchange round at
# 128 and not at 64 would produce exactly this kink.
#
# This costs no cluster-scale time: the schedule builder runs offline on one
# node, calling the SAME builders production calls rather than a lookalike
# adjacency graph (a re-derived 1-ring graph reports 8 rounds where the real
# depth-3 graph reports 12-14).
#
# WHAT WOULD EXPLAIN THE KINK: exchange rounds, maximum neighbour degree, or
# worst-rank cell count rising from 64 to 128 by more than the rank count
# does. A step is paid at the worst rank, so a max-to-mean cell ratio that
# grows is a slowdown even when the mean halves properly.
#
# WHAT WOULD NOT: rounds and degree flat, and the worst-rank cell count
# falling by about half. Then the split is fine at 128 and the kink is
# elsewhere -- which is what the threads-per-rank sweep tests.
#
# Production weights are passed explicitly (cell_width = nlev + 2,
# edge_width = nlev): the default 1:1 weighting builds a DIFFERENT schedule,
# and a round count from it is not production's.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/cpu_partition_64_vs_128.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
echo "provenance: $($PY -c 'import legoesm.parallel.sharded_dynamics as m; print(m.__file__)')"
echo "sha=$(git -C "$REPO" rev-parse --short HEAD)"

srun "$PY" - <<'PYEOF'
import numpy as np
from legoesm.grids.mpas_voronoi import build_icosahedral_voronoi_mesh
from legoesm.parallel.sharded_dynamics import spmd_schedule_cost

NLEV = 26
mesh = build_icosahedral_voronoi_mesh(subdivisions=9, lloyd_iterations=0)
print(f"mesh cells={mesh.n_cells}", flush=True)
rows = []
for nd in (32, 64, 128, 256):
    c = spmd_schedule_cost(mesh, nd, method="sfc", reorder_target=nd,
                           cell_width=NLEV + 2, edge_width=NLEV)
    mean_cells = mesh.n_cells / nd
    rows.append((nd, c["n_rounds"], c.get("n_rounds_greedy"), c["max_degree"],
                 c["max_local_cells"], mean_cells,
                 c["max_local_cells"] / mean_cells, c["n_neighbour_pairs"]))
    print(f"nd={nd} done", flush=True)

print(f"\n{'ranks':>6}{'rounds':>8}{'greedy':>8}{'degree':>8}"
      f"{'max_cells':>11}{'mean':>10}{'max/mean':>10}{'pairs':>8}")
for r in rows:
    print(f"{r[0]:>6}{r[1]:>8}{str(r[2]):>8}{r[3]:>8}{r[4]:>11}"
          f"{r[5]:>10.0f}{r[6]:>10.2f}{r[7]:>8}")

a = {r[0]: r for r in rows}
if 64 in a and 128 in a:
    for i, name in ((1, "rounds"), (3, "max degree"), (6, "max/mean cells")):
        lo, hi = a[64][i], a[128][i]
        verdict = "WORSE" if hi > lo else ("same" if hi == lo else "better")
        print(f"64 -> 128 {name:>15}: {lo} -> {hi}  {verdict}")
print("PART_AB_DONE")
PYEOF
