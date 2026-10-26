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
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
from legoesm.parallel.sharded_dynamics import spmd_schedule_cost

NLEV, SUB, METHOD = 26, 9, "sfc"

# The SAME construction the bench does, in the same order: build, then reorder
# for the rank count, which is what the ladder's arms ran (reorder_for
# defaults to the device count). Scoring an unreordered mesh would describe a
# partition no run used, and the cost function must then be told the ownership
# is already baked in.
base = create_voronoi_mesh(subdivision_level=SUB, lloyd_iterations=0)
print(f"mesh nCells={base.nCells} nEdges={base.nEdges}", flush=True)

rows = []
for nd in (32, 64, 128, 256):
    mesh = reorder_voronoi_for_sharding(base, nd, method=METHOD)
    c = spmd_schedule_cost(mesh, nd, method=METHOD, already_reordered=True,
                           cell_width=NLEV + 2, edge_width=NLEV)
    mean = mesh.nCells / nd
    rows.append((nd, c["n_rounds"], c.get("n_rounds_greedy"), c["max_degree"],
                 c["max_local_cells"], mean, c["max_local_cells"] / mean,
                 c["n_neighbour_pairs"]))
    print(f"nd={nd} rounds={c['n_rounds']} degree={c['max_degree']} "
          f"max_cells={c['max_local_cells']}", flush=True)

print(f"\n{'ranks':>6}{'rounds':>8}{'greedy':>8}{'degree':>8}"
      f"{'max_cells':>11}{'mean':>10}{'max/mean':>10}{'pairs':>8}")
for r in rows:
    print(f"{r[0]:>6}{r[1]:>8}{str(r[2]):>8}{r[3]:>8}{r[4]:>11}"
          f"{r[5]:>10.0f}{r[6]:>10.2f}{r[7]:>8}")

a = {r[0]: r for r in rows}
if 64 in a and 128 in a:
    worse = []
    for i, name in ((1, "rounds"), (3, "max degree"), (6, "max/mean cells")):
        lo, hi = a[64][i], a[128][i]
        tag = "WORSE" if hi > lo else ("same" if hi == lo else "better")
        if tag == "WORSE":
            worse.append(name)
        print(f"64 -> 128 {name:>15}: {lo} -> {hi}  {tag}")
    if worse:
        print(f"\nThe split DEGRADES at 128 in: {', '.join(worse)}. That is a "
              f"candidate cause of the 822 -> 1240 ms kink and it is not the "
              f"thread pool.")
    else:
        print("\nThe split does NOT degrade at 128 on any of these measures, "
              "so the leading alternative to the thread-pool explanation is "
              "refuted and the kink is elsewhere.")
print("PART_AB_DONE")
PYEOF
