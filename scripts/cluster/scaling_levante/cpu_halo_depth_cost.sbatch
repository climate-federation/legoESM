#!/bin/bash -l
#SBATCH --job-name=halo_cost
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=halo_cost_%j.log
# ===========================================================================
# HOW MUCH REDUNDANT WORK DOES THE WIDE HALO BUY AT 128 RANKS?
#
# The CPU atmosphere costs 51% MORE time going 64 -> 128 ranks. The partition
# was eliminated (it degrades only smoothly) and the per-rank core share was
# eliminated (more cores wins at every local problem size, down to an eighth
# of what a rank holds at 128 ranks). This is the next candidate, and the
# receipts point straight at it.
#
# WHAT THE RECEIPTS SAY. Those runs carry halo_depth 9 -- the wide halo,
# three times the base depth of 3 -- and the exchange index array they build
# has this shape per rank:
#
#     ranks    owned cells   index entries   ratio
#       16        163840        585133        3.6
#       32         81920        314020        3.8
#       64         40960        175578        4.3
#      128         20480         96942        4.7
#
# The wide halo trades communication for REDUNDANT COMPUTE: each rank carries
# a thicker skirt of cells it does not own and evaluates tendencies on them so
# it can skip exchanges. That trade gets worse as the owned region shrinks,
# because the skirt is a surface and the interior is a volume. At 128 ranks
# roughly four fifths of what a rank computes is halo.
#
# WHAT IS COMPUTED HERE, offline and free: the padded local cell and edge
# counts, exchange rounds and neighbour pairs at halo depth 3 and 9, for 32
# through 256 ranks, from the SAME builders production calls and at
# production weights. That gives the redundancy factor -- local cells over
# owned cells -- for both depths at each rank count.
#
# WHAT IT CAN AND CANNOT SETTLE. It quantifies the redundant WORK exactly. It
# does NOT price the exchanges the wide halo saves, so a depth that computes
# more can still win overall. The output is therefore a cost table and an
# arithmetic estimate of where the trade turns, never a recommendation: the
# halo depth is a production setting and changing it is a separate decision.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/cpu_halo_depth_cost.sbatch
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
BASE, WIDE = 3, 9

base = create_voronoi_mesh(subdivision_level=SUB, lloyd_iterations=0)
print(f"mesh nCells={base.nCells}", flush=True)

rows = []
for nd in (32, 64, 128, 256):
    mesh = reorder_voronoi_for_sharding(base, nd, method=METHOD)
    owned = base.nCells / nd
    cell = {}
    for depth in (BASE, WIDE):
        c = spmd_schedule_cost(mesh, nd, method=METHOD, already_reordered=True,
                               halo_depth=depth, cell_width=NLEV + 2,
                               edge_width=NLEV)
        cell[depth] = (c["max_local_cells"], c["n_rounds"],
                       c["n_neighbour_pairs"])
    rows.append((nd, owned, cell))
    print(f"nd={nd} done", flush=True)

print(f"\n{'ranks':>6}{'owned':>9}"
      f"{'d3 local':>10}{'d3 xN':>7}{'d3 rnds':>9}"
      f"{'d9 local':>10}{'d9 xN':>7}{'d9 rnds':>9}{'extra work':>12}")
for nd, owned, c in rows:
    l3, r3, _ = c[BASE]
    l9, r9, _ = c[WIDE]
    print(f"{nd:>6}{owned:>9.0f}"
          f"{l3:>10}{l3 / owned:>7.2f}{r3:>9}"
          f"{l9:>10}{l9 / owned:>7.2f}{r9:>9}"
          f"{l9 / l3:>11.2f}x")

print("\nThe wide halo buys FEWER EXCHANGES at the price of MORE CELLS. The\n"
      "rounds column is per halo fill and the local column is what every\n"
      "rank evaluates tendencies on. Where the extra work grows faster than\n"
      "the rounds fall, the trade has turned -- but this job does not price\n"
      "an exchange, so it cannot say that it HAS turned. That needs the\n"
      "measured per-collective cost.")
print("HALO_COST_DONE")
PYEOF
