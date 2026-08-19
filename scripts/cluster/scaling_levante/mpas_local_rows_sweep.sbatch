#!/bin/bash -l
#SBATCH --job-name=mpas_rows_sweep
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=mpas_rows_sweep.%j.log
# ===========================================================================
# WHY DOES THE MPAS LANE GAIN ALMOST NOTHING FROM 8 TO 16 GPUs?
#
# Measured 2^n ladder, icosahedral L9, wide halo, 60 timed steps:
#   1 GPU 173.84 | 2 81.76 | 4 42.03 | 8 23.06 | 16 18.91 | 32 7.20
#   | 64 5.74 | 128 5.09 ms
# 8 -> 16 is a 1.22x speedup; 16 -> 32 is 2.63x, i.e. SUPER-linear.  Both
# 16-GPU receipts agree (19.16 and 18.91, per-rank spread 0.01 ms), so this
# is not a slow node and not a straggler -- every rank is equally slow.
#
# A step time that does not fall when the owned work halves means the work
# each device actually performs did not halve.  The wide (communication-
# avoiding) halo makes each device compute a ring of GHOST cells as well as
# its own, and that ring is rebuilt from the partition at every device
# count.  This job measures the ring OFFLINE, on CPU, from the production
# partition builders -- no GPU time -- for the same mesh, the same
# reorder target and the same halo depth the ladder ran.
#
# For each device count it prints owned rows, total local rows, the ratio,
# the schedule's round count and the padded/actual bytes per fill.
#
#   CONFIRMS the ghost-blow-up explanation: local/owned at 16 is a clear
#     outlier above the trend of its neighbours (e.g. >= 1.5x the value
#     interpolated between 8 and 32), or the round count spikes there.
#   REFUTES it: local/owned and the round count both sit on a smooth trend
#     through 8/16/32 -- the stall is then in the compiled kernels or the
#     collectives, and only the GPU no-comm/no-staging arms can name it.
#
# Narrow depth 3 is measured alongside as the control: it is the same
# partition machinery with a thinner ring, so a defect in the ring build
# shows up as a much larger wide/narrow gap at the outlier device count.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_local_rows_sweep.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export JAX_PLATFORMS=cpu
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
cd "$SUBMIT_DIR" || exit 2
# WORKTREE-IMPORT TRAP: the venv's editable install resolves legoesm.* to
# the MAIN checkout, so without this PYTHONPATH the job audits code this
# worktree does not have.
export PYTHONPATH="$SUBMIT_DIR/src:$(ls -d "$SUBMIT_DIR"/packages/* | tr '\n' ':' | sed 's/:$//'):${PYTHONPATH:-}"

"$PY" - <<'EOF'
import numpy as np
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
from legoesm.parallel.sharded_dynamics import (
    SPMD_HALO_DEPTH, _build_voronoi_partition_infra,
    _build_halo_send_maps, _build_ppermute_schedule,
)

SUBDIV = 9
NLEV = 26
# The wide step evaluates the RHS three times behind one fill, so its halo
# is three narrow depths deep -- the same arithmetic the step itself uses
# (_halo_depth_eff = SPMD_HALO_DEPTH * _wide_evals).
WIDE_DEPTH = SPMD_HALO_DEPTH * 3
DEVICES = (8, 16, 32, 64, 128)
cell_w = (NLEV + 2) * 4
edge_w = NLEV * 4

print(f"subdivision {SUBDIV}, narrow depth {SPMD_HALO_DEPTH}, "
      f"wide depth {WIDE_DEPTH}")
print(f"{'ndev':>5} {'depth':>6} {'owned':>10} {'local':>10} {'ratio':>7} "
      f"{'rounds':>7} {'actualMB':>9} {'paddedMB':>9} {'infl':>6}")

rows = {}
for n_dev in DEVICES:
    # reorder target matches the bench's --reorder-for <ndev>, which the
    # ladder passed for every point. Rebuild the mesh each time: the
    # reorder permutes it in place-equivalent fashion.
    mesh = create_voronoi_mesh(subdivision_level=SUBDIV, lloyd_iterations=0)
    mesh = reorder_voronoi_for_sharding(mesh, n_dev)
    nC, nE = int(mesh.nCells), int(mesh.nEdges)
    cells_per, edges_per = nC // n_dev, nE // n_dev
    for depth in (SPMD_HALO_DEPTH, WIDE_DEPTH):
        infra = _build_voronoi_partition_infra(mesh, n_dev, halo_depth=depth)
        max_lc, max_le = infra[5], infra[6]
        parts, owner = infra[7], infra[8]
        pairs, cs, cr, es, er = _build_halo_send_maps(
            parts, owner, n_dev, cells_per, edges_per)
        sched = _build_ppermute_schedule(parts, owner, n_dev, cells_per,
                                         edges_per, max_lc, max_le)
        n_rounds = sched["n_rounds"]
        hc, he = sched["halo_cells_per_round"], sched["halo_edges_per_round"]
        actual = (sum(len(v) for v in cs.values()) * cell_w
                  + sum(len(v) for v in es.values()) * edge_w)
        padded = 0
        for r in range(n_rounds):
            n_pairs_r = sum(1 for (a, b) in sched["ppermute_perms"][r]
                            if a != b)
            padded += n_pairs_r * (hc[r] * cell_w + he[r] * edge_w)
        ratio = max_lc / cells_per
        infl = padded / actual if actual else float("nan")
        rows[(n_dev, depth)] = (cells_per, max_lc, ratio, n_rounds)
        print(f"{n_dev:>5} {depth:>6} {cells_per:>10} {max_lc:>10} "
              f"{ratio:7.3f} {n_rounds:>7} {actual/1e6:9.2f} "
              f"{padded/1e6:9.2f} {infl:6.3f}")

# Gate written before submit: is 16 an outlier, or on the trend?
print()
r8 = rows[(8, WIDE_DEPTH)][2]
r16 = rows[(16, WIDE_DEPTH)][2]
r32 = rows[(32, WIDE_DEPTH)][2]
interp = (r8 + r32) / 2
print(f"wide local/owned  8: {r8:.3f}  16: {r16:.3f}  32: {r32:.3f}  "
      f"(midpoint of 8 and 32 = {interp:.3f})")
n8, n16, n32 = (rows[(d, WIDE_DEPTH)][3] for d in (8, 16, 32))
print(f"wide rounds       8: {n8}  16: {n16}  32: {n32}")
if r16 >= 1.5 * interp:
    print("VERDICT: CONFIRMS — the 16-device ghost ring is an outlier; the "
          "extra local rows are the missing speedup")
elif n16 > max(n8, n32):
    print("VERDICT: CONFIRMS via the schedule — 16 devices colour into more "
          "rounds than either neighbour")
else:
    print("VERDICT: REFUTES — ring size and round count are both on trend; "
          "the 8->16 stall is in the kernels or the collectives, and only "
          "the GPU no-comm / no-staging arms can name it")
EOF
echo "DONE"
