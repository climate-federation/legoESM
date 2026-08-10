#!/bin/bash -l
#SBATCH --job-name=mpas_wh_rounds
#SBATCH --account=bb1596
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=mpas_wh_rounds.%j.log
# ===========================================================================
# WIDE-HALO discriminator, part 1 (offline, CPU): does the depth-9 halo
# comm graph still colour to ~11 rounds per fill at s9/np64, or does the
# deeper ring reach more distant ranks and inflate the chromatic index?
#
# WHY. The MPAS GPU lane is bound by (sequential collectives per step) x
# (~155-310 us latency floor). The step exchanges once per RK3 stage:
# 11 coloured rounds x 3 fills = 33 collectives. Communication-avoiding
# wide halo (depth 9 = 3 stages x 3 stencil hops, exactly tight for
# SSP-RK3 per the shrinking-validity argument) fills ONCE per step. The
# win is (33 -> n_rounds(depth9)); if n_rounds(depth9) ~ 11-14 the step
# model predicts ~2x. If the depth-9 ring reaches many new ranks and the
# graph colours to 25+, the lever is dead before implementation.
#
# (a) NUMBER: n_rounds and max_degree at halo_depth {3, 9} for s9@64
#     (reorder-for 128, sfc — the census/production config), plus halo
#     cell/edge payload growth.
# (b) CONFIRMS lever if rounds(9) <= ~14 (step model: 33 rounds -> ~
#     rounds(9), latency-bound); REFUTES if rounds(9) >= ~25.
# (c) Offline on CPU via the COMMITTED estimator spmd_schedule_cost —
#     the same builders production calls. No GPU-hour needed.
#
# KNOWN-ANSWER CHECK: depth 3 must reproduce 11 rounds (census job
# 26791341: 33 CP / 3 fills). If it does not, the probe config drifted
# from the census config and NEITHER number may be quoted.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_wide_halo_rounds.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export JAX_PLATFORMS=cpu
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
cd "$SUBMIT_DIR" || exit 2
OUTDIR="${OUTDIR:-/scratch/b/b381103/legoesm_scaling/wh_rounds_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"

"$PY" - "$OUTDIR" <<'EOF'
import json, sys, subprocess
outdir = sys.argv[1]
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.parallel.sharded_dynamics import spmd_schedule_cost

sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                     text=True).stdout.strip()
mesh = create_voronoi_mesh(subdivision_level=9, lloyd_iterations=0)
rows = []
for depth in (3, 9):
    cost = spmd_schedule_cost(mesh, 64, method="sfc", reorder_target=128,
                              halo_depth=depth)
    row = {"git_sha": sha, "subdivision": 9, "n_dev": 64,
           "reorder_target": 128, "method": "sfc", "halo_depth": depth,
           "n_rounds": cost["n_rounds"], "max_degree": cost["max_degree"],
           "coloring_method": cost.get("coloring_method")}
    for k in ("halo_cells_total", "halo_edges_total", "max_local_cells",
              "max_local_edges", "production_strategy"):
        if k in cost:
            row[k] = cost[k]
    rows.append(row)
    print("ROW", json.dumps(row), flush=True)

with open(f"{outdir}/wide_halo_rounds.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

d3 = rows[0]
if d3["n_rounds"] != 11:
    print(f"KNOWN_ANSWER_FAILED: depth-3 rounds = {d3['n_rounds']}, "
          "census says 11 — config drifted, quote NOTHING", flush=True)
    sys.exit(4)
d9 = rows[1]
print(f"KNOWN_ANSWER_OK (depth3=11). depth9: rounds={d9['n_rounds']} "
      f"degree={d9['max_degree']} — step model: 33 -> {d9['n_rounds']} "
      "sequential collectives", flush=True)
EOF
RC=$?
echo "PROBE_EXIT=$RC"
[ $RC -eq 0 ] && echo "WH_ROUNDS_DONE" || echo "WH_ROUNDS_FAILED"
exit $RC
