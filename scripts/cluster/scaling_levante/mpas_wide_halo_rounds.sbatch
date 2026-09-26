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
#
# THE KNOWN-ANSWER GATE PINS max_degree, NOT n_rounds -- and an earlier
# version of this gate pinned the wrong one. The reference run (sha
# 13ce1df4) recorded n_rounds 11 and max_degree 11 at depth 3. On current
# code the same call returns n_rounds 12 with max_degree still 11, and
# deterministically so over repeated trials. That is not drift in the
# partition: the schedule builder now chooses a SIZE-AWARE colouring, which
# deliberately spends one extra round to cut the padded payload (the
# padding audit measured inflation falling from 2.93 to 2.16 that way), and
# it reports the greedy alternative alongside -- greedy 11, size-aware 12.
# So n_rounds is a POLICY OUTPUT and a gate on it fires whenever the policy
# improves. max_degree is the property of the communication GRAPH that the
# reference actually pinned, and it has not moved.
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
              "max_local_edges", "production_strategy", "n_rounds_greedy",
              "partition_fingerprint"):
        if k in cost:
            row[k] = cost[k]
    rows.append(row)
    print("ROW", json.dumps(row), flush=True)

with open(f"{outdir}/wide_halo_rounds.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

d3 = rows[0]
# GLM: pin the GREEDY round count beside the degree. max_degree pins only
# the MAXIMUM vertex degree, so a split that redistributes edges at equal
# maximum passes it; n_rounds_greedy is deterministic given the graph and
# the builder, and is stable across the colouring-policy change that made
# n_rounds itself unpinnable.
if d3["max_degree"] != 11 or d3.get("n_rounds_greedy") != 11:
    print(f"KNOWN_ANSWER_FAILED: depth-3 max_degree = {d3['max_degree']} "
          f"and n_rounds_greedy = {d3.get('n_rounds_greedy')}, reference "
          "13ce1df4 says 11 and 11 — the communication graph moved, so the "
          "split is not the one the reference measured, quote NOTHING",
          flush=True)
    sys.exit(4)
d9 = rows[1]
print(f"KNOWN_ANSWER_OK: depth3 max_degree={d3['max_degree']} "
      f"greedy={d3['n_rounds_greedy']} rounds={d3['n_rounds']} "
      f"(the last is a colouring-policy output and is NOT gated). "
      f"depth9: rounds={d9['n_rounds']} "
      f"degree={d9['max_degree']} — step model: 33 -> {d9['n_rounds']} "
      "sequential collectives", flush=True)
if d9["n_rounds"] >= 25:
    print("VERDICT: REFUTED — depth-9 graph inflates the round count",
          flush=True)
    sys.exit(3)
EOF
RC=$?
echo "PROBE_EXIT=$RC"
[ $RC -eq 0 ] && echo "WH_ROUNDS_DONE" || echo "WH_ROUNDS_FAILED"
exit $RC
