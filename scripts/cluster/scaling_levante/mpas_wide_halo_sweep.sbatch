#!/bin/bash -l
#SBATCH --job-name=mpas_wh_sweep
#SBATCH --account=bb1596
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=mpas_wh_sweep.%j.log
# ===========================================================================
# WIDE-HALO sweep across DEVICE COUNT (offline, CPU): where does the
# communication-avoiding wide halo stop paying?
#
# WHY. The CPU MPAS atmosphere strong-scaling curve at subdivision 9 is
# flat and non-monotonic (4 nodes 919 ms, 8 nodes 1005, 16 nodes 822,
# 32 nodes 1240, 64 nodes 1095, 128 nodes 682 -- 32x the machine buys
# 1.35x). Every one of those arms ran LEGOESM_MPAS_WIDE_HALO=1, i.e.
# ONE fill per step of a depth-9 halo instead of THREE fills of depth 3.
# That trade sells bytes to buy latency. The halo's cell count grows
# faster than linearly in depth once the halo ring is comparable to the
# owned block, so the trade must invert somewhere up the ladder -- and
# above that point the pin is costing time, not saving it.
#
# (a) NUMBER, per device count nd in {16,32,64,128,256,512} at
#     subdivision 9, sfc: payload cells received per STEP and sequential
#     collectives per STEP, for depth 3 (3 fills) and depth 9 (1 fill).
#     byte_ratio = halo9 / (3 * halo3); coll_ratio = rounds9 / (3*rounds3).
# (b) CONFIRMS that wide halo is the wrong pin at the top of the ladder
#     if byte_ratio rises above ~1.3 by nd=128. REFUTES if byte_ratio
#     stays within 1.0 +- 0.15 across the whole sweep, in which case wide
#     halo is free on bytes and the flat curve is something else.
# (c) Offline on one CPU core via the COMMITTED estimator
#     spmd_schedule_cost -- the same builders production calls. No GPU
#     hour, and no allocation can answer it more cheaply.
#
# KNOWN-ANSWER CHECK: depth 3 at nd=64 with reorder_target=128 must
# reproduce 11 rounds (census job 26791341: 33 collective-permutes over
# 3 fills). If it does not, the probe config has drifted from the census
# config and NO number in this file may be quoted.
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
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_wide_halo_sweep.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export JAX_PLATFORMS=cpu
# Without this the venv resolves legoesm to the MAIN checkout and the
# probe silently measures a different tree (see the provenance guard
# below, which is what caught it).
PP="$SUBMIT_DIR/src"
for _p in atmosphere core coupler ice land ml ocean tools; do
  PP="$PP:$SUBMIT_DIR/packages/$_p"
done
export PYTHONPATH="$PP:${PYTHONPATH:-}"
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
cd "$SUBMIT_DIR" || exit 2
OUTDIR="${OUTDIR:-/scratch/b/b381103/legoesm_scaling/wh_sweep_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"

"$PY" - "$OUTDIR" <<'EOF'
import json, os, sys, subprocess
outdir = sys.argv[1]
from legoesm.grids.voronoi import create_voronoi_mesh
import legoesm.parallel.sharded_dynamics as _sd
from legoesm.parallel.sharded_dynamics import spmd_schedule_cost

# PROVENANCE. A previous run of this job imported the MAIN checkout rather
# than this worktree and died on an unexpected keyword, which is the
# cheap version of the same mistake silently answering a different
# question. Refuse unless the module came from the tree we submitted from.
_want = os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())
print("estimator module:", _sd.__file__, flush=True)
if not _sd.__file__.startswith(_want):
    sys.exit(f"REFUSED: imported {_sd.__file__}, expected it under "
             f"{_want} — set PYTHONPATH to this worktree's packages")

SUBDIV = 9
NLEV = 26          # the production atmosphere column
FILLS = {3: 3, 9: 1}        # SSP-RK3: 3 evals; depth 9 covers all 3
sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                     text=True).stdout.strip()
mesh = create_voronoi_mesh(subdivision_level=SUBDIV, lloyd_iterations=0)

def score(nd, depth):
    # Profile device 0: ppermute moves the round's PADDED extent, so the
    # per-round halo_cells is what device 0 actually puts on the wire.
    c = spmd_schedule_cost(mesh, nd, method="sfc", reorder_target=nd,
                           halo_depth=depth, round_profile_for_device=0,
                           cell_width=NLEV + 2, edge_width=NLEV)
    assert c["production_weighted"], c["weights_note"]
    payload = sum(r["halo_cells"] for r in c["round_profile"])
    return {"n_dev": nd, "halo_depth": depth,
            "n_rounds": c["n_rounds"], "max_degree": c["max_degree"],
            "cells_per_device": c["cells_per_device"],
            "max_local_cells": c["max_local_cells"],
            "production_strategy": c["production_strategy"],
            "halo_cells_per_fill_dev0": payload,
            "fills_per_step": FILLS[depth],
            "halo_cells_per_step_dev0": payload * FILLS[depth],
            "collectives_per_step": c["n_rounds"] * FILLS[depth]}

# --- known-answer gate: must run BEFORE anything is quoted -------------
known = spmd_schedule_cost(mesh, 64, method="sfc", reorder_target=128,
                           halo_depth=3, cell_width=NLEV + 2,
                           edge_width=NLEV)
print(f"KNOWN_ANSWER depth3 nd=64 target=128 -> max_degree="
      f"{known['max_degree']} (reference 13ce1df4: 11), partition_fingerprint="
      f"{known['partition_fingerprint']}, rounds={known['n_rounds']} "
      f"greedy={known['n_rounds_greedy']} "
      f"coloring={known['coloring_method']} "
      f"production_weighted={known['production_weighted']}", flush=True)
if known["max_degree"] != 11:
    print("KNOWN_ANSWER_FAILED -- the communication graph moved, so the "
          "split is not the one the reference measured; quote NOTHING",
          flush=True)
    sys.exit(4)

rows = []
for nd in (16, 32, 64, 128, 256, 512):
    per = {}
    for depth in (3, 9):
        r = score(nd, depth)
        r["git_sha"] = sha
        r["subdivision"] = SUBDIV
        rows.append(r)
        per[depth] = r
        print("ROW", json.dumps(r), flush=True)
    b = per[9]["halo_cells_per_step_dev0"] / per[3]["halo_cells_per_step_dev0"]
    k = per[9]["collectives_per_step"] / per[3]["collectives_per_step"]
    print(f"RATIO nd={nd:4d}  bytes_wide/narrow={b:.3f}  "
          f"collectives_wide/narrow={k:.3f}", flush=True)

with open(f"{outdir}/wide_halo_sweep.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("SWEEP_DONE", flush=True)
EOF
RC=$?
echo "PROBE_EXIT=$RC"
exit $RC
