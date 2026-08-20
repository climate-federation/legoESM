#!/bin/bash -l
#SBATCH --job-name=mpas_round_attr
#SBATCH --account=bb1596
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=mpas_nsys_round_attribution.%j.log
# ===========================================================================
# Is the MPAS halo collective's 216 us median PAYLOAD or WAIT?  CPU-only,
# ZERO GPU hours — it re-reads an nsys trace that already exists.
#
# WHY IT MATTERS: the two mechanisms imply opposite engineering, and the
# spread between them is the whole uncertainty in whether a round-depth
# partitioner (7-14 days) is worth building.
#   payload-bound -> cutting ROUND COUNT saves only ~30 us x rounds removed,
#                    because the same bytes still have to move.  LOW end.
#   wait-bound    -> the collective idles on the slowest peer; cutting rounds
#                    saves much more, but so would fixing the imbalance.
#
# ALREADY ESTABLISHED from the same trace (rank 0, s9/np64, job 26680051):
#   * median 216.2 us, and the durations are REPRODUCIBLE per schedule slot
#     (most positions CV <= 1.5 % across 9 steps, against 63 % overall), so
#     random rank jitter is REFUTED as the dominant term.
#   * SendRecv is 5.94 of 9.72 ms = 61 % of the step, and rank 0 is not even
#     the critical rank (it sits in fewer colour classes than max_degree).
#
# WHAT IS UNRESOLVED, and is the whole point of this job: the trace and the
# schedule must describe the SAME partition before their per-round numbers
# can be paired.  They currently do not — scoring s9/np64/sfc with
# reorder_target = 64 puts device 0 in 7 rounds, but the trace shows 8 per
# fill (288 calls = 12 steps x 24, i.e. 8 rounds x 3 fills).  The probe
# REFUSES to pair in that state rather than reporting a number.
#
# HYPOTHESIS UNDER TEST: the traced run passed `--reorder-for 128` (every
# sibling MPAS launcher in this directory does; that run's own launcher is a
# throwaway and is gone, so this cannot be read off the manifest).  Ownership
# built for 128 but run at 64 is a DIFFERENT partition, hence a different
# round count.  The scorer models exactly that case via reorder_target.
#   CONFIRMS: reorder_target=128 puts device 0 in 8 rounds, so 8 x 3 = 24
#             divides the 288 calls and the pairing becomes legitimate.
#   REFUTES:  it does not, in which case the trace's provenance is not
#             recoverable and the payload/wait question needs a NEW nsys
#             capture with a recorded --reorder-for (a GPU job), not this one.
#
# Arm 1 is the cheap decisive test.  Arm 2 only runs if arm 1 confirms, and
# produces the actual correlation.
#
# RUN SERIALLY AND WITH REAL MEMORY: two concurrent s9 mesh builds OOM-killed
# each other on the shared interactive node.  s9 is 2.62 M cells (a 1.8 GB
# cached npz) and the 64-device partition infra on top of it is what needs
# the 200 G, not the trace.
#
# SUBMIT (from the repo root):
#   sbatch scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }

TRACE=/scratch/b/b381103/legoesm_scaling/s9_nsys_j26680051/s9np64_rank0.sqlite
OUT=results/a1/mpas_schedule_cost
mkdir -p "$OUT"

if [ ! -r "$TRACE" ]; then
  echo "[attr] trace not readable: $TRACE"
  exit 2
fi

echo "=== [attr] arm 1/2: which reorder_target reproduces the traced schedule? ==="
date -u +"[attr] start %Y-%m-%dT%H:%M:%SZ"
"$PY" - "$TRACE" <<'PYEOF'
import sys, json, sqlite3
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.parallel.sharded_dynamics import spmd_schedule_cost

con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
n_calls = con.execute(
    "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    "JOIN StringIds s ON k.demangledName = s.id "
    "WHERE s.value LIKE 'ncclDevKernel_SendRecv%'").fetchone()[0]
con.close()
print(f"[attr] trace holds {n_calls} SendRecv calls", flush=True)

mesh = create_voronoi_mesh(subdivision_level=9, lloyd_iterations=0)
found = {}
for target in (64, 128):
    cost = spmd_schedule_cost(mesh, 64, method="sfc", reorder_target=target,
                              round_profile_for_device=0)
    dev_rounds = len(cost["round_profile"])
    # A candidate is viable only if some whole number of halo fills per step
    # makes the device's round count divide the trace exactly.
    fills = [k for k in range(1, 9) if n_calls % (dev_rounds * k) == 0]
    print(f"[attr] reorder_target={target}: n_rounds={cost['n_rounds']} "
          f"max_degree={cost['max_degree']} device0_rounds={dev_rounds} "
          f"viable_fills_per_step={fills}", flush=True)
    if fills:
        found[target] = {"device0_rounds": dev_rounds, "fills": fills,
                         "n_rounds": cost["n_rounds"],
                         "max_degree": cost["max_degree"]}

with open("results/a1/mpas_schedule_cost/reorder_target_probe.json", "w") as f:
    json.dump({"n_calls": n_calls, "viable": found}, f, indent=2)

if not found:
    print("[attr] NO reorder_target reproduces the traced schedule. The "
          "trace's provenance is not recoverable; answering payload-vs-wait "
          "needs a NEW capture with a recorded --reorder-for.", flush=True)
    raise SystemExit(3)
print(f"[attr] viable targets: {sorted(found)}", flush=True)
PYEOF
RC1=$?
echo "[attr] arm 1 exit=$RC1"
date -u +"[attr] end   %Y-%m-%dT%H:%M:%SZ"

if [ "$RC1" -ne 0 ]; then
  echo "[attr] arm 1 did not identify the traced schedule — NOT running the"
  echo "[attr] correlation, because pairing rounds against unrelated"
  echo "[attr] collectives is how a confident wrong answer gets made."
  echo "ATTR_UNRESOLVED"
  exit "$RC1"
fi

echo "=== [attr] arm 2/2: payload correlation at the confirmed target ==="
date -u +"[attr] start %Y-%m-%dT%H:%M:%SZ"
TARGET=$("$PY" -c '
import json
d = json.load(open("results/a1/mpas_schedule_cost/reorder_target_probe.json"))
# Prefer a target that is NOT the device count: reorder_target == n_dev is
# the scorer default and would already have aligned, so if a non-default
# target is viable it is the informative one.
viable = sorted(int(k) for k in d["viable"])
print(next((t for t in viable if t != 64), viable[0]))')
echo "[attr] using reorder_target=$TARGET"
"$PY" scripts/bench/analyze_nsys_halo_rounds.py \
    --sqlite "$TRACE" \
    --subdivision 9 --n-devices 64 --partition-method sfc --lloyd 0 \
    --reorder-target "$TARGET" --nlev 26 \
    --out "$OUT/nsys_halo_rounds_s9np64.json"
RC2=$?
echo "[attr] arm 2 exit=$RC2"
date -u +"[attr] end   %Y-%m-%dT%H:%M:%SZ"
if [ "$RC2" -ne 0 ]; then
  echo "ATTR_FAILED_ARM2"
  exit "$RC2"
fi
echo "ATTR_DONE"
