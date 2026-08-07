#!/bin/bash -l
#SBATCH --job-name=mpas_payload_ab
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=mpas_payload_ab.%j.log
# ===========================================================================
# PAYLOAD A/B: does the halo collective's 216 us scale with what it sends?
#
# Cleaner than the skew route, which needs cross-rank timestamps that four
# independent nsys sessions do not share (each report has its own t=0), and
# which can only ever show ASYMMETRY.  This changes the PAYLOAD at a FIXED
# schedule and watches the collective itself.
#
# THE LEVER.  --physics kessler attaches the q_v/q_c/q_r tracers, so the
# packed cell record (_pack_cell_state: T | p_s | phis | tracers) goes from
#   W = nlev*(1+0)+2 =  28   (physics none)
# to
#   W = nlev*(1+3)+2 = 106   (kessler)
# = 3.79x the CELL term.  Edge payload (nlev per entity) is unchanged, so the
# total per-round bytes rise by less than 3.79x; the analysis prices the
# exact mixture per round from the schedule.  Mesh, device count, partition
# method and reorder target are IDENTICAL across arms, so the schedule --
# round count, partner map, per-round entity counts -- does not move.
#
# PREDICTION, written before the run.
#   PAYLOAD-bound: the SendRecv median tracks the byte increase. With the
#     measured at-scale fabric (29.7 us latency, 12.1 GB/s) a 216 us round
#     that is mostly transfer becomes several hundred us.
#   WAIT-bound: the median barely moves -- a rank idling on a slower peer
#     idles just as long when the message is bigger.
#   CONFIRMS payload if the median rises by >=2x; CONFIRMS wait if it rises
#     <=1.2x; anything between is mixed and is reported as such.
#
# THE CONFOUND, stated because it is real: kessler also adds COMPUTE (RK
# tracer advection), so STEP time is not a clean readout and is NOT the
# observable here.  The observable is the ncclDevKernel_SendRecv duration
# distribution itself, which added compute does not lengthen.  It can shift
# ARRIVAL times, which is exactly why this is reported as a payload test and
# not as a wait test.
#
# Single-rank nsys is sufficient: this compares a distribution against itself
# across arms, so no cross-report timebase is involved.
#
# SUBMIT:  sbatch scripts/cluster/scaling_levante/mpas_payload_ab.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }

# Pin the collective-affecting XLA flags so an inherited env cannot combine or
# pipeline the permutes differently between arms (codex: the repo's own notes
# say the compiled executable may do so, which would shear the comparison).
unset LEGOESM_XLA_OVERLAP
export LEGOESM_XLA_OVERLAP=0

NSYS_BIN="${NSYS_BIN:-/sw/spack-levante/cuda-12.2.0-2ttufp/nsight-systems-2023.2.3/target-linux-x64/nsys}"
[ -x "$NSYS_BIN" ] || { echo "nsys not executable: $NSYS_BIN"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_payload_ab_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"
export NSYS_BIN NSYS_OUTDIR="$OUTDIR" PROFILE_RANKS=0 PY_BIN="$PY"
WRAPPER="${SUBMIT_DIR}/scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh"

run_arm () {  # $1=label  $2=physics
  echo "=== [ab] arm $1 (physics=$2) ==="
  # Each arm needs its own profile filename; the wrapper derives it from the
  # rank, so give each arm its own directory.
  local d="$OUTDIR/$1"; mkdir -p "$d"
  NSYS_OUTDIR="$d" JAX_ENABLE_X64=0 srun --ntasks=64 \
    --distribution=block:block --cpu-bind=cores --kill-on-bad-exit=1 \
    "$WRAPPER" scripts/bench/bench_mpas_spmd_scaling.py \
      --n-devices 64 --subdivision 9 --nlev 26 --steps 12 --warmup 3 \
      --lloyd 0 --partition-method sfc --reorder-for 128 --multicontroller \
      --physics "$2" --out "$d/s9np64.jsonl"
  local rc=$?
  echo "[ab] arm $1 exit=$rc"
  # An exit code is not evidence. Require the COLLECTIVE to be present, not
  # merely that a file exists: the previous capture wrote four valid-looking
  # sqlite files of which two contained no SendRecv kernel at all, and a
  # file-existence gate passed it.
  local n
  n=$("$PY" - "$d/rank_0.sqlite" <<'PYEOF'
import sqlite3, sys
try:
    con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    print(con.execute(
        "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName = s.id "
        "WHERE s.value LIKE 'ncclDevKernel_SendRecv%'").fetchone()[0])
except Exception:
    print(0)
PYEOF
)
  echo "[ab] arm $1 SendRecv kernels: $n"
  [ "${n:-0}" -gt 0 ] || { echo "[ab] arm $1 produced NO collectives"; return 90; }
  return $rc
}

run_arm dry none;      RC_A=$?
run_arm wet kessler;   RC_B=$?

echo "[ab] dry rc=$RC_A  wet rc=$RC_B"
if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then
  echo "AB_FAILED"; exit 1
fi

echo "=== [ab] SendRecv duration distributions ==="
"$PY" - "$OUTDIR" <<'PYEOF'
import sqlite3, statistics as st, sys, json, os
base = sys.argv[1]
out = {}
for arm in ("dry", "wet"):
    con = sqlite3.connect(f"file:{base}/{arm}/rank_0.sqlite?mode=ro", uri=True)
    d = sorted((e - s) / 1e3 for s, e in con.execute(
        "SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName = s.id "
        "WHERE s.value LIKE 'ncclDevKernel_SendRecv%' ORDER BY k.start"))
    con.close()
    out[arm] = {"n": len(d), "median_us": st.median(d),
                "p25_us": d[len(d)//4], "p75_us": d[3*len(d)//4],
                "total_ms": sum(d)/1e3}
    print(f"{arm:4s}: n={len(d):5d} p25={out[arm]['p25_us']:8.1f} "
          f"median={out[arm]['median_us']:8.1f} p75={out[arm]['p75_us']:8.1f} "
          f"total={out[arm]['total_ms']:8.2f} ms")
r = out["wet"]["median_us"] / out["dry"]["median_us"]
out["median_ratio_wet_over_dry"] = r
# Cell payload rises 106/28 = 3.79x; edges are unchanged, so the total byte
# rise is less. The reading is stated, the verdict is not.
print(f"\nmedian ratio wet/dry = {r:.2f}x   (cell-record width 28 -> 106)")
print("counts MUST match across arms; a difference means the schedule moved "
      "and the arms are not comparable.")
if out["dry"]["n"] != out["wet"]["n"]:
    print(f"MISMATCHED COLLECTIVE COUNTS: {out['dry']['n']} vs {out['wet']['n']}")
with open(os.path.join(base, "payload_ab.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"JSON: {base}/payload_ab.json")
PYEOF
echo "AB_DONE"
