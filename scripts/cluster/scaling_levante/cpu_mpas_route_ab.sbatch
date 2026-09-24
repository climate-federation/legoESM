#!/bin/bash -l
#SBATCH --job-name=mpas_route_ab
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=mpas_route_ab_%j.log
# ===========================================================================
# DOES THE MPAS ATMOSPHERE CPU PLATEAU GO AWAY ON THE mpi4jax LANE?
#
# The CPU ladder times the MPAS atmosphere through the multi-controller SPMD
# lane (jax.distributed + shard_map), whose CPU collectives are gloo. On the
# same ranks gloo costs 124 us per neighbour exchange and 1.3 ms per global
# reduction even on ib0, vs 4 us / 12 us for mpi4jax. The ocean MPAS CPU lane
# already runs through mpi4jax and scales 62x to 512 ranks; the atmosphere
# lane plateaus. An mpi4jax (route-A) MPAS atmosphere lane EXISTS
# (run_cpu_mpi_scaling.py --grid icosahedral -> voronoi_mpi) and no ladder
# has timed it at the full core share.
#
# ARMS: same mesh (subdivision 8, 50 Lloyd iterations, cached), 40 levels,
# float32, no physics, strong scaling, 4 ranks/node x 32 cores, SFC
# partition on both. Ranks 4, 8, 16, 32; each rank count runs A, B, A, B.
# The two lanes differ in MORE than transport (route A exchanges per RK
# stage, route B uses the wide halo), so this compares LANES, not
# transports.
#
# PRE-REGISTERED: CONFIRM (route A is the CPU lane to scale) if A at 32 ranks
# <= 0.7x B at 32 AND A's 4->32 speedup exceeds B's. REFUTE if A >= B at 32.
# Offline cannot answer it: the cost is the multi-node transport.
#
# SUBMIT (from the nature worktree): sbatch scripts/cluster/scaling_levante/cpu_mpas_route_ab.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export JAX_PLATFORMS=cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || exit 2
export LD_LIBRARY_PATH="/sw/spack-levante/gcc-13.4.0-c4mgfk/lib64:/sw/spack-levante/openmpi-4.1.2-mnmady/lib:${LD_LIBRARY_PATH:-}"
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_GLOO_IFACE=ib0
export LEGOESM_VORONOI_PARTITION=sfc
export JAX_ENABLE_X64=0
OUTDIR="${OUTDIR:-/scratch/b/b381103/legoesm_scaling/mpas_route_ab_j${SLURM_JOB_ID}}"
[ -e "$OUTDIR" ] && { echo "REFUSING: $OUTDIR exists" >&2; exit 1; }
mkdir -p "$OUTDIR"
SUB=8; NLEV=40; STEPS=30; WARMUP=5; ARM_TIMEOUT=1500
CPT=$(( ${SLURM_CPUS_ON_NODE:?} / 4 ))
echo "sha $(git rev-parse --short HEAD) cpus/rank $CPT s$SUB L$NLEV f32 outdir $OUTDIR"

arm() {  # arm <A|B> <nd> <rep>
  local lane="$1" nd="$2" rep="$3" nn=$(( ($2 + 3) / 4 ))
  local tag="${lane}_d${nd}_r${rep}" cmd
  if [[ $lane == A ]]; then
    cmd=(scripts/bench/run_cpu_mpi_scaling.py --grid icosahedral --mode strong
         --resolution "$SUB" --n-levels "$NLEV" --precision float32 --physics none
         --n-warmup "$WARMUP" --n-timing "$STEPS" --output-dir "$OUTDIR/$tag")
  else
    cmd=(scripts/bench/bench_mpas_spmd_scaling.py --multicontroller --n-devices "$nd"
         --subdivision "$SUB" --nlev "$NLEV" --lloyd 50 --wide-halo --partition-method sfc
         --steps "$STEPS" --warmup "$WARMUP" --out "$OUTDIR/$tag.jsonl")
  fi
  echo "=== $tag nodes=$nn $(date +%T)"
  timeout --kill-after=30 "$ARM_TIMEOUT" srun --export=ALL --nodes="$nn" --ntasks="$nd" \
      --ntasks-per-node=4 --cpus-per-task="$CPT" --cpu-bind=cores --kill-on-bad-exit=1 \
      "$PY" "${cmd[@]}" < /dev/null > "$OUTDIR/$tag.log" 2>&1 || echo "ARM FAILED $tag rc=$?"
  tail -n 3 "$OUTDIR/$tag.log"
}
for nd in 4 8 16 32; do
  for rep in 1 2; do arm A "$nd" "$rep"; arm B "$nd" "$rep"; done
done

$PY - "$OUTDIR" <<'PYEOF'
import glob, json, os, sys, collections
out = sys.argv[1]; rows = collections.defaultdict(list)
for f in glob.glob(os.path.join(out, "B_*.jsonl")):
    lane, d, r = os.path.basename(f)[:-6].split("_")
    for l in open(f):
        rec = json.loads(l); rows[("B", int(d[1:]))].append(rec["steady_median_ms"])
for f in glob.glob(os.path.join(out, "A_*", "**", "*.json"), recursive=True):
    d = json.load(open(f))
    if "time_per_step_ms" in d:
        rows[("A", int(d["n_ranks"]))].append(d["time_per_step_ms"])
print(f"{'ranks':>6} {'A mpi4jax ms':>22} {'B gloo ms':>22}")
for nd in (4, 8, 16, 32):
    print(f"{nd:>6} {str([round(x) for x in rows[('A', nd)]]):>22} {str([round(x) for x in rows[('B', nd)]]):>22}")
print("NOTE: A reports a MEAN over timed steps, B a MEDIAN; both listed per replicate.")
print("MPAS_ROUTE_AB_DONE")
PYEOF
