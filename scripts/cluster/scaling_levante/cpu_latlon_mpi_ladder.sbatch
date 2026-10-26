#!/bin/bash -l
#SBATCH --job-name=latlon_mpi_ladder
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=latlon_mpi_ladder_%j.log
# ===========================================================================
# THE LAT-LON ATMOSPHERE ON THE OTHER TRANSPORT, AT THE LADDER'S OWN SIZES.
#
# Every CPU lane that plateaus records transport=gloo (jax.distributed CPU
# collectives under shard_map); the one lane that scales 62x records
# mpi4jax. On one instrument, same ranks, same minutes, 8 ranks across 2
# nodes (job 27668470): a neighbour exchange is 318 us / 0.11 GB/s on gloo
# against 4.2 us / 3.67 GB/s on MPI; a global sum 1640 us against 11.9 us.
# That is a primitive measurement. Both reviewers made the same point: a
# primitive ratio does not prove the LANES differ for that reason. This job
# is the lane-level test.
#
# It runs the EXISTING MPI atmosphere driver (scripts/bench/
# run_cpu_mpi_scaling.py, the same C-grid FV primitive-equation core as the
# SPMD ladder, decomposed over MPI via make_latlon_mpi_step) at exactly the
# SPMD ladder's problem -- 4096 x 8192 x 26, single precision, no physics --
# and exactly its rank counts and layout: 32 to 512 ranks, 4 ranks per node,
# 64 threads per rank. That lane has NEVER been measured on this machine.
#
# COMPARATOR, NOT GATE. The SPMD (gloo) lane's strong-scaling receipts are
#     32:2120  64:1387  128:1110  256:1015  512:997 ms/step
# There is no prior MPI-lane receipt to hold this job to, so those numbers
# are the comparison, and the internal gates are: every rank agrees on the
# rank count, the step is finite, and the rows are monotone enough to be a
# curve. Pre-registered reading, at 512 ranks:
#     MPI lane <= 500 ms   -> transport is the dominant CPU term (gloo 997)
#     MPI lane >= 900 ms   -> the plateau is NOT the transport; look at the
#                             lane's own work/imbalance next
#     between              -> transport is part of it, not all of it
# and the 128 -> 512 speedup (gloo: 1.11x) is reported next to it.
#
# ONE STATISTIC CAVEAT, stated up front so nobody reads past it: this driver
# reports time_per_step_ms as the MEAN over --n-timing steps between two
# barriers, while the SPMD ladder reports a MEDIAN over 60-step fused blocks.
# A mean is pulled up by stragglers, a median is not, so if anything this
# comparison is biased AGAINST the MPI lane.
#
# Both decompositions the driver offers run: the 1-D latitude band (what the
# gloo ladder ran) and the 2-D pencil, so the transport question and the
# decomposition question are answered on the same lane in the same job.
# ===========================================================================
set -uo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
export JAX_PLATFORMS=cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
# mpi4py/mpi4jax were built against this OpenMPI; the module system does not
# prepend it from a batch shell (memory: mpi-tests-on-levante).
export LD_LIBRARY_PATH="/sw/spack-levante/gcc-13.4.0-c4mgfk/lib64:/sw/spack-levante/openmpi-4.1.2-mnmady/lib:${LD_LIBRARY_PATH:-}"

RES=4096            # n_lat; the driver sets n_lon = 2 * n_lat = 8192
NLEV=26
PREC=float32
PHYS=none
N_WARMUP=3
N_TIMING=60         # matches the SPMD ladder's 60-step blocks
LADDER="${LADDER:-512 256 128 64 32}"
DECOMPS="${DECOMPS:-band 2d}"
ARM_TIMEOUT=1500    # 10 arms at this cap is 4.2 h inside a 6 h wall
OUTDIR="${OUTDIR:-/scratch/b/b381103/legoesm_scaling/latlon_mpi_ladder_j${SLURM_JOB_ID}}"
[ -e "$OUTDIR" ] && { echo "REFUSING: $OUTDIR exists" >&2; exit 1; }
mkdir -p "$OUTDIR"

echo "== header =="
echo "sha        $(git -C "$REPO" rev-parse --short HEAD)"
echo "py         $PY"
echo "provenance $($PY -c 'import legoesm.parallel.latlon_mpi as m; print(m.__file__)')"
echo "mpi        $($PY -c 'from mpi4py import MPI; print(MPI.Get_library_version().strip()[:40])')"
echo "problem    ${RES}x$((2 * RES))x${NLEV} ${PREC} physics=${PHYS} strong; n_timing=${N_TIMING}"
echo "layout     4 ranks/node, 64 threads/rank; ladder ${LADDER}; decompositions ${DECOMPS}"
echo "comparator gloo lane 32:2120 64:1387 128:1110 256:1015 512:997 ms/step (median of 60-step blocks)"

want=0; ok=0
for DEC in $DECOMPS; do
  for ND in $LADDER; do
    NN=$(( ND / 4 )); want=$(( want + 1 ))
    extra=(); [ "$DEC" = 2d ] && extra=(--latlon-2d)
    echo "=== ${DEC} nd=${ND} nodes=${NN} $(date +%T)"
    if timeout --kill-after=30 "$ARM_TIMEOUT" srun --export=ALL --nodes="$NN" \
            --ntasks="$ND" --ntasks-per-node=4 --cpus-per-task=64 --cpu-bind=cores \
            --kill-on-bad-exit=1 \
            "$PY" scripts/bench/run_cpu_mpi_scaling.py --grid latlon --mode strong \
            --resolution "$RES" --n-levels "$NLEV" --precision "$PREC" --physics "$PHYS" \
            --n-warmup "$N_WARMUP" --n-timing "$N_TIMING" \
            ${extra[@]+"${extra[@]}"} --output-dir "$OUTDIR"; then
        ok=$(( ok + 1 ))
    else
        echo "ARM FAILED ${DEC} nd=${ND} rc=$? (124 = timeout)"
    fi
  done
done
echo "=== arms: $ok of $want completed ==="

$PY - "$OUTDIR" <<'PYEOF'
import glob, json, math, os, sys
out = sys.argv[1]
gloo = {32: 2120.1, 64: 1387.2, 128: 1110.0, 256: 1014.5, 512: 996.7}
rows = {}
for f in glob.glob(os.path.join(out, "**", "latlon*_none_strong_r4096_n*_float32.json"),
                   recursive=True):
    d = json.load(open(f))
    dec = d.get("decomposition") or ("2d" if "_2d_" in os.path.basename(f) else "band")
    nd = int(d["n_ranks"]); ms = float(d["time_per_step_ms"])
    if not (math.isfinite(ms) and ms > 0):
        print(f"REFUSED {os.path.basename(f)}: time_per_step_ms={ms!r}"); continue
    if d.get("finite_ok") is False or d.get("valid") is False:
        print(f"REFUSED {os.path.basename(f)}: marked invalid/non-finite"); continue
    rows[(dec, nd)] = ms

print("\nstatistic caveat: MPI rows are a MEAN over timed steps between barriers; "
      "the gloo comparator is a MEDIAN of 60-step blocks. Stragglers raise a mean, "
      "so this reads AGAINST the MPI lane if anything.")
print(f"\n{'ranks':>6} {'gloo band':>10} {'MPI band':>10} {'MPI 2-D':>10} "
      f"{'gloo/MPI band':>14}")
for nd in sorted(gloo):
    b = rows.get(("band", nd)); t = rows.get(("2d", nd))
    r = f"{gloo[nd] / b:.2f}x" if b else "-"
    print(f"{nd:>6} {gloo[nd]:>10.0f} {b if b is None else round(b):>10} "
          f"{t if t is None else round(t):>10} {r:>14}")

b128, b512 = rows.get(("band", 128)), rows.get(("band", 512))
if b512 is None:
    print("\nVERDICT WITHHELD: no 512-rank MPI band row. The pre-registered reading "
          "is at 512 ranks and nothing substitutes for it.")
else:
    print(f"\nMPI band at 512 ranks: {b512:.0f} ms (gloo 997). "
          + (f"128 -> 512 speedup {b128 / b512:.2f}x (gloo 1.11x)." if b128 else
             "no 128-rank row for the speedup."))
    if b512 <= 500:
        print("TRANSPORT DOMINATES: the same core on MPI is at least 2x faster at "
              "512 ranks. The CPU plateau is the communication library.")
    elif b512 >= 900:
        print("NOT THE TRANSPORT: the same core on MPI plateaus where gloo does. "
              "The non-shrinking cost is in the lane's own work or imbalance; the "
              "primitive gap does not reach the step.")
    else:
        print("PARTIAL: the transport is part of the plateau and not all of it.")
t512 = rows.get(("2d", 512))
if b512 and t512:
    print(f"decomposition on the MPI lane at 512 ranks: band {b512:.0f}, 2-D {t512:.0f}, "
          f"band/2-D {b512 / t512:.2f}x -- on the SAME transport, so this ratio "
          f"is the decomposition alone.")
print("LATLON_MPI_LADDER_DONE")
PYEOF
