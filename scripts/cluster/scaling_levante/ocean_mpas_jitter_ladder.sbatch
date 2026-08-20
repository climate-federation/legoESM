#!/bin/bash -l
#SBATCH --job-name=oc_jitlad
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=oc_jitlad.%j.log
# Ocean-MPAS rank-count term, round 3 — the two top discriminators from
# the GLM-5.2 consult (BSP tail-latency model; PLAUSIBLE until these):
#
# ARM A: JAX-free spin+barrier jitter probe at 32 vs 128 ranks,
#   segment 2.5 ms (~the ocean step's inter-sync gap; ~100 syncs/step).
#   CONFIRMS jitter if bsp/solo amplification at 128 ranks is large
#   enough that (amp128-amp32) x ~100 segments x 2.5 ms ~ D (43 ms),
#   i.e. amplification difference >~ 0.17. REFUTES if both ~1.0x —
#   then the term is in the mpi4jax/JAX host-callback pipeline itself.
#
# ARM B: SAME-MESH rank ladder s8@{32,64,128} (global size FIXED,
#   only ranks vary; tiles 20480/10240/5120). Splits global-size from
#   rank-count: the matched-tile pair (s7@32 vs s8@128) conflated them.
#   Read: per-doubling speedups vs the atm lane's pure cells/rank law.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv-mpi/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/oc_jitlad_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0

echo "=== ARM A: jitter probe ==="
for NP in 32 128; do
  NODES=$(( NP / 32 )); [ "$NODES" -lt 1 ] && NODES=1
  srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=32 \
      --distribution=block:cyclic --cpu-bind=cores --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_bsp_jitter.py --segment-ms 2.5 --iters 400 \
      --out "$OUTDIR/jitter_np${NP}.jsonl" < /dev/null \
    || { echo "jitter np$NP FAILED"; rc=1; }
done

echo "=== ARM B: s8 same-mesh ladder ==="
COMMON="--mode strong --nlev 10 --precision float64 --partition-method geometric \
  --steps 12 --warmup 2 --block-steps 8 --blocks 2 --probe-steps 3 \
  --halo-refresh auto --barotropic-solver explicit_substep --n-substeps 10"
for NP in 32 64 128; do
  NODES=$(( NP / 32 )); [ "$NODES" -lt 1 ] && NODES=1
  srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=32 \
      --distribution=block:cyclic --cpu-bind=cores --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_ocean_mpas_scaling.py \
      --subdivision 8 $COMMON \
      --out "$OUTDIR/s8np${NP}.jsonl" < /dev/null \
    || { echo "s8 np$NP FAILED"; rc=1; }
done

echo "=== RECEIPTS ==="
"$PY" - "$OUTDIR" <<'EOF' || rc=1
import json, math, sys, pathlib
out = pathlib.Path(sys.argv[1])
J = {}
for np_ in (32, 128):
    r = json.loads(open(out / f"jitter_np{np_}.jsonl").readline())
    assert r["n_ranks"] == np_
    J[np_] = r
    print(f"jitter np{np_}: solo {r['solo_median_ms']:.3f} ms, "
          f"bsp {r['bsp_median_ms']:.3f} ms, amp {r['amplification']:.3f}, "
          f"p95 {r['bsp_p95_ms']:.3f}")
d_amp = J[128]["amplification"] - J[32]["amplification"]
pred = d_amp * 100 * 2.5
print(f"amp delta {d_amp:.3f} -> predicted step-delta contribution "
      f"~{pred:.1f} ms vs measured D=43 ms")
T = {}
for np_ in (32, 64, 128):
    rows = [json.loads(l) for l in open(out / f"s8np{np_}.jsonl")]
    assert len(rows) == 1
    r = rows[0]
    assert r["metadata"]["n_ranks"] == np_, np_
    assert math.isfinite(r["steady_median_ms"]) and r["steady_median_ms"] > 0
    T[np_] = r["steady_median_ms"]
    print(f"s8 np{np_}: {T[np_]:9.2f} ms")
print(f"s8 per-doubling speedups: 32->64 {T[32]/T[64]:.3f}, "
      f"64->128 {T[64]/T[128]:.3f} (ideal 2.0)")
EOF
echo "DONE rc=$rc"; exit $rc
