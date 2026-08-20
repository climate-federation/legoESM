#!/bin/bash -l
#SBATCH --job-name=scan_census
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --exclude=l50100
#SBATCH --output=scan_census.%j.log
# ===========================================================================
# Combined MPAS discriminators (one allocation, 4 sruns):
#   0. parity gate s6@4 wide=1  — guards the wide arms (the 26849483
#      gate correctly REFUSED a mask regression; metric reverted).
#   1. loop_a : default step, per-step Python loop  (baseline)
#   2. scan_b : default step, --timed-scan (ONE jit(lax.scan) window,
#      one sync) — host-dispatch/pipelining share discriminator.
#      Gate: scan/loop <= 0.85 host share REAL; >= 0.95 null.
#   3. wide_c : wide halo + rank-0 XLA dump — census of the compiled
#      wide executable. CP must equal 11 (offline rounds receipt,
#      job 26845329) or the 26846337 0.970 verdict is VOID.
# Protocol otherwise byte-identical to the 26846337 arms.
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_scan_census.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/scan_census_j${SLURM_JOB_ID}}"
DUMP="$OUTDIR/dump_rank0"
mkdir -p "$DUMP"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"

cat > "$OUTDIR/rank_wrapper.sh" <<'WRAP'
#!/usr/bin/env bash
set -uo pipefail
if [ "${SLURM_PROCID:-1}" = "0" ] && [ "${WANT_DUMP:-0}" = "1" ]; then
  export XLA_FLAGS="--xla_dump_to=${DUMP_DIR}${XLA_FLAGS:+ $XLA_FLAGS}"
fi
exec "$PY_BIN" "$@"
WRAP
chmod +x "$OUTDIR/rank_wrapper.sh"
export PY_BIN="$PY" DUMP_DIR="$DUMP"
rc=0

echo "=== arm 0: parity gate s6@4 wide=1 ==="
WANT_DUMP=0 LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO=0 \
JAX_ENABLE_X64=0 \
  srun --nodes=1 --ntasks=4 --ntasks-per-node=4 --gpus-per-node=4 \
    --gpu-bind=none --kill-on-bad-exit=1 \
  "$OUTDIR/rank_wrapper.sh" scripts/bench/bench_mpas_spmd_scaling.py \
    --multicontroller --n-devices 4 --subdivision 6 --nlev 8 \
    --steps 4 --warmup 1 --lloyd 0 --partition-method sfc \
    --halo-strategy ppermute --parity-gate \
    --out "$OUTDIR/parity.jsonl" \
  || { echo "WIDE PARITY GATE FAILED — wide arm not run"; rc=5; }

run64 () {  # tag wide dump extra_args...
  local TAG=$1 WH=$2 DP=$3; shift 3
  echo "=== s9@64 [$TAG] wide=$WH dump=$DP extra='$*' ==="
  WANT_DUMP=$DP LEGOESM_MPAS_WIDE_HALO=$WH LEGOESM_MPAS_RAGGED_HALO=0 \
  JAX_ENABLE_X64=0 \
    srun --nodes=16 --ntasks=64 --ntasks-per-node=4 --gpus-per-node=4 \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$OUTDIR/rank_wrapper.sh" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices 64 --subdivision 9 --nlev 26 \
      --steps 12 --warmup 3 --lloyd 0 --partition-method sfc \
      --reorder-for 64 "$@" \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
}

run64 loop_a 0 0
run64 scan_b 0 0 --timed-scan
if [ "$rc" -ne 5 ]; then
  run64 wide_c 1 1
fi

echo "=== RECEIPTS ==="
set +e
if ls "$DUMP" | grep -q after_optimizations.txt; then
  "$PY" scripts/bench/census_xla_dump.py --dump-dir "$DUMP" \
    --out "$OUTDIR/census_wide.json" && cat "$OUTDIR/census_wide.json"
fi
"$PY" - "$OUTDIR" <<'EOF'
import json, math, sys, pathlib
out = pathlib.Path(sys.argv[1])
rc = 0

def row(tag):
    rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    assert len(rows) == 1, (tag, len(rows))
    r = rows[0]
    assert r["n_devices"] == 64 and r["platform"] == "gpu", tag
    assert r["halo_strategy_effective"] == "ppermute", tag
    return r

try:
    a, b = row("loop_a"), row("scan_b")
    assert not a["timed_scan"] and not a["wide_halo"]
    assert b["timed_scan"] and not b["wide_halo"]
    t_loop = a["steady_median_ms"]
    t_scan = b["scan_median_ms"]
    ratio = t_scan / t_loop
    print(f"loop_a {t_loop:.2f} ms | scan_b {t_scan:.2f} ms | "
          f"scan/loop {ratio:.3f}")
    if ratio <= 0.85:
        print("SCAN VERDICT: host per-step sync/dispatch REAL — fold "
              "steps into scan in production timing, re-baseline lanes")
    elif ratio >= 0.95:
        print("SCAN VERDICT: null — skew start-spread instrument next")
    else:
        print("SCAN VERDICT: intermediate")
except Exception as e:
    print(f"scan arms incomplete: {e}"); rc = 1

try:
    c = row("wide_c")
    assert c["wide_halo"] and c["halo_depth"] == 9
    print(f"wide_c {c['steady_median_ms']:.2f} ms")
    cen = json.load(open(out / "census_wide.json"))
    cp = (cen["counts"].get("collective-permute", 0)
          + cen["counts"].get("collective-permute-start", 0))
    print(f"CENSUS: wide CP = {cp} (offline rounds receipt: 11; "
          f"default executable: 33)")
    if cp != 11:
        print("CENSUS VERDICT: wide executable does NOT hold 11 CP — "
              "the 26846337 0.970 verdict is VOID"); rc = 4
    else:
        print("CENSUS VERDICT: 11 CP confirmed — 0.970 verdict stands")
except Exception as e:
    print(f"census incomplete: {e}"); rc = rc or 3

sys.exit(rc)
EOF
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc
