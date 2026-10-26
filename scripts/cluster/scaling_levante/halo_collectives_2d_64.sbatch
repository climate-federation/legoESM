#!/bin/bash -l
#SBATCH --job-name=halo_coll_2d
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=halo_coll_2d.%j.log
# The 2-D (single-level) halo fill that the ocean-MPAS barotropic PCG issues
# once per iteration (20 iterations + 8 other exchanges per step at 128 GPUs,
# each = 12-14 SEQUENTIAL ppermute rounds).  The 3-D micro-bench (job 26818265,
# 700x130 f32 per pair) measured ppermute 1008 us vs ragged 478 / uniform 499 at
# nd=16, i.e. ~92 us PER ROUND at 364 KB; a 2-D field is ~200x1 f32 = 800 B per
# pair, so the round is pure latency and one grouped collective may win by more.
# Falsifiability, written BEFORE submit:
#   numbers  : per_fill_us for ppermute(K rounds) / ragged / uniform at width 1,
#              nd=32 and nd=128 (degree 12, symmetric alias-free offsets).
#   CONFIRMS : best single collective <= 0.5x ppermute at nd=128 -> routing the
#              PCG's 2-D exchanges through it saves >= 0.5 x 28 x per_fill per
#              ocean step (quantified from the rows), worth wiring.
#   REFUTES  : best single collective >= 0.8x ppermute -> the 2-D transport is
#              not the lever; the iteration count is.
# Offline impossible: XLA:CPU has no ragged-all-to-all thunk and the round
# latency is a fabric property.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
# Accept a submit directory that is either the repo root or the launcher
# directory itself.  Slurm copies the batch script into its spool dir, so
# BASH_SOURCE cannot be used to find the repo (job 27591170 died in 8 s on an
# unbound REPO after being submitted from the launcher directory).
ENV_SH="${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
[[ -f "$ENV_SH" ]] || ENV_SH="${SUBMIT_DIR}/_env.sh"
[[ -f "$ENV_SH" ]] || { echo "cannot find _env.sh from $SUBMIT_DIR" >&2; exit 2; }
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_REPO="${LEGOESM_REPO:-$(cd "$(dirname "$ENV_SH")/../../.." && pwd)}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "$ENV_SH"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/halo_c2d64_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR="$OUTDIR/jitcache"
OFFS="1,-1,2,-2,5,-5,7,-7,6,-6,11,-11"
ROWS="${ROWS:-200}"
rc=0
run_leg () {  # nd nodes width
  local ND=$1 NODES=$2 W=$3
  srun --nodes="$NODES" --ntasks="$ND" --ntasks-per-node=4 \
      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_halo_collectives.py \
      --multicontroller --n-devices "$ND" --offsets "$OFFS" \
      --rows "$ROWS" --width "$W" --strategies ppermute,ragged,uniform \
      --out "$OUTDIR/nd${ND}_w${W}.jsonl" || { echo "nd$ND w$W FAILED"; rc=1; }
}
run_leg 32   8 1
run_leg 64 16 1
run_leg 64 16 130   # 3-D control
"$PY" - "$OUTDIR" "$OFFS" "$ROWS" <<'EOF2' || rc=1
import json, sys, pathlib, math
out, offs, rows = pathlib.Path(sys.argv[1]), [int(t) for t in sys.argv[2].split(",")], int(sys.argv[3])
for nd, w in ((32, 1), (64, 1), (64, 130)):
    rs = {json.loads(l)["strategy"]: json.loads(l) for l in open(out / f"nd{nd}_w{w}.jsonl")}
    assert set(rs) == {"ppermute", "ragged", "uniform"}, sorted(rs)
    for r in rs.values():
        assert r["n_devices"] == nd and r["offsets"] == offs and r["rows"] == rows \
            and r["width"] == w and r["verified_exact"] is True \
            and math.isfinite(r["per_fill_us"]) and r["per_fill_us"] > 0 \
            and r["env_overlap"] == "0", r
    pp = rs["ppermute"]["per_fill_us"]
    best = min(rs["ragged"]["per_fill_us"], rs["uniform"]["per_fill_us"])
    print(f"nd{nd} w{w}: ppermute({rs['ppermute']['n_collectives_per_fill']}r) {pp:9.1f} us | "
          f"ragged {rs['ragged']['per_fill_us']:9.1f} | uniform {rs['uniform']['per_fill_us']:9.1f} | "
          f"best/ppermute {best/pp:.3f}")
EOF2
echo "DONE rc=$rc"; exit $rc
