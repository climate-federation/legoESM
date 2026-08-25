#!/bin/bash -l
#SBATCH --job-name=ovl_hlo
#SBATCH --account=bd1179_gpu
#SBATCH --partition=gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=00:28:00
#SBATCH --output=ovl_hlo.%j.log
# ===========================================================================
# DID XLA ACTUALLY OVERLAP THE HALO WITH THE INTERIOR PASS?
#
# Compile the interior/rim overlap step (LEGOESM_MPAS_HALO_OVERLAP=1) with the
# latency-hiding scheduler ON (LEGOESM_XLA_OVERLAP=1), dump the optimized HLO,
# and check for the async collective form. GLM's gate, decisive and free:
#   - PROOF of overlap: __ncclCollectivePermuteStart / ...Done PAIRS exist, and
#     interior tendency FUSIONS sit between a Start and its matching Done.
#   - NO overlap: only synchronous __ncclCollectivePermute (or __ncclSendRecv)
#     appears -> the scheduler did not hide anything, do NOT spend a big
#     allocation timing it.
# Two GPUs, subdivision 6 (~20k cells/device -> ppermute), dump on rank 0.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_overlap_hlo.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
# The overlap DEPENDS on the latency-hiding scheduler; without it the collective
# and the interior fusions serialise on one stream and the whole lever is inert.
export LEGOESM_XLA_OVERLAP=1
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/ovl_hlo_j${SLURM_JOB_ID}"; DUMP="$OUTDIR/dump0"
mkdir -p "$DUMP"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR=""   # a cached compile does not re-dump
export LEGOESM_MPAS_HALO_OVERLAP=1

cat > "$OUTDIR/wrap.sh" <<'WRAP'
#!/usr/bin/env bash
set -uo pipefail
if [ "${SLURM_PROCID:-1}" = "0" ]; then
  export XLA_FLAGS="--xla_dump_to=${DUMP_DIR}${XLA_FLAGS:+ $XLA_FLAGS}"
fi
exec "$PY_BIN" "$@"
WRAP
chmod +x "$OUTDIR/wrap.sh"
export DUMP_DIR="$DUMP" PY_BIN="$PY"

JAX_ENABLE_X64=0 timeout 900 srun --ntasks=2 --gpus-per-node=2 --gpu-bind=none \
  --kill-on-bad-exit=1 "$OUTDIR/wrap.sh" scripts/bench/bench_mpas_spmd_scaling.py \
    --n-devices 2 --subdivision 6 --nlev 26 --steps 4 --warmup 1 \
    --halo-strategy ppermute --partition-method metis --multicontroller \
    --out "$OUTDIR/s6np2.jsonl"
RC=$?; echo "[hlo] srun exit=$RC"

echo "=== async collective census in the step module ==="
"$PY" - "$DUMP" <<'PYEOF'
import glob, os, re, sys
d = sys.argv[1]
mods = glob.glob(os.path.join(d, "*after_optimizations.txt"))
if not mods:
    print("NO optimized modules dumped"); sys.exit(4)
# Scan EVERY module; the collectives live in whichever one holds the step,
# not necessarily the largest. Pick the module with the most permutes.
def _score(pp):
    t = open(pp).read()
    return (t.count("collective-permute") + t.count("__ncclCollectivePermute")
            + t.count("all-gather") + t.count("__ncclAllGather"))
mod = max(mods, key=_score)
print(f"scanned {len(mods)} modules; step module:", os.path.basename(mod))
text = open(mod).read()
starts = len(re.findall(r"collective-permute-start", text))
dones = len(re.findall(r"collective-permute-done", text))
sync = len(re.findall(r"(?<!-)collective-permute\(", text))
ncstart = len(re.findall(r"__ncclCollectivePermuteStart", text))
ncdone = len(re.findall(r"__ncclCollectivePermuteDone", text))
print(f"collective-permute-start: {starts}   -done: {dones}")
print(f"__ncclCollectivePermuteStart: {ncstart}   -Done: {ncdone}")
print(f"synchronous collective-permute(: {sync}")
# Between a start and its done, is there a fusion (interior compute)?
lines = text.splitlines()
between_fusion = 0
depth = 0
for ln in lines:
    if "collective-permute-start" in ln:
        depth += 1
    elif "collective-permute-done" in ln:
        depth = max(0, depth - 1)
    elif depth > 0 and "fusion(" in ln:
        between_fusion += 1
print(f"fusion lines textually between a start and a done: {between_fusion}")
if (starts + ncstart) > 0 and between_fusion > 0:
    print("VERDICT: OVERLAP FIRED — async collectives with compute between "
          "start and done. Worth a timed A/B.")
elif (starts + ncstart) > 0:
    print("VERDICT: async collectives exist but no compute between them in "
          "text order — inspect operands; overlap may be marginal.")
else:
    print("VERDICT: NO async collectives — latency-hiding did not convert the "
          "ppermute; overlap is inert. Do NOT time it until this is fixed.")
    sys.exit(6)
PYEOF
echo "HLO_DONE rc=$RC"
