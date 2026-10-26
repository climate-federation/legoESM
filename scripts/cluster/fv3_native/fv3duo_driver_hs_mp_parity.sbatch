#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=duo_hs_mp
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/duo_hs_mp_parity_%j.log
# Driver-level parity for Held-Suarez on the multi-process SPMD duo lane:
#   arm A: ONE process   -> ModelDriver's certified NumPy HS path
#   arm B: SIX processes -> the face-stacked JAX twin (the only path that
#          can run there), XLA:CPU collectives=mpi, jax.distributed via SLURM
#          (needs >1 hostname, hence --nodes=2)
# Same deck, same steps; the final snapshots are diffed field by field.
# PASS: every field within REL_PEAK_MAX (peak-relative), 1e-11 = the twin's
# per-step pin to the authority in test_fv3_physics_coupling (user call
# 2026-09-21; measured 2.7e-13 over 144 steps, jobs 9902213/9903984, so a
# day of steps does not compound it).  Any NaN, missing snapshot, or
# non-zero exit REFUSES.
set -uo pipefail
WT=${WT:-/burg-archive/glab/users/pg2328/legoESM_codex_m8}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
WRAPLIB=/burg-archive/glab/users/pg2328/mpiwrapper/install/lib64/libmpiwrapper.so
cd "$WT" || exit 1
export PYTHONPATH="$WT/packages/core:$WT/packages/atmosphere:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ocean:$WT/packages/ml:$WT/packages/tools:$WT/src"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
export LEGOESM_JIT_CACHE_DIR=""
export XLA_FLAGS="--xla_cpu_max_isa=AVX"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
source /etc/profile.d/modules.sh
module load openmpi/gcc/64/4.1.7a1
for v in $(env | sed -n 's/^\(MPITRAMPOLINE_[A-Z_]*\)=.*/\1/p'); do unset "$v"; done
# gloo, NOT mpi, for the DRIVER: run_amip's launcher (maybe_init_jax_distributed)
# initialises MPI through mpi4py, and the mpi collectives then call MPI_Init a
# second time -- "Open MPI has detected that this process has attempted to
# initialize MPI more than once", job 9902192.  The m6 gate never imports mpi4py,
# which is why the mpi venue works there.  gloo is the correctness venue here;
# putting the driver on the mpi venue needs a no-mpi4py boot (open).
export JAX_CPU_COLLECTIVES_IMPLEMENTATION=${COLL:-gloo}
[ "$JAX_CPU_COLLECTIVES_IMPLEMENTATION" = mpi ] && export MPITRAMPOLINE_LIB="$WRAPLIB"
LAUNCH="$WT/scripts/cluster/fv3_native/bind_rank.sh"

N=${N:-12}; NLEV=${NLEV:-5}; DAYS=${DAYS:-1}; DT=${DT:-600}
# KT unset: arm B is the 6-rank FACE layout.  KT=k (with PAD): arm B is the
# 6*k*k-rank WINDOW layout (--fv3-duo-windows); submit with matching
# --ntasks, e.g. --nodes=4 --ntasks=24 --ntasks-per-node=6 --export=ALL,KT=2,PAD=5,N=24
KT=${KT:-}; PAD=${PAD:-}
WINFLAGS=""; [ -n "$KT" ] && WINFLAGS="--fv3-duo-windows $KT --fv3-duo-window-pad $PAD"
REL_PEAK_MAX=${REL_PEAK_MAX:-1e-11}
OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/duo_hs_mp_parity_$SLURM_JOB_ID
mkdir -p "$OUT"
echo "sha $(git rev-parse --short HEAD) dirty $(git status --porcelain | wc -l)  nodes $SLURM_JOB_NODELIST  C$N nlev=$NLEV days=$DAYS dt=$DT  arm B: ${SLURM_NTASKS} ranks ${WINFLAGS:-face layout}"

COMMON=(scripts/run/run_amip.py --dataset analytical --grid-type cubed_sphere
        --discretization fv3_duo --resolution "$N" --nlev "$NLEV" --precision fp64
        --radiation none --convection none --microphysics none --turbulence none
        --gravity-wave-drag none --allow-disabled-physics --held-suarez-forcing
        --days "$DAYS" --dt "$DT" --diag-days "$DAYS")

echo "===== arm A: 1 process (NumPy HS) ====="
srun -N 1 -n 1 --mpi=pmix "$PY" "${COMMON[@]}" --output "$OUT/serial" > "$OUT/serial.log" 2>&1
RC_A=$?
echo "arm A rc=$RC_A  $(grep -vE "cuda|jax_plugins|xla_bridge|versions_helpers|for d in range|_check_cuda|plugin_module" "$OUT/serial.log" | grep -cE 'Error') non-CUDA-noise error lines"

echo "===== arm B: $SLURM_NTASKS processes (JAX HS twin${KT:+, windows kt=$KT pad=$PAD}) ====="
srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_NTASKS" --mpi=pmix --kill-on-bad-exit=0 $LAUNCH "$PY" "${COMMON[@]}" \
  --distributed --distributed-mode spmd $WINFLAGS --output "$OUT/mp6" > "$OUT/mp6.log" 2>&1
RC_B=$?
echo "arm B rc=$RC_B  $(grep -vE "cuda|jax_plugins|xla_bridge|versions_helpers|for d in range|_check_cuda|plugin_module" "$OUT/mp6.log" | grep -cE 'Error') non-CUDA-noise error lines"
grep -m1 -E "process_count|multi-process|_fv3_duo_apply_held_suarez_jax|NotImplementedError" "$OUT/mp6.log" || true

if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then
  echo "REFUSED: an arm exited non-zero (A=$RC_A B=$RC_B); tails:"
  tail -n 15 "$OUT/serial.log"; tail -n 25 "$OUT/mp6.log"; exit 2
fi

"$PY" - "$OUT" "$REL_PEAK_MAX" <<'PYEOF'
import sys, glob, numpy as np
out, tol = sys.argv[1], float(sys.argv[2])
def last(d):
    fs = sorted(glob.glob(f"{d}/fv3duo_snapshot_step_*.npz"))
    if not fs:
        print(f"REFUSED: no snapshot under {d}"); sys.exit(2)
    return fs[-1], np.load(fs[-1])
fa, a = last(f"{out}/serial"); fb, b = last(f"{out}/mp6")
print("A:", fa.split('/')[-1], " B:", fb.split('/')[-1])
if fa.split('/')[-1] != fb.split('/')[-1]:
    print("REFUSED: the two arms wrote different final steps"); sys.exit(2)
rc, worst = 0, 0.0
for nm in sorted(set(a.files) & set(b.files)):
    x, y = a[nm], b[nm]
    if x.shape != y.shape or not np.issubdtype(x.dtype, np.floating):
        continue
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        print(f"  {nm:8s} REFUSED: non-finite"); rc = 2; continue
    peak = float(np.abs(x).max()); d = float(np.abs(x - y).max())
    rel = d / peak if peak > 0 else (0.0 if d == 0 else float('inf'))
    worst = max(worst, rel)
    tag = "ok" if rel <= tol else "DIFF"
    if rel > tol: rc = 1
    print(f"  {nm:8s} {tag:4s} max|d| {d:.3e}  peak {peak:.3e}  rel-to-peak {rel:.3e}")
print(f"VERDICT: {'PASS' if rc == 0 else 'FAIL'} worst rel-to-peak {worst:.3e} (gate {tol:g})")
sys.exit(rc)
PYEOF
