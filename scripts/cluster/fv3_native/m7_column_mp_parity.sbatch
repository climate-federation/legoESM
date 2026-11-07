#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=m7_col_mp
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m7/col_mp_parity_%j.log
# M7 gate G2: the FV3 duo COLUMN lane (route A, _run_mpas) under multi-
# process SPMD.  Three arms on the same deck (C24 km5, Kessler, 1 day):
#   A  one process, ONE device                      (the certified column lane)
#   A6 one process, 6 host devices (face layout)     (G1 at the driver level)
#   B  SLURM_NTASKS processes, one device each: face layout (6 ranks) or the
#      window layout (KT=k, PAD=p, 6*k*k ranks; e.g. --nodes=3 --ntasks=24
#      --ntasks-per-node=8 --export=ALL,KT=2,PAD=4)
# Claims: B == A6 BITWISE (same partitioned program, G2); A6 and B == A at the
# closed lane's parity gate 1e-11 of peak (the sharded step is not bitwise with
# one device, MEASURED 2026-10-04).  Compared on the day-1 CHECKPOINT (the six
# host faces the column lane persists), never on printed summaries.
set -uo pipefail
WT=${WT:-/burg-archive/glab/users/pg2328/legoESM_codex_m8}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$WT" || exit 1
export PYTHONPATH="$WT/packages/core:$WT/packages/atmosphere:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ocean:$WT/packages/ml:$WT/packages/tools:$WT/src"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
export LEGOESM_JIT_CACHE_DIR=""
export XLA_FLAGS="--xla_cpu_max_isa=AVX"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
source /etc/profile.d/modules.sh
module load openmpi/gcc/64/4.1.7a1
for v in $(env | sed -n 's/^\(MPITRAMPOLINE_[A-Z_]*\)=.*/\1/p'); do unset "$v"; done
export JAX_CPU_COLLECTIVES_IMPLEMENTATION=${COLL:-gloo}
LAUNCH="$WT/scripts/cluster/fv3_native/bind_rank.sh"
N=${N:-24}; NLEV=${NLEV:-5}; DAYS=${DAYS:-1}; DT=${DT:-600}
KT=${KT:-}; PAD=${PAD:-}
WINFLAGS=""; [ -n "$KT" ] && WINFLAGS="--fv3-duo-windows $KT --fv3-duo-window-pad $PAD"
REL_PEAK_MAX=${REL_PEAK_MAX:-1e-11}
OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/m7/col_mp_parity_$SLURM_JOB_ID
mkdir -p "$OUT"
echo "sha $(git rev-parse --short HEAD) dirty $(git status --porcelain | grep -v '^??' | wc -l)  nodes $SLURM_JOB_NODELIST  C$N nlev=$NLEV days=$DAYS dt=$DT  arm B: ${SLURM_NTASKS} ranks ${WINFLAGS:-face layout}"
COMMON=(scripts/run/run_amip.py --dataset analytical --grid-type cubed_sphere
        --discretization fv3_duo --fv3-duo-column-lane --resolution "$N" --nlev "$NLEV"
        --precision fp64 --radiation none --convection none --microphysics kessler
        --turbulence none --gravity-wave-drag none --allow-disabled-physics
        --days "$DAYS" --dt "$DT" --diag-days "$DAYS" --checkpoint-days "$DAYS")
echo "===== arm A: 1 process, 1 device ====="
srun -N 1 -n 1 --mpi=pmix "$PY" "${COMMON[@]}" --output "$OUT/one" > "$OUT/one.log" 2>&1
RC_A=$?; echo "arm A rc=$RC_A"
echo "===== arm A6: 1 process, ${SLURM_NTASKS} host devices ${WINFLAGS:-(face layout)} ====="
XLA_FLAGS="--xla_cpu_max_isa=AVX --xla_force_host_platform_device_count=${SLURM_NTASKS}" \
  srun -N 1 -n 1 --mpi=pmix "$PY" "${COMMON[@]}" $WINFLAGS --output "$OUT/one_sharded" > "$OUT/one_sharded.log" 2>&1
RC_A6=$?; echo "arm A6 rc=$RC_A6"
echo "===== arms C / C6: the CLOSED lane (no column model), same deck, 1 device / ${SLURM_NTASKS} host devices ====="
CLOSED=(); for a in "${COMMON[@]}"; do [ "$a" = "--fv3-duo-column-lane" ] || CLOSED+=("$a"); done
srun -N 1 -n 1 --mpi=pmix "$PY" "${CLOSED[@]}" --output "$OUT/closed_one" > "$OUT/closed_one.log" 2>&1
RC_C=$?; echo "arm C rc=$RC_C"
XLA_FLAGS="--xla_cpu_max_isa=AVX --xla_force_host_platform_device_count=${SLURM_NTASKS}" \
  srun -N 1 -n 1 --mpi=pmix "$PY" "${CLOSED[@]}" $WINFLAGS --output "$OUT/closed_sharded" > "$OUT/closed_sharded.log" 2>&1
RC_C6=$?; echo "arm C6 rc=$RC_C6"
echo "===== arm B: $SLURM_NTASKS processes ${WINFLAGS:-(face layout)} ====="
srun -N "$SLURM_JOB_NUM_NODES" -n "$SLURM_NTASKS" --mpi=pmix --kill-on-bad-exit=0 $LAUNCH "$PY" "${COMMON[@]}" \
  --distributed --distributed-mode spmd $WINFLAGS --output "$OUT/mp" > "$OUT/mp.log" 2>&1
RC_B=$?; echo "arm B rc=$RC_B"
for a in one one_sharded closed_one closed_sharded mp; do
  echo "-- $a: $(grep -m1 -E 'fv3_duo layout' "$OUT/$a.log" | sed 's/^ *//' | cut -c1-120); $(grep -E 'MPAS run (COMPLETED|BLOWUP)|Traceback|Error' "$OUT/$a.log" | grep -v 'cuda\|xla_bridge\|CUDA' | tail -2 | cut -c1-160)"
done
if [ "$RC_A" -ne 0 ] || [ "$RC_A6" -ne 0 ] || [ "$RC_B" -ne 0 ] || [ "$RC_C" -ne 0 ] || [ "$RC_C6" -ne 0 ]; then
  echo "REFUSED: an arm exited non-zero (A=$RC_A A6=$RC_A6 B=$RC_B C=$RC_C C6=$RC_C6); tails:"
  tail -n 12 "$OUT/one.log"; tail -n 12 "$OUT/one_sharded.log"; tail -n 30 "$OUT/mp.log"; exit 2
fi
"$PY" - "$OUT" "$REL_PEAK_MAX" <<'PYEOF'
import sys, glob, numpy as np
out, tol = sys.argv[1], float(sys.argv[2])
def last(d):
    fs = sorted(glob.glob(f"{d}/checkpoint_day_*.npz"))
    if not fs:
        print(f"REFUSED: no checkpoint under {d}"); sys.exit(2)
    return fs[-1].split("/")[-1], np.load(fs[-1])
na, a = last(f"{out}/one"); n6, a6 = last(f"{out}/one_sharded"); nb, b = last(f"{out}/mp")
print("files:", na, n6, nb)
if not (na == n6 == nb):
    print("REFUSED: the arms wrote different final checkpoints"); sys.exit(2)
keys = [k for k in a.files if k.startswith(("state_", "press_", "q_"))]
# tracers are gated MASS-weighted (delp*q) at 1e-9 of peak: a limiter-branch
# flip of the sharded dynamics moves a few cells by ~1e-10 of peak once
# (MEASURED, test_fv3_duo_column_spmd.REL_TRACER); winds/T/delp/ps at the gate
TOL_TRACER = 1e-9
def cmp(x, y, label, bitwise):
    rc, worst = 0, 0.0
    for nm in sorted(keys):
        u, v = np.asarray(x[nm]), np.asarray(y[nm])
        if u.shape != v.shape or not np.issubdtype(u.dtype, np.floating):
            continue
        if not (np.isfinite(u).all() and np.isfinite(v).all()):
            print(f"  {label} {nm:12s} REFUSED: non-finite"); rc = 2; continue
        if bitwise:                       # RAW leaves, never a product
            same = u.tobytes() == v.tobytes()
            if not same:
                rc = 1; print(f"  {label} {nm:12s} NOT BITWISE max|d| {np.abs(u - v).max():.3e}")
            continue
        tol_nm = tol
        if nm.startswith("q_") and u.ndim == 4:
            u, v = u * np.asarray(x["state_delp"]), v * np.asarray(y["state_delp"])
            tol_nm = max(tol, TOL_TRACER)
        peak = float(np.abs(v).max()); d = float(np.abs(u - v).max())
        rel = d / peak if peak > 0 else (0.0 if d == 0 else float("inf"))
        worst = max(worst, rel)
        if rel > tol_nm: rc = 1
        print(f"  {label} {nm:12s} {'ok  ' if rel <= tol_nm else 'DIFF'} max|d| {d:.3e} peak {peak:.3e} rel {rel:.3e} (gate {tol_nm:g})")
    return rc, worst
rc1, _ = cmp(b, a6, "G2 mp==1proc-sharded", True)
print(f"G2 (multi-process vs single-process, same device count): {'BITWISE' if rc1 == 0 else 'NOT bitwise'}")
# the CLOSED lane's own sharded-vs-one-device difference on the SAME deck is
# the reference for G1: the column seam must add nothing to that class
def last_snap(d):
    fs = sorted(glob.glob(f"{d}/fv3duo_snapshot_step_*.npz"))
    if not fs:
        print(f"REFUSED: no snapshot under {d}"); sys.exit(2)
    return np.load(fs[-1])
c, c6 = last_snap(f"{out}/closed_one"), last_snap(f"{out}/closed_sharded")
w_closed = 0.0
for nm in ("u", "v", "pt", "delp"):
    x, y = np.asarray(c6[nm]), np.asarray(c[nm])
    rel = float(np.abs(x - y).max() / max(float(np.abs(y).max()), 1e-300))
    w_closed = max(w_closed, rel)
    print(f"  CLOSED lane sharded vs one device {nm:6s} rel {rel:.3e}")
if not (0.0 < w_closed < 1e-9):
    print(f"REFUSED: the closed lane's own sharded-vs-one-device difference "
          f"{w_closed:.3e} is outside (0, 1e-9): the reference is degenerate or "
          f"the shared layout regressed; it may not set the gate"); sys.exit(2)
tol_g1 = max(tol, 10.0 * w_closed)
print(f"G1 gate = max({tol:g}, 10 x closed lane's worst {w_closed:.3e}) = {tol_g1:.3e}")
tol = tol_g1
rc2, w2 = cmp(a6, a, "G1 sharded vs one device", False)
rc3, w3 = cmp(b, a, "mp vs one device", False)
print(f"VERDICT: G2 {'PASS' if rc1 == 0 else 'FAIL'}; G1 {'PASS' if rc2 == 0 else 'FAIL'} worst {w2:.3e}; mp-vs-one {'PASS' if rc3 == 0 else 'FAIL'} worst {w3:.3e} (gate {tol:.3e}, closed-lane worst {w_closed:.3e})")
sys.exit(max(rc1, rc2, rc3))
PYEOF
