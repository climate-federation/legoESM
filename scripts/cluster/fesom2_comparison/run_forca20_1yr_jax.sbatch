#!/bin/bash
# One-year FESOM2-JAX run on FORCA20 (32 GPUs = 8 nodes, dist_32), self-chaining.
# Adapted from fesom_jax/scripts/runs/run_forca20_hindcast.sbatch (AWI): same chain
# logic (canary -> bounded per-job step budget -> archival restart -> resubmit),
# our account/paths, TOTAL = one model year at dt=180.
# Python: the AWI-certified env (jax 0.10.1, zarr 2.x) + PYTHONPATH to our fesom_jax
# clone (our legoESM venv carries zarr 3, which fesom_jax's output writer refuses).
#SBATCH --job-name=forca20_1yr
#SBATCH --account=bd1179_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=11:45:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoesm_pg/ocean_hires_runs/logs/forca20_1yr.%j.out
source /sw/etc/profile.levante 2>/dev/null
set -uo pipefail
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
FJ=/work/bd1083/b309178/diffESM/fesom_jax
LEGO=/work/bd1083/b309178/diffESM/legoesm_pg/wt_hires_ocean   # committed branch feat/hires-ocean-fesom-forca20 (stable paths for the chain)
export PYTHONPATH=$FJ
cd $FJ
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export JDIST=1 GPUS_PER_NODE=4 PYTHONUNBUFFERED=1
export FESOM_REUSE_EXE=1
export JAX_COMPILATION_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_pg/ocean_hires_runs/jax_cache/forca20_1yr
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1.0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

CONFIG=${CONFIG:-$LEGO/config/omip/fesom_forca20_1yr_jax.yaml}
MESH=/work/ab0995/a270088/fesom_jax_meshes/mesh_forca20
POOL=/work/ab0995/a270088/meshes/FORCA20
# 16 GPUs = 4 nodes x 4, fixed together with the SBATCH --nodes=4 header.  This is
# the partition AWI ran FORCA20 production on, and for the same reason: 4-node
# allocations start far sooner on this partition than 8-node ones (measured here,
# 15.5 h queued for 8 nodes with no start).  ~0.61 s/step instead of ~0.33.
PART=dist_16
NODES=4
OUT=/work/bd1083/b309178/diffESM/legoesm_pg/ocean_hires_runs/forca20_1yr
ARCHIVE=$OUT/restart_archive
SELF=$LEGO/scripts/cluster/fesom2_comparison/run_forca20_1yr_jax.sbatch
YEAR=1958
TOTAL=${TOTAL:-175200}      # one model year (1958, 365-day calendar) at dt=180
# Job boundaries MUST sit on calendar-month boundaries: the model's monthly/daily mean
# accumulators do not persist across jobs and the next job re-creates the period store, so a
# mid-month boundary would overwrite that month's mean with a partial one (codex review).
CANARY=14880                # first (cold) job = January (31 d x 480 steps/d): finiteness + throughput
PERJOB=${PERJOB:-43200}     # cap per resumed job; the actual STEPS is the largest month boundary <= CUR+PERJOB
                            # 43200 steps = 90 d ~ 7.3 h at the AWI-measured 0.61 s/step at dist_16,
                            # leaving >4 h headroom under the 11:45 limit; the chain takes ~6 jobs
GUARD_SECS=${GUARD_SECS:-1800}
CHUNK=48
mkdir -p $OUT $(dirname $SELF) /work/bd1083/b309178/diffESM/legoesm_pg/ocean_hires_runs/logs
# chain ownership: exactly one job of this chain may be alive (a second cold submission would race on
# .jobcount / the output stores)
# (the parent that resubmitted us may still be COMPLETING: it is excluded via PARENT_JOB, and the
# successor is submitted with --dependency=afterany:<parent> so the two never overlap)
Q=$(squeue -h -u $USER -n forca20_1yr -o %i) || { echo "ABORT: squeue failed — cannot prove single chain ownership"; exit 1; }
OTHERS=$(printf '%s\n' $Q | grep -v -e "^$SLURM_JOB_ID$" -e "^${PARENT_JOB:-none}$" || true)
if [ -n "$OTHERS" ]; then echo "ABORT: another forca20_1yr job is queued/running ($OTHERS) — one chain job at a time"; exit 1; fi
# model code pin: clean tree + recorded HEAD (an uncommitted edit would change the executable at a restart seam)
DIRTY=$(git -C $FJ status --porcelain --untracked-files=all) || { echo "ABORT: git status failed"; exit 1; }
[ -z "$DIRTY" ] || { echo "ABORT: fesom_jax clone not clean (incl. untracked; it is on PYTHONPATH):"; echo "$DIRTY" | head; exit 1; }
SHA_NOW=$(git -C $FJ rev-parse HEAD) && [ -n "$SHA_NOW" ] || { echo "ABORT: cannot read fesom_jax HEAD"; exit 1; }
CFG_SHA_NOW=$(sha256sum "$CONFIG" | cut -d' ' -f1) && [ -n "$CFG_SHA_NOW" ] || { echo "ABORT: cannot hash $CONFIG"; exit 1; }
# the yaml's forcing year must match the --year we pass (run_from_config.py reads --year, not the yaml)
YAML_YEAR=$($PY -c "import yaml,sys;print(yaml.safe_load(open('$CONFIG'))['forcing']['start_year'])")
[ "$YAML_YEAR" = "$YEAR" ] || { echo "ABORT: yaml forcing.start_year=$YAML_YEAR != YEAR=$YEAR"; exit 1; }

if [ -e $OUT/.jobcount ]; then CNT=$(cat $OUT/.jobcount) || { echo "ABORT: .jobcount exists but is unreadable"; exit 1; }; else CNT=0; fi
case "$CNT" in ''|*[!0-9]*) echo "ABORT: .jobcount corrupt ($CNT)"; exit 1;; esac
CNT=$((CNT+1)); echo $CNT > $OUT/.jobcount
if [ "$CNT" -gt 16 ]; then echo "ABORT: jobcount $CNT > 12 (runaway guard)"; exit 1; fi
RESTART_DIR=""
if [ -f "$ARCHIVE/restart.latest" ]; then
  LTAG=$(cat "$ARCHIVE/restart.latest"); [ -n "$LTAG" ] && RESTART_DIR="$ARCHIVE/$LTAG"
fi
CUR=0
if [ -n "$RESTART_DIR" ]; then
  CUR=$( $PY -c "import json;print(int(json.load(open('$RESTART_DIR/.zattrs')).get('step',0)))" 2>/dev/null || echo FAIL )
  case "$CUR" in ''|*[!0-9]*)
    echo "ABORT: restart pointer -> $RESTART_DIR but its step is unreadable — refusing the silent cold-start fallback"; exit 1;;
  esac
fi
echo "=== $(date) forca20_1yr job#$CNT: CUR=$CUR / TOTAL=$TOTAL (restart=${RESTART_DIR:-none}) nodes=$SLURM_JOB_NODELIST ==="
if [ "$CUR" -ge "$TOTAL" ]; then echo "FORCA20_1YR_DONE ($CUR steps = 1 model year)"; exit 0; fi
if [ "$CUR" -eq 0 ]; then
  if [ -n "$(ls -A "$ARCHIVE" 2>/dev/null)" ] && [ "${FORCE_COLD:-0}" != "1" ]; then
    echo "ABORT: cold start requested but $ARCHIVE is non-empty — refusing to overwrite an existing chain (FORCE_COLD=1 overrides)"; exit 1
  fi
  STEPS=$CANARY; RIN=""; CNT=1; echo $CNT > $OUT/.jobcount
  echo "$SHA_NOW" > $OUT/.fesom_sha; echo "$CFG_SHA_NOW" > $OUT/.config_sha
else
  SHA_CHAIN=$(cat $OUT/.fesom_sha 2>/dev/null || echo none)
  [ "$SHA_NOW" = "$SHA_CHAIN" ] || { echo "ABORT: fesom_jax HEAD $SHA_NOW != chain SHA $SHA_CHAIN"; exit 1; }
  CFG_CHAIN=$(cat $OUT/.config_sha 2>/dev/null || echo none)
  [ "$CFG_SHA_NOW" = "$CFG_CHAIN" ] || { echo "ABORT: $CONFIG changed since cold start (sha $CFG_SHA_NOW != $CFG_CHAIN)"; exit 1; }
  STEPS=$($PY - "$CUR" "$PERJOB" "$TOTAL" <<'PYEOF'
import sys
cur, perjob, total = map(int, sys.argv[1:])
spd = 480  # steps per day at dt=180 (365-day calendar, 1958)
bounds = [d * spd for d in (31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)]
ok = [b for b in bounds if cur < b <= min(cur + perjob, total)]
if not ok:
    sys.exit(f"no month boundary in ({cur}, {min(cur + perjob, total)}] — PERJOB too small")
print(ok[-1] - cur)
PYEOF
  ) || { echo "ABORT: $STEPS"; exit 1; }
  RIN="--restart-in $RESTART_DIR"
fi
echo "running $STEPS steps from $CUR (${RIN:-cold})"
LOGFILE=/work/bd1083/b309178/diffESM/legoesm_pg/ocean_hires_runs/logs/forca20_1yr.${SLURM_JOB_ID}.out
do_resubmit() {
  cd $LEGO && sbatch --dependency=afterany:$SLURM_JOB_ID \
    --export=ALL,CONFIG="$CONFIG",PERJOB="$PERJOB",TOTAL="$TOTAL",GUARD_SECS="$GUARD_SECS",PARENT_JOB="$SLURM_JOB_ID" $SELF
}
# --cpus-per-task explicitly: srun refuses to reconcile SLURM_CPUS_PER_TASK with
# SLURM_TRES_PER_TASK when the two disagree, and it does disagree here (127 vs 64),
# which killed job 27339578 in seconds after 27 h of queueing. The CLI value wins.
srun --cpu-bind=none --cpus-per-task=64 "$PY" scripts/run_from_config.py "$CONFIG" --year $YEAR \
    --mesh-dir "$MESH" --dist-dir "$POOL" --ic-dir "$MESH" --partition "$PART" \
    --steps "$STEPS" --chunk-steps "$CHUNK" --ragged --local-forcing \
    --diagnostics --progress \
    --monthly-out "$OUT/monthly" --monthly-start-step 0 \
    --daily-out "$OUT/daily" --daily-start-step 0 \
    --restart-out "$OUT/restart" \
    --restart-archive-out "$ARCHIVE" --restart-archive-period month --restart-archive-length 1 \
    $RIN &
SRUN_PID=$!
t0=$SECONDS; HUNG=0
while kill -0 $SRUN_PID 2>/dev/null; do
  grep -q "chunk 1/" "$LOGFILE" 2>/dev/null && break
  [ $((SECONDS-t0)) -gt $GUARD_SECS ] && { HUNG=1; break; }
  sleep 30
done
if [ "$HUNG" -eq 1 ]; then
  HC=$(( $(cat $OUT/.hangcount 2>/dev/null || echo 0) + 1 )); echo $HC > $OUT/.hangcount
  echo "GUARD: COMPILE-HANG — no 'chunk 1/' in ${GUARD_SECS}s (consecutive hang #$HC); killing srun"
  kill $SRUN_PID 2>/dev/null; sleep 5; kill -9 $SRUN_PID 2>/dev/null; wait $SRUN_PID 2>/dev/null
  if [ "$HC" -ge 3 ]; then echo "FORCA20_1YR_STOP (3 consecutive compile-hangs)"; exit 1; fi
  echo "GUARD: resubmitting from $CUR/$TOTAL (restart intact)"; do_resubmit || { echo "FORCA20_1YR_STOP (resubmit failed)"; exit 1; }; exit 0
fi
echo 0 > $OUT/.hangcount
wait $SRUN_PID; R=$?
echo "forca20_1yr job#$CNT exit=$R"
if [ "$R" -ne 0 ]; then
  # the model publishes its end-of-segment archival restart + restart.latest BEFORE the finite check
  # runs, so a NON-FINITE segment leaves a poisoned pointer: park it so a manual retry cannot resume it.
  # A crash / node loss / time-limit kill leaves a VALID last-month pointer: keep it (manual resubmit resumes).
  if grep -q "NG5_R0_NONFINITE" "$LOGFILE" 2>/dev/null && [ -f "$ARCHIVE/restart.latest" ]; then
    mv -f "$ARCHIVE/restart.latest" "$ARCHIVE/restart.latest.FAILED.$SLURM_JOB_ID"
    echo "non-finite state: restart.latest parked as restart.latest.FAILED.$SLURM_JOB_ID (earlier monthly restarts remain in $ARCHIVE)"
  fi
  echo "FORCA20_1YR_STOP (exit=$R — inspect $OUT; no auto-continue; a manual resubmit resumes from restart.latest if present)"; exit $R
fi
NEWCUR=$((CUR+STEPS))
if [ "$NEWCUR" -lt "$TOTAL" ]; then echo "resubmitting from $NEWCUR/$TOTAL"; do_resubmit || { echo "FORCA20_1YR_STOP (resubmit failed)"; exit 1; }
else echo "FORCA20_1YR_DONE ($NEWCUR steps = 1 model year)"; fi
