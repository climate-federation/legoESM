#!/bin/bash
#SBATCH --account=CUSTOMIZE_ACCOUNT
#SBATCH --partition=CUSTOMIZE_PARTITION
#SBATCH --job-name=compare_reanalysis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=results/compare_reanalysis/campaign_%j.out

# ---------------------------------------------------------------------------
# LES-informed clubb_lite correction campaign vs ERA5 (docs/COMPARE_REANALYSIS.md):
#   run AMIP/CMIP -> time-mean -> compare to ERA5 -> rank worst columns ->
#   spin off a column LES -> diagnose the closure coefficient -> RE-RUN, keeping
#   ONLY bias-improving rounds (the monotonic acceptance gate). Writes the
#   corrected clubb_lite config to $OUT (deploy with deploy_correction).
#
# RESUMABLE + SELF-REQUEUING (the point of running it under SLURM): the per-round
# checkpoint is written ATOMICALLY (temp+fsync+os.replace -- iter 185), and on the
# pre-timeout SIGUSR1 (--signal=B:USR1@120 above) this script auto-requeues ITSELF
# (scontrol requeue) so a multi-day run survives the wall-clock limit with NO manual
# re-submission -- the requeued job re-runs from the top and --resumes from the last
# completed round (a deploy-grade grid fingerprint in the checkpoint refuses a
# wrong-grid resume -- iter 186). A campaign that CONVERGES before the limit exits
# normally => no requeue (no infinite loop). stdout is line-buffered for real-time
# progress in this .out log (iter 191).
#
# CUSTOMIZE: the #SBATCH account/partition/gres/time above, the REPO/PY/CONFIG/
# ERA5 paths below, and the campaign knobs in COMMON. Both done-criterion modes:
#   AMIP (default):  sbatch <this>
#   CMIP (coupled):  MODE=cmip COUPLED_PRESET=aquaplanet sbatch <this>
# Other knobs: --diagnosis-method clubb_coefficient|prandtl_number|c_eps OR
# --coefficients C_K,Pr_t (simultaneous multi-coefficient), --iterations, --n-worst,
# --les-budget. Auto-requeues itself on the wall-clock limit; re-submitting the SAME
# script by hand also resumes (the -f $CKPT check), e.g. after a node failure.
# ---------------------------------------------------------------------------
set +e
REPO="${LEGOESM_REPO:-$PWD}"
cd "$REPO" || { echo "cannot cd to REPO=$REPO"; exit 2; }
PY="${LEGOESM_PY:-.venv/bin/python}"
CONFIG="${CONFIG:-configs/amip_clubb_lite.json}"   # base ExperimentConfig (turbulence=clubb_lite)
# Create a starter CONFIG with:
#   python scripts/experiment/write_amip_clubb_lite_config.py "$CONFIG" --resolution 8 --nlev 10
ERA5="${ERA5:-/path/to/era5.zarr}"                 # ERA5 reanalysis zarr store
OUT="${OUT:-results/compare_reanalysis/corrected_clubb.json}"
CKPT="${CKPT:-results/compare_reanalysis/campaign.ckpt.json}"
mkdir -p "$(dirname "$OUT")" "$(dirname "$CKPT")"

MODE="${MODE:-amip}"                               # amip (prescribed SST) | cmip (coupled ocean)
COMMON=(--config "$CONFIG" --era5-zarr "$ERA5" --mode "$MODE"
        --diagnosis-method clubb_coefficient --iterations 10 --n-worst 64
        --les-budget 8 --orographic-forcing auto --out "$OUT")
# CMIP needs a coupled-ocean preset (aquaplanet|full_coupled|slab_simple|slab_richards|
# slab_pft|slab_carbon). Run CMIP with:  MODE=cmip COUPLED_PRESET=aquaplanet sbatch <this>
if [ "$MODE" = "cmip" ]; then COMMON+=(--coupled-preset "${COUPLED_PRESET:-aquaplanet}"); fi

echo "=== HOST ==="; hostname; date

# 1) PRE-FLIGHT (iter 183/184): construct + validate the WHOLE campaign
#    (config/units/grid/scheme/method + ERA5 load + output-path writability) and
#    report the compute estimate in SECONDS -- abort the multi-day run on any
#    construction error instead of burning hours to crash late.
JAX_ENABLE_X64=1 "$PY" scripts/run/run_correction_campaign.py "${COMMON[@]}" --dry-run
if [ "$?" -ne 0 ]; then echo "DRY-RUN FAILED -- fix the config/paths; aborting."; exit 1; fi

# 2) RUN (resumable + self-requeuing): auto-add --resume if a checkpoint already
#    exists, so a re-run continues the accumulated correction.
RESUME=()
if [ -f "$CKPT" ]; then RESUME=(--resume "$CKPT"); echo "resuming from $CKPT"; fi

# On the pre-timeout SIGUSR1, requeue THIS job (race-free: scontrol kills the
# current execution before re-pending, so two jobs never share one campaign/
# checkpoint -- unlike `sbatch "$0"`). The requeued job re-runs from the top and
# --resumes from the last ATOMIC checkpoint, losing at most the in-progress round.
_requeue() {
  echo "[requeue] wall-clock limit approaching -- requeueing job ${SLURM_JOB_ID:-?} to resume from $CKPT"
  if [ -n "${SLURM_JOB_ID:-}" ]; then scontrol requeue "$SLURM_JOB_ID"; fi
}
trap _requeue USR1

# Run in the background + `wait` so bash can service the trap while the (long)
# campaign runs; without `&` + `wait` the trap could not fire until the campaign
# returned, defeating the pre-timeout requeue.
JAX_ENABLE_X64=1 "$PY" scripts/run/run_correction_campaign.py "${COMMON[@]}" \
  --checkpoint "$CKPT" "${RESUME[@]}" &
_pid=$!
wait "$_pid"; rc="$?"
# `wait` returns >128 when the SIGUSR1 trap interrupted it; keep waiting for the
# campaign to actually exit (it is torn down by the requeue) so the logged status
# reflects the real campaign, not the signal.
while [ "$rc" -gt 128 ] && kill -0 "$_pid" 2>/dev/null; do wait "$_pid"; rc="$?"; done
echo "campaign_rc=$rc"; date; echo "=== DONE ==="
exit "$rc"
