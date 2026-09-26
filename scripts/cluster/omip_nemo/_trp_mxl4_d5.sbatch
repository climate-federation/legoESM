#!/bin/bash
#SBATCH --job-name=trp_mxl4
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trp_mxl4_%j.out
#
# THE PRODUCTION CONTROL, RE-RUN WITH THE WINDOW MEAN.
#
# Same command line as trp_pinned_d30 (results/omip_nemo/trp_pinned_d30/
# run_manifest.json) plus --state-accumulate, so every 5-day snapshot also
# carries T_mean/S_mean: the 5-day window mean that NEMO's grid_T file IS
# (tos hourly-sampled mean; to/so thickness-weighted mean, XIOS
# file_def_nemo-oce.xml:79-80). ONE VARIABLE: the diagnostic; the trajectory
# is byte-identical to the control (read-only accumulator; the instantaneous
# T/S in each snapshot must match the control's to roundoff -- that is the
# first check on this run).
#
# PRE-REGISTERED, nino3 SST bias at day 30 vs NEMO record 5 (days 25-30):
#   instantaneous 00 UTC snapshot (the standing headline)   = +1.13 (control)
#   T_mean window-matched                                     = ?
#   * T_mean bias <= +0.6  -> half or more of the day-30 headline is clock
#     phase; the equatorial ledger is re-based on the window mean;
#   * T_mean bias >= +1.0  -> the day-30 lens is real at the daily mean and
#     the clock explains only the day-5 number; the search stays in mixing.
# ARM: mixing length nn_mxl=2 (ORCA1 namelist_cfg) == our tke_mxl_choice 4, vs
# the production card's 3. ONE variable vs trp_stmean_uvw_d5 (same HEAD family,
# 6-hourly snapshots added for the clock-phase probes). Motivation MEASURED
# 2026-09-26: like-for-like turbocline (avt<5e-4, NEMO's nlb10 rule) ours 19.3 m
# at all four clock phases vs NEMO mldkz5 35.7 m in 220-240E |lat|<=2, day 1-5.
# PRE-REGISTERED: 4-phase turbocline moves from 19 toward 36 m AND the day-5
# window-mean SST-T(10) lens from 0.178 toward NEMO's 0.108 -> the mixing-length
# choice is the lever (and the production card's 3 is a harmonization defect);
# turbocline unchanged (< +3 m) -> refuted, look at TKE sources (etau/Langmuir
# delivered values) not the length. Choice 4 crashed a run on 2026-08-06; a
# crash here is itself the finding (report, do not stabilise).
set -uo pipefail
# detached worktree at the pinned commit; _omip_merge stays free for probe commits
ROOT=/burg-archive/glab/users/pg2328/_arm_mxl4; cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
CMD=$(cat /burg-archive/glab/users/pg2328/legoESM/omip_nemo/_trp_mxl4_d5.txt)
case "$CMD" in *"--snapshot-every-days 0.25"*"--state-accumulate"*|*"--snapshot-every-days 5"*"--state-accumulate"*) ;; *) echo "FATAL: arm flags missing"; exit 1;; esac
[ "$(git -C $ROOT rev-parse --short HEAD)" = "8ad12a2ae" ] || { echo "FATAL: tree HEAD is not 8ad12a2ae"; exit 1; }
echo "CMD: $PY scripts/run/run_omip_core2.py $CMD"
$PY -u scripts/run/run_omip_core2.py $CMD
rc=$?
echo "=== card exit: $rc ==="
ls /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_mxl4_d5/*.npz 2>/dev/null | tail -25
exit $rc
