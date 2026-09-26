#!/bin/bash
#SBATCH --job-name=trp_z0
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trp_z0_%j.out
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
# ARM B: surface TKE held at NEMO's z=0 W-point (--tke-surface-bc-level nemo_z0,
# the #1690/#1738 placement) vs the production command's interior_pinned. ONE
# variable vs trp_stmean_uvw_d5; same detached worktree (8ad12a2ae), no code
# change. Motivation: the turbulent layer is 19 m vs NEMO's 36 m from the FIRST
# SIX HOURS (card _turbocline_history); the pinned interior row receives no
# downward TKE transport (test_run_omip_core2_surface_bc_level_cli docstring);
# the earlier z0 A/B was scored on an insensitive metric (memory #1690).
# PRE-REGISTERED: 4-phase turbocline (mldkz5 rule) 19 -> >= 28 m and the
# day-5 window-mean lens 0.178 -> <= 0.14 confirms placement as a lever;
# turbocline within +3 m refutes it (collapse intrinsic to the solve/sources).
set -uo pipefail
# detached worktree at the pinned commit; _omip_merge stays free for probe commits
ROOT=/burg-archive/glab/users/pg2328/_arm_mxl4; cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
CMD=$(cat /burg-archive/glab/users/pg2328/legoESM/omip_nemo/_trp_z0_d5.txt)
case "$CMD" in *"--snapshot-every-days 0.25"*"--state-accumulate"*|*"--snapshot-every-days 5"*"--state-accumulate"*) ;; *) echo "FATAL: arm flags missing"; exit 1;; esac
[ "$(git -C $ROOT rev-parse --short HEAD)" = "8ad12a2ae" ] || { echo "FATAL: tree HEAD is not 8ad12a2ae"; exit 1; }
echo "CMD: $PY scripts/run/run_omip_core2.py $CMD"
$PY -u scripts/run/run_omip_core2.py $CMD
rc=$?
echo "=== card exit: $rc ==="
ls /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_z0_d5/*.npz 2>/dev/null | tail -25
exit $rc
