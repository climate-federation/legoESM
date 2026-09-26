#!/bin/bash
# NEMO-literal TKE step arm (2026-09-26): nemo_z0 + carried avm_k/dissl + literal
# zdftke matrix/solver. One variable vs ctl: the TKE tridiagonal step itself.
#SBATCH --job-name=trp_lit
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trp_ctl_%j.out
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
# ARM A: TKE buoyancy term discretised as NEMO does (--tke-buoyancy-sink
# nemo_explicit: the whole -K_T*N2 on the RHS with the previous-step K_T,
# zdftke.F90:417-420) vs the card's implicit_linearized. ONE variable vs
# trp_stmean_uvw_d5. Motivation: turbulent layer 19 m vs NEMO 36 m from the
# first six hours; codex named the implicit sink as damping imported TKE in
# proportion to the NEW e, suppressing penetration into stratified layers.
# PRE-REGISTERED: 4-phase turbocline (mldkz5 rule) 19 -> >= 28 m and the day-5
# window-mean lens 0.178 -> <= 0.14 confirms; within +3 m refutes. A crash /
# negative-TKE flood is itself the finding (report, do not stabilise).
# detached worktree at the pinned commit; _omip_merge stays free for probe commits
ROOT=/burg-archive/glab/users/pg2328/_arm_literal; cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
CMD=$(cat /burg-archive/glab/users/pg2328/legoESM/omip_nemo/_trp_literal_d2.txt)
case "$CMD" in *"--snapshot-every-days 0.0833333"*"--state-accumulate"*) ;; *) echo "FATAL: arm flags missing"; exit 1;; esac
[ "$(git -C $ROOT rev-parse --short HEAD)" = "1eef1e1a3" ] || { echo "FATAL: tree HEAD is not 1eef1e1a3"; exit 1; }
echo "CMD: $PY scripts/run/run_omip_core2.py $CMD"
$PY -u scripts/run/run_omip_core2.py $CMD
rc=$?
echo "=== card exit: $rc ==="
ls /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_literal_d2/*.npz 2>/dev/null | tail -25
exit $rc
