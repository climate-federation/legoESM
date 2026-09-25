#!/bin/bash
#SBATCH --job-name=trp_ph4evd
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trp_ph4evd_%j.out
#
# ENHANCED VERTICAL DIFFUSION, RE-TESTED WITH PHASE-RESOLVED SAMPLING.
#
# The 2026-09-25 phase probe (job 9990917) measured the nino3 SST bias against
# NEMO's days-1-5 mean at four clock phases of day 4-5 of the pinned control:
# +0.43/+0.49 at 00 UTC (afternoon), +0.25 (06), +0.14 (12, pre-dawn), +0.25
# (18); 4-phase mean +0.27. HALF the headline warm bias is the afternoon
# diurnal warm layer; our SST-T(10 m) is +0.45 at 00 UTC and still +0.15 at
# pre-dawn. The earlier EVD refutation (63d495e91, "lens moved 4%") was read
# on 00 UTC snapshots, i.e. on the afternoon layer, which nightly convection
# cannot touch by construction. It is therefore re-tested on the statistic it
# can move: the daily-mean bias and the pre-dawn residual stratification.
#
# ONE VARIABLE against trp_phase4_d5 (same tree eb06dac06, same command line
# + the five convection flags of the NEMO-exact EVD arm _arm_trp_evdeq_d30).
#
# PRE-REGISTERED (nino3, day 4-5, against NEMO record 0):
#   * 4-phase-mean bias falls from +0.27 to <= +0.13 AND pre-dawn SST-T10
#     falls from +0.15 to <= +0.05 -> EVD is the lever for the daily-mean
#     lens; the 63d495e91 refutation is withdrawn as mis-instrumented;
#   * 4-phase-mean bias moves by < 0.05 -> EVD refuted on the right
#     statistic, and the daily-mean residual has another cause;
#   * in between -> partial; report the dose, no verdict.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
CMD=$(cat /burg-archive/glab/users/pg2328/legoESM/omip_nemo/_trp_phase4evd_d5.txt)
case "$CMD" in *"--snapshot-every-days 0.25"*) ;; *) echo "FATAL: cadence not 0.25"; exit 1;; esac
echo "CMD: $PY scripts/run/run_omip_core2.py $CMD"
$PY -u scripts/run/run_omip_core2.py $CMD
rc=$?
echo "=== card exit: $rc ==="
ls /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_phase4evd_d5/*.npz 2>/dev/null | tail -25
exit $rc
