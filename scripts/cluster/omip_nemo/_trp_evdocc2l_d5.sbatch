#!/bin/bash
#SBATCH --job-name=trp_evdocc2l
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trp_evdocc2l_%j.out
#
# DOES OUR EVD FIRE AT NIGHT IN THE COLD TONGUE, AS NEMO'S DOES?
#
# NEMO's 5-day-mean avt in 220-240E |lat|<=2 at day 5 is 54/49/44/37/30
# m2/s at 1/2/3/4.5/5.8 m = rn_evd=100 firing 54%..30% of the time (all
# night, every night). The NEMO-exact EVD arm (trp_phase4evd_d5) moved the
# window-mean nino3 SST by +0.0004 K although the lever engaged elsewhere.
# The 2026-09-06 review named the RUNTIME firing fraction as the
# discriminator; it was never run. Same arm + NEMO's TWO-LEVEL trigger MIN(rn2, rn2b) (--convection-two-level), plus
# --evd-occupancy-every-hours 1 (hourly firing fraction at the top 3
# interfaces of the 220-240E box + box-mean T at 0.5..50 m by local hour)
# and the window-mean accumulators.
#
# PRE-REGISTERED (evd_occupancy.csv, days 1-5, local hours 0-6):
#   * night firing fraction at interface 1-2 >= 0.3 and the day-5 T_mean
#     lens (T(0.5)-T(10)) unchanged vs control -> the trigger fires but the
#     delivered K does not mix (solve / placement defect) -> read the solve;
#   * night firing fraction < 0.05 -> the trigger never sees an unstable
#     top interface: the state handed to it (time level, N2 form, where the
#     night cooling lands) is the defect -> instrument the top-cell budget.
#   NEMO's implied occupancy: ~0.5 at 1 m, ~0.3 at 5.8 m (5-day mean / 100).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
CMD=$(cat /burg-archive/glab/users/pg2328/legoESM/omip_nemo/_trp_evdocc2l_d5.txt)
case "$CMD" in *"--snapshot-every-days 0.25"*) ;; *) echo "FATAL: cadence not 0.25"; exit 1;; esac
echo "CMD: $PY scripts/run/run_omip_core2.py $CMD"
$PY -u scripts/run/run_omip_core2.py $CMD
rc=$?
echo "=== card exit: $rc ==="
ls /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_evdocc2l_d5/*.npz 2>/dev/null | tail -25
exit $rc
