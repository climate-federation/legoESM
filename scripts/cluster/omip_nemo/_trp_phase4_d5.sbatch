#!/bin/bash
#SBATCH --job-name=trp_phase4
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trp_phase4_%j.out
#
# IS THE EQUATORIAL SURFACE WARM LENS PARTLY THE CLOCK?
#
# Every equatorial SST comparison in this campaign pairs OUR instantaneous
# 00:00 UTC snapshot with NEMO's FIVE-DAY MEAN. At 200-270E, 00:00 UTC is
# 13:20-18:00 local -- the afternoon peak of the diurnal warm layer that
# --dm2dc builds in the 1 m top cell. The measured "lens" (+1.1 C at 0-2 m,
# +0.4 at 2-10 m, column heat matching NEMO) has exactly the vertical shape
# of an afternoon diurnal warm layer. feedback_snapshot_vs_mean_sampling
# already forbids this pairing for K_M/TKE/EUC; SST was never checked.
#
# ONE VARIABLE against trp_pinned_d30 (the production control, same tree,
# same command line): --snapshot-every-days 5 -> 0.25, i.e. snapshots at
# 00, 06, 12, 18 UTC, and --years 5/365. Nothing physical changes; only the
# sampling of the same trajectory.
#
# PRE-REGISTERED, nino3 and eq_pacific SST bias vs NEMO record 0 (days 1-5
# mean) at each of the four phases of day 5, and the mean of the four:
#   * 00 UTC warmest and the 4-phase spread >= 0.3 C -> a large part of the
#     "lens" is sampling phase; the standing nino3 bias is re-based on the
#     4-phase mean and the whole equatorial ledger is re-read against it;
#   * spread < 0.1 C -> the lens is real at every clock time; sampling is
#     exonerated for SST and the search returns to the mixing state.
# The 4-phase mean is a crude daily mean (aliases the diurnal cycle at 6 h);
# the driver accumulator (a real window mean) is the follow-up if branch 1.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
CMD=$(cat /burg-archive/glab/users/pg2328/legoESM/omip_nemo/_trp_phase4_d5.txt)
case "$CMD" in *"--snapshot-every-days 0.25"*) ;; *) echo "FATAL: cadence not 0.25"; exit 1;; esac
echo "CMD: $PY scripts/run/run_omip_core2.py $CMD"
$PY -u scripts/run/run_omip_core2.py $CMD
rc=$?
echo "=== card exit: $rc ==="
ls /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_phase4_d5/*.npz 2>/dev/null | tail -25
exit $rc
