#!/bin/bash
#SBATCH --job-name=eqspinup
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eqspinup_%j.out
#
# WHEN DOES THE EQUATORIAL MIXING DIVERGE FROM THE ORACLE -- DAY 1 OR DAY 20?
#
# The chain is closed to a self-consistent under-mixing state (K_M 5-6% of
# NEMO's at 3-15 m, surface current +97%, undercurrent starved) and the seed
# budget (5549c5311) shows the TKE EQUATION itself matches NEMO's term by term
# within ~20% at 2-8 m on NEMO's own state. A faithful equation with a 170x
# energy deficit in the run means the RUN's STATE selected the low-mixing
# branch. The one thing never measured is WHEN: every comparison so far is
# day 30 or later. The daily-snapshot run (trp_daily_d15, --kprofile-snapshots)
# already holds days 1-15; NEMO's GATEWAY record 0 is the days 1-5 mean.
#
# PRE-REGISTERED, on our stored K_M/K_H (not reconstructed) in the 200-260E
# +/-2 deg box against NEMO's days-1-5 mean avm/avt:
#   * ours at 5-25 m already <= 0.1x NEMO's on days 1-2 -> the split happens
#     in the FIRST DAYS of spin-up; the next instrument is an HOURLY oracle
#     re-run of days 0-5 (cold start, file_def with avm/avt/en/uo hourly) set
#     beside an hourly run of ours;
#   * ours within 0.3-3x of NEMO's mean through day 5 and falling below 0.1x
#     only by days 10-15 -> a slow state-driven collapse; the daily pair d5-d30
#     is the instrument and the hourly oracle run is NOT built;
#   * ours >= NEMO's at all days -> the day-30 deficit is not a decay at all
#     and the day-30 measurement is re-examined before anything else.
# A five-day MEAN against daily snapshots is a coarse pairing; only order-of-
# magnitude readings are taken from it, which is what the branches ask.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
echo "HEAD: $(git -C $M log --oneline -1)"
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
W=$(ls $N/ORCA1_5d_20000101_*grid_W.nc | head -1)
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_daily_d15
SNAPS=""; for d in 01 02 03 05 08 11 14; do SNAPS="$SNAPS $R/snapshot_day00$d.npz"; done
echo "NEMO grid_W: $W (record 0 = days 1-5 mean)"
$PY -u scripts/validate/ocean_fidelity/equatorial_diffusivity_oracle.py \
  --nemo-gridw "$W" --rec 0 --legoesm-snapshot $SNAPS
rc=$?
echo "=== card exit: $rc ==="
exit $rc
