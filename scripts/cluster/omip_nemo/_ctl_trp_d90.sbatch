#!/bin/bash
#SBATCH --job-name=trpctl90
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trpctl90_%j.out
#
# THE 30-DAY PROTOCOL CANNOT ANSWER THE QUESTION WE ARE ASKING.
#
# Thirty days is a local wind response. An equatorial undercurrent equilibrates
# through boundary-wave round trips, which take MONTHS, so every ours-versus-
# NEMO number in the undercurrent thread compares TRANSIENTS and none of them
# excludes an initial-condition phase explanation. That was GLM's closing
# point and it is the reason this run exists. The user chose this option over
# a perturbation ensemble and over parking the thread.
#
# The oracle is already available: the NEMO gateway file covers 20000101 to
# 20000331, eighteen 5-day records, i.e. out to day 90.
#
# A FRESH 90-DAY LEG, NOT A CONTINUATION, AND THE REASON IS DAY NUMBERING.
# The control's leg runs from its own restart_leg1.npz and its diagnostics
# start at step 0 / day 0.000, so restart_leg1 IS the day-0 state and snapshot
# days are LEG-RELATIVE. Continuing from restart_leg2 (day 30) would restart
# the numbering, so snapshot_day0005 would mean absolute day 35 and every
# scorer call would need an offset -- and the scorer derives its oracle record
# as DAY/5-1 from the snapshot name. Mis-paired days have cost this session
# twice already. Restarting from the DAY-0 state instead keeps absolute days
# and scorer records aligned for free, at the price of re-integrating the
# first thirty days, about ten GPU-hours.
#
# BUILT PROGRAMMATICALLY from the control's own run_manifest command line,
# with exactly four values changed and the token count asserted unchanged at
# 122: --years to 90/365, --restart-from to the control's day-0 state,
# --restart-save and --output into a new directory. Nothing else differs, so
# this is the same configuration as trp_faithz0_d30 run three times as long.
#
# WHAT IT WILL PRODUCE AND WHAT WOULD REFUTE WHAT (pre-registered, because a
# run costing more than one GPU-hour has to name its numbers first):
#   * the 220E undercurrent core speed and 0-400 m net zonal transport at
#     every 5-day mark out to day 90, ours against NEMO's matching record.
#   * If our transport deficit at 220E STOPS growing and settles at a roughly
#     constant fraction of NEMO's, the 30-day divergence was a spin-up
#     transient and the equilibrium difference is that fraction.
#   * If the deficit keeps growing through day 90, it is not a transient and
#     a momentum sink becomes the leading explanation.
#   * If our transport catches NEMO's, the whole undercurrent thread was a
#     transient artefact and closes.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_ENABLE_X64=1
CMD=$(cat /burg-archive/glab/users/pg2328/legoESM/omip_nemo/_ctl_trp_d90.txt)
echo "CMD: $PY scripts/run/run_omip_core2.py $CMD"
$PY -u scripts/run/run_omip_core2.py $CMD
rc=$?
echo "=== card exit: $rc ==="
ls /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_faithz0_d90/*.npz 2>/dev/null | tail -5
exit $rc
