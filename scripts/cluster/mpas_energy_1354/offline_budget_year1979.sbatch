#!/usr/bin/env bash
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=en1354_offline
#SBATCH --output=%x_%j.out
# #1354 offline energy budget: CONTROLS first -- two runs of the same
# production lane that also wrote the in-run interval-mean tracker series
# (short_main_v2 days 11-20; july_main days 192-201, series counted from its
# start_day 181) compared term by term with the offline instrument -- then the
# year1979 production checkpoints in 5-day windows.  CPU only.
set -uo pipefail
# PROBE must sit on a filesystem the compute node sees (not login-node /tmp).
PROBE=${PROBE:?absolute path to scripts/validate/mpas_energy_budget_offline.py}
PY=/work/bb1596/b309178/mambaforge/envs/diffesm/bin/python
RUNS=/work/bb1596/b309178/diffESM/amip_runs
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
echo "=== CONTROL short_main_v2"
$PY -I "$PROBE" $RUNS/short_main_v2 --days 10 20 --control $RUNS/short_main_v2/timeseries.npz
echo "=== CONTROL july_main"
$PY -I "$PROBE" $RUNS/july_main --days 191 201 --control $RUNS/july_main/timeseries.npz --control-day-offset 181
echo "=== PRODUCTION year1979"
$PY -I "$PROBE" $RUNS/year1979 --days $(seq 5 5 80)
