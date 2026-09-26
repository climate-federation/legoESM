#!/bin/bash
#SBATCH --job-name=tkesfc
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/tkesfc_%j.out
# Is the surface TKE VALUE (rn_ebb*taum/rho0) the same in both models? Our
# e at 1 m is 3.2x NEMO's e at 1 m under interior_pinned, i.e. our pinned
# value itself; NEMO's en(1) at z=0 is in the restart. Same taum -> the
# profile shape is the defect; taum 1.7x -> the BULK STRESS is upstream.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
git -C $ROOT rev-parse --short HEAD | grep -q bb0771ff5 || { echo "worktree HEAD moved"; exit 2; }
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_HOURLY
for PAIR in "12:snapshot_day0000.500" "24:snapshot_day0001" "36:snapshot_day0001.500" "48:snapshot_day0002"; do
  H=${PAIR%%:*}; S=$R/trp_evdocc_d5/${PAIR#*:}.npz
  echo "##### step $H"
  $PY -u scripts/validate/ocean_fidelity/equatorial_tke_length_vs_nemo_restart.py --restart-glob "$N/ORCA1_000000${H}_restart_oce_*.nc" --snapshot $S --n-levels 3 --nemo-t-file $N/ORCA1_1h_20000101_20000102_grid_T_1h.nc --nemo-rec $((H-1)) 2>&1 | grep -E "^\[surface\]|SystemExit|Error"
done
echo "=== card exit: $? ==="
