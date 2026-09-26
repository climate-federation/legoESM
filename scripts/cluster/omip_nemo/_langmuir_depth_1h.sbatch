#!/bin/bash
#SBATCH --job-name=hlc
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/hlc_%j.out
# Langmuir depth from each model's own column: NEMO restarts (LC run) hours
# 4/8/12/16/24, ours (EVD arm) hours 6/12/18/24. Same formula, numpy.
# PRE-REGISTERED: h_lc(NEMO) 18-21 m and h_lc(ours) 12-15 m from hour 6 on,
# with NEMO's hour-4 value already ~18 m -> the LC depth follows the
# stratification integral and our 5-15 m N2 excess is upstream of it.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
git -C $ROOT rev-parse --short HEAD | grep -q c51f44b2e || { echo "worktree HEAD moved"; exit 2; }
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_evdocc_d5
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_HOURLY
$PY -u scripts/validate/ocean_fidelity/equatorial_langmuir_depth_vs_nemo.py --items \
  "nemo_h04=nemo:$N/ORCA1_00000004_restart_oce_*.nc" "nemo_h08=nemo:$N/ORCA1_00000008_restart_oce_*.nc" \
  "nemo_h12=nemo:$N/ORCA1_00000012_restart_oce_*.nc" "nemo_h16=nemo:$N/ORCA1_00000016_restart_oce_*.nc" \
  "nemo_h24=nemo:$N/ORCA1_00000024_restart_oce_*.nc" \
  "ours_h06=ours:$R/snapshot_day0000.250.npz" "ours_h12=ours:$R/snapshot_day0000.500.npz" \
  "ours_h18=ours:$R/snapshot_day0000.750.npz" "ours_h24=ours:$R/snapshot_day0001.npz" 2>&1 | grep -vE "cuInit|jax_plugins|plugin_module|_check_cuda|cuda_versions|Traceback|^\s*\^|Jax plugin"
echo "=== card exit: $? ==="
