#!/bin/bash
#SBATCH --job-name=score_stmean5
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/score_stmean5_%j.out
# trp_statemean_d30 at day 5 vs NEMO record 0 (days 0-5 mean):
#   1. trajectory check: instantaneous T at day 5 must equal the control's
#      (trp_pinned_d30) to roundoff -- the accumulator is read-only;
#   2. instantaneous 00 UTC row (the standing +1.13 headline), and
#   3. the T_mean WINDOW MEAN row (days 0-5), the matched statistic.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
N=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
A=$R/trp_statemean_d30; C=$R/trp_pinned_d30
$PY - <<PYEOF
import numpy as np
a=np.load("$A/snapshot_day0005.npz"); c=np.load("$C/snapshot_day0005.npz")
print("arm keys with mean:", [k for k in a.files if "mean" in k])
print("time_days arm/control:", float(a["time_days"]), float(c["time_days"]))
d=np.abs(np.nan_to_num(a["T"])-np.nan_to_num(c["T"])); print("TRAJECTORY CHECK max|T_arm - T_control| at day 5 = %.3e (want roundoff)"%d.max())
print("window: n_steps=%s window_s=%s"%(a["flux_mean_n_steps"], a["flux_mean_window_s"]))
PYEOF
echo "##### instantaneous 00 UTC day 30 (arm, control) vs NEMO record 5"
$PY -u scripts/validate/ocean_fidelity/sst_phase_vs_nemo.py --nemo-gridt $N --rec 0 --snapshots $A/snapshot_day0005.npz $C/snapshot_day0005.npz
echo "##### WINDOW MEAN days 0-5 (T_mean / T_mean_hw) vs NEMO record 5"
$PY -u scripts/validate/ocean_fidelity/sst_phase_vs_nemo.py --nemo-gridt $N --rec 0 --use-mean-fields --mean --snapshots $A/snapshot_day0005.npz
echo "=== card exit: $? ==="
