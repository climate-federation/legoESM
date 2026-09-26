#!/bin/bash
#SBATCH --job-name=ri_d2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ri_d2_%j.out
# Full shear/N2/Ri profiles (ours vs NEMO hourly) for a 2-day arm at hours
# 4, 12, 24, 36: does our Richardson number sit 6x below NEMO's in the
# afternoon layer (hour-24 Prandtl 1.6 vs NEMO's cap of 10)?
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
git -C $ROOT rev-parse --short HEAD | grep -q 035ad86bd || { echo "worktree HEAD moved"; exit 2; }
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
ARM=${ARM:-trp_ctl_d2}
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/$ARM
B=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00
NOISE="^(WARNING|INFO|E0000|W0000|I0000|.*XLA|.*constant.folding|.*slow_operation|.*xla_dump|.*isn't necessarily)"
for H in 4 12 24 36; do
  S=$(ls $R/snapshot_day*.npz | awk -v h=$H '{ if (match($0,/day([0-9.]+)\.npz/,m) && (m[1]*24-h<0.95 && h-m[1]*24<0.95)) print }' | head -1)
  [ -z "$S" ] && { echo "##### hour $H: no snapshot"; continue; }
  REC=$(python3 -c "import re,sys;print(int(float(re.search(r'day([0-9.]+)\.npz',sys.argv[1]).group(1))*24))" $S)
  echo "##### $ARM hour $H ($S) vs NEMO RUN_HOURLY rec $REC"
  $PY -u scripts/validate/ocean_fidelity/equatorial_shear_ri_vs_nemo.py --snapshot $S --nemo-dir $B/RUN_HOURLY --nemo-stem ORCA1_1h_20000101_20000102 --nemo-suffix _1h --rec $REC --n-levels 18 2>&1 | grep -vE "$NOISE"
done
echo "=== card exit: 0 ==="
