#!/bin/bash
#SBATCH --job-name=fwdil
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fwdil_%j.out
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$ROOT/results/omip_nemo
for g in mpas trp; do
  if [ $g = mpas ]; then V=$R/mpas_unified_180d; F=$R/mpas_fwnemo_180d; else V=$R/trp_unified180; F=$R/trp_fwnemo_180d; fi
  echo "########## $g: ref = virtual d15 as proxy IC? NO -- use the earliest common snapshot: virtual d15 vs real d15 vs each other, anomalies relative to virtual d15"
  ls $V/snapshot_day0015.npz $F/snapshot_day0015.npz $V/snapshot_day0030.npz $F/snapshot_day0030.npz
  $PY scripts/validate/ocean_fidelity/freshwater_dilution_depth_profile.py \
    --ref $V/snapshot_day0015.npz \
    --arm "virt_d30=$V/snapshot_day0030.npz" --arm "real_d15=$F/snapshot_day0015.npz" --arm "real_d30=$F/snapshot_day0030.npz"
done
echo DONE
