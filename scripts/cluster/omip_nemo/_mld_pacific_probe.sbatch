#!/bin/bash
#SBATCH --job-name=mldpac
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mldpac_%j.out
# Pacific MLD anomaly probe (user report 2026-08-25): basin x band MLD bias,
# equator-line artifact detector, NW-Pacific box, Pacific zoom PNGs, plus the
# existing T-vs-S MLD decomposition (fresh-lens vs mixing discriminator).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
OUT=$ROOT/results/omip_nemo/nemolev_trp_long180

for pair in "0030 5" "0090 17"; do
  set -- $pair
  echo "=============== day $1 vs rec $2 ==============="
  $PY scripts/validate/ocean_fidelity/mld_basin_vs_nemo.py \
    --legoesm-snapshot $OUT/snapshot_day$1.npz \
    --nemo-gridt "$GW_T" --nemo-time-idx $2 \
    --png $ROOT/results/omip_nemo/mld_pacific_d$1.png
done

echo "=============== T-vs-S decomposition (d90) ==============="
# --snapshot takes label=path (a bare path parses to an empty file name).
$PY scripts/validate/ocean_fidelity/mld_decomposition.py \
  --snapshot d90=$OUT/snapshot_day0090.npz 2>&1 | tail -30
echo DONE
