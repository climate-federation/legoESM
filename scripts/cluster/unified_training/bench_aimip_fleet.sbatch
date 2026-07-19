#!/usr/bin/env bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=bench_fleet
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/scripts/cluster/unified_training/logs/bench_fleet_%j.out
# AIMIP fleet benchmark (D8): annual global-mean tas anomaly time series of
# the published AIMIP Phase-1 fleet (S3) + ERA5 + our AMIP-inference
# variants, with the E1 bias/RMSE + E2 train/holdout trend metrics table.
#   sbatch scripts/cluster/unified_training/bench_aimip_fleet.sbatch
set -euo pipefail

source /burg-archive/glab/users/pg2328/legoESM/scripts/cluster/unified_training/_env.sh
cd "$UT_REPO"

OUT_DIR="${OUT_DIR:-results/bench_aimip_fleet}"
mkdir -p "$OUT_DIR"

LEGO_ARGS=()
for csv in results/aimip_fleet_paper/legoesm_*_amip*annual*.csv; do
  [ -f "$csv" ] && LEGO_ARGS+=(--legoesm-csv "$csv")
done

"$UT_PYTHON" -u scripts/plot/plot_aimip_fleet_annual.py \
  --anomaly \
  --out "$OUT_DIR/aimip_fleet_tas_annual.png" \
  --csv "$OUT_DIR/aimip_fleet_tas_annual.csv" \
  "${LEGO_ARGS[@]}"
echo "[bench-fleet] done $(date)"
ls -l "$OUT_DIR"
