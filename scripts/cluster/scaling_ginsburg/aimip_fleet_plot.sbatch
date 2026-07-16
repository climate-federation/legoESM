#!/bin/bash
# Reproduce AIMIP Phase-1 paper Fig 3: annual global-mean 2-m air temperature
# ANOMALY (rel. 1979-2014), 1979-2024, for the PUBLISHED AIMIP fleet
# (ACE2.1-ERA5 / NeuralGCM / NeuralGCM-HRD / ArchesWeather) + ERA5 reference,
# downloaded from the public AIMIP S3 bucket (s3://ai-mip via DKRZ). CPU only
# (S3 download + matplotlib) -- no GPU, no legoesm dycore. Our 6h-trained
# models are NOT overlaid (they cannot honestly run 46-yr AMIP inference).
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=aimip_fleet_plot
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=results/aimip_fleet_paper/logs/%x_%j.out
set -uo pipefail
source "$SLURM_SUBMIT_DIR/scripts/cluster/scaling_ginsburg/_env.sh"
cd "$REPO"
OUT=results/aimip_fleet_paper
mkdir -p "$OUT/logs"
echo "[fleet] $(date) reproducing AIMIP Phase-1 Fig 3 (anomaly) -> $OUT"
"$PY_GPU" -u scripts/plot/plot_aimip_fleet_annual.py \
  --anomaly \
  --out "$OUT/aimip_fleet_tas_anomaly.png" \
  --csv "$OUT/aimip_fleet_tas_anomaly.csv"
echo "[fleet] done $(date)"
ls -la "$OUT"/*.png "$OUT"/*.csv 2>/dev/null
