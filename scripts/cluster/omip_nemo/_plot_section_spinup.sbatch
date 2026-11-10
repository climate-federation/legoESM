#!/bin/bash
#SBATCH --job-name=spinup
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/spinup_%j.out
#
# The day-30 transport excess: was it there all along, or did it appear?
#
# The viscosity schedule steps at days 8 and 16, so those are marked -- as
# SUSPECTS, not as an answer. Marking a candidate on a figure is only honest
# if the caption says what it is, which the plotter's own subtitle does.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge
R=/burg-archive/glab/users/pg2328/legoESM
cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cpu
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
TAG=${TAG:-_halfvisc}

$PY -u scripts/plot/plot_section_transport_spinup.py \
  --dir "$R/results/omip_nemo/section_transport" --tag "$TAG" \
  --days 5,10,15,20,25,30 \
  --mark-days 8,16 \
  --mark-label "viscosity-schedule segment boundaries (a suspect, untested)" \
  --out "$R/results/omip_nemo/section_transport/spinup${TAG}.png"
echo "=== rc=$? ==="
