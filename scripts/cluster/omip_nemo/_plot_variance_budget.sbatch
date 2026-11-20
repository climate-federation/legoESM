#!/bin/bash
#SBATCH --job-name=varplot
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/varplot_%j.out
#
# The variance budget, as a figure. Pure JSON in, PNG out -- no model, no
# netCDF, no recomputation, so nothing here can change a number. Both inputs
# are the committed reports:
#
#   NEMO's own operator budget   <- tracer_variance_budget_nemo.py
#   ours vs NEMO, matched state  <- compare_tendencies_nemo.py --variance-budget
#
# The two live under DIFFERENT roots because their cards use different ones:
# the budget card writes under the legoESM results tree, the Mode-A card under
# the merge checkout. Both paths are asserted readable before matplotlib runs,
# so a moved file fails loudly here instead of producing a figure with a stale
# or missing panel.
set -euo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cpu
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

NEMO_JSON=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/variance_budget_nemo/variance_budget.json
OURS_JSON=$ROOT/omip_nemo/variance_ours/tendency_match_rec1.json
OUTPNG=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/variance_budget_nemo/variance_budget.png

for f in "$NEMO_JSON" "$OURS_JSON"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

$PY -u scripts/plot/plot_variance_budget.py \
  --nemo-budget "$NEMO_JSON" \
  --ours-json "$OURS_JSON" \
  --band=0.89,1.49 \
  --out "$OUTPNG"
