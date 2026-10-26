#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=les_decomp
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_decomp_%j.out
set +e
# Reads an EXISTING multi-case tuning directory and answers one question:
# is the joint loss dominated by a genuinely hard case, or by a case whose
# reference has so little vertical structure that normalising by its own
# spread inflates the score? Also regenerates the per-case figures, which the
# previous campaign wrote as five byte-identical copies of the bomex panel.
#
# CPU only: this reads saved profiles, it does not roll the SCM forward.
: "${DECOMP_WT:?set DECOMP_WT to the pinned worktree to run from}"
WT=$DECOMP_WT
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results
OUT=${DECOMP_DIR:-$R/scm_les_turbulence/all_cases}

echo "=============== unit tests for the new instruments ==============="
# The probe and the plotter selector are both instruments a conclusion will be
# quoted from, so they are tested before their numbers are read.
$PY -m pytest -q \
    tests/unit/test_scm_les_score_decomposition.py \
    tests/unit/test_plot_scm_les_turbulence_case_select.py 2>&1 | tail -15
echo "pytest rc=${PIPESTATUS[0]}"

echo "=============== score decomposition ==============="
$PY scripts/validate/scm_les_score_decomposition.py "$OUT" \
    --json-out "$OUT/score_decomposition.json" 2>&1 | tail -80
echo "decomp rc=${PIPESTATUS[0]}"

echo "=============== per-case figures (one per CASE this time) ==============="
for c in $(${PY} -c "
import json,sys
print(' '.join(json.load(open('$OUT/tuned_parameters.json'))['protocol']['cases']))
"); do
  $PY scripts/plot/plot_scm_les_turbulence.py "$OUT" --case "$c" \
      --out "$OUT/profiles_vs_les_$c.png" && echo "  plotted $c"
done
# md5 them: identical hashes across cases is the exact defect being fixed.
md5sum $OUT/profiles_vs_les_*.png

echo "=============== summary figure ==============="
$PY scripts/plot/plot_scm_les_tuning_summary.py "$OUT" 2>&1 | tail -5
echo "summary rc=${PIPESTATUS[0]}"
echo "DONE $(date +%T)"
