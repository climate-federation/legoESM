#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=gmtscore
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/_omip_merge/omip_nemo/gmtscore_%j.out
#
# The pre-registered verdict for the MPAS Treguier arm: does giving MPAS the
# adaptive eddy coefficient narrow the MPAS-vs-tripole gap?
#
# The SAME scorer, the SAME oracle, the SAME target grid and the SAME masking
# are used for both arms — only --mpas changes — so the two scorecards are
# comparable to each other. Running one arm alone would answer "how far is
# MPAS from NEMO", which is not the question.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

RES=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
NEMOT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
# The scorer takes SNAPSHOT FILES, not run directories.  `snapshot_final.npz`
# is used for all three because the Treguier arm has no `snapshot_day0030.npz`
# (day 30 coincided with its last step, so only the final snapshot was
# written).  Mixing file names across arms would be a protocol asymmetry;
# --expect-day 30 below makes the scorer verify each one really is day 30
# rather than trusting the name.
TRP=$RES/trp_orca1faithtint_d30/snapshot_final.npz
OUT=$RES/score_gmtreguier

for f in "$NEMOT" "$TRP"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

RC=0
score() {   # score <label> <mpas-dir>
  local label="$1" mpas="$2"
  echo "=== $label : $mpas ==="
  [ -f "$mpas" ] || { echo "FATAL: no such file $mpas"; RC=1; return; }
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -u \
    scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole "$TRP" --mpas "$mpas" --nemo-gridt "$NEMOT" \
    --expect-day 30 --label-mpas "$label" \
    --out-dir "$OUT/$label" 2>&1 | tail -80
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: scorer failed for $label"; RC=1; }
}

# ONE variable apart: same card, the arm adds --gm-treguier.
score baseline  $RES/mpas_orca1faithtint_d30/snapshot_final.npz
score treguier  $RES/mpas_gmtreguier_d30/snapshot_final.npz

echo "DONE rc=$RC"
exit $RC
