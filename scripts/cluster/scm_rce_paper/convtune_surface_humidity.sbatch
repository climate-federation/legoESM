#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=convtune_qsat
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/convtune_qsat_%j.out
set +e
# ---------------------------------------------------------------------------
# Attribute the SCM-RCE evaporation deficit (tuned columns 1.2-1.9 mm/day vs
# the CRM's 2.40) to one of the two multiplicands of the bulk formula:
#
#     E = rho * C_E * |U| * (q_sat(SST) - q_a)
#
# The probe measures the DRIVER on both sides at the same level, with the same
# saturation curve and the same humidity convention, and prints the exact
# identity  E_ratio = driver_ratio * transfer_ratio  as its own self-check.
#
# Gates run FIRST: the attribution is only as good as the saturation call, so
# the unit tests pin a hand-computed Tetens value and prove the self-check is
# non-vacuous by perturbing a row.  Red gates abort before any number is
# printed.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
REPO=/burg-archive/glab/users/pg2328/legoESM
REF=$REPO/results/rcemip_ref_sam300
OUT=${OUT:?export OUT=<arm results dir>}

cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"; echo "host: $(hostname)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }
NCK=$(ls $OUT/scheme_*.json 2>/dev/null | wc -l)
echo "OUT=$OUT   checkpoints present: $NCK / 10"
[ "$NCK" -eq 10 ] || { echo "ABORT: incomplete arm ($NCK/10)"; exit 92; }
echo "aux 0D flux files:"; ls -l $REF/aux_0D 2>/dev/null || echo "  (none -- will fall back to E=P)"

export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
export MPLBACKEND=Agg
export TMPDIR=${SLURM_TMPDIR:-/burg-archive/glab/users/pg2328/_tmp}
mkdir -p "$TMPDIR"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo; echo "############ probe unit tests ############"
JAX_ENABLE_X64=1 $PY -m pytest -q \
  $WT/tests/unit/test_check_scm_rce_surface_humidity.py 2>&1 | tail -15
TEST_RC=${PIPESTATUS[0]}
echo "TEST_RC=$TEST_RC"
[ $TEST_RC -eq 0 ] || { echo "ABORT: probe gates red"; exit $TEST_RC; }

for COND in tuned prior; do
  echo; echo "############ surface humidity: $COND ############"
  JAX_ENABLE_X64=1 $PY $WT/scripts/validate/check_scm_rce_surface_humidity.py \
    $OUT --reference-dir $REF --condition $COND \
    --json-out $OUT/surface_humidity_$COND.json 2>&1 \
    | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader"
  echo "RC_$COND=${PIPESTATUS[0]}"
done

echo; echo "COMPLETED"
