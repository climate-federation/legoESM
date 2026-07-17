#!/bin/bash
# ---------------------------------------------------------------------------
# Controlled A/B for the CLUBB sub-grid cloud-fraction -> radiation lever
# (marine-stratocumulus over-bright planetary-albedo bias).
#
# CONTROLLED COMPARISON (CLAUDE.md rule): the two arms differ in EXACTLY ONE
# variable, --use-clubb-cloud-fraction; everything else (grid, levels, dt,
# forcing + its sampling, days, convection=bechtold, turbulence=clubb, clouds,
# microphysics, precision, rollout) is byte-identical.  BOTH arms run diagnostic
# CLUBB turbulence (so louis->clubb is NOT a confound) and the per-step rollout
# (--per-step-rollout: the compiled SegmentCarry does not yet thread the
# cloud-fraction carry; run() raises for compiled+flag rather than silently
# no-op).
#
#   Arm A (baseline):  --no-use-clubb-cloud-fraction  -> RH (sundqvist) grid cf
#   Arm B (treatment): --use-clubb-cloud-fraction      -> diagnostic CLUBB PDF cf
#
# Read albedo (rsut / planetary albedo, esp. marine-Sc decks) from both output
# dirs on the SAME grid/metric; B should be LESS bright where CLUBB is less
# overcast than sundqvist over a saturated marine BL.
#
# PREREQUISITE: run the 2-day smoke first (SMOKE=1) to confirm CLUBB+bechtold+
# latlon+cf runs stably before spending the full A/B.
# ---------------------------------------------------------------------------
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=clubbcf_ab
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM_convcmp/results/clubb_cf_ab/slurm_%j.out

set -u
WT=/burg-archive/glab/users/pg2328/legoESM_convcmp
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1

DAYS="${DAYS:-90}"
[ "${SMOKE:-0}" = "1" ] && DAYS=2
OUT=$WT/results/clubb_cf_ab
mkdir -p "$OUT"
cd "$WT"

# Shared, byte-identical base for BOTH arms.  latlon24 = hydrostatic PE (the path
# the cloud-fraction carry is wired through); override convection->bechtold (the
# over-bright mass-flux scheme this lever targets) + turbulence->clubb (the cf
# producer).  --per-step-rollout carries cloud_fraction.
COMMON=(
  --config config/amip/amip_production_latlon24.yaml
  --convection bechtold
  --turbulence clubb
  --per-step-rollout
  --days "$DAYS"
)

echo "=== $($PY --version); DAYS=$DAYS; $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) ==="

echo "############ Arm A (baseline: RH sundqvist cloud fraction) ############"
$PY -u scripts/run/run_amip.py "${COMMON[@]}" \
    --no-use-clubb-cloud-fraction \
    --output "$OUT/armA_sundqvist_cf"
rcA=$?; echo "### Arm A exit=$rcA"

echo "############ Arm B (treatment: diagnostic CLUBB PDF cloud fraction) ############"
$PY -u scripts/run/run_amip.py "${COMMON[@]}" \
    --use-clubb-cloud-fraction \
    --output "$OUT/armB_clubb_cf"
rcB=$?; echo "### Arm B exit=$rcB"

echo "=== A/B done: armA=$rcA armB=$rcB (0=clean).  Compare rsut/albedo on the"
echo "=== SAME grid+metric between $OUT/armA_sundqvist_cf and $OUT/armB_clubb_cf ==="
[ $rcA -ne 0 ] || [ $rcB -ne 0 ] && exit 1 || exit 0
