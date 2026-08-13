#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=convtune_ab
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/convtune_ab_%j.out
set +e
# ---------------------------------------------------------------------------
# A/B ON THE SUSPECT.  The convection=none control loses 0.854 mm/day of water
# while reporting no precipitation.  With convection off, the only steps that
# touch moisture are turbulence (a SOURCE at the surface), radiation (none) and
# microphysics -- which cannot rain a column carrying ~1e-4 kg/kg of condensate.
# That leaves a non-conservative step, and the in-scheme hard saturation
# adjustment is the one this campaign turned ON.
#
# ONE VARIABLE: identical column, guard on vs off, convection off so no
# convective plumbing can confound it.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
REPO=/burg-archive/glab/users/pg2328/legoESM
REF=$REPO/results/rcemip_ref_sam300

cd $WT || exit 90
HEAD_SHA=$(git rev-parse HEAD); echo "PINNED sha: $HEAD_SHA"; echo "host: $(hostname)"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 91; }

export JAX_PLATFORMS=cpu
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$WT
export LEGOESM_JIT_CACHE_DIR=""
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

for ARMFLAG in "--hard-saturation-adjustment" "--no-hard-saturation-adjustment"; do
  echo; echo "############ convection=none  $ARMFLAG ############"
  JAX_ENABLE_X64=1 $PY $WT/scripts/validate/scm_rce_water_budget.py \
    --reference-dir $REF --convection none --microphysics morrison \
    --days-short 5 --days-long 10 --analysis-days 2 $ARMFLAG 2>&1 \
    | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader"
  echo "RC=${PIPESTATUS[0]}"
done

# And the same A/B with a convecting scheme, so the answer is not special to
# a column with no convective condensate source.
for ARMFLAG in "--hard-saturation-adjustment" "--no-hard-saturation-adjustment"; do
  echo; echo "############ convection=sbm  $ARMFLAG ############"
  JAX_ENABLE_X64=1 $PY $WT/scripts/validate/scm_rce_water_budget.py \
    --reference-dir $REF --convection sbm --microphysics morrison \
    --days-short 20 --days-long 25 --analysis-days 2 $ARMFLAG 2>&1 \
    | grep -v "^E[0-9]\|pjrt_executable\|cpu_aot_loader"
  echo "RC=${PIPESTATUS[0]}"
done
echo ALLDONE
