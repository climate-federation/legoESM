#!/bin/bash
#SBATCH --account=glab
#SBATCH -p burst
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o /burg-archive/glab/users/pg2328/legoESM/logs/per_sst_%A_%a.out
#SBATCH --array=0-9
set +e
WT=${WT:?}; PIN=${PIN:?}
cd $WT; [ "$(git rev-parse HEAD)" = "$PIN" ] || { echo PINFAIL; exit 1; }
git diff --quiet || { echo DIRTY; exit 1; }
export JAX_PLATFORMS=cpu PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/|tr '\n' ':')$WT
SCHEMES=(dca kuo mass_flux edmf sbm zhang_mcfarlane kain_fritsch emanuel tiedtke bechtold)
SCH=${SCHEMES[$SLURM_ARRAY_TASK_ID]}
R=/burg-archive/glab/users/pg2328/legoESM/results
JAX_ENABLE_X64=1 /burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -u \
  scripts/validate/scm_rce_per_sst_profiles.py --scheme $SCH --ssts 295,300,305 \
  --out $R/scm_rce_tqtune/per_sst_joint/per_sst_${SCH}.json 2>&1 | grep --line-buffered -vE "cuda|pjrt|^E[0-9]"
echo "RC=${PIPESTATUS[0]}"
