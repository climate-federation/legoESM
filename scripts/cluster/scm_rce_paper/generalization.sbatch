#!/bin/bash
#SBATCH --account=glab
#SBATCH -p burst
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 20:00:00
#SBATCH -o /burg-archive/glab/users/pg2328/legoESM/logs/generalization_%A_%a.out
#SBATCH --array=0-4
set +e
WT=${WT:?}; PIN=${PIN:?}
cd $WT; [ "$(git rev-parse HEAD)" = "$PIN" ] || { echo PINFAIL; exit 1; }
git diff --quiet || { echo DIRTY; exit 1; }
export JAX_PLATFORMS=cpu PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/|tr '\n' ':')$WT
# The five schemes whose JOINT params actually differ from single (the rest
# returned defaults under the joint budget, so single-vs-joint is moot there).
SCHEMES=(bechtold zhang_mcfarlane mass_flux sbm dca)
SCH=${SCHEMES[$SLURM_ARRAY_TASK_ID]}
R=/burg-archive/glab/users/pg2328/legoESM/results
JAX_ENABLE_X64=1 /burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -u \
  scripts/validate/scm_rce_generalization.py --schemes $SCH --ssts 295,300,305 \
  --out $R/scm_rce_tqtune/generalization_${SCH}.json 2>&1 | grep --line-buffered -vE "cuda|pjrt|^E[0-9]"
echo "RC=${PIPESTATUS[0]}"
