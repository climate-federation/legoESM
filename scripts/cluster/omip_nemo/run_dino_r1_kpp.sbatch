#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=dino_r1kpp
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/dino_r1kpp_%j.out

# DINO comparison-run candidate: r1_exact preset with the ONE stable
# deviation the ablation (job 8818038) identified — vmix=kpp instead of
# TKE (our TKE NaNs day ~30 under the exact-paper floors; kpp survived
# with healthy max|u|=1.13).  720 d = 24 NEMO monthly records.
set -e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/nemo-gaps-p4p5; cd $WT
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; nvidia-smi -L | head -1; date; git -C $WT log --oneline -1
OUT=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/dino_r1_kpp
mkdir -p $OUT
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/run/run_dino.py \
  --preset r1_exact --vmix kpp --grid latlon --days 720 \
  --snapshot-every-days 30 --allow-multiyear --output-dir $OUT
echo "run_rc=$?"
# Regenerate the DINO comparison figures against the NEMO 4-yr reference.
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY scripts/validate/compare_dino_nemo.py \
  --legoesm-dir $OUT \
  --nemo-gridt /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/tests/DINO_R1/EXP00/DINO_1m_grid_T.nc \
  --nemo-gridu /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/tests/DINO_R1/EXP00/DINO_1m_grid_U.nc \
  --month 24 \
  --out-dir /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/dino_compare_plots
echo "compare_rc=$?"
echo "=== DONE ==="; date
