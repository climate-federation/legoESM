#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=dino_r1ex
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/dino_r1ex_%j.out

# legoESM DINO with the Level-1 r1_exact preset (S-EOS, TKE w/ wind+taum,
# seasonal forcing, nemo_quadratic drag, EIV off, stabilizers off) — the
# comparison run against the NEMO DINO_R1 4-yr reference.  2 years,
# monthly snapshots (matches the reference's monthly grid_T cadence).
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/nemo-gaps-p4p5; cd $WT
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
echo "=== worktree ==="; git -C $WT log --oneline -1
OUT=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/dino_r1_exact
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_dino.py \
  --preset r1_exact --grid latlon --days 720 --allow-multiyear \
  --snapshot-every-days 30 --output-dir $OUT 2>&1 | tail -60
echo "run_rc=${PIPESTATUS[0]}"; date
ls -la $OUT/ 2>/dev/null | tail -5
echo "=== DONE ==="
