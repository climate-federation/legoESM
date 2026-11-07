#!/bin/bash
#SBATCH --account=glab --job-name=codex_m8 --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=01:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_m8_%j.log
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps
cd "$REPO" || exit 1
codex exec -m gpt-5.6-sol --sandbox read-only -C "$REPO" "$(cat $G/review_m8_claim.md)

CONTEXT FILES: packages/core/legoesm/grids/fv3_duo_window_spmd.py packages/core/legoesm/grids/fv3_duo_spmd.py packages/core/legoesm/core/fv3_dsw_phase_3d.py packages/core/legoesm/core/fv3_cgrid_phase_3d.py
" 2>&1
echo "CODEX_RC=$?"
echo DONE codex_m8
