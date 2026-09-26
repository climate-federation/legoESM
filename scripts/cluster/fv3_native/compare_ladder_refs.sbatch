#!/bin/bash
#SBATCH --account=glab --job-name=cmp_refs --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/cmp_refs_%j.log
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/fv3_duo_gaps
cd "$REPO" || exit 1
for c in 48 96 192; do echo "===== C$c: loop-arm C-pressure ref (v1) vs batched-arm ref (v2) ====="; "$PY" scripts/validate/fv3_native/compare_ref_npz.py $G/ladder_ref_c$c.npz $G/ladder_ref_c${c}_v2.npz; echo "RC=$?"; done
echo DONE cmp_refs
