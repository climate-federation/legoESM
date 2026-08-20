#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=clubb_pc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/clubb_pc_%j.out
set +e
# WHICH CASE contributes the 1e15 gradient? Stop hypothesising and enumerate.
#
# Three hypotheses have now been refuted by measurement:
#   - chaos amplifying a rounding difference: smagorinsky has the LARGEST
#     gradient and an EXACTLY 1.0 objective at defaults, where chaos predicts
#     the largest discrepancy;
#   - the dry rt == 0 configuration: cbl (dry) is as well-conditioned as bomex
#     (moist) at every rollout length from 15 to 1440 steps, |grad| 0.13 .. 2.1;
#   - the reference-spread floor: restricted to the variables each case actually
#     scores, the largest 1/spread amplification is 284 (astex qv), not 1e8.
#
# And on bomex ALONE the campaign's own objective is perfectly behaved: loss
# exactly 1.0 at the defaults, |grad|max 0.282, and clubb TRAINS (0.0753 ->
# 0.0635, rel 0.844, 33 parameters). So the pathology is contributed by a
# specific case, and eight single-case runs name it.
#
# PRE-REGISTERED, per case, from the driver's own preflight print:
#   the culprit shows |grad|max >= ~1e13 (|dL| ~ lr * sum|g| ~ 0.1 * 33 * |g|,
#   so |dL| = 1e15 needs |g| ~ 3e14) and/or a preflight loss != 1.0.
#   Every case O(1) would REFUTE "one bad case" and point at the multi-case
#   AGGREGATION itself, which is then the only thing left.
: "${E_WT:?set E_WT}"
cd "$E_WT" || exit 1
echo "PINNED sha: $(git rev-parse HEAD)  host: $(hostname)"
export PYTHONPATH=$E_WT/src:$(ls -d "$E_WT"/packages/*/ | tr '\n' ':')
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results
# ONE case per job, set by CASES. The all-eight loop hit the walltime on
# wangara -- 2880 steps at dt=10 s with the slowest closure, whose COMPILE alone
# exceeded an hour -- and took the four unrun cases down with it. A slow case
# must not starve the others.
for C in ${CASES:?set CASES to the case name(s) to enumerate}; do
  echo "########## $C ##########"
  $PY scripts/run/run_scm_les_turbulence_tuning.py \
      --cases "$C:$R/les_ref/$C" \
      --outdir "$R/scm_les_turbulence/_clubb_pc_$C" \
      --nlev 48 --steps 1 --schemes clubb \
      --joint-aggregation default_relative --surface-flux-to-closure 2>&1 \
    | grep -vE "slow_operation|^E[0-9]{4}|XLA has|If you|constant-folding|^$"
done
echo "DONE $(date +%T)"
