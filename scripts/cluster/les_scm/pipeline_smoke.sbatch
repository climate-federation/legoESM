#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --job-name=les_tunesmoke
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/les_tunesmoke_%j.out
set +e
# ---------------------------------------------------------------------------
# PIPELINE smoke for the turbulence-tuning campaign, run against the SMOKE LES
# output rather than the production reference.
#
# THIS PRODUCES NO PHYSICS RESULT. The smoke LES is 32x32x40 over 15 minutes;
# its profiles are not a converged BOMEX and nothing here may be quoted as a
# scheme ranking. The point is to exercise the whole chain -- deck -> SCM
# column, 9 arms, the arms-differ-only-in-turbulence control, scoring against
# an LES reference, the preflight gradient freeze, MUON + line search, and the
# output/plot writers -- while the production LES is still integrating, so the
# real campaign does not discover a plumbing bug hours from now.
#
# The smoke LES wrote 4 frames at t = 0, 0.083, 0.167, 0.25 h, so a trailing
# 0.1 h analysis window holds 2 frames (the loader refuses fewer).
# ---------------------------------------------------------------------------
WT=/burg/glab/users/pg2328/les_scm_bridge_wt
cd $WT || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"
echo "host: $(hostname)"; nvidia-smi -L
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

REPO=/burg-archive/glab/users/pg2328/legoESM
OUT=$REPO/results/scm_les_turbulence/_pipeline_smoke_bomex
rm -rf $OUT

$PY scripts/run/run_scm_les_turbulence_tuning.py \
    --case bomex \
    --les-dir $REPO/results/les_smoke/bomex \
    --outdir $OUT \
    --nlev 32 --dt 60 --analysis-hours 0.1 \
    --chunk-steps 5 --steps 2
rc=$?
echo "pipeline rc=$rc"

if [ -f "$OUT/profiles.npz" ]; then
  $PY scripts/plot/plot_scm_les_turbulence.py "$OUT"
  echo "plot rc=$?"
fi
echo "=== ranking ==="
cat "$OUT/ranking.csv" 2>/dev/null
echo "DONE $(date +%T)"
exit $rc
