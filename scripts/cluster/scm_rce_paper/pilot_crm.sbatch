#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --job-name=scmrce_pilot
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/scmrce_pilot_%j.out
set +e
WT=/burg/glab/users/pg2328/scmrce_wt
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"
echo "host: $(hostname)"; nvidia-smi -L
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

# PREC is passed in via --export (fp64 | fp32)
PREC=${PREC:-fp64}
OUT=/burg-archive/glab/users/pg2328/legoESM/results/scm_rce_paper/pilot_${PREC}
PRECFLAGS="--precision float64"
if [ "$PREC" = "fp32" ]; then
  export LEGOESM_RCEMIP_PLANE_FP32=1
  PRECFLAGS="--precision float32"
fi
echo "PREC=$PREC PRECFLAGS=$PRECFLAGS OUT=$OUT"

# Recipe = the VALIDATED iter-229 production plane-CRM configuration
# (docs/science/specs/CRM_implementation.md:554-564: 132^2 x 30, dx=2 km,
# dt=20 s, van_leer TVD, Smag c_s=0.2, hyperdiff 5e6, N_ACOUSTIC=12,
# off-centering beta=0.2, semi-implicit), at n=128 to match the campaign's
# reference dir name. ONLY deliberate deviation: radiation rrtmgp (not gray)
# so the CRM reference and the SCM share a radiation scheme -- the confound
# run_scm_rce_campaign.py:1934-1943 warns about.
# 4320 steps @ dt=20 s = 1.0 sim day.
JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_rcemip_plane.py \
  --nx 128 --ny 128 --nlev 30 --dx 2000 --dt 20.0 \
  --steps 4320 \
  --semi-implicit --n-acoustic-substeps 12 --off-centering 0.2 \
  --smag-cs 0.2 --hyperdiff 5e6 \
  --radiation rrtmgp --radiation-interval 150 \
  --microphysics kessler \
  $PRECFLAGS \
  --print-every 120 \
  --snapshot-days 0.25 \
  --snapshot3d-days "0.5,1.0" \
  --checkpoint-every 2160 \
  --output $OUT 2>&1
rc=$?
echo "run rc=$rc"
echo "--- vol files ---"; ls -la $OUT/snapshots3d/ 2>&1 | tail -5
echo "--- sfc files ---"; ls -la $OUT/snapshots/ 2>&1 | tail -5
echo ALLDONE
