#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --job-name=scmrce_crm
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/scmrce_crm_%j.out
set +e
WT=/burg/glab/users/pg2328/scmrce_wt
cd $WT
echo "PINNED sha: $(git rev-parse HEAD)"
echo "host: $(hostname)"; nvidia-smi -L
export PYTHONPATH=$WT/src:$(ls -d $WT/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
OUT=/burg-archive/glab/users/pg2328/legoESM/results/rcemip1_n128_ocean

# ---------------------------------------------------------------------------
# CRM reference for the SCM-RCE convection-scheme intercomparison.
#
# Recipe = the VALIDATED iter-229 production plane-CRM configuration
# (docs/science/specs/CRM_implementation.md:554-564: 132^2 x 30, dx=2 km,
# dt=20 s, van_leer TVD scalars + centered momentum, Smagorinsky c_s=0.2,
# hyperdiff 5e6, N_ACOUSTIC=12, off-centering beta=0.2, semi-implicit),
# at n=128 to match the campaign's reference-dir name.
#
# TWO deliberate deviations from iter-229, both stated so they are not
# mistaken for the validated config:
#   1. radiation = rrtmgp (iter-229 used gray).  REQUIRED: the SCM campaign
#      runs RRTMGP, and run_scm_rce_campaign.py:1934-1943 warns that a
#      gray CRM vs an RRTMGP SCM is a radiation confound.  This matches
#      the two sides.
#   2. radiation-interval = 90 steps = 1800 s, the SAM/WRF/CM1 recommended
#      maximum (the script WARNs above 1800 s; 150 steps = 3000 s would trip
#      that warning).
#
# theta' seed: REQUIRED.  A 1-day unseeded pilot (job 9248485) stayed
# laminar -- max|w| ~ 9e-4 m/s, i.e. no convection at all -- while theta'
# grew monotonically under radiative destabilisation.  The band-noise seed
# breaks the horizontal symmetry so convection can initiate.
#
# 345600 steps @ dt=20 s = 80.0 sim days.  Snapshots (3D volumes) are taken
# at days 62..80 every 2 days = 10 volumes, so --last-reference-files 5
# (days 72,74,76,78,80) has margin, and all are far past spin-up.
# ---------------------------------------------------------------------------
JAX_ENABLE_X64=1 $PY $WT/scripts/run/run_rcemip_plane.py \
  --nx 128 --ny 128 --nlev 30 --dx 2000 --dt 20.0 \
  --steps 345600 \
  --semi-implicit --n-acoustic-substeps 12 --off-centering 0.2 \
  --smag-cs 0.2 --hyperdiff 5e6 \
  --radiation rrtmgp --radiation-interval 90 \
  --microphysics kessler \
  --precision float64 \
  --theta-noise-amp 0.5 --seed-kind band_noise --seed-kmax 6 \
  --print-every 360 \
  --snapshot-days 0.5 \
  --snapshot3d-days "62,64,66,68,70,72,74,76,78,80" \
  --checkpoint-every 21600 \
  --restart "${RESTART:-}" \
  --output $OUT 2>&1
rc=$?
echo "run rc=$rc"
echo "--- vol files ---"; ls -la $OUT/snapshots3d/ 2>&1 | tail -12
echo "--- sfc files (count) ---"; ls $OUT/snapshots/ 2>&1 | wc -l
echo ALLDONE
